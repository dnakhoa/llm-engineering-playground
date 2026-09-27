"""The Checks CLI: cumulative suites, a Spend Cap, and the Reader's own agent.

Tested from the outside: through ``main`` (what the Reader types) and
``run_checks`` (what the Grader calls), Offline with the replay transport, and
"live" with the HTTP transport swapped for a fake so nothing leaves the machine.

Cost arithmetic used below, from the Spine 1 recording on claude-sonnet-5
($2 in / $10 out per million tokens):

    call 1:  812 in, 61 out  -> $0.002234
    call 2:  921 in, 58 out  -> $0.002422
    call 3: 1003 in, 47 out  -> $0.002476
    one Case                 -> $0.007132

Ticket: docs/tickets/graded-attacked-budgeted/04-checks-cli-spend-cap.md
"""
from __future__ import annotations

import builtins
import socket
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from checks import CheckResult  # noqa: E402
from checks.cli import main, parse_modules, run_checks  # noqa: E402
from checks.suites import Suite  # noqa: E402
from company.runner import RECORDINGS_DIR  # noqa: E402
from llm import credentials  # noqa: E402
from llm.replay import ReplayTransport  # noqa: E402
from tests.fixtures.checks_cli_suites import LOOP, SECOND, SUITES  # noqa: E402

AGENTS = "tests.fixtures.case_runner_agents"
RECORDING = RECORDINGS_DIR / "upgrade-to-pro.recording.json"


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def never_asks(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the Checks asked for input")

    monkeypatch.setattr(builtins, "input", refuse)


class FakeHttp:
    """Stands in for llm.transport.HttpTransport in "live" tests."""

    instances = []

    def __init__(self, *, provider="", timeout=120.0):
        self.provider = provider
        self.requests = []
        self._replay = ReplayTransport(RECORDING)
        FakeHttp.instances.append(self)

    def send(self, request):
        self.requests.append(request)
        if request.model == "claude-sonnet-5":
            return self._replay.send(request)
        if request.path.endswith("/responses"):
            return {
                "status": "completed",
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "Done! You're on Pro now."}]}],
                "usage": {"input_tokens": 50, "output_tokens": 9},
            }
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Done! You're on Pro now."},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 50000, "completion_tokens": 9000},
        }


@pytest.fixture
def fake_http(monkeypatch):
    import llm.transport

    FakeHttp.instances = []
    monkeypatch.setattr(llm.transport, "HttpTransport", FakeHttp)
    return FakeHttp


#: Every variable the Checks CLI's model choice reads, cleared for each test.
_MODEL_ENV = credentials.ENV_VARS


@pytest.fixture(autouse=True)
def reader_env(monkeypatch, tmp_path):
    """No keys in the environment, and the CLI reads a .env the test writes.

    `main()` loads `.env` into os.environ, which monkeypatch does not track, so
    the environment is restored by hand afterwards.
    """
    import os

    import checks.cli

    saved = dict(os.environ)
    for name in _MODEL_ENV:
        os.environ.pop(name, None)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(checks.cli, "ENV_FILE", env_file)
    yield env_file
    os.environ.clear()
    os.environ.update(saved)


def _lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


# ── The Reader's first run ────────────────────────────────────────────────────


def test_an_offline_run_of_module_1_passes_and_says_so(capsys, no_network, never_asks):
    code = main(["--modules", "1", "--agent", "flagship.loop:run"])

    out = _lines(capsys.readouterr().out)
    assert code == 0
    assert out[0].startswith("Checks for modules 1 to 1")
    assert any(line.startswith("Spend Cap: $1.00") for line in out[:3])
    assert "PASS  upgrade-to-pro: plan changed to Pro exactly once" in "\n".join(out)
    assert out[-1] == "Passed through module 1 of 1."


def test_the_spend_cap_is_printed_before_any_check_runs(capsys, no_network):
    main(["--modules", "1", "--spend-cap", "0.50"])

    out = _lines(capsys.readouterr().out)
    cap = next(i for i, line in enumerate(out) if line.startswith("Spend Cap: $0.50"))
    first_result = next(i for i, line in enumerate(out) if line.startswith(("PASS", "FAIL")))
    assert cap < first_result
    assert "claude-sonnet-5" in out[cap]


def test_a_failing_agent_does_not_pass_module_1(capsys, no_network):
    code = main(["--modules", "1", "--agent", AGENTS + ":changes_someone_elses_plan"])

    out = _lines(capsys.readouterr().out)
    assert code == 1
    assert any(line.startswith("FAIL  upgrade-to-pro") for line in out)
    assert out[-1] == "Passed through module 0 of 1."


# ── Cumulative suites ─────────────────────────────────────────────────────────


def test_running_module_2_includes_module_1s_suite(no_network):
    report = run_checks(through=2, agent="flagship.loop:run", suites=SUITES)

    assert [(r.module, r.case_id) for r in report.results if r.status == "pass"] == [
        (1, "upgrade-to-pro"),
        (2, "upgrade-to-pro-again"),
    ]
    assert report.passed_through == 2


def test_running_module_1_leaves_later_suites_out(no_network):
    report = run_checks(through=1, agent="flagship.loop:run", suites=SUITES)

    assert {r.module for r in report.results} == {1}


def test_a_module_range_always_starts_at_module_1():
    assert parse_modules("3") == 3
    assert parse_modules("1-3") == 3
    assert parse_modules("1..3") == 3
    with pytest.raises(ValueError, match="every earlier module"):
        parse_modules("2-3")


# ── Live-only Checks ──────────────────────────────────────────────────────────


def test_live_only_checks_are_reported_as_skipped_offline(no_network):
    report = run_checks(through=2, agent="flagship.loop:run", suites=SUITES)

    skipped = [r for r in report.results if r.status == "skip"]
    assert [r.name for r in skipped] == ["reply_is_polite"]
    assert "live only" in skipped[0].detail.lower()
    # A skip is not a failure.
    assert report.passed_through == 2


def test_live_only_checks_run_in_live_mode(fake_http, never_asks):
    report = run_checks(
        through=2, agent="flagship.loop:run", mode="live", suites=SUITES
    )

    assert [r.status for r in report.results if r.module == 2] == ["pass", "pass"]


# ── The Spend Cap ─────────────────────────────────────────────────────────────


def test_a_spend_cap_below_the_run_cost_stops_it_and_reports_what_finished(
    capsys, no_network
):
    # Module 1's Case finishes at $0.007132. Module 2's first call brings the
    # total to $0.009366, past the $0.009 cap, so its second call never goes out.
    report = run_checks(
        through=2, agent="flagship.loop:run", spend_cap_usd=0.009, suites=SUITES
    )

    assert report.stopped_by_spend_cap is True
    assert report.spent_usd == pytest.approx(0.009366)
    by_case = {(r.case_id, r.name): r.status for r in report.results}
    assert by_case[("upgrade-to-pro", "plan_changed_to_pro_exactly_once")] == "pass"
    assert by_case[("upgrade-to-pro-again", "plan_changed_to_pro_exactly_once")] == "stopped"
    assert report.passed_through == 1

    out = _lines(capsys.readouterr().out)
    assert any(line.startswith("Stopped by the Spend Cap") for line in out)
    assert out[-1] == "Passed through module 1 of 2."


def test_a_spend_cap_stops_a_case_mid_loop_and_later_cases_do_not_run(no_network):
    # $0.003: call 1 ($0.002234) is under it, call 2 takes the total to
    # $0.004656, and call 3 is refused. Module 2 never starts.
    report = run_checks(
        through=2, agent="flagship.loop:run", spend_cap_usd=0.003, suites=SUITES
    )

    assert report.spent_usd == pytest.approx(0.004656)
    assert [(r.module, r.status) for r in report.results] == [
        (1, "stopped"),
        (2, "not run"),
        (2, "not run"),
    ]
    assert report.passed_through == 0


# ── The Reader's own agent ────────────────────────────────────────────────────


def test_an_agent_file_outside_the_repo_is_loaded_and_graded(tmp_path, capsys, no_network):
    agent_file = tmp_path / "my_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            '''
            from llm.types import ToolCall

            def run(customer_turn, env):
                env.act(ToolCall(id="c1", name="change_plan",
                                 arguments={"account_id": "acct_1001", "plan": "pro"}))
                return "You are on Pro."
            '''
        ),
        encoding="utf-8",
    )

    code = main(["--modules", "1", "--agent", "{}:run".format(agent_file)])

    out = _lines(capsys.readouterr().out)
    assert code == 0
    assert str(agent_file) in out[0]
    assert "PASS  upgrade-to-pro: plan changed to Pro exactly once" in "\n".join(out)


# ── Live mode and local servers ───────────────────────────────────────────────


def test_live_mode_prints_the_cap_and_asks_for_nothing(capsys, fake_http, never_asks):
    code = main(
        ["--modules", "1", "--mode", "live", "--spend-cap", "0.25", "--model", "claude-sonnet-5"]
    )

    out = _lines(capsys.readouterr().out)
    assert code == 0
    assert any(line.startswith("Spend Cap: $0.25") for line in out[:3])
    assert fake_http.instances and fake_http.instances[0].provider == "anthropic"
    assert out[-1] == "Passed through module 1 of 1."


LOCAL = ["--mode", "live", "--model", "qwen3:8b", "--base-url", "http://localhost:11434/v1"]


def _recording_suite(outcomes):
    """Module 1's Case, with one Check that keeps the Outcome it is given."""

    def keeps_the_outcome(outcome):
        outcomes.append(outcome)
        return CheckResult("outcome kept", True, "kept.")

    return Suite(module=1, title="Loop", cases=("upgrade-to-pro",), checks=(keeps_the_outcome,))


def test_a_local_server_needs_no_registry_entry_and_the_cap_does_not_apply(
    capsys, fake_http, never_asks
):
    code = main(
        ["--modules", "1", "--agent", AGENTS + ":claims_without_acting", "--spend-cap", "0"]
        + LOCAL
    )

    out = _lines(capsys.readouterr().out)
    text = "\n".join(out)
    assert "Spend Cap does not apply" in text
    assert "qwen3:8b" in text and "http://localhost:11434/v1" in text
    request = fake_http.instances[0].requests[0]
    assert request.model == "qwen3:8b"
    assert request.base_url == "http://localhost:11434/v1"
    assert request.path == "/chat/completions"
    # A $0 Spend Cap refuses the first call of any run it applies to, because
    # the cap is checked before every call. So the Case going out and being
    # graded, rather than stopped, shows that no cap was applied.
    assert "STOP" not in text
    assert "Stopped by the Spend Cap" not in text
    assert "FAIL  upgrade-to-pro" in text
    assert code == 1


def test_a_local_server_is_priced_at_zero_and_runs_under_no_spend_cap(fake_http, never_asks):
    outcomes = []
    report = run_checks(
        through=1,
        agent=AGENTS + ":claims_without_acting",
        mode="live",
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        spend_cap_usd=0,
        suites=(_recording_suite(outcomes),),
    )

    (outcome,) = outcomes
    # The call went out and was metered: the fake server reports 50,000 tokens
    # in and 9,000 out, which any registry price would make cost something.
    assert outcome.steps == 1
    assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (50000, 9000)
    assert outcome.cost_usd == 0
    assert outcome.spend_cap_reached is False
    assert report.spend_cap_usd is None
    assert report.spent_usd == 0
    assert report.stopped_by_spend_cap is False


def test_a_local_server_cannot_run_offline(capsys):
    code = main(
        ["--modules", "1", "--model", "qwen3:8b", "--base-url", "http://localhost:11434/v1"]
    )

    assert code == 2
    assert "live" in capsys.readouterr().err


def test_an_offline_run_on_another_model_reports_the_replay_mismatch(capsys, no_network):
    code = main(["--modules", "1", "--model", "claude-opus-5-5"])

    out = capsys.readouterr().out
    assert code == 1
    assert "ERROR upgrade-to-pro" in out
    assert "No recording" in out


def test_the_real_suites_are_discovered_in_module_order():
    from checks.suites import discover_suites

    suites = discover_suites()
    assert suites[0].module == 1 and suites[0].title == "Loop"
    assert [s.module for s in suites] == sorted(s.module for s in suites)
    assert LOOP.checks == suites[0].checks
    assert SECOND.module == 2


# ── Which model, when --model is not given ────────────────────────────────────
#
# The CLI loads the root .env and uses llm.default_model(), the same policy as
# ask(): a Reader with only an OpenAI key runs live on an OpenAI model, not on
# a Claude model their key cannot call. Offline replays recordings made on one
# model, so it stays on that model and needs no key at all.


def _write_env(env_file, **values):
    template = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    lines = [template] + ["{}={}".format(name, value) for name, value in values.items()]
    env_file.write_text("\n".join(lines) + "\n")


def test_live_mode_without_a_model_uses_the_key_in_the_readers_env(
    capsys, fake_http, never_asks, reader_env
):
    _write_env(reader_env, OPENAI_API_KEY="sk-real-openai")

    main(["--modules", "1", "--mode", "live", "--agent", AGENTS + ":claims_without_acting"])

    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" not in captured.out + captured.err
    assert "· live · gpt-6-luna" in captured.out
    assert fake_http.instances[0].provider == "openai"


def test_live_mode_without_a_model_follows_llm_model(capsys, fake_http, reader_env):
    _write_env(reader_env, LLM_MODEL="claude-sonnet-5", ANTHROPIC_API_KEY="sk-ant-real")

    code = main(["--modules", "1", "--mode", "live"])

    assert "· live · claude-sonnet-5" in capsys.readouterr().out
    assert code == 0


def test_live_mode_without_a_model_runs_on_the_readers_local_server(
    capsys, fake_http, never_asks, reader_env
):
    _write_env(reader_env, LLM_PROVIDER="ollama", LLM_MODEL="llama3.2",
               OPENAI_BASE_URL="http://localhost:11434/v1")

    main(["--modules", "1", "--mode", "live", "--agent", AGENTS + ":claims_without_acting"])

    out = capsys.readouterr().out
    assert "Spend Cap does not apply" in out
    request = fake_http.instances[0].requests[0]
    assert (request.model, request.base_url) == ("llama3.2", "http://localhost:11434/v1")


def test_live_mode_with_no_key_anywhere_says_how_to_configure_one(capsys, fake_http, reader_env):
    _write_env(reader_env)

    code = main(["--modules", "1", "--mode", "live"])

    assert code == 2
    assert "LLM_MODEL" in capsys.readouterr().err
    assert fake_http.instances == []


def test_offline_mode_needs_no_key_whatever_the_env_holds(capsys, no_network, reader_env):
    _write_env(reader_env, OPENAI_API_KEY="sk-real-openai")

    code = main(["--modules", "1"])

    out = capsys.readouterr().out
    assert code == 0
    assert "· offline · claude-sonnet-5" in out


def test_offline_mode_with_no_env_file_and_no_key_passes(capsys, no_network, reader_env):
    assert not reader_env.exists()

    assert main(["--modules", "1"]) == 0


def test_live_mode_with_a_model_still_reads_the_key_from_the_readers_env(
    fake_http, reader_env
):
    import os

    _write_env(reader_env, ANTHROPIC_API_KEY="sk-ant-real")

    main(["--modules", "1", "--mode", "live", "--model", "claude-sonnet-5"])

    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-real"


def test_a_variable_already_set_beats_the_env_file(fake_http, reader_env, monkeypatch):
    import os

    _write_env(reader_env, ANTHROPIC_API_KEY="sk-ant-from-file")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-from-shell"

    main(["--modules", "1", "--mode", "live", "--model", "claude-sonnet-5"])

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"
