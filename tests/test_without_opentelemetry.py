"""Spines 1 to 3 run without the OpenTelemetry SDK; Spine 4 says how to get it.

Only Spine 4 (Observed) grades traces, so a Reader on modules 1 to 3 must not
need ``opentelemetry-sdk``: a Case still runs and its Checks still pass, with
an empty trace. Anything that does need a trace, a run through module 4 or a
span exporter, stops before it starts, with exit code 2 ("could not start")
and the ``pip install opentelemetry-sdk`` line. Never exit code 1, which means
a Check failed: a missing package says nothing about the Reader's agent.

The SDK is made missing in-process: ``sys.modules["opentelemetry"] = None``
makes every ``import opentelemetry...`` fail, and the course's own packages
are imported afresh under that block, then put back as they were. So these
tests run under an environment that has the SDK installed too.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HINT = "pip install opentelemetry-sdk"

#: The packages imported afresh with the SDK blocked, and put back afterwards.
_FRESH = ("company", "checks", "flagship", "tests.fixtures", "opentelemetry")


def _fresh(name):
    return any(name == package or name.startswith(package + ".") for package in _FRESH)


@contextmanager
def opentelemetry_missing():
    """Import the course as if ``opentelemetry-sdk`` were not installed."""
    saved = {name: module for name, module in sys.modules.items() if _fresh(name)}
    for name in saved:
        del sys.modules[name]
    sys.modules["opentelemetry"] = None  # every import of it now raises ImportError
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _fresh(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def no_sdk(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = dict(os.environ)
    with opentelemetry_missing():
        import checks.cli

        monkeypatch.setattr(checks.cli, "ENV_FILE", tmp_path / ".env")
        yield
    os.environ.clear()
    os.environ.update(saved)


def test_the_block_really_hides_the_sdk(no_sdk):
    with pytest.raises(ImportError):
        import opentelemetry.sdk.trace  # noqa: F401
    from company import tracing

    assert tracing.available() is False


def test_a_case_runs_without_the_sdk_and_has_an_empty_trace(no_sdk):
    from company.runner import load_case, run_case

    outcome = run_case(load_case("upgrade-to-pro"), "flagship.loop:run", mode="offline")

    assert outcome.resolved is True
    assert outcome.trace.spans == ()


def test_the_spine_4_agent_still_names_itself_without_the_sdk(no_sdk):
    # describe_agent is a no-op on an untraced Case, not an error.
    from company.runner import load_case, run_case

    outcome = run_case(load_case("upgrade-to-pro"), "flagship.observed:run", mode="offline")

    assert outcome.resolved is True


def test_an_agent_that_opens_its_own_span_still_runs_without_the_sdk(no_sdk):
    from company.runner import load_case, run_case

    def opens_a_span(customer_turn, env):
        with env.tracer.start_as_current_span("plan the reply") as span:
            span.set_attribute("acme.turn", customer_turn)
        return "Noted."

    outcome = run_case(load_case("upgrade-to-pro"), opens_a_span, mode="offline")

    assert outcome.replies == ("Noted.",)
    assert outcome.trace.spans == ()


def test_a_span_exporter_without_the_sdk_says_how_to_install_it(no_sdk):
    from company.runner import load_case, run_case

    with pytest.raises(ImportError) as raised:
        run_case(load_case("upgrade-to-pro"), "flagship.loop:run", mode="offline",
                 exporters=[object()])

    assert HINT in str(raised.value)


def test_the_checks_cli_passes_modules_1_to_3_without_the_sdk(no_sdk, capsys):
    from checks.cli import main

    code = main(["--modules", "3", "--mode", "offline"])

    out = capsys.readouterr().out.strip().splitlines()
    assert code == 0, "\n".join(out)
    assert out[-1] == "Passed through module 3 of 3."


@pytest.mark.parametrize("argv", [["--modules", "4"], ["--modules", "1-4"], []])
def test_module_4_without_the_sdk_cannot_start_and_says_why(no_sdk, capsys, argv):
    from checks.cli import main

    code = main(argv + ["--mode", "offline"])

    captured = capsys.readouterr()
    assert code == 2  # could not start; never 1, which would blame the agent
    assert HINT in captured.err
    assert "Module 4" in captured.err
    assert "PASS" not in captured.out and "FAIL" not in captured.out


def test_the_grader_without_the_sdk_cannot_start_module_4_and_writes_no_badge(
        no_sdk, capsys, tmp_path):
    from checks.grader import main

    badge_file = tmp_path / "badge.json"
    code = main(["--modules", "4", "--agent", "tests.fixtures.grader_agents:upgrades_without_a_model",
                 "--badge-file", str(badge_file)])

    assert code == 2
    assert HINT in capsys.readouterr().err
    assert not badge_file.exists()


def test_the_grader_grades_module_1_without_the_sdk(no_sdk, capsys, tmp_path):
    from checks.grader import main

    badge_file = tmp_path / "badge.json"
    code = main(["--modules", "1", "--agent", "tests.fixtures.grader_agents:upgrades_without_a_model",
                 "--badge-file", str(badge_file)])

    assert code == 0, capsys.readouterr().out
    assert json.loads(badge_file.read_text(encoding="utf-8"))["message"].startswith(
        "through module 1")


@pytest.mark.parametrize("notebook", [
    "spine/01-loop/loop.ipynb",
    "spine/02-knowledge/knowledge.ipynb",
    "spine/03-graded/graded.ipynb",
])
def test_the_spine_1_to_3_notebooks_run_without_the_sdk(no_sdk, monkeypatch, notebook):
    path = ROOT / notebook
    monkeypatch.chdir(path.parent)
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]

    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert "opentelemetry" not in source, "cell {} needs the SDK".format(index)
        try:
            exec(compile(source, "{} cell {}".format(path.name, index), "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure message
            pytest.fail("cell {} raised {!r}:\n{}".format(index, error, source))
