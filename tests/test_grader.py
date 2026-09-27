"""The Grader: the Checks, explained, with a badge and a line to share.

Tested from the outside, through ``checks.grader.main`` (the command the
Grader skill runs), ``grade`` (the same run with stand-in suites) and
``explain`` (the lesson link for one failed Check), Offline with the replay
transport and nothing on the network.

Ticket: docs/tickets/graded-attacked-budgeted/11-grader-skill.md
"""
from __future__ import annotations

import builtins
import json
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from checks.grader import COURSE_URL, main  # noqa: E402

ROOT = Path(__file__).parent.parent
AGENTS = "tests.fixtures.grader_agents"
FAILING = "tests.fixtures.case_runner_agents:changes_someone_elses_plan"
BLOB = COURSE_URL + "/blob/main/"


@pytest.fixture(autouse=True)
def offline_world(monkeypatch, tmp_path):
    """No network, no terminal questions, no keys, and a .env the test owns."""
    import checks.cli

    def refuse(*args, **kwargs):
        raise AssertionError("the Grader tried to open a network connection")

    def never(*args, **kwargs):
        raise AssertionError("the Grader asked for input")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(builtins, "input", never)
    monkeypatch.setattr(checks.cli, "ENV_FILE", tmp_path / ".env")
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _failed(module, name, case="upgrade-to-pro", status="fail", detail="wrong."):
    from checks.cli import CheckRun

    return CheckRun(module, case, name, status, detail)


# ── Failures, explained ───────────────────────────────────────────────────────


def _heading_exists(link):
    """The link names a lesson file in this repo and a heading it really has."""
    from checks.grader import anchor, headings

    assert link.startswith(BLOB + "spine/"), link
    path, _, fragment = link[len(BLOB):].partition("#")
    readme = ROOT / path
    assert readme.is_file(), link
    assert fragment in {anchor(h) for h in headings(readme)}, link


def test_every_real_check_links_to_a_lesson_heading_that_exists():
    from checks.grader import explain
    from checks.suites import discover_suites

    suites = discover_suites()
    assert {suite.module for suite in suites} >= {1, 2, 3, 4}
    for suite in suites:
        for check in suite.checks:
            _heading_exists(explain(_failed(suite.module, check.__name__), "offline"))


def test_a_check_links_to_the_section_where_a_lesson_names_it():
    from checks.grader import explain

    assert explain(_failed(3, "reply_meets_the_judge_rubric"), "live") == (
        BLOB + "spine/03-graded/README.md#the-judge-rubric-is-live-only")
    assert explain(_failed(2, "the_agent_finishes_the_case"), "offline") == (
        BLOB + "spine/02-knowledge/README.md#the-checks")
    assert explain(_failed(4, "every_tool_call_has_a_tool_span"), "offline") == (
        BLOB + "spine/04-observed/README.md#the-checks")
    # "no forbidden Action is attempted", in the verdict table: case and marks aside.
    assert explain(_failed(3, "no_forbidden_action_is_attempted"), "offline") == (
        BLOB + "spine/03-graded/README.md#one-verdict")


def test_a_check_no_lesson_names_links_to_its_modules_checks_section():
    from checks.grader import explain

    assert explain(_failed(1, "plan_changed_to_pro_exactly_once"), "offline") == (
        BLOB + "spine/01-loop/README.md#the-check")


def test_links_come_from_the_lessons_text_not_a_table(monkeypatch, tmp_path):
    import checks.grader

    lesson = tmp_path / "09-new" / "README.md"
    lesson.parent.mkdir()
    lesson.write_text(
        "# Spine 9\n\n## Intro\n\nNothing here.\n\n"
        "## Where it is taught\n\nThe Check **Brand new Check\nholds** reads state.\n\n"
        "```\n## not a heading\n```\n\n## The Checks\n\nA list.\n",
        encoding="utf-8")
    monkeypatch.setattr(checks.grader, "SPINE", tmp_path)

    base = BLOB + "spine/09-new/README.md"
    assert checks.grader.explain(_failed(9, "brand_new_check_holds"), "offline") == (
        base + "#where-it-is-taught")
    assert checks.grader.explain(_failed(9, "one_nobody_wrote_up"), "offline") == (
        base + "#the-checks")


def test_a_failure_links_to_the_lesson_section_that_covers_it(capsys, tmp_path):
    code = main(["--modules", "1", "--agent", FAILING,
                 "--badge-file", str(tmp_path / "badge.json")])

    out = capsys.readouterr().out
    assert code == 1
    assert "What to read next:" in out
    assert BLOB + "spine/01-loop/README.md#the-check" in out
    lines = _lines(out)
    assert lines[-2] == "Passed through module 0 of 1."
    assert lines[-1].startswith("Share: ")
    assert COURSE_URL in lines[-1]


def test_an_agent_that_raises_is_sent_to_the_agent_contract(capsys, tmp_path):
    main(["--modules", "1", "--agent", AGENTS + ":raises",
          "--badge-file", str(tmp_path / "badge.json")])

    link = BLOB + "spine/01-loop/README.md#the-agent-contract"
    assert link in capsys.readouterr().out
    _heading_exists(link)


def test_an_offline_replay_mismatch_is_sent_to_the_offline_lesson(capsys, tmp_path):
    main(["--modules", "1", "--model", "claude-opus-5-5",
          "--badge-file", str(tmp_path / "badge.json")])

    link = BLOB + "spine/01-loop/README.md#offline-why-this-costs-nothing"
    assert link in capsys.readouterr().out
    _heading_exists(link)


# ── The result badge ──────────────────────────────────────────────────────────

PASSES = AGENTS + ":upgrades_without_a_model"
SHIELDS_KEYS = {"schemaVersion", "label", "message", "color"}


def _suite(module, *checks):
    from checks.suites import Suite
    from tests.fixtures.checks_cli_suites import AGAIN

    return Suite(module=module, title="Module {}".format(module), cases=(AGAIN,),
                 checks=checks)


def _always_fails(outcome):
    from checks import CheckResult

    return CheckResult("always fails", False, "it always does.")


def _module_1():
    from tests.fixtures.checks_cli_suites import LOOP

    return LOOP


def _passes():
    from checks.spine_1_loop import plan_changed_to_pro_exactly_once

    return plan_changed_to_pro_exactly_once


def _badge(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quiet(**kwargs):
    from checks.grader import grade

    return grade(out=lambda line: None, **kwargs)


def test_passing_checks_write_a_shields_endpoint_badge_labelled_offline(capsys, tmp_path):
    badge_file = tmp_path / "badge.json"
    code = main(["--modules", "1", "--agent", PASSES, "--badge-file", str(badge_file)])

    assert code == 0
    badge = _badge(badge_file)
    assert set(badge) == SHIELDS_KEYS
    assert badge["schemaVersion"] == 1
    assert badge["label"] == "course Checks · offline"
    assert badge["message"] == "through module 1 · 1 Check passed"
    out = capsys.readouterr().out
    assert "https://img.shields.io/endpoint?url=" in out
    assert "]({})".format(COURSE_URL) in out


def test_an_offline_badge_counts_only_checks_that_ran_and_names_the_live_only_ones(tmp_path):
    from tests.fixtures.checks_cli_suites import SUITES

    badge_file = tmp_path / "badge.json"
    result = _quiet(through=2, agent=PASSES, suites=SUITES, badge_file=badge_file)

    # Module 2 has a passing Check and a live-only judge that Offline skips:
    # two Checks ran and passed, and the judge is named, not counted.
    assert result.badge_module == 2
    badge = _badge(badge_file)
    assert badge["label"] == "course Checks · offline"
    assert badge["message"] == (
        "through module 2 · 2 Checks passed · not run offline: reply is polite")


def test_a_live_run_is_not_labelled_offline(tmp_path):
    from company.runner import RECORDINGS_DIR
    from llm.replay import ReplayTransport

    badge_file = tmp_path / "badge.json"
    _quiet(through=1, agent=PASSES, mode="live", suites=(_module_1(),),
           transport=ReplayTransport(RECORDINGS_DIR / "upgrade-to-pro.recording.json"),
           badge_file=badge_file)

    badge = _badge(badge_file)
    assert badge["label"] == "course Checks · live"
    assert "offline" not in badge["message"]


def test_the_badge_never_claims_a_module_whose_checks_failed(tmp_path):
    badge_file = tmp_path / "badge.json"
    result = _quiet(through=3, agent=PASSES,
                    suites=(_module_1(), _suite(2, _always_fails), _suite(3, _passes())),
                    badge_file=badge_file)

    assert result.report.passed_through == 1
    assert _badge(badge_file)["message"] == "through module 1 · 1 Check passed"


def test_a_module_whose_checks_were_all_skipped_is_not_claimed(tmp_path):
    from tests.fixtures.checks_cli_suites import reply_is_polite

    badge_file = tmp_path / "badge.json"
    result = _quiet(through=3, agent=PASSES,
                    suites=(_module_1(), _suite(2, reply_is_polite), _suite(3, _passes())),
                    badge_file=badge_file)

    # The Checks CLI does not fail a skip, so it says module 3; the badge,
    # which counts only Checks that ran, stops where nothing ran.
    assert result.report.passed_through == 3
    assert result.badge_module == 1
    assert _badge(badge_file)["message"].startswith("through module 1 · 1 Check passed")


def test_the_badge_names_only_the_milestones_that_passed(tmp_path):
    badge_file = tmp_path / "badge.json"
    _quiet(through=4, agent=PASSES,
           suites=(_module_1(), _suite(2, _passes()), _suite(3, _passes()), _suite(4, _passes())),
           badge_file=badge_file)

    message = _badge(badge_file)["message"]
    assert message == "Graded ✓ — through module 4 · 4 Checks passed"


def test_an_unresolved_case_blocks_the_graded_badge_and_share_line(capsys, tmp_path):
    # Upgrades as asked, then gives the customer an empty reply: Spine 1's Check
    # passes and the Knowledge suite does not run the Case, but it is not resolved.
    badge_file = tmp_path / "badge.json"
    code = main(["--agent", "tests.fixtures.graded_agents:upgrades_then_says_nothing",
                 "--badge-file", str(badge_file)])

    out = capsys.readouterr().out
    lines = _lines(out)
    assert code == 1, out
    assert "FAIL  upgrade-to-pro: the agent finishes the Case." in out
    assert lines[-2].startswith("Passed through module 2 of ")
    message = _badge(badge_file)["message"]
    assert message.startswith("through module 2 · ") and "Graded" not in message
    assert "Spine module 2," in lines[-1] and "Graded" not in lines[-1]


def test_a_warning_never_counts_against_the_badge_or_towards_it(tmp_path):
    from checks import CheckResult

    def warns(outcome):
        return CheckResult.warn("names itself", "No name on the agent span.")

    badge_file = tmp_path / "badge.json"
    suite = _suite(1, _passes(), warns)

    result = _quiet(through=1, agent=PASSES, suites=(suite,), badge_file=badge_file)

    assert result.report.passed is True
    assert result.badge_module == 1
    assert "1 Check passed" in _badge(badge_file)["message"]  # the warning is not a pass


def test_a_module_with_only_warnings_is_not_claimed(tmp_path):
    from checks import CheckResult

    def warns(outcome):
        return CheckResult.warn("names itself", "No name on the agent span.")

    result = _quiet(through=1, agent=PASSES, suites=(_suite(1, warns),),
                    badge_file=tmp_path / "badge.json")

    assert result.report.passed is True
    assert result.badge_module == 0  # a warning proved nothing, like a skip


def test_the_grader_links_each_warning_to_its_lesson(capsys, tmp_path):
    main(["--modules", "4", "--agent", "flagship.knowledge:run",
          "--badge-file", str(tmp_path / "badge.json")])

    from checks.grader import check_link

    out = capsys.readouterr().out
    assert "Warnings, which fail nothing:" in out
    assert "the agent span names the agent" in out
    link = check_link(4, "the_agent_span_names_the_agent")
    assert link.startswith(BLOB + "spine/04-observed/README.md#")
    assert link in out
    _heading_exists(link)


def test_a_failing_run_writes_no_new_badge(tmp_path):
    badge_file = tmp_path / "badge.json"
    main(["--modules", "1", "--agent", FAILING, "--badge-file", str(badge_file)])

    assert not badge_file.exists()


def test_a_failing_run_takes_back_an_earlier_badge_rather_than_leave_its_claim(tmp_path):
    badge_file = tmp_path / "badge.json"
    main(["--modules", "1", "--agent", PASSES, "--badge-file", str(badge_file)])
    main(["--modules", "1", "--agent", FAILING, "--badge-file", str(badge_file)])

    badge = _badge(badge_file)
    assert set(badge) == SHIELDS_KEYS
    assert badge["message"] == "not passing"


def test_the_reference_agent_earns_no_badge(capsys, tmp_path):
    badge_file = tmp_path / "badge.json"
    code = main(["--modules", "1", "--agent", "flagship.loop:run",
                 "--badge-file", str(badge_file)])

    assert code == 0
    assert not badge_file.exists()
    assert "reference" in capsys.readouterr().out.lower()


def _reference_forms(tmp_path):
    """Every way to name the reference Flagship Agent, or hand it over."""
    import flagship.loop

    reexport = tmp_path / "my_agent.py"
    reexport.write_text("from flagship.loop import run  # noqa: F401\n", encoding="utf-8")
    return {
        "dotted": "flagship.loop:run",
        "path": str(ROOT / "flagship" / "loop.py") + ":run",
        "relative path": "flagship/loop.py:run",
        "dot-slash path": "./flagship/loop.py:run",
        "callable": flagship.loop.run,
        "a Reader file that re-exports it": str(reexport) + ":run",
    }


@pytest.mark.parametrize("form", [
    "dotted", "path", "relative path", "dot-slash path", "callable",
    "a Reader file that re-exports it",
])
def test_the_reference_agent_earns_no_badge_however_it_is_named(
        monkeypatch, tmp_path, form):
    # The Grader decides by where the agent's code lives, not how it is spelled.
    monkeypatch.chdir(ROOT)
    agent = _reference_forms(tmp_path)[form]
    badge_file = tmp_path / "badge.json"

    result = _quiet(through=1, agent=agent, suites=(_module_1(),), badge_file=badge_file)

    assert result.report.passed is True
    assert result.badge_module == 0
    assert not badge_file.exists()


def test_a_readers_file_outside_flagship_still_earns_a_badge(monkeypatch, tmp_path):
    # Named like the reference agent's file, but it is the Reader's own code.
    mine = tmp_path / "flagship" / "loop.py"
    mine.parent.mkdir()
    mine.write_text((ROOT / "tests" / "fixtures" / "grader_agents.py").read_text(
        encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    badge_file = tmp_path / "badge.json"

    result = _quiet(through=1, agent="flagship/loop.py:upgrades_without_a_model",
                    suites=(_module_1(),), badge_file=badge_file)

    assert result.badge_module == 1
    assert badge_file.exists()


def test_the_badge_markdown_points_at_the_readers_github_repo(capsys, tmp_path):
    import subprocess

    repo = tmp_path / "my-agent"
    (repo / "badges").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                    "git@github.com:reader/my-agent.git"], check=True)

    main(["--modules", "1", "--agent", PASSES,
          "--badge-file", str(repo / "badges" / "checks.json")])

    raw = "https%3A%2F%2Fraw.githubusercontent.com%2Freader%2Fmy-agent%2FHEAD%2Fbadges%2Fchecks.json"
    assert "https://img.shields.io/endpoint?url={})]({})".format(raw, COURSE_URL) in (
        capsys.readouterr().out)


# ── The summary and the share line ────────────────────────────────────────────


def test_the_share_line_follows_the_passed_through_line(capsys, tmp_path):
    main(["--modules", "1", "--agent", PASSES, "--badge-file", str(tmp_path / "b.json")])

    lines = _lines(capsys.readouterr().out)
    assert lines[-2] == "Passed through module 1 of 1."
    assert lines[-1] == (
        "Share: My agent passed 1 course Check through Spine module 1, offline. "
        + COURSE_URL)


def test_the_share_line_claims_what_the_badge_does_and_no_more(tmp_path):
    from tests.fixtures.checks_cli_suites import SUITES

    result = _quiet(through=2, agent=PASSES, suites=SUITES, badge_file=None)

    assert result.share == (
        "Share: My agent passed 2 course Checks through Spine module 2, offline "
        "(not run offline: reply is polite). " + COURSE_URL)


def test_the_share_line_names_no_milestone_a_failure_blocks(tmp_path):
    result = _quiet(through=4, agent=PASSES,
                    suites=(_module_1(), _suite(2, _passes()), _suite(3, _always_fails),
                            _suite(4, _passes())),
                    badge_file=None)

    assert "module 2" in result.share
    for word in ("Graded", "Attacked", "Budgeted"):
        assert word not in result.share
    assert result.share.endswith(COURSE_URL)


def test_a_run_that_passes_nothing_still_shares_the_course(tmp_path):
    result = _quiet(through=1, agent=FAILING, suites=(_module_1(),), badge_file=None)

    assert "passed" not in result.share
    assert result.share.startswith("Share: ") and result.share.endswith(COURSE_URL)


# ── The skill ─────────────────────────────────────────────────────────────────

SKILL = ROOT / "skills" / "grader" / "SKILL.md"
INSTALL = "npx skills add dnakhoa/llm-engineering-playground --skill grader"


def _frontmatter(text):
    """The Agent Skills frontmatter: ``key: value`` lines between two ``---``."""
    assert text.startswith("---\n"), "SKILL.md starts with YAML frontmatter"
    block, _, body = text[4:].partition("\n---\n")
    fields = {}
    for line in block.splitlines():
        if line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields, body


def test_the_skill_follows_the_agent_skills_format():
    # https://agentskills.io/specification: name matches the folder, lowercase
    # letters, digits and single hyphens, at most 64; a description of at most 1024.
    import re

    fields, body = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert fields["name"] == SKILL.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    assert len(fields["name"]) <= 64
    assert 0 < len(fields["description"]) <= 1024
    assert set(fields) <= {"name", "description", "license", "compatibility",
                           "metadata", "allowed-tools"}
    assert body.strip()


def _section(body, number):
    """The body of the skill's ``## <number>.`` section."""
    import re

    match = re.search(r"^## {}\. .*?$(.*?)(?=^## |\Z)".format(number), body, re.M | re.S)
    assert match, "SKILL.md has a section {}".format(number)
    return match.group(1)


def _bash_blocks(text):
    """Each ```bash block in ``text``: (where it starts, its commands)."""
    import re

    return [(m.start(), m.group(1))
            for m in re.finditer(r"^```bash\n(.*?)^```", text, re.M | re.S)]


def _prose(text):
    """Text on one line, whatever the line breaks: a sentence can wrap anywhere."""
    return " ".join(text.split())


def test_the_skill_runs_the_grader_command_offline_unless_the_reader_asks():
    import re

    fields, body = _frontmatter(SKILL.read_text(encoding="utf-8"))
    run = _section(body, 3)
    blocks = _bash_blocks(run)

    # The command a coding agent runs by default is section 3's first: Offline.
    assert blocks, "section 3 shows the command to run"
    first_at, default = blocks[0]
    assert "python -m checks.grader" in default
    assert "--mode" not in default, default
    assert "Offline is the default" in _prose(run[:first_at])
    # Going live is conditioned on the Reader asking, and never the agent's idea.
    prose = _prose(run)
    assert "Go live only when the Reader asks for a live run in this conversation." in prose
    assert "Never add `--mode live` on your own" in prose
    ask = re.search(r"Go\s+live\s+only\s+when\s+the\s+Reader\s+asks", run)
    # A live command appears only after that rule, and in no other section.
    live = [at for at, commands in blocks if "--mode live" in commands]
    assert live and all(at > ask.start() for at in live)
    for number in (1, 2, 4, 5, 6):
        assert not any("--mode live" in commands
                       for _, commands in _bash_blocks(_section(body, number))), number
    assert "Offline by default" in fields["description"]
    assert "live only when the Reader explicitly asks" in fields["description"]


def test_the_skill_ends_on_the_summary_and_the_share_line():
    _, body = _frontmatter(SKILL.read_text(encoding="utf-8"))

    assert "Passed through module N of M." in body
    assert "Share:" in body


def test_the_readme_shows_the_one_command_install_above_the_fold():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_section = readme.index("\n## ")

    assert INSTALL in readme[:first_section]
    assert "git clone" not in readme[:first_section]
    assert INSTALL in SKILL.read_text(encoding="utf-8")


def test_the_skill_reports_a_missing_package_as_setup_not_as_the_agent_failing():
    # Exit code 2 with a pip line means the Checks never ran: telling the
    # Reader "your agent failed" there would blame their agent for a package.
    _, body = _frontmatter(SKILL.read_text(encoding="utf-8"))
    setup = _prose(_section(body, 1))

    assert "exits 2" in setup
    assert "pip install" in setup
    assert "pip install -r requirements.txt" in setup
    assert "never say the agent failed" in setup.lower()
    assert "opentelemetry-sdk" in setup and "module 4" in setup.lower()
