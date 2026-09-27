"""The Grader's command: run the Checks, explain each failure, and say what passed.

    python -m checks.grader --modules 2 --agent path/to/my_agent.py:run
    python -m checks.grader --modules 2 --agent path/to/my_agent.py:run --mode live --spend-cap 0.50

It takes every flag ``python -m checks`` takes and runs the same Checks the same
way: Offline unless ``--mode live`` is given. After the per-Check lines it adds
what the Grader skill (``skills/grader/SKILL.md``) relays to the Reader:

- each failure, with a link to the Spine lesson section that covers it. The
  link is found from the Check's own name: the lesson section that names it,
  else its module's section on its Checks. No table of Checks is kept here, so
  a new suite's Checks get links by existing;
- once the Checks pass, a shields.io endpoint badge file (``--badge-file``)
  that counts only the Checks that ran, is labelled offline or live, and on an
  Offline run names the live-only Checks it did not run;
- the "Passed through module N of M." line;
- one line of share text with the course link.

Nothing here asks a question on the terminal or goes live on its own.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

from company.runner import LIVE, OFFLINE, Agent, load_agent

from . import cli, is_live_only
from .cli import (
    DEFAULT_AGENT,
    DEFAULT_SPEND_CAP_USD,
    ERROR,
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckRun,
    Report,
    _model_and_base_url,
    _parser,
    parse_modules,
    run_checks,
)
from .suites import Suite, discover_suites

COURSE_URL = "https://github.com/dnakhoa/llm-engineering-playground"
ROOT = Path(__file__).resolve().parent.parent
SPINE = ROOT / "spine"
#: The reference Flagship Agent's code: an agent from here earns no badge.
FLAGSHIP = ROOT / "flagship"
DEFAULT_BADGE_FILE = "grader-badge.json"

#: The thesis's three milestones and the Spine module that earns each.
MILESTONES: Tuple[Tuple[int, str], ...] = ((3, "Graded"), (5, "Attacked"), (6, "Budgeted"))

#: A lesson's section on its own Checks: "The Check", "The Checks", "The Graded suite".
_CHECKS_HEADING = re.compile(r"The (\w+ )?(Checks?|suite)\b")

#: An error is the run failing, not a Check: where each kind is taught, as
#: (Spine module, lesson heading). A test holds each heading to its lesson.
AGENT_RAISED = (1, "The agent contract")
REPLAY_MISMATCH = (1, "Offline: why this costs nothing")
LIVE_ERROR = (1, "Run the Checks on your agent")


# ── Lesson links ──────────────────────────────────────────────────────────────


def anchor(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def _lessons() -> List[Tuple[int, Path]]:
    """Every ``spine/NN-<name>/README.md``, with its module number, in order."""
    found = []
    for readme in sorted(SPINE.glob("[0-9][0-9]-*/README.md")):
        found.append((int(readme.parent.name[:2]), readme))
    return found


def lesson_readme(module: int) -> Optional[Path]:
    """``spine/NN-<name>/README.md`` for Spine module ``module``, if it has one."""
    return next((readme for number, readme in _lessons() if number == module), None)


def sections(readme: Path) -> List[Tuple[str, str]]:
    """A lesson's level-two sections, (heading, body), outside code blocks."""
    found: List[Tuple[str, List[str]]] = []
    fenced = False
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            found.append((line[3:].strip(), []))
        elif not fenced and found:
            found[-1][1].append(line)
    return [(heading, "\n".join(body)) for heading, body in found]


def headings(readme: Path) -> List[str]:
    """The level-two headings of a lesson, outside code blocks."""
    return [heading for heading, _ in sections(readme)]


def _plain(text: str) -> str:
    """Prose without Markdown emphasis or code marks, lowercase, on one line."""
    return " ".join(re.sub(r"[*_`]", " ", text).lower().split())


def _link(readme: Path, heading: Optional[str]) -> str:
    url = "{}/blob/main/spine/{}".format(COURSE_URL, readme.relative_to(SPINE).as_posix())
    return url + "#" + anchor(heading) if heading else url


def lesson_link(module: int, heading: str) -> str:
    """A course link to ``heading`` in Spine module ``module``'s lesson."""
    readme = lesson_readme(module)
    if readme is None:
        raise FileNotFoundError("There is no lesson for Spine module {}.".format(module))
    return _link(readme, heading)


def check_link(module: int, name: str) -> str:
    """Where the Check called ``name`` (a function name) is taught.

    The first lesson section that names the Check in prose, looking in its own
    module's lesson first; else that lesson's section on its Checks; else the
    lesson itself. It reads the lessons each time, so a Check a lesson starts
    naming, or a new module's suite, needs no change here.
    """
    words = re.compile(r"\b{}\b".format(re.escape(_plain(name.replace("_", " ")))))
    lessons = _lessons()
    own = [readme for number, readme in lessons if number == module]
    for readme in own + [readme for number, readme in lessons if number != module]:
        for heading, body in sections(readme):
            if words.search(_plain(body)):
                return _link(readme, heading)
    if not own:
        return "{}/tree/main/spine".format(COURSE_URL)
    checks_heading = next(
        (h for h in headings(own[0]) if _CHECKS_HEADING.match(h)), None)
    return _link(own[0], checks_heading)


def explain(result: CheckRun, mode: str) -> str:
    """The lesson link for one failed, errored or warning Check."""
    if result.status == ERROR:
        if result.detail.startswith("The agent raised"):
            return lesson_link(*AGENT_RAISED)
        return lesson_link(*(REPLAY_MISMATCH if mode == OFFLINE else LIVE_ERROR))
    return check_link(result.module, result.name)


# ── The badge and the share line ──────────────────────────────────────────────


@dataclass(frozen=True)
class Claim:
    """What a run earned: modules 1 to ``module``, on Checks that ran."""

    module: int
    mode: str
    #: Checks that ran and passed in the claimed modules. A skip never counts.
    passed: int = 0
    #: Live-only Checks in the claimed modules an Offline run did not run.
    not_run: Tuple[str, ...] = ()


def claim(report: Report, mode: str, suites: Sequence[Suite]) -> Claim:
    """The last module N such that modules 1 to N each had a Check that ran and
    passed, and none that did anything but pass, warn or skip.

    Stricter than "Passed through module N": a skipped Check proved nothing, so
    a module whose every Check was skipped is not claimed, and the count is of
    the Checks that ran and passed. A warning neither blocks a claim nor counts
    towards one.
    """
    module = 0
    for number in range(1, report.through + 1):
        mine = [r for r in report.results if r.module == number]
        if not any(r.status == PASS for r in mine):
            break
        if any(r.status not in (PASS, WARN, SKIP) for r in mine):
            break
        module = number
    claimed = [r for r in report.results if r.module <= module]
    live_only = {check.__name__ for suite in suites for check in suite.checks
                 if is_live_only(check)}
    not_run: List[str] = []
    for result in claimed:
        label = result.name.replace("_", " ")
        if result.status == SKIP and result.name in live_only and label not in not_run:
            not_run.append(label)
    return Claim(module=module, mode=mode,
                 passed=sum(1 for r in claimed if r.status == PASS),
                 not_run=tuple(not_run))


def milestones(module: int) -> List[str]:
    """The milestone names up to and including ``module``: Graded, Attacked, Budgeted."""
    return [name for at, name in MILESTONES if at <= module]


def _checks(count: int) -> str:
    return "{} Check{}".format(count, "" if count == 1 else "s")


def _not_run(earned: Claim) -> str:
    return "not run offline: {}".format(", ".join(earned.not_run))


def badge(earned: Claim) -> Dict[str, object]:
    """A shields.io endpoint badge for ``earned``, and nothing more.

    It holds only the endpoint schema's keys, which shields.io checks:
    https://shields.io/badges/endpoint-badge
    """
    label = "course Checks · {}".format(earned.mode)
    if earned.module == 0:
        return {"schemaVersion": 1, "label": label, "message": "not passing",
                "color": "lightgrey"}
    names = " · ".join("{} ✓".format(name) for name in milestones(earned.module))
    parts = ["through module {}".format(earned.module), _checks(earned.passed) + " passed"]
    if earned.not_run:
        parts.append(_not_run(earned))
    message = " · ".join(parts)
    return {
        "schemaVersion": 1,
        "label": label,
        "message": "{} — {}".format(names, message) if names else message,
        "color": "brightgreen" if earned.mode == LIVE else "blue",
    }


def _raw_url(badge_file: Path) -> Optional[str]:
    """The raw.githubusercontent.com URL the committed badge file will have, if
    the file sits in a git checkout whose origin is on GitHub."""
    folder = badge_file.resolve().parent
    try:
        top = subprocess.run(["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
        remote = subprocess.run(["git", "-C", str(folder), "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?$", remote.stdout.strip())
    if top.returncode != 0 or remote.returncode != 0 or not match:
        return None
    path = badge_file.resolve().relative_to(Path(top.stdout.strip()).resolve()).as_posix()
    return "https://raw.githubusercontent.com/{}/{}/HEAD/{}".format(
        match.group(1), match.group(2), path)


def badge_markdown(raw_url: Optional[str]) -> str:
    """The Markdown for a README: the endpoint badge, linking back to the course."""
    url = quote(raw_url, safe="") if raw_url else "<raw URL of your committed badge file>"
    return "[![course Checks](https://img.shields.io/endpoint?url={})]({})".format(
        url, COURSE_URL)


def _is_reference(agent: Agent) -> bool:
    """Whether ``agent``'s code is the reference Flagship Agent's.

    Decided by where the resolved callable's code lives, not by how it was
    named: its module's file is inside this repo's ``flagship/`` folder. So
    ``flagship.loop:run``, ``flagship/loop.py:run``, the function itself and a
    Reader's file that only re-exports it are all the reference agent.
    """
    target = inspect.unwrap(agent)
    target = getattr(target, "func", target)  # a functools.partial
    target = getattr(target, "__func__", target)  # a bound method
    # A function's own globals are its module's, even for a Reader's file that
    # another file of the same name has since replaced in sys.modules.
    source = getattr(target, "__globals__", {}).get("__file__")
    if not source:
        module = sys.modules.get(getattr(target, "__module__", None) or "")
        source = getattr(module, "__file__", None)
    if not source:
        return False
    return FLAGSHIP in Path(source).resolve().parents


def share_text(earned: Claim) -> str:
    """One line the Reader can paste into a post. It claims what the badge does."""
    if earned.module == 0:
        return ("Share: I'm building an agent that has to pass the course Checks, "
                "one Spine module at a time: {}".format(COURSE_URL))
    names = " · ".join("{} ✓".format(name) for name in milestones(earned.module))
    return "Share: My agent passed {} through Spine module {}{}, {}{}. {}".format(
        _checks(earned.passed).replace(" ", " course ", 1),
        earned.module,
        " (" + names + ")" if names else "",
        earned.mode,
        " (" + _not_run(earned) + ")" if earned.not_run else "",
        COURSE_URL,
    )


# ── The run ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Grade:
    """A Grader run: the Checks' report, and what the Grader added to it."""

    report: Report
    mode: str
    badge_module: int
    badge: Optional[Dict[str, object]]
    share: str


def grade(
    *,
    through: int,
    agent: Union[str, Agent] = DEFAULT_AGENT,
    mode: str = OFFLINE,
    spend_cap_usd: float = DEFAULT_SPEND_CAP_USD,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    suites: Optional[Sequence[Suite]] = None,
    transport=None,
    badge_file: Union[str, Path, None] = DEFAULT_BADGE_FILE,
    out: Optional[Callable[[str], None]] = None,
) -> Grade:
    """Run the Checks through module ``through``, explain what failed, and
    write the result badge to ``badge_file`` (``None``: write none).

    The badge is written once a module passes. A run that passes nothing
    creates no badge, but replaces one an earlier run wrote, so a committed
    badge never keeps claiming a module that now fails. The reference agent
    earns no badge: it is not the Reader's work.
    """
    write = out or (lambda line: print(line, flush=True))
    selected = tuple(suites) if suites is not None else discover_suites()
    lines: List[str] = []
    kwargs = {} if model is None else {"model": model}
    # Loaded once, so a Reader's file runs once, and the badge decision below
    # is about the very code the Checks graded.
    run_agent = load_agent(agent)
    label = agent if isinstance(agent, str) else getattr(agent, "__name__", repr(agent))
    report = run_checks(
        through=through, agent=run_agent, agent_name=label, mode=mode,
        spend_cap_usd=spend_cap_usd, base_url=base_url, suites=selected,
        transport=transport, out=lines.append, **kwargs,
    )
    # The Checks' own summary line moves to the end, just above the share text.
    summary = lines.pop() if lines and lines[-1].startswith("Passed through") else None
    for line in lines:
        write(line)

    failures = [r for r in report.results if r.status in (FAIL, ERROR)]
    if failures:
        write("")
        write("What to read next:")
        for result in failures:
            write("  {}: {}".format(result.case_id, result.name.replace("_", " ")))
            write("    {}".format(explain(result, mode)))
    warnings = [r for r in report.results if r.status == WARN]
    if warnings:
        write("")
        write("Warnings, which fail nothing:")
        for result in warnings:
            write("  {}: {}".format(result.case_id, result.name.replace("_", " ")))
            write("    {}".format(explain(result, mode)))

    earned = claim(report, mode, selected)
    written: Optional[Dict[str, object]] = None
    if badge_file is not None:
        path = Path(badge_file)
        write("")
        if _is_reference(run_agent):
            earned = Claim(module=0, mode=mode)
            write("No badge: {} is the reference Flagship Agent, not yours. "
                  "Point --agent at your own agent to earn one.".format(label))
        elif earned.module or path.exists():
            written = badge(earned)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(written, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            if earned.module:
                write("Badge: wrote {} ({}: {}). Commit it, then add to your README:".format(
                    path, written["label"], written["message"]))
                write("  " + badge_markdown(_raw_url(path)))
            else:
                write("Badge: {} now says \"not passing\"; it claimed a module this run "
                      "did not pass.".format(path))
        else:
            write("No badge yet: it is written once module 1 has a Check that ran and passed.")

    share = share_text(earned)
    write("")
    if summary:
        write(summary)
    write(share)
    return Grade(report=report, mode=mode, badge_module=earned.module, badge=written,
                 share=share)


# ── Command line ──────────────────────────────────────────────────────────────


def _grader_parser() -> argparse.ArgumentParser:
    parser = _parser()
    parser.prog = "python -m checks.grader"
    parser.description = "Run the Checks, explain each failure, and write a result badge."
    parser.add_argument(
        "--badge-file", default=DEFAULT_BADGE_FILE, metavar="PATH",
        help="where to write the shields.io endpoint badge once the Checks pass "
             "(default: {})".format(DEFAULT_BADGE_FILE))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The same exit codes as ``python -m checks``."""
    args = _grader_parser().parse_args(argv)
    try:
        cli.load_env()
        model, base_url = _model_and_base_url(args)
        suites = discover_suites()
        through = parse_modules(args.modules) if args.modules else suites[-1].module
        result = grade(
            through=through, agent=args.agent, mode=args.mode,
            spend_cap_usd=args.spend_cap, model=model, base_url=base_url, suites=suites,
            badge_file=args.badge_file,
        )
    except cli.START_ERRORS as error:
        print("checks: {}".format(cli.start_error(error)), file=sys.stderr)
        return 2
    if result.report.stopped_reason:
        return 2
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
