"""
Scanner behind the stale-reference lint (see tests/test_stale_references.py).

Data files:
    tests/stale_lint_denylist.json  — retired model IDs and dead API patterns, each
                                      with a replacement hint.
    tests/stale_lint_baseline.txt   — files that already fail, one relative path per
                                      line. Their matches are reported as baselined so
                                      CI stays green. The baseline shrinks as the
                                      Appendix refresh lands and is deleted by ticket 17.

A match is allowed when its line carries the marker `historical-exception` inside a
comment, written in whatever comment syntax the file uses:

    <!-- historical-exception -->        Markdown
    # historical-exception               Python
    // historical-exception              TypeScript

Notebooks are read as JSON — no nbformat dependency — and each cell is scanned with
line numbers relative to that cell.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DENYLIST_PATH = Path(__file__).resolve().parent / "stale_lint_denylist.json"
BASELINE_PATH = Path(__file__).resolve().parent / "stale_lint_baseline.txt"

HISTORICAL_EXCEPTION_MARKER = "historical-exception"

#: The marker only counts inside a comment, so prose that merely names it — this
#: module, the ticket, the spec — is not itself an exception.
HISTORICAL_EXCEPTION_RE = re.compile(
    r"(?:<!--|#|//|/\*|\*|;)\s*" + HISTORICAL_EXCEPTION_MARKER, re.IGNORECASE
)

SCANNED_SUFFIXES = frozenset({".md", ".py", ".ts", ".tsx", ".ipynb"})

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".github",
        ".claude",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "dist",
        "build",
    }
)

#: The lint's own files: they quote denylisted references on purpose.
EXCLUDED_PATHS = frozenset(
    {
        "tests/stale_lint.py",
        "tests/test_stale_references.py",
        "tests/stale_lint_denylist.json",
        "tests/stale_lint_baseline.txt",
        "tests/fixtures/stale_lint",
    }
)


@dataclass(frozen=True)
class DenylistEntry:
    """One retired model ID or dead API, with the replacement to reach for instead."""

    id: str
    pattern: "re.Pattern[str]"
    hint: str
    description: str = ""


@dataclass(frozen=True)
class Violation:
    """A denylisted reference found at a specific place."""

    path: str
    line: int
    entry_id: str
    hint: str
    excerpt: str
    cell: Optional[int] = None

    @property
    def location(self) -> str:
        if self.cell is None:
            return f"{self.path}:{self.line}"
        return f"{self.path}:cell {self.cell}:{self.line}"

    def format(self) -> str:
        return f"{self.location}: [{self.entry_id}] {self.excerpt} → {self.hint}"


def load_denylist(path: Path = DENYLIST_PATH) -> tuple[DenylistEntry, ...]:
    """Read the denylist data file and compile its patterns."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in raw["entries"]:
        entries.append(
            DenylistEntry(
                id=item["id"],
                pattern=re.compile(item["pattern"]),
                hint=item["hint"],
                description=item.get("description", ""),
            )
        )
    return tuple(entries)


def load_baseline(path: Path = BASELINE_PATH) -> frozenset[str]:
    """Read the baseline: relative paths of files that already fail."""
    if not path.exists():
        return frozenset()
    paths = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            paths.append(line)
    return frozenset(paths)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_at(text: str, line_number: int) -> str:
    lines = text.splitlines()
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def _is_excepted(line: str) -> bool:
    return HISTORICAL_EXCEPTION_RE.search(line) is not None


def scan_text(
    text: str,
    entries: Sequence[DenylistEntry],
    path: str,
    cell: Optional[int] = None,
) -> list[Violation]:
    """Find every denylisted reference in `text` that is not marked as historical.

    Patterns may span lines; a match is reported at the line it starts on, and that
    line is the one that carries — or does not carry — the historical-exception marker.
    """
    violations = []
    for entry in entries:
        for match in entry.pattern.finditer(text):
            line_number = _line_of(text, match.start())
            line = _line_at(text, line_number)
            if _is_excepted(line):
                continue
            violations.append(
                Violation(
                    path=path,
                    line=line_number,
                    entry_id=entry.id,
                    hint=entry.hint,
                    excerpt=line.strip()[:120] or match.group(0).strip()[:120],
                    cell=cell,
                )
            )
    violations.sort(key=lambda v: (v.line, v.entry_id))
    return violations


def _scan_notebook(text: str, entries: Sequence[DenylistEntry], path: str) -> list[Violation]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        # A notebook that is not valid JSON is the notebook smoke test's problem.
        return []
    violations = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        violations.extend(scan_text(source, entries, path=path, cell=index))
    return violations


def scan_file(
    file_path: Path,
    entries: Sequence[DenylistEntry],
    root: Path = REPO_ROOT,
) -> list[Violation]:
    """Scan one file. Paths in the result are relative to `root`."""
    file_path = Path(file_path)
    relative = file_path.resolve().relative_to(Path(root).resolve()).as_posix()
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if file_path.suffix == ".ipynb":
        return _scan_notebook(text, entries, path=relative)
    return scan_text(text, entries, path=relative)


def _is_excluded(relative: str) -> bool:
    return any(
        relative == excluded or relative.startswith(excluded + "/")
        for excluded in EXCLUDED_PATHS
    )


def iter_scannable_files(root: Path = REPO_ROOT) -> Iterator[Path]:
    """Every Markdown, Python, TypeScript and notebook file the lint covers."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if set(relative.parts) & EXCLUDED_DIR_NAMES:
            continue
        if _is_excluded(relative.as_posix()):
            continue
        yield path


def scan_repo(
    root: Path = REPO_ROOT,
    entries: Optional[Sequence[DenylistEntry]] = None,
    baseline: Optional[Iterable[str]] = None,
) -> list[Violation]:
    """Scan the repo, skipping files listed in `baseline`.

    Pass `baseline=frozenset()` to see every violation, baselined ones included.
    """
    entries = load_denylist() if entries is None else entries
    skip = load_baseline() if baseline is None else frozenset(baseline)
    violations = []
    for path in iter_scannable_files(root):
        relative = path.relative_to(Path(root)).as_posix()
        if relative in skip:
            continue
        violations.extend(scan_file(path, entries, root=root))
    return violations


def main() -> int:
    """`python tests/stale_lint.py` — print the current violations and the baseline size."""
    baseline = load_baseline()
    violations = scan_repo(baseline=baseline)
    for violation in violations:
        print(violation.format())
    print(f"{len(violations)} stale reference(s) outside a baseline of {len(baseline)} file(s).")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
