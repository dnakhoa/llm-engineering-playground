"""
Internal link check — finds relative links in Markdown and notebook markdown cells
whose target file or folder does not exist (see tests/test_repo_layout.py).

What counts as a link:
    [text](target)            inline Markdown links and images
    href="target" / src="…"   HTML attributes inside Markdown

External links (http, https, mailto, …) and in-page anchors (`#section`) are not
checked. Fenced code blocks and inline code spans are skipped: code samples
describe the Reader's own project, not files in this repo.

Run: python tests/link_check.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
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

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
HTML_LINK_RE = re.compile(r"\b(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
EXTERNAL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


@dataclass(frozen=True)
class DanglingLink:
    """A relative link whose target does not exist."""

    path: str
    line: int
    target: str
    cell: Optional[int] = None

    def format(self) -> str:
        where = f"{self.path}:{self.line}" if self.cell is None else f"{self.path}:cell {self.cell}:{self.line}"
        return f"{where}: dangling link → {self.target}"


def iter_links(text: str) -> Iterator[tuple[int, str]]:
    """Yield (line number, target) for every link outside code."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        prose = INLINE_CODE_RE.sub("", line)
        for regex in (MARKDOWN_LINK_RE, HTML_LINK_RE):
            for match in regex.finditer(prose):
                yield number, match.group(1)


def _is_internal(target: str) -> bool:
    return not (target.startswith("#") or EXTERNAL_RE.match(target) or target.startswith("//"))


def resolve(source: Path, target: str, root: Path) -> Path:
    """The file a relative link points at. A leading `/` means the repo root."""
    path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if path_part.startswith("/"):
        return root / path_part.lstrip("/")
    return source.parent / path_part


def check_text(
    text: str, source: Path, root: Path, cell: Optional[int] = None
) -> list[DanglingLink]:
    relative = source.resolve().relative_to(root.resolve()).as_posix()
    dangling = []
    for line, target in iter_links(text):
        if not _is_internal(target):
            continue
        if not resolve(source, target, root).exists():
            dangling.append(DanglingLink(path=relative, line=line, target=target, cell=cell))
    return dangling


def check_file(path: Path, root: Path = REPO_ROOT) -> list[DanglingLink]:
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return check_text(text, path, root)
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return []
    dangling = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        dangling.extend(check_text(source, path, root, cell=index))
    return dangling


def iter_documents(root: Path = REPO_ROOT) -> Iterator[Path]:
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".ipynb"}:
            continue
        if set(path.relative_to(root).parts) & EXCLUDED_DIR_NAMES:
            continue
        yield path


def check_repo(root: Path = REPO_ROOT) -> list[DanglingLink]:
    dangling = []
    for path in iter_documents(root):
        dangling.extend(check_file(path, root))
    return dangling


def main() -> int:
    dangling = check_repo()
    for link in dangling:
        print(link.format())
    print(f"{len(dangling)} dangling internal link(s).")
    return 1 if dangling else 0


if __name__ == "__main__":
    raise SystemExit(main())
