"""
Appendix path setup — every Appendix script and notebook still reaches the provider
layer (llm/), shared/ and the root .env from its own folder, now that the topics sit
one level deeper in appendix/.

The Appendix code cannot run in CI: it calls the providers and needs their SDKs. So
this test reads each script and each notebook code cell, evaluates the very path
expressions the file hands to sys.path.insert / sys.path.append / load_dotenv, and asks
Python's import machinery whether the shared/ modules the file imports load through
those entries (and the same for llm/). A script is evaluated with its own __file__. A notebook is evaluated
against its own folder, which is the working directory Jupyter gives it.

Nothing else notices a path one level off. load_dotenv returns False silently, so the
notebook fails later on a missing API key. A script fails only when someone runs it,
with ModuleNotFoundError: No module named 'provider'.

Run: pytest tests/test_appendix_paths.py -v
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Iterator, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import link_check  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APPENDIX = REPO_ROOT / "appendix"
SHARED = REPO_ROOT / "shared"
LLM = REPO_ROOT / "llm"
ROOT_ENV = REPO_ROOT / ".env"

#: Top-level import names that live in shared/.
SHARED_MODULES = frozenset(
    {SHARED.name}
    | {p.stem for p in SHARED.glob("*.py")}
    | {p.name for p in SHARED.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}
)

#: Top-level import names outside the Appendix that Appendix code reaches through sys.path,
#: and the folder each must load from: shared/'s modules, and the provider layer.
REPO_MODULES = {**{name: SHARED for name in SHARED_MODULES}, LLM.name: LLM}

#: Code that does path setup. A cell or script carrying one of these that does not
#: parse cannot be checked, so it is reported instead of skipped.
_PATH_SETUP = re.compile(
    r"sys\.path|load_dotenv|^\s*(?:from|import)\s+(?:%s)\b" % "|".join(sorted(REPO_MODULES)),
    re.M,
)

#: The only names and attributes a path expression may use. Evaluating nothing else
#: means reading a file never runs its code.
_NAMES = {"os": os, "Path": Path, "pathlib": pathlib, "str": str}
_ATTRS = frozenset(
    {"path", "join", "dirname", "abspath", "realpath", "normpath", "Path",
     "resolve", "absolute", "parent", "parents", "joinpath"}
)


class Unresolvable(Exception):
    """A path expression needs more than os.path / pathlib to evaluate."""


def _evaluate(node: ast.expr, names: dict) -> object:
    try:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Attribute) and node.attr in _ATTRS:
            return getattr(_evaluate(node.value, names), node.attr)
        if isinstance(node, ast.Subscript):  # Path(__file__).parents[2]
            return _evaluate(node.value, names)[_evaluate(node.slice, names)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):  # root / ".env"
            return _evaluate(node.left, names) / _evaluate(node.right, names)
        if isinstance(node, ast.Call) and not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args):
            return _evaluate(node.func, names)(*(_evaluate(a, names) for a in node.args))
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise Unresolvable(f"{ast.unparse(node)} ({error})") from error
    raise Unresolvable(ast.unparse(node))


def _resolve_path(node: ast.expr, names: dict, base: Path) -> Path:
    value = _evaluate(node, names)
    if not isinstance(value, (str, os.PathLike)):
        raise Unresolvable(f"{ast.unparse(node)} is not a path")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _is_sys_path_call(func: ast.expr) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"insert", "append"}
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def _is_load_dotenv(func: ast.expr) -> bool:
    return (isinstance(func, ast.Name) and func.id == "load_dotenv") or (
        isinstance(func, ast.Attribute) and func.attr == "load_dotenv"
    )


@dataclass
class PathSetup:
    """What one Appendix file does to reach code and config outside its own folder."""

    path: Path
    #: Where the file's relative paths start: its folder.
    base: Path
    sys_path: list = field(default_factory=list)  # (where, entry it adds to sys.path)
    shared_imports: list = field(default_factory=list)  # (where, dotted module from shared/ or llm/)
    dotenv_paths: list = field(default_factory=list)  # (where, .env it loads by explicit path)
    unresolved: list = field(default_factory=list)  # path setup this test could not evaluate

    @property
    def label(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


def _sources(path: Path) -> Iterator[tuple[str, str]]:
    """(where, Python source) for a script, or for each code cell of a notebook."""
    if path.suffix != ".ipynb":
        yield "", path.read_text(encoding="utf-8")
        return
    for index, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        # %magic and !shell lines are IPython, not Python: blank them, keeping line numbers.
        lines = ("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())
        yield f"cell {index} ", "\n".join(lines)


def scan(path: Path) -> PathSetup:
    setup = PathSetup(path=path, base=path.parent)
    names = dict(_NAMES)
    if path.suffix != ".ipynb":  # Jupyter defines no __file__
        names["__file__"] = str(path)

    for cell, source in _sources(path):
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            if _PATH_SETUP.search(source):
                setup.unresolved.append(f"{cell}does not parse: {error.msg} (line {error.lineno})")
            continue

        # Module-level constants built from __file__, e.g. ROOT = Path(__file__).parents[2].
        for node in tree.body:
            if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                try:
                    value = _evaluate(node.value, names)
                except Unresolvable:
                    continue
                names.update({target.id: value for target in node.targets})

        for node in ast.walk(tree):
            where = f"{cell}line {getattr(node, 'lineno', '?')}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                setup.shared_imports += [
                    (where, module) for module in _imported_modules(node) if module.split(".")[0] in REPO_MODULES
                ]
                continue
            if not isinstance(node, ast.Call):
                continue
            if _is_sys_path_call(node.func):
                argument, found = (node.args[-1] if node.args else None), setup.sys_path
            elif _is_load_dotenv(node.func):
                keyword = next((k.value for k in node.keywords if k.arg == "dotenv_path"), None)
                argument, found = (node.args[0] if node.args else keyword), setup.dotenv_paths
            else:
                continue
            if argument is None:
                continue  # load_dotenv() with no path searches upward from the file for .env
            try:
                found.append((where, _resolve_path(argument, names, setup.base)))
            except Unresolvable as error:
                setup.unresolved.append(f"{where}: {error}")
    return setup


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module] if node.level == 0 and node.module else []  # relative imports stay local


def locate(module: str, search_path: list[Path]) -> Optional[Path]:
    """Where `import module` loads from when only `search_path` is on sys.path."""
    locations = [str(p) for p in search_path]
    spec = None
    parts = module.split(".")
    for depth in range(1, len(parts) + 1):
        spec = PathFinder.find_spec(".".join(parts[:depth]), locations)
        if spec is None:
            return None
        locations = list(spec.submodule_search_locations or [])
    if spec.origin and spec.has_location:
        return Path(spec.origin).resolve()
    return Path(list(spec.submodule_search_locations)[0]).resolve()  # namespace package


def _appendix_files(suffix: str) -> list[Path]:
    return sorted(
        path
        for path in APPENDIX.rglob(f"*{suffix}")
        if not set(path.relative_to(REPO_ROOT).parts) & link_check.EXCLUDED_DIR_NAMES
    )


SETUPS = [scan(path) for path in _appendix_files(".py") + _appendix_files(".ipynb")]
WITH_SYS_PATH = [s for s in SETUPS if s.sys_path]
IMPORTING_SHARED = [s for s in SETUPS if s.shared_imports]
LOADING_DOTENV = [s for s in SETUPS if s.dotenv_paths]


def _ids(setups: list[PathSetup]) -> list[str]:
    return [s.label for s in setups]


def _shown(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if REPO_ROOT in (path, *path.parents) else str(path)


class TestAppendixPathSetup:
    def test_the_scan_sees_the_path_setup_it_guards(self):
        # An empty scan would pass every check below on nothing. Each file here stands
        # for one form of path setup the scan has to read.
        importing = set(_ids(IMPORTING_SHARED))
        loading = set(_ids(LOADING_DOTENV))

        assert "appendix/agent-harness/harness_example.py" in importing  # os.path.join(dirname(__file__), ..)
        assert "appendix/agent-harness/loops/outcome_loop.py" in importing  # a topic subfolder, three levels
        assert "appendix/rag/rag_example.py" in importing  # insert and import inside a function
        assert "appendix/multimodal/multimodal_example.py" in loading  # Path(__file__).parent... / ".env"
        assert "appendix/rag/rag_systems.ipynb" in loading  # notebook, relative to its folder

    def test_path_setup_can_be_read_without_running_it(self):
        unresolved = [f"{s.label} {problem}" for s in SETUPS for problem in s.unresolved]

        assert unresolved == [], "build sys.path and .env paths from __file__ with os.path or pathlib"

    @pytest.mark.parametrize("setup", WITH_SYS_PATH, ids=_ids(WITH_SYS_PATH))
    def test_every_sys_path_entry_is_a_folder(self, setup):
        missing = [f"{where}: {_shown(entry)}" for where, entry in setup.sys_path if not entry.is_dir()]

        assert missing == [], f"{setup.label} puts folders that do not exist on sys.path"

    @pytest.mark.parametrize("setup", IMPORTING_SHARED, ids=_ids(IMPORTING_SHARED))
    def test_shared_and_llm_imports_load_from_the_repo(self, setup):
        # A script's own folder is sys.path[0]; a notebook's working directory is too.
        search_path = [entry for _, entry in setup.sys_path] + [setup.base]
        wrong = []
        for where, module in setup.shared_imports:
            found = locate(module, search_path)
            home = REPO_MODULES[module.split(".")[0]].resolve()
            if found is None or home not in (found, *found.parents):
                wrong.append(f"{where}: import {module} -> {_shown(found) if found else 'ModuleNotFoundError'}")

        assert wrong == [], f"{setup.label} cannot import shared/ or llm/ from its folder"

    @pytest.mark.parametrize("setup", LOADING_DOTENV, ids=_ids(LOADING_DOTENV))
    def test_dotenv_path_is_the_root_env(self, setup):
        wrong = [f"{where}: {_shown(path)}" for where, path in setup.dotenv_paths if path != ROOT_ENV]

        assert wrong == [], f"{setup.label} loads a .env other than the root .env, so load_dotenv finds nothing"


class TestScan:
    def test_script_paths_start_at_its___file__(self, tmp_path):
        script = tmp_path / "topic" / "script.py"
        script.parent.mkdir()
        script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "sys.path.insert(0, str(ROOT / 'shared'))\n"
            "from provider import chat\n",
            encoding="utf-8",
        )

        setup = scan(script)

        assert setup.sys_path == [("line 4", (tmp_path / "shared").resolve())]
        assert setup.shared_imports == [("line 5", "provider")]
        assert setup.unresolved == []

    def test_notebook_paths_start_at_its_folder_past_ipython_lines(self, tmp_path):
        notebook = tmp_path / "topic" / "nb.ipynb"
        notebook.parent.mkdir()
        notebook.write_text(
            json.dumps({"cells": [
                {"cell_type": "markdown", "source": ["load_dotenv('nowhere')"]},
                {"cell_type": "code", "source": [
                    "%pip install -q python-dotenv\n",
                    "from dotenv import load_dotenv\n",
                    'load_dotenv(dotenv_path="../.env")\n',
                    "load_dotenv()\n",
                ]},
            ]}),
            encoding="utf-8",
        )

        setup = scan(notebook)

        assert setup.dotenv_paths == [("cell 1 line 3", (tmp_path / ".env").resolve())]
        assert setup.unresolved == []

    def test_path_setup_it_cannot_evaluate_is_reported_not_skipped(self, tmp_path):
        script = tmp_path / "script.py"
        script.write_text("import sys\nsys.path.insert(0, find_root())\n", encoding="utf-8")
        notebook = tmp_path / "nb.ipynb"
        notebook.write_text(
            json.dumps({"cells": [{"cell_type": "code", "source": ["load_dotenv('../.env'"]}]}),
            encoding="utf-8",
        )

        assert [problem.split(":")[0] for problem in scan(script).unresolved] == ["line 2"]
        assert [problem.split(":")[0] for problem in scan(notebook).unresolved] == ["cell 0 does not parse"]
