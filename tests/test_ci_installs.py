"""
CI installs what the tests it runs import.

Each CI job that runs pytest installs its own packages, so a test importing a package
the job never installs passes on any machine that happens to have it and fails in CI
with ModuleNotFoundError. This reads every job in .github/workflows/ci.yml, takes its
`pip install` lines and the files its `pytest` runs collect, follows those files'
imports through the repo's own modules (function-level imports included), and asks
that every package from outside the standard library is one the job installs.

Run: pytest tests/test_ci_installs.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import sysconfig
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Import name -> the name pip installs it under, where the two differ.
DISTRIBUTIONS = {"dotenv": "python-dotenv", "yaml": "pyyaml"}

_PIP = re.compile(r"^(?:python3?\s+-m\s+)?pip3?\s+install\s+(?P<args>.+)$")
_PYTEST = re.compile(r"^(?:python3?\s+-m\s+)?pytest(?:\s+(?P<args>.*))?$")
#: pytest's default norecursedirs, plus bytecode caches.
_SKIP_DIRS = {"build", "dist", "node_modules", "venv", "CVS", "_darcs", "__pycache__"}


def _normalize(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


@dataclass
class Job:
    name: str
    installs: Set[str] = field(default_factory=set)  # normalized distribution names
    pytest_runs: List[List[str]] = field(default_factory=list)  # each run's arguments


def _requirements(path: Path) -> Set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(_normalize(re.split(r"[\s\[<>=!~;@]", line, maxsplit=1)[0]))
    return names


def _installs(args: str, root: Path) -> Set[str]:
    tokens, names = args.split(), set()
    for i, token in enumerate(tokens):
        if i and tokens[i - 1] in ("-r", "--requirement"):
            names |= _requirements(root / token)
        elif not token.startswith("-"):
            names.add(_normalize(re.split(r"[\[<>=!~;@]", token.strip("'\""), maxsplit=1)[0]))
    return names


def parse_jobs(workflow: str, root: Path) -> List[Job]:
    """The workflow's jobs, each with what it pip-installs and how it runs pytest.

    Commands are read from every line of a job, so a `run: |` block counts as well as
    a one-line `run:`.
    """
    jobs: List[Job] = []
    in_jobs = False
    for line in workflow.splitlines():
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if re.match(r"^\S", line):
            in_jobs = False
        if not in_jobs:
            continue
        header = re.match(r"^  ([\w-]+):\s*$", line)
        if header:
            jobs.append(Job(header.group(1)))
            continue
        if not jobs:
            continue
        command = re.sub(r"^\s*(?:-\s+)?(?:run:\s*)?", "", line)
        command = re.split(r"\s+#", command, maxsplit=1)[0].strip()
        pip, run = _PIP.match(command), _PYTEST.match(command)
        if pip:
            jobs[-1].installs |= _installs(pip.group("args"), root)
        elif run:
            jobs[-1].pytest_runs.append((run.group("args") or "").split())
    return jobs


# ── What a pytest run imports ────────────────────────────────────────────────


def _collected(args: Sequence[str], root: Path) -> Iterator[Path]:
    """The files pytest collects for one run: named files, and test_*.py, *_test.py
    and conftest.py under named folders (the repo root when none is named)."""
    targets = [root / arg.split("::", 1)[0] for arg in args if not arg.startswith("-")]
    for target in [t for t in targets if t.exists()] if targets else [root]:
        if target.is_file():
            yield target.resolve()
            continue
        for path in sorted(target.rglob("*.py")):
            relative = path.relative_to(target).parts[:-1]
            if any(part.startswith(".") or part in _SKIP_DIRS for part in relative):
                continue
            if path.name == "conftest.py" or path.name.startswith("test_") or path.stem.endswith("_test"):
                yield path.resolve()


def _module_files(base: Path, parts: Sequence[str]) -> List[Path]:
    """The files `import a.b.c` runs from `base`: each package's __init__.py, then the module."""
    files, path = [], base
    for part in parts:
        path = path / part
        if (path / "__init__.py").is_file():
            files.append(path / "__init__.py")
        elif path.with_suffix(".py").is_file():
            files.append(path.with_suffix(".py"))
            break
        elif not path.is_dir():
            break
    return [f.resolve() for f in files]


def _is_stdlib(name: str) -> bool:
    names = getattr(sys, "stdlib_module_names", None)  # Python 3.10+
    if names is not None:
        return name in names
    if name in sys.builtin_module_names:
        return True
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return spec is not None  # not installed: the standard library always is
    origin = Path(spec.origin).resolve()
    stdlib = [Path(sysconfig.get_paths()[key]).resolve() for key in ("stdlib", "platstdlib")]
    in_stdlib = any(base == origin or base in origin.parents for base in stdlib)
    return in_stdlib and not {"site-packages", "dist-packages"} & set(origin.parts)


def third_party_imports(files: Sequence[Path], root: Path) -> Dict[str, Set[Path]]:
    """Top-level name of every non-stdlib, non-repo module the files import, directly or
    through the repo's own modules, mapped to the files that import it."""
    found: Dict[str, Set[Path]] = {}
    queue, seen = list(files), set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        in_package = (path.parent / "__init__.py").is_file()
        # A script-style module's own folder is on sys.path (pytest puts a test file's
        # folder there); a module inside a package imports absolutely from the root.
        bases = [root] if in_package else [path.parent, root]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.level:
                package = path.parent
                for _ in range(node.level - 1):
                    package = package.parent
                parts = node.module.split(".") if node.module else []
                queue += _module_files(package, parts)
                for alias in node.names:
                    queue += _module_files(package, parts + [alias.name])
                continue
            if isinstance(node, ast.Import):
                imports = [(alias.name.split("."), []) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [(node.module.split("."), [alias.name for alias in node.names])]
            else:
                continue
            for parts, names in imports:
                top = parts[0]
                base = next(
                    (b for b in bases if (b / top).is_dir() or (b / top).with_suffix(".py").is_file()),
                    None,
                )
                if base is not None:
                    queue += _module_files(base, parts)
                    for name in names:
                        queue += _module_files(base, parts + [name])
                elif top != "__future__" and not _is_stdlib(top):
                    found.setdefault(top, set()).add(path)
    return found


def missing_installs(job: Job, root: Path) -> Dict[str, List[str]]:
    """pip name -> repo files a job's pytest runs import it from, for each package the
    job does not install."""
    files = [f for run in job.pytest_runs for f in _collected(run, root)]
    missing = {}
    for name, importers in third_party_imports(files, root).items():
        distribution = DISTRIBUTIONS.get(name, name)
        if _normalize(distribution) not in job.installs:
            missing[distribution] = sorted(str(p.relative_to(root)) for p in importers)
    return missing


# ── The real workflow ────────────────────────────────────────────────────────

JOBS = parse_jobs(CI_WORKFLOW.read_text(encoding="utf-8"), REPO_ROOT)
TESTING_JOBS = [job for job in JOBS if job.pytest_runs]


def test_the_workflow_has_jobs_that_run_pytest():
    # Finding none means the workflow changed shape and every check below passes on nothing.
    assert {"unit-tests", "notebook-smoke"} <= {job.name for job in TESTING_JOBS}


@pytest.mark.parametrize("job", TESTING_JOBS, ids=[job.name for job in TESTING_JOBS])
def test_the_job_installs_every_package_its_tests_import(job):
    missing = missing_installs(job, REPO_ROOT)

    assert missing == {}, (
        "CI job {!r} runs pytest on files importing packages its `pip install` lines never "
        "install, so they fail there with ModuleNotFoundError: {}. Add them to that job's "
        "pip install line (or map an import name to its pip name in DISTRIBUTIONS).".format(
            job.name, missing
        )
    )


# ── The check itself ─────────────────────────────────────────────────────────


def test_a_package_imported_inside_a_test_function_is_reported_until_ci_installs_it(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "helper.py").write_text("import json\nimport tiktoken\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("from .core import run\n")
    (tmp_path / "pkg" / "core.py").write_text("import types\n\ndef run():\n    import requests\n")
    (tmp_path / "tests" / "test_it.py").write_text(
        textwrap.dedent(
            """
            import sys
            import helper
            from pkg import run

            def _env():
                from dotenv import dotenv_values
                return dotenv_values(".env.example")
            """
        )
    )
    workflow = textwrap.dedent(
        """
        on: [push]
        jobs:
          unit-tests:
            steps:
              - name: Install
                run: pip install pytest tiktoken
              - run: pytest tests/ -v
          lint:
            steps:
              - run: |
                  pip install ruff
        """
    )

    job = next(j for j in parse_jobs(workflow, tmp_path) if j.name == "unit-tests")
    assert missing_installs(job, tmp_path) == {
        "python-dotenv": ["tests/test_it.py"],
        "requests": ["pkg/core.py"],
    }

    fixed = workflow.replace("pip install pytest tiktoken", "pip install pytest tiktoken python-dotenv requests")
    job = next(j for j in parse_jobs(fixed, tmp_path) if j.name == "unit-tests")
    assert missing_installs(job, tmp_path) == {}


def test_a_requirements_file_counts_as_installed(tmp_path):
    (tmp_path / "requirements-ci.txt").write_text("# CI\npython-dotenv>=1.0.0\npytest\n")

    workflow = "jobs:\n  unit-tests:\n    steps:\n      - run: python -m pip install -r requirements-ci.txt\n"
    (job,) = parse_jobs(workflow, tmp_path)

    assert job.installs == {"python-dotenv", "pytest"}
