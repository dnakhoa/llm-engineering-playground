"""The Spine 1 lesson: its notebook runs Offline, top to bottom, and opens in Colab.

The notebook is executed cell by cell from its own folder, the way Jupyter runs
it locally, with the network blocked. Colab runs the same cells from /content,
outside any checkout, so there the setup cell clones the course first. That
path is run too: from a folder outside the checkout, with `git` allowed only
local repositories and the course's GitHub URL mapped to a snapshot of this
working tree, and run again the way a Reader re-runs the setup cell or
restarts the runtime (/content keeps the clone).

Where the course lives is what the root README tells every Reader to
`git clone`. The Colab badges and the setup cell must name that repository.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LESSON = REPO / "spine" / "01-loop"
NOTEBOOK = LESSON / "loop.ipynb"
CLONE_NAME = "llm-engineering-playground"

COLAB_LINK = re.compile(r"https://colab\.research\.google\.com/github/[^\s)\"'\]]+")
GITHUB_CLONE_URL = re.compile(r"https://github\.com/[^\s\"']+?\.git")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells():
    return ["".join(cell["source"]) for cell in _notebook()["cells"]
            if cell["cell_type"] == "code"]


def _course_url():
    """The one URL the root README tells a Reader to `git clone`."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    urls = set(re.findall(r"git clone (" + GITHUB_CLONE_URL.pattern + ")", readme))
    assert len(urls) == 1, "the root README should name one clone URL, found {}".format(urls)
    return urls.pop()


def _colab_url_of_this_notebook():
    owner_repo = _course_url()[len("https://github.com/"):-len(".git")]
    return "https://colab.research.google.com/github/{}/blob/main/{}".format(
        owner_repo, NOTEBOOK.relative_to(REPO).as_posix())


def _is_course(folder):
    return (folder / "company").is_dir() and (folder / "llm").is_dir()


def test_the_notebook_is_a_colab_ready_v4_notebook():
    nb = _notebook()

    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["language"] == "python"
    assert "colab" in nb["metadata"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_the_lesson_badge_opens_this_notebook_from_the_course_repo():
    readme = (LESSON / "README.md").read_text(encoding="utf-8")

    assert COLAB_LINK.findall(readme) == [_colab_url_of_this_notebook()]


def test_the_notebook_badge_opens_this_notebook_from_the_course_repo():
    first_cell = "".join(_notebook()["cells"][0]["source"])

    assert COLAB_LINK.findall(first_cell) == [_colab_url_of_this_notebook()]


def test_the_setup_cell_clones_the_course_repo():
    setup = _code_cells()[0]

    assert GITHUB_CLONE_URL.findall(setup) == [_course_url()]


def test_the_notebook_runs_offline_and_its_check_passes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the notebook tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.chdir(LESSON)
    monkeypatch.setattr(sys, "path", list(sys.path))

    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            exec(compile(source, "loop.ipynb cell {}".format(index), "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure message
            pytest.fail("cell {} raised {!r}:\n{}".format(index, error, source))

    assert namespace["outcome"].resolved is True
    assert namespace["check"].passed is True
    assert namespace["looping"].step_limit_reached is True


# Runs in a fresh interpreter, as a Colab runtime does: every code cell in
# order, then the setup cell once more. Prints what the test asserts on.
COLAB_SESSION = textwrap.dedent("""
    import json
    import socket
    import sys

    def refuse(*args, **kwargs):
        raise AssertionError("the notebook tried to open a network connection")

    socket.socket.connect = refuse
    socket.create_connection = refuse

    notebook = json.load(open(sys.argv[1], encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    namespace = {"__name__": "__main__"}
    courses = []
    runs = list(enumerate(cells)) + [(0, cells[0])]
    for index, source in runs:
        exec(compile(source, "loop.ipynb code cell {}".format(index), "exec"), namespace)
        if index == 0:
            courses.append(str(namespace["COURSE"]))

    import company
    print(json.dumps({
        "courses": courses,
        "company": company.__file__,
        "resolved": namespace["outcome"].resolved,
        "passed": namespace["check"].passed,
    }))
""")


def _isolated_git_env(**extra):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_") and key != "PYTHONPATH"}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_TERMINAL_PROMPT="0", **extra)
    return env


def _snapshot_of_this_working_tree(origin):
    """A local repository holding the files a clone of this course would get."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True, text=True,
    ).stdout.split("\0")
    for name in filter(None, listed):
        source = REPO / name
        if source.is_file():
            (origin / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, origin / name)
    env = _isolated_git_env()
    for command in (["init", "-q"], ["add", "-A"],
                    ["-c", "user.name=test", "-c", "user.email=test@example.invalid",
                     "commit", "-q", "-m", "snapshot"]):
        subprocess.run(["git", "-C", str(origin), *command], check=True, env=env)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is needed to clone the course")
def test_the_notebook_runs_on_colab_and_again_after_a_restart(tmp_path):
    origin = tmp_path / "origin"
    content = tmp_path.resolve() / "content"  # a notebook's cwd is a resolved path
    origin.mkdir()
    content.mkdir()
    assert not any(_is_course(folder) for folder in [content, *content.parents]), \
        "the Colab run must start outside any checkout of the course"
    _snapshot_of_this_working_tree(origin)

    # Colab's `git clone` of the course URL gets this snapshot. Any other URL
    # fails: git may use local repositories only, so nothing reaches a network.
    env = _isolated_git_env(
        GIT_ALLOW_PROTOCOL="file",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="url.{}.insteadOf".format(origin.as_uri()),
        GIT_CONFIG_VALUE_0=_course_url(),
    )
    clone = content / CLONE_NAME

    def run_session(label):
        session = subprocess.run(
            [sys.executable, "-c", COLAB_SESSION, str(NOTEBOOK)],
            cwd=str(content), env=env, capture_output=True, text=True, timeout=120,
        )
        assert session.returncode == 0, "{} failed:\n{}".format(label, session.stderr[-3000:])
        return json.loads(session.stdout.strip().splitlines()[-1])

    first = run_session("the first Colab run")
    assert first["courses"] == [str(clone), str(clone)]
    assert Path(first["company"]).resolve().is_relative_to(clone.resolve())
    assert first["resolved"] is True
    assert first["passed"] is True

    # A restarted runtime keeps /content, so the clone is still there and the
    # setup cell must reuse it; nothing is left to clone from.
    shutil.rmtree(origin)
    again = run_session("the run after a runtime restart")
    assert again["courses"] == [str(clone), str(clone)]
    assert again["resolved"] is True
    assert again["passed"] is True
