"""The Spine 1 lesson: its notebook runs Offline, top to bottom, and opens in Colab.

The notebook is executed cell by cell from its own folder, the way Jupyter runs
it locally, with the network blocked. Colab runs the same cells; the only
difference is that the setup cell clones the course first.
"""
from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LESSON = REPO / "spine" / "01-loop"
NOTEBOOK = LESSON / "loop.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


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


def test_the_lesson_badge_opens_this_notebook_in_colab():
    readme = (LESSON / "README.md").read_text(encoding="utf-8")
    badge = re.search(
        r"https://colab\.research\.google\.com/github/[^/]+/[^/]+/blob/main/(\S+?\.ipynb)",
        readme,
    )

    assert badge is not None
    assert (REPO / badge.group(1)) == NOTEBOOK


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
