"""The Spine 4 lesson: it states the pinned convention version, and its notebook
runs Offline, top to bottom, and opens in Colab.

The Colab clone-and-restart path is exercised for Spine 1's notebook
(tests/test_spine_1_lesson.py); this notebook's setup cell must be the same
cell, so that test covers it too.
"""
from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from checks.spine_4_observed import every_tool_call_has_a_tool_span  # noqa: E402
from company.tracing import SCHEMA_URL, SEMCONV_VERSION  # noqa: E402

LESSON = REPO / "spine" / "04-observed"
README = LESSON / "README.md"
NOTEBOOK = LESSON / "observed.ipynb"
SPINE_1_NOTEBOOK = REPO / "spine" / "01-loop" / "loop.ipynb"

COLAB_LINK = re.compile(r"https://colab\.research\.google\.com/github/[^\s)\"'\]]+")


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code_cells(path):
    return ["".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code"]


def _colab_url():
    first_cell = "".join(_cells(SPINE_1_NOTEBOOK)[0]["source"])
    spine_1 = COLAB_LINK.findall(first_cell)[0]
    return spine_1.replace("spine/01-loop/loop.ipynb", "spine/04-observed/observed.ipynb")


def test_the_lesson_states_the_version_the_code_pins():
    readme = README.read_text(encoding="utf-8")

    assert "OpenTelemetry semantic conventions {}**".format(SEMCONV_VERSION) in readme
    assert "`SEMCONV_VERSION`" in readme
    assert SCHEMA_URL in readme


def test_the_notebook_is_a_colab_ready_v4_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["language"] == "python"
    assert "colab" in nb["metadata"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_the_lesson_and_notebook_badges_open_this_notebook():
    readme = README.read_text(encoding="utf-8")
    first_cell = "".join(_cells(NOTEBOOK)[0]["source"])

    assert COLAB_LINK.findall(readme) == [_colab_url()]
    assert COLAB_LINK.findall(first_cell) == [_colab_url()]


def test_the_setup_cell_is_spine_1s_setup_cell():
    assert _code_cells(NOTEBOOK)[0] == _code_cells(SPINE_1_NOTEBOOK)[0]


def test_the_notebook_runs_offline_and_its_checks_pass(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the notebook tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.chdir(LESSON)
    monkeypatch.setattr(sys, "path", list(sys.path))

    namespace: dict = {"__name__": "__main__"}
    for index, source in enumerate(_code_cells(NOTEBOOK)):
        try:
            exec(compile(source, "observed.ipynb code cell {}".format(index), "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure message
            pytest.fail("code cell {} raised {!r}:\n{}".format(index, error, source))

    assert len(namespace["outcome"].trace.chat_spans) == 3
    assert namespace["untraced"].resolved is True
    assert every_tool_call_has_a_tool_span(namespace["untraced"]).passed is False
    assert namespace["refused_span"].status == "ERROR"
    assert len(namespace["exported"]) > 0
    assert namespace["report"].passed is True
