"""The Spine 2 lesson: its notebook runs Offline, top to bottom, and opens in Colab.

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
LESSON = REPO / "spine" / "02-knowledge"
NOTEBOOK = LESSON / "knowledge.ipynb"
SPINE_1_NOTEBOOK = REPO / "spine" / "01-loop" / "loop.ipynb"

COLAB_LINK = re.compile(r"https://colab\.research\.google\.com/github/[^\s)\"'\]]+")


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code_cells(path):
    return ["".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code"]


def _colab_url():
    first_cell = "".join(_cells(SPINE_1_NOTEBOOK)[0]["source"])
    spine_1 = COLAB_LINK.findall(first_cell)[0]
    return spine_1.replace("spine/01-loop/loop.ipynb", "spine/02-knowledge/knowledge.ipynb")


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
    readme = (LESSON / "README.md").read_text(encoding="utf-8")
    first_cell = "".join(_cells(NOTEBOOK)[0]["source"])

    assert COLAB_LINK.findall(readme) == [_colab_url()]
    assert COLAB_LINK.findall(first_cell) == [_colab_url()]


def test_the_setup_cell_is_spine_1s_setup_cell():
    assert _code_cells(NOTEBOOK)[0] == _code_cells(SPINE_1_NOTEBOOK)[0]


SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
NEGATION = re.compile(r"\b(?:no|not|never|without|nothing|needn)\b|n['’]t\b", re.IGNORECASE)


def _sentences(text):
    return [" ".join(s.split()) for s in SENTENCE_BREAK.split(text) if s.strip()]


def test_the_lesson_says_a_new_system_prompt_needs_new_recordings():
    # The two phrases must meet in one sentence: "re-record" also appears where
    # the lesson talks about adding Actions, and "system prompt" in "Try it".
    # tests/test_case_runner.py proves the claim; this proves the lesson makes it.
    readme = (LESSON / "README.md").read_text(encoding="utf-8")

    said = [
        s for s in _sentences(readme)
        if "system prompt" in s.lower() and "re-record" in s.lower()
    ]

    assert said, "the lesson never says a changed system prompt needs re-recording"
    assert [s for s in said if NEGATION.search(s)] == [], (
        "the lesson says a changed system prompt does not need re-recording"
    )


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
            exec(compile(source, "knowledge.ipynb code cell {}".format(index), "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure message
            pytest.fail("code cell {} raised {!r}:\n{}".format(index, error, source))

    assert namespace["refund"].resolved is True
    assert namespace["window"].resolved is True
    assert namespace["remembers"].resolved is True
    assert namespace["forgot"] is not None  # the Spine 1 loop's replay mismatch
    assert namespace["report"].passed is True
