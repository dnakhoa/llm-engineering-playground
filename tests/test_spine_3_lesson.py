"""The Spine 3 lesson: it documents the Case format, and its notebook runs Offline.

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
LESSON = REPO / "spine" / "03-graded"
README = LESSON / "README.md"
NOTEBOOK = LESSON / "graded.ipynb"
SPINE_1_NOTEBOOK = REPO / "spine" / "01-loop" / "loop.ipynb"
CASES_DIR = REPO / "company" / "cases"

COLAB_LINK = re.compile(r"https://colab\.research\.google\.com/github/[^\s)\"'\]]+")


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code_cells(path):
    return ["".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code"]


def _colab_url():
    first_cell = "".join(_cells(SPINE_1_NOTEBOOK)[0]["source"])
    spine_1 = COLAB_LINK.findall(first_cell)[0]
    return spine_1.replace("spine/01-loop/loop.ipynb", "spine/03-graded/graded.ipynb")


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
    first_cell = "".join(_cells(NOTEBOOK)[0]["source"])

    assert COLAB_LINK.findall(README.read_text(encoding="utf-8")) == [_colab_url()]
    assert COLAB_LINK.findall(first_cell) == [_colab_url()]


def test_the_setup_cell_is_spine_1s_setup_cell():
    assert _code_cells(NOTEBOOK)[0] == _code_cells(SPINE_1_NOTEBOOK)[0]


def _case_format_section():
    readme = README.read_text(encoding="utf-8")
    match = re.search(r"^## The Case format\n(.*?)(?=^## )", readme, re.MULTILINE | re.DOTALL)
    assert match, "the lesson has no '## The Case format' section"
    return match.group(1)


def _keys(value, prefix=""):
    """Every key in a Case file, nested ones as ``customer.account_id``."""
    if not isinstance(value, dict):
        return set()
    found = set()
    for key, inner in value.items():
        path = prefix + key
        found.add(path)
        if key not in ("expected_state_change", "tags"):  # their keys are data, not format
            found |= _keys(inner, path + ".")
    return found


def test_the_lesson_documents_every_field_a_case_file_uses():
    section = _case_format_section()
    used = set().union(*(
        _keys(json.loads(path.read_text(encoding="utf-8"))) for path in CASES_DIR.glob("*.json")
    ))

    undocumented = sorted(key for key in used if "`{}`".format(key) not in section)
    assert undocumented == []


def test_the_lesson_documents_every_field_the_case_loader_reads():
    # A field only a future Case uses is still part of the format.
    from company.runner import Case

    section = _case_format_section()
    loader_fields = {"id", "opening_message", "follow_ups", "actions", "knowledge_base",
                     "expected_state_change", "forbidden_actions", "recording", "tags",
                     "judge_rubric", "customer.account_id"}
    assert {f for f in Case.__dataclass_fields__} - {"customer_account_id"} <= loader_fields
    assert sorted(f for f in loader_fields if "`{}`".format(f) not in section) == []


def test_the_notebook_has_a_write_your_own_case_exercise():
    markdown = [
        "".join(c["source"]) for c in _cells(NOTEBOOK) if c["cell_type"] == "markdown"
    ]
    assert any("Your turn" in text and "Case" in text for text in markdown)
    assert any("load_case(" in source for source in _code_cells(NOTEBOOK))


def test_the_notebook_runs_offline_and_writes_nothing_into_the_repo(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the notebook tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.chdir(LESSON)
    monkeypatch.setattr(sys, "path", list(sys.path))
    cases_before = sorted(CASES_DIR.iterdir())

    namespace: dict = {"__name__": "__main__"}
    for index, source in enumerate(_code_cells(NOTEBOOK)):
        try:
            exec(compile(source, "graded.ipynb code cell {}".format(index), "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure message
            pytest.fail("code cell {} raised {!r}:\n{}".format(index, error, source))

    assert namespace["report"].passed is True
    assert namespace["regression"].passed is False
    assert namespace["regression"].passed_through == 2
    assert namespace["my_case_reference"].resolved is True
    assert namespace["my_case_incident"].resolved is False
    assert sorted(CASES_DIR.iterdir()) == cases_before
