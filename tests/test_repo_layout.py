"""
Repository layout — the numbered modules live in the Appendix, and no internal link dangles.

Run: pytest tests/test_repo_layout.py -v
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import link_check  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APPENDIX = REPO_ROOT / "appendix"

#: Every Appendix topic folder. Each has a README; the Python topics also pin
#: their own requirements.txt.
APPENDIX_TOPICS = (
    "foundations",
    "prompt-engineering",
    "rag",
    "fine-tuning",
    "evaluation",
    "deployment",
    "optimization",
    "agent-frameworks",
    "observability",
    "evalops",
    "guardrails",
    "memory",
    "context-engineering",
    "agent-harness",
    "mcp",
    "multimodal",
    "graph-engineering",
    "typescript",
)
NON_PYTHON_TOPICS = frozenset({"typescript"})


class TestAppendixLayout:
    def test_no_numbered_module_folders_remain_at_the_root(self):
        numbered = sorted(p.name for p in REPO_ROOT.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))

        assert numbered == []

    @pytest.mark.parametrize("folder", ["capstone", "kaggle", "typescript"])
    def test_retired_or_moved_root_folder_is_gone(self, folder):
        assert not (REPO_ROOT / folder).exists()

    def test_appendix_holds_exactly_the_topic_folders(self):
        folders = sorted(p.name for p in APPENDIX.iterdir() if p.is_dir())

        assert folders == sorted(APPENDIX_TOPICS)

    @pytest.mark.parametrize("topic", APPENDIX_TOPICS)
    def test_topic_has_a_readme_and_its_requirements(self, topic):
        assert (APPENDIX / topic / "README.md").is_file()
        if topic not in NON_PYTHON_TOPICS:
            assert (APPENDIX / topic / "requirements.txt").is_file()

    @pytest.mark.parametrize("topic", APPENDIX_TOPICS)
    def test_appendix_index_links_every_topic(self, topic):
        index = APPENDIX / "README.md"
        targets = {
            link_check.resolve(index, target, REPO_ROOT).resolve()
            for _, target in link_check.iter_links(index.read_text(encoding="utf-8"))
        }

        assert (APPENDIX / topic).resolve() in targets or (APPENDIX / topic / "README.md").resolve() in targets

    def test_appendix_notebooks_are_discoverable(self):
        assert list(APPENDIX.rglob("*.ipynb")), "the CI notebook smoke test would find no Appendix notebooks"


class TestLinkCheck:
    def test_dangling_link_is_reported_with_its_line(self, tmp_path):
        page = tmp_path / "page.md"
        (tmp_path / "exists.md").write_text("x", encoding="utf-8")
        page.write_text(
            "[ok](exists.md) [web](https://example.com) [anchor](#top)\n"
            "[gone](missing/README.md#setup)\n"
            "```\n[in code](also-missing.md)\n```\n"
            "`[inline](inline-missing.md)`\n",
            encoding="utf-8",
        )

        dangling = link_check.check_file(page, root=tmp_path)

        assert [(d.line, d.target) for d in dangling] == [(2, "missing/README.md#setup")]

    def test_notebook_markdown_cells_are_checked(self, tmp_path):
        notebook = tmp_path / "nb.ipynb"
        notebook.write_text(
            '{"cells": [{"cell_type": "code", "source": ["x = \\"[a](gone.md)\\""]},'
            ' {"cell_type": "markdown", "source": ["**Next:** [b](gone.md)"]}]}',
            encoding="utf-8",
        )

        dangling = link_check.check_file(notebook, root=tmp_path)

        assert [(d.cell, d.target) for d in dangling] == [(1, "gone.md")]

    def test_repo_has_no_dangling_internal_links(self):
        dangling = link_check.check_repo(REPO_ROOT)

        assert dangling == [], "\n" + "\n".join(d.format() for d in dangling)
