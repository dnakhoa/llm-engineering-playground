"""
Stale-reference lint — fails when the course mentions a retired model ID or a dead API.

Scans Markdown, Python, TypeScript and notebooks against the denylist in
`tests/stale_lint_denylist.json`. Each denylist entry carries a replacement hint.
A match is allowed when its line carries the historical-exception marker
(`historical-exception`), so prose about the past does not have to be deleted.

Files listed in `tests/stale_lint_baseline.txt` are known-stale legacy files: their
matches are reported as baselined, not as failures, so CI stays green while the
Appendix refresh works through them. Ticket 17 deletes the baseline.

Run: pytest tests/test_stale_references.py -v
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import stale_lint  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "stale_lint"


@pytest.fixture(scope="module")
def entries():
    return stale_lint.load_denylist()


class TestDenylistDataFile:
    def test_denylist_is_a_data_file(self):
        assert stale_lint.DENYLIST_PATH.exists()
        assert stale_lint.DENYLIST_PATH.suffix == ".json"

    def test_every_entry_has_an_id_a_pattern_and_a_replacement_hint(self, entries):
        assert entries
        for entry in entries:
            assert entry.id
            assert entry.pattern.pattern
            assert entry.hint.strip(), f"{entry.id} has no replacement hint"

    def test_entry_ids_are_unique(self, entries):
        ids = [entry.id for entry in entries]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "sample, expected_id",
        [
            ('model="gpt-4o-mini"', "gpt-4o-mini"),
            ('model="gpt-3.5-turbo"', "gpt-3.5-turbo"),
            ('model = "gpt-4"', "gpt-4-as-default"),
            ("meta-llama/Llama-2-7b-chat-hf", "llama-2"),
            ("from langchain.chains import RetrievalQA", "langchain-legacy-imports"),
            ("chain = LLMChain(llm=llm, prompt=prompt)", "langchain-legacy-chains"),
            ("client.beta.assistants.create(name='helper')", "openai-assistants-api"),
            (
                'client.chat.completions.create(\n    model=model,\n    tools=tools,\n)',
                "openai-chat-completions-tool-calls",
            ),
            ("MCP protocol version 2025-06-18", "mcp-2025-06-18"),
        ],
    )
    def test_denylisted_reference_is_detected(self, entries, sample, expected_id):
        found = stale_lint.scan_text(sample, entries, path="sample.py")
        assert expected_id in {v.entry_id for v in found}, f"{sample!r} was not flagged"

    @pytest.mark.parametrize(
        "sample",
        [
            'model="gpt-5.1"',
            'model="claude-sonnet-5"',
            'model="llama3.2"',
            "from langchain_openai import ChatOpenAI",
            "client.responses.create(model=model, tools=tools)",
            "MCP protocol version 2026-07-28",
        ],
    )
    def test_current_reference_is_not_flagged(self, entries, sample):
        assert stale_lint.scan_text(sample, entries, path="sample.py") == []


class TestFailureReporting:
    def test_new_file_with_a_retired_model_fails_with_file_line_and_hint(self, entries, tmp_path):
        new_file = tmp_path / "17-new-module" / "README.md"
        new_file.parent.mkdir()
        new_file.write_text(
            "# New module\n"
            "\n"
            "Pick a cheap model:\n"
            '\n```python\nmodel = "gpt-4o-mini"\n```\n',
            encoding="utf-8",
        )

        violations = stale_lint.scan_file(new_file, entries, root=tmp_path)

        assert len(violations) == 1
        violation = violations[0]
        assert violation.path == "17-new-module/README.md"
        assert violation.line == 6
        assert violation.entry_id == "gpt-4o-mini"
        assert violation.hint
        report = violation.format()
        assert "17-new-module/README.md:6" in report
        assert violation.hint in report
        assert "gpt-4o-mini" in report

    def test_notebook_match_reports_its_cell_and_line(self, entries):
        violations = stale_lint.scan_file(
            FIXTURES / "retired_notebook.ipynb", entries, root=FIXTURES
        )

        assert [(v.entry_id, v.cell, v.line) for v in violations] == [("gpt-4o-mini", 1, 2)]
        assert "cell 1" in violations[0].format()

    def test_typescript_file_is_scanned(self, entries):
        violations = stale_lint.scan_file(FIXTURES / "retired.ts", entries, root=FIXTURES)

        assert [v.entry_id for v in violations] == ["gpt-4o-mini"]


class TestHistoricalException:
    def test_marker_suppresses_a_match(self, entries):
        violations = stale_lint.scan_file(
            FIXTURES / "historical_prose.md", entries, root=FIXTURES
        )

        assert violations == []

    def test_same_fixture_without_its_markers_would_fail(self, entries):
        text = (FIXTURES / "historical_prose.md").read_text(encoding="utf-8")
        unmarked = re.sub(r"\s*<!--\s*historical-exception\s*-->", "", text)

        violations = stale_lint.scan_text(unmarked, entries, path="historical_prose.md")

        assert len(violations) >= 2, "fixture must contain real matches for the marker to suppress"

    def test_marker_only_suppresses_its_own_line(self, entries):
        text = (
            'model = "gpt-4o-mini"  # historical-exception: what the 2024 course used\n'
            'model = "gpt-3.5-turbo"\n'
        )

        violations = stale_lint.scan_text(text, entries, path="sample.py")

        assert [(v.entry_id, v.line) for v in violations] == [("gpt-3.5-turbo", 2)]


class TestBaseline:
    def test_baseline_file_exists_and_lists_the_legacy_provider(self):
        baseline = stale_lint.load_baseline()

        assert "shared/provider.py" in baseline

    def test_repo_has_no_stale_references_outside_the_baseline(self, entries):
        violations = stale_lint.scan_repo(entries=entries, baseline=stale_lint.load_baseline())

        assert violations == [], "\n" + "\n".join(v.format() for v in violations)

    def test_every_baseline_entry_still_exists_and_still_fails(self, entries):
        baseline = stale_lint.load_baseline()
        failing = {v.path for v in stale_lint.scan_repo(entries=entries, baseline=frozenset())}

        clean = sorted(path for path in baseline if path not in failing)

        assert clean == [], (
            "these files no longer contain stale references — "
            "remove them from tests/stale_lint_baseline.txt:\n" + "\n".join(clean)
        )
