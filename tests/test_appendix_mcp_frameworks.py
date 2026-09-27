"""
The MCP pages teach the 2026-07-28 spec, and the frameworks page maps onto the Spine
(ticket 15).

Two Appendix pages go stale faster than the rest: MCP changed shape in 2026-07-28
(no initialize handshake, `server/discover`, `InputRequiredResult` instead of
server-initiated requests, Tasks moved to an extension, cacheable list results), and
the agent frameworks ship new majors every few months. This checks what a Reader
would copy from them:

1. The MCP pages cite the 2026-07-28 spec and name its changes, and name no older
   spec version except on a line marked `historical-exception`.
2. The example server runs on the current MCP SDK (`mcp` 2.x): it answers
   `server/discover`, speaks 2026-07-28 with no handshake, puts cache hints on
   `tools/list`, and asks for a confirmation through an `InputRequiredResult`. These
   tests need the `mcp` package (appendix/mcp/requirements.txt) and are skipped
   without it.
3. The frameworks page covers LangGraph 1.x, the OpenAI Agents SDK and Google ADK
   2.0, and every concept it maps links to the Spine lesson where the Reader built
   that concept by hand.

Run: pytest tests/test_appendix_mcp_frameworks.py -v
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = REPO_ROOT / "appendix" / "mcp"
MCP_README = MCP_DIR / "README.md"
SERVER = MCP_DIR / "servers" / "example_server.py"
FRAMEWORKS_README = REPO_ROOT / "appendix" / "agent-frameworks" / "README.md"

SPEC = "2026-07-28"
SPEC_URL = f"https://modelcontextprotocol.io/specification/{SPEC}"
#: MCP spec revisions before 2026-07-28 (modelcontextprotocol.io/specification/versioning).
OLDER_SPECS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
MARKER = "historical-exception"


def _lines(path: Path) -> list[str]:
    """Every line of a page, script, or notebook cell."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return text.splitlines()
    lines = []
    for cell in json.loads(text)["cells"]:
        source = cell["source"]
        lines += ("".join(source) if isinstance(source, list) else source).splitlines()
    return lines


MCP_FILES = sorted(
    p for p in MCP_DIR.rglob("*") if p.suffix in (".md", ".py", ".ipynb") and "__pycache__" not in p.parts
)


# ── 1. The MCP pages ──────────────────────────────────────────────────────────


def test_the_mcp_page_cites_the_2026_07_28_spec():
    assert SPEC_URL in MCP_README.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", MCP_FILES, ids=[p.relative_to(REPO_ROOT).as_posix() for p in MCP_FILES])
def test_no_older_spec_version_is_named_outside_the_marker(path):
    found = [
        line.strip()[:100]
        for line in _lines(path)
        if any(version in line for version in OLDER_SPECS) and MARKER not in line
    ]

    assert found == [], f"{path.name} names an MCP spec older than {SPEC}"


@pytest.mark.parametrize(
    "change",
    [
        "initialize",  # the handshake that is gone
        "server/discover",
        "InputRequiredResult",
        "inputResponses",
        "io.modelcontextprotocol/tasks",
        "ttlMs",
        "cacheScope",
        "Mcp-Session-Id",
    ],
)
def test_the_mcp_page_teaches_the_2026_07_28_changes(change):
    assert change in MCP_README.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", MCP_FILES, ids=[p.relative_to(REPO_ROOT).as_posix() for p in MCP_FILES])
def test_no_mcp_page_uses_the_removed_fastmcp_module(path):
    """mcp 2.x renamed FastMCP to MCPServer; importing mcp.server.fastmcp raises."""
    found = [line.strip() for line in _lines(path) if "mcp.server.fastmcp" in line and MARKER not in line]

    assert found == []


# ── 2. The example server, on the current SDK ────────────────────────────────


def _server_module():
    pytest.importorskip("mcp")
    if int(importlib.metadata.version("mcp").split(".")[0]) < 2:
        pytest.skip("the example server needs mcp 2.x")
    spec = importlib.util.spec_from_file_location("appendix_mcp_example_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(coro):
    return asyncio.run(coro)


def test_the_server_speaks_2026_07_28_without_a_handshake():
    from mcp import Client

    server = _server_module().build_server()

    async def main():
        async with Client(server, mode=SPEC) as client:
            discovered = await client.session.discover()
            return client.protocol_version, discovered.supported_versions

    version, supported = _run(main())

    assert version == SPEC
    assert SPEC in supported


def test_tools_list_carries_a_cache_hint():
    from mcp import Client

    server = _server_module().build_server()

    async def main():
        async with Client(server, mode=SPEC, cache=None) as client:
            return await client.list_tools()

    listing = _run(main())

    assert listing.ttl_ms > 0
    assert listing.cache_scope in ("public", "private")
    # Deterministic order, so a client's cache and the model's prompt cache both hit.
    assert [tool.name for tool in listing.tools] == sorted(tool.name for tool in listing.tools)


def _delete(answer: str):
    from mcp import Client, types

    server = _server_module().build_server()

    async def on_elicit(_context, _params):
        if answer == "accept":
            return types.ElicitResult(action="accept", content={"confirm": True})
        return types.ElicitResult(action="decline")

    async def main():
        async with Client(server, mode=SPEC, elicitation_callback=on_elicit) as client:
            first = await client.session.call_tool("notes_delete", {"note_id": 1}, allow_input_required=True)
            final = await client.call_tool("notes_delete", {"note_id": 1})
            listing = await client.call_tool("notes_list", {})
            return first, final.content[0].text, listing.content[0].text

    return _run(main())


def test_a_destructive_tool_asks_for_confirmation_with_an_input_required_result():
    first, _, _ = _delete("accept")

    assert first.result_type == "input_required"
    [request] = first.input_requests.values()
    assert request.method == "elicitation/create"


def test_the_retry_with_the_answer_completes_the_call():
    _, final, listing = _delete("accept")

    assert "deleted" in final.lower()
    assert "Welcome" not in listing


def test_a_declined_confirmation_keeps_the_note():
    _, final, listing = _delete("decline")

    assert "not deleted" in final.lower()
    assert "Welcome" in listing


def test_the_server_runs_over_stdio():
    _server_module()
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async def main():
        async with Client(params) as client:
            return client.protocol_version, [t.name for t in (await client.list_tools()).tools]

    version, tools = _run(asyncio.wait_for(main(), timeout=60))

    assert version == SPEC
    assert "notes_search" in tools


# ── 3. The agent frameworks page ─────────────────────────────────────────────


def _frameworks_page() -> str:
    return FRAMEWORKS_README.read_text(encoding="utf-8")


@pytest.mark.parametrize("framework", [r"LangGraph 1\.", r"OpenAI Agents SDK", r"(?:Google )?ADK 2\."])
def test_the_frameworks_page_covers_each_current_framework(framework):
    headings = [line for line in _frameworks_page().splitlines() if line.startswith("#")]

    assert any(re.search(framework, heading) for heading in headings)


def _concept_rows() -> list[str]:
    """Data rows of the table that maps each hand-built concept onto the frameworks."""
    lines = _frameworks_page().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## ") and "by hand" in line)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows.append(line)
    return rows[1:]  # drop the header row


def test_the_frameworks_page_maps_concepts_onto_all_three_frameworks():
    rows = _concept_rows()

    assert len(rows) >= 5
    for row in rows:
        assert row.count("|") >= 6, f"a concept row needs a cell per framework: {row[:80]}"


def test_every_mapped_concept_links_to_a_spine_lesson():
    unlinked = [row[:80] for row in _concept_rows() if "](../../spine/" not in row]

    assert unlinked == []
