# 15 — Appendix refresh B: MCP 2026-07-28 and agent frameworks

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **MCP.** The Appendix MCP material is rewritten to the 2026-07-28 spec: stateless requests, server/discover, InputRequiredResult instead of server-initiated elicitation and sampling, Tasks as an extension, and cacheable lists. Every spec fact is verified against the spec's own changelog.
- **Agent frameworks page.** Covers LangGraph 1.x, the OpenAI Agents SDK and Google ADK 2.0, and maps each back to what the Reader built by hand.
- **Lint baseline.** The files in this batch are removed from the lint baseline.

**Blocked by:** 13

**Status:** done

- [ ] The MCP material cites the 2026-07-28 spec, and no 2025-06-18 references remain outside the historical-exception marker.
- [ ] The MCP example server runs against the current MCP SDK, or the page states that the SDK isn't current yet (verified).
- [ ] The frameworks page links each concept to a Spine lesson.
- [ ] The files in this batch are gone from the lint baseline.

## Comments

- [ ] These three files are removed from `tests/stale_lint_baseline.txt` AND from the `LATER_BATCH` exclusion tuple in `tests/test_appendix_code.py`, so the Appendix code test (registry model IDs, no direct vendor chat calls, no temperature sent straight to a vendor API, temperature support asked via `accepts_sampling_at`) holds them too: `appendix/mcp/mcp_example.py`, `appendix/mcp/servers/example_server.py`, `appendix/agent-frameworks/agentic_workflows.ipynb` (which still has `ChatOpenAI(model="gpt-4o-mini", temperature=0)`). <!-- historical-exception -->
