# 02 — Stale-reference lint with a legacy baseline

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A CI-enforced test that fails when Markdown, Python, TypeScript or notebooks mention a retired model ID or a dead API. Examples: gpt-4o-mini, gpt-3.5-turbo, gpt-4 used as a default, Llama-2, legacy LangChain chains and imports, OpenAI Assistants API, OpenAI tool calls through Chat Completions, and MCP 2025-06-18. Each denylist entry carries a replacement hint, and a historical-exception marker allows prose about the past. This is the expand step of an expand–contract sequence: the lint starts with a baseline of files that already fail, so CI stays green. Later tickets shrink the baseline.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The denylist is a data file with a replacement hint for each entry.
- [ ] Adding a new file that contains a denylisted reference fails the test with the file, line and hint.
- [ ] A fixture proves that the historical-exception marker suppresses a match.
- [ ] The baseline lists every currently failing file, and CI is green.
- [ ] The test runs in the existing CI workflow.
