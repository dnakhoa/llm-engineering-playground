# 14 — Appendix refresh A: code and notebooks

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md · ADR 0002

**What to build:** Appendix Python code, notebooks and TypeScript examples use current model IDs from the registry. Python calls go through the new provider layer. OpenAI tool calls use the Responses API. Legacy LangChain chains and imports are removed or rewritten. This is a migrate batch: every file fixed is removed from the lint baseline.

**Blocked by:** 01, 13

**Status:** done

- [ ] No Appendix code or notebook file remains in the lint baseline.
- [ ] Existing tests and the notebook smoke test pass.
- [ ] No Appendix code sends temperature to a model that rejects it.
