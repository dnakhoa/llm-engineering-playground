# 13 — Move modules 00–16 into the Appendix

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md · ADR 0001

**What to build:** This is a wide refactor. The numbered module folders move into Appendix topic folders:

- prompt engineering
- RAG
- fine-tuning
- evaluation
- deployment
- optimization
- agent frameworks
- observability
- EvalOps
- guardrails
- memory
- context engineering
- agent harness
- MCP
- multimodal
- graph engineering
- foundations

TypeScript moves into the Appendix too. The capstone and Kaggle folders are removed. Every internal link, the tests, the CI notebook discovery and the lint baseline paths are updated.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] No numbered module folders remain, and the Appendix index lists every topic.
- [ ] A link check finds no dangling internal links.
- [ ] The existing tests and the CI notebook smoke test pass at the new paths.
- [ ] The lint baseline is updated to the new paths, and CI is green.
