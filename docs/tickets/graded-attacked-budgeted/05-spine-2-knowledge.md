# 05 — Spine 2 Knowledge

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **Knowledge Base.** Acme Notes help-centre and policy articles covering the refund window, proration and plan limits.
- **Retrieval.** A default lexical retriever (pure Python, no embeddings), plus an embeddings retriever taught as an upgrade behind the same interface.
- **New Action.** "Issue refund", which enforces the refund window, proration and the maximum refund.
- **Memory.** Memory across the turns of a multi-turn Case.
- **Cases.** Cases where the correct Action depends on the retrieved policy.
- **Lesson and extension.** A Spine 2 lesson, a notebook, and the reference Flagship Agent extended.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] A prorated-refund Case passes only when the refunded amount matches the policy article.
- [ ] A refund outside the window is refused by the Action.
- [ ] A multi-turn Case that needs an earlier turn's detail passes.
- [ ] All Knowledge Checks pass offline, and the Spine 1 Checks still pass.
