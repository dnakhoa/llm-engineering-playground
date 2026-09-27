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

## Comments

**Decision G2(a), 2026-09-26 (ADR 0005).** Each Case now declares its Actions:

- [ ] A Case file lists the Actions it allows. The runner offers the agent only those, and a call to an undeclared Action is refused and recorded as attempted but not executed.
- [ ] The upgrade-to-Pro Case declares `look_up_account` and `change_plan`. Adding `issue_refund` leaves its recording and its Offline Check unchanged. Test this explicitly: add the new Action, then replay the old Case.
- [ ] The system prompt stays in the recording. Changing it *should* require re-recording, and the lesson says so.
