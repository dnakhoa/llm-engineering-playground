# 06 — Spine 3 Graded

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **Lesson.** Teaches Backend-state assertions against LLM-as-judge rubrics (and when each is appropriate) and how to turn an incident into a new Case.
- **Judge rubric.** Checks that only run live, reported as skipped in offline mode.
- **Graded suite.** A Case set that covers the Actions built so far.
- **CI.** A workflow step that runs the Offline Checks on every PR.
- **Notebook.** Includes an exercise where the Reader writes their own Case.

**Blocked by:** 04, 05

**Status:** ready-for-agent

- [ ] The Graded suite passes offline against the reference agent.
- [ ] A seeded regression in the reference agent (for example, refunding twice) fails the suite.
- [ ] CI runs the Offline Checks and stays green without secrets.
- [ ] The Case authoring format is documented in the lesson.
