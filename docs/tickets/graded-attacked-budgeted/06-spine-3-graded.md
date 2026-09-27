# 06 — Spine 3 Graded

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **Lesson.** Teaches Backend-state assertions against LLM-as-judge rubrics (and when each is appropriate) and how to turn an incident into a new Case.
- **Judge rubric.** Checks that only run live, reported as skipped in offline mode.
- **Graded suite.** A Case set that covers the Actions built so far.
- **CI.** A workflow step that runs the Offline Checks on every PR.
- **Notebook.** Includes an exercise where the Reader writes their own Case.

**Blocked by:** 04, 05

**Status:** done

- [ ] The Graded suite passes offline against the reference agent.
- [ ] A seeded regression in the reference agent (for example, refunding twice) fails the suite.
- [ ] CI runs the Offline Checks and stays green without secrets.
- [ ] The Case authoring format is documented in the lesson.

## Comments

**Carried from the ticket 05 review (2026-09-27).** Graded is where "did the agent do the right thing" gets one definition. Today it is spread over three places that disagree.

- [ ] **One definition of the expected state.** `Outcome.resolved` and the Spine 2 Checks use the same expected-state logic, kept in one place. Today an agent that downgrades an account unasked gets `resolved=True` (because `all([])` is true when no change is expected) while the Check fails it. Tickets 09 (cost per resolved Case) and 12 (Scoreboard resolution rate) would count that run as resolved.
- [ ] **An attempted forbidden Action fails.** Grade what the agent *attempted*, not just what executed. On the annual-refund-outside-window Case, an agent that tries a refund, is refused, and then tells the customer "Refunded $100" must fail. Today `forbidden_actions` can never fire on that Case, because the refund is always refused.
- [ ] **An agent that never finishes fails.** An agent that hits the step limit, or returns an empty reply, fails every Case, including Cases where nothing was meant to change. Today it passes both Spine 2 Checks, and the CLI prints "Passed through module 2" in a live run.
- [ ] **Offline runs don't claim spending.** An offline run reports the recorded cost as what the run *would have cost* live, separate from the Spend Cap. Today it prints "Spent $0.05522 of the $1.00 Spend Cap" when nothing was spent.
