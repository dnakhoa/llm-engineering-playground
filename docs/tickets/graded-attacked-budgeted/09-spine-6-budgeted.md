# 09 — Spine 6 Budgeted

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A cost-per-resolved-Case Check. It passes when the cost is at or below a stated limit and the Graded pass rate is no lower than the baseline. A Spine 6 lesson walks through prompt caching, model routing and effort levels, and the Check report shows each lever's effect. The reference agent is extended with these levers, and there's a notebook.

**Blocked by:** 06, 07

**Status:** ready-for-agent

- [ ] The report shows cost per resolved Case before and after each lever.
- [ ] A change that lowers cost but drops the Graded pass rate fails the Check.
- [ ] Cost is computed from registry prices and matches the trace.

## Comments

**Decisions of 2026-09-27:**

- [ ] **G6 — the judge is not part of "resolved".** Cost per resolved Case uses the Backend-state verdict only. Judge scores are reported separately and never change "resolved".
- [ ] **Cost comes from the spans.** Every Case, including Spine 5's and Spine 6's own, has a trace (ADR 0006: the trace Checks run on every Case). Cost per resolved Case is summed from the spans and must equal the Outcome's cost.
- [ ] **ADR 0006.** Caching, routing and effort change the request, so Spine 6's reference agent gets its own recordings for Spine 6's Cases. Earlier recordings stay as they are.
