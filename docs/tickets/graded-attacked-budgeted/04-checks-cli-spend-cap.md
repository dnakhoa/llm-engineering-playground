# 04 — Checks CLI: Spend Cap, cumulative suites, the Reader's own agent

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A Checks command that takes a module range, an agent reference, a mode (offline or live) and a Spend Cap. It runs each module's Check suite together with the suites of every earlier module, prints a result per Check, and ends with a "passed through module N" line. The Spend Cap uses registry prices: it's printed before the run starts, the run stops when it's reached, and partial results are reported. Checks that only work live are reported as skipped in offline mode.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] A Spend Cap below a run's cost stops the run and reports what finished.
- [ ] An agent reference outside the reference implementation is loaded and graded.
- [ ] Running module N includes the Check suites of modules 1 to N-1.
- [ ] Offline mode never touches the network. Live mode prints the cap and asks for nothing interactively.
