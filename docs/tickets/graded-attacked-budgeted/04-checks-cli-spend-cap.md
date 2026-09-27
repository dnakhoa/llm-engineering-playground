# 04 — Checks CLI: Spend Cap, cumulative suites, the Reader's own agent

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A Checks command that takes a module range, an agent reference, a mode (offline or live) and a Spend Cap. It runs each module's Check suite together with the suites of every earlier module, prints a result per Check, and ends with a "passed through module N" line. The Spend Cap uses registry prices: it's printed before the run starts, the run stops when it's reached, and partial results are reported. Checks that only work live are reported as skipped in offline mode.

**Blocked by:** 03

**Status:** done

- [ ] A Spend Cap below a run's cost stops the run and reports what finished.
- [ ] An agent reference outside the reference implementation is loaded and graded.
- [ ] Running module N includes the Check suites of modules 1 to N-1.
- [ ] Offline mode never touches the network. Live mode prints the cap and asks for nothing interactively.

## Comments

**Carried from the stage A review (2026-09-26).** These belong here because the Spend Cap prices every call and live mode is where a Reader first chooses a model:

- [ ] **Cache pricing (S4).** The registry gains cache-read and cache-write prices, verified against vendor docs like every other field. Cost on an Outcome bills `cached_input_tokens` at the cache price. Otherwise the Spend Cap and the Budgeted module overstate the cost of cached runs.
- [ ] **Local servers (S1).** A Reader can use a local OpenAI-compatible server (for example Ollama) by giving a base URL and a model name, with no registry entry needed. It is priced at zero, and the Checks output says the Spend Cap doesn't apply. Qwen stays out of the registry until Alibaba's docs give prices on a vendor page.
- [ ] **Sampling depends on effort (S3).** Sampling support depends on the effort level, not just the model. For GPT-6, temperature is accepted when effort is `none` and dropped otherwise, following the registry's `sampling_note`. An outgoing-request test covers both cases.

**Carried from the stage B review (2026-09-26).** Stage B was merged without these test-only fixes. The code behaves correctly; these tests make sure it keeps doing so:

- [ ] **Claim without acting (G1).** A fixture agent replies "Done! You're on Pro now." and never calls `change_plan`. Its Outcome is unresolved and the Spine 1 Check fails. Today every never-acting fixture replies with text that claims nothing, so a Check that read the reply instead of Backend state would still pass.
- [ ] **Exactly once, and only for the Case's account.** An agent that upgrades, downgrades and upgrades again fails the Check. So does one that changes a different account's plan.
- [ ] **Full transcript.** The transcript test asserts every turn, including tool results and refusals, not just the first and last.
