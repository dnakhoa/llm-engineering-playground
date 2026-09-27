# 07 — Spine 4 Observed

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** Every Case run emits an OpenTelemetry trace following the GenAI semantic conventions (an agent span, chat spans, tool spans), with tokens and cost on the spans and the convention version pinned. The Outcome carries the trace. Trace-structure Checks verify the instrumentation is complete. Includes a Spine 4 lesson (how to export to any OTel backend), a notebook, and the reference agent extended.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] A trace-structure Check fails when a tool call has no span.
- [ ] The spans carry token usage and cost that match the Outcome.
- [ ] The pinned convention version is stated in the lesson and in the code.
- [ ] Earlier Checks still pass.

## Comments

**Rebuild note (2026-09-27).** A first build (branch `ticket/07`, 9343bdc) was held because it was cut before ticket 06 changed `Outcome` (the five-part `verdict` behind `resolved`). Rebuild on the current base and reuse that branch's tracing design where it still fits.

- [ ] Spans are emitted in addition to 06's `verdict`, not in place of it. `flagship/observed.py` builds on `flagship/knowledge.py` as it stands after 06, including `the_agent_finishes_the_case`.
- [ ] **Each suite lists only its own Checks.** The cumulative runner from ticket 04 already runs earlier suites, so the Spine 4 suite doesn't re-import earlier modules' Checks. Also remove Spine 3's re-import of Spine 2's Checks, which prints nine duplicate PASS lines.
- [ ] Tests that assert the "Passed through module N" line don't hard-code the highest module number. Today 06's CI test asserts "module 3 of" and breaks when Spine 4 lands.
- [ ] `opentelemetry-sdk` goes in `requirements.txt` and the CI unit-tests install line, next to 15's `mcp>=2.2`.
