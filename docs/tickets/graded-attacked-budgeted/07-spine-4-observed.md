# 07 — Spine 4 Observed

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** Every Case run emits an OpenTelemetry trace following the GenAI semantic conventions (an agent span, chat spans, tool spans), with tokens and cost on the spans and the convention version pinned. The Outcome carries the trace. Trace-structure Checks verify the instrumentation is complete. Includes a Spine 4 lesson (how to export to any OTel backend), a notebook, and the reference agent extended.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] A trace-structure Check fails when a tool call has no span.
- [ ] The spans carry token usage and cost that match the Outcome.
- [ ] The pinned convention version is stated in the lesson and in the code.
- [ ] Earlier Checks still pass.
