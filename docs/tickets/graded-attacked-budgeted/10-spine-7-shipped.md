# 10 — Spine 7 Shipped

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **Service.** The final Flagship Agent served behind an HTTP endpoint with streaming.
- **Container.** A container build for the service.
- **Hugging Face Space recipe.** Defaults to replaying recorded Cases. An optional live mode uses the operator's key under a Spend Cap, and visitors are never asked for keys.
- **Check.** A served-endpoint Check that sends a Case through the endpoint.
- **Lesson and notebook.** A Spine 7 lesson and notebook.

**Blocked by:** 08, 09

**Status:** ready-for-agent

- [ ] The served-endpoint Check passes offline against a locally started service.
- [ ] The container builds, and the service in it passes the same Check.
- [ ] The Space recipe runs in replay mode with no secrets configured.
- [ ] The Check suites of all Spine modules pass against the final agent.
