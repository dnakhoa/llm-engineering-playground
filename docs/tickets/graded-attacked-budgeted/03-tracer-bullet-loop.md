# 03 — Tracer bullet: Acme Notes Backend, one Case, Case runner, Spine 1 Loop

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md · ADR 0001, ADR 0003

**What to build:** This ticket builds the thinnest end-to-end path through the course:

- **Backend.** A deterministic in-memory Acme Notes Backend with accounts and plans (Free, Pro, Team). It is reset for every Case and can export its state.
- **Actions.** Two Actions exposed as tools: look up account, and change plan. Each enforces "only the Case's own customer".
- **Case.** One Case in which a customer asks to upgrade to Pro.
- **Case runner.** It runs a Case against an agent loaded from an importable reference and returns an Outcome: the final Backend state and diff, Actions attempted and executed, transcript, usage and cost, and resolved or unresolved.
- **Reference agent.** The Spine 1 reference Flagship Agent: a small from-scratch loop with tool calls, stop conditions and a step limit.
- **Check.** One Offline Check (Backend plan changed to Pro, exactly once), passing through a replay recording.
- **Lesson.** The Spine 1 lesson and a Colab-ready notebook.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Running the Case in Offline mode against the reference agent passes its Check with no network.
- [ ] An agent that never calls the Action produces an unresolved Outcome, and the Check fails.
- [ ] The step limit stops a looping agent and is reported in the Outcome.
- [ ] A cross-account plan change is refused by the Action itself, not the prompt.
- [ ] The Spine 1 lesson walks through the loop, and the notebook opens in Colab.
