# 11 — Grader skill

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A Claude Code skill in the open Agent Skills format. It runs the Checks command against the Reader's agent, in Offline mode by default and live only when the Reader asks. It explains each failure with a link to the Spine lesson section that covers it, and ends with the "passed through module N" line.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Invoking the skill runs Offline Checks and never goes live unasked.
- [ ] Each failure maps to a lesson link.
- [ ] The skill follows the Agent Skills format so other tools can load it.
