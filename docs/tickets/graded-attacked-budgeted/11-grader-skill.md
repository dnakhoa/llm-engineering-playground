# 11 — Grader skill

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A Claude Code skill in the open Agent Skills format. It runs the Checks command against the Reader's agent, in Offline mode by default and live only when the Reader asks. It explains each failure with a link to the Spine lesson section that covers it, and ends with the "passed through module N" line.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Invoking the skill runs Offline Checks and never goes live unasked.
- [ ] Each failure maps to a lesson link.
- [ ] The skill follows the Agent Skills format so other tools can load it.

## Comments

**Reach additions, 2026-09-26.** Each is a way for a Reader's progress to bring new Readers:

- [ ] **Install in one command.** The Grader installs into a skill-capable coding agent with one command and no clone. The same command appears in the README above the fold.
- [ ] **Shareable result badge.** Once the Checks pass, the Grader writes a small result file the Reader can commit. It feeds a shields.io endpoint badge ("Graded ✓ · Attacked ✓ · Budgeted ✓ — through module N") that links back to this course. Offline results are labelled as offline, and the badge can never claim a module whose Checks didn't pass.
- [ ] **Share text.** The "passed through module N" summary ends with one line of share text the Reader can paste into a post, including the course link.

**Rebuild note (2026-09-27).** A first build (branch `ticket/11`, a1c01f7) was held because it was cut before 06 and 07. Rebuild on the current base and reuse its design where it still fits.

- [ ] **The badge counts only Checks that ran.** A skipped Check (such as 06's live-only judge in an offline run) never counts toward a badge claim. An offline badge reads "offline" and says which live-only Checks it didn't run.
- [ ] Lesson links are generated from the real Check names in `checks/spine_*`, including 06's and 07's, not from a table written in advance.
