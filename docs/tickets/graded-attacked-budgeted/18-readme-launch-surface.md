# 18 — README, CHANGELOG, demo, Drop template

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **README.** Rewritten as the landing page. Above the fold it has the thesis, a one-line audience, a mermaid Spine diagram, the demo link, the Offline quick start and the Scoreboard excerpt with its date and label. Badges stay only if they link somewhere real. The superlatives, star comparison and hour totals are removed. A "What's New" section sits at the top.
- **demo.py.** Runs the finished agent on a sample Case, offline by default.
- **CHANGELOG.** A new major entry that describes only what's true.
- **Drop template.** Covers the model and API refresh, one new Case or attack, the Scoreboard update and the "What's New" entry.
- **Launch checklist.** Lists the human steps: repo rename, Space publish, Scoreboard #1 live run, and the social posts.

**Blocked by:** 10, 11, 12, 17

**Status:** ready-for-agent

- [ ] Cloning and running the demo works offline with no key.
- [ ] Every README link and badge resolves to a real target.
- [ ] The Drop template exists, and the first "What's New" entry is written.
- [ ] The launch checklist lists each outward-facing step as needing the maintainer's go-ahead.

## Comments

**Reach additions, 2026-09-26** (from studying why ai-engineering-from-scratch has 58k stars and 10k forks):

- [ ] **Goal-based entry table** above the fold: "I want my agent to… → start at Spine N". For example: survive prompt injection → 5; stop refunding the wrong amount → 3; cost less per Case → 6; answer from our docs → 2; go live → 7.
- [ ] **"Add the Grader in 30 seconds"**: the one-command Grader install from ticket 11, placed next to the Offline quick start.
- [ ] **Point of difference, said once, near the top**: the Checks grade *your own* agent against Backend state, not self-reported output. State it plainly, without superlatives.
- [ ] **Readers' badges**: a short section showing the Reader result badge from ticket 11 and how to add it to your own repo.
- [ ] **Invite contributions through the Drop**: the Drop template invites the community to submit a new attack Case or an incident Case by PR, with the best one credited in the next "What's New".
