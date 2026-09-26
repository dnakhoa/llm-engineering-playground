# 19 — Launch kit: reach beyond the README

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** Everything the maintainer needs to launch in one sitting. All of it is drafted and committed, and nothing is posted.

- **Translated landing pages.** Translated READMEs for the landing page only, in Chinese (Simplified), Japanese, Korean, Hindi, Spanish, Portuguese and Vietnamese. English stays canonical and the lessons are not translated. Each page says it is machine-assisted and links to the English lessons. This reverses the English-only launch choice (Q16) on the maintainer's instruction of 2026-09-26: "much more reach".
- **Terminal recording.** A short, reproducible recording of the Grader failing a naive agent on an Attacked Case, then passing the hardened agent. It is built offline from the replay recordings, so anyone can regenerate it, and it is embedded in the README.
- **Post drafts** for LinkedIn, an X thread, Show HN, and r/LocalLLaMA and r/MachineLearning. Each leads with the thesis, the terminal recording and one Scoreboard finding, and names the point of difference once: the Checks grade your own agent against Backend state.
- **Launch timing note.** Launch with a Scoreboard update that includes the newest frontier model. A model release is a news hook for the post, and it proves the monthly Drop cadence works.
- **Human checklist.** Repo rename, GitHub description, topics and homepage, publishing the Space, the Scoreboard #1 live run, and each post. Every item is marked as needing the maintainer's explicit go-ahead.

**Blocked by:** 18

**Status:** ready-for-agent

- [ ] Every translated landing page links to a real English target (the link check covers `i18n/`).
- [ ] The terminal recording regenerates offline with one command.
- [ ] Each post draft fits its platform's length limit and contains no unverifiable claims (no star comparisons, no "most comprehensive").
- [ ] The checklist marks every outward-facing step as needing the maintainer's go-ahead.
