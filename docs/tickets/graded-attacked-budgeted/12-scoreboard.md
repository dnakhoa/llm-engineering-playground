# 12 — Scoreboard command and format

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** A maintainer command that runs the Scoreboard Case set across the registry's model list in live mode under a Spend Cap. For each model it records resolution rate, Attacked pass rate and cost per resolved Case, and writes a dated markdown table and a JSON file. Every output carries the label "our Cases, our Checks — not a general benchmark". Running the command offline against recordings is how it's tested. Producing Scoreboard #1 needs the maintainer's live keys and is a human step.

**Blocked by:** 08, 09

**Status:** ready-for-agent

- [ ] An offline run over recordings produces the dated table and JSON.
- [ ] The Spend Cap stops the run and marks which models are incomplete.
- [ ] The label appears on every output.
- [ ] The Scoreboard #1 run is documented as a one-command maintainer step (ready-for-human).

## Comments

**Decisions of 2026-09-27:**

- [ ] **G6.** The Scoreboard's resolution rate uses the Backend-state verdict. LLM-judged reply quality is its own column, labelled "LLM-judged", with the judge model named.
- [ ] **G5.** Each Scoreboard row is judged by a model from a different provider than the one being scored, and the judge model is named in the output.
