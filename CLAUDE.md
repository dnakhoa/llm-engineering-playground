# LLM Engineering Playground

Read `CONTEXT.md` (glossary) and `docs/adr/` before changing course structure or the Flagship Agent.

## Agent skills

### Issue tracker

Local markdown: specs in `docs/specs/`, one file per ticket in `docs/tickets/<slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
