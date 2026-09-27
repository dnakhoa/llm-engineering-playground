# 17 — Contract: delete the lint baseline and the old provider API

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:** This is the contract step of the expand–contract sequence. The lint baseline is deleted, so the lint now applies to the whole repo. The old provider API is deleted once nothing calls it.

**Blocked by:** 14, 15, 16

**Status:** ready-for-agent

- [ ] The baseline no longer exists, and CI is green.
- [ ] Nothing imports the old provider API.
