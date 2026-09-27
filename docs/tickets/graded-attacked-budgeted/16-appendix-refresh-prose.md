# 16 — Appendix refresh C: prose, dates, intros

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md

**What to build:**

- **Model tables.** Appendix prose (foundations model tables, fine-tuning, multimodal, evaluation, optimization, context engineering, harness) reflects the current landscape, verified against vendor docs.
- **Service changes.** It covers the model tables, effort replacing sampling parameters, OpenAI self-serve fine-tuning being closed to new users (with open-weight fine-tuning as the primary path), and current multimodal models.
- **Benchmarks.** Saturated benchmarks are noted as saturated.
- **Page hygiene.** Every Appendix page gets a "last verified" date. Empty "Why this matters" intros are filled.
- **Lint baseline.** The remaining prose files are removed from the lint baseline.

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] Every Appendix page shows a "last verified" date.
- [ ] No page has an empty intro.
- [ ] Every model, price or deprecation fact has a vendor source or is removed.
- [ ] The files in this batch are gone from the lint baseline.

## Comments

- [ ] Fix the pre-existing `ImportError` in `appendix/agent-harness/harness_example.py`: running it imports `loops/research_loop.py`, whose `from ..harness.journal import Journal` is a relative import beyond the top-level package.
- [ ] Move `appendix/fine-tuning/finetune_example.py` off the old trl `SFTTrainer` signature (`tokenizer=`, `dataset_text_field=`, `max_seq_length=`, `packing=` passed to `SFTTrainer` itself) onto the current trl API, verified against trl's own docs.
