# Contributing

Contributions are welcome — bug fixes, better examples, new exercises, improved explanations.

## What's most useful

- **Fix broken code** — API changes (LangChain, OpenAI) break examples fast
- **Add exercises** — practical challenges at the end of any module
- **Improve notebooks** — richer explanations, better visualisations
- **New modules** — suggest topics via an issue first

## How to contribute

1. **Fork** the repo and create a branch: `git checkout -b fix/module-02-imports`
2. **Make your change** — keep PRs focused on one thing
3. **Test your code** — run the example or notebook and confirm it produces reasonable output
4. **Open a PR** with a short description of what changed and why

## Standards

- Code must run without errors when `OPENAI_API_KEY` is set
- New notebooks follow the existing structure: Setup → Concepts → Code → Exercises
- Keep `requirements.txt` files per-module in sync with imports
- Do not commit `.env` files or API keys

## Reporting issues

Open a GitHub issue with:
- Module number and file name
- Error message or unexpected behaviour
- Python version and OS
