# Issue tracker: Local Markdown

Specs and tickets for this repo live as markdown files under `docs/`, reviewed in PRs like any other change.

## Conventions

- One feature per slug.
- The spec is `docs/specs/<feature-slug>.md`.
- Implementation tickets go one file per ticket at `docs/tickets/<feature-slug>/<NN>-<slug>.md`, numbered from `01`. Never put all tickets in a single combined file.
- Triage state is a `Status:` line near the top of each file. See `triage-labels.md` for the role strings.
- Blocking is a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists has `Status: done`.
- Comments and conversation history are appended at the bottom of the file under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create the spec or ticket file at the paths above, creating directories if needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the ticket number directly.

## Wayfinding operations

Used by `/wayfinder`.

- **Map**: `docs/tickets/<effort>/map.md` holds the Notes, Decisions-so-far and Fog sections.
- **Child ticket**: `docs/tickets/<effort>/NN-<slug>.md`, with the question in the body.
  - A `Type:` line records the ticket type: `research`, `prototype`, `grilling` or `task`.
  - A `Status:` line records `claimed` or `resolved`.
- **Frontier**: the open, unblocked, unclaimed files in `docs/tickets/<effort>/`. The lowest number wins.
- **Claim**: set `Status: claimed` and save before doing any work.
- **Resolve**:
  1. Append the answer under an `## Answer` heading.
  2. Set `Status: resolved`.
  3. Append a context pointer (a one-line gist plus a link) to the Decisions-so-far section of `map.md`.
