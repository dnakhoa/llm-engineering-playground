# Offline, each module's Cases are graded against that module's own reference agent; verdict Checks run on every later Case

Replay matches every outgoing request exactly, including the system prompt and the retrieved Knowledge Base text as well as the tool list (ADR 0005). Later modules have to change those. Spine 5 hardens the prompt, and Spine 6 adds caching and routing. If every earlier Case were replayed against the latest reference agent, each module would force re-recording every earlier Case with a live key. So we decided:

- **Offline:** each module's Cases replay against the reference agent of the module that recorded them. Old recordings stay fixed evidence for the agent that made them.
- **Live:** a Reader's own agent is graded on the Cases of every module up to the chosen one. That is the run that proves hardening and earns the Grader badge.
- **Verdict everywhere:** the verdict Checks (`@on_every_case`) run on every later module's Cases too, as do the trace Checks once a module has traces. A Case can then never go ungraded on "did the agent finish and reach the expected state" just because a later suite owns it.

## Considered Options

- **Re-record every earlier Case whenever a module changes the request:** rejected. It needs a live key at every Drop, and cumulative Offline Checks would go red between Drops.
- **Ignore the system prompt when matching replays:** rejected, for the same reason as in ADR 0005. A replay could then pass for a request the model never saw.
- **Copy earlier Cases into later suites instead of `@on_every_case`:** rejected. The copies drift, as Spine 4's hand-copied list of four Case IDs already showed.
