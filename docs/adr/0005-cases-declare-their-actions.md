# Each Case declares the Actions the agent may use

A replay recording captures the whole outgoing request, including the tool list. With one global tool list, every new Spine module's Action (such as `issue_refund` in Spine 2 or `escalate_case` in Spine 5) would change every existing recording. Every earlier module's Offline Checks would then fail with a replay mismatch, and fixing them would mean re-recording with a live key. So each Case lists the Actions it allows, and the runner offers the agent only those. This keeps recordings stable as the course grows. It also shows least-privilege tool exposure, which the Attacked module teaches as a defence.

## Considered Options

- **Re-record every Drop**: rejected. It needs a live key for every module change and breaks cumulative Offline Checks between Drops.
- **Ignore the tool list when matching replays**: rejected. A replay could then pass for a request the model never saw, which is exactly the silent staleness replay exists to catch.
