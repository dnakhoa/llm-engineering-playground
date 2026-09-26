# 01 — Model registry and capability-aware provider layer

Spec: docs/specs/graded-attacked-budgeted.md · Glossary: CONTEXT.md · ADR 0002

**What to build:** A single provider call that takes normalized messages, normalized tools and intent-level options (effort, max output), and works across Anthropic (Messages), OpenAI (Responses API), Google Gemini, and OpenAI-compatible servers (DeepSeek, Qwen, xAI, local). It returns normalized text, tool calls, usage and stop reason. The call consults a model registry (one data file of current models: ID, provider, prices, context window, accepts sampling parameters, effort levels, API surface) and drops or translates options each model doesn't support. It also offers a replay transport that records normalized request/response pairs and replays them, failing loudly on a mismatch.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Registry entries are verified against vendor docs on the day they're written, and each entry notes its source and verification date. Unverified models are left out.
- [ ] For every registry model, a stub-transport test shows that temperature is never sent where it's rejected, and that effort is mapped to that provider's control.
- [ ] Tool calls for current OpenAI models go through the Responses API, verified by an outgoing-request test.
- [ ] Normalized tools round-trip into each provider's tool shape, and tool calls come back in normalized form.
- [ ] Record mode writes a recording, replay mode returns it, and a changed request raises a clear mismatch error.
- [ ] The whole test suite runs with no network or keys.
