# Spec: Graded, Attacked, Budgeted — reposition the course around one Flagship Agent

Status: ready-for-agent

Glossary: `CONTEXT.md`. Decisions: ADR 0001 (Flagship Agent as spine), ADR 0002 (provider-agnostic, capability-aware), ADR 0003 (no agent framework).

## Problem Statement

The course has 3 stars and 0 forks. Newer repos cover the same ground and are growing fast: ai-engineering-from-scratch (57.9k), learn-harness-engineering (16.1k, already teaches loop and graph engineering), and learn-claude-code (77.6k).

The course also teaches things that are no longer true:

- Most runnable code defaults to GPT-4o-era models. There's `gpt-3.5-turbo`, `gpt-4` and Llama-2 in places.
- It recommends the Responses API but never uses it.
- It teaches MCP `2025-06-18`, which has been superseded by `2026-07-28`.
- It uses LangChain APIs from before 1.0.
- Its shared provider layer sends `temperature=0.7`, which the current Claude models reject.
- The CHANGELOG claims stale references were retired. They weren't.

The README lists five audiences, claims to be "the most comprehensive", compares itself with repos 30,000× larger, and has badges that link nowhere.

A **Reader** (a software engineer who already calls LLM APIs and must now ship an agent) sees nothing here that answers their real question. That question is how to know an agent is safe, correct and affordable enough to ship.

## Solution

Rebuild the course around one thesis: **"Your agent isn't done until it's graded, attacked and budgeted."**

The Reader builds a single **Flagship Agent** from scratch. It's a support/operations agent for **Acme Notes**, a fictional note-taking SaaS. The agent resolves **Cases** by answering from the Company's **Knowledge Base** and taking **Actions** on its deterministic mock **Backend**.

Seven **Spine** modules each add one capability, and each ends with **Checks** the Reader runs against *their own* agent:

1. Loop
2. Knowledge
3. Graded
4. Observed
5. Attacked
6. Budgeted
7. Shipped

Checks run on the Reader's own key under a hard **Spend Cap**. **Offline Checks** replay recorded model responses and need no key. A Claude Code **Grader** skill runs the Checks and explains failures. A dated **Scoreboard** of quality and cost per resolved Case across current models ships with the launch and is updated with every monthly **Drop**.

The old modules move to an **Appendix** and get a correctness-only refresh: current model IDs, no dead APIs, and a "last verified" date. A CI lint keeps them correct afterwards. The README is rewritten around the thesis, with a working Hugging Face Space and a Colab badge for each Spine module.

## User Stories

### Reader: first contact

1. As a Reader, I want the README's first screen to state the thesis and what I will have at the end, so that I can decide in seconds whether this course is for me.
2. As a Reader, I want to see the seven Spine modules as a single diagram, so that I understand the course is one agent getting hardened, not a pile of topics.
3. As a Reader, I want a live demo of the finished Flagship Agent resolving a Case, so that I can see the end state before investing time.
4. As a Reader, I want to run the finished agent's Offline Checks locally with no API key, so that my first run costs nothing and cannot fail on credentials.
5. As a Reader, I want a single command that runs the finished agent on a sample Case and prints its Outcome, so that I get a result within a minute of cloning.
6. As a Reader, I want every Spine module to open in Colab with one click, so that I can follow along without a local setup.
7. As a Reader, I want the README to show a current Scoreboard excerpt with its date, so that I can see the course is maintained and measures real models.
8. As a Reader, I want no claims I cannot verify (star comparisons, "most comprehensive", hour totals that don't add up), so that I trust the rest of the content.

### Reader: building the Flagship Agent

9. As a Reader, I want to build the agent loop myself from provider SDK calls, without a framework, so that I understand every step it takes.
10. As a Reader, I want the reference Flagship Agent at each module's end state to be small enough to read in one sitting, so that I can compare it with mine.
11. As a Reader, I want to point the Checks at my own agent implementation, so that I am graded on what I built, not on the reference.
12. As a Reader, I want to use any supported provider (Anthropic, OpenAI, Google, DeepSeek, Qwen, xAI, or a local OpenAI-compatible server), so that I can learn with the model I actually use at work.
13. As a Reader, I want the provider layer to handle each model's quirks (effort instead of temperature, Responses API for OpenAI tool calls), so that switching models doesn't break my agent with a 400 error.
14. As a Reader, I want to see *why* the provider layer drops or translates parameters, so that I learn how current models differ.
15. As a Reader, I want Actions such as refunding, changing a plan or escalating a Case to be tools my agent calls against the Backend, so that my agent does real work, not just chat.
16. As a Reader, I want the Backend to be deterministic and reset for every Case, so that the same Case always starts from the same state.
17. As a Reader, I want the Knowledge Base to contain Acme Notes' real-looking policies (refund window, proration, plan limits), so that correct answers depend on retrieval.
18. As a Reader, I want Knowledge retrieval to work without an embeddings API by default, so that Offline Checks stay free, with embeddings shown as an upgrade.

### Reader: Graded

19. As a Reader, I want Checks that assert on Backend state ("refunded exactly $20 to account A, once"), so that grading is deterministic and not a matter of taste.
20. As a Reader, I want an LLM-as-judge rubric for reply quality alongside the state assertions, so that I learn when a judge is appropriate and when it isn't.
21. As a Reader, I want to write my own Case and add it to the suite, so that I practise turning a production incident into a regression test.
22. As a Reader, I want the Checks to run in my CI with Offline Checks, so that regressions block merges without secrets.
23. As a Reader, I want the Checks for a module to include every earlier module's Checks, so that hardening never silently breaks earlier behaviour.

### Reader: Observed

24. As a Reader, I want every Case to produce a trace following the OpenTelemetry GenAI conventions (agent, chat, tool spans), so that my agent plugs into any tracing backend.
25. As a Reader, I want a Check that verifies the trace's structure, so that I know my instrumentation is complete, not just present.
26. As a Reader, I want token and cost numbers on every trace, so that the Budgeted module has data to work from.

### Reader: Attacked

27. As a Reader, I want Checks that plant a hostile article in the Knowledge Base, so that I see indirect prompt injection succeed against a naive agent and then fail against mine.
28. As a Reader, I want Checks for injection in the customer's own message, so that I cover direct injection too.
29. As a Reader, I want Checks where the customer asks for an Action on someone else's account, so that I learn authorization belongs in the tool layer, not the prompt.
30. As a Reader, I want Checks where a refund exceeds policy, so that I learn to put limits and escalation around high-risk Actions.
31. As a Reader, I want the Attacked Checks to assert "no forbidden Action happened" on Backend state, so that a polite refusal that still issued the refund fails.
32. As a Reader, I want attacks that evolve with each Drop, so that my agent keeps being tested against new techniques.

### Reader: Budgeted

33. As a Reader, I want a Check that my cost per resolved Case stays under a stated limit without lowering the Graded pass rate, so that I learn cost and quality trade-offs together.
34. As a Reader, I want to apply prompt caching, model routing and effort levels and see each one's effect in the Check report, so that I learn which lever moves cost.
35. As a Reader, I want a hard Spend Cap on every Check run, printed up front, so that a course about budgets never surprises me with a bill.
36. As a Reader, I want the run to stop cleanly when the Spend Cap is reached and report what finished, so that a runaway loop can't drain my key.

### Reader: Shipped

37. As a Reader, I want to serve my agent behind an HTTP endpoint with streaming, so that it's deployable like a real service.
38. As a Reader, I want a Check that sends a Case through the served endpoint, so that "shipped" is graded like everything else.
39. As a Reader, I want a container build and a Hugging Face Space recipe, so that I can put my own agent online.

### Reader: Grader

40. As a Reader using Claude Code, I want a Grader skill that runs my Checks and explains each failure with a pointer to the relevant lesson, so that I get unstuck without reading the whole module again.
41. As a Reader using another agent tool, I want the Grader to follow the open Agent Skills format, so that it works outside Claude Code where supported.
42. As a Reader, I want the Grader to run Offline Checks by default and live Checks only when I ask, so that it never spends my money unprompted.

### Reader: Appendix

43. As a Reader, I want Appendix topics (prompt engineering, fine-tuning, multimodal, graph engineering, agent frameworks, MCP deep dive, TypeScript) to stay available, so that I can go deeper where the Spine only touches a topic.
44. As a Reader, I want every Appendix page to show a "last verified" date, so that I know how fresh it is.
45. As a Reader, I want Appendix code to use current model IDs and live APIs, so that nothing I copy fails on the first run.
46. As a Reader, I want the MCP material to teach the `2026-07-28` spec, so that I build servers that match current clients.
47. As a Reader, I want the fine-tuning material to reflect that OpenAI self-serve fine-tuning is closed to new users, with open-weight fine-tuning as the primary path, so that I don't plan around a closing service.
48. As a Reader, I want agent-framework material (LangGraph 1.x, OpenAI Agents SDK, Google ADK 2.0) in the Appendix, so that I can map what I built by hand onto a framework later.
49. As a Reader, I want Spine modules to link to the Appendix page that goes deeper, so that the Spine stays short without losing depth.

### Maintainer

50. As the maintainer, I want one source of truth for current models, their prices and their supported parameters, so that a Drop means editing one file.
51. As the maintainer, I want a CI lint that fails on retired model IDs and dead APIs, so that the correctness refresh stays correct between Drops.
52. As the maintainer, I want the lint to allow a marked exception for historical discussion, so that a sentence like "GPT-4o introduced…" doesn't need to be deleted.
53. As the maintainer, I want to record live model responses for a Case once and replay them as Offline Checks, so that CI and first-contact runs are free and deterministic.
54. As the maintainer, I want the replay to fail loudly when a request no longer matches a recording, so that stale recordings don't pass silently.
55. As the maintainer, I want one command that runs every Case across the model list and writes a dated Scoreboard, so that each Drop's Scoreboard is reproducible.
56. As the maintainer, I want the Scoreboard labelled as "our Cases, our Checks", so that it isn't mistaken for a general benchmark.
57. As the maintainer, I want the Scoreboard run to respect a Spend Cap too, so that a Drop has a known upper bound on spend.
58. As the maintainer, I want a Drop template (model and API refresh, one new Case or attack, Scoreboard update, "What's New" entry), so that every monthly Drop has the same small shape.
59. As the maintainer, I want the CHANGELOG entries to describe only what is true, so that the old false claim doesn't happen again.
60. As the maintainer, I want the absorbed capstone and the removed Kaggle folder to leave no dangling links, so that the repo looks alive.
61. As the maintainer, I want the tests to run without network access or API keys, so that CI stays green for contributors.

### Sharing

62. As the maintainer, I want a README that works as the landing page for a LinkedIn or X post (thesis, diagram, demo link, Scoreboard excerpt above the fold), so that people who click through from a post convert to stars.
63. As a Reader sharing my progress, I want the Grader to print a short "passed through module N" summary, so that I can show evidence of what my agent withstands.

## Implementation Decisions

### Repository shape

- Top-level areas:
  - **Spine**: seven modules in order.
  - **Appendix**: topic folders.
  - **Company**: Acme Notes' Backend, Knowledge Base and Cases.
  - **Flagship Agent**: the reference implementation.
  - **Checks**, the **Case runner**, the **provider layer**, the **Grader** skill, and the **Scoreboard** output.
- The numbered module folders `00` to `16` go away. Their content moves to Appendix topic folders. Spine modules are new lessons that reuse that content and link into it.
- `capstone/` is absorbed and removed. Its ideas carry over (memory, guardrail and cost tracking); its LangChain code does not (ADR 0003).
- `demo.py` becomes the one-command "run the finished agent on a sample Case" entry point, Offline by default.
- `kaggle/` is removed. `typescript/` moves to the Appendix.

### Provider layer (seam 2)

- The public interface is one call that takes normalized messages, a normalized tool list and a small set of intent-level options (effort, max output). It returns a normalized response: text, tool calls, usage and stop reason.
- Normalized tools have a name, a description and a JSON-schema input. The layer translates them to Anthropic Messages tool use, the OpenAI Responses API (required for tool calls on current OpenAI models), Google Gemini, and OpenAI-compatible Chat Completions for DeepSeek, Qwen, xAI and local servers.
- It knows each model's capabilities through the **model registry**: a single data file listing each current model's ID, provider, prices, context window, whether it accepts sampling parameters, supported effort levels, and API surface. It drops or translates unsupported options instead of passing them through. Temperature is never sent to models that reject it.
- It supports a **replay transport**: in record mode it stores each normalized request and response. In replay mode it returns the recorded response for a matching request and fails with a clear error on a mismatch.
- The old `shared/provider.py` API is removed, and its callers in Appendix code migrate to the new layer.

### Company: Acme Notes

- The **Backend** is an in-memory, deterministic model of accounts, plans (Free, Pro, Team), subscriptions, invoices, refunds and escalations. It is seeded from fixture data, reset for every Case, and able to export its state for assertions.
- **Actions** are the Backend operations exposed as agent tools: look up account, change plan, issue refund, escalate Case, add note. Each Action enforces its own authorization (the acting Case's customer only) and policy limits (refund window, maximum refund, proration), independently of the prompt.
- The **Knowledge Base** is a set of Acme Notes help-centre and policy articles in markdown. Default retrieval is lexical (BM25-style, pure Python) so Offline Checks need no embeddings. An embeddings retriever is taught in Spine module 2 as an upgrade behind the same interface.
- A **Case** is a data file with:
  - the customer identity and opening message, plus optional follow-up turns;
  - optional Knowledge Base overrides (for planted hostile articles);
  - the expected Backend state change, and forbidden Actions;
  - optional judge rubric;
  - tags (module, category, attack type).

### Case runner (seam 1)

- The interface is to run a Case against an agent and get an **Outcome**. The Outcome holds:
  - the final Backend state and the diff from the seed;
  - the list of Actions attempted and executed;
  - the transcript;
  - token usage and cost;
  - the trace;
  - a resolved/unresolved flag.
- The agent contract is a callable that receives the Case's customer turn and an environment (Actions as tools, Knowledge Base retriever, provider layer, tracer). The runner loads the agent from an importable reference, so a Reader can point it at their own code.
- The runner enforces the **Spend Cap** across a run, using usage multiplied by registry prices. It prints the cap before starting, stops at the cap, and reports partial results.
- Offline and live modes are chosen per run. Offline uses the replay transport.

### Checks

- A Check is a pass/fail assertion over an Outcome, or over a set of Outcomes for aggregate checks.
- Each Spine module owns a Check suite, and a module's run includes every earlier module's suite.
- Kinds of Checks:
  - Backend-state assertions (Graded).
  - LLM-as-judge rubric Checks, live only, marked as such.
  - No-forbidden-Action assertions (Attacked).
  - Trace-structure assertions against the OpenTelemetry GenAI conventions, pinned to a named convention version because they are still in Development (Observed).
  - Cost per resolved Case, at or below the limit, with the Graded pass rate not lower than baseline (Budgeted).
  - Served-endpoint Checks (Shipped).
- The Checks CLI takes a module range, an agent reference, a mode (offline or live) and a Spend Cap. It prints a per-Check result and a one-line "passed through module N" summary.

### Spine modules

- Each Spine module has a lesson README, a Colab-ready notebook, the reference Flagship Agent's changes for that step, and its Check suite. They are:
  1. **Loop**: agent loop, tool calling, stop conditions, step limits.
  2. **Knowledge**: retrieval over the Knowledge Base, memory across a Case's turns.
  3. **Graded**: state assertions, judge rubrics, writing Cases, CI.
  4. **Observed**: OpenTelemetry GenAI tracing, cost on spans.
  5. **Attacked**: direct and indirect injection, tool-layer authorization, policy limits, escalation.
  6. **Budgeted**: caching, routing, effort levels, cost per resolved Case.
  7. **Shipped**: HTTP service with streaming, container, Hugging Face Space.
- The reference Flagship Agent's final state stays small enough to read in one sitting. Each module's lesson shows the change it makes.

### Grader

- A Claude Code skill in the open Agent Skills format. It runs the Checks CLI (Offline by default, live only when asked), reads failures, and explains each one with a link to the Spine lesson section that covers it.

### Scoreboard

- One maintainer command runs the Scoreboard Case set across the registry's model list, in live mode under a Spend Cap. It writes a dated markdown table and a JSON file (resolution rate, Attacked pass rate, cost per resolved Case, per model).
- The README embeds the latest excerpt with its date and the label "our Cases, our Checks — not a general benchmark".

### Appendix correctness refresh

- Every Appendix page:
  - uses current model IDs from the registry;
  - removes dead or retired APIs (OpenAI Assistants API, OpenAI Evals platform and Agent Builder, pre-1.0 LangChain chains and imports, `chat.completions` for OpenAI tool calls, MCP `2025-06-18`);
  - reflects that OpenAI self-serve fine-tuning is closed to new users;
  - gains a "last verified" date.
- There is no new material beyond what correctness requires. Broken or empty "Why this matters" intros are filled or removed.
- The MCP material is rewritten to the `2026-07-28` spec: stateless requests, `server/discover`, the multi-round-trip `InputRequiredResult` replacing server-initiated elicitation and sampling, and Tasks as an extension.

### Stale-reference lint

- A denylist data file of retired model IDs and dead API patterns, each with a replacement hint.
- A test that scans Markdown, Python, TypeScript and notebooks and fails on any match not carrying an inline historical-exception marker.
- The test runs in CI.

### README and repo presentation

- The README is rewritten, with these above the fold:
  - the thesis;
  - one line on who it's for;
  - a Spine diagram (mermaid);
  - the demo link;
  - the Offline quick start;
  - the Scoreboard excerpt.
- Badges are removed unless they link somewhere real (Hugging Face Space, CI status, Colab).
- The superlatives, star-comparison table and hour totals are removed.
- A "What's New" section is added at the top for Drops.
- The CHANGELOG gets a new major entry describing only what is true.

### Dependencies

- Python 3.11+. Current `anthropic` (1.x), `openai` and `google-genai` SDKs.
- LangChain and LangGraph only in the Appendix framework pages.
- OpenTelemetry SDK with the GenAI conventions.
- No other new core dependencies without need.

## Testing Decisions

- Good tests assert external behaviour through the highest seam. For Checks and the Flagship Agent, that means an Outcome from the Case runner. They do not test loop internals, retriever scoring or prompt text.
- **Seam 1, the Case runner**, is tested in Offline mode using the replay transport:
  - the reference agent at each module's end state passes that module's cumulative suite;
  - a deliberately naive agent fails the Attacked suite;
  - a Spend Cap below a run's cost stops the run and reports partial results;
  - a replay mismatch raises a clear error;
  - an agent reference that points outside the repo is loaded and graded.
- **Seam 2, the provider layer**, is tested on the outgoing request. For each registry model it asserts that unsupported options are dropped or translated (no temperature where rejected; the Responses API for OpenAI tool calls) and that normalized tools round-trip into each provider's shape. These tests use a stub transport, not the network.
- **The Backend** gets a small set of direct tests of Action policy (refund window, cross-account denial), because Checks rely on its determinism. Everything else about it is covered through the Case runner.
- **The stale-reference lint** is itself a test. It includes a fixture proving that the historical-exception marker works.
- The whole test suite runs with no network and no API keys. Live-only Checks (judge rubrics) are skipped in Offline mode and reported as skipped.
- Prior art: the existing module tests already avoid API calls, and the existing CI notebook smoke test validates notebook structure. Both are kept and extended. The old per-module tests are replaced where their subject moves into the Spine.

## Out of Scope

- Translations, a docs website, video, certification tracks.
- A general-purpose benchmark. The Scoreboard measures only this course's Cases and Checks.
- New Appendix material beyond correctness (for example new multimodal or graph topics).
- Stripe test mode or any real external backend (possible later Appendix bonus).
- A TypeScript version of the Spine (TypeScript stays an Appendix page).
- Posting to LinkedIn or X, renaming the GitHub repo, and publishing the Hugging Face Space. Each is an outward-facing action that needs the maintainer's explicit go-ahead at launch.

## Further Notes

- **Hugging Face Space mode**: the public Space defaults to replaying recorded Cases, which is free and deterministic. A live mode, if enabled, uses the maintainer's key under a Spend Cap. Visitors are never asked to paste API keys into the Space.
- **Repo rename** to `agent-to-production` is agreed. It happens at launch, and GitHub redirects old URLs.
- **Drop cadence** is monthly with a fixed shape (see story 58). Scoreboard #1 ships with the launch.
- Model facts used here (current IDs, prices, the MCP `2026-07-28` changes, OpenAI deprecations) come from research on 2026-09-26. Each must be re-verified against vendor docs when the model registry and the Appendix refresh are written, and any not confirmed by a vendor source are left out.
