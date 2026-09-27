# Spine 4 · Observed

_Last verified: 2026-09-27_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnakhoa/llm-engineering-playground/blob/main/spine/04-observed/observed.ipynb)

**What your agent can do at the end:** every Case it works leaves an
OpenTelemetry trace: one agent span, a chat span for each model call and a
tool span for each tool call, with tokens and cost on them. Any tracing backend
that speaks OpenTelemetry can show it, and the Budgeted module prices your
agent from it.

**What you build:** almost nothing, which is the point. The trace comes on the
Outcome next to Spine 3's `verdict`, not in place of it: the verdict says
whether the Case was resolved, the trace shows how the agent got there. The tracing lives in
the environment your agent already works through. Your agent adds the one
thing only it knows: its name. The Checks then make sure the trace is complete,
not just present.

## The pinned convention version

The spans follow the OpenTelemetry GenAI semantic conventions, pinned to
**OpenTelemetry semantic conventions 1.41.0**. The pin is in the code, as
`SEMCONV_VERSION` in `company/tracing.py`, and every span carries it as its
schema URL, `https://opentelemetry.io/schemas/1.41.0`.

Why pin at all: the GenAI conventions are still marked Development, which means
their names can change in any release, and they have. `gen_ai.system` was
replaced by `gen_ai.provider.name`, and later releases of the main
semantic-conventions repository mark every `gen_ai.*` name deprecated there,
because the GenAI conventions moved to a repository of their own. 1.41.0 is a
release that defines every name this course uses, in the repository most tools
read. When you move to a newer version, you change the constants in
`company/tracing.py`; the Checks read the same constants, so they move with it.

## What a trace looks like

```text
invoke_agent acme-support            INTERNAL  the Case: every turn, totals
├── chat claude-sonnet-5             CLIENT    one model call: tokens, cost
├── execute_tool look_up_account     INTERNAL  one tool call: name, call id
├── chat claude-sonnet-5
├── execute_tool change_plan
└── chat claude-sonnet-5
```

That is `upgrade-to-pro`, replayed Offline: three model calls and two tool
calls, all in one trace under one agent span.

| Span | Attributes it carries |
|---|---|
| `invoke_agent {name}` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.conversation.id` (the Case id), `gen_ai.agent.name` and `gen_ai.agent.version` (from your agent), the Case's total tokens and cost |
| `chat {model}` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `acme.cost_usd` |
| `execute_tool {tool}` | `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `gen_ai.tool.description` |

Three details worth knowing:

- **Input tokens include cached tokens.** The conventions say the cache-read and
  cache-write counts are part of `gen_ai.usage.input_tokens`, not added to it.
  The provider layer already counts them that way for every vendor.
- **Cost is not in the conventions.** They define tokens, not money, so cost
  goes on `acme.cost_usd`, in US dollars at registry prices. It is in our own
  namespace because `gen_ai.*` belongs to the conventions: a name we invented
  there could collide with one they add later.
- **Errors are marked.** A refused Action is an `execute_tool` span with status
  `ERROR` and `error.type` `tool_error`. A Case the step limit or the Spend Cap
  stopped is an agent span with `error.type` `step_limit_reached` or
  `spend_cap_reached`. In a backend, that is one filter for "what went wrong".

The prompts and replies themselves (`gen_ai.input.messages`,
`gen_ai.output.messages`) are Opt-In in the conventions, and this course leaves
them off. A support agent's prompts hold customers' names and accounts, and a
trace goes to more places than a database does. The transcript is on the
Outcome if you need it.

## Where the spans come from

Your agent already calls the model through `env.complete` and runs tools
through `env.act`, so that is where the spans are made (`company/runner.py`).
It is the same place an instrumentation library hooks into a framework's model
client. The Case runner opens the agent span around the whole Case, every turn
of it.

That gives one rule: **every tool call goes through `env.act`.** A tool your
agent runs some other way, say a Knowledge Base search it answers itself with
`run_search`, still works, and the Backend still ends right. It just leaves no
span, so in a tracing backend the search never happened. The Checks fail it.

The one thing the environment cannot know is which agent this is. The
conventions put that on the agent span as `gen_ai.agent.name`, "if provided by
the application", so the Spine 4 agent, `flagship/observed.py`, provides it:

```python
AGENT_NAME = "acme-support"
AGENT_VERSION = "spine-4"


def run(customer_turn, env):
    env.describe_agent(AGENT_NAME, version=AGENT_VERSION, description=AGENT_DESCRIPTION)
    return knowledge_run(customer_turn, env)
```

`knowledge_run` is the Spine 2 loop that Spine 3 graded, untouched. The name
becomes the span name too, `invoke_agent acme-support`. In a backend
it is the filter that finds this agent's traces among every other service's,
and the version tells this week's regression from last week's. The name goes
on a span, never into a prompt, so every request is still the recorded one.

`env.tracer` is an OpenTelemetry tracer on the same trace. A span your agent
starts with it, around a planning step for example, lands under the agent span.

## The trace on the Outcome

```python
from company.runner import load_case, run_case

outcome = run_case(load_case("upgrade-to-pro"), "flagship.observed:run")
for span in outcome.trace.spans:
    print(span.name, span.attributes.get("gen_ai.usage.input_tokens"))
```

`outcome.trace` is a `Trace`: plain records (`SpanRecord`) with a name, kind,
ids, parent id, attributes and status. `trace.chat_spans`, `trace.tool_spans`
and `trace.agent_spans` pick out one operation. A Check reads it without
touching the OpenTelemetry SDK.

## Export to any OpenTelemetry backend

The same spans go, as they end, to every exporter you pass to `run_case`. An
exporter is the standard OpenTelemetry `SpanExporter`, so anything that
accepts OpenTelemetry takes this trace: Jaeger, Grafana Tempo, Honeycomb,
Langfuse, Phoenix, your cloud's tracing service.

Print the spans, with nothing to install beyond the SDK:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

run_case(load_case("upgrade-to-pro"), "flagship.observed:run", exporters=[ConsoleSpanExporter()])
```

Send them over OTLP, the protocol every backend accepts. Jaeger on your laptop
is one container, with its UI at <http://localhost:16686>:

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/jaeger:latest
pip install opentelemetry-exporter-otlp-proto-http
```

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
for case_id in ["upgrade-to-pro", "downgrade-with-prorated-refund"]:
    run_case(load_case(case_id), "flagship.observed:run", exporters=[exporter])
```

For a hosted backend, leave out `endpoint` and set the standard variables
instead: `OTEL_EXPORTER_OTLP_ENDPOINT` to the address your vendor gives you and
`OTEL_EXPORTER_OTLP_HEADERS` to its API key header. Keep the key in your `.env`,
never in code. One exporter serves every Case: the runner flushes it after each
Case and never shuts it down. The spans report `service.name`
`acme-notes-support-agent`, or whatever you set in `OTEL_SERVICE_NAME`.

## The Checks

`checks/spine_4_observed.py` runs six Checks on every Case so far, the
single-turn upgrade, the Knowledge Base Cases and the multi-turn one:

- **Trace is one tree under the agent span.** One trace, one `invoke_agent`
  span at its root, everything else below it.
- **Spans follow GenAI conventions 1.41.0.** Every span has the pinned schema
  URL, and each GenAI span has its operation's name, kind and required
  attributes. A span of your own that is not a GenAI operation is left alone.
- **Every model call has a chat span.** As many `chat` spans as model calls.
- **Every tool call has a tool span.** Each tool call the model asked for has
  an `execute_tool` span with that call's id. This is the one that catches a
  tool run outside `env.act`.
- **Span tokens and cost match the Outcome.** The chat spans' tokens and cost
  add up to the Outcome's usage and cost, and so do the agent span's totals.
  The Budgeted module prices your agent from the trace, so a trace that
  undercounts would make it look cheaper than it is.
- **The agent span names the agent.** `gen_ai.agent.name` is set, so a backend
  can find your agent's traces. This one is a warning: it prints `WARN` and
  never fails a module, the exit code or your badge. The conventions set the
  name only "if provided by the application", so an agent that leaves it out
  breaks none of them. Name yours anyway, with `env.describe_agent`: in a
  backend full of other services' traces, an unnamed agent is hard to find.

The Spine 2 agent passes the first five as it is, because the environment
traces it, and gets the naming warning until it names itself.

The suite lists only these six. A module 4 run runs modules 1 to 3 first, and
the Checks on the verdict hold on every Case, so each Case here is graded on the
whole verdict once, and a Case a later suite adds is graded on it too without
listing those Checks again. Listing them would print their results twice.

## Installing the SDK

The tracing needs the OpenTelemetry SDK, which is in the course's
`requirements.txt`. On its own:

```bash
pip install opentelemetry-sdk
```

The notebook installs it for you if it is missing, which is what happens on
Colab.

## Run the Checks on your agent

```bash
python -m checks --modules 4 --agent path/to/my_agent.py:run      # modules 1 to 4, Offline
```

`--modules 4` runs every earlier module's Checks too. Without `--agent`, the
CLI grades the reference agent at the latest module, `flagship.observed:run`.

## Try it

- Run the notebook, Offline, top to bottom.
- Export a few Cases to Jaeger. Then run an agent that calls an Action its
  Case does not declare, and find the refusal in Jaeger by its error status.
- Add a span of your own with `env.tracer` around your agent's turn, and check
  it lands under the agent span.
- Run a live Case with prompt caching on, and read the cache-read tokens on the
  second chat span.

## Going further

[Observability](../../appendix/observability/) in the Appendix covers metrics,
dashboards and the hosted tracing tools in more depth. Spine 6, Budgeted, reads
cost per resolved Case from the numbers this module puts on the spans.
