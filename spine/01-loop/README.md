# Spine 1 · Loop

_Last verified: 2026-09-26_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnakhoa/llm-engineering-playground/blob/main/spine/01-loop/loop.ipynb)

**What your agent can do at the end:** resolve one real Case. A customer on
Acme Notes' Free plan asks to move to Pro, and your agent looks the account up,
changes the plan, and replies. A Check confirms the plan changed, once.

**What you build:** the agent loop. A model call, tool calls, two stop
conditions and a step limit, in about twenty lines of Python with no framework.

## The world your agent works in

Acme Notes is a note-taking SaaS with Free, Pro and Team plans (see
[CONTEXT.md](../../CONTEXT.md) for every term used here).

- **The Backend** (`company/backend.py`) is Acme Notes' deterministic mock of
  accounts and plans. Every Case starts from the same seed
  (`company/fixtures/accounts.json`), and `export_state()` gives the Checks
  something exact to assert on.
- **Actions** are the Backend operations your agent calls as tools. Spine 1
  has two: `look_up_account` and `change_plan`.
- **A Case** (`company/cases/upgrade-to-pro.json`) is one customer request:
  who the customer is, what they say, and which Backend change resolves it.

## The agent contract

The Case runner calls your agent like this:

```python
reply = agent(customer_turn, env)
```

`env` gives the agent everything it is allowed to touch:

| | |
|---|---|
| `env.tools` | the Actions, as `ToolSpec`s the model can call |
| `env.act(call)` | run one tool call against the Backend, get a `ToolResult` |
| `env.complete(messages, tools=..., system=...)` | one model call through the [provider layer](../../llm/README.md), with the model and transport already chosen |
| `env.step_limit` | how many model calls this Case allows |

Because the agent never picks its own transport, the same code runs Offline
(replayed, free) and live (your key), and the Checks can grade it either way.

## The loop

This is `flagship/loop.py`, the reference Flagship Agent at the end of Spine 1:

```python
def run(customer_turn, env):
    messages = [Message.user(customer_turn)]

    for _ in range(env.step_limit):
        response = env.complete(messages, tools=env.tools, system=SYSTEM)
        messages.append(Message.assistant(response.text or None, response.tool_calls))

        if not response.tool_calls:  # stop condition: the model has answered
            return response.text

        results = [env.act(call) for call in response.tool_calls]
        messages.append(Message.tool(results))

    # Stop condition: out of steps. The Case runner reports it on the Outcome.
    raise StepLimitReached(env.step_limit)
```

Read it line by line:

1. **The transcript is a list you own.** Every model turn and every tool
   result is appended to `messages`, and the whole list goes back to the model
   on each call. The model has no memory between calls; this list is it.
2. **The model asks for tools; your code runs them.** A `Response` carries
   `tool_calls`. The agent runs each one with `env.act` and sends the results
   back as a tool turn. The provider layer turns that into whatever the chosen
   provider expects (a `tool_result` block, a `function_call_output`, a
   `functionResponse`).
3. **Stop condition one: the model answered.** A turn with no tool calls is the
   final reply.
4. **Stop condition two: the step limit.** A model that keeps calling tools
   would keep spending money. After `env.step_limit` calls the loop stops and
   raises `StepLimitReached`.

The runner enforces the step limit too: `env.complete` refuses to make a call
past the limit. A Reader's agent without a limit of its own, stuck in a loop,
is stopped all the same, and the Outcome says `step_limit_reached=True`.

## Authorization belongs in the Action

The system prompt says "only act on the customer's own account". That is a
request, not a control: the customer's message can argue with it. So
`change_plan` and `look_up_account` check the account against the Case's
customer inside the Action and refuse anything else:

```python
actions = Backend.seeded().actions_for("acct_1001")
actions.run(ToolCall(id="c1", name="change_plan",
                     arguments={"account_id": "acct_1002", "plan": "free"}))
# ToolResult(is_error=True,
#            content="Refused: acct_1002 is not the account of the customer on this Case.")
```

The refusal goes back to the model as an error result, so the agent can
explain it to the customer. The attempt is still logged: the Outcome lists it
under `actions_attempted` and not under `actions_executed`. The Attacked module
builds on exactly this.

## The Outcome

```python
from company.runner import load_case, run_case

outcome = run_case(load_case("upgrade-to-pro"), "flagship.loop:run")
```

`run_case` accepts the agent itself or an importable reference such as
`"my_agent.loop:run"`, so you can point it at your own code. It returns an
Outcome with:

- `final_state` and `diff`: the Backend after the Case, and every field that
  changed, for example `{"accounts.acct_1001.plan": {"before": "free", "after": "pro"}}`;
- `actions_attempted` and `actions_executed`;
- `transcript`: the customer turn, every model turn, every tool result;
- `usage` and `cost_usd`, priced from the model registry;
- `steps`, `step_limit` and `step_limit_reached`;
- `spend_cap_reached`: the run's Spend Cap stopped this Case;
- `resolved`: the Case's expected change happened, no forbidden Action ran,
  and neither the step limit nor the Spend Cap stopped the run.

## The Check

`checks/spine_1_loop.py` holds this module's one Check: the customer's plan
ended on Pro, moved there by exactly one `change_plan`. It reads Backend state
and executed Actions, not the reply. An agent that says "Done, you're on Pro!"
without calling the Action is unresolved and fails.

## Offline: why this costs nothing

`mode="offline"` (the default) replays
`company/recordings/upgrade-to-pro.recording.json` through the provider
layer's replay transport. The model replies in it were written by hand and the
requests captured from the reference loop, so it is a reviewed recording with
nothing private in it. A replay answers only a request it has seen: change the
prompt, a tool, the model or an option that reaches the wire, and it raises
`ReplayMismatchError` naming the fields that moved.

That is also why a Reader's own agent needs a live run to be graded: its
requests are its own. Run it on your key with `mode="live"`, or record it once
with `RecordingTransport` (see the [provider layer](../../llm/README.md)) and
replay it free from then on.

## Run the Checks on your agent

```bash
python -m checks --modules 1 --agent path/to/my_agent.py:run            # Offline, free
python -m checks --modules 1 --agent path/to/my_agent.py:run \
    --mode live --spend-cap 0.50                                         # your key, capped
python -m checks --modules 1 --agent path/to/my_agent.py:run --mode live \
    --model qwen3:8b --base-url http://localhost:11434/v1                # a local server
```

The agent can live anywhere on disk; `package.module:function` works too. A
live run reads your keys from the root `.env`, and without `--model` runs on the
model your `.env` selects, the same one `llm.default_model()` picks; an Offline
run stays on the model its recordings were made on. The
Spend Cap is printed before anything runs, and a run that reaches it stops and
reports what finished. From module 2 on, `--modules N` also runs every earlier
module's Checks. The last line, "Passed through module N of M.", is the one to
share.

## Try it

- Run the notebook, Offline, top to bottom.
- Write your own `run(customer_turn, env)` without looking at the reference,
  and run it live against the Case.
- Lower `step_limit` until your agent can no longer resolve the Case. How many
  model calls does this Case need, at minimum?
- Ask your agent, live, to change the plan of `acct_1002`. Check
  `actions_attempted` against `actions_executed`.

## Going further

The Appendix covers agent frameworks (LangGraph, the OpenAI Agents SDK, Google
ADK) for when you want to map this loop onto one. The course builds the loop
without one on purpose: see
[ADR 0003](../../docs/adr/0003-flagship-agent-built-without-a-framework.md).
