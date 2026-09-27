# Module 07: Agent Frameworks

_Last verified: 2026-09-27, against LangGraph 1.2.12, the OpenAI Agents SDK 0.22.3 and Google ADK 2.10.0_

> **Why this matters:** In the Spine you build the Flagship Agent's loop by hand, with no framework ([ADR 0003](../../docs/adr/0003-flagship-agent-built-without-a-framework.md)), so you can read every step it takes. Sooner or later you will meet a framework in someone else's codebase, or want one of its runtime features. This page maps what you built onto the three you are most likely to meet — so a framework's name for a thing never hides what the thing is.

## Learning Objectives
- Recognise the agent loop, tools, step limits, memory and human approval in LangGraph 1.x, the OpenAI Agents SDK and Google ADK 2.0
- Know what each framework adds on top of the loop you wrote, and what it hides
- Build the same loop as a LangGraph `StateGraph`, over the course's provider layer ([agentic_workflows.ipynb](agentic_workflows.ipynb))
- Apply framework-agnostic multi-agent patterns: supervisor/worker, pipeline vs barrier, adversarial verification, swarm

## 🧭 What you built by hand, in each framework

Each row is a concept from the Spine. The link goes to the lesson where you built it yourself; the other columns name the same thing in each framework.

| Concept | Where you built it | LangGraph 1.x | OpenAI Agents SDK | Google ADK 2.0 |
|---------|--------------------|---------------|-------------------|----------------|
| **The agent loop**: call the model, run its tool calls, send results back, repeat | [Spine 1 · The loop](../../spine/01-loop/README.md#the-loop) | A `StateGraph` with a model node, a tool node and a conditional edge back | `Runner.run(agent, input)` runs the loop for you | A `Runner` runs an `Agent`; in 2.0 the agent is a node in the workflow graph engine |
| **Tools / Actions**: a name, a description, a JSON schema, and your function behind it | [Spine 1 · The agent contract](../../spine/01-loop/README.md#the-agent-contract) | Plain functions called from a node (or LangChain `@tool`s with the prebuilt `ToolNode`) | `@function_tool` on a Python function; the schema comes from its signature | Plain functions in `Agent(tools=[...])`, wrapped as `FunctionTool` |
| **Stop conditions and the step limit** | [Spine 1 · The loop](../../spine/01-loop/README.md#the-loop) | The graph ends at `END`; `recursion_limit` in the run config raises `GraphRecursionError` | A final output ends the run; `max_turns` (default 10) raises `MaxTurnsExceeded` | The agent's final response ends the run; `RunConfig(max_llm_calls=...)` caps model calls |
| **Authorization belongs in the Action**, not the prompt | [Spine 1 · Authorization belongs in the Action](../../spine/01-loop/README.md#authorization-belongs-in-the-action) | Inside the tool function or the node that runs it | Inside the tool; tool guardrails (`@tool_input_guardrail`) run before each call | Inside the tool; `before_tool_callback` can inspect or block a call |
| **The Outcome**: what the run did, for a Check to assert on | [Spine 1 · The Outcome](../../spine/01-loop/README.md#the-outcome) | The final state returned by `invoke()` | A `RunResult`: `final_output`, `new_items`, usage | The stream of `Event`s the `Runner` yields, and the session's state |
| **Retrieval over the Knowledge Base** | [Spine 2 · How the agent reaches the Knowledge Base](../../spine/02-knowledge/README.md#how-the-agent-reaches-the-knowledge-base) | A retrieval tool, or a retrieval node before the model node | A function tool (or a hosted `FileSearchTool` on OpenAI's side) | A function tool, or one of ADK's grounding tools |
| **Each Case declares its Actions**: the agent is offered only those | [Spine 2 · Each Case declares its Actions](../../spine/02-knowledge/README.md#each-case-declares-its-actions) | Build the tool list per run and pass it to the model node | Build the `Agent(tools=...)` per Case, or `agent.clone(tools=...)` | Build the `Agent(tools=...)` per Case |
| **Policy belongs in the Action**: refund window, limits, proration | [Spine 2 · Policy belongs in the Action](../../spine/02-knowledge/README.md#policy-belongs-in-the-action) | In the tool function — the graph cannot enforce it for you | In the tool function; a guardrail can add a check, not replace it | In the tool function; a callback can add a check, not replace it |
| **Memory across a Case's turns** | [Spine 2 · Memory across a Case's turns](../../spine/02-knowledge/README.md#memory-across-a-cases-turns) | A checkpointer (`InMemorySaver`, or a database one) keyed by `thread_id` | A `Session` (`SQLiteSession("case-42")`) passed to `Runner.run` | A `SessionService` (`InMemorySessionService`) holding the session's events and state |
| **Offline: replaying recorded model responses** | [Spine 1 · Offline: why this costs nothing](../../spine/01-loop/README.md#offline-why-this-costs-nothing) | Nothing built in: swap the model call in your node | A custom `Model` implementation, or the SDK's testing helpers | A custom model class |

Two concepts come later in the Spine and are not mapped here yet: **human approval and escalation** (the Attacked module) and **tracing** (the Observed module). The frameworks name them `interrupt()` + `Command(resume=...)` (LangGraph), `needs_approval` on a tool + `RunState` (OpenAI Agents SDK), and human-input nodes in a workflow graph (ADK 2.0); and each ships its own tracing — the OpenAI Agents SDK traces by default and exports to OpenAI unless you disable it (`OPENAI_AGENTS_DISABLE_TRACING=1`) or replace its processors.

## LangGraph 1.x

[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) models an agent as a graph of nodes over a shared, typed state. A node is a plain function that returns an update to the state; edges, including conditional ones, decide what runs next.

- **What it adds**: explicit, inspectable control flow; cycles; checkpointers that persist state per `thread_id`; `interrupt()` to pause for a human and `Command(resume=...)` to continue; streaming of each step.
- **What it hides**: little — you still write the model call. That is why the notebook runs LangGraph over this course's provider layer (`llm.complete()`) instead of LangChain's chat-model wrappers: LangGraph 1.x does not require them.

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]     # each node's messages are appended

def model_node(state):   # one model call through llm.complete(), as in Spine 1
    response = call_model(state["messages"], tools=TOOL_SPECS)
    return {"messages": [Message.assistant(response.text or None, response.tool_calls)]}

def tools_node(state):   # run what the model asked for
    return {"messages": [Message.tool([run_tool(c) for c in state["messages"][-1].tool_calls])]}

graph = StateGraph(AgentState)
graph.add_node("model", model_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
graph.add_edge("tools", "model")
app = graph.compile()

app.invoke({"messages": [Message.user("What is 15% of 840?")]}, config={"recursion_limit": 10})
```

The notebook builds this graph, a multi-agent router, and a human approval step with `interrupt()`.

## OpenAI Agents SDK

The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) is a small runtime around the loop: **agents** (a model with instructions and tools), **handoffs** and agents-as-tools for delegation, **guardrails** (input, output and tool guardrails), **sessions** for memory, and built-in **tracing**. It calls OpenAI models through the Responses API by default; other providers go through its model interfaces and adapters.

- **What it adds**: the loop, tool dispatch and turn limit; delegation between agents; guardrails that run alongside the loop and fail fast; pausing for human approval (`needs_approval`) and resuming from a serialized `RunState`; tracing on by default.
- **What it hides**: the loop itself. A run is one `Runner.run()` call, so what the model was sent each turn is in the trace, not in your code.

```python
from agents import Agent, Runner, SQLiteSession, function_tool

@function_tool
def look_up_account(account_id: str) -> str:
    """Look up an Acme Notes account: plan, status and invoices."""
    return backend.look_up_account(account_id)   # authorization and policy stay in here

agent = Agent(
    name="Acme Notes support",
    instructions="Resolve the customer's request. Act only on their own account.",
    model="gpt-6-luna",
    tools=[look_up_account],
)

result = await Runner.run(agent, "Can you move me to Pro?", session=SQLiteSession("case-42"), max_turns=8)
print(result.final_output)
```

## Google ADK 2.0

[Google's Agent Development Kit](https://google.github.io/adk-docs/) reached 2.0 for Python on 19 May 2026. 2.0 introduces a Workflow Runtime: agents, tools and functions are nodes in a graph-based execution engine, and `BaseAgent` now subclasses `BaseNode`. On top of single agents it offers three ways to compose work: **graph-based workflows** (`Workflow` with explicit edges), **dynamic workflows** in your own code, and the prebuilt **workflow agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`).

- **What it adds**: sessions and state, callbacks around every agent, model and tool call (`before_tool_callback` and the rest), deterministic graph workflows that mix code and model calls, human-input nodes, and a runner with a dev UI and API server.
- **What it hides**: the loop and the event log. A run is a stream of `Event`s; 2.0 added `node_info` and `output` fields to them, which matters if you store sessions yourself.

```python
from google.adk import Agent, Workflow

def look_up_account(account_id: str) -> dict:
    """Look up an Acme Notes account: plan, status and invoices."""
    return backend.look_up_account(account_id)   # authorization and policy stay in here

support = Agent(
    name="support",
    model="gemini-3.8-flash",
    instruction="Resolve the customer's request. Act only on their own account.",
    tools=[look_up_account],
)

def log_outcome(node_input: str):
    """A plain function node: no model call."""
    return f"Resolved: {node_input}"

root_agent = Workflow(name="support_flow", edges=[("START", support, log_outcome)])
```

## 🧠 Choosing

| If you need… | Reach for |
|--------------|-----------|
| To read and own every step (the Flagship Agent) | The loop you wrote in [Spine 1](../../spine/01-loop/README.md) |
| Explicit control flow, checkpoints, human interrupts, any provider | LangGraph 1.x |
| A small runtime with handoffs, guardrails and tracing, mostly on OpenAI models | OpenAI Agents SDK |
| Deterministic graph workflows mixing code and agents, on Google Cloud or Gemini | Google ADK 2.0 |

Whichever you pick, keep authorization and policy inside the Actions, and keep the Checks. A framework changes how the loop is written, not what your agent must withstand.

---

## 🚀 Framework-agnostic Multi-Agent Patterns

These patterns work the same with or without a framework.

### Agent-Computer Interface (ACI) Design

Anthropic's research on building effective agents found that **tool design matters more than prompt design**. They spent more time optimizing tools than the overall agent prompt for their SWE-bench agent.

**Core ACI principles**:

```python
# ❌ BAD: Relative paths break when agent changes directories
@mcp.tool()
def edit_file(path: str, content: str) -> str:
    """Edit a file."""

# ✅ GOOD: Absolute paths always work — model never makes this mistake
@mcp.tool()
def edit_file(absolute_path: str, content: str) -> str:
    """Edit a file. absolute_path must be a full absolute path (e.g., /home/user/project/file.py)."""
```

**ACI checklist**:
- [ ] Put yourself in the model's shoes — is it obvious how to use this tool?
- [ ] Include example usage, edge cases, and format requirements in descriptions
- [ ] Change parameter names to make mistakes harder (poka-yoke)
- [ ] Use absolute paths, explicit formats, and constrained types over free-form inputs
- [ ] Test with many examples — watch what mistakes the model makes, then fix the tool

See the [MCP page](../mcp/README.md) for tool design in depth.

### Supervisor / Worker Architecture

The supervisor holds the task decomposition and coordination logic. Workers are stateless executors that receive only the context slice they need.

```python
# Pattern: supervisor with isolated workers
async def supervisor_workflow(task: str, workers: dict) -> str:
    """
    Supervisor breaks down task, dispatches to specialist workers,
    synthesizes results. Workers never see each other's context.
    """
    # Step 1: Supervisor decomposes
    subtasks = await supervisor_agent.plan(task)

    # Step 2: Dispatch in parallel — each worker gets ONLY its slice
    results = await asyncio.gather(*[
        worker_agent.execute(
            task=subtask,
            context=subtask.relevant_context  # NOT the full parent context
        )
        for subtask in subtasks
    ])

    # Step 3: Supervisor synthesizes
    return await supervisor_agent.synthesize(task, results)


# Context isolation is critical — passing full context to every worker:
#   ✗ Wastes tokens (workers see irrelevant history)
#   ✗ Increases error rate (noise → confused worker)
#   ✗ Breaks privacy (worker A sees worker B's results before synthesis)
```

### Pipeline vs Barrier Synchronization

**Pipeline** (default): Output of stage A flows directly into stage B without waiting for other items. Item 1 can be in stage 3 while Item 2 is in stage 1.

**Barrier**: All items must complete stage N before any item starts stage N+1.

```python
# ✅ Pipeline — correct for most tasks
# Item A flows through all stages independently of Item B
# Wall-clock = slowest single item, not sum-of-slowest-per-stage

async def pipeline(items, *stages):
    """Each item progresses through stages independently."""
    tasks = [run_item_through_stages(item, stages) for item in items]
    return await asyncio.gather(*tasks)

async def run_item_through_stages(item, stages):
    result = item
    for stage in stages:
        result = await stage(result)
    return result


# ✅ Barrier — only when synthesis needs ALL prior results
# Use when: dedup across full result set, cross-item comparison, voting

async def barrier_workflow(items, find_fn, verify_fn):
    # All finders run in parallel (independent)
    all_findings = await asyncio.gather(*[find_fn(item) for item in items])

    # Barrier: dedup across ALL findings before verification
    flat = [f for findings in all_findings for f in findings]
    unique = dedup_by_key(flat, key="id")           # genuinely needs ALL

    # Now verify each unique finding (pipeline again)
    verified = await asyncio.gather(*[verify_fn(f) for f in unique])
    return [v for v in verified if v.is_real]
```

**Decision rule**: if your code would write `await parallel(all_items)` then immediately `for result in results:` — that's a pipeline, not a barrier. Rewrite as pipeline.

### Adversarial Verification

For high-stakes agent outputs, spawn N independent agents to try to REFUTE each finding:

```python
async def adversarial_verify(finding: str, n_voters: int = 3) -> bool:
    """
    Spawn N independent skeptic agents.
    Claim survives only if majority CANNOT refute it.
    """
    votes = await asyncio.gather(*[
        skeptic_agent(
            f"Try hard to refute this claim. "
            f"Default refuted=True if uncertain.\n"
            f"Claim: {finding}"
        )
        for _ in range(n_voters)
    ])
    refuted = sum(1 for v in votes if v.refuted)
    return refuted < n_voters // 2 + 1  # majority must fail to refute


# Perspective-diverse verification (stronger than identical skeptics):
LENSES = ["correctness", "security", "reproducibility"]

async def diverse_verify(finding: str) -> dict:
    """Each verifier uses a different failure lens."""
    verdicts = await asyncio.gather(*[
        verifier_agent(f"Verify via {lens} lens: {finding}")
        for lens in LENSES
    ])
    return {
        "finding": finding,
        "passes": sum(1 for v in verdicts if v.passes),
        "verdicts": verdicts
    }
```

### Swarm Coordination

Agents operate peer-to-peer, picking tasks from a shared queue — no central coordinator. Scales horizontally for embarrassingly parallel workloads.

```python
import asyncio
from asyncio import Queue

async def swarm_worker(agent_id: int, task_queue: Queue, results: list):
    """Worker pulls tasks until queue is empty."""
    while True:
        try:
            task = task_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        result = await agent.execute(task)
        results.append(result)
        task_queue.task_done()

async def swarm(tasks: list, n_workers: int = 5) -> list:
    queue = Queue()
    for task in tasks:
        queue.put_nowait(task)

    results = []
    workers = [
        asyncio.create_task(swarm_worker(i, queue, results))
        for i in range(n_workers)
    ]
    await asyncio.gather(*workers)
    return results
```

**Use supervisor/worker when**: task decomposition is complex, subtask dependencies exist, or you need a synthesis step.
**Use swarm when**: tasks are independent, uniform, and embarrassingly parallel (e.g., analyze 1,000 documents).

---

## 📁 Project Structure

```
appendix/agent-frameworks/
├── README.md
├── requirements.txt
├── agentic_workflows.ipynb       # ★ The loop by hand, then as a LangGraph graph; routing; HITL
├── configs/
│   └── agent_configs.yaml
├── examples/
│   ├── multi_agent_workflow.py   # Multi-agent LangGraph demo
│   └── human_in_loop.py          # HITL workflow
└── skills/
    └── skill_library.py          # Search, code, analysis, knowledge skills
```

## 🛠️ Required Dependencies

```bash
pip install -r requirements.txt          # LangGraph 1.x, for the notebook and examples
pip install "openai-agents>=0.22"        # optional: the OpenAI Agents SDK snippet
pip install "google-adk>=2.0"            # optional: the Google ADK 2.0 snippet
```

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Agent loops infinitely | Set a step limit: `recursion_limit` (LangGraph), `max_turns` (Agents SDK), `max_llm_calls` (ADK) |
| Agent calls wrong tool | Improve tool descriptions; add "Do NOT use for..." disclaimers |
| LangGraph state not persisting, or `interrupt()` fails | Compile the graph with a checkpointer and pass a `thread_id` in the config |
| Multi-agent context pollution | Isolate worker context; don't pass full parent context |

## 🧪 Hands-On Exercises

1. **Same Case, three ways**: Take the Spine 1 Case (Free → Pro) and run it through your hand-written loop, a LangGraph graph and one of the two SDKs. Point the Case runner at each (`run_case(case, your_agent)`). Do all three pass the Spine 1 Check? Which one made it hardest to see what the model was sent?

2. **Move the limit**: In each framework, set the step limit to 2 and give the agent a Case that needs 3 tool calls. What does each one do when it runs out — raise, return partial output, or answer anyway?

3. **Guardrail vs Action**: Put the "own account only" rule in an OpenAI Agents SDK tool guardrail instead of inside the tool. Then write a customer message that gets past the guardrail. Why does the Spine keep the rule inside the Action?

4. **LangGraph State Machine**: Build a LangGraph workflow with at least 3 nodes and 2 conditional edges. Add a loop that retries up to 3 times before giving up.

5. **Human-in-the-Loop**: In the notebook's review graph, resume with `Command(resume="no")`, then route a rejected draft back to the drafting node with the reviewer's feedback.

## 📚 Resources

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graphs, state, checkpointers, interrupts
- [LangGraph changelog](https://github.com/langchain-ai/langgraph/releases)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — agents, handoffs, guardrails, sessions, tracing
- [OpenAI Agents SDK: human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- [Google ADK](https://google.github.io/adk-docs/) and [Welcome to ADK 2.0](https://adk.dev/2.0/) — graph-based, dynamic and collaborative workflows
- [ADK changelog](https://github.com/google/adk-python/blob/main/CHANGELOG.md)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — ACI principles
