# Module 16: Graph Engineering

> **Why this matters:** A loop says "keep going until done." A graph says "here is exactly what can happen next." Once an agent system has branches, retries, human approvals, and parallel work, the graph *is* the system — and the same idea shows up again in the data layer, where knowledge graphs answer the multi-hop questions vector search can't.

**Difficulty:** ⭐⭐⭐ Advanced · **Time:** ~3h · **Prerequisites:** Modules 02 (RAG), 07 (Agents), 13 (Harness)

## Learning Objectives

- Model an agent system as a typed state graph: nodes, edges, conditional edges, cycles
- Choose between a loop and a graph — and know why production systems nest them
- Design graph state with reducers so parallel branches merge instead of clobbering
- Use checkpointers for durability, resume, time-travel debugging, and human-in-the-loop interrupts
- Fan out dynamically (map-reduce over an unknown number of items) and fan back in
- Express supervisor and swarm topologies as graphs, and know when to stop drawing
- Build a knowledge graph from unstructured text: entity and relation extraction, resolution
- Implement Graph RAG (local, global, and hybrid retrieval) and measure whether it beats vector RAG
- Model change over time with a temporal knowledge graph instead of overwriting facts
- Evaluate and debug at node level; recognize the standard graph failure modes

---

## 📚 Part 1 — Graphs as Control Flow

### The core abstraction

Three pieces, and nothing else:

| Piece | What it is | Practical form |
|-------|-----------|----------------|
| **State** | The typed object every node reads and writes | A `TypedDict` / dataclass with a reducer per field |
| **Node** | A function `state -> partial state update` | An LLM call, a tool call, plain Python |
| **Edge** | What runs next | Static (`a → b`) or conditional (`route(state) -> "b" | "c" | END`) |

Execution proceeds in **supersteps**: all nodes scheduled for this step run (potentially in parallel), their updates are merged into state via reducers, then the next step is scheduled. This is why reducers matter — see below.

```
        ┌─────────┐
        │  START  │
        └────┬────┘
             ▼
        ┌─────────┐        ┌──────────┐
        │ retrieve│───────▶│  grade   │
        └─────────┘        └────┬─────┘
                                │ conditional edge
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
        ┌──────────┐      ┌──────────┐       ┌─────────┐
        │ rewrite  │      │ generate │       │  END    │
        └────┬─────┘      └────┬─────┘       └─────────┘
             │                 │
             └───── cycle ─────┘  (rewrite → retrieve, bounded by state["attempts"])
```

That cycle is the point. A DAG can't express "retrieve, judge, and if the evidence is weak, rewrite the query and try again" — a graph with a conditional edge can, and unlike a bare `while` loop the topology is inspectable, replayable, and testable node by node.

### Minimal graph (LangGraph)

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    docs: Annotated[list[str], add]   # reducer: branches APPEND, not overwrite
    answer: str
    attempts: int


def retrieve(state: State) -> dict:
    return {"docs": search(state["question"]), "attempts": state["attempts"] + 1}


def grade(state: State) -> dict:
    return {"relevant": judge_relevance(state["question"], state["docs"])}


def rewrite(state: State) -> dict:
    return {"question": rewrite_query(state["question"])}


def generate(state: State) -> dict:
    return {"answer": answer_from(state["question"], state["docs"])}


def route(state: State) -> str:
    """The stop condition lives here — in code, not in the model's judgement."""
    if state.get("relevant"):
        return "generate"
    if state["attempts"] >= 3:        # mechanical stop: never trust the semantic one alone
        return "generate"
    return "rewrite"


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route, ["rewrite", "generate"])
builder.add_edge("rewrite", "retrieve")   # the cycle
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"question": "...", "docs": [], "answer": "", "attempts": 0})
```

> **Read `route` again.** Every conditional edge is a stop condition or a branch policy, written in ordinary code you can unit-test without an API key. Moving decisions out of prose and into `route` functions is most of what "graph engineering" buys you.

### State design: reducers are not optional

The single most common graph bug: two parallel branches both return `{"docs": [...]}`, and one silently overwrites the other.

| Field kind | Reducer | Why |
|-----------|---------|-----|
| Accumulating list (docs, findings, messages) | `add` / append | Parallel branches must merge |
| Scalar you want *last write wins* | none (default overwrite) | Only safe if exactly one writer per superstep |
| Counter | custom `lambda a, b: a + b` | Two increments in one step must both land |
| Dict of per-item results | custom merge keyed by item id | Avoids order dependence |

Design rules:
1. **One writer per scalar field per superstep.** If two nodes can write the same scalar in the same step, you need a reducer or a different field.
2. **Keep state small and typed.** State is passed to every node and checkpointed on every transition — a fat state object is a cost and a debugging problem. Put bulk data behind a reference (file path, doc id) and keep pointers in state.
3. **Separate durable facts from scratch.** Confirmed findings persist; the current draft doesn't need to.

### Dynamic fan-out (map-reduce)

When you don't know at build time how many branches you need, generate them at runtime:

```python
from langgraph.types import Send

def fan_out(state: State) -> list[Send]:
    """One 'analyze' node instance per discovered file — decided at runtime."""
    return [Send("analyze", {"file": f}) for f in state["files"]]

builder.add_conditional_edges("discover", fan_out, ["analyze"])
builder.add_edge("analyze", "synthesize")   # implicit fan-in
```

Each `analyze` instance returns a partial update; the reducer on that field merges them; `synthesize` runs once, after all of them. Two rules from Module 13 carry over unchanged: cap the fan-out width, and **log what you dropped** if you truncate to top-N. A silent cap reads as "we covered everything."

### Checkpointers: durability, resume, time travel, interrupts

A checkpointer persists state after every superstep. That one mechanism gives you four things:

| Capability | How it works |
|-----------|--------------|
| **Crash resume** | Re-invoke with the same `thread_id`; execution continues from the last checkpoint |
| **Human-in-the-loop** | `interrupt_before=["apply_changes"]`, inspect state, then resume — the pause is durable, so it can last days |
| **Time travel** | Load an old checkpoint, edit state, re-run from there — the debugger for agent systems |
| **Multi-turn memory** | The thread's state *is* the conversation state |

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = builder.compile(checkpointer=saver, interrupt_before=["apply_changes"])
    cfg = {"configurable": {"thread_id": "run-42"}}

    graph.invoke(initial_state, cfg)      # runs until the interrupt, then returns

    snapshot = graph.get_state(cfg)       # inspect what it wants to do
    print(snapshot.next, snapshot.values["proposed_diff"])

    graph.update_state(cfg, {"approved": True})
    graph.invoke(None, cfg)               # resume from the checkpoint
```

This is the productized version of Module 13's durable journal. The journal teaches you the mechanics; a checkpointer gives you node-level granularity, state snapshots, and replay for free. Use a journal for scripts; use a checkpointer once you have branches or human gates.

> ⚠️ **Checkpoint everything, but keep secrets out of state.** Checkpointed state is written to durable storage and replayed into future runs — a token in state is a token in your database and in every later trace.

### Agent topologies as graphs

Module 07's multi-agent patterns are graph shapes:

| Topology | Graph form | Trade-off |
|----------|-----------|-----------|
| **Sequential pipeline** | Static chain | Predictable, no adaptivity |
| **Router** | One conditional edge from START | Cheap specialization |
| **Supervisor / workers** | Hub node with conditional edges to workers, workers edge back | Central control, hub is a bottleneck and a context sink |
| **Swarm / handoff** | Peer nodes each able to route to any other | Flexible, harder to bound and audit |
| **Hierarchical teams** | Subgraph per team, each with its own supervisor | Scales; watch state translation at boundaries |

Two things that go wrong in practice: **context leakage** (the hub forwards everything to every worker — isolate worker inputs, see Module 12) and **handoff ping-pong** (two peers route to each other forever — count handoffs in state and force a decision at the cap).

### When *not* to reach for a graph

- **Two or three fixed steps** — a function that calls three functions is clearer than a graph.
- **A single unbounded loop** — a loop with a novelty gate (Module 13) is the honest expression; wrapping it in a one-node graph adds ceremony.
- **You're drawing the graph to look sophisticated.** Nodes are surface area: each one is a state contract, a failure point, and a checkpoint. Ten nodes that could be three is a maintenance cost with no payoff.

Reach for a graph when you have **branching + cycles + durability or approval requirements**. That combination is where hand-rolled control flow stops being reviewable.

---

## 🕸️ Part 2 — Knowledge Graphs & Graph RAG

The same structure, one layer down. Vector RAG retrieves *chunks that look like the question*. Some questions aren't answerable that way:

| Question | Vector RAG | Why |
|----------|-----------|-----|
| "What did the Q3 report say about churn?" | ✅ Works | The answer sits in one chunk |
| "Which of our suppliers are also customers of our competitors?" | ❌ Fails | Requires joining facts across documents |
| "How did the ownership chain of Acme change from 2019 to 2024?" | ❌ Fails | Requires traversal and time |
| "What are the main themes across all 4,000 support tickets?" | ❌ Fails | Requires whole-corpus aggregation, not top-k |

A **knowledge graph** stores entities as nodes and relationships as typed edges, so those become traversals and aggregations rather than similarity searches.

### Pipeline: text → graph

```
documents ──▶ chunk ──▶ extract (entities + relations) ──▶ resolve ──▶ store ──▶ community summaries
                            │                                 │
                        LLM w/ strict schema          dedupe & merge aliases
```

**1. Extraction** — an LLM with a strict schema and a closed entity-type list. Open-ended extraction produces an unusable graph where every document invents new relation names.

```python
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string",
                         "enum": ["PERSON", "COMPANY", "PRODUCT", "LOCATION", "EVENT"]},
                "description": {"type": "string"},
            },
            "required": ["name", "type"], "additionalProperties": False}},
        "relations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "type": {"type": "string",
                         "enum": ["WORKS_AT", "ACQUIRED", "SUPPLIES", "COMPETES_WITH",
                                  "LOCATED_IN", "PARTNERED_WITH"]},
                "evidence": {"type": "string"},   # the sentence it came from
            },
            "required": ["source", "target", "type", "evidence"],
            "additionalProperties": False}},
    },
    "required": ["entities", "relations"], "additionalProperties": False,
}
```

**Always extract `evidence`.** A graph edge without a source sentence is an unfalsifiable claim — you can't audit it, and you can't cite it in an answer.

**2. Entity resolution** — the step that decides whether your graph is useful. "Acme", "Acme Corp.", "ACME Corporation" and "the company" must collapse to one node, or every traversal dead-ends.

| Strategy | Cost | Use |
|----------|------|-----|
| Exact + normalized string match | Free | First pass, always |
| Alias table / known-entity list | Cheap | Domains with a canonical registry (tickers, SKUs, employees) |
| Embedding similarity above a threshold | Moderate | Long tail of name variants |
| LLM adjudication on near-duplicates | Expensive | Only the ambiguous pairs the cheaper passes flag |

Cascade them cheapest-first and log every merge — bad merges are much harder to detect later than missed ones.

**3. Storage** — a property graph (Neo4j, Kùzu, Memgraph) if you need Cypher-style traversal at scale; plain `networkx` or SQLite if the graph fits in memory. Start in memory. Most course-sized and many production graphs do.

### Retrieval modes

| Mode | Question shape | Mechanism |
|------|---------------|-----------|
| **Local** | "Tell me about X" | Anchor on entities in the query, expand 1–2 hops, return the subgraph + source chunks |
| **Global** | "What are the main themes?" | Pre-computed community summaries; map over communities, then reduce |
| **Path** | "How is X connected to Y?" | Shortest / all paths between anchors, returned as an explained chain |
| **Hybrid** | Most real traffic | Vector search for passages + graph expansion for structure, then re-rank the union |

**Global search** is what makes Graph RAG distinctive. Cluster the graph into communities (e.g. Leiden), summarize each community once at index time, then answer corpus-level questions by mapping over summaries instead of retrieving top-k chunks. The cost lives at index time, which is exactly the trade you want for a corpus queried many times.

```python
# Hybrid retrieval — the default production shape
def hybrid_retrieve(query: str, k: int = 8) -> list[str]:
    passages = vector_store.search(query, k=k)              # what looks like the question
    anchors  = extract_entities(query)                      # what the question is about
    subgraph = graph.expand(anchors, hops=2, limit=40)      # structure around it

    context = [p.text for p in passages]
    context += [f"{s} —[{t}]→ {o}   (source: {ev})"         # edges carry their evidence
                for s, t, o, ev in subgraph.edges_with_evidence()]
    return rerank(query, dedupe(context))[:k]
```

### Temporal knowledge graphs

Facts expire. "Alice works at Acme" was true in 2023 and false in 2025. Overwriting it loses history; keeping both without time bounds makes the graph self-contradictory. The fix is to make validity a property of the edge:

```python
edge = {
    "source": "Alice", "type": "WORKS_AT", "target": "Acme",
    "valid_from": "2023-02-01",
    "valid_to": "2025-06-30",     # None = still valid
    "invalidated_by": "doc_8891", # what superseded it
    "evidence": "Alice joined Acme as VP Eng in February 2023.",
}
```

Then retrieval takes an as-of time, and a new fact **invalidates** rather than overwrites the old one. This is what makes an agent's memory (Module 11) auditable: you can ask not just "what do we believe?" but "what did we believe last March, and what changed it?"

### Does Graph RAG actually beat vector RAG?

Sometimes. It is strictly more expensive, so measure before adopting.

| Dimension | Vector RAG | Graph RAG |
|-----------|-----------|-----------|
| Index cost | Low (embed once) | **High** — extraction is an LLM call per chunk, plus community summaries |
| Query cost | Low | Moderate–high (traversal + more context) |
| Multi-hop questions | Weak | **Strong** |
| Corpus-level questions | Very weak | **Strong** (global search) |
| Freshness / incremental update | Easy | Harder — new facts touch resolution and communities |
| Explainability | Chunk citations | **Explicit reasoning chains** with per-edge evidence |
| Failure mode | Missing context | **Wrong context stated confidently** (bad edge) |

**Evaluate it like any other retrieval change (Module 04/09):** build a question set that includes single-hop, multi-hop, and corpus-level questions; measure recall and faithfulness for vector-only, graph-only, and hybrid. If multi-hop and corpus-level questions are <10% of your traffic, hybrid-on-demand — graph expansion only when the query names multiple entities — usually beats graph-everywhere.

> ⚠️ **The dangerous failure mode is a confidently wrong edge.** A missing chunk produces "I don't know"; a hallucinated `ACQUIRED` edge produces a fluent, cited, false answer. This is why every edge carries `evidence` and why extraction runs against a closed relation vocabulary.

---

## 🏗️ Project Structure

```
16-graph-engineering/
├── README.md
├── requirements.txt
├── graph_engineering.ipynb          ★ Interactive notebook
└── examples/
    ├── __init__.py
    ├── state_graph.py               Conditional edges, cycles, reducers, fan-out (no deps)
    └── knowledge_graph.py           Text → graph → local/global/path retrieval (no deps)
```

Both examples run with **no API key and no graph database** — they use a small in-memory graph and a scripted extractor so you can study the mechanics, then swap in a real LLM and store.

```bash
cd 16-graph-engineering
python examples/state_graph.py
python examples/knowledge_graph.py
```

---

## ☠️ Graph Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Lost update** | Parallel branch results vanish | Add a reducer; one writer per scalar per superstep |
| **Unbounded cycle** | Graph never reaches END | Put a counter in state and check it in the `route` function |
| **State bloat** | Slow checkpoints, huge traces | Store references, not payloads; separate scratch from durable state |
| **Node soup** | 15 nodes, unclear ownership | Collapse nodes with no branch or checkpoint value |
| **Handoff ping-pong** | Two agents route to each other forever | Count handoffs; force a decision at the cap |
| **Context leakage** | Every worker sees everything | Pass scoped inputs per worker (Module 12) |
| **Interrupt deadlock** | Run waits forever | Timeout on pending approvals; surface them in a queue |
| **Non-deterministic replay** | Time travel produces a different path | No wall-clock/random inside nodes — pass them in as state |
| **Unresolved entities** | Traversals dead-end; duplicate nodes | Cascade resolution cheapest-first; log every merge |
| **Relation-name explosion** | Hundreds of near-synonym edge types | Closed enum in the extraction schema |
| **Unfalsifiable edges** | Wrong answers you can't trace | Require `evidence` on every edge; drop edges without it |
| **Stale facts** | Confidently outdated answers | Temporal edges with `valid_from`/`valid_to`, retrieval as-of a time |

---

## 🧪 Hands-On Exercises

1. **Reducer Bug**: Build a 3-node graph where two parallel nodes both write `docs`. Run it without a reducer and count how many docs survive. Add `Annotated[list, add]` and re-run. Explain the superstep semantics you just observed.

2. **Bounded Cycle**: Implement the retrieve → grade → rewrite cycle above with a grader that *never* returns relevant. Verify it terminates at exactly 3 attempts. Then remove the attempts check and observe what a missing mechanical stop condition costs you.

3. **Time Travel Debug**: Run a 5-node graph with a checkpointer. Load the checkpoint before node 4, mutate one state field, resume, and confirm you got a different outcome from the same prefix. This is the agent debugger — use it on your own graph.

4. **Human Interrupt**: Add `interrupt_before` on a node that writes files. Approve one run, reject another with a reason, and verify the rejection reaches the model as feedback rather than silently ending the run.

5. **Dynamic Fan-Out**: Use `Send` to spawn one node per item over a list of 20 items, with a hard cap of 8 concurrent. Log which items were processed and which were dropped by the cap — then check whether your final report *says* it was capped.

6. **Loop vs Graph**: Take exercise 2 from Module 13 (journal resume) and re-implement it as a graph with a checkpointer. Compare lines of code, what each recovers after a crash, and what each lets you inspect afterwards.

7. **Extraction Schema**: Extract entities and relations from 5 news paragraphs, once with an open-ended prompt and once with the closed enum schema above. Count distinct relation types produced by each. Which graph could you actually query?

8. **Entity Resolution Cascade**: Take 40 entity mentions with deliberate variants ("Acme", "ACME Corp", "Acme Corporation", "acme"). Resolve with (a) exact match, (b) normalized match, (c) embedding threshold. Report precision and recall of merges at each stage, and find one *wrong* merge you'd have to catch by hand.

9. **Multi-Hop Showdown**: Write 15 questions — 5 single-hop, 5 multi-hop, 5 corpus-level. Answer all 15 with vector RAG, graph RAG, and hybrid. Report accuracy per category and total cost. Decide which mode you'd actually ship, and say why.

10. **Global Search**: Cluster a 200-document graph into communities, summarize each, and answer "what are the main themes?" from the summaries. Compare to a top-k vector answer for the same question. Note the index-time cost you paid.

11. **Temporal Facts**: Model one entity whose employer changes twice. Answer "where does X work?" as-of three different dates. Then break it: overwrite instead of invalidating, and show which query now returns a wrong answer.

12. **Failure Injection**: Insert one deliberately false edge into a 50-edge graph. Ask a question whose answer traverses it. Show the confident wrong answer, then show how per-edge `evidence` makes it detectable.

---

## 🔗 Integration with Other Modules

- **Module 02 (RAG)**: Graph RAG is the advanced retrieval mode — this module is its deep dive
- **Module 04 / 09 (Evaluation, EvalOps)**: Node-level evals; the multi-hop question set is how you justify a graph
- **Module 07 (Agentic Workflows)**: Supervisor, swarm, and hierarchical patterns are graph topologies
- **Module 08 (LLM Ops)**: A trace *is* a graph — align trace spans with node names for free debuggability
- **Module 11 (Memory)**: A temporal knowledge graph is auditable long-term memory
- **Module 12 (Context Engineering)**: Scoped node inputs prevent context leakage between branches
- **Module 13 (Harness & Loops)**: Loops and graphs are complementary — see "Loop or graph?" in Module 13

## 📚 Resources

- [LangGraph: Low-level concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/) — state, nodes, edges, supersteps, reducers
- [LangGraph: Persistence & checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/) — resume, time travel, threads
- [LangGraph: Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — interrupts and approval gates
- [Microsoft Research: GraphRAG](https://microsoft.github.io/graphrag/) — community summaries, local vs global search
- [Neo4j: LLM knowledge graph builder](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/) — extraction and property-graph storage
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — when orchestration beats a single agent
- [Zep/Graphiti: temporal knowledge graphs](https://github.com/getzep/graphiti) — bi-temporal edge validity for agent memory

## 📖 References

- Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (2024)
- Traag et al., "From Louvain to Leiden: guaranteeing well-connected communities" (2019)
- Peng et al., "Graph Retrieval-Augmented Generation: A Survey" (2024)
- Rasmussen et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (2025)

---

**Loops give agents persistence. Graphs give them structure you can review, replay, and trust.**
