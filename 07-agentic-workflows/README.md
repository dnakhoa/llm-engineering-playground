# Agentic Development with LangChain & LangGraph


> **Why this matters:** Agents are the future of LLM applications — systems that reason, plan, use tools, and recover from errors. Getting agent architecture right determines whether your system is reliable or a liability.


## 🎯 Learning Objectives
- Master the agentic development workflow
- Build multi-agent systems with specialized roles
- Implement dynamic routing and orchestration
- Create reusable skills and tool libraries
- Structure knowledge bases for agents

## 📚 Core Concepts

### 1. Agent Architecture Patterns

#### Single Agent vs Multi-Agent Systems
```
Single Agent:           Multi-Agent (Specialized):
┌─────────────┐         ┌──────────┐  ┌──────────┐  ┌──────────┐
│   General   │         │ Research │  │  Coder   │  │ Reviewer │
│    Agent    │   OR    │  Agent   │→ │  Agent   │→ │  Agent   │
└─────────────┘         └──────────┘  └──────────┘  └──────────┘
                              ↓           ↓           ↓
                         ┌──────────────────────────────┐
                         │      Orchestrator/Router     │
                         └──────────────────────────────┘
```

### 2. Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Skills** | Reusable capabilities | Python functions + Tool decorators |
| **Tools** | External integrations | LangChain Tools, API wrappers |
| **Knowledge** | Domain context | Vector stores, Document loaders |
| **Router** | Task distribution | LLM-based classification, State machines |
| **Specialist Agents** | Domain experts | Role-prompted agents with specific tools |
| **Orchestrator** | Coordination | LangGraph StateGraph, Supervisor |

### 3. LangGraph Fundamentals

LangGraph extends LangChain with:
- **State Management**: Typed state objects
- **Graph Flows**: Nodes (functions) and Edges (transitions)
- **Cycles**: Loops for iterative refinement
- **Human-in-the-loop**: Checkpoints for approval

```python
# Basic Graph Structure
from langgraph.graph import StateGraph, END

workflow = StateGraph(StateSchema)
workflow.add_node("researcher", research_node)
workflow.add_node("coder", code_node)
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", END)
```

## 🏗️ Development Workflow

### Phase 1: Define Skills & Tools
```python
# skills/search_skills.py
from langchain.tools import tool
from typing import List, Dict

@tool
def search_academic_papers(query: str, year_range: tuple = None) -> str:
    """Search academic papers with advanced filtering."""
    # Implementation
    pass

@tool
def execute_code(code: str, language: str = "python") -> str:
    """Safely execute code in sandboxed environment."""
    # Implementation
    pass
```

### Phase 2: Build Knowledge Base
```python
# knowledge/loader.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_domain_knowledge(domain: str) -> Chroma:
    """Load and index domain-specific documents."""
    documents = load_documents(f"data/{domain}/")
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(documents, embeddings)
```

### Phase 3: Design Specialist Agents
```python
# agents/specialists.py
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

def create_research_agent(llm, tools, knowledge_base):
    """Create a specialized research agent."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Research Analyst.
        Expertise: Academic literature review, data synthesis.
        Tools: {tools}
        Knowledge: Use the provided vector store for domain context.
        Style: Cite sources, be precise, flag uncertainties."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    return create_tool_calling_agent(llm, tools, prompt)
```

### Phase 4: Implement Router
```python
# router/dynamic_router.py
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    target_agent: str = Field(description="research, coding, review, or general")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Why this route was chosen")

def create_router(llm):
    """LLM-based dynamic router."""
    parser = PydanticOutputParser(pydantic_object=RouteDecision)
    # Implementation with few-shot examples
```

### Phase 5: Orchestrate with LangGraph
```python
# orchestrator/workflow.py
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_step: str
    results: dict
    history: List[dict]

def build_multi_agent_graph():
    graph = StateGraph(AgentState)
    
    # Add specialist nodes
    graph.add_node("router", router_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    
    # Define edges with conditional routing
    graph.add_conditional_edges(
        "router",
        route_based_on_decision,
        {
            "research": "researcher",
            "coding": "coder",
            "review": "reviewer",
            "general": "general_agent"
        }
    )
    
    # Add loops for refinement
    graph.add_edge("reviewer", "router")  # Can route back for revisions
    
    return graph.compile()
```

## 📁 Project Structure

```
agentic-workflows/
├── skills/
│   ├── __init__.py
│   └── skill_library.py         # Search, code, analysis, knowledge skills
├── tools/
│   └── __init__.py
├── knowledge/
│   └── __init__.py
├── agents/
│   └── __init__.py
├── router/
│   └── __init__.py
├── orchestrator/
│   └── __init__.py
├── examples/
│   ├── __init__.py
│   ├── multi_agent_workflow.py   # ★ Multi-agent LangGraph demo
│   └── human_in_loop.py          # ★ HITL workflow
└── agentic_workflows.ipynb       # ★ Interactive notebook
```

## 🔧 Best Practices

### 1. Skill Design
- **Atomic**: Each skill does one thing well
- **Composable**: Skills can be combined
- **Observable**: Log inputs/outputs for debugging
- **Safe**: Validate inputs, handle errors gracefully

### 2. Agent Specialization
- **Clear Roles**: Define expertise boundaries
- **Tool Scoping**: Give agents only needed tools
- **Context Limits**: Provide relevant knowledge only
- **Prompt Templates**: Consistent role definitions

### 3. Routing Strategies
- **Hierarchical**: High-level → Specific routing
- **Confidence-based**: Fallback on low confidence
- **Learning**: Improve routes from feedback
- **Multi-criteria**: Consider cost, latency, expertise

### 4. State Management
- **Immutable History**: Keep conversation history
- **Structured Results**: Typed output schemas
- **Checkpointing**: Save state for recovery
- **Metadata**: Track token usage, timing, costs

## 🚀 Advanced Patterns

### 1. Hierarchical Teams
```
CEO/Supervisor
    ↓
Project Manager (routes to)
    ├─ Research Team Lead → Researchers
    ├─ Engineering Lead → Coders
    └─ QA Lead → Reviewers
```

### 2. Competitive Collaboration
- Multiple agents solve same problem
- Voting mechanism selects best answer
- Useful for critical decisions

### 3. Iterative Refinement
```
Generate → Critique → Revise → Validate → Deploy
    ↑                                    │
    └──────────── Loop ──────────────────┘
```

### 4. Human-in-the-Loop
- Critical steps require approval
- Agents propose, humans decide
- Audit trail for compliance

---

## 🚀 Advanced Multi-Agent Patterns (2025-2026)

### Agent SDKs — Production-Ready Agent Frameworks

Major providers now offer dedicated agent SDKs that handle the boilerplate of tool calling, state management, and orchestration:

| SDK | Provider | Key Features |
|-----|----------|-------------|
| **OpenAI Agents SDK** | OpenAI | Multi-agent orchestration, handoffs, guardrails, sandboxing |
| **Anthropic Agent SDK** | Anthropic | Claude-native tool use, MCP integration, human-in-the-loop |
| **LangGraph** | LangChain | State machine graphs, checkpointing, streaming |
| **Strands Agents SDK** | AWS | Bedrock-native, enterprise integration |

**When to use SDKs vs raw API**:
- SDKs reduce boilerplate but add abstraction layers that can obscure prompts/responses
- Start with raw API to understand the mechanics, then adopt SDKs for production
- Always verify what's happening under the hood — incorrect assumptions are a common source of errors

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

## 📖 Next Steps

1. **Start Simple**: Build a single agent with 2-3 tools
2. **Add Specialization**: Create 2-3 specialist agents
3. **Implement Router**: Add dynamic routing logic
4. **Build Graph**: Orchestrate with LangGraph
5. **Add HITL**: Include human approval steps
6. **Optimize**: Monitor, measure, improve

## 🛠️ Required Dependencies

```bash
pip install langchain langchain-core langchain-community
pip install langgraph
pip install chromadb sentence-transformers
pip install pydantic pyyaml
```

## 📝 Example Use Cases

1. **Research Assistant**: Search → Synthesize → Cite
2. **Code Generation**: Plan → Code → Test → Review
3. **Customer Support**: Triage → Solve → Escalate
4. **Data Analysis**: Query → Analyze → Visualize → Report
5. **Content Creation**: Research → Draft → Edit → Publish

---



## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Agent loops infinitely | Add max iterations limit and novelty gate |
| Agent calls wrong tool | Improve tool descriptions; add "Do NOT use for..." disclaimers |
| LangGraph state not persisting | Add checkpointer to graph compilation |
| Multi-agent context pollution | Isolate worker context; don't pass full parent context |

## 📚 Resources

- [LangGraph](https://langchain-ai.github.io/langgraph/) — state machine agents
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — OpenAI's agent framework
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — ACI principles

## 🧪 Hands-On Exercises

1. **Single Agent with Tools**: Build a single agent with 3 tools (calculator, web search, text summarizer). Give it a complex query that requires using all 3 tools. How does it decide which tool to use?

2. **Multi-Agent Debate**: Create two agents with opposing viewpoints. Have them debate a topic for 5 rounds. Who "wins"? How would you evaluate the quality of the debate?

3. **LangGraph State Machine**: Build a LangGraph workflow with at least 3 nodes and 2 conditional edges. Add a loop that retries up to 3 times before giving up.

4. **Router Improvement**: Replace the keyword-based router in `multi_agent_workflow.py` with an LLM-based classifier. Compare routing accuracy on 20 test queries.

5. **Human-in-the-Loop**: Add a checkpoint before the "review" step that requires human approval. What happens if the human rejects? What if they suggest changes?

---

**Ready to build?** Start with `examples/multi_agent_workflow.py` and work up to complex multi-agent systems!
