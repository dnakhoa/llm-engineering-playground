# Agentic Development with LangChain & LangGraph

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

## 🧪 Hands-On Exercises

1. **Single Agent with Tools**: Build a single agent with 3 tools (calculator, web search, text summarizer). Give it a complex query that requires using all 3 tools. How does it decide which tool to use?

2. **Multi-Agent Debate**: Create two agents with opposing viewpoints. Have them debate a topic for 5 rounds. Who "wins"? How would you evaluate the quality of the debate?

3. **LangGraph State Machine**: Build a LangGraph workflow with at least 3 nodes and 2 conditional edges. Add a loop that retries up to 3 times before giving up.

4. **Router Improvement**: Replace the keyword-based router in `multi_agent_workflow.py` with an LLM-based classifier. Compare routing accuracy on 20 test queries.

5. **Human-in-the-Loop**: Add a checkpoint before the "review" step that requires human approval. What happens if the human rejects? What if they suggest changes?

---

**Ready to build?** Start with `examples/multi_agent_workflow.py` and work up to complex multi-agent systems!
