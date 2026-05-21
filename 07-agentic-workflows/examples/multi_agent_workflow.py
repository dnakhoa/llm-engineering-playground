"""
Multi-Agent Collaborative Workflow with LangGraph

This example demonstrates a complete multi-agent system with:
- Specialist agents (Researcher, Coder, Reviewer)
- Dynamic routing based on task type
- State management with typed StateGraph
- Iterative refinement loops
- Human-in-the-loop checkpoints

Architecture:
    User Request → Router → [Researcher | Coder | Reviewer] → Output
                          ↑              ↓
                          └── Refinement Loop
"""

from typing import TypedDict, Annotated, List, Literal, Optional, Dict, Any
import operator
from datetime import datetime

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from skills.skill_library import get_skills_by_category


# ============================================================================
# STATE SCHEMA
# ============================================================================

class AgentState(TypedDict):
    """
    Defines the state structure that flows through the graph.
    
    All fields are annotated with how they should be updated:
    - operator.add: Append new items to list (for messages)
    - direct replacement: Overwrite with new value
    """
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    task_type: str  # research, coding, review, general
    results: Dict[str, Any]
    history: List[Dict[str, Any]]
    iteration_count: int
    requires_human_approval: bool
    final_output: Optional[str]


# ============================================================================
# SPECIALIST AGENTS
# ============================================================================

def create_agent_prompt(role: str, expertise: str, tools_info: str) -> SystemMessage:
    """Create a role-specific system prompt for specialist agents."""
    return SystemMessage(content=f"""You are a {role}.

EXPERTISE:
{expertise}

AVAILABLE TOOLS:
{tools_info}

GUIDELINES:
- Be precise and thorough in your domain
- Cite sources when providing information
- Flag uncertainties clearly
- Build upon previous work in the conversation
- Keep responses structured and actionable

OUTPUT FORMAT:
1. Summary of what you're doing
2. Step-by-step execution
3. Results with evidence
4. Recommendations for next steps
""")


class ResearcherAgent:
    """Specialist agent for research and information gathering."""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = get_skills_by_category("search") + get_skills_by_category("knowledge")
        self.prompt = create_agent_prompt(
            role="Senior Research Analyst",
            expertise="- Academic literature review\n- Market research\n- Data synthesis\n- Source verification",
            tools_info="Web search, Academic paper search, Knowledge base queries"
        )
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute research task."""
        print("\n🔍 [RESEARCHER] Starting research task...")
        
        messages = state["messages"]
        if not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content="Please research this topic."))
        
        # Add system prompt
        all_messages = [self.prompt] + messages[-10:]  # Last 10 messages for context
        
        # Execute with tools (mock - in production use agent executor)
        response = self.llm.invoke(all_messages)
        
        # Mock research result
        research_result = {
            "sources_found": 5,
            "key_findings": [
                "Finding 1: Recent advances in the field",
                "Finding 2: Industry best practices",
                "Finding 3: Emerging trends"
            ],
            "confidence": 0.87
        }
        
        print(f"✓ [RESEARCHER] Found {research_result['sources_found']} sources")
        
        return {
            "messages": [AIMessage(content=f"Research complete. Found {research_result['sources_found']} relevant sources.")],
            "current_step": "research_complete",
            "results": {"research": research_result},
            "history": state["history"] + [{"step": "research", "result": research_result}],
            "iteration_count": state["iteration_count"] + 1
        }


class CoderAgent:
    """Specialist agent for code generation and implementation."""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = get_skills_by_category("code") + get_skills_by_category("utility")
        self.prompt = create_agent_prompt(
            role="Senior Software Engineer",
            expertise="- Clean code architecture\n- Test-driven development\n- Performance optimization\n- Security best practices",
            tools_info="Code execution, Syntax validation, Unit test generation, Code search"
        )
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute coding task."""
        print("\n💻 [CODER] Starting code implementation...")
        
        # Mock code generation
        code_result = {
            "language": "python",
            "lines_of_code": 45,
            "functions_created": 3,
            "tests_generated": 5,
            "syntax_valid": True,
            "code_snippet": """
def process_data(data):
    \"\"\"Process input data with validation.\"\"\"
    if not data:
        raise ValueError("Empty data")
    
    result = []
    for item in data:
        if validate_item(item):
            result.append(transform(item))
    
    return result
"""
        }
        
        print(f"✓ [CODER] Generated {code_result['lines_of_code']} lines of code")
        
        return {
            "messages": [AIMessage(content=f"Code generated: {code_result['lines_of_code']} lines, {code_result['tests_generated']} tests.")],
            "current_step": "coding_complete",
            "results": {"code": code_result},
            "history": state["history"] + [{"step": "coding", "result": code_result}],
            "iteration_count": state["iteration_count"] + 1
        }


class ReviewerAgent:
    """Specialist agent for code review and quality assurance."""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = get_skills_by_category("analysis") + get_skills_by_category("code")
        self.prompt = create_agent_prompt(
            role="Senior Code Reviewer",
            expertise="- Code quality assessment\n- Bug detection\n- Security vulnerabilities\n- Performance bottlenecks\n- Best practices compliance",
            tools_info="Sentiment analysis, Code validation, Key phrase extraction"
        )
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute review task."""
        print("\n🔎 [REVIEWER] Starting code review...")
        
        # Get code from previous step
        code_result = state.get("results", {}).get("code", {})
        
        # Mock review
        review_result = {
            "quality_score": 8.5,
            "issues_found": [
                {"severity": "low", "description": "Missing docstring in helper function"},
                {"severity": "medium", "description": "Consider adding type hints"}
            ],
            "suggestions": [
                "Add comprehensive error handling",
                "Include performance benchmarks"
            ],
            "approved": True,
            "requires_changes": False
        }
        
        print(f"✓ [REVIEWER] Quality score: {review_result['quality_score']}/10")
        
        return {
            "messages": [AIMessage(content=f"Review complete. Quality: {review_result['quality_score']}/10. Issues: {len(review_result['issues_found'])}")],
            "current_step": "review_complete",
            "results": {"review": review_result},
            "history": state["history"] + [{"step": "review", "result": review_result}],
            "iteration_count": state["iteration_count"] + 1,
            "requires_human_approval": not review_result["approved"]
        }


# ============================================================================
# ROUTER
# ============================================================================

class DynamicRouter:
    """LLM-based dynamic router that determines which specialist should handle a task."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Analyze task and route to appropriate specialist."""
        print("\n🎯 [ROUTER] Analyzing task...")
        
        # Get the user's request
        messages = state["messages"]
        user_request = messages[-1].content if messages else ""
        
        # Simple keyword-based routing (replace with LLM classification)
        task_type = self.classify_task(user_request)
        
        print(f"✓ [ROUTER] Routed to: {task_type.upper()}")
        
        return {
            "messages": [AIMessage(content=f"Routing task to {task_type} specialist.")],
            "current_step": "routed",
            "task_type": task_type,
            "history": state["history"] + [{"step": "routing", "task_type": task_type}]
        }
    
    def classify_task(self, request: str) -> str:
        """Classify task type based on keywords."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["research", "search", "find", "study", "analyze"]):
            return "research"
        elif any(word in request_lower for word in ["code", "implement", "build", "develop", "function", "script"]):
            return "coding"
        elif any(word in request_lower for word in ["review", "check", "validate", "test", "audit"]):
            return "review"
        else:
            return "general"


def route_based_on_task_type(state: AgentState) -> Literal["researcher", "coder", "reviewer", "general_agent"]:
    """Conditional edge function to route to appropriate node."""
    task_type = state.get("task_type", "general")
    
    routing_map = {
        "research": "researcher",
        "coding": "coder",
        "review": "reviewer"
    }
    
    target = routing_map.get(task_type, "general_agent")
    print(f"→ Routing to: {target}")
    return target


def should_continue(state: AgentState) -> Literal["router", "END"]:
    """Determine if workflow should continue or end."""
    # Check if review is complete and approved
    review_result = state.get("results", {}).get("review", {})
    
    if review_result.get("approved", False) and not review_result.get("requires_changes", True):
        print("✓ Workflow complete - approved")
        return "END"
    
    # Check iteration limit
    if state.get("iteration_count", 0) >= 3:
        print("⚠ Max iterations reached")
        return "END"
    
    # Continue for refinement
    print("↺ Continuing for refinement")
    return "router"


# ============================================================================
# GENERAL AGENT (fallback)
# ============================================================================

class GeneralAgent:
    """General-purpose agent for tasks that don't fit specialist categories."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = SystemMessage(content="""You are a helpful AI assistant.
Provide clear, accurate, and helpful responses.
If a task requires specialized knowledge, indicate that.""")
    
    def __call__(self, state: AgentState) -> AgentState:
        """Handle general tasks."""
        print("\n🤖 [GENERAL] Handling general task...")
        
        messages = state["messages"]
        all_messages = [self.prompt] + messages[-10:]
        
        response = self.llm.invoke(all_messages)
        
        return {
            "messages": [response],
            "current_step": "general_complete",
            "final_output": response.content,
            "history": state["history"] + [{"step": "general", "response": response.content}]
        }


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_multi_agent_graph(llm=None):
    """
    Build the complete multi-agent workflow graph.
    
    Returns:
        Compiled LangGraph workflow
    """
    if llm is None:
        # Use mock LLM for demo (replace with actual LLM)
        from unittest.mock import Mock
        llm = Mock()
        llm.invoke = lambda msgs: AIMessage(content="Mock response")
    
    # Initialize agents
    router = DynamicRouter(llm)
    researcher = ResearcherAgent(llm)
    coder = CoderAgent(llm)
    reviewer = ReviewerAgent(llm)
    general = GeneralAgent(llm)
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("researcher", researcher)
    workflow.add_node("coder", coder)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("general_agent", general)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_based_on_task_type,
        {
            "researcher": "researcher",
            "coder": "coder",
            "reviewer": "reviewer",
            "general_agent": "general_agent"
        }
    )
    
    # Add edges from specialists to reviewer (for coding tasks) or end
    workflow.add_edge("researcher", "general_agent")  # Research → Final output
    workflow.add_edge("coder", "reviewer")  # Code → Review
    workflow.add_edge("general_agent", "END")
    
    # Add conditional edge from reviewer (loop back or end)
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "router": "router",  # Loop back for revisions
            "END": "END"
        }
    )
    
    # Compile with checkpointing for human-in-the-loop
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_example():
    """Demonstrate the multi-agent workflow."""
    print("="*60)
    print("MULTI-AGENT COLLABORATIVE WORKFLOW DEMO")
    print("="*60)
    
    # Build the graph
    app = build_multi_agent_graph()
    
    # Example 1: Coding task
    print("\n" + "="*60)
    print("EXAMPLE 1: Code Generation Task")
    print("="*60)
    
    initial_state = {
        "messages": [HumanMessage(content="Create a Python function to sort a list using merge sort")],
        "current_step": "start",
        "task_type": "",
        "results": {},
        "history": [],
        "iteration_count": 0,
        "requires_human_approval": False,
        "final_output": None
    }
    
    # Run workflow (mock - won't execute without real LLM)
    print("\n⚠️  Note: This is a structural demo. To run with real LLM:")
    print("   1. Set OPENAI_API_KEY environment variable")
    print("   2. Replace mock LLM with ChatOpenAI()")
    print("   3. Uncomment the app.invoke() call below\n")
    
    # result = app.invoke(initial_state, config={"configurable": {"thread_id": "demo_1"}})
    
    # Print expected flow
    print("Expected workflow:")
    print("  1. Router → classifies as 'coding'")
    print("  2. Coder → implements merge sort")
    print("  3. Reviewer → reviews code quality")
    print("  4. END → outputs final code")
    
    # Example 2: Research task
    print("\n" + "="*60)
    print("EXAMPLE 2: Research Task")
    print("="*60)
    
    initial_state["messages"] = [HumanMessage(content="Research recent advances in quantum computing")]
    initial_state["history"] = []
    initial_state["results"] = {}
    initial_state["iteration_count"] = 0
    
    print("\nExpected workflow:")
    print("  1. Router → classifies as 'research'")
    print("  2. Researcher → finds academic papers")
    print("  3. General → synthesizes findings")
    print("  4. END → outputs research summary")
    
    print("\n" + "="*60)
    print("KEY FEATURES DEMONSTRATED")
    print("="*60)
    print("""
✓ Specialist agents with domain expertise
✓ Dynamic routing based on task type
✓ State management with typed schemas
✓ Conditional edges for workflow control
✓ Iterative refinement loops
✓ Human-in-the-loop checkpoints
✓ Complete audit trail in history
    """)


if __name__ == "__main__":
    run_example()
