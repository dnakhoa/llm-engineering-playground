"""
Human-in-the-Loop Workflow with LangGraph

This example demonstrates how to add human approval checkpoints
to multi-agent workflows for critical decisions, safety, and quality control.

Key Features:
- Interrupt execution at defined checkpoints
- Present context to human reviewer
- Capture approval/rejection with feedback
- Resume workflow with human input
- Audit trail of all approvals
"""

from typing import TypedDict, Annotated, List, Literal, Optional, Dict, Any
import operator
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# STATE SCHEMA WITH HUMAN INPUT
# ============================================================================

class HumanReviewState(TypedDict):
    """State schema that includes human review fields."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    pending_approval: bool  # True if waiting for human
    approval_status: Optional[str]  # approved, rejected, pending
    human_feedback: Optional[str]
    review_context: Optional[Dict[str, Any]]  # Context for human reviewer
    iteration_count: int
    final_output: Optional[str]


# ============================================================================
# AGENT NODES
# ============================================================================

class ContentGenerator:
    """Agent that generates content requiring human review."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: HumanReviewState) -> HumanReviewState:
        """Generate content that needs approval."""
        print("\n✍️ [GENERATOR] Creating content...")
        
        # Mock content generation
        generated_content = {
            "type": "article",
            "title": "The Future of AI in Healthcare",
            "content": "Artificial intelligence is revolutionizing healthcare...",
            "word_count": 500,
            "confidence": 0.92,
            "sources_cited": 3
        }
        
        print(f"✓ Generated {generated_content['type']}: {generated_content['title']}")
        
        return {
            "messages": [AIMessage(content=f"Generated article: {generated_content['title']}")],
            "current_step": "content_generated",
            "pending_approval": True,
            "review_context": {
                "action": "publish_article",
                "content_summary": generated_content,
                "risk_level": "medium",
                "requires_expertise": "medical accuracy"
            }
        }


class CodeGenerator:
    """Agent that generates code requiring human review."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: HumanReviewState) -> HumanReviewState:
        """Generate code that needs security review."""
        print("\n💻 [CODER] Writing production code...")
        
        generated_code = {
            "language": "python",
            "purpose": "Database migration script",
            "lines": 75,
            "security_sensitive": True,
            "affects_production": True,
            "code_preview": "def migrate_users():\n    # Migration logic..."
        }
        
        print(f"✓ Generated {generated_code['purpose']} ({generated_code['lines']} lines)")
        
        return {
            "messages": [AIMessage(content=f"Generated code: {generated_code['purpose']}")],
            "current_step": "code_generated",
            "pending_approval": True,
            "review_context": {
                "action": "deploy_code",
                "code_summary": generated_code,
                "risk_level": "high",
                "requires_expertise": "security review",
                "rollback_plan": "Available"
            }
        }


class DecisionMaker:
    """Agent that makes recommendations requiring human approval."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: HumanReviewState) -> HumanReviewState:
        """Make a decision recommendation."""
        print("\n🤔 [DECIDER] Analyzing options...")
        
        recommendation = {
            "decision_type": "vendor_selection",
            "recommendation": "Select Vendor A",
            "alternatives": ["Vendor B", "Vendor C"],
            "rationale": "Best cost-performance ratio",
            "estimated_impact": "$50k annual savings",
            "risks": ["Vendor lock-in", "Migration effort"]
        }
        
        print(f"✓ Recommendation: {recommendation['recommendation']}")
        
        return {
            "messages": [AIMessage(content=f"Recommendation: {recommendation['recommendation']}")],
            "current_step": "decision_made",
            "pending_approval": True,
            "review_context": {
                "action": "approve_decision",
                "recommendation": recommendation,
                "risk_level": "high",
                "financial_impact": True,
                "stakeholders": ["Engineering", "Finance", "Legal"]
            }
        }


# ============================================================================
# HUMAN REVIEW NODE
# ============================================================================

def human_review_node(state: HumanReviewState) -> HumanReviewState:
    """
    This node represents a human review checkpoint.
    
    In production, this would:
    1. Pause the workflow
    2. Send notification to human reviewer
    3. Wait for approval/rejection via UI/API
    4. Resume with human's decision
    
    For this demo, we simulate the human input.
    """
    print("\n⏸️ [HUMAN REVIEW] Waiting for approval...")
    
    review_context = state.get("review_context", {})
    
    # Display context for human reviewer
    print("\n" + "="*60)
    print("REVIEW REQUIRED")
    print("="*60)
    print(f"Action: {review_context.get('action', 'Unknown')}")
    print(f"Risk Level: {review_context.get('risk_level', 'Unknown')}")
    print(f"Requires Expertise: {review_context.get('requires_expertise', 'General')}")
    
    if "content_summary" in review_context:
        summary = review_context["content_summary"]
        print(f"\nContent: {summary.get('title', 'N/A')}")
        print(f"Type: {summary.get('type', 'N/A')}")
    
    if "code_summary" in review_context:
        code = review_context["code_summary"]
        print(f"\nCode Purpose: {code.get('purpose', 'N/A')}")
        print(f"Lines: {code.get('lines', 'N/A')}")
        print(f"Security Sensitive: {code.get('security_sensitive', False)}")
    
    if "recommendation" in review_context:
        rec = review_context["recommendation"]
        print(f"\nRecommendation: {rec.get('recommendation', 'N/A')}")
        print(f"Impact: {rec.get('estimated_impact', 'N/A')}")
    
    print("\n" + "="*60)
    print("SIMULATED HUMAN DECISION (in production, wait for real input)")
    print("="*60)
    
    # Simulate human decision (in production, this comes from UI/API)
    # For demo, we'll approve medium risk, reject high risk
    risk_level = review_context.get("risk_level", "medium")
    
    if risk_level == "high":
        approval_status = "rejected"
        human_feedback = "Please address security concerns and resubmit for review."
        print(f"\n❌ REJECTED: {human_feedback}")
    else:
        approval_status = "approved"
        human_feedback = "Approved with minor suggestions added to comments."
        print(f"\n✅ APPROVED: {human_feedback}")
    
    return {
        "messages": [
            HumanMessage(
                content=f"Human review: {approval_status}. Feedback: {human_feedback}"
            )
        ],
        "current_step": "human_reviewed",
        "pending_approval": False,
        "approval_status": approval_status,
        "human_feedback": human_feedback,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_generation(state: HumanReviewState) -> Literal["human_review", "END"]:
    """Route to human review if pending approval."""
    if state.get("pending_approval", False):
        print("→ Routing to human review")
        return "human_review"
    print("→ No approval needed, ending")
    return "END"


def route_after_review(state: HumanReviewState) -> Literal["finalize", "revise", "END"]:
    """Route based on human decision."""
    approval_status = state.get("approval_status", "pending")
    
    if approval_status == "approved":
        print("→ Approved, proceeding to finalize")
        return "finalize"
    elif approval_status == "rejected":
        # Check if we've tried too many times
        if state.get("iteration_count", 0) >= 2:
            print("→ Rejected but max iterations reached, ending")
            return "END"
        print("→ Rejected, routing back for revision")
        return "revise"
    else:
        print("→ Unknown status, ending")
        return "END"


def determine_revision_target(state: HumanReviewState) -> Literal["content_generator", "code_generator", "decision_maker"]:
    """Determine which agent should revise based on context."""
    context = state.get("review_context", {})
    action = context.get("action", "")
    
    if "publish" in action or "content" in action:
        return "content_generator"
    elif "deploy" in action or "code" in action:
        return "code_generator"
    elif "approve" in action or "decision" in action:
        return "decision_maker"
    else:
        return "content_generator"  # Default


# ============================================================================
# FINALIZATION NODE
# ============================================================================

class Finalizer:
    """Finalize and output approved content."""
    
    def __call__(self, state: HumanReviewState) -> HumanReviewState:
        """Produce final output."""
        print("\n✅ [FINALIZER] Producing final output...")
        
        feedback = state.get("human_feedback", "")
        
        final_output = {
            "status": "approved_and_published",
            "timestamp": datetime.now().isoformat(),
            "human_feedback_incorporated": feedback,
            "audit_trail": {
                "generations": state.get("iteration_count", 1),
                "approvals": 1,
                "rejections": 0
            }
        }
        
        print(f"✓ Final output ready")
        
        return {
            "messages": [AIMessage(content="Content approved and finalized.")],
            "current_step": "complete",
            "final_output": str(final_output),
            "pending_approval": False
        }


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_human_in_loop_graph(llm=None):
    """
    Build a workflow with human-in-the-loop checkpoints.
    
    Returns:
        Compiled LangGraph workflow with interrupt capability
    """
    if llm is None:
        from unittest.mock import Mock
        llm = Mock()
        llm.invoke = lambda msgs: AIMessage(content="Mock response")
    
    # Initialize agents
    content_gen = ContentGenerator(llm)
    code_gen = CodeGenerator(llm)
    decision_maker = DecisionMaker(llm)
    finalizer = Finalizer()
    
    # Create graph
    workflow = StateGraph(HumanReviewState)
    
    # Add nodes
    workflow.add_node("content_generator", content_gen)
    workflow.add_node("code_generator", code_gen)
    workflow.add_node("decision_maker", decision_maker)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalizer)
    
    # Set entry point (example: start with content generation)
    workflow.set_entry_point("content_generator")
    
    # Route from generators to human review
    workflow.add_conditional_edges(
        "content_generator",
        route_after_generation,
        {"human_review": "human_review", "END": "END"}
    )
    
    workflow.add_conditional_edges(
        "code_generator",
        route_after_generation,
        {"human_review": "human_review", "END": "END"}
    )
    
    workflow.add_conditional_edges(
        "decision_maker",
        route_after_generation,
        {"human_review": "human_review", "END": "END"}
    )
    
    # Route from human review based on decision
    workflow.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "finalize": "finalize",
            "revise": "revise",  # Will be handled by conditional edge below
            "END": "END"
        }
    )
    
    # Handle revision routing
    workflow.add_conditional_edges(
        "human_review",
        determine_revision_target,
        {
            "content_generator": "content_generator",
            "code_generator": "code_generator",
            "decision_maker": "decision_maker"
        }
    )
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", "END")
    
    # Compile with memory for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_hitl_example():
    """Demonstrate human-in-the-loop workflow."""
    print("="*60)
    print("HUMAN-IN-THE-LOOP WORKFLOW DEMO")
    print("="*60)
    
    app = build_human_in_loop_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="Write an article about AI in healthcare")],
        "current_step": "start",
        "pending_approval": False,
        "approval_status": None,
        "human_feedback": None,
        "review_context": None,
        "iteration_count": 0,
        "final_output": None
    }
    
    print("\n⚠️  Note: This is a structural demo with simulated human input.")
    print("   In production:")
    print("   1. Workflow interrupts at human_review node")
    print("   2. Notification sent to reviewer (email, Slack, UI)")
    print("   3. Reviewer approves/rejects via interface")
    print("   4. Workflow resumes with decision\n")
    
    print("Expected flow for MEDIUM RISK content:")
    print("  1. Content Generator → creates article")
    print("  2. Human Review → approves (simulated)")
    print("  3. Finalizer → publishes content")
    print("  4. END\n")
    
    print("Expected flow for HIGH RISK code:")
    print("  1. Code Generator → writes production code")
    print("  2. Human Review → rejects due to security (simulated)")
    print("  3. Code Generator → revises based on feedback")
    print("  4. Human Review → approves revised version")
    print("  5. Finalizer → deploys code")
    print("  6. END\n")
    
    print("="*60)
    print("KEY HITL FEATURES")
    print("="*60)
    print("""
✓ Interrupt at defined checkpoints
✓ Context-rich review requests
✓ Approval/rejection with feedback
✓ Revision loops based on feedback
✓ Complete audit trail
✓ Thread persistence for long reviews
    """)
    
    # To run with actual interruption:
    # config = {"configurable": {"thread_id": "review_123"}}
    # result = app.invoke(initial_state, config=config)
    # 
    # To resume after human input:
    # app.update_state(config, {"approval_status": "approved", "human_feedback": "Looks good!"})
    # result = app.invoke(None, config=config)


if __name__ == "__main__":
    run_hitl_example()
