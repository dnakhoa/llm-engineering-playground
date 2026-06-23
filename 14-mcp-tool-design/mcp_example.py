"""
Module 14: MCP & Tool Design
Demonstrations that work WITHOUT running an MCP server:
  1. Tool description quality audit
  2. Direct function-calling comparison (good vs bad tool design)
  3. Error handling patterns
  4. Schema design comparison
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from provider import get_llm_client

client, model = get_llm_client()

# ──────────────────────────────────────────────────────────────────────────────
# 1. TOOL DESCRIPTION QUALITY AUDIT
# ──────────────────────────────────────────────────────────────────────────────

from tools.tool_design import ToolValidator, describe_tool


def demo_tool_validator():
    print("\n=== Demo 1: Tool Description Quality Audit ===")

    validator = ToolValidator()

    # ❌ Poorly designed tool
    def search(q: str):
        """Searches and returns results."""
        pass

    # ✅ Well designed tool
    def notes_search(
        query: str,
        limit: int = 5,
        tag: str = None,
    ) -> str:
        """
        Search notes in the knowledge base by keyword or tag.

        Use when the user asks to find, look up, or retrieve notes about a topic.
        Returns up to `limit` matching notes with ID, title, and excerpt.

        Do NOT use for creating notes (use notes_create) or listing all notes (use notes_list).

        limit: Maximum results (1-20, default 5).
        tag: Optional tag filter (e.g. 'meeting', 'project-x').
        """
        pass

    print("\nBad tool:")
    print(validator.report(search))

    print("\nGood tool:")
    print(validator.report(notes_search))


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROUTING COMPARISON — GOOD VS BAD DESCRIPTIONS
# ──────────────────────────────────────────────────────────────────────────────

def demo_routing_comparison():
    """
    Shows how tool description quality affects routing accuracy.
    We give the LLM two tool sets and measure routing correctness.
    """
    print("\n=== Demo 2: Routing Quality — Good vs Bad Descriptions ===")

    # Bad descriptions — generic, no routing signal
    bad_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Searches and returns data",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create",
                "description": "Creates a new item",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"}
                    },
                    "required": ["data"]
                }
            }
        }
    ]

    # Good descriptions — specific routing signals
    good_tools = [
        {
            "type": "function",
            "function": {
                "name": "notes_search",
                "description": (
                    "Search notes in the knowledge base by keyword.\n\n"
                    "Use when the user asks to find, look up, or retrieve information from saved notes. "
                    "Returns matching notes with ID, title, and excerpt.\n\n"
                    "Do NOT use for creating new notes (use notes_create instead)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms"},
                        "limit": {"type": "integer", "description": "Max results (1-20)", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "notes_create",
                "description": (
                    "Create a new note in the knowledge base.\n\n"
                    "Use when the user asks to save, write, or add a note, memo, or piece of information. "
                    "Returns the ID of the created note.\n\n"
                    "Do NOT use to look up existing notes (use notes_search instead)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short descriptive title"},
                        "content": {"type": "string", "description": "Full note content"},
                        "tags": {"type": "string", "description": "Comma-separated tags. Optional.", "default": ""}
                    },
                    "required": ["title", "content"]
                }
            }
        }
    ]

    test_tasks = [
        "Find all notes about machine learning",
        "Save a note about today's meeting: discussed Q3 roadmap",
        "Look up what I wrote about context engineering",
        "Create a new note titled 'Ideas' with content 'Build MCP server for calendar'",
    ]

    print("\n  Bad tool descriptions:")
    for task in test_tasks[:2]:
        response = client.chat.completions.create(
            model=model,
            max_tokens=100,
            tools=bad_tools,
            tool_choice="auto",
            messages=[{"role": "user", "content": task}]
        )
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            call = choice.message.tool_calls[0]
            print(f"    Task: {task[:50]}")
            print(f"    → Called: {call.function.name}({call.function.arguments[:60]})")
        else:
            print(f"    Task: {task[:50]}")
            print(f"    → No tool called (agent gave up)")

    print("\n  Good tool descriptions:")
    for task in test_tasks[:2]:
        response = client.chat.completions.create(
            model=model,
            max_tokens=100,
            tools=good_tools,
            tool_choice="auto",
            messages=[{"role": "user", "content": task}]
        )
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            call = choice.message.tool_calls[0]
            print(f"    Task: {task[:50]}")
            print(f"    → Called: {call.function.name}({call.function.arguments[:60]})")
        else:
            print(f"    Task: {task[:50]}")
            print(f"    → No tool called")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ERROR HANDLING PATTERNS
# ──────────────────────────────────────────────────────────────────────────────

def demo_error_handling():
    """
    Compares actionable vs non-actionable error messages.
    Shows how error quality affects agent recovery.
    """
    print("\n=== Demo 3: Error Handling — Actionable vs Generic ===")

    # Scenario: agent tries to schedule a meeting with bad inputs

    def bad_error_tool(date: str, attendees: str) -> str:
        """Schedule a meeting."""
        if "@" not in attendees:
            raise ValueError("Invalid input")  # ❌ agent can't act on this
        return "Meeting scheduled."

    def good_error_tool(date: str, attendees: str) -> str:
        """
        Schedule a meeting.

        Use when user asks to book, schedule, or set up a meeting.
        date: ISO 8601 datetime string (e.g. '2026-06-23T14:30:00')
        attendees: Comma-separated email addresses (e.g. 'alice@co.com,bob@co.com')
        """
        errors = []
        if "T" not in date or len(date) < 16:
            errors.append(
                f"Invalid date format '{date}'. "
                "Use ISO 8601: '2026-06-23T14:30:00' (YYYY-MM-DDTHH:MM:SS)."
            )
        invalid_emails = [e.strip() for e in attendees.split(",") if "@" not in e.strip()]
        if invalid_emails:
            errors.append(
                f"Invalid email address(es): {invalid_emails}. "
                "All attendees must be valid email addresses."
            )
        if errors:
            return "Cannot schedule meeting:\n" + "\n".join(f"- {e}" for e in errors)
        return "Meeting scheduled successfully."

    bad_inputs = [
        ("June 23rd at 2pm", "alice, bob"),       # both fields wrong
        ("2026-06-23T14:30:00", "alice, bob@co"),  # attendees wrong
    ]

    print("\n  Bad error messages:")
    for date, attendees in bad_inputs:
        try:
            result = bad_error_tool(date, attendees)
        except ValueError as e:
            result = str(e)
        print(f"    Input: date='{date}', attendees='{attendees}'")
        print(f"    Error: {result}")
        print(f"    (Agent cannot recover — doesn't know what to fix)")

    print("\n  Good error messages:")
    for date, attendees in bad_inputs:
        result = good_error_tool(date, attendees)
        print(f"    Input: date='{date}', attendees='{attendees}'")
        print(f"    Error: {result}")
        print(f"    (Agent can retry with corrected inputs)")


# ──────────────────────────────────────────────────────────────────────────────
# 4. SCHEMA DESIGN COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def demo_schema_design():
    """
    Compares ambiguous vs precise schema design.
    """
    print("\n=== Demo 4: Schema Design — Ambiguous vs Precise ===")

    # Ambiguous schema — agent must guess format
    ambiguous_schema = {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification to a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {"type": "string"},
                    "message": {"type": "string"},
                    "priority": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["user", "message"]
            }
        }
    }

    # Precise schema — format constraints eliminate ambiguity
    precise_schema = {
        "type": "function",
        "function": {
            "name": "notify_user",
            "description": (
                "Send a notification to a user via the specified channel.\n\n"
                "Use when user asks to notify, alert, ping, or message someone. "
                "Returns confirmation with delivery timestamp.\n\n"
                "Do NOT use for bulk notifications (use batch_notify instead)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID (UUID format: 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')"
                    },
                    "message": {
                        "type": "string",
                        "description": "Notification message body (max 500 characters)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": "Delivery priority. urgent = immediate push, high = within 1 min",
                        "default": "normal"
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["email", "slack", "sms", "push"],
                        "description": "Delivery channel. sms requires phone number on file.",
                        "default": "email"
                    }
                },
                "required": ["user_id", "message"]
            }
        }
    }

    task = "Send an urgent Slack message to user abc-123 saying 'System maintenance in 5 minutes'"

    for schema_name, schema in [("Ambiguous", ambiguous_schema), ("Precise", precise_schema)]:
        response = client.chat.completions.create(
            model=model,
            max_tokens=150,
            tools=[schema],
            tool_choice="auto",
            messages=[{"role": "user", "content": task}]
        )
        choice = response.choices[0]
        print(f"\n  {schema_name} schema:")
        if choice.finish_reason == "tool_calls":
            call = choice.message.tool_calls[0]
            args = json.loads(call.function.arguments)
            print(f"    Called: {call.function.name}")
            for k, v in args.items():
                print(f"      {k}: {v!r}")
        else:
            print(f"    No tool call made")


# ──────────────────────────────────────────────────────────────────────────────
# 5. DESCRIBE_TOOL HELPER
# ──────────────────────────────────────────────────────────────────────────────

def demo_describe_tool():
    print("\n=== Demo 5: describe_tool() Helper ===")

    description = describe_tool(
        when_to_use="Use when the user asks to create, schedule, or book a calendar event or meeting.",
        returns="The created event ID and a shareable link.",
        not_for="Searching existing events (use calendar_search) or sending standalone emails.",
        param_notes={
            "start_datetime": "ISO 8601: '2026-06-23T14:30:00' (YYYY-MM-DDTHH:MM:SS)",
            "attendees":      "Comma-separated emails: 'alice@co.com,bob@co.com'",
            "calendar":       "One of: personal, work, team",
        }
    )

    print("Generated description:")
    print("-" * 40)
    print(description)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 14: MCP & Tool Design")
    print("=" * 60)

    demo_tool_validator()
    demo_routing_comparison()
    demo_error_handling()
    demo_schema_design()
    demo_describe_tool()

    print("\n✅ All MCP tool design demos complete.")
    print("\nKey principles:")
    print("  1. Description = routing signal. State WHEN to use, not just WHAT it does.")
    print("  2. Schema = ambiguity reducer. Use Literal[], add examples, describe every param.")
    print("  3. Errors = recovery opportunities. Make them actionable.")
    print("  4. Namespace_verb_noun naming prevents collisions.")
    print("  5. Split on trigger condition, consolidate on shared params.")
    print("\nTo run the full MCP server:")
    print("  pip install mcp")
    print("  mcp dev servers/example_server.py")
