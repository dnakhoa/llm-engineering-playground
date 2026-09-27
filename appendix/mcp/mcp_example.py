"""
Module 14: MCP & Tool Design (MCP spec 2026-07-28)

Demonstrations:
  1. Tool description quality audit                      (no model, no server)
  2. Routing comparison: good vs bad tool descriptions   (calls your model)
  3. Error handling patterns                             (no model, no server)
  4. Schema design comparison                            (calls your model)
  5. describe_tool() helper                              (no model, no server)
  6. The 2026-07-28 wire, in process                     (needs the MCP SDK, no model)

Demos 2 and 4 go through the course's provider layer (llm/), so any model in
llm/models.json works and the registry decides what each one is sent. They are
skipped when no model is configured (see .env.example). Demo 6 connects a client to
servers/example_server.py in the same process and shows server/discover, the cache
hints on tools/list, and an InputRequiredResult round trip.
"""
import asyncio
import os
import sys

# The provider layer (llm/) and the .env file live at the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from llm import Message, ToolSpec, complete, configured_registry, default_model  # noqa: E402
from llm.transport import HttpTransport  # noqa: E402
from tools.tool_design import ToolValidator, describe_tool  # noqa: E402

REGISTRY = configured_registry()


def _model():
    """The configured model's spec, or None when no model is configured."""
    try:
        return REGISTRY.get(default_model(REGISTRY))
    except RuntimeError as err:
        print(f"  (skipped: {err})")
        return None


def _first_tool_call(spec, tools, task):
    response = complete(
        model=spec.model_id,
        messages=[Message.user(task)],
        tools=tools,
        transport=HttpTransport(provider=spec.provider),
        registry=REGISTRY,
    )
    return response.tool_calls[0] if response.tool_calls else None


# ──────────────────────────────────────────────────────────────────────────────
# 1. TOOL DESCRIPTION QUALITY AUDIT
# ──────────────────────────────────────────────────────────────────────────────


def demo_tool_validator():
    print("\n=== Demo 1: Tool Description Quality Audit ===")

    validator = ToolValidator()

    # ❌ Poorly designed tool
    def search(q: str):
        """Searches and returns results."""

    # ✅ Well designed tool
    def notes_search(query: str, limit: int = 5, tag: str = None) -> str:
        """
        Search notes in the knowledge base by keyword or tag.

        Use when the user asks to find, look up, or retrieve notes about a topic.
        Returns up to `limit` matching notes with ID, title, and excerpt.

        Do NOT use for creating notes (use notes_create) or listing all notes (use notes_list).

        limit: Maximum results (1-20, default 5).
        tag: Optional tag filter (e.g. 'meeting', 'project-x').
        """

    print("\nBad tool:")
    print(validator.report(search))

    print("\nGood tool:")
    print(validator.report(notes_search))


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROUTING COMPARISON — GOOD VS BAD DESCRIPTIONS
# ──────────────────────────────────────────────────────────────────────────────

# Bad descriptions — generic, no routing signal
BAD_TOOLS = [
    ToolSpec(
        name="search",
        description="Searches and returns data",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    ),
    ToolSpec(
        name="create",
        description="Creates a new item",
        input_schema={"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]},
    ),
]

# Good descriptions — specific routing signals
GOOD_TOOLS = [
    ToolSpec(
        name="notes_search",
        description=(
            "Search notes in the knowledge base by keyword.\n\n"
            "Use when the user asks to find, look up, or retrieve information from saved notes. "
            "Returns matching notes with ID, title, and excerpt.\n\n"
            "Do NOT use for creating new notes (use notes_create instead)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "description": "Max results (1-20)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="notes_create",
        description=(
            "Create a new note in the knowledge base.\n\n"
            "Use when the user asks to save, write, or add a note, memo, or piece of information. "
            "Returns the ID of the created note.\n\n"
            "Do NOT use to look up existing notes (use notes_search instead)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short descriptive title"},
                "content": {"type": "string", "description": "Full note content"},
                "tags": {"type": "string", "description": "Comma-separated tags. Optional.", "default": ""},
            },
            "required": ["title", "content"],
        },
    ),
]


def demo_routing_comparison():
    """How tool description quality affects routing."""
    print("\n=== Demo 2: Routing Quality — Good vs Bad Descriptions ===")
    spec = _model()
    if spec is None:
        return

    tasks = [
        "Find all notes about machine learning",
        "Save a note about today's meeting: discussed Q3 roadmap",
    ]
    for label, tools in (("Bad", BAD_TOOLS), ("Good", GOOD_TOOLS)):
        print(f"\n  {label} tool descriptions ({spec.model_id}):")
        for task in tasks:
            call = _first_tool_call(spec, tools, task)
            print(f"    Task: {task[:50]}")
            if call:
                print(f"    → Called: {call.name}({dict(call.arguments)})")
            else:
                print("    → No tool called")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ERROR HANDLING PATTERNS
# ──────────────────────────────────────────────────────────────────────────────


def demo_error_handling():
    """Actionable vs non-actionable error messages, and how they affect recovery."""
    print("\n=== Demo 3: Error Handling — Actionable vs Generic ===")

    def bad_error_tool(date: str, attendees: str) -> str:
        """Schedule a meeting."""
        if "@" not in attendees:
            raise ValueError("Invalid input")  # ❌ the agent can't act on this
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
        ("June 23rd at 2pm", "alice, bob"),  # both fields wrong
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
        print("    (Agent cannot recover — doesn't know what to fix)")

    print("\n  Good error messages:")
    for date, attendees in bad_inputs:
        result = good_error_tool(date, attendees)
        print(f"    Input: date='{date}', attendees='{attendees}'")
        print(f"    Error: {result}")
        print("    (Agent can retry with corrected inputs)")


# ──────────────────────────────────────────────────────────────────────────────
# 4. SCHEMA DESIGN COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

# Ambiguous schema — the agent must guess formats
AMBIGUOUS_SCHEMA = ToolSpec(
    name="send_notification",
    description="Send a notification to a user.",
    input_schema={
        "type": "object",
        "properties": {
            "user": {"type": "string"},
            "message": {"type": "string"},
            "priority": {"type": "string"},
            "channel": {"type": "string"},
        },
        "required": ["user", "message"],
    },
)

# Precise schema — format constraints remove the ambiguity
PRECISE_SCHEMA = ToolSpec(
    name="notify_user",
    description=(
        "Send a notification to a user via the specified channel.\n\n"
        "Use when user asks to notify, alert, ping, or message someone. "
        "Returns confirmation with delivery timestamp.\n\n"
        "Do NOT use for bulk notifications (use batch_notify instead)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "User ID (UUID format: 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')",
            },
            "message": {"type": "string", "description": "Notification message body (max 500 characters)"},
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "description": "Delivery priority. urgent = immediate push, high = within 1 min",
                "default": "normal",
            },
            "channel": {
                "type": "string",
                "enum": ["email", "slack", "sms", "push"],
                "description": "Delivery channel. sms requires phone number on file.",
                "default": "email",
            },
        },
        "required": ["user_id", "message"],
    },
)


def demo_schema_design():
    """Ambiguous vs precise schema design."""
    print("\n=== Demo 4: Schema Design — Ambiguous vs Precise ===")
    spec = _model()
    if spec is None:
        return

    task = "Send an urgent Slack message to user abc-123 saying 'System maintenance in 5 minutes'"
    for label, schema in (("Ambiguous", AMBIGUOUS_SCHEMA), ("Precise", PRECISE_SCHEMA)):
        call = _first_tool_call(spec, [schema], task)
        print(f"\n  {label} schema:")
        if call:
            print(f"    Called: {call.name}")
            for key, value in call.arguments.items():
                print(f"      {key}: {value!r}")
        else:
            print("    No tool call made")


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
            "attendees": "Comma-separated emails: 'alice@co.com,bob@co.com'",
            "calendar": "One of: personal, work, team",
        },
    )

    print("Generated description:")
    print("-" * 40)
    print(description)


# ──────────────────────────────────────────────────────────────────────────────
# 6. THE 2026-07-28 WIRE, IN PROCESS
# ──────────────────────────────────────────────────────────────────────────────


def demo_protocol():
    """server/discover, cacheable lists and an InputRequiredResult, against the example server."""
    print("\n=== Demo 6: The MCP 2026-07-28 wire, in process ===")
    try:
        from mcp import Client, types
    except ImportError:
        print("  (skipped: pip install -r requirements.txt for the MCP SDK 2.x)")
        return
    from servers.example_server import SPEC_VERSION, build_server

    async def confirm(_context, params):
        print(f"    client shows the user: {params.message!r} → accept")
        return types.ElicitResult(action="accept", content={"confirm": True})

    async def main():
        # No initialize handshake: the client pins 2026-07-28 and every request
        # carries the version in its _meta.
        async with Client(build_server(), mode=SPEC_VERSION, elicitation_callback=confirm) as client:
            discovered = await client.session.discover()
            print(f"  server/discover → supportedVersions={discovered.supported_versions}")

            listing = await client.list_tools()
            print(
                f"  tools/list → {[t.name for t in listing.tools]}\n"
                f"               ttlMs={listing.ttl_ms}, cacheScope={listing.cache_scope!r}"
            )

            # Drive one round by hand to see the InputRequiredResult itself.
            first = await client.session.call_tool("notes_delete", {"note_id": 1}, allow_input_required=True)
            [(key, request)] = first.input_requests.items()
            print(f"  tools/call notes_delete → resultType={first.result_type!r}")
            print(f"    inputRequests[{key!r}].method = {request.method!r}")

            # Client.call_tool answers the input request and retries with inputResponses.
            final = await client.call_tool("notes_delete", {"note_id": 1})
            print(f"  retry with inputResponses → {final.content[0].text}")

    asyncio.run(main())


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 14: MCP & Tool Design (MCP 2026-07-28)")
    print("=" * 60)

    demo_tool_validator()
    demo_routing_comparison()
    demo_error_handling()
    demo_schema_design()
    demo_describe_tool()
    demo_protocol()

    print("\n✅ All MCP tool design demos complete.")
    print("\nKey principles:")
    print("  1. Description = routing signal. State WHEN to use, not just WHAT it does.")
    print("  2. Schema = ambiguity reducer. Use Literal[], add examples, describe every param.")
    print("  3. Errors = recovery opportunities. Make them actionable.")
    print("  4. Namespace_verb_noun naming prevents collisions.")
    print("  5. Split on trigger condition, consolidate on shared params.")
    print("\nTo run the full MCP server:")
    print("  pip install -r requirements.txt")
    print("  python servers/example_server.py")
