# Module 14: MCP & Tool Design


> **Why this matters:** Tools are how agents interact with the world. Bad tool design causes routing failures, hallucinated arguments, and unreliable agents. Good ACI design has higher ROI than prompt engineering.


## 🎯 Learning Objectives
- Understand the Model Context Protocol (MCP) architecture and primitives
- Build a complete MCP server in Python using FastMCP
- Write tool descriptions that agents can reliably route on
- Design tool schemas that minimize ambiguity and hallucinated arguments
- Handle errors at both the protocol and business-logic levels
- Decide when to split vs consolidate tools

## 📚 What is MCP?

The **Model Context Protocol** (MCP, spec version `2025-06-18`) is an open standard for connecting AI models to external tools, data sources, and services. It replaces ad-hoc function-calling integrations with a structured, discoverable interface.

**Why it matters**: Before MCP, every LLM application had to implement its own tool-calling format. MCP provides a standard so any MCP-compatible client (Claude, Cursor, Windsurf, etc.) can use any MCP-compatible server without custom integration.

```
┌────────────────────────────────────────────────────────────────┐
│                     MCP Architecture                           │
│                                                                │
│  ┌─────────────┐    JSON-RPC 2.0    ┌───────────────────────┐ │
│  │    Host     │◄──────────────────▶│      MCP Server       │ │
│  │ (Claude,    │                    │                       │ │
│  │  Cursor,    │    ┌───────────┐   │  ┌─────────────────┐  │ │
│  │  your app)  │    │  Client   │   │  │  Tools          │  │ │
│  │             │    │ (1 per    │   │  │  Resources      │  │ │
│  └─────────────┘    │ server)   │   │  │  Prompts        │  │ │
│                     └───────────┘   │  │  Sampling       │  │ │
│                                     │  └─────────────────┘  │ │
│                                     └───────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## 🧩 Core Primitives

| Primitive | Direction | Description |
|-----------|-----------|-------------|
| **Tools** | Model → Server | Functions the model can call (read/write). Require user approval. |
| **Resources** | Server → Client | File-like context the client can read (APIs, files, DB records). |
| **Prompts** | Server → Client | Reusable prompt templates with parameters, surfaced in UIs. |
| **Sampling** | Server → Model | Server requests an LLM completion from the host — enables nested agents. |

**Tools** are the most commonly used primitive and the focus of this module.

## 🚀 Building an MCP Server

### Installation

```bash
pip install mcp   # installs mcp + FastMCP
```

### Minimal Server (stdio transport)

```python
# servers/weather_server.py
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("weather-service")

@mcp.tool()
async def get_current_weather(
    latitude: float,
    longitude: float,
) -> str:
    """
    Get current weather conditions for a geographic location.

    Use when the user asks about current weather, temperature, or conditions
    at a specific location by coordinates. Returns temperature, wind speed,
    and weather description.

    Do NOT use for weather forecasts (use get_forecast instead).
    """
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude, "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m,weather_code",
            }
        )
        data = resp.json()["current"]
        return (
            f"Temperature: {data['temperature_2m']}°C | "
            f"Wind: {data['wind_speed_10m']} km/h | "
            f"Code: {data['weather_code']}"
        )


@mcp.resource("weather://alerts/{state_code}")
async def get_weather_alerts(state_code: str) -> str:
    """
    Active weather alerts for a US state.
    state_code: Two-letter US state code (e.g. CA, NY, TX).
    """
    # Implementation would call weather.gov API
    return f"No active alerts for {state_code}"


if __name__ == "__main__":
    mcp.run()  # default: stdio transport
    # mcp.run(transport="streamable-http", port=8080)  # HTTP transport
```

### Running the Server

```bash
# stdio (used by Claude Desktop, Claude Code)
python servers/weather_server.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector python servers/weather_server.py

# HTTP transport (for remote/hosted servers)
python servers/weather_server.py --transport streamable-http --port 8080
```

### Registering with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/servers/weather_server.py"]
    }
  }
}
```

---

## ✍️ Tool Design Principles

Good tool design is the difference between an agent that reliably routes to the right tool and one that hallucinates calls or ignores available tools.

### 1. Descriptions That Agents Can Route On

The `description` is the **primary routing signal** — the model reads it to decide whether to call this tool.

```python
# ❌ BAD: Describes WHAT it does, not WHEN to use it
@mcp.tool()
def search_documents(query: str) -> str:
    """Searches documents and returns results."""

# ✅ GOOD: Tells the model WHEN to call this tool
@mcp.tool()
def search_documents(query: str) -> str:
    """
    Search the internal knowledge base for company documents, policies, and procedures.

    Use when the user asks about internal company information, policies, HR procedures,
    or any topic that might be documented internally. Returns the top 3 matching
    document excerpts with source citations.

    Do NOT use for general web search (use web_search instead) or real-time data.
    """
```

**Description checklist:**
- [ ] States WHEN to use the tool (the trigger condition)
- [ ] States what it returns (output format)
- [ ] States what it does NOT handle (disambiguation from similar tools)
- [ ] Includes format/constraint hints for parameters
- [ ] Is under 200 words (longer descriptions dilute the routing signal)

### 2. Schema Design to Reduce Ambiguity

```python
from typing import Literal

@mcp.tool()
def create_calendar_event(
    title: str,
    start_datetime: str,   # ISO 8601 format: "2026-06-23T14:30:00"
    duration_minutes: int,
    calendar: Literal["personal", "work", "team"] = "personal",
    attendees: list[str] = None,  # list of email addresses
    description: str = "",
) -> dict:
    """
    Create a new calendar event and send invites to attendees.

    Use when user asks to schedule, book, or create a meeting or event.
    start_datetime must be in ISO 8601 format (e.g. "2026-06-23T14:30:00").
    Returns the created event ID and a shareable link.
    """
    ...
```

**Schema checklist:**
- [ ] Use `Literal` for fixed-choice parameters (replaces prose constraints)
- [ ] Include format examples in parameter descriptions ("ISO 8601: '2026-06-23T14:30:00'")
- [ ] Only mark truly required fields as required — use defaults elsewhere
- [ ] Add `description` to every parameter, not just the tool
- [ ] Prefer primitive types (str, int, bool) over free-form dicts
- [ ] Provide `outputSchema` for structured return values

### 3. Error Handling — Two Layers

MCP distinguishes between two kinds of errors:

```python
# Layer 1: Protocol errors (JSON-RPC level)
# — Unknown tool, invalid arguments, serialization failure
# — The client surfaces these automatically; you don't handle them in tool code

# Layer 2: Tool execution errors (business logic)
# — API failures, bad inputs, rate limits, timeouts
# — Return as isError=True with an actionable message the LLM can act on

from mcp.types import TextContent

@mcp.tool()
async def send_email(to: str, subject: str, body: str) -> list[TextContent]:
    """Send an email via the configured SMTP server."""
    try:
        result = await smtp_client.send(to=to, subject=subject, body=body)
        return [TextContent(type="text", text=f"Email sent. Message ID: {result.id}")]
    except InvalidAddressError:
        # Actionable error — LLM can ask user to fix the address
        return [TextContent(
            type="text",
            text=f"Invalid email address: '{to}'. "
                 "Please provide a valid email address (e.g. name@domain.com)."
        )]
    except RateLimitError as e:
        # Retriable error with wait hint
        return [TextContent(
            type="text",
            text=f"Rate limit reached. Wait {e.retry_after_seconds}s and retry."
        )]
    except Exception as e:
        # Generic — don't expose internals, but give enough to debug
        return [TextContent(
            type="text",
            text=f"Email send failed: {type(e).__name__}. Try again or check credentials."
        )]
```

**Error message guidelines:**
- Include what failed and why (in user-understandable terms)
- State what the model or user can do to fix it
- Never expose stack traces or internal URLs
- For retriable errors: include the wait time

### 4. Consolidate vs Split

```
SPLIT when:                          CONSOLIDATE when:
───────────────────────────────      ────────────────────────────────
Different required params            Share >80% of parameters
Different use cases / triggers       Always called together in sequence
Different side effects               Few optional params, same trigger
One is read, one is write            Would create micro-tools <3 params
Different safety profiles            
```

```python
# ❌ Over-split (would be called together every time)
@mcp.tool()
def get_user_id(username: str) -> str: ...
@mcp.tool()
def get_user_profile(user_id: str) -> dict: ...

# ✅ Consolidated (one trigger, one semantic action)
@mcp.tool()
def get_user(username: str) -> dict:
    """Get full user profile by username."""
    user_id = db.lookup_id(username)
    return db.get_profile(user_id)


# ✅ Correctly split (different triggers, different side effects)
@mcp.tool()
def search_emails(query: str, limit: int = 10) -> list[dict]:
    """Search emails — read only, safe to call frequently."""

@mcp.tool()
def delete_email(email_id: str) -> dict:
    """Delete a specific email — irreversible, requires confirmation."""
```

### 5. Tool Naming Conventions

```python
# ✅ namespace_verb_noun — clear, collision-resistant
calendar_create_event
calendar_list_events
calendar_delete_event
email_search
email_send
db_query
db_insert

# ❌ Vague or generic — causes routing confusion
search        # search what?
create        # create what?
get_data      # what data?
process       # process how?
```

---

## 🔗 Resources

Resources expose file-like data for the client to read into context:

```python
@mcp.resource("db://tables/{table_name}/schema")
async def get_table_schema(table_name: str) -> str:
    """
    Database table schema — use to understand column names and types
    before writing SQL queries.
    """
    schema = await db.get_schema(table_name)
    return schema.to_json()


@mcp.resource("config://app")
async def get_app_config() -> str:
    """Current application configuration (read-only)."""
    return json.dumps(load_config(), indent=2)
```

Resources differ from tools:
- Resources are **read-only** and explicitly fetched by the client
- Tools are **callable** by the model (may have side effects)
- Resources appear as "attachable context", tools appear as "callable functions"

---

## 🔐 Secure MCP Tunnels

For production deployments, expose MCP servers securely without making them public:

```bash
# OpenAI's Secure MCP Tunnel — connects private servers to ChatGPT
# without exposing them to the public internet
npx @openai/mcp-tunnel --server http://localhost:8080/mcp
```

**Why tunnels matter**:
- MCP servers often need access to internal databases, APIs, and services
- Direct exposure creates security risks
- Tunnels provide authenticated, encrypted access without public endpoints
- Works with ChatGPT Apps SDK, Claude, and other MCP clients

### Computer Use as a Tool

Modern models can interact with computers directly — clicking, typing, taking screenshots. This is exposed as a tool type in both OpenAI and Anthropic APIs:

```python
# Anthropic Computer Use
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=4096,
    tools=[{
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
        "display_number": 0,
    }],
    messages=[{
        "role": "user",
        "content": "Open a browser and search for 'LLM engineering courses'."
    }]
)
```

**When to use Computer Use**:
- Automating tasks that require GUI interaction
- Testing web applications
- Scraping dynamic content that requires browser interaction
- Legacy system integration where no API exists

**CAUTION**: Computer Use has broad capabilities — always run in sandboxed environments with appropriate guardrails.

---

## 📋 Prompt Templates

Prompts define reusable templates that UIs can surface:

```python
@mcp.prompt()
def code_review_prompt(language: str, focus: str = "correctness") -> str:
    """Standard code review prompt. Use from slash-command /review."""
    return (
        f"Review the following {language} code for {focus}. "
        "For each issue found: (1) state the severity [critical/warning/info], "
        "(2) explain the problem, (3) provide the corrected code. "
        "Only report real issues with high confidence."
    )
```

---

## 🏗️ Project Structure

```
14-mcp-tool-design/
├── README.md
├── requirements.txt
├── mcp_example.py               ★ Standalone demo (no server needed)
├── servers/
│   ├── __init__.py
│   └── example_server.py        ★ Complete MCP server (run with mcp dev)
└── tools/
    ├── __init__.py
    └── tool_design.py           Tool design helpers and validators
```


## 📚 Resources

- [MCP Specification](https://modelcontextprotocol.io/) — Model Context Protocol
- [A2A Protocol](https://a2a-protocol.org/) — Agent-to-Agent standard
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — ACI principles

## 🧪 Hands-On Exercises

1. **Build a File System Server**: Create an MCP server with 4 tools: `file_read`, `file_write`, `file_list`, and `file_delete`. Apply the split/consolidate principle — should read and write be one tool or two? Why?

2. **Description Quality Test**: Take any 3 tools from the official MCP server repo. Rewrite their descriptions to include: (a) when to use, (b) what it returns, (c) what it doesn't handle. Call the same agent task with old vs new descriptions. Does routing improve?

3. **Schema Ambiguity Hunt**: Build a tool for scheduling meetings with a free-form `time` parameter (e.g., "next Tuesday at 3pm"). Have an agent call it 20 times. How often does the agent pass invalid formats? Now add a strict `start_datetime: str  # ISO 8601` with an example. Does error rate drop?

4. **Error Recovery Test**: Build a tool that throws 3 different error types. Write actionable error messages for each. Have an agent encounter each error and observe whether it: (a) retries with a fix, (b) tries an alternative approach, (c) gives up. Compare with and without actionable messages.

5. **Multi-Server Discovery**: Start 3 MCP servers with overlapping capabilities (e.g., two calendar tools). Connect an agent to all three. How does it choose between them? Refine descriptions until the agent consistently picks the right one.

6. **Resource vs Tool**: Design a database access pattern. Which operations should be Resources (schema discovery, config) and which should be Tools (insert, update, query)? Build both and compare how an agent uses them.

---

## 📚 References

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-06-18) — official spec (current: 2025-06-18)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — FastMCP
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — dev tool for testing servers
- [MCP Registry](https://registry.modelcontextprotocol.io) — discover community servers
- [Official Reference Servers](https://github.com/modelcontextprotocol/servers) — filesystem, git, GitHub, PostgreSQL, etc.
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — ACI design principles
- [OpenAI Secure MCP Tunnels](https://platform.openai.com/docs/guides/secure-mcp-tunnels) — production MCP security
- [MCPEvol-Bench](https://arxiv.org/abs/2607.14642) — benchmarking LLM agents across MCP server evolutions

## 🔗 Integration with Other Modules

- **Module 07 (Agents)**: Agents use tools — this module shows how to build good ones
- **Module 10 (Guardrails)**: Tool validation and safety belong in the server
- **Module 08 (LLM Ops)**: Add tracing to every tool call for observability
- **Module 12 (Context)**: Tool output is context — design small, focused outputs

---

**Good tools make agents more capable. Bad tools make them unpredictable. Design matters.**


---

## Agent-to-Agent (A2A) Protocol

While MCP connects agents to **tools**, the **Agent2Agent (A2A) protocol** enables agents to communicate with **each other**. It's a Google-led standard gaining traction alongside MCP.

### MCP vs A2A

| | MCP | A2A |
|--|-----|-----|
| **Connects** | Agent ↔ Tool | Agent ↔ Agent |
| **Direction** | Agent calls tool | Agents collaborate |
| **Use case** | Database query, API call | Multi-agent orchestration |
| **Standard** | Anthropic-led | Google-led |

### A2A Architecture

```
┌─────────────┐     A2A Protocol     ┌─────────────┐
│   Agent A   │◄────────────────────▶│   Agent B   │
│  (Research) │   task delegation     │  (Coding)   │
└─────────────┘   status updates      └─────────────┘
       │            results                │
       ▼                                   ▼
┌─────────────┐                    ┌─────────────┐
│  MCP Tools  │                    │  MCP Tools  │
│  (search,   │                    │  (code exec,│
│   browse)   │                    │   test)     │
└─────────────┘                    └─────────────┘
```

### When to Use A2A

| Scenario | Why A2A |
|----------|---------|
| Multi-agent teams | Agents need to delegate subtasks to specialists |
| Cross-org collaboration | Different orgs' agents work together |
| Heterogeneous agents | Agents built on different frameworks need to interoperate |
| Complex workflows | Task decomposition across specialized agents |

### A2A + MCP Together

Most production agent systems use both:
- **MCP** for tool access (databases, APIs, file systems)
- **A2A** for agent collaboration (delegation, synthesis, verification)

```python
# Conceptual: A2A agent interaction
agent_a = ResearchAgent(mcp_tools=["search", "browse"])
agent_b = CoderAgent(mcp_tools=["code_exec", "test"])

# Agent A discovers Agent B via A2A
task = a2a_client.create_task(
    agent="coder-agent",
    description="Write a Python function to parse the research findings",
    context=research_results
)

# Agent B executes and returns results
result = a2a_client.wait_for_result(task.id)
```

### A2A Resources

- [A2A Protocol Spec](https://a2a-protocol.org/) — official specification
- [Google A2A GitHub](https://github.com/google/a2a-python) — Python SDK
- [A2A + MCP Integration](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — Google blog post

