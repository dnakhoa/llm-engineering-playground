# Module 14: MCP & Tool Design

_Last verified: 2026-09-27, against the MCP specification 2026-07-28 and the MCP Python SDK 2.2.0_

> **Why this matters:** Tools are how agents interact with the world. Bad tool design causes routing failures, hallucinated arguments, and unreliable agents. Good ACI design has higher ROI than prompt engineering.

## Learning Objectives
- Understand the Model Context Protocol (MCP) architecture and primitives, as of spec `2026-07-28`
- Know what `2026-07-28` changed: stateless requests, `server/discover`, `InputRequiredResult`, Tasks as an extension, cacheable lists
- Build a complete MCP server in Python with the SDK's `MCPServer`
- Write tool descriptions that agents can reliably route on
- Design tool schemas that minimize ambiguity and hallucinated arguments
- Handle errors at both the protocol and business-logic levels
- Decide when to split vs consolidate tools

## 📚 What is MCP?

The **Model Context Protocol** ([MCP, spec version `2026-07-28`](https://modelcontextprotocol.io/specification/2026-07-28)) is an open standard for connecting AI models to external tools, data sources, and services. It replaces ad-hoc function-calling integrations with a structured, discoverable interface.

**Why it matters**: Before MCP, every LLM application had to implement its own tool-calling format. MCP provides a standard so any MCP-compatible client (Claude, Cursor, Windsurf, etc.) can use any MCP-compatible server without custom integration.

```
┌────────────────────────────────────────────────────────────────┐
│                  MCP Architecture (2026-07-28)                 │
│                                                                │
│  ┌─────────────┐   JSON-RPC 2.0     ┌───────────────────────┐  │
│  │    Host     │   one request,     │      MCP Server       │  │
│  │ (Claude,    │   one response —   │                       │  │
│  │  Cursor,    │   each carries its │  ┌─────────────────┐  │  │
│  │  your app)  │   own _meta        │  │  Tools          │  │  │
│  │  ┌───────┐  │◄──────────────────▶│  │  Resources      │  │  │
│  │  │Client │  │                    │  │  Prompts        │  │  │
│  │  └───────┘  │                    │  └─────────────────┘  │  │
│  └─────────────┘                    └───────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 🆕 What changed in 2026-07-28

Each row is from the spec's own [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog). The previous revision was `2025-11-25`. <!-- historical-exception -->

| Change | Before | In 2026-07-28 |
|--------|--------|---------------|
| **Stateless requests** (SEP-2575) | An `initialize` / `notifications/initialized` handshake opened a session | No handshake. Every request carries `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` in its `_meta`; clients SHOULD send `io.modelcontextprotocol/clientInfo`, servers SHOULD put `io.modelcontextprotocol/serverInfo` in each result's `_meta`. A version the server does not speak gets an `UnsupportedProtocolVersionError` |
| **No protocol sessions** (SEP-2567) | Streamable HTTP tracked a session with the `Mcp-Session-Id` header | The header and protocol-level sessions are gone. `tools/list`, `resources/list` and `prompts/list` no longer vary per connection. State that must survive between calls lives in server-minted handles passed as ordinary tool arguments |
| **`server/discover`** (SEP-2575) | Capabilities came back from `initialize` | Servers MUST implement `server/discover`, which returns supported versions, capabilities and identity. Clients MAY call it first; on stdio it is the backward-compatibility probe |
| **`InputRequiredResult`** (SEP-2322) | A server sent its own requests mid-call: `elicitation/create`, `sampling/createMessage`, `roots/list` | Multi Round-Trip Requests: the server returns an `InputRequiredResult` (`resultType: "input_required"`) whose `inputRequests` carry those requests; the client retries the original request with `inputResponses` |
| **`resultType` on every result** (SEP-2322) | — | `"complete"` for ordinary results, `"input_required"` for the interim ones |
| **Tasks as an extension** (SEP-2663) | Experimental tasks in the core protocol | Moved to the official extension `io.modelcontextprotocol/tasks`: poll with `tasks/get`, send input with `tasks/update`; the blocking `tasks/result` and `tasks/list` are removed |
| **Cacheable lists** (SEP-2549) | Clients re-listed on `listChanged` notifications or polled | `tools/list`, `prompts/list`, `resources/list`, `resources/read` and `resources/templates/list` results carry `ttlMs` (a freshness hint) and `cacheScope` (`"public"` or `"private"`). Servers SHOULD return `tools/list` in a deterministic order |
| **`subscriptions/listen`** (SEP-2575) | An HTTP GET stream and `resources/subscribe` | One long-lived POST response stream the client opts in to per notification type |
| **Removed** (SEP-2575) | `ping`, `logging/setLevel`, `notifications/roots/list_changed`, SSE resumability (`Last-Event-ID`) | Log level is set per request in `_meta` (`io.modelcontextprotocol/logLevel`); a broken stream loses the request, and the client re-issues it with a new ID |
| **Deprecated** (SEP-2577) | Roots, Sampling and Logging | Still working, but new implementations should not add them: pass paths as tool parameters or resource URIs, call your LLM provider directly, log to `stderr` or OpenTelemetry |

### Stateless requests

Every request stands alone. There is nothing to set up first, so any server replica can answer any request:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "notes_search",
    "arguments": { "query": "refund" },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": { "name": "my-agent", "version": "1.0.0" },
      "io.modelcontextprotocol/clientCapabilities": { "elicitation": { "form": {} } }
    }
  }
}
```

### `server/discover`

A client can ask for a server's versions, capabilities and identity in one call, before anything else:

```json
{
  "jsonrpc": "2.0",
  "id": "discover-1",
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": { "tools": {}, "resources": {} },
    "_meta": { "io.modelcontextprotocol/serverInfo": { "name": "knowledge-base-server", "version": "2.0.0" } }
  }
}
```

### `InputRequiredResult`: asking the user without a server-initiated request

A server that needs more input — a confirmation, a missing field — answers the call with an `InputRequiredResult`. The client gathers the answer and retries the same call with `inputResponses` (and the opaque `requestState`, if the server sent one):

```
Client ── tools/call notes_delete {note_id: 1} ───────────────────────▶ Server
Client ◀── resultType: "input_required", inputRequests: {confirm: elicitation/create} ──
          (client asks the user: "Delete note 1?")
Client ── tools/call notes_delete {note_id: 1}, inputResponses: {confirm: accept} ──▶
Client ◀── resultType: "complete", content: "Note 1 deleted." ────────────────
```

In the Python SDK a tool asks through a resolver; the SDK builds the `InputRequiredResult` and resumes on the retry (from [`servers/example_server.py`](servers/example_server.py)):

```python
from typing import Annotated
from pydantic import BaseModel
from mcp.server.mcpserver import AcceptedElicitation, Elicit, ElicitationResult, MCPServer, Resolve

class DeleteConfirmation(BaseModel):
    confirm: bool

def confirm_delete(note_id: int) -> Elicit[DeleteConfirmation]:
    return Elicit(f"Delete note {note_id}? This cannot be undone.", DeleteConfirmation)

@mcp.tool()
def notes_delete(
    note_id: int,
    confirmation: Annotated[ElicitationResult[DeleteConfirmation], Resolve(confirm_delete)],
) -> str:
    """Delete a note by ID. Irreversible: the user is asked to confirm first."""
    if not (isinstance(confirmation, AcceptedElicitation) and confirmation.data.confirm):
        return f"Note {note_id} not deleted: the user did not confirm."
    ...
```

`confirmation` is not part of the tool's input schema: the model never fills it, the user does.

### Cacheable lists

List results carry freshness hints, so a client re-lists only when its copy goes stale. Set them once for the server:

```python
from mcp.server.caching import CacheHint

mcp = MCPServer(
    "knowledge-base-server",
    cache_hints={
        "tools/list": CacheHint(ttl_ms=300_000, scope="public"),    # same for every caller
        "resources/read": CacheHint(ttl_ms=30_000, scope="private"),  # per-user data
    },
)
```

`cacheScope: "public"` lets a shared intermediary cache the result; `"private"` keeps it to the caller. Register tools in a fixed order too: a deterministic `tools/list` is what lets both the client's cache and the model's prompt cache hit.

### Tasks, as an extension

Long-running work (a CI run, a batch job, a human approval) no longer blocks the call. With the [Tasks extension](https://github.com/modelcontextprotocol/ext-tasks) (`io.modelcontextprotocol/tasks`), a client that declares the extension in its capabilities can get back a task handle instead of a result, poll it with `tasks/get`, and send input with `tasks/update`. The Python SDK 2.2.0 ships the extension's capability plumbing but no task helpers, so the example server here does not use it.

## 🧩 Core Primitives

| Primitive | Offered by | Description |
|-----------|------------|-------------|
| **Tools** | Server | Functions the model can call (read/write). Hosts should keep a human in the loop before running them. |
| **Resources** | Server | File-like context the client can read (APIs, files, DB records). |
| **Prompts** | Server | Reusable prompt templates with parameters, surfaced in UIs. |
| **Elicitation** | Client | The user answers a server's question. Asked through an `InputRequiredResult`, never a mid-call server request. |

Sampling (a server asking the host's model for a completion) and Roots still exist but are deprecated in `2026-07-28`. **Tools** are the most commonly used primitive and the focus of this module.

## 🚀 Building an MCP Server

### Installation

```bash
pip install -r requirements.txt   # mcp>=2.2: the SDK that speaks 2026-07-28
```

The SDK's 2.x line renamed `FastMCP` to `MCPServer` (see the [v1 → v2 migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/)). The decorators are the same.

### Minimal Server (stdio transport)

```python
# servers/weather_server.py
from mcp.server.mcpserver import MCPServer
import httpx

mcp = MCPServer("weather-service")

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
    # mcp.run("streamable-http", port=8080)  # HTTP transport
```

On stdio, stdout is the protocol stream: log to `stderr`, never `print()` to stdout.

### Running the Server

```bash
# stdio (used by Claude Desktop, Claude Code)
python servers/example_server.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector python servers/example_server.py

# HTTP transport (for remote/hosted servers)
python servers/example_server.py --transport streamable-http --port 8080
```

### Talking to it from Python

The SDK's `Client` connects to a URL, a stdio command, or — for tests — a server object in the same process. It probes `server/discover` and speaks `2026-07-28` when the server does:

```python
import asyncio
from mcp import Client, StdioServerParameters

async def main():
    server = StdioServerParameters(command="python", args=["servers/example_server.py"])
    async with Client(server) as client:
        print(client.protocol_version)              # 2026-07-28
        tools = await client.list_tools()           # cached for ttlMs
        print([t.name for t in tools.tools])
        result = await client.call_tool("notes_search", {"query": "welcome"})
        print(result.content[0].text)

asyncio.run(main())
```

`python mcp_example.py` (demo 6) runs this against the example server in process and prints the `InputRequiredResult` round trip.

### Registering with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "python",
      "args": ["/path/to/servers/example_server.py"]
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
- [ ] Provide `outputSchema` for structured return values (any JSON Schema 2020-12 keywords are allowed in `2026-07-28`)

### 3. Error Handling — Two Layers

MCP distinguishes between two kinds of errors:

```python
# Layer 1: Protocol errors (JSON-RPC level)
# — Unknown tool, invalid arguments, unsupported protocol version
# — The SDK returns these; you don't handle them in tool code

# Layer 2: Tool execution errors (business logic)
# — API failures, bad inputs, rate limits, timeouts
# — Returned as a result with isError=True and an actionable message the LLM can act on.
#   In MCPServer, raise ToolError: its message becomes the isError result.

from mcp.server.mcpserver.exceptions import ToolError

@mcp.tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via the configured SMTP server."""
    try:
        result = await smtp_client.send(to=to, subject=subject, body=body)
        return f"Email sent. Message ID: {result.id}"
    except InvalidAddressError:
        # Actionable error — the LLM can ask the user to fix the address
        raise ToolError(
            f"Invalid email address: '{to}'. "
            "Please provide a valid email address (e.g. name@domain.com)."
        )
    except RateLimitError as e:
        # Retriable error with wait hint
        raise ToolError(f"Rate limit reached. Wait {e.retry_after_seconds}s and retry.")
    except Exception as e:
        # Generic — don't expose internals, but give enough to debug
        raise ToolError(f"Email send failed: {type(e).__name__}. Try again or check credentials.")
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
    """Delete a specific email — irreversible, asks the user to confirm."""
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
- A missing resource is now JSON-RPC `-32602` (Invalid Params)

---

## 🔐 Secure MCP Tunnels

For production deployments, expose MCP servers securely without making them public. OpenAI's [Secure MCP Tunnels](https://platform.openai.com/docs/guides/secure-mcp-tunnels) connect a private server to ChatGPT without exposing it to the public internet.

**Why tunnels matter**:
- MCP servers often need access to internal databases, APIs, and services
- Direct exposure creates security risks
- Tunnels provide authenticated, encrypted access without public endpoints

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
appendix/mcp/
├── README.md
├── requirements.txt
├── mcp_example.py               ★ Tool-design demos + the 2026-07-28 wire in process
├── mcp_example.ipynb
├── servers/
│   ├── __init__.py
│   └── example_server.py        ★ Complete MCP server (stdio or streamable HTTP)
└── tools/
    ├── __init__.py
    └── tool_design.py           Tool design helpers and validators
```

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Agent doesn't call the right tool | Rewrite description with clear trigger conditions |
| Tool arguments are wrong format | Add examples and format hints to parameter descriptions |
| `No module named 'mcp.server.fastmcp'` | You have the 2.x SDK: import `MCPServer` from `mcp.server.mcpserver` <!-- historical-exception --> |
| Client gets `UnsupportedProtocolVersionError` | The two sides share no protocol version: call `server/discover` and pick one from `supportedVersions` |
| stdio client fails to parse the server's messages | Something printed to stdout; log to `stderr` instead |
| Error messages not actionable | Include what failed, why, and what to do next |

## 🧪 Hands-On Exercises

1. **Build a File System Server**: Create an MCP server with 4 tools: `file_read`, `file_write`, `file_list`, and `file_delete`. Apply the split/consolidate principle — should read and write be one tool or two? Make `file_delete` ask for confirmation through an `InputRequiredResult`.

2. **Description Quality Test**: Take any 3 tools from the official MCP server repo. Rewrite their descriptions to include: (a) when to use, (b) what it returns, (c) what it doesn't handle. Call the same agent task with old vs new descriptions. Does routing improve?

3. **Schema Ambiguity Hunt**: Build a tool for scheduling meetings with a free-form `time` parameter (e.g., "next Tuesday at 3pm"). Have an agent call it 20 times. How often does the agent pass invalid formats? Now add a strict `start_datetime: str  # ISO 8601` with an example. Does error rate drop?

4. **Error Recovery Test**: Build a tool that raises 3 different `ToolError`s. Write actionable error messages for each. Have an agent encounter each error and observe whether it: (a) retries with a fix, (b) tries an alternative approach, (c) gives up. Compare with and without actionable messages.

5. **Cache hints**: Give `tools/list` a `ttlMs` of 5 minutes, then call `list_tools()` twice from one `Client`. How many requests reach the server? What changes when you set `cache_mode` or `cache=None` on the client?

6. **Resource vs Tool**: Design a database access pattern. Which operations should be Resources (schema discovery, config) and which should be Tools (insert, update, query)? Build both and compare how an agent uses them.

---

## 📚 References

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — the current spec
- [What changed in 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog) — the spec's changelog
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr) — `InputRequiredResult` in full
- [Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover) — `server/discover`
- [Tasks extension](https://github.com/modelcontextprotocol/ext-tasks) — `io.modelcontextprotocol/tasks`
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — `MCPServer` and `Client` ([v1 → v2 migration](https://py.sdk.modelcontextprotocol.io/v2/migration/))
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — dev tool for testing servers
- [MCP Registry](https://registry.modelcontextprotocol.io) — discover community servers
- [Official Reference Servers](https://github.com/modelcontextprotocol/servers) — filesystem, git, GitHub, PostgreSQL, etc.
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — ACI design principles
- [OpenAI Secure MCP Tunnels](https://platform.openai.com/docs/guides/secure-mcp-tunnels) — production MCP security
- [MCPEvol-Bench](https://arxiv.org/abs/2607.14642) — benchmarking LLM agents across MCP server evolutions

## 🔗 Where this meets the Spine

- **[Spine 1 · Loop](../../spine/01-loop/README.md)**: the Flagship Agent's Actions are tools; the description and schema rules here are why they route
- **[Spine 1 · Authorization belongs in the Action](../../spine/01-loop/README.md#authorization-belongs-in-the-action)**: validation and safety belong in the server, not the prompt
- **[Agent frameworks](../agent-frameworks/README.md)**: every framework there can consume MCP servers as tools

---

**Good tools make agents more capable. Bad tools make them unpredictable. Design matters.**


---

## Agent-to-Agent (A2A) Protocol

While MCP connects agents to **tools**, the **Agent2Agent (A2A) protocol** enables agents to communicate with **each other**.

### MCP vs A2A

| | MCP | A2A |
|--|-----|-----|
| **Connects** | Agent ↔ Tool | Agent ↔ Agent |
| **Direction** | Agent calls tool | Agents collaborate |
| **Use case** | Database query, API call | Multi-agent orchestration |
| **Originated at** | Anthropic | Google |

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
- [A2A Python SDK](https://github.com/a2aproject/a2a-python) — Python SDK
- [A2A + MCP Integration](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — Google blog post
