"""
A complete MCP server for protocol version 2026-07-28, on the current Python SDK (mcp 2.x).

It shows:
- Tools (read and write, with actionable errors), registered in a fixed order
- A destructive tool that asks the user to confirm through an InputRequiredResult
- Resources (a schema and a config) and a prompt template
- Cache hints (ttlMs / cacheScope) on the list results

What 2026-07-28 changes, and where this file meets it:
- There is no initialize handshake and no protocol session. Every request carries its
  protocol version and client capabilities in `_meta`, and a client may call
  `server/discover` first. The SDK answers both; nothing here keeps per-connection
  state. The note IDs are the "server-minted handles" the spec asks for: a client
  passes them back as ordinary tool arguments.
- A server can no longer send `elicitation/create` in the middle of a call. It returns
  an InputRequiredResult instead, and the client retries the call with its answer in
  `inputResponses`. `notes_delete` does this through the SDK's `Resolve(...)` /
  `Elicit(...)` resolver, which builds that result and resumes on the retry.
- `tools/list`, `prompts/list`, `resources/list`, `resources/templates/list` and
  `resources/read` results carry `ttlMs` and `cacheScope`, set from `CACHE_HINTS`.

Run it (pip install -r appendix/mcp/requirements.txt):
    python servers/example_server.py                              (stdio)
    python servers/example_server.py --transport streamable-http  (HTTP on :8080)
    npx @modelcontextprotocol/inspector python servers/example_server.py

Nothing is printed to stdout: on the stdio transport, stdout is the protocol stream.
"""

import argparse
import json
import sqlite3
import sys
from typing import Annotated, Optional

SPEC_VERSION = "2026-07-28"

try:
    from pydantic import BaseModel

    from mcp.server.caching import CacheHint
    from mcp.server.mcpserver import (
        AcceptedElicitation,
        Elicit,
        ElicitationResult,
        MCPServer,
        Resolve,
    )

    HAS_MCP = True
except ImportError:  # mcp 1.x has no MCPServer; mcp 2.x does
    HAS_MCP = False

STRUCTURE = """\
MCP server structure (install the MCP SDK to run it: pip install -r requirements.txt):

  Tools:
    notes_create(title, content, tags)  add a new note
    notes_delete(note_id)               delete a note, after the user confirms
    notes_list(limit)                   browse recent notes
    notes_search(query, limit, tag)     search the knowledge base
    notes_update(note_id, ...)          edit an existing note

  Resources:
    db://notes/schema                   table schema for SQL context
    config://server                     server configuration

  Prompts:
    note_summary_prompt(note_id)        summarize a specific note
"""

if HAS_MCP:

    #: The catalogue is the same for every caller and changes only on redeploy, so
    #: clients and shared caches may keep it for five minutes. Resource reads are
    #: private and short-lived: the safe default once a resource holds per-user data.
    CACHE_HINTS = {
        "tools/list": CacheHint(ttl_ms=300_000, scope="public"),
        "prompts/list": CacheHint(ttl_ms=300_000, scope="public"),
        "resources/list": CacheHint(ttl_ms=300_000, scope="public"),
        "resources/templates/list": CacheHint(ttl_ms=300_000, scope="public"),
        "resources/read": CacheHint(ttl_ms=30_000, scope="private"),
    }

    class DeleteConfirmation(BaseModel):
        confirm: bool

    def _open_db() -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute(
            """
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        db.execute(
            "INSERT INTO notes (title, content, tags) VALUES (?,?,?)",
            ("Welcome", "This is the first note.", "intro,example"),
        )
        db.commit()
        return db

    def build_server() -> MCPServer:
        """A fresh server over a fresh in-memory notes table."""
        db = _open_db()
        mcp = MCPServer("knowledge-base-server", version="2.0.0", cache_hints=CACHE_HINTS)

        # ─── TOOLS ─────────────────────────────────────────────────────────────
        # Registered in alphabetical order: the spec asks for a deterministic
        # tools/list, so a client's cache and the model's prompt cache both hit.

        @mcp.tool()
        def notes_create(title: str, content: str, tags: str = "") -> str:
            """
            Create a new note in the knowledge base.

            Use when the user asks to save, write, or add a note, memo, or piece of information.
            Returns the ID of the created note.

            Args:
                title: Short descriptive title for the note (required)
                content: Full note content (required)
                tags: Comma-separated tags (e.g. "meeting,project-x,2026"). Optional.
            """
            if not title.strip():
                return "Error: title cannot be empty. Please provide a descriptive title."
            if not content.strip():
                return "Error: content cannot be empty."
            cursor = db.execute(
                "INSERT INTO notes (title, content, tags) VALUES (?,?,?)",
                (title.strip(), content.strip(), tags.strip()),
            )
            db.commit()
            return f"Note created successfully. ID: {cursor.lastrowid}"

        def confirm_delete(note_id: int) -> Elicit[DeleteConfirmation]:
            """Ask the user before deleting. On 2026-07-28 the SDK turns this into an
            InputRequiredResult and resumes when the client retries with the answer."""
            return Elicit(f"Delete note {note_id}? This cannot be undone.", DeleteConfirmation)

        @mcp.tool()
        def notes_delete(
            note_id: int,
            confirmation: Annotated[ElicitationResult[DeleteConfirmation], Resolve(confirm_delete)],
        ) -> str:
            """
            Delete a note by ID. Irreversible: the user is asked to confirm first.

            Use when the user asks to delete or remove a specific note.
            Do NOT use if you don't have the note ID — search first with notes_search.
            """
            accepted = isinstance(confirmation, AcceptedElicitation) and confirmation.data.confirm
            if not accepted:
                return f"Note {note_id} not deleted: the user did not confirm."
            deleted = db.execute("DELETE FROM notes WHERE id=?", (note_id,)).rowcount
            db.commit()
            if not deleted:
                return f"Note ID {note_id} not found. Use notes_search to find the correct ID first."
            return f"Note {note_id} deleted."

        @mcp.tool()
        def notes_list(limit: int = 10) -> str:
            """
            List recent notes with titles and IDs.

            Use when the user wants to browse or see all notes, or when they ask
            'what notes do I have?' Returns up to `limit` most recent notes.

            For keyword search, use notes_search instead.
            """
            rows = db.execute(
                "SELECT id, title, created_at FROM notes ORDER BY id DESC LIMIT ?",
                (max(1, min(50, limit)),),
            ).fetchall()
            if not rows:
                return "No notes found. Create one with notes_create."
            lines = [f"[ID:{r[0]}] {r[1]} ({r[2][:10]})" for r in rows]
            return f"{len(rows)} notes:\n" + "\n".join(lines)

        @mcp.tool()
        def notes_search(query: str, limit: int = 5, tag: Optional[str] = None) -> str:
            """
            Search notes in the knowledge base by keyword or tag.

            Use when the user asks to find, look up, or retrieve notes about a topic.
            Returns up to `limit` matching notes with ID, title, and excerpt.

            Do NOT use for creating or editing notes (use notes_create or notes_update).
            Do NOT use for listing all notes (use notes_list instead).

            Args:
                query: Search terms (searched in title and content)
                limit: Maximum results to return (1-20, default 5)
                tag: Optional tag filter (e.g. "intro", "project-x")
            """
            limit = max(1, min(20, limit))  # clamp to a safe range
            sql = "SELECT id, title, content FROM notes WHERE (title LIKE ? OR content LIKE ?)"
            params: list = [f"%{query}%", f"%{query}%"]
            if tag:
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")
            rows = db.execute(sql + " LIMIT ?", (*params, limit)).fetchall()
            if not rows:
                return f"No notes found matching '{query}'" + (f" with tag '{tag}'" if tag else "")
            results = []
            for row in rows:
                excerpt = row[2][:120] + ("..." if len(row[2]) > 120 else "")
                results.append(f"[ID:{row[0]}] {row[1]}\n  {excerpt}")
            return f"Found {len(results)} note(s):\n\n" + "\n\n".join(results)

        @mcp.tool()
        def notes_update(
            note_id: int,
            content: Optional[str] = None,
            title: Optional[str] = None,
            tags: Optional[str] = None,
        ) -> str:
            """
            Update an existing note's title, content, or tags.

            Use when the user asks to edit, modify, or update a specific note by ID.
            At least one of title, content, or tags must be provided.
            Returns confirmation with the updated note ID.

            Do NOT use if you don't have the note ID — search first with notes_search.
            """
            fields = {"title": title, "content": content, "tags": tags}
            updates = {name: value for name, value in fields.items() if value is not None}
            if not updates:
                return "Error: provide at least one of title, content, or tags to update."
            if not db.execute("SELECT id FROM notes WHERE id=?", (note_id,)).fetchone():
                return f"Note ID {note_id} not found. Use notes_search to find the correct ID first."
            assignments = ", ".join(f"{name}=?" for name in updates)  # names from the fixed dict above
            db.execute(f"UPDATE notes SET {assignments} WHERE id=?", (*updates.values(), note_id))
            db.commit()
            return f"Note {note_id} updated successfully."

        # ─── RESOURCES ─────────────────────────────────────────────────────────

        @mcp.resource("db://notes/schema")
        def notes_schema() -> str:
            """
            Database schema for the notes table.
            Include in context before writing SQL queries or explaining data structure.
            """
            return json.dumps(
                {
                    "table": "notes",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True, "auto": True},
                        {"name": "title", "type": "TEXT", "required": True},
                        {"name": "content", "type": "TEXT", "required": True},
                        {"name": "tags", "type": "TEXT", "default": ""},
                        {"name": "created_at", "type": "TEXT", "default": "datetime('now')"},
                    ],
                },
                indent=2,
            )

        @mcp.resource("config://server")
        def server_config() -> str:
            """Current server configuration (read-only)."""
            return json.dumps(
                {
                    "server_name": "knowledge-base-server",
                    "version": "2.0.0",
                    "mcp_spec": SPEC_VERSION,
                    "capabilities": ["tools", "resources", "prompts"],
                    "max_results_per_query": 20,
                },
                indent=2,
            )

        # ─── PROMPT TEMPLATES ──────────────────────────────────────────────────

        @mcp.prompt()
        def note_summary_prompt(note_id: int) -> str:
            """Generate a concise summary of a specific note."""
            row = db.execute("SELECT title, content FROM notes WHERE id=?", (note_id,)).fetchone()
            if not row:
                return f"Note {note_id} not found."
            return (
                "Summarize the following note in 2-3 bullet points. "
                f"Be concise and factual.\n\nTitle: {row[0]}\n\nContent:\n{row[1]}"
            )

        return mcp


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge-base MCP server (MCP 2026-07-28)")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    if not HAS_MCP:
        # stderr, never stdout: stdout is the stdio protocol stream.
        print(STRUCTURE, file=sys.stderr)
        sys.exit("The MCP Python SDK 2.x is not installed: pip install -r appendix/mcp/requirements.txt")

    server = build_server()
    if args.transport == "stdio":
        server.run()
    else:
        print(f"Serving MCP {SPEC_VERSION} on http://localhost:{args.port}/mcp", file=sys.stderr)
        server.run("streamable-http", port=args.port)


if __name__ == "__main__":
    main()
