"""
Complete MCP server example demonstrating:
- Tools (read + write, with proper error handling)
- Resources (schema, config)
- Prompt templates
- Correct naming, description, and schema design

Run with:
    python servers/example_server.py           (stdio)
    mcp dev servers/example_server.py          (inspector UI)
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# FastMCP requires: pip install mcp
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("This file shows the structure — run after installing mcp.\n")

if HAS_MCP:
    mcp = FastMCP("knowledge-base-server")

    # ─── In-memory SQLite for demo ────────────────────────────────────────────
    DB_PATH = ":memory:"
    _db = sqlite3.connect(DB_PATH, check_same_thread=False)
    _db.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    _db.execute("INSERT INTO notes (title, content, tags) VALUES (?,?,?)",
                ("Welcome", "This is the first note.", "intro,example"))
    _db.commit()

    # ─── TOOLS ───────────────────────────────────────────────────────────────

    @mcp.tool()
    def notes_search(
        query: str,
        limit: int = 5,
        tag: Optional[str] = None,
    ) -> str:
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
        try:
            limit = max(1, min(20, limit))  # clamp to safe range
            if tag:
                rows = _db.execute(
                    "SELECT id, title, content FROM notes "
                    "WHERE (title LIKE ? OR content LIKE ?) AND tags LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", f"%{tag}%", limit)
                ).fetchall()
            else:
                rows = _db.execute(
                    "SELECT id, title, content FROM notes "
                    "WHERE title LIKE ? OR content LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit)
                ).fetchall()

            if not rows:
                return f"No notes found matching '{query}'" + (f" with tag '{tag}'" if tag else "")

            results = []
            for row in rows:
                excerpt = row[2][:120] + ("..." if len(row[2]) > 120 else "")
                results.append(f"[ID:{row[0]}] {row[1]}\n  {excerpt}")
            return f"Found {len(results)} note(s):\n\n" + "\n\n".join(results)

        except Exception as e:
            return f"Search failed: {type(e).__name__}. Try a simpler query."

    @mcp.tool()
    def notes_create(
        title: str,
        content: str,
        tags: str = "",
    ) -> str:
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

        cursor = _db.execute(
            "INSERT INTO notes (title, content, tags) VALUES (?,?,?)",
            (title.strip(), content.strip(), tags.strip())
        )
        _db.commit()
        return f"Note created successfully. ID: {cursor.lastrowid}"

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
        if not any([title, content, tags]):
            return "Error: provide at least one of title, content, or tags to update."

        row = _db.execute("SELECT id FROM notes WHERE id=?", (note_id,)).fetchone()
        if not row:
            return (f"Note ID {note_id} not found. "
                    "Use notes_search to find the correct ID first.")

        updates, vals = [], []
        if title is not None:
            updates.append("title=?"); vals.append(title)
        if content is not None:
            updates.append("content=?"); vals.append(content)
        if tags is not None:
            updates.append("tags=?"); vals.append(tags)
        vals.append(note_id)

        _db.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id=?", vals)
        _db.commit()
        return f"Note {note_id} updated successfully."

    @mcp.tool()
    def notes_list(limit: int = 10) -> str:
        """
        List recent notes with titles and IDs.

        Use when the user wants to browse or see all notes, or when they ask
        'what notes do I have?' Returns up to `limit` most recent notes.

        For keyword search, use notes_search instead.
        """
        rows = _db.execute(
            "SELECT id, title, created_at FROM notes ORDER BY id DESC LIMIT ?",
            (min(50, limit),)
        ).fetchall()
        if not rows:
            return "No notes found. Create one with notes_create."
        lines = [f"[ID:{r[0]}] {r[1]} ({r[2][:10]})" for r in rows]
        return f"{len(rows)} notes:\n" + "\n".join(lines)

    # ─── RESOURCES ───────────────────────────────────────────────────────────

    @mcp.resource("db://notes/schema")
    def notes_schema() -> str:
        """
        Database schema for the notes table.
        Include in context before writing SQL queries or explaining data structure.
        """
        return json.dumps({
            "table": "notes",
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True, "auto": True},
                {"name": "title",      "type": "TEXT",    "required": True},
                {"name": "content",    "type": "TEXT",    "required": True},
                {"name": "tags",       "type": "TEXT",    "default": ""},
                {"name": "created_at", "type": "TEXT",    "default": "datetime('now')"},
            ]
        }, indent=2)

    @mcp.resource("config://server")
    def server_config() -> str:
        """Current server configuration (read-only)."""
        return json.dumps({
            "server_name": "knowledge-base-server",
            "version": "1.0.0",
            "mcp_spec": "2025-06-18",
            "capabilities": ["tools", "resources", "prompts"],
            "max_results_per_query": 20,
        }, indent=2)

    # ─── PROMPT TEMPLATES ────────────────────────────────────────────────────

    @mcp.prompt()
    def note_summary_prompt(note_id: int) -> str:
        """Generate a concise summary of a specific note."""
        row = _db.execute("SELECT title, content FROM notes WHERE id=?", (note_id,)).fetchone()
        if not row:
            return f"Note {note_id} not found."
        return (
            f"Summarize the following note in 2-3 bullet points. "
            f"Be concise and factual.\n\nTitle: {row[0]}\n\nContent:\n{row[1]}"
        )

    if __name__ == "__main__":
        print("Starting knowledge-base MCP server...")
        print("Connect via: mcp dev servers/example_server.py")
        mcp.run()

else:
    # Fallback: show structure without mcp package
    print("""
MCP Server Structure (install 'mcp' package to run):

  Tools:
    notes_search(query, limit, tag)     → search knowledge base
    notes_create(title, content, tags)  → add a new note
    notes_update(note_id, ...)          → edit existing note
    notes_list(limit)                   → browse recent notes

  Resources:
    db://notes/schema                   → table schema for SQL context
    config://server                     → server configuration

  Prompts:
    note_summary_prompt(note_id)        → summarize a specific note

Install and run:
  pip install mcp
  python servers/example_server.py
""")
