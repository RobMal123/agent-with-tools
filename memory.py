"""
memory.py — Long-term memory helpers for the agent.

Facts are stored as a flat JSON dict in memory.json (project root).
This module is imported by:
  - tools.py   (save_memory / list_memories tools)
  - graph.py   (format_memories_for_prompt — injected into every system message)
  - app.py     (sidebar display + per-key delete)
"""

import json
import os

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(_BASE_DIR, "memory.json")


# ── CRUD ───────────────────────────────────────────────────────────────────────

def load_memories() -> dict:
    """Return all stored memories as an ordered dict (insertion order)."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_memory_entry(key: str, value: str) -> None:
    """Upsert a single fact."""
    memories = load_memories()
    memories[key] = value
    _write(memories)


def delete_memory_entry(key: str) -> bool:
    """Delete a fact by key. Returns True if it existed."""
    memories = load_memories()
    if key not in memories:
        return False
    del memories[key]
    _write(memories)
    return True


def clear_all_memories() -> int:
    """Delete every stored memory. Returns the number that were cleared."""
    memories = load_memories()
    count = len(memories)
    _write({})
    return count


# ── Internal ───────────────────────────────────────────────────────────────────

def _write(memories: dict) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


# ── Prompt injection ───────────────────────────────────────────────────────────

def format_memories_for_prompt() -> str:
    """
    Return a formatted block for injection into the system prompt.
    Returns an empty string when no memories exist.
    """
    memories = load_memories()
    if not memories:
        return ""
    lines = [f"- {k}: {v}" for k, v in memories.items()]
    return (
        "## What you remember about the user\n"
        "(These facts are already known to you — answer any recall question directly from them. "
        "Do NOT call a tool to read or list a fact you can already see here.)\n"
        + "\n".join(lines)
    )
