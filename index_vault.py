"""
index_vault.py — bulk-index the configured Obsidian vault into semantic search.

Run once after connecting a vault (and any time you want to refresh the index):

    python index_vault.py

Reads OBSIDIAN_VAULT from .env, creates the agent's "AI Assistant" home note, then
indexes every Markdown note in the vault (your existing notes + the agent's) into
ChromaDB so search_documents can recall them. Safe to re-run — stale chunks are
replaced. Requires Ollama running with the embedding model pulled
(ollama pull nomic-embed-text).
"""

from dotenv import load_dotenv

load_dotenv()  # must run before importing tools so OBSIDIAN_VAULT is read

import tools

if not tools.VAULT_ENABLED:
    raise SystemExit(
        "No Obsidian vault configured. Set OBSIDIAN_VAULT in .env to your vault path."
    )

print(f"Vault:       {tools.VAULT_DIR}")
print(f"Agent notes: {tools.KNOWLEDGE_DIR}")
print(f"Home note:   {tools.ensure_vault_home()}")
print("Indexing vault…")

result = tools.index_vault(verbose=True)

print(f"\nIndexed {result['indexed']} note(s), skipped {result['skipped']}.")
for err in result["errors"]:
    print(f"  ! {err}")
