# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Run the Streamlit UI
streamlit run app.py

# Run the FastAPI server
uvicorn main:app --reload

# Run tests (real LLM calls — see Testing section)
pytest test_agent.py -v

# Run a single test
pytest test_agent.py::test_vision_distinguishes_colors -v
```

All commands must be run from the `files/` directory. The virtual environment is in `venv/`.

The test suite makes **real Ollama calls** and needs these models pulled:
`ollama pull llama3.2` · `ollama pull gemma3:4b` · `ollama pull nomic-embed-text`.
Tests skip cleanly (not fail) when Ollama is down or a model is missing.

## Architecture

This is a **LangGraph ReAct agent** with a Streamlit UI and FastAPI backend. The core loop is:

```
START → call_model → (tool calls?) → call_tools → call_model → ... → END
```

`graph.py` builds and compiles this loop. `build_agent(model_name, use_memory=True, vision_model=None)` is the factory — it calls `set_agent_model()`, creates a fresh `ChatOllama` LLM, binds the TOOLS list, builds a separate vision LLM, wires up the graph nodes, and attaches a `MemorySaver` checkpointer for per-thread conversation history. The global `agent = build_agent()` at the bottom of `graph.py` is used by `main.py`; `app.py` manages its own agent instance in `st.session_state.agent` so the model can be switched at runtime.

**`call_model` injects memory + date + routes vision per turn:**
- Long-term memories (`format_memories_for_prompt()`) are loaded fresh on every call and appended to `SYSTEM_PROMPT`, so the LLM always sees the latest facts without a rebuild.
- **The current date** (`datetime.now()`) is injected fresh on every call, with a nudge to use `brave_search` (and its `freshness` arg) and trust search results over its training cutoff for time-sensitive topics. Without this the model doesn't know "today" and answers recent questions from stale training data.
- If the latest human message contains an `image_url` content part, the turn is routed to the **dedicated vision model** with a short `VISION_SYSTEM_PROMPT` and **no tools** (see Vision section). All other turns use the primary model with the full system prompt + tools.

**Import structure:** all files use flat absolute imports (e.g. `from graph import agent`). The project is run as a script directory, not an installed package — `app.py` and `main.py` both prepend `sys.path.insert(0, dirname(__file__))`, and `conftest.py` does the same for pytest. Relative imports (`.state`, `.tools`) do **not** work here. **Do not add an `__init__.py`** to `files/` — it flips pytest into package-import mode and breaks `pytest test_agent.py` with `ModuleNotFoundError: graph` (one was removed for exactly this reason).

**Tool-calling quirks (small Ollama models):**
- Single-string args are passed as `value=` rather than a named parameter. Define every tool with `@tool` and an explicit parameter name (e.g. `query: str`), not inherited from a pre-built class. See `brave_search` as the reference.
- **Nested dict / object params are unreliable.** Use flat string params instead (e.g. `doc_type: str = ""`, not `metadata: dict`). All v2 tools follow this.
- Small models reliably do **one** tool call per turn but often skip a chained second call — orchestrate multi-step flows app-side (see Meeting summary flow).

**Env vars are read at import time** — `tools.py` reads `BRAVE_SEARCH_API_KEY` when `BraveSearchWrapper` is instantiated at module level. `load_dotenv()` must run in `app.py` / `main.py` *before* importing `graph` or `tools`, otherwise keys are empty.

## Key files

| File | Role |
|------|------|
| `graph.py` | Agent graph, `build_agent()` factory, memory injection + vision routing |
| `tools.py` | All 14 tool definitions + `TOOLS` list, secondary-LLM + vectorstore helpers |
| `state.py` | `AgentState` TypedDict — just `messages` with `add_messages` reducer |
| `prompts.py` | `SYSTEM_PROMPT` (full, tool-aware) + `VISION_SYSTEM_PROMPT` (short, image turns) |
| `memory.py` | Long-term memory CRUD over `memory.json` + `format_memories_for_prompt()` |
| `app.py` | Streamlit UI: model selector, chat persistence, image upload, transcription, memory + knowledge sidebars |
| `conftest.py` | Puts `files/` on `sys.path` so `pytest` resolves sibling modules |

## Tools

`TOOLS` (13): `brave_search`, `python_repl`, `list_directory`, `read_file`, `write_md_file`, `transcribe_audio`, `save_memory`, `list_memories`, `index_document`, `search_documents`, `structure_thoughts`, `log_improvement`, `analyze_improvements`.

Adding a new tool: define it with `@tool` in `tools.py`, append it to `TOOLS` at the bottom. The agent picks it up automatically — no changes to `graph.py` needed.

- **`brave_search`** returns up to **5** results as a readable numbered `[n] title / snippet / Source: <url>` list (not raw JSON), so the model synthesises instead of pasting. An optional `freshness` arg (`pd`/`pw`/`pm`/`py` = past day/week/month/year) maps to Brave's recency filter for time-sensitive queries; default off so evergreen searches are unaffected. It degrades gracefully: if `BRAVE_SEARCH_API_KEY` is unset or the request fails, it returns an `Error: …` string telling the model to answer from its own knowledge — it never raises (raising crashes the whole agent run). The non-tool helper `extract_search_sources()` parses that output back into `{title, url}` so `main.py` can attach clickable sources to the chat bubble.
- **`python_repl`** executes Python **in-process** (no real sandbox) and is **disabled by default** — set `ENABLE_CODE_EXECUTION=true` to enable it. The `os.chdir(workspace/)` prefix only sets the working dir so generated files land in `workspace/`; it is not a security boundary.
- **`transcribe_audio`** lazy-loads faster-whisper into `_whisper_cache` (~970 MB for `small`, downloads to `~/.cache/huggingface/hub/`). Saves transcripts to `knowledge/meetings/<stem>.md` and auto-indexes them. Set `WHISPER_MODEL=medium` for better non-English/noisy accuracy.
- **`structure_thoughts` / `analyze_improvements`** call a secondary `temperature=0` LLM via `_get_structure_llm()`. `build_agent()` calls `set_agent_model()` so this secondary LLM tracks the active model.

## FastAPI backend (`main.py`)

`main.py` serves the React UI in `../Research Companion/`. Notable behaviour:

- **Per-request model switching.** `_get_agent(model)` caches one compiled agent per model name (cheap — `ChatOllama` loads lazily on first invoke); `/chat` and `/chat/stream` accept an optional `model` field, and `/api/status` returns `default_model` (= `AGENT_MODEL`) for the UI to pre-select. Each agent has its own `MemorySaver`, which is fine because the UI starts a fresh conversation when the model changes (history is never shared across models). Single-user/localhost, so no locking around the cache.
- **Streaming.** `/chat/stream` uses `stream_mode=["messages", "updates"]`: it streams token deltas **only** from the `call_model` node (so the echoed prompt, raw tool output, and inner-tool LLMs never leak into the reply), then sends a final `{tools, sources}` SSE frame before `[DONE]`. The non-streaming `/chat` returns `tool_calls_made` + `sources` in its JSON. (The earlier `stream_mode="values"` emitted whole-state snapshots, which dumped the user's message + raw search results into the bubble — fixed.)
- **`/api/models`** excludes embedding models (any name containing `embed`) so they aren't offered as chat models.

## Long-term memory

`memory.py` stores facts as a flat JSON dict in `memory.json` (project root). `save_memory` / `list_memories` are agent tools (deletion is intentionally **not** an agent tool — it's exposed only via the UI / `DELETE /api/memory/{key}` → `delete_memory_entry`); `format_memories_for_prompt()` is injected into the system prompt on every `call_model`. The app sidebar (🧠 Memory) shows entries with per-key delete + "Clear all". `memory.json` is gitignored.

## Semantic search (RAG)

ChromaDB (local, embedded, in `chroma_db/`) + Ollama embeddings (`nomic-embed-text`, override with `EMBED_MODEL`). The store is lazy-loaded and cached in `tools._vectorstore_cache`; the persist dir is `tools._CHROMA_DIR`.

- **`index_document`** / **`_index_file`** chunk with `RecursiveCharacterTextSplitter` (800/100) and upsert with metadata `{source, type, date, indexed_at}`. `type` is inferred from the parent folder (`meetings/`→`meeting`, etc.); `date` from a `YYYY-MM-DD` filename pattern or mtime. Re-indexing the same `source` deletes stale chunks first.
- **`search_documents(query, doc_type="", top_k=4)`** does similarity search with an optional `{"type": {"$eq": doc_type}}` filter.
- Files created by `transcribe_audio` / `structure_thoughts` / `log_improvement` auto-index via `_auto_index` (errors swallowed).

## Vision / multimodal

Ollama vision models can't process an image when **tools** are bound or a **long system prompt** precedes it — and the primary models here can't do vision at all (`llama3.2` is text-only; the custom `gemma4` build advertises a `vision` capability that doesn't actually work in Ollama). So image turns are routed to a **dedicated vision model**.

- `build_agent(..., vision_model=None)` → defaults to `VISION_MODEL` env or `gemma3:4b`. `gemma3` has working vision in Ollama but **does not support tools** — fine, image turns don't need them.
- `call_model` detects an `image_url` part in the latest human message and invokes the plain vision LLM with `VISION_SYSTEM_PROMPT` only. Tool use + full prompt resume on the next text turn. The two models swap in/out of VRAM as you alternate (a few seconds' load latency — normal).
- `app.py` builds the multimodal `HumanMessage` (content list with a `data:` URL). Because `st.file_uploader` can return `None` on the `st.chat_input` rerun, the uploaded image bytes are stashed in `st.session_state["_queued_img"]` and consumed on submit.

## Data directories

| Directory / file | Contents | Committed |
|------------------|----------|-----------|
| `audio_in/` | Source audio for transcription | No (`.gitkeep`) |
| `knowledge/meetings/` | Transcripts (`<stem>.md`) + summaries (`<stem>_summary.md`) + extracted JSON | No (`.gitkeep`) |
| `knowledge/ideas/` | `structure_thoughts` output | No (`.gitkeep`) |
| `knowledge/projects/` `reports/` | Project plans / reports | No (`.gitkeep`) |
| `knowledge/improvements/` | `log_improvement` entries | No (`.gitkeep`) |
| `workspace/` | Files written by `python_repl` | No (`.gitkeep`) |
| `chroma_db/` | ChromaDB vector store | No |
| `memory.json` | Long-term memory | No |
| `chats/` | Saved conversation JSON (`<thread_id>.json`) | No |
| `transcriptions/`, `reports/` | Legacy paths — still read for backward compat | No |

## Environment variables

| Var | Purpose | Default |
|-----|---------|---------|
| `BRAVE_SEARCH_API_KEY` | `brave_search` (degrades gracefully if unset) | — |
| `OLLAMA_BASE_URL` | Ollama host | `http://localhost:11434` |
| `AGENT_MODEL` | Primary reasoning model (text + tools); the UI can override it per request | `gemma4:e4b` |
| `VISION_MODEL` | Dedicated vision model for image turns | `gemma3:4b` |
| `EMBED_MODEL` | RAG embedding model | `nomic-embed-text` |
| `WHISPER_MODEL` | faster-whisper size | `small` |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | LangSmith tracing | off |

## Testing

`test_agent.py` uses **real LLM calls — no mocks of the model.** Guiding principle: a verification that can't produce a *wrong* answer when the system is broken isn't a verification. Each LLM test asserts on an outcome impossible to fake unless the real machinery works:

- **tool calling** → a 9-digit product no small model can do in its head (proves the REPL path)
- **vision** → reads specific digits / names red-green-blue (the regression guard for the vision saga; a blind model answering "Black" to everything fails it)
- **memory** → recalls an unguessable codeword (proves prompt injection)
- **RAG** → retrieves a unique invented fact (proves embeddings + vector search)

Tests `skipif` Ollama is down or a model isn't pulled. Fixtures isolate side effects: `preserve_memory` snapshots/restores `memory.json`; `temp_vectorstore` points ChromaDB at a throwaway dir. That's isolation, not mocking — the real LLM/embeddings/vector search still run.

**Verify with the exact command the user runs** (`pytest …`, not just `python -m pytest …`) — they differ in `sys.path` handling.

## LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=lsv2_...`, `LANGCHAIN_PROJECT=ai-research-agent` in `.env` — no code changes. LangChain/LangGraph auto-detect them because `load_dotenv()` runs before any LangChain import. Every `agent.invoke()` produces a trace of the full ReAct loop. The sidebar shows a green `🔍 LangSmith tracing · <project>` badge when active.

## Meeting summary flow

The sidebar **▶ Run** button sets `st.session_state.auto_save_summary` and sends a single-step prompt (transcribe + summarise). After the agent responds, `app.py` checks that flag and itself: saves the reply to `knowledge/meetings/<stem>_summary.md`, indexes it, runs a second LLM pass (`_extract_meeting_json`) to pull `tasks` / `decisions` / `people` into `<stem>_data.json`. This app-side orchestration bypasses the model's unreliable chained tool calls — small models reliably complete one tool call per turn but skip the second.
