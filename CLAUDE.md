# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Run the Streamlit UI
streamlit run app.py

# Run the FastAPI server
uvicorn main:app --reload

# Run tests
pytest test_agent.py -v

# Run a single test
pytest test_agent.py::test_read_file_success -v
```

All commands must be run from the `files/` directory. The virtual environment is in `venv/`.

## Architecture

This is a **LangGraph ReAct agent** with a Streamlit UI and FastAPI backend. The core loop is:

```
START → call_model → (tool calls?) → call_tools → call_model → ... → END
```

`graph.py` builds and compiles this loop. `build_agent(model_name)` is the factory — it creates a fresh `ChatOllama` LLM, binds the TOOLS list to it, wires up the graph nodes, and attaches a `MemorySaver` checkpointer for per-thread conversation history. The global `agent = build_agent()` at the bottom of `graph.py` is used by `main.py`; `app.py` manages its own agent instance in `st.session_state.agent` so the model can be switched at runtime.

**Import structure:** all files use flat absolute imports (e.g. `from graph import agent`). The project is run as a script directory, not an installed package — `app.py` and `main.py` both prepend `sys.path.insert(0, dirname(__file__))` so sibling modules resolve correctly. Relative imports (`.state`, `.tools`) do **not** work here.

**Tool calling quirk:** Ollama models pass single-string tool arguments as `value=` rather than a named parameter. All tools that take a single string input must be defined with `@tool` and an explicit parameter name (e.g. `query: str`, not inherited from a pre-built class like `TavilySearchResults`). See `brave_search` in `tools.py` as the reference pattern.

**Env vars are read at import time** — `tools.py` reads `BRAVE_SEARCH_API_KEY` when `BraveSearchWrapper` is instantiated at module level. `load_dotenv()` must be called in `app.py` / `main.py` *before* importing `graph` or `tools`, otherwise the key is empty.

## Key files

| File | Role |
|------|------|
| `graph.py` | Agent graph, `build_agent()` factory |
| `tools.py` | All tool definitions + `TOOLS` list |
| `state.py` | `AgentState` TypedDict — just `messages` with `add_messages` reducer |
| `prompts.py` | `SYSTEM_PROMPT` — update here to change agent behaviour |
| `app.py` | Streamlit UI: model selector, chat persistence, transcription sidebar |

## Tools

Adding a new tool: define it with `@tool` in `tools.py`, append it to the `TOOLS` list at the bottom. The agent picks it up automatically — no changes to `graph.py` needed.

The `transcribe_audio` tool lazy-loads the faster-whisper model into `_whisper_cache` on first call. The Whisper model downloads to `~/.cache/huggingface/hub/` (~970 MB for `small`). Set `WHISPER_MODEL=medium` in `.env` for better accuracy on non-English or noisy audio.

## Data directories

| Directory | Contents | Committed |
|-----------|----------|-----------|
| `audio_in/` | Source audio files for transcription | No (`.gitkeep` only) |
| `transcriptions/` | Raw transcripts (`<stem>.md`) + meeting summaries (`<stem>_summary.md`) | No |
| `chats/` | Saved conversation JSON (`<thread_id>.json`) | No |
| `reports/` | Any `.md` reports written by the agent | No |

## LangSmith tracing

Set these three vars in `.env` to enable — no code changes required:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=ai-research-agent
```

LangChain/LangGraph auto-detect them because `load_dotenv()` runs before any LangChain import. Every `agent.invoke()` call produces a trace in the LangSmith UI showing the full ReAct loop: LLM calls, tool inputs/outputs, latency, and token counts. The sidebar shows a green `🔍 LangSmith tracing · <project>` badge when active.

## Meeting summary flow

The sidebar **▶ Run** button sets `st.session_state.auto_save_summary` and sends a single-step prompt (transcribe + summarise). After the agent responds, `app.py` checks that flag and saves the reply to `transcriptions/<stem>_summary.md` itself — rather than asking the model to call `write_md_file`. This bypass exists because small local models reliably complete one tool call per turn but often skip a chained second call.
