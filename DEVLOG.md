# Dev Log — 2026-05-27

## What we built today

Starting point: a LangGraph agent template using **OpenAI GPT-4o-mini** and **Tavily Search**.  
End state: a fully local, privacy-first research and meeting assistant.

---

## Session summary

### 1. Switched to local stack (Ollama + Brave Search)

- Replaced `ChatOpenAI` → `ChatOllama` (`langchain-ollama`) with `llama3.2` as default model
- Replaced Tavily → **Brave Search API** (`langchain_community.utilities.brave_search`)
- Wrapped `BraveSearch` in a `@tool`-decorated function to fix an Ollama schema mismatch
  where the model was passing `value=` instead of `query=`
- Updated `requirements.txt`, `.env.example`, and all import paths
- Fixed a `ModuleNotFoundError` caused by the project folder being named `files` instead of
  `agent` — converted all relative package imports to flat absolute imports
- Added `load_dotenv()` before any module imports so `.env` is read before tool init

### 2. Added chat persistence + model selector

- **Saved chats**: every conversation auto-saves to `chats/<thread_id>.json` after each
  exchange; the sidebar lists all chats (newest first) with Load / Delete buttons
- Loading a chat restores display history *and* re-seeds LangGraph's `MemorySaver` so
  follow-up questions have context
- **Model selector**: queries `http://localhost:11434/api/tags` on startup; switching models
  rebuilds the agent and saves the current chat first
- **Refresh model list** button picks up newly pulled Ollama models without restarting

### 3. Added Markdown file tool

- `write_md_file(path, content)` — creates or overwrites any `.md` file; parent directories
  are created automatically
- Combined with the existing `read_file` tool, the agent can now maintain a personal
  knowledge base of notes and reports

### 4. Added local audio transcription

- `transcribe_audio(file_path)` — transcribes any audio file using **faster-whisper**
  (local, CPU-friendly, int8 quantised; ~4× faster than openai-whisper)
- Detected language is logged; raw transcript saved to `transcriptions/<stem>.md`
- Whisper model is lazy-loaded and cached in memory so subsequent calls are instant
- Model size is configurable via `WHISPER_MODEL` env var (default: `small`, ~970 MB)
- Requires **ffmpeg** on PATH for `.m4a` / `.mp3` decoding
- Sidebar **🎙️ Transcription** section lists all files in `audio_in/` with status dots:
  🔴 nothing · 🔵 transcript only · ✅ transcript + summary
- One-click **▶ Run** button triggers transcription + meeting summary generation
- Meeting summary is **auto-saved by the app** (not relying on the model to call
  `write_md_file`) — this was a key reliability fix for small local models that
  sometimes skip chained tool calls

### 5. Added directory browser tool

- `list_directory(path)` — lists files and folders with size and last-modified date
- Paths are resolved relative to the project root so short paths like `"transcriptions"`
  just work
- Enables the agent to discover and read files autonomously, e.g. finding a
  meeting summary and answering questions about it

---

## Final tool set

| Tool | Purpose |
|------|---------|
| `brave_search` | Web search |
| `python_repl` | Local code execution |
| `list_directory` | Browse the file system |
| `read_file` | Read text files |
| `write_md_file` | Create / update Markdown files |
| `transcribe_audio` | Local speech-to-text via Whisper |

## Stack

| Component | Choice |
|-----------|--------|
| LLM | Ollama `llama3.2` (local) |
| Agent framework | LangGraph |
| Search | Brave Search API |
| Transcription | faster-whisper (local) |
| UI | Streamlit |
| API | FastAPI |
