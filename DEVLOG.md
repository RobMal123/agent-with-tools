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

---

# Dev Log — 2026-06-01

## What we built today

Extended the agent with **persistent long-term memory**, **semantic document search (RAG)**,
**multimodal image input**, and **smarter audio file naming**.

---

## Session summary

### 1. Long-term memory across conversations

- New `memory.py` module — thin JSON store (`memory.json`) with `load / save / delete / clear`
  helpers and a `format_memories_for_prompt()` function
- Three new tools: `save_memory(key, value)`, `list_memories()`, `delete_memory(key)`
- Memories are injected into **every** system prompt inside `graph.py`'s `call_model` node,
  loaded fresh on each LLM call so changes take effect immediately without an agent rebuild
- System prompt updated to guide the model: save personal context proactively, call
  `list_memories` at the start of relevant topics
- Sidebar **🧠 Memory** section shows all stored facts with per-key 🗑 delete buttons,
  a **Clear all** button, and an **Index docs** button (see RAG below)

### 2. Semantic search / RAG over local documents

- **ChromaDB** (embedded, no server) as the local vector store, persisted to `chroma_db/`
- **Ollama embeddings** via `langchain-ollama` (`OllamaEmbeddings`) — default model:
  `nomic-embed-text` (configurable via `EMBED_MODEL` env var)
- `_index_file(abs_path)` helper: reads a file → chunks with `RecursiveCharacterTextSplitter`
  (800 chars, 100 overlap) → deletes stale chunks for that source → upserts into ChromaDB
- Two new tools:
  - `index_document(file_path)` — manually index any `.md / .txt / .py / .json / .csv`
  - `search_documents(query)` — semantic similarity search, returns top-4 chunks with sources
- Transcripts are **auto-indexed** (`_auto_index`) immediately after `transcribe_audio` saves
  them — no manual step needed
- The agent demonstrated adaptive behaviour in practice: it called `list_directory` to
  discover an un-indexed file, tried `search_documents` (empty result), called
  `index_document` itself, then searched again successfully
- New deps: `chromadb>=0.5.0`, `langchain-chroma>=0.1.0`

### 3. Multimodal image input

- `base64` import added; image file uploader added above the chat input bar
  (png / jpg / jpeg / webp / gif)
- When an image is attached, a multimodal `HumanMessage` is built:
  `[{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}]`
- Works with any vision-capable Ollama model (e.g. `gemma3`)
- Image is shown inline in the chat bubble above the user's text
- Base64 bytes are stripped before saving to `chats/*.json` to keep file sizes small;
  a `*(image attached)*` note is added to the message text instead

### 4. Audio upload renaming

- Previously, uploaded audio was saved immediately with its original filename
- Now: uploading a file shows a **naming form** (text input pre-filled with the original
  stem + live filename preview) and a **💾 Save** button
- Final filename format: `<custom_name>-<YYYY-MM-DD>.<ext>`
  e.g. `Morgonmöte-2026-06-01.m4a`
- Date is appended automatically at save time — user only types the meeting name
- Dedup key changed from `uploaded.name` → `f"{name}_{size}"` so two recordings with the
  same source filename each get their own naming form

---

## Final tool set

| Tool | Purpose |
|------|---------|
| `brave_search` | Web search |
| `python_repl` | Local code execution |
| `list_directory` | Browse the file system |
| `read_file` | Read text files |
| `write_md_file` | Create / update Markdown files |
| `transcribe_audio` | Local speech-to-text via Whisper (auto-indexes transcript) |
| `save_memory` | Persist a fact across all conversations |
| `list_memories` | Recall all stored facts |
| `delete_memory` | Forget a specific fact |
| `index_document` | Add a file to the semantic search index |
| `search_documents` | Semantic similarity search over indexed documents |

## Stack

| Component | Choice |
|-----------|--------|
| LLM | Ollama (default `llama3.2`, tested with `gemma3` for vision) |
| Agent framework | LangGraph |
| Search | Brave Search API |
| Transcription | faster-whisper (local) |
| Memory | JSON flat file (`memory.json`) |
| Vector store | ChromaDB (local, embedded) |
| Embeddings | Ollama `nomic-embed-text` |
| UI | Streamlit |
| API | FastAPI |
