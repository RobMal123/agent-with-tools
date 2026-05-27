# AI Research Agent

A local-first AI agent built with **LangGraph + Ollama** that can search the web, run Python code, transcribe audio, and manage files — all running on your own machine.

## Architecture

```
User query
    ↓
[call_model] ← Local LLM (Ollama) decides: respond or use a tool?
    ↓ (tool needed)
[call_tools] ← Executes the tool
    ↓
[call_model] ← LLM synthesises the tool result
    ↓ (done)
Final response
```

## Tools

| Tool | What it does |
|------|-------------|
| `brave_search` | Searches the web via Brave Search API |
| `python_repl` | Writes and runs Python code locally |
| `list_directory` | Browses folders to discover files |
| `read_file` | Reads `.txt`, `.md`, `.py`, `.json`, `.csv` files |
| `write_md_file` | Creates or overwrites Markdown files |
| `transcribe_audio` | Transcribes audio locally with Whisper (faster-whisper) |

## Requirements

- [Ollama](https://ollama.com) running locally with at least one tool-capable model pulled  
  (e.g. `ollama pull llama3.2`)
- [ffmpeg](https://ffmpeg.org) on PATH (required for audio transcription)
- A [Brave Search API key](https://brave.com/search/api/) (free tier: 2 000 queries/month)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
# Edit .env — add your BRAVE_SEARCH_API_KEY

# Make sure Ollama is running, then start the UI
streamlit run app.py
```

## Running

**Streamlit UI** (recommended):
```bash
streamlit run app.py
```

**FastAPI server**:
```bash
uvicorn main:app --reload
# Docs at http://localhost:8000/docs
```

## Project structure

```
.
├── app.py              # Streamlit chat UI
├── main.py             # FastAPI REST API
├── graph.py            # LangGraph agent (core logic)
├── tools.py            # Tool definitions
├── state.py            # AgentState schema
├── prompts.py          # System prompt
├── requirements.txt
├── .env.example
├── audio_in/           # Drop audio files here for transcription
├── transcriptions/     # Raw transcripts + meeting summaries saved here
└── chats/              # Persisted conversation history
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAVE_SEARCH_API_KEY` | — | Required. Get at brave.com/search/api |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Override if Ollama runs elsewhere |
| `WHISPER_MODEL` | `small` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large-v3` |

## Ideas for extending

- Add a **vector database tool** (Chroma / Qdrant) for RAG over your documents
- Add **calendar integration** to auto-create action items from meeting summaries
- Deploy with a **Dockerfile** behind a reverse proxy
- Add **LangSmith tracing** for observability
