"""
main.py — FastAPI server for the AI Research Agent

Endpoints:
    GET  /health                       → health check
    POST /chat                         → single-turn or multi-turn chat (text + optional image)
    POST /chat/stream                  → streaming response (SSE framing over POST, text-only)

    GET  /api/models                   → list installed Ollama models
    GET  /api/status                   → LangSmith tracing status

    GET  /api/chats                    → list saved chats (newest-first)
    POST /api/chats/{thread_id}/load   → load a saved chat with full messages
    DELETE /api/chats/{thread_id}      → delete a saved chat

    GET  /api/memory                   → all long-term memories
    DELETE /api/memory/{key}           → delete one memory entry
    DELETE /api/memory                 → clear all memories

    POST /api/knowledge/index-all      → index knowledge/ + legacy dirs
    POST /api/knowledge/index          → index knowledge/ only
    POST /api/knowledge/save-idea      → save last assistant reply as idea

    GET  /api/audio                    → list audio files + transcript/summary status
    POST /api/audio/upload             → upload audio file to audio_in/

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uuid
import json
import os
import re
import secrets
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # must run before graph/tools are imported so env vars are set
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import agent
from memory import load_memories, delete_memory_entry, clear_all_memories
from tools import _index_file

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHATS_DIR        = os.path.join(_BASE_DIR, "chats")
AUDIO_IN_DIR     = os.path.join(_BASE_DIR, "audio_in")
KNOWLEDGE_DIR    = os.path.join(_BASE_DIR, "knowledge")
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".flac", ".webm"}

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(AUDIO_IN_DIR, exist_ok=True)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Agent",
    description="LangGraph-powered agent with web search, code execution, and file reading.",
    version="2.0.0",
)

# Restrict cross-origin access to the local UI origins only. Do NOT use "*":
# every endpoint here is unauthenticated and the agent can execute code and
# read/write files, so a wildcard lets any website you visit drive the agent
# cross-origin (drive-by RCE / data exfiltration). Add ports here if you run the
# UI elsewhere — note localhost and 127.0.0.1 are distinct browser origins.
ALLOWED_ORIGINS = [
    "http://localhost:8080", "http://127.0.0.1:8080",  # Lovable/Vite dev server (bun dev)
    "http://localhost:3001", "http://127.0.0.1:3001",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    # Authorization / X-API-Token must be allow-listed or the browser's preflight for
    # token-authenticated requests fails (Authorization is not a CORS-safelisted header).
    allow_headers=["Content-Type", "Authorization", "X-API-Token"],
)


# CSRF / cross-origin side-effect guard. CORS only controls whether a page may
# READ a response; it does NOT stop the browser from SENDING "simple" requests
# (EventSource GETs, text/plain or multipart POSTs) whose server-side side effects
# still run. Browsers always attach an Origin header to such cross-origin requests
# and JS cannot forge it, so we reject any request whose Origin is present but not
# allow-listed. Requests with no Origin (curl, server-to-server) are unaffected.
@app.middleware("http")
async def enforce_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
    return await call_next(request)


# ── Authentication ───────────────────────────────────────────────────────────
# Optional shared-secret token. When API_TOKEN is set, every request (except the
# health check and CORS preflight) must present it via "Authorization: Bearer <token>"
# or an "X-API-Token" header. Browsers are already covered by the Origin guard above;
# the token additionally protects non-browser clients (curl/scripts/other hosts) and is
# strongly recommended before binding the server to anything other than localhost.
API_TOKEN = os.environ.get("API_TOKEN", "").strip()
_AUTH_EXEMPT_PATHS = {"/health"}

if not API_TOKEN:
    print(
        "WARNING: API_TOKEN is not set - API authentication is DISABLED. Set API_TOKEN in "
        ".env (and VITE_API_TOKEN in the UI) to require a token. Strongly recommended before "
        "exposing the server beyond localhost.",
        file=sys.stderr,
    )


def _request_token(request: Request) -> str | None:
    """Pull the token from the Authorization header or the X-API-Token header."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-api-token")


@app.middleware("http")
async def require_token(request: Request, call_next):
    if (
        API_TOKEN
        and request.method != "OPTIONS"
        and request.url.path not in _AUTH_EXEMPT_PATHS
    ):
        provided = _request_token(request)
        if not provided or not secrets.compare_digest(provided, API_TOKEN):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    image_b64: str | None = None   # base64-encoded image (no data: prefix)
    image_mime: str | None = None  # e.g. "image/png"


class StreamRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    tool_calls_made: list[str]


class SaveIdeaRequest(BaseModel):
    content: str


# ── Internal helpers ───────────────────────────────────────────────────────────

# Saved chats are stored as <thread_id>.json, so a thread_id must never contain
# path separators or "..". UUIDs (the only ids the UI generates) pass this.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _safe_thread_id(thread_id: str) -> str:
    """Reject any thread_id that could escape the chats/ directory."""
    if not _SAFE_ID_RE.match(thread_id):
        raise HTTPException(status_code=400, detail="Invalid thread_id")
    return thread_id


def _get_ollama_models() -> list[str]:
    """Query local Ollama for installed models."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return sorted(models) if models else ["llama3.2"]
    except Exception:
        return ["llama3.2"]


def _list_chats_raw() -> list[dict]:
    """Return all saved chat dicts, newest-first."""
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHATS_DIR, fname), encoding="utf-8") as f:
                    chats.append(json.load(f))
            except Exception:
                pass
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)


def _save_chat(thread_id: str, title: str, model: str, messages: list) -> None:
    """Persist the conversation to chats/<thread_id>.json."""
    _safe_thread_id(thread_id)
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    # Preserve original created_at if the file already exists
    created_at = datetime.now().isoformat(timespec="seconds")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                created_at = json.load(f).get("created_at", created_at)
        except Exception:
            pass
    data = {
        "thread_id": thread_id,
        "title": title,
        "model": model,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Core chat endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Single request-response chat.
    Supports plain text and multimodal (text + image) turns.
    Pass the same thread_id to maintain conversation history.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Build LangChain message — multimodal when an image is provided
    if request.image_b64 and request.image_mime:
        lc_content = [
            {"type": "text", "text": request.message},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{request.image_mime};base64,{request.image_b64}"},
            },
        ]
    else:
        lc_content = request.message

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=lc_content)]},
            config=config,
        )
    except Exception:
        # Log the real error server-side; don't leak internals (paths/stack) to the client.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Agent error while processing the request")

    reply = result["messages"][-1].content
    tool_calls_made = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        tool_calls_made=list(set(tool_calls_made)),
    )


@app.post("/chat/stream")
def chat_stream(request: StreamRequest):
    """
    Streaming endpoint — text turns only. Uses SSE framing over a POST body so the prompt
    travels in the body and the token in the Authorization header (neither leaks into the
    URL / server logs). The React UI falls back to POST /chat for image turns.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        for event in agent.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                data = json.dumps({"token": last_msg.content, "thread_id": thread_id})
                yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── /api/models & /api/status ──────────────────────────────────────────────────

@app.get("/api/models")
def get_models() -> list[str]:
    """Return all installed Ollama models."""
    return _get_ollama_models()


@app.get("/api/status")
def get_status():
    """Return LangSmith tracing status for the sidebar badge."""
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.environ.get("LANGCHAIN_PROJECT", "default") if tracing else None
    return {"tracing_enabled": tracing, "langsmith_project": project}


# ── /api/chats ─────────────────────────────────────────────────────────────────

@app.get("/api/chats")
def list_chats():
    """Return all saved chats (summary only — no messages) sorted newest-first."""
    return [
        {
            "thread_id": c["thread_id"],
            "title": c.get("title", "Untitled"),
            "model": c.get("model", ""),
            "updated_at": c.get("updated_at", ""),
            "created_at": c.get("created_at", ""),
        }
        for c in _list_chats_raw()
    ]


@app.post("/api/chats/{thread_id}/load")
def load_chat(thread_id: str):
    """Return a saved chat including its full message list."""
    _safe_thread_id(thread_id)
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.delete("/api/chats/{thread_id}")
def delete_chat(thread_id: str):
    """Permanently delete a saved chat."""
    _safe_thread_id(thread_id)
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat not found")
    os.remove(path)
    return {"ok": True}


# ── /api/memory ────────────────────────────────────────────────────────────────

@app.get("/api/memory")
def get_memory() -> dict:
    """Return all long-term memories as a flat key→value dict."""
    return load_memories()


@app.delete("/api/memory/{key}")
def delete_memory_key(key: str):
    """Delete a single memory entry by key."""
    existed = delete_memory_entry(key)
    if not existed:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"ok": True}


@app.delete("/api/memory")
def clear_memory():
    """Delete all stored memories."""
    n = clear_all_memories()
    return {"cleared": n}


# ── /api/knowledge ─────────────────────────────────────────────────────────────

@app.post("/api/knowledge/index-all")
def index_all_docs():
    """Index all .md files in knowledge/ plus legacy transcriptions/ and reports/ dirs."""
    indexed = 0
    for scan in [
        KNOWLEDGE_DIR,
        os.path.join(_BASE_DIR, "transcriptions"),
        os.path.join(_BASE_DIR, "reports"),
    ]:
        if os.path.exists(scan):
            for root, _, files in os.walk(scan):
                for fn in files:
                    if fn.endswith(".md"):
                        _index_file(os.path.join(root, fn))
                        indexed += 1
    return {"indexed": indexed}


@app.post("/api/knowledge/index")
def index_knowledge():
    """Index all .md files in the knowledge/ directory only."""
    indexed = 0
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fn in files:
            if fn.endswith(".md"):
                _index_file(os.path.join(root, fn))
                indexed += 1
    return {"indexed": indexed}


@app.post("/api/knowledge/save-idea")
def save_idea(body: SaveIdeaRequest):
    """Save content as a timestamped idea file in knowledge/ideas/ and index it."""
    stem     = datetime.now().strftime("idea_%Y-%m-%d_%H-%M")
    ideas_dir = os.path.join(KNOWLEDGE_DIR, "ideas")
    os.makedirs(ideas_dir, exist_ok=True)
    path = os.path.join(ideas_dir, f"{stem}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# Idea – {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{body.content}\n"
        )
    _index_file(path, "idea")
    return {"saved": f"knowledge/ideas/{stem}.md"}


# ── /api/audio ─────────────────────────────────────────────────────────────────

@app.get("/api/audio")
def list_audio():
    """
    List all audio files in audio_in/ with their transcript/summary status.
    Status logic:
      🔴 has_transcript=False, has_summary=False
      🔵 has_transcript=True,  has_summary=False
      ✅ has_transcript=True,  has_summary=True
    """
    meetings_dir       = os.path.join(KNOWLEDGE_DIR, "meetings")
    transcriptions_dir = os.path.join(_BASE_DIR, "transcriptions")  # legacy

    files = []
    if os.path.exists(AUDIO_IN_DIR):
        for fname in sorted(os.listdir(AUDIO_IN_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                continue
            stem = os.path.splitext(fname)[0]
            size = os.path.getsize(os.path.join(AUDIO_IN_DIR, fname))
            has_transcript = (
                os.path.exists(os.path.join(meetings_dir, f"{stem}.md"))
                or os.path.exists(os.path.join(transcriptions_dir, f"{stem}.md"))
            )
            has_summary = (
                os.path.exists(os.path.join(meetings_dir, f"{stem}_summary.md"))
                or os.path.exists(os.path.join(transcriptions_dir, f"{stem}_summary.md"))
            )
            files.append(
                {
                    "filename": fname,
                    "size_bytes": size,
                    "has_transcript": has_transcript,
                    "has_summary": has_summary,
                }
            )
    return files


@app.post("/api/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    custom_name: str = Form(...),
):
    """
    Upload an audio file to audio_in/ with a custom name + date suffix.
    The final filename will be: <custom_name>-YYYY-MM-DD.<ext>
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext!r}")

    # Sanitize the user-supplied name to a bare filename stem: strip any path
    # components and allow only safe characters, so it cannot escape audio_in/.
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", custom_name.strip()).strip("_")[:80]
    if not safe_stem:
        raise HTTPException(status_code=400, detail="custom_name must contain letters or digits")

    date_str   = datetime.now().strftime("%Y-%m-%d")
    final_name = f"{safe_stem}-{date_str}{ext}"
    dest       = os.path.join(AUDIO_IN_DIR, final_name)

    # Defense in depth: never write outside audio_in/.
    if os.path.dirname(os.path.realpath(dest)) != os.path.realpath(AUDIO_IN_DIR):
        raise HTTPException(status_code=400, detail="Invalid destination path")

    # Stream to disk in chunks with a hard size cap so a huge upload can't
    # exhaust memory/disk. Remove the partial file if the limit is exceeded.
    MAX_AUDIO_BYTES = 200 * 1024 * 1024  # 200 MB
    total = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 200 MB)")
                f.write(chunk)
    except HTTPException:
        if os.path.exists(dest):
            os.remove(dest)
        raise

    return {"saved": final_name}
