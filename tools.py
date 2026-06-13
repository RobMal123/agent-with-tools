"""
tools.py — All tool definitions + TOOLS list

Adding a new tool: define it with @tool here, append to TOOLS at the bottom.
The agent picks it up automatically — no changes to graph.py needed.
"""

import os
import re
import json
from datetime import datetime

from langchain_core.tools import tool
from langchain_community.utilities.brave_search import BraveSearchWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage as _HM

from memory import load_memories, save_memory_entry


# ── Paths ───────────────────────────────────────────────────────────────────────
_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

# ── Obsidian vault integration ───────────────────────────────────────────────────
# An Obsidian vault is just a folder of Markdown files on disk. We treat the vault
# as the agent's knowledge home: it READS / SEARCHES the whole vault but only WRITES
# into a dedicated subfolder (VAULT_AI_SUBDIR) so the user's own notes are never
# touched or overwritten. Set OBSIDIAN_VAULT in .env to enable; when it is empty or
# the path doesn't exist the agent falls back to the in-project knowledge/ folder
# (so first-run and the test suite — which never loads .env — behave exactly as before).
VAULT_DIR       = os.environ.get("OBSIDIAN_VAULT", "").strip().strip('"')
VAULT_ENABLED   = bool(VAULT_DIR) and os.path.isdir(VAULT_DIR)
VAULT_AI_SUBDIR = (os.environ.get("OBSIDIAN_AI_SUBDIR", "AI Assistant") or "AI Assistant").strip()

AUDIO_IN_DIR     = os.path.join(_BASE_DIR, "audio_in")
# Knowledge lives inside the vault subfolder when a vault is configured, else in-project.
KNOWLEDGE_DIR    = os.path.join(VAULT_DIR, VAULT_AI_SUBDIR) if VAULT_ENABLED \
                   else os.path.join(_BASE_DIR, "knowledge")
MEETINGS_DIR     = os.path.join(KNOWLEDGE_DIR, "meetings")
IDEAS_DIR        = os.path.join(KNOWLEDGE_DIR, "ideas")
PROJECTS_DIR     = os.path.join(KNOWLEDGE_DIR, "projects")
REPORTS_DIR      = os.path.join(KNOWLEDGE_DIR, "reports")
IMPROVEMENTS_DIR = os.path.join(KNOWLEDGE_DIR, "improvements")
VIDEOS_DIR       = os.path.join(KNOWLEDGE_DIR, "videos")
WORKSPACE_DIR    = os.path.join(_BASE_DIR, "workspace")
_CHROMA_DIR      = os.path.join(_BASE_DIR, "chroma_db")
# Legacy — kept so existing files in transcriptions/ still resolve via read_file
TRANSCRIPTIONS_DIR = os.path.join(_BASE_DIR, "transcriptions")

# Ensure all runtime directories exist at import time (creates the agent's subfolder
# tree inside the vault on first run — that subfolder IS the integration point).
for _d in (MEETINGS_DIR, IDEAS_DIR, PROJECTS_DIR, REPORTS_DIR,
           IMPROVEMENTS_DIR, VIDEOS_DIR, WORKSPACE_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Filesystem sandbox ────────────────────────────────────────────────────────────
# File tools are confined to a set of allowed roots: always the project directory,
# plus the Obsidian vault when configured. This keeps the agent from reaching
# arbitrary paths while letting it read/write across both the project and the vault.
def _allowed_roots() -> list[str]:
    roots = [os.path.realpath(_BASE_DIR)]
    if VAULT_ENABLED:
        roots.append(os.path.realpath(VAULT_DIR))
    return roots


def _resolve_in_base(path: str, base: str | None = None) -> str | None:
    """
    Resolve an agent-supplied path and confine it to an allowed root (the project
    directory, or the Obsidian vault when configured). Relative paths are taken
    relative to `base` (default: the project root); absolute paths are allowed only
    if they fall inside an allowed root. Returns the real absolute path, or None if
    it would escape every allowed root (symlinks are resolved first).
    """
    candidate = path if os.path.isabs(path) else os.path.join(base or _BASE_DIR, path)
    real = os.path.realpath(candidate)
    real_nc = os.path.normcase(real)
    for root in _allowed_roots():
        root_nc = os.path.normcase(root)
        if real_nc == root_nc or real_nc.startswith(root_nc + os.sep):
            return real
    return None


# ── Obsidian-native note formatting (YAML frontmatter + [[wikilinks]]) ────────────
# Notes the agent writes get a frontmatter block (type/date/tags/source) so they are
# filterable in Obsidian (tag search, Dataview), plus a wikilink back to a per-type
# hub note so they show up connected in the graph view instead of as orphans.
def _yaml_escape(v: str) -> str:
    """Quote a scalar for YAML frontmatter if it contains characters that need it."""
    s = str(v).replace("\n", " ").replace('"', "'").strip()
    if s == "" or re.search(r'[:#\[\]{}",&*?|<>=!%@`]', s):
        return f'"{s}"'
    return s


def _frontmatter(title: str, note_type: str, tags=None, source: str = "",
                 date: str = "", extra: dict | None = None) -> str:
    """Build a YAML frontmatter block, terminated with a blank line so it can be
    prepended directly to note content."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    tags = list(tags or [])
    for t in (note_type, "ai-generated"):
        if t and t not in tags:
            tags.append(t)
    lines = ["---",
             f"title: {_yaml_escape(title)}",
             f"type: {_yaml_escape(note_type or 'note')}",
             f"date: {date}",
             f"created: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "tags:"]
    lines += [f"  - {t}" for t in tags]
    if source:
        lines.append(f"source: {_yaml_escape(source)}")
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {_yaml_escape(v)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"   # blank line between frontmatter and content


# Per-type hub (Map-Of-Content) note each generated note links back to. The hubs need
# not exist yet — Obsidian resolves the link when clicked; the home note is created by
# ensure_vault_home().
_TYPE_HUB = {
    "meeting":     "Meetings",
    "idea":        "Ideas",
    "project":     "Projects",
    "report":      "Reports",
    "improvement": "Improvements",
    "video":       "Videos",
}


def _hub_link(note_type: str) -> str:
    """A one-line wikilink connecting a note to its type hub and the vault home note."""
    home = VAULT_AI_SUBDIR or "AI Assistant"
    hub  = _TYPE_HUB.get(note_type)
    return f"Part of [[{hub}]] · [[{home}]]" if hub else f"Part of [[{home}]]"


# ── Agent model tracking ────────────────────────────────────────────────────────
# build_agent() in graph.py calls set_agent_model() so tools that spin up a
# secondary LLM (structure_thoughts, analyze_improvements) use the right model.

_AGENT_MODEL: str = "llama3.2"
_structure_llm_cache: dict = {}


def set_agent_model(name: str) -> None:
    """Called by build_agent() whenever the active Ollama model changes."""
    global _AGENT_MODEL
    _AGENT_MODEL = name
    _structure_llm_cache.clear()          # force rebuild on next call


def _get_structure_llm() -> ChatOllama:
    """Lazy-load a dedicated temperature=0 LLM for structured-output tools."""
    if _structure_llm_cache.get("model") != _AGENT_MODEL:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        _structure_llm_cache["llm"]   = ChatOllama(
            model=_AGENT_MODEL, temperature=0, base_url=base_url
        )
        _structure_llm_cache["model"] = _AGENT_MODEL
    return _structure_llm_cache["llm"]


# ── Web Search ───────────────────────────────────────────────────────────────────
# Wrapped with @tool so Ollama models receive an explicit `query` parameter
# schema instead of the generic `value` arg that pre-built tools expose.

_brave = BraveSearchWrapper(
    api_key=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
    search_kwargs={"count": 5},
)


@tool
def brave_search(query: str, freshness: str = "") -> str:
    """Search the web for current information — recent events, facts, or anything you're unsure about.

    Write a single specific, focused query (key terms, not a whole sentence) — a precise query
    returns far better results than a vague one. Returns up to 5 results as a numbered list of
    title / snippet / source URL. Synthesise the findings in your own words and cite sources;
    do not paste the raw results back to the user.

    freshness (optional): limit results by recency for time-sensitive queries —
    "pd" past day, "pw" past week, "pm" past month, "py" past year. Leave empty for no time
    filter (best for evergreen facts). Use it when the user wants the latest / most recent info.
    """
    # Degrade gracefully (like every other tool) instead of crashing the agent
    # when the API key is missing or the request fails.
    if not os.environ.get("BRAVE_SEARCH_API_KEY", "").strip():
        return ("Error: web search is unavailable because BRAVE_SEARCH_API_KEY is not set. "
                "Answer from your own knowledge instead.")
    try:
        now = datetime.now()
        # Append current month+year to the query so Brave's ranking favours current results.
        # Without this, queries like "latest AI news" return 2024/2025 results because the
        # query itself carries no date signal.
        dated_query = f"{query} {now.strftime('%B %Y')}"
        fresh = (freshness or "").strip().lower()
        if fresh in ("pd", "pw", "pm", "py"):
            # Per-call recency filter: build a wrapper with Brave's freshness window so only the
            # default (no-filter) path keeps using the shared module-level _brave instance.
            _fresh_brave = BraveSearchWrapper(
                api_key=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": 5, "freshness": fresh},
            )
            raw = _fresh_brave.run(dated_query)
        else:
            raw = _brave.run(dated_query)
    except Exception as e:
        return (f"Error: web search failed ({e}). "
                "Answer from your own knowledge instead.")

    # _brave.run() returns a JSON string: [{"title","link","snippet"}, ...]. Reformat it into a
    # compact, readable list so the model synthesises the facts instead of echoing raw JSON, and
    # truncate long snippets (Brave concatenates description + extra_snippets) to keep context tight.
    try:
        results = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # unexpected shape — hand back what we got rather than crash the run
    if not results:
        return (f"No web results found for '{query}'. Try a different query, or answer from "
                "your own knowledge instead.")

    lines = []
    for i, item in enumerate(results, 1):
        title   = (item.get("title") or "").strip()
        snippet = " ".join((item.get("snippet") or "").split())
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        link = (item.get("link") or "").strip()
        lines.append(f"[{i}] {title}\n{snippet}\nSource: {link}")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"Web search results (searched at {now}; synthesise in your own words and cite [n]; do not paste verbatim):\n\n"
        + "\n\n".join(lines)
    )


def extract_search_sources(tool_output: str) -> list[dict]:
    """
    Parse the "[n] title / snippet / Source: url" blocks that brave_search emits into a list
    of {title, url} dicts (deduped by url, original order preserved). Returns [] for any other
    tool output. Lives next to brave_search so the output format and this parser stay in sync —
    main.py calls it to attach clickable sources to the chat bubble.
    """
    sources: list[dict] = []
    seen: set = set()
    current_title = ""
    for line in (tool_output or "").splitlines():
        line = line.strip()
        m = re.match(r"^\[\d+\]\s*(.*)", line)
        if m:
            current_title = m.group(1).strip()
        elif line.startswith("Source: "):
            url = line[len("Source: "):].strip()
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": current_title or url, "url": url})
            current_title = ""
    return sources


# ── Python REPL (DISABLED by default) ────────────────────────────────────────────
# WARNING: this executes arbitrary Python IN-PROCESS via exec(). The os.chdir below
# only sets the working directory to workspace/ for convenience — it is NOT a security
# sandbox: the code can read/write any file, spawn processes, and reach the network as
# this server's user. Because a poisoned web/search/file result can steer the agent
# here (prompt injection → RCE), it is OFF unless you explicitly opt in by setting
# ENABLE_CODE_EXECUTION=true — and only do that in a trusted, local-only setup.

_repl_instance = PythonREPLTool()


def _code_execution_enabled() -> bool:
    return os.environ.get("ENABLE_CODE_EXECUTION", "").strip().lower() in ("1", "true", "yes", "on")


@tool
def python_repl(code: str) -> str:
    """
    Write and execute Python code for calculations, data analysis, or file generation.
    Working directory is set to workspace/ — files written to disk appear there.
    State (variables, imports) persists across calls within the same session.

    Disabled by default for safety; returns an error unless code execution is enabled.
    """
    if not _code_execution_enabled():
        return ("Error: code execution is disabled. python_repl runs code in-process with no "
                "sandbox, so it is off by default. To enable it, set ENABLE_CODE_EXECUTION=true "
                "in the environment (trusted local-only setups only). For now, compute the answer "
                "yourself or use another tool.")
    sandboxed = (
        f"import os; os.makedirs({repr(WORKSPACE_DIR)}, exist_ok=True); "
        f"os.chdir({repr(WORKSPACE_DIR)})\n{code}"
    )
    return _repl_instance.run(sandboxed)


# ── File tools ───────────────────────────────────────────────────────────────────

@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a local file.
    Supports .txt, .md, .py, .json, and .csv files.
    Use this to inspect or summarize existing files.
    """
    allowed_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in allowed_extensions:
        return f"Error: Only {allowed_extensions} files are supported."

    abs_path = _resolve_in_base(file_path)
    if abs_path is None:
        return "Error: access denied — path is outside the project or vault."

    if not os.path.exists(abs_path):
        return f"Error: File not found at '{file_path}'."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            return content[:8000] + "\n\n[...file truncated at 8000 characters]"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_md_file(file_path: str, content: str) -> str:
    """
    Create or overwrite a Markdown (.md) note in the knowledge base (the Obsidian
    vault when one is configured). Use this to save notes, summaries, reports, or
    research findings.

    Give a path relative to the knowledge home, e.g. 'reports/ai-news.md' or
    'meetings/standup.md' — it lands in the right folder of the vault. Parent
    folders are created automatically. YAML frontmatter (type/date/tags) is added
    for you if you don't write your own. Use Obsidian [[wikilinks]] in the body to
    connect related notes.
    """
    if not file_path.endswith(".md"):
        return "Error: file_path must end with .md"

    # Relative note paths resolve under the knowledge home (the vault subfolder);
    # absolute paths are still allowed as long as they stay inside an allowed root.
    abs_path = _resolve_in_base(file_path, base=KNOWLEDGE_DIR)
    if abs_path is None:
        return "Error: access denied — path is outside the project or vault."
    parent   = os.path.dirname(abs_path)

    # Make notes first-class in Obsidian: prepend frontmatter unless the model
    # already supplied its own block.
    if not content.lstrip().startswith("---"):
        title = os.path.splitext(os.path.basename(abs_path))[0]
        content = _frontmatter(title=title, note_type=_infer_doc_type(abs_path)) + content

    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to: {abs_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# ── Directory browser ────────────────────────────────────────────────────────────

@tool
def list_directory(path: str = ".") -> str:
    """
    List the files and folders inside a directory.
    Defaults to the project root (".") if no path is given.
    Paths are relative to the project folder, e.g. "knowledge/meetings", "knowledge/improvements".
    Returns each entry with its type (file/dir), size in KB, and last-modified date.
    """
    target = _resolve_in_base(path)
    if target is None:
        return "Error: access denied — path is outside the project or vault."

    if not os.path.exists(target):
        return f"Error: path not found — '{target}'"
    if not os.path.isdir(target):
        return f"Error: '{target}' is a file. Use read_file to read it."

    entries = []
    try:
        for name in sorted(os.listdir(target)):
            full  = os.path.join(target, name)
            kind  = "dir " if os.path.isdir(full) else "file"
            mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
            if os.path.isdir(full):
                entries.append(f"[{kind}]  {name}/  ({mtime})")
            else:
                size_kb = os.path.getsize(full) / 1024
                entries.append(f"[{kind}]  {name}  {size_kb:.1f} KB  ({mtime})")
    except PermissionError:
        return f"Error: permission denied reading '{target}'"

    if not entries:
        return f"Directory '{target}' is empty."

    return f"Contents of: {target}\n" + "─" * 60 + "\n" + "\n".join(entries)


# ── Audio transcription ──────────────────────────────────────────────────────────
# Uses faster-whisper (local, CPU-friendly, int8 quantised).
# Model is lazy-loaded and cached; default: "small" (~970 MB).
# Override with WHISPER_MODEL env var.

_whisper_cache: dict = {}


def _get_whisper_model():
    """Lazy-load and cache the faster-whisper model."""
    model_name = os.environ.get("WHISPER_MODEL", "small")
    if model_name not in _whisper_cache:
        try:
            from faster_whisper import WhisperModel
            _whisper_cache[model_name] = WhisperModel(
                model_name, device="cpu", compute_type="int8"
            )
        except ImportError:
            return None, "faster-whisper is not installed. Run: pip install faster-whisper"
        except Exception as e:
            return None, f"Failed to load Whisper model '{model_name}': {e}"
    return _whisper_cache[model_name], None


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file to text using the local Whisper model (runs fully offline).
    Supported formats: .m4a, .mp3, .wav, .mp4, .ogg, .flac, .webm

    The raw transcription is saved to knowledge/meetings/<stem>.md and auto-indexed
    for semantic search. Returns the full transcription text.
    """
    supported = {".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".flac", ".webm"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in supported:
        return f"Error: unsupported format '{ext}'. Supported: {', '.join(sorted(supported))}"

    abs_path = _resolve_in_base(file_path)
    if abs_path is None:
        return "Error: access denied — path is outside the project or vault."
    if not os.path.exists(abs_path):
        return f"Error: file not found — '{file_path}'"

    model, err = _get_whisper_model()
    if err:
        return f"Error: {err}"

    try:
        segments, info = model.transcribe(abs_path, beam_size=5)
        text     = " ".join(seg.text.strip() for seg in segments).strip()
        language = info.language

        stem      = os.path.splitext(os.path.basename(abs_path))[0]
        md_path   = os.path.join(MEETINGS_DIR, f"{stem}.md")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        md_content = (
            _frontmatter(title=f"Transcription: {stem}", note_type="meeting",
                         tags=["transcript"], source=abs_path,
                         extra={"language": language})
            + f"# Transcription: {stem}\n\n"
            + f"{_hub_link('meeting')}\n\n"
            + f"**Date:** {timestamp}  \n"
            + f"**Language detected:** {language}  \n"
            + f"**Source:** {abs_path}\n\n"
            + f"---\n\n{text}\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        _auto_index(md_path, "meeting")

        return (
            f"Transcription complete.\n"
            f"Language: {language} | Saved to: {md_path}\n\n"
            f"---\n\n{text}"
        )

    except Exception as e:
        return f"Error during transcription: {e}"


# ── Video understanding ──────────────────────────────────────────────────────────
# url → yt-dlp download → [Whisper transcript] + [ffmpeg scene-detect keyframes →
# Moondream query()] → one timestamp-aligned markdown file in knowledge/videos/.
# The pipeline lives in video_understanding.py and is imported lazily so tools.py
# stays importable (and fast) when torch / transformers / yt-dlp aren't installed.

@tool
def process_video(url: str) -> str:
    """
    Analyze a video from a URL (YouTube, TikTok, X/Twitter): downloads it, transcribes
    the speech, and describes what is shown on screen, merged into ONE timestamp-aligned
    markdown file. Each line is "[MM:SS] SPOKEN: ..." (speech) or "[MM:SS] VISUAL: ..."
    (on-screen text + scene description).

    Use this whenever the user shares a video link and wants it summarised, fact-checked,
    or discussed. Returns the saved file path plus the timeline (truncated if long — call
    read_file on the path for the full version). Results are cached: re-running the same
    video is instant, but a NEW video takes a few minutes to process — mention that.
    """
    try:
        from video_understanding import process_video_impl
    except ImportError as e:
        return (f"Error: video processing dependencies missing ({e}). "
                "Run: pip install yt-dlp torch transformers einops accelerate")
    try:
        return process_video_impl(url)
    except Exception as e:   # never raise — an exception crashes the whole agent run
        return f"Error: video processing failed ({e})"


# ── Long-term memory tools ───────────────────────────────────────────────────────

@tool
def save_memory(key: str, value: str) -> str:
    """
    Save a fact or preference about the user to long-term memory.
    This persists across ALL conversations — use it to remember names,
    preferences, context, or anything the user asks you to keep in mind.

    Examples:
        save_memory("user_name", "Rob")
        save_memory("preferred_language", "Python")
        save_memory("company", "Acme Corp")

    Use a short snake_case key and a concise value.
    """
    save_memory_entry(key, value)
    return f"Remembered: {key} = {value}"


@tool
def list_memories() -> str:
    """
    List all facts currently stored in long-term memory.
    Call this to recall what you know about the user before answering
    questions where personal context matters.
    """
    memories = load_memories()
    if not memories:
        return "No memories stored yet."
    lines = [f"- **{k}**: {v}" for k, v in memories.items()]
    return "**Long-term memories:**\n" + "\n".join(lines)


# delete_memory is intentionally NOT an agent tool. Deletion is destructive, and small
# models reflexively call it to "answer" recall questions ("what is my…?"), wiping data.
# Memory deletion is available to the user via the UI / API (DELETE /api/memory/{key}),
# which calls memory.delete_memory_entry directly.


# ── Semantic search (RAG) ────────────────────────────────────────────────────────
# Uses ChromaDB (local, embedded) + Ollama embeddings.
# Default model: nomic-embed-text  →  ollama pull nomic-embed-text
# Override with EMBED_MODEL env var.

_vectorstore_cache: dict = {}


def _get_vectorstore():
    """Lazy-load and cache the ChromaDB vector store. Returns (store, error_str)."""
    if "store" not in _vectorstore_cache:
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
            base_url    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            embeddings  = OllamaEmbeddings(model=embed_model, base_url=base_url)
            _vectorstore_cache["store"] = Chroma(
                collection_name="agent_docs",
                embedding_function=embeddings,
                persist_directory=_CHROMA_DIR,
            )
        except ImportError as e:
            return None, f"Missing dependency: {e}. Run: pip install langchain-chroma"
        except Exception as e:
            return None, f"Failed to initialise vector store: {e}"
    return _vectorstore_cache["store"], None


def _infer_doc_type(abs_path: str, doc_type: str = "") -> str:
    """Infer document type from the file's parent folder, or use an explicit override."""
    if doc_type:
        return doc_type
    parent = os.path.basename(os.path.dirname(abs_path)).lower()
    return {
        "meetings":     "meeting",
        "ideas":        "idea",
        "projects":     "project",
        "reports":      "report",
        "improvements": "improvement",
        "videos":       "video",
        "transcriptions": "meeting",   # legacy folder
    }.get(parent, "document")


def _infer_date(abs_path: str) -> str:
    """Infer date from a YYYY-MM-DD pattern in the filename, or fall back to mtime."""
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1)
    try:
        return datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _index_file(abs_path: str, doc_type: str = "") -> str:
    """
    Chunk a file and upsert into ChromaDB with type/date metadata.
    Public helper so app.py can call it directly.
    Returns a human-readable status string.
    """
    try:
        with open(abs_path, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return f"Error reading '{abs_path}': {e}"

    store, err = _get_vectorstore()
    if err:
        return f"Error: {err}"

    inferred_type = _infer_doc_type(abs_path, doc_type)
    inferred_date = _infer_date(abs_path)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks   = splitter.create_documents(
            [text],
            metadatas=[{
                "source":     abs_path,
                "type":       inferred_type,
                "date":       inferred_date,
                "indexed_at": datetime.now().isoformat(),
            }],
        )

        # Upsert: remove stale chunks for this source, then re-add
        try:
            store._collection.delete(where={"source": abs_path})
        except Exception:
            pass

        store.add_documents(chunks)
        return (
            f"Indexed {len(chunks)} chunk(s) "
            f"[type={inferred_type}, date={inferred_date}] "
            f"from: {os.path.basename(abs_path)}"
        )
    except Exception as e:
        return f"Error indexing '{abs_path}': {e}"


def _auto_index(abs_path: str, doc_type: str = "") -> None:
    """Silently index a file after creation. Non-fatal — errors are swallowed."""
    try:
        _index_file(abs_path, doc_type)
    except Exception:
        pass


# ── Obsidian vault: bulk indexing + home note ────────────────────────────────────
_VAULT_SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules"}


def index_vault(verbose: bool = False) -> dict:
    """
    Index every Markdown note in the configured Obsidian vault (the user's existing
    notes plus the agent's own) into the semantic search store, so search_documents
    can recall them. Skips Obsidian/system folders. Safe to re-run — each file's
    stale chunks are replaced. Returns {indexed, skipped, errors}.
    """
    if not VAULT_ENABLED:
        return {"indexed": 0, "skipped": 0,
                "errors": ["No Obsidian vault configured — set OBSIDIAN_VAULT in .env."]}

    ensure_vault_home()
    indexed = skipped = 0
    errors: list[str] = []
    for root, dirs, files in os.walk(VAULT_DIR):
        dirs[:] = [d for d in dirs if d not in _VAULT_SKIP_DIRS]
        for fn in files:
            if not fn.lower().endswith(".md"):
                continue
            status = _index_file(os.path.join(root, fn))
            if status.startswith("Indexed"):
                indexed += 1
                if verbose:
                    print(status)
            else:
                skipped += 1
                errors.append(status)
    return {"indexed": indexed, "skipped": skipped, "errors": errors}


def ensure_vault_home() -> str | None:
    """
    Create the agent's home note (a Map-Of-Content) at the root of its vault
    subfolder, linking to the per-type hubs so the graph view has structure.
    Idempotent — never overwrites an existing home note. Returns its path, or None
    when no vault is configured.
    """
    if not VAULT_ENABLED:
        return None
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    home = os.path.join(KNOWLEDGE_DIR, f"{VAULT_AI_SUBDIR}.md")
    if os.path.exists(home):
        return home
    body = (
        f"# {VAULT_AI_SUBDIR}\n\n"
        "Home for notes created by the local AI research assistant. The assistant "
        "reads and semantically searches this whole vault, and writes its own notes "
        "into the folders below (kept in sync with search).\n\n"
        "## Sections\n"
        "- [[Meetings]] — transcripts & summaries\n"
        "- [[Ideas]] — structured thoughts\n"
        "- [[Projects]] — project plans\n"
        "- [[Reports]] — research & reports\n"
        "- [[Improvements]] — friction & bug log\n"
        "- [[Videos]] — video breakdowns (spoken + visual timelines)\n"
    )
    with open(home, "w", encoding="utf-8") as f:
        f.write(_frontmatter(title=VAULT_AI_SUBDIR, note_type="moc", tags=["moc"]) + body)
    return home


@tool
def index_document(file_path: str, doc_type: str = "") -> str:
    """
    Add a local text or Markdown file to the semantic search index.
    Paths are relative to the project folder, e.g. 'knowledge/meetings/standup.md'.

    doc_type (optional): "meeting", "idea", "project", "report", "improvement"
    If omitted, the type is inferred from the file's parent folder name.

    Re-indexing the same file is safe — old chunks are replaced automatically.
    Requires: ollama pull nomic-embed-text
    """
    allowed = {".txt", ".md", ".py", ".json", ".csv"}
    if os.path.splitext(file_path)[1].lower() not in allowed:
        return f"Error: only {allowed} files can be indexed."

    abs_path = _resolve_in_base(file_path)
    if abs_path is None:
        return "Error: access denied — path is outside the project or vault."
    if not os.path.exists(abs_path):
        return f"Error: file not found — '{file_path}'"

    return _index_file(abs_path, doc_type)


@tool
def search_documents(query: str, doc_type: str = "", top_k: int = 4) -> str:
    """
    Search across all indexed documents using semantic similarity.

    query:    what you're looking for (natural language)
    doc_type: filter by type — "meeting", "idea", "improvement", "report", "project"
              leave empty to search across all types
    top_k:    number of results to return (default 4)

    Examples:
        search_documents("action items from last week")
        search_documents("UI friction", "improvement")
        search_documents("project goals", "project", 6)
    """
    store, err = _get_vectorstore()
    if err:
        return f"Error: {err}"

    try:
        where   = {"type": {"$eq": doc_type}} if doc_type else None
        results = store.similarity_search(query, k=int(top_k), filter=where)
    except Exception as e:
        return f"Error during search: {e}"

    if not results:
        tip = f" of type '{doc_type}'" if doc_type else ""
        return (
            f"No relevant documents found{tip}. "
            "Use index_document to add files to the search index first."
        )

    parts = []
    for i, doc in enumerate(results, 1):
        meta        = doc.metadata
        source_name = os.path.basename(meta.get("source", "unknown"))
        type_tag    = meta.get("type", "")
        date_tag    = meta.get("date", "")
        header      = f"**[{i}] {source_name}**"
        if type_tag or date_tag:
            header += f"  `{type_tag} · {date_tag}`"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# ── Structure & knowledge tools ──────────────────────────────────────────────────

_TEMPLATES = {
    "ideas": """\
## Summary
{summary}

## Key Points
{key_points}

## Open Questions
{open_questions}

## Next Steps
{next_steps}""",

    "problem": """\
## Problem Description
{problem_description}

## Root Causes
{root_causes}

## Impact
{impact}

## Possible Solutions
{possible_solutions}

## Recommended Next Step
{recommended_next_step}""",

    "project": """\
## Goal
{goal}

## Tasks
{tasks}

## Dependencies
{dependencies}

## Risks
{risks}

## Next Action
{next_action}""",
}

_TEMPLATE_FIELDS = {
    "ideas":   ["summary", "key_points", "open_questions", "next_steps"],
    "problem": ["problem_description", "root_causes", "impact",
                "possible_solutions", "recommended_next_step"],
    "project": ["goal", "tasks", "dependencies", "risks", "next_action"],
}


@tool
def structure_thoughts(input_text: str, template_type: str) -> str:
    """
    Structure raw thoughts, ideas, or notes into a clean, consistent markdown document.

    template_type must be one of:
      "ideas"   → Summary · Key Points · Open Questions · Next Steps
      "problem" → Problem Description · Root Causes · Impact · Possible Solutions · Recommended Next Step
      "project" → Goal · Tasks · Dependencies · Risks · Next Action

    The output is saved to knowledge/ideas/ and indexed for future retrieval.
    Returns the structured document.
    """
    template_type = template_type.lower().strip()
    if template_type not in _TEMPLATES:
        return f"Error: template_type must be one of {list(_TEMPLATES.keys())}"

    fields     = _TEMPLATE_FIELDS[template_type]
    field_list = "\n".join(f'"{f}": "..."' for f in fields)

    prompt = f"""You are a structured-thinking assistant. Fill in each field below based on the input.
Be concise. Return ONLY a JSON object with these exact keys — no markdown, no explanation.

Fields required:
{{{field_list}}}

For list-type fields (key_points, tasks, etc.), use a newline-separated string of "- item" lines.

Input:
{input_text}"""

    try:
        llm      = _get_structure_llm()
        response = llm.invoke([_HM(content=prompt)])
        raw      = response.content.strip()

        # Extract JSON from the response (model sometimes wraps in ```json blocks)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
        data = json.loads(json_match.group())

        # Fill the template
        filled = _TEMPLATES[template_type].format(**{k: data.get(k, "—") for k in fields})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        full_doc  = (
            _frontmatter(title=f"{template_type.capitalize()} – {timestamp}",
                         note_type="idea", tags=[template_type],
                         source=input_text[:80])
            + f"# {template_type.capitalize()} – {timestamp}\n\n"
            + f"{_hub_link('idea')}\n\n"
            + f"{filled}\n\n"
            + f"---\n*Source input:* {input_text[:200]}{'…' if len(input_text) > 200 else ''}\n"
        )

        # Save to knowledge/ideas/
        stem     = datetime.now().strftime(f"{template_type}_%Y-%m-%d_%H-%M")
        out_path = os.path.join(IDEAS_DIR, f"{stem}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_doc)
        _auto_index(out_path, "idea")

        return f"Saved to {out_path}\n\n{full_doc}"

    except Exception as e:
        return f"Error structuring thoughts: {e}\n\nRaw model output:\n{raw if 'raw' in dir() else 'n/a'}"


@tool
def log_improvement(input_text: str) -> str:
    """
    Log a friction point, bug, or improvement idea to the improvements backlog.
    The entry is saved to knowledge/improvements/ and indexed for analysis.

    Format input_text with labelled sections for best results:
        Problem: the search is slow with many documents
        Context: happens when knowledge base has 50+ files
        Impact: users wait 5+ seconds per query
        Suggestion: add an index cache layer
        Questions: is this a ChromaDB limit or embedding model?

    Plain text also works — everything is treated as the Problem description.
    """
    # Parse labelled sections if present, otherwise treat all as Problem
    sections = {"problem": "", "context": "", "impact": "", "suggestion": "", "questions": ""}
    current  = "problem"
    label_map = {
        "problem:":    "problem",
        "context:":    "context",
        "impact:":     "impact",
        "suggestion:": "suggestion",
        "questions:":  "questions",
    }

    for line in input_text.strip().splitlines():
        matched = False
        for label, key in label_map.items():
            if line.lower().startswith(label):
                current = key
                sections[current] = line[len(label):].strip()
                matched = True
                break
        if not matched:
            sep = "\n" if sections[current] else ""
            sections[current] += sep + line

    # If nothing reached the named sections, everything is in "problem" already
    if not sections["problem"].strip():
        sections["problem"] = input_text.strip()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fname     = datetime.now().strftime("improvement_%Y-%m-%d_%H-%M.md")
    out_path  = os.path.join(IMPROVEMENTS_DIR, fname)

    md = _frontmatter(title=f"Improvement – {timestamp}", note_type="improvement",
                      source=sections["problem"][:80]) + f"""\
# Improvement – {timestamp}

{_hub_link('improvement')}

## Problem
{sections['problem'] or '—'}

## Context
{sections['context'] or '—'}

## Why it matters
{sections['impact'] or '—'}

## Suggested Change
{sections['suggestion'] or '—'}

## Open Questions
{sections['questions'] or '—'}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    _auto_index(out_path, "improvement")

    return f"Improvement logged to {out_path}"


@tool
def analyze_improvements() -> str:
    """
    Analyze all logged improvements to identify recurring themes and suggest priorities.
    Returns a structured markdown summary.

    Use this periodically once the backlog has several entries.
    """
    if not os.path.exists(IMPROVEMENTS_DIR):
        return "No improvements logged yet. Use log_improvement to start building the backlog."

    files = sorted(f for f in os.listdir(IMPROVEMENTS_DIR) if f.endswith(".md"))
    if not files:
        return "Improvements folder exists but is empty. Use log_improvement to add entries."

    snippets = []
    for fname in files[-20:]:   # cap at 20 to stay within context
        path = os.path.join(IMPROVEMENTS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                snippets.append(f"### {fname}\n{f.read()}")
        except Exception:
            pass

    combined = "\n\n---\n\n".join(snippets)
    prompt   = f"""\
Analyze the following improvement logs and produce a structured markdown report with:

1. **Recurring Themes** — group similar issues together
2. **Highest Impact Items** — which issues matter most and why
3. **Suggested Priority Order** — ordered list of what to fix first

Be concise. Base everything strictly on the logs provided.

IMPROVEMENT LOGS:
{combined[:6000]}"""

    try:
        llm      = _get_structure_llm()
        response = llm.invoke([_HM(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error during analysis: {e}"


# ── Export ───────────────────────────────────────────────────────────────────────
TOOLS = [
    brave_search,
    python_repl,
    list_directory,
    read_file,
    write_md_file,
    transcribe_audio,
    process_video,
    save_memory,
    list_memories,
    index_document,
    search_documents,
    structure_thoughts,
    log_improvement,
    analyze_improvements,
]
