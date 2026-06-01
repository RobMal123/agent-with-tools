import os
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.utilities.brave_search import BraveSearchWrapper
from langchain_experimental.tools import PythonREPLTool

from memory import load_memories, save_memory_entry, delete_memory_entry

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
AUDIO_IN_DIR       = os.path.join(_BASE_DIR, "audio_in")
TRANSCRIPTIONS_DIR = os.path.join(_BASE_DIR, "transcriptions")
_CHROMA_DIR        = os.path.join(_BASE_DIR, "chroma_db")


# --- Web Search ---
# Brave Search API — get a free key at https://brave.com/search/api/
# Wrapped with @tool so Ollama models receive an explicit `query` parameter
# schema instead of the generic `value` arg that pre-built tools expose.
_brave = BraveSearchWrapper(
    api_key=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
    search_kwargs={"count": 3},
)

@tool
def brave_search(query: str) -> str:
    """Search the web for current information. Use this for recent events, facts, or anything you're unsure about."""
    return _brave.run(query)


# --- Python REPL ---
# Lets the agent write and run Python code for calculations, data analysis, etc.
# WARNING: only use in sandboxed environments in production!
python_repl = PythonREPLTool()


# --- File tools ---

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

    if not os.path.exists(file_path):
        return f"Error: File not found at '{file_path}'."

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Limit to ~8000 chars to stay within context window
        if len(content) > 8000:
            return content[:8000] + "\n\n[...file truncated at 8000 characters]"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_md_file(file_path: str, content: str) -> str:
    """
    Create or overwrite a Markdown (.md) file with the given content.
    Use this to save notes, summaries, reports, or research findings to disk.
    The file_path must end with .md (e.g. 'notes.md' or 'reports/summary.md').
    Parent directories are created automatically if they do not exist.
    """
    if not file_path.endswith(".md"):
        return "Error: file_path must end with .md"

    abs_path = os.path.abspath(file_path)
    parent = os.path.dirname(abs_path)

    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to: {abs_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# --- Directory browser ---

@tool
def list_directory(path: str = ".") -> str:
    """
    List the files and folders inside a directory.
    Defaults to the project root (".") if no path is given.
    Use this to discover what files exist before reading them.
    Paths are relative to the project folder (e.g. "transcriptions", "audio_in").
    Returns each entry with its type (file/dir), size in KB, and last-modified date.
    """
    # Resolve relative to project root so short paths like "transcriptions" just work
    target = os.path.join(_BASE_DIR, path) if not os.path.isabs(path) else path
    target = os.path.normpath(target)

    if not os.path.exists(target):
        return f"Error: path not found — '{target}'"
    if not os.path.isdir(target):
        return f"Error: '{target}' is a file, not a directory. Use read_file to read it."

    entries = []
    try:
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            kind = "dir " if os.path.isdir(full) else "file"
            size_kb = os.path.getsize(full) / 1024 if os.path.isfile(full) else 0
            mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
            if os.path.isdir(full):
                entries.append(f"[{kind}]  {name}/  ({mtime})")
            else:
                entries.append(f"[{kind}]  {name}  {size_kb:.1f} KB  ({mtime})")
    except PermissionError:
        return f"Error: permission denied reading '{target}'"

    if not entries:
        return f"Directory '{target}' is empty."

    header = f"Contents of: {target}\n" + "─" * 60
    return header + "\n" + "\n".join(entries)


# --- Audio transcription ---
# Uses faster-whisper (local, CPU-friendly, int8 quantised).
# Model is lazy-loaded and cached so it only downloads/loads once per session.
# Default model: "small" (~970 MB).  Override with WHISPER_MODEL env var.

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

    The raw transcription is saved automatically to transcriptions/<stem>.md.
    Returns the full transcription text so you can analyse it, create a meeting
    summary, extract action items, etc.

    Paths are resolved relative to the project folder, so you can pass just
    'audio_in/meeting.m4a' without a full absolute path.
    """
    supported = {".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".flac", ".webm"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in supported:
        return f"Error: unsupported format '{ext}'. Supported: {', '.join(sorted(supported))}"

    # Resolve path — try as-is, then relative to the project directory
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        alt = os.path.join(_BASE_DIR, file_path)
        if os.path.exists(alt):
            abs_path = alt
        else:
            return f"Error: file not found — tried '{abs_path}' and '{alt}'"

    model, err = _get_whisper_model()
    if err:
        return f"Error: {err}"

    try:
        segments, info = model.transcribe(abs_path, beam_size=5)
        # Consume the generator (transcription happens here)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        language = info.language

        # Save raw transcription as .md
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
        md_path = os.path.join(TRANSCRIPTIONS_DIR, f"{stem}.md")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        md_content = (
            f"# Transcription: {stem}\n\n"
            f"**Date:** {timestamp}  \n"
            f"**Language detected:** {language}  \n"
            f"**Source:** {abs_path}\n\n"
            f"---\n\n"
            f"{text}\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Auto-index the transcript so it's searchable immediately
        _auto_index(md_path)

        return (
            f"Transcription complete.\n"
            f"Language: {language} | Saved to: {md_path}\n\n"
            f"---\n\n{text}"
        )

    except Exception as e:
        return f"Error during transcription: {e}"


# ── Long-term memory tools ─────────────────────────────────────────────────────

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


@tool
def delete_memory(key: str) -> str:
    """
    Delete a specific fact from long-term memory by its key.
    Use list_memories first to see which keys exist.
    """
    removed = delete_memory_entry(key)
    if removed:
        return f"Deleted memory: '{key}'"
    return f"No memory found with key '{key}'. Use list_memories to see available keys."


# ── Semantic search (RAG) ──────────────────────────────────────────────────────
# Uses ChromaDB (local, embedded, no server) + Ollama embeddings.
# Default embedding model: nomic-embed-text (must be pulled: ollama pull nomic-embed-text)
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


def _index_file(abs_path: str) -> str:
    """
    Core indexing logic — chunks a file and upserts it into ChromaDB.
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

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks   = splitter.create_documents(
            [text],
            metadatas=[{"source": abs_path, "indexed_at": datetime.now().isoformat()}],
        )

        # Remove stale chunks from this source before re-indexing
        try:
            store._collection.delete(where={"source": abs_path})
        except Exception:
            pass

        store.add_documents(chunks)
        return f"Indexed {len(chunks)} chunk(s) from: {os.path.basename(abs_path)}"
    except Exception as e:
        return f"Error indexing '{abs_path}': {e}"


def _auto_index(abs_path: str) -> None:
    """Silently index a file after it's created (e.g. after transcription). Non-fatal."""
    try:
        _index_file(abs_path)
    except Exception:
        pass


@tool
def index_document(file_path: str) -> str:
    """
    Add a local text or Markdown file to the semantic search index so it can
    be found with search_documents later.

    Paths are relative to the project folder, e.g.:
        index_document("transcriptions/meeting.md")
        index_document("reports/research.md")

    Re-indexing the same file is safe — old chunks are replaced automatically.
    Requires the Ollama nomic-embed-text model:  ollama pull nomic-embed-text
    """
    allowed = {".txt", ".md", ".py", ".json", ".csv"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed:
        return f"Error: only {allowed} files can be indexed."

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        alt = os.path.join(_BASE_DIR, file_path)
        if os.path.exists(alt):
            abs_path = alt
        else:
            return f"Error: file not found — tried '{file_path}' and relative to project root."

    return _index_file(abs_path)


@tool
def search_documents(query: str) -> str:
    """
    Search across all indexed documents (transcriptions, notes, reports) using
    semantic similarity. Returns the most relevant passages with their source files.

    Use this to answer questions about past meetings, saved research, or any
    content that has been indexed with index_document.

    Requires at least one document to have been indexed first.
    """
    store, err = _get_vectorstore()
    if err:
        return f"Error: {err}"

    try:
        results = store.similarity_search(query, k=4)
    except Exception as e:
        return f"Error during search: {e}"

    if not results:
        return (
            "No relevant documents found. "
            "Use index_document to add files to the search index first."
        )

    parts = []
    for i, doc in enumerate(results, 1):
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"**[{i}] {source_name}**\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# Export all tools as a list — this is what the agent will use
TOOLS = [
    brave_search,
    python_repl,
    list_directory,
    read_file,
    write_md_file,
    transcribe_audio,
    save_memory,
    list_memories,
    delete_memory,
    index_document,
    search_documents,
]
