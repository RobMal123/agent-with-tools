"""
app.py — Streamlit chat interface

Run with: streamlit run app.py
"""

import streamlit as st
import uuid
import json
import os
import sys
import base64
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # must run before graph/tools are imported so env vars are set
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import build_agent
from memory import load_memories, delete_memory_entry, clear_all_memories
from tools import _index_file

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHATS_DIR        = os.path.join(_BASE_DIR, "chats")
AUDIO_IN_DIR     = os.path.join(_BASE_DIR, "audio_in")
TRANSCRIPTIONS_DIR = os.path.join(_BASE_DIR, "transcriptions")
os.makedirs(CHATS_DIR, exist_ok=True)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".flac", ".webm"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_ollama_models() -> list:
    """Query local Ollama for installed models."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return sorted(models) if models else ["llama3.2"]
    except Exception:
        return ["llama3.2"]


def _existing_created_at(path: str) -> str:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("created_at", datetime.now().isoformat(timespec="seconds"))
        except Exception:
            pass
    return datetime.now().isoformat(timespec="seconds")


def save_chat(thread_id: str, title: str, model: str, messages: list) -> None:
    """Persist the current conversation to chats/<thread_id>.json.
    Image bytes are stripped before saving to keep JSON files small;
    a note is appended to the message text so the history is still readable.
    """
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    clean_messages = []
    for m in messages:
        if "image_b64" in m:
            stripped = {k: v for k, v in m.items() if k not in ("image_b64", "image_mime")}
            stripped["content"] = stripped["content"] + " *(image attached)*"
            clean_messages.append(stripped)
        else:
            clean_messages.append(m)
    data = {
        "thread_id": thread_id,
        "title": title,
        "model": model,
        "created_at": _existing_created_at(path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": clean_messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_chats() -> list:
    """Return all saved chats sorted newest-first."""
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHATS_DIR, fname), encoding="utf-8") as f:
                    chats.append(json.load(f))
            except Exception:
                pass
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)


def delete_chat(thread_id: str) -> None:
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    if os.path.exists(path):
        os.remove(path)


def load_chat_into_state(chat: dict) -> None:
    """Restore a saved chat into session state and re-seed agent memory."""
    st.session_state.thread_id = chat["thread_id"]
    st.session_state.display_messages = list(chat["messages"])

    # Rebuild LangGraph memory so follow-up questions have context
    lc_messages = []
    for m in chat["messages"]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant" and m.get("content"):
            lc_messages.append(AIMessage(content=m["content"]))

    if lc_messages:
        config = {"configurable": {"thread_id": chat["thread_id"]}}
        try:
            st.session_state.agent.update_state(config, {"messages": lc_messages})
        except Exception:
            pass  # non-fatal — user can still view history


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Research Agent")

# ── Session-state init ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "available_models" not in st.session_state:
    st.session_state.available_models = get_ollama_models()
if "selected_model" not in st.session_state:
    models = st.session_state.available_models
    st.session_state.selected_model = models[0] if models else "llama3.2"
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(model_name=st.session_state.selected_model)
if "saved_uploads" not in st.session_state:
    # Tracks filenames already written to audio_in/ this session to prevent
    # double-saves when Streamlit reruns with the uploader widget still populated.
    st.session_state.saved_uploads = set()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # --- Model picker ---
    st.header("⚙️ Model")
    models = st.session_state.available_models
    current_idx = models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
    chosen_model = st.selectbox(
        "Ollama model",
        options=models,
        index=current_idx,
        label_visibility="collapsed",
    )
    if chosen_model != st.session_state.selected_model:
        # Save current chat, then rebuild agent for new model
        if st.session_state.display_messages:
            first = st.session_state.display_messages[0]["content"]
            save_chat(
                st.session_state.thread_id,
                first[:60] + ("…" if len(first) > 60 else ""),
                st.session_state.selected_model,
                st.session_state.display_messages,
            )
        st.session_state.selected_model = chosen_model
        st.session_state.agent = build_agent(model_name=chosen_model)
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()

    if st.button("🔄 Refresh model list", use_container_width=True):
        st.session_state.available_models = get_ollama_models()
        st.rerun()

    # LangSmith tracing status
    _tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    if _tracing:
        _project = os.environ.get("LANGCHAIN_PROJECT", "default")
        st.success(f"🔍 LangSmith tracing · `{_project}`")
    else:
        st.caption("🔍 LangSmith tracing off")

    st.divider()

    # --- Conversation controls ---
    st.header("💬 Conversation")
    st.code(f"Thread: {st.session_state.thread_id[:8]}...")

    if st.button("➕ New conversation", use_container_width=True):
        if st.session_state.display_messages:
            first = st.session_state.display_messages[0]["content"]
            save_chat(
                st.session_state.thread_id,
                first[:60] + ("…" if len(first) > 60 else ""),
                st.session_state.selected_model,
                st.session_state.display_messages,
            )
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()

    st.divider()

    # --- Audio transcription ---
    st.header("🎙️ Transcription")

    # Upload widget — shows a naming form before saving
    uploaded = st.file_uploader(
        "Upload audio",
        type=["m4a", "mp3", "wav", "mp4", "ogg", "flac", "webm"],
        label_visibility="collapsed",
        key="audio_uploader",
    )
    if uploaded is not None:
        # Use name+size as a dedup key so the form doesn't re-appear after saving
        _upload_key = f"{uploaded.name}_{uploaded.size}"
        if _upload_key not in st.session_state.saved_uploads:
            # Show naming form
            _default_stem = os.path.splitext(uploaded.name)[0]
            _ext          = os.path.splitext(uploaded.name)[1].lower()
            _date_str     = datetime.now().strftime("%Y-%m-%d")
            _custom_name  = st.text_input(
                "Recording name",
                value=_default_stem,
                key="audio_custom_name",
                placeholder="e.g. Morgonmöte",
                help="Date is added automatically: name-YYYY-MM-DD",
            )
            st.caption(f"Will save as: **{_custom_name.strip() or '…'}-{_date_str}{_ext}**")
            if st.button("💾 Save to audio_in/", use_container_width=True, key="save_audio_btn"):
                if _custom_name.strip():
                    _final_name = f"{_custom_name.strip()}-{_date_str}{_ext}"
                    os.makedirs(AUDIO_IN_DIR, exist_ok=True)
                    with open(os.path.join(AUDIO_IN_DIR, _final_name), "wb") as _f:
                        _f.write(uploaded.getbuffer())
                    st.session_state.saved_uploads.add(_upload_key)
                    st.rerun()
                else:
                    st.error("Please enter a name before saving.")

    audio_files = []
    if os.path.exists(AUDIO_IN_DIR):
        for fname in sorted(os.listdir(AUDIO_IN_DIR)):
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(fname)

    if not audio_files:
        st.caption("No audio files found in `audio_in/`.")
    else:
        for af in audio_files:
            size_kb = os.path.getsize(os.path.join(AUDIO_IN_DIR, af)) // 1024
            stem = os.path.splitext(af)[0]
            has_transcript = os.path.exists(os.path.join(TRANSCRIPTIONS_DIR, f"{stem}.md"))
            has_summary    = os.path.exists(os.path.join(TRANSCRIPTIONS_DIR, f"{stem}_summary.md"))
            # 🔴 nothing  🔵 transcript only  ✅ both done
            if has_transcript and has_summary:
                status = "✅"
            elif has_transcript:
                status = "🔵"
            else:
                status = "🔴"
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                st.caption(f"{status} **{af}**  \n`{size_kb} KB`")
            with col_btn:
                if st.button("▶ Run", key=f"trans_{af}", use_container_width=True):
                    # Set flag so the app saves the agent's reply automatically —
                    # avoids relying on the model to chain a second write_md_file call.
                    st.session_state.auto_save_summary = {
                        "path": f"transcriptions/{stem}_summary.md",
                        "stem": stem,
                    }
                    st.session_state.pending_input = (
                        f'Transcribe the audio file "audio_in/{af}" and produce a '
                        f'detailed meeting summary. Structure it with these sections: '
                        f'**Overview**, **Key Topics Discussed**, '
                        f'**Decisions Made**, and **Action Items**.'
                    )

    st.divider()

    # --- Saved chats ---
    st.header("🗂️ Saved Chats")
    saved = list_chats()
    if not saved:
        st.caption("No saved chats yet.")
    else:
        for chat in saved:
            is_active = chat["thread_id"] == st.session_state.thread_id
            label = ("▶ " if is_active else "") + chat["title"][:38]
            with st.expander(label, expanded=False):
                st.caption(f"Model: `{chat.get('model', '?')}` · {chat.get('updated_at', '')[:16]}")
                col_load, col_del = st.columns(2)
                with col_load:
                    if st.button("Load", key=f"load_{chat['thread_id']}", use_container_width=True):
                        load_chat_into_state(chat)
                        st.rerun()
                with col_del:
                    if st.button("Delete", key=f"del_{chat['thread_id']}", use_container_width=True):
                        delete_chat(chat["thread_id"])
                        if chat["thread_id"] == st.session_state.thread_id:
                            st.session_state.thread_id = str(uuid.uuid4())
                            st.session_state.display_messages = []
                        st.rerun()

    st.divider()

    # --- Long-term memory ---
    _memories = load_memories()
    st.header(f"🧠 Memory ({len(_memories)})")

    if not _memories:
        st.caption("No memories yet. Tell the agent: *'Remember that my name is …'*")
    else:
        for _key, _val in list(_memories.items()):
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.caption(f"**{_key}**: {_val}")
            with col_del:
                if st.button("🗑", key=f"mem_del_{_key}", use_container_width=True,
                             help=f"Forget '{_key}'"):
                    delete_memory_entry(_key)
                    st.rerun()

    _col_idx, _col_clr = st.columns(2)
    with _col_idx:
        if st.button("📚 Index docs", use_container_width=True,
                     help="Index transcriptions/ and reports/ for semantic search"):
            _indexed = 0
            for _dir in ["transcriptions", "reports"]:
                _dp = os.path.join(_BASE_DIR, _dir)
                if os.path.exists(_dp):
                    for _fn in sorted(os.listdir(_dp)):
                        if _fn.endswith(".md"):
                            _index_file(os.path.join(_dp, _fn))
                            _indexed += 1
            st.success(f"Indexed {_indexed} file(s)")
    with _col_clr:
        if st.button("🗑 Clear all", use_container_width=True,
                     help="Delete ALL stored memories"):
            n = clear_all_memories()
            st.success(f"Cleared {n} memory/memories")
            st.rerun()

    st.divider()

    # --- Examples ---
    st.header("💡 Examples")
    examples = [
        "What are the latest LangGraph features in 2025?",
        "Calculate the compound interest on 10000 SEK at 5% for 10 years",
        "Search for 3 AI companies in Stockholm and summarize what they do",
        "Write a short summary of today's top AI news and save it as notes.md",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending_input = ex


# ── Caption ────────────────────────────────────────────────────────────────────
st.caption(
    f"Model: `{st.session_state.selected_model}` · "
    "Tools: Brave Search · Python REPL · Browse Dir · Read File · Write MD · Transcribe Audio · Memory · RAG"
)

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_b64"):
            st.image(base64.b64decode(msg["image_b64"]), use_container_width=True)
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🔧 Tools used: {', '.join(msg['tools_used'])}")

# ── Chat input ─────────────────────────────────────────────────────────────────

# Image attachment — sits above the chat input bar
attached_image = st.file_uploader(
    "📎 Attach an image (optional — vision models only)",
    type=["png", "jpg", "jpeg", "webp", "gif"],
    key="chat_image_upload",
    label_visibility="collapsed",
)

user_input = st.chat_input("Ask me anything...")
if hasattr(st.session_state, "pending_input"):
    user_input = st.session_state.pending_input
    del st.session_state.pending_input

if user_input:
    # Build LangChain message — multimodal when an image is attached
    if attached_image is not None:
        img_bytes = attached_image.getvalue()
        img_b64   = base64.b64encode(img_bytes).decode()
        mime      = attached_image.type or "image/png"
        lc_content = [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]
        user_display = {"role": "user", "content": user_input,
                        "image_b64": img_b64, "image_mime": mime}
    else:
        lc_content   = user_input
        user_display = {"role": "user", "content": user_input}

    # Show user message immediately
    st.session_state.display_messages.append(user_display)
    with st.chat_message("user"):
        if attached_image is not None:
            st.image(attached_image, use_container_width=True)
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=lc_content)]},
                config=config,
            )

        reply = result["messages"][-1].content
        tools_used = []
        for m in result["messages"]:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tools_used.append(tc["name"])

        st.markdown(reply)
        if tools_used:
            st.caption(f"🔧 Tools used: {', '.join(set(tools_used))}")

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": reply,
        "tools_used": list(set(tools_used)),
    })

    # ── Auto-save meeting summary ──────────────────────────────────────────────
    # The model reliably transcribes but sometimes skips the write_md_file call.
    # When the transcription button was used, we save the reply ourselves instead
    # of depending on the model to chain a second tool call.
    if "auto_save_summary" in st.session_state:
        if "transcribe_audio" in tools_used:
            save_info = st.session_state.pop("auto_save_summary")
            summary_path = os.path.join(_BASE_DIR, save_info["path"])
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            md_content = (
                f"# Meeting Summary: {save_info['stem']}\n\n"
                f"**Generated:** {timestamp}\n\n"
                f"---\n\n{reply}\n"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            st.success(f"💾 Summary saved to `{save_info['path']}`")
        else:
            # Transcription wasn't called (model may have answered from memory),
            # clear the flag so it doesn't fire on the next unrelated message.
            del st.session_state["auto_save_summary"]

    # Auto-save chat after every exchange
    first = st.session_state.display_messages[0]["content"]
    save_chat(
        st.session_state.thread_id,
        first[:60] + ("…" if len(first) > 60 else ""),
        st.session_state.selected_model,
        st.session_state.display_messages,
    )
