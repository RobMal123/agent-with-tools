"""
app.py — Streamlit chat interface

Run with: streamlit run app.py
"""

import streamlit as st
import uuid
import json
import os
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # must run before graph/tools are imported so env vars are set
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import build_agent

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
    """Persist the current conversation to chats/<thread_id>.json."""
    path = os.path.join(CHATS_DIR, f"{thread_id}.json")
    data = {
        "thread_id": thread_id,
        "title": title,
        "model": model,
        "created_at": _existing_created_at(path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
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

    # Upload widget — saves directly to audio_in/
    uploaded = st.file_uploader(
        "Upload audio",
        type=["m4a", "mp3", "wav", "mp4", "ogg", "flac", "webm"],
        label_visibility="collapsed",
        key="audio_uploader",
    )
    if uploaded is not None and uploaded.name not in st.session_state.saved_uploads:
        os.makedirs(AUDIO_IN_DIR, exist_ok=True)
        dest = os.path.join(AUDIO_IN_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.saved_uploads.add(uploaded.name)
        st.rerun()  # refresh file list; uploader stays populated but won't re-save

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
    "Tools: Brave Search · Python REPL · Browse Dir · Read File · Write MD · Transcribe Audio"
)

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🔧 Tools used: {', '.join(msg['tools_used'])}")

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")
if hasattr(st.session_state, "pending_input"):
    user_input = st.session_state.pending_input
    del st.session_state.pending_input

if user_input:
    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
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
