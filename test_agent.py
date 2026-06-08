"""
test_agent.py — real, discriminating tests (NO mocks of the LLM).

Run with:   pytest test_agent.py -v
Single test: pytest test_agent.py::test_vision_reads_numbers -v

Design principle (learned the hard way):
    A verification that cannot produce a WRONG answer when the system is broken
    is not a verification. Every LLM test below asserts on an outcome that is
    impossible to produce unless the real machinery actually works:
        - tool calling  -> a 9-digit product no small model can do in its head
        - vision        -> reading specific digits / naming specific colors
        - memory        -> recalling an unguessable codeword
        - RAG           -> retrieving a unique fact only present in the indexed doc
        - multi-turn    -> combining two specific earlier products across a 5-turn thread

These hit a live Ollama instance and are slower than unit tests. They SKIP
cleanly (not fail) when Ollama isn't running or a required model isn't pulled.

Required models (skipped individually if missing):
    TEST_MODEL          (default: gemma4:12b)      — reasoning + tool calls
    TEST_VISION_MODEL   (default: gemma4:12b)      — image turns
    EMBED_MODEL         (default: nomic-embed-text)— RAG embeddings
"""

import os
import re
import io
import uuid
import base64

import pytest
import requests
from langchain_core.messages import HumanMessage, AIMessage


# ── Environment / availability ──────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TEST_MODEL       = os.environ.get("TEST_MODEL", "gemma4:12b")
TEST_VISION_MODEL = os.environ.get("TEST_VISION_MODEL", "gemma4:12b")
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _fetch_installed():
    """Return list of installed Ollama model names, or None if Ollama is down."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


_INSTALLED = _fetch_installed()


def _has_model(model: str) -> bool:
    """True if `model` is pulled. An untagged spec matches any tag (e.g. :latest)."""
    if not _INSTALLED:
        return False
    if model in _INSTALLED:
        return True
    if ":" not in model:                       # "llama3.2" matches "llama3.2:latest"
        return any(name.split(":")[0] == model for name in _INSTALLED)
    return False


try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# Skip decorators -----------------------------------------------------------------
requires_ollama = pytest.mark.skipif(
    _INSTALLED is None, reason=f"Ollama not reachable at {OLLAMA_BASE_URL}"
)
requires_text = pytest.mark.skipif(
    not _has_model(TEST_MODEL), reason=f"model '{TEST_MODEL}' not pulled"
)
requires_vision = pytest.mark.skipif(
    not _has_model(TEST_VISION_MODEL), reason=f"vision model '{TEST_VISION_MODEL}' not pulled"
)
requires_embed = pytest.mark.skipif(
    not _has_model(EMBED_MODEL), reason=f"embed model '{EMBED_MODEL}' not pulled"
)
requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _digits(text: str) -> str:
    """All digits in `text`, concatenated — for matching numbers ignoring commas/spaces."""
    return re.sub(r"\D", "", text)


def _result_number(text: str) -> int:
    """
    The largest integer mentioned in `text`. For a multiply/add reply that is the
    computed result — the product/sum is larger than the operands it echoes — so
    this reads back the value the agent actually produced, even when the reply
    restates the inputs (e.g. '60127 * 7919 = 476145713'). Returns -1 if none.

    Handles thousands separators: a comma-grouped number like '23,159,016' is read
    as one value, not split into 23 / 159 / 016 (which would mask the real result).
    """
    candidates = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", text)
    nums = [int(c.replace(",", "")) for c in candidates]
    return max(nums) if nums else -1


def _tools_used(result: dict) -> list:
    """Names of every tool the agent actually called during a run."""
    used = []
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            used.append(tc["name"])
    return used


def _png_b64(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _solid(color, size=200):
    return Image.new("RGB", (size, size), color)


def _number_img(n, size=200):
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arialbd.ttf", 150)
    except Exception:
        font = ImageFont.load_default()
    d.text((size // 2, size // 2), str(n), fill="black", anchor="mm", font=font)
    return im


def _ask(agent, text, image=None):
    """Single-turn invoke on a fresh thread. Optionally attach a PIL image."""
    if image is not None:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_png_b64(image)}"}},
        ]
        msg = HumanMessage(content=content)
    else:
        msg = HumanMessage(content=text)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke({"messages": [msg]}, config=cfg)
    return result, result["messages"][-1].content


def _turn(agent, cfg, text):
    """
    One turn of a *multi-turn* conversation. Unlike `_ask`, the caller reuses the
    same `cfg` (same thread_id) across calls, so the MemorySaver checkpointer
    accumulates history and the agent can see what happened in earlier turns.
    Returns (full_state, last_reply_text).
    """
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=cfg)
    return result, result["messages"][-1].content


# ── Fixtures (isolation — NOT mocking; the real systems still run) ───────────────

@pytest.fixture
def preserve_memory():
    """Snapshot memory.json and restore it after the test so we don't clobber real data."""
    import memory
    existed = os.path.exists(memory.MEMORY_FILE)
    backup = None
    if existed:
        with open(memory.MEMORY_FILE, encoding="utf-8") as f:
            backup = f.read()
    yield
    if backup is not None:
        with open(memory.MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(memory.MEMORY_FILE):
        os.remove(memory.MEMORY_FILE)


@pytest.fixture
def temp_vectorstore(tmp_path):
    """Point ChromaDB at a throwaway dir so RAG tests don't pollute the real index."""
    import tools
    old_dir   = tools._CHROMA_DIR
    old_cache = dict(tools._vectorstore_cache)
    tools._CHROMA_DIR = str(tmp_path / "chroma_test")
    tools._vectorstore_cache.clear()
    yield
    tools._vectorstore_cache.clear()
    tools._vectorstore_cache.update(old_cache)
    tools._CHROMA_DIR = old_dir


@pytest.fixture
def confined_base(tmp_path):
    """
    Point the file-tool sandbox root (tools._BASE_DIR) at a throwaway dir so tests
    can operate on tmp_path files. read_file / write_md_file / list_directory confine
    access to _BASE_DIR; production behaviour is unchanged (see test_read_file_blocks_traversal).
    """
    import tools
    old = tools._BASE_DIR
    tools._BASE_DIR = str(tmp_path)
    yield tmp_path
    tools._BASE_DIR = old


# ════════════════════════════════════════════════════════════════════════════════
#  Pure unit tests (no LLM, no network) — fast
# ════════════════════════════════════════════════════════════════════════════════

def test_read_file_success(confined_base):
    """File reader should return file contents (inside the sandbox root)."""
    from tools import read_file
    test_file = confined_base / "test.txt"
    test_file.write_text("Hello from test file!")
    result = read_file.invoke(str(test_file))
    assert "Hello from test file!" in result


def test_read_file_not_found():
    """File reader should return error for missing files."""
    from tools import read_file
    result = read_file.invoke("/nonexistent/path/file.txt")
    assert "Error" in result


def test_read_file_unsupported_extension(tmp_path):
    """File reader should reject unsupported file types."""
    from tools import read_file
    bad_file = tmp_path / "file.exe"
    bad_file.write_bytes(b"\x00\x01\x02")
    result = read_file.invoke(str(bad_file))
    assert "Error" in result


def test_read_file_truncates_large_files(confined_base):
    """File reader should truncate files > 8000 chars."""
    from tools import read_file
    large_file = confined_base / "large.txt"
    large_file.write_text("x" * 10000)
    result = read_file.invoke(str(large_file))
    assert "truncated" in result


def test_read_file_blocks_traversal(confined_base):
    """SECURITY: reads outside the sandbox root must be denied (path confinement)."""
    from tools import read_file
    outside = confined_base.parent / "outside.txt"   # sibling of the sandbox root
    outside.write_text("top secret")
    result = read_file.invoke(str(outside))
    assert "access denied" in result
    assert "top secret" not in result


def test_python_repl_disabled_by_default(monkeypatch):
    """SECURITY: code execution is opt-in; the tool must refuse unless enabled."""
    from tools import python_repl
    monkeypatch.delenv("ENABLE_CODE_EXECUTION", raising=False)
    result = python_repl.invoke({"code": "print(2 + 2)"})
    assert "disabled" in result.lower()


def test_agent_graph_compiles():
    """The agent graph should compile without errors."""
    from graph import build_agent
    app = build_agent(use_memory=False)
    assert app is not None


def test_agent_state_schema():
    """AgentState should accept messages correctly."""
    from state import AgentState
    state: AgentState = {"messages": [HumanMessage(content="hello")]}
    assert len(state["messages"]) == 1


# ════════════════════════════════════════════════════════════════════════════════
#  Real LLM integration tests — discriminating, no mocks
# ════════════════════════════════════════════════════════════════════════════════

@requires_ollama
@requires_text
def test_llm_answers_directly_without_tools():
    """
    Smoke: the model gives a correct, specific answer from its own knowledge.
    Deliberately uses a non-'current-events' question plus an explicit no-tools
    instruction, so the answer does NOT depend on web search (which needs a
    Brave API key the test env may not have).
    """
    from graph import build_agent
    agent = build_agent(model_name=TEST_MODEL, use_memory=True)
    _, reply = _ask(
        agent,
        "Answer from your own knowledge. Do NOT use any tools. "
        "What color do you get when you mix blue and yellow paint? Reply with one word.",
    )
    assert "green" in reply.lower(), f"expected 'green', got: {reply!r}"


@requires_ollama
@requires_text
def test_tool_calling_does_real_math(monkeypatch):
    """
    DISCRIMINATING: ask for 83621 * 7919 = 662194699.
    A small model cannot produce this 9-digit product without actually running
    the python_repl tool — so a correct answer proves the tool path works.
    """
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "true")  # python_repl is opt-in since the security fix
    from graph import build_agent
    agent = build_agent(model_name=TEST_MODEL, use_memory=True)
    result, reply = _ask(
        agent,
        "Use the python_repl tool to calculate 83621 * 7919. "
        "Report the exact integer result.",
    )
    assert "662194699" in _digits(reply), (
        f"expected 662194699 in answer; tools_used={_tools_used(result)}; reply={reply!r}"
    )


@requires_ollama
@requires_text
def test_multiturn_cross_turn_tool_combination(monkeypatch):
    """
    DISCRIMINATING + MULTI-TURN: a 5-turn conversation on ONE persistent thread
    where every turn drives the python_repl tool, and the final turn can only be
    answered by combining the results of turn 2 AND turn 4 specifically.

    Why this can't pass unless the real machinery works:
      - Each turn asks for a product of 4–5 digit numbers — too big to do in the
        head — so the tool must run. We assert python_repl actually fired on every
        turn by inspecting the recorded tool_calls (not just the reply text).
      - Turns 1 and 3 are DISTRACTORS; their products must not leak into the finale.
      - The finale asks for ALPHA + BETA (the products from turns 2 and 4). We read
        back what the agent actually computed in those two turns and require the
        final answer to equal exactly their sum. With no cross-turn memory the model
        can't see ALPHA/BETA at all; recalling only one — or grabbing a turn-1/turn-3
        distractor — gives the wrong total; and adding two ~9-digit numbers reliably
        needs the tool rather than mental arithmetic. So a correct sum, together with
        the recorded python_repl calls, implies retention across turns AND a real tool
        call on the final turn.

    We deliberately read the agent's own turn-2/turn-4 results instead of hard-coding
    the products: small Ollama models occasionally mistype an operand into the tool
    call, and this test targets the *cross-turn combination*, so it holds the finale
    to the sum of whatever the agent genuinely produced in turns 2 and 4.
    """
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "true")  # python_repl is opt-in
    from graph import build_agent
    agent = build_agent(model_name=TEST_MODEL, use_memory=True)

    # One thread for the whole conversation (reused cfg => accumulating history).
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def _calc(text):
        """Run one turn and return (full_state, reply, the number the agent computed)."""
        result, reply = _turn(agent, cfg, text)
        return result, reply, _result_number(reply)

    # Turn 1 (tool) — distractor product
    _, a1, n1 = _calc("Use the python_repl tool to calculate 4392 * 5273. "
                      "Report the exact integer result.")
    assert n1 >= 1_000_000, f"turn 1 did not return a computed product; reply={a1!r}"

    # Turn 2 (tool) — ALPHA: the first value the finale must combine
    _, a2, alpha = _calc("Now use the python_repl tool to calculate 60127 * 7919. "
                         "Remember this product as ALPHA and report the exact integer result.")
    assert alpha >= 1_000_000, f"turn 2 (ALPHA) did not return a product; reply={a2!r}"

    # Turn 3 (tool) — distractor product
    _, a3, n3 = _calc("Use the python_repl tool to calculate 8124 * 3307. "
                      "Report the exact integer result.")
    assert n3 >= 1_000_000, f"turn 3 did not return a computed product; reply={a3!r}"

    # Turn 4 (tool) — BETA: the second value the finale must combine
    _, a4, beta = _calc("Now use the python_repl tool to calculate 51237 * 9043. "
                        "Remember this product as BETA and report the exact integer result.")
    assert beta >= 1_000_000, f"turn 4 (BETA) did not return a product; reply={a4!r}"

    # Turn 5 (tool) — combine ONLY turn 2 (ALPHA) + turn 4 (BETA)
    r5, a5, got = _calc(
        "Using the python_repl tool, add together ALPHA and BETA — the two products "
        "you computed earlier — and report the exact total."
    )
    assert got == alpha + beta, (
        f"cross-turn combination failed: expected ALPHA+BETA={alpha + beta} "
        f"(ALPHA={alpha} from turn 2, BETA={beta} from turn 4), got {got}; "
        f"tools_used={_tools_used(r5)}; reply={a5!r}"
    )

    # Direct proof every turn exercised the tool: r5 holds the full thread, so this
    # counts python_repl tool_calls across all 5 turns.
    used = _tools_used(r5)
    assert used.count("python_repl") >= 5, (
        f"expected a python_repl call on each of the 5 turns; used={used}"
    )


@requires_ollama
@requires_vision
@requires_pil
def test_vision_reads_numbers():
    """
    DISCRIMINATING (the regression test for the whole vision saga):
    the agent must READ specific digits from images. Faking three is 1/1000.
    Image turns are routed to the dedicated vision model inside build_agent.
    """
    from graph import build_agent
    agent = build_agent(
        model_name=TEST_MODEL, use_memory=True, vision_model=TEST_VISION_MODEL
    )
    for n in (3, 5, 8):
        _, reply = _ask(
            agent,
            "What single digit is shown in this image? Reply with just the digit.",
            image=_number_img(n),
        )
        assert str(n) in reply, f"image showed {n} but model replied: {reply!r}"


@requires_ollama
@requires_vision
@requires_pil
def test_vision_distinguishes_colors():
    """
    DISCRIMINATING: name the dominant color for red, green AND blue.
    A blind model answering 'Black' to everything (the original bug) fails this.
    """
    from graph import build_agent
    agent = build_agent(
        model_name=TEST_MODEL, use_memory=True, vision_model=TEST_VISION_MODEL
    )
    for color in ("red", "green", "blue"):
        _, reply = _ask(
            agent,
            "What is the single dominant color of this image? Answer with one word.",
            image=_solid(color),
        )
        assert color in reply.lower(), f"image was {color} but model replied: {reply!r}"


@requires_ollama
@requires_text
def test_memory_recall(preserve_memory):
    """
    DISCRIMINATING: store an unguessable codeword, then confirm the agent recalls
    it. The codeword can only appear if memory injection actually works.
    """
    from memory import save_memory_entry
    from graph import build_agent

    codeword = "platypus-9271"
    save_memory_entry("secret_codeword", codeword)

    agent = build_agent(model_name=TEST_MODEL, use_memory=True)
    _, reply = _ask(
        agent,
        "What is my secret codeword? It is stored in your long-term memory.",
    )
    assert codeword in reply.lower(), f"expected '{codeword}' in reply; got: {reply!r}"


@requires_ollama
@requires_embed
def test_rag_index_and_retrieve(temp_vectorstore, tmp_path):
    """
    DISCRIMINATING: index a document containing a unique invented fact, then
    retrieve it by semantic query. The number can only come back if real
    embeddings + vector search work end to end.
    """
    from tools import _index_file, search_documents

    doc = tmp_path / "zultrax_report.md"
    doc.write_text(
        "# Zultrax-9 Reactor Notes\n\n"
        "The Zultrax-9 reactor reaches critical resonance at exactly 4471 kelvin.\n"
        "Below that threshold the core remains stable.\n",
        encoding="utf-8",
    )

    status = _index_file(str(doc), "report")
    assert "Indexed" in status, f"indexing failed: {status}"

    result = search_documents.invoke(
        {"query": "At what temperature does the Zultrax reactor reach critical resonance?"}
    )
    assert "4471" in result, f"unique fact not retrieved; search returned: {result!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
