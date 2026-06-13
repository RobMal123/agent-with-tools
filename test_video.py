"""
test_video.py — tests for the video understanding pipeline (video_understanding.py).

Run with:   pytest test_video.py -v
Full e2e:   set RUN_VIDEO_E2E=1, then pytest test_video.py -v
            (downloads a real 19s YouTube clip + the Moondream weights on first run)

Same design principle as test_agent.py — discriminating assertions, no mocks of the
real machinery:
    - scene detection  -> must find the cuts of a synthetic 3-scene video AT the
                          right timestamps (a flat-fps sampler or a broken parser
                          produces the wrong count/times)
    - Moondream OCR    -> must read an unguessable rendered string from an image
                          (a blind/captioning-only path can't produce it)
    - e2e              -> a real video must yield BOTH interleaved SPOKEN and VISUAL
                          lines, and mention what is actually said/shown in the clip

Heavy tests SKIP cleanly when their prerequisite (ffmpeg, torch, cached weights,
network opt-in) is missing, mirroring how test_agent.py skips without Ollama.
"""

import os
import re
import shutil
import subprocess
from importlib.util import find_spec

import pytest

import video_understanding as vu


_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_TORCH_STACK = all(find_spec(m) is not None for m in ("torch", "transformers", "PIL"))


def _moondream_available() -> bool:
    """True when the pinned Moondream weights are already in the HF cache, or the
    user explicitly allows the ~4 GB download (MOONDREAM_TEST_DOWNLOAD=1)."""
    if not _HAS_TORCH_STACK:
        return False
    if os.environ.get("MOONDREAM_TEST_DOWNLOAD", "") == "1":
        return True
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False
    folder = "models--" + vu.DEFAULT_MOONDREAM_MODEL.replace("/", "--")
    return os.path.isdir(os.path.join(HF_HUB_CACHE, folder))


requires_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
requires_moondream = pytest.mark.skipif(
    not _moondream_available(),
    reason="torch/transformers missing, or Moondream weights not cached "
           "(set MOONDREAM_TEST_DOWNLOAD=1 to allow the download)",
)
requires_e2e = pytest.mark.skipif(
    os.environ.get("RUN_VIDEO_E2E", "") != "1",
    reason="set RUN_VIDEO_E2E=1 to run the full network e2e (downloads a real video)",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_videos_dir(tmp_path, monkeypatch):
    """Point the output/cache dir at a throwaway folder. Works because
    video_understanding looks tools.VIDEOS_DIR up late, per call."""
    import tools
    monkeypatch.setattr(tools, "VIDEOS_DIR", str(tmp_path / "videos"))
    return tmp_path / "videos"


@pytest.fixture
def temp_vectorstore(tmp_path):
    """Same isolation as in test_agent.py: throwaway ChromaDB so auto-indexing
    of test output never pollutes the real semantic search index."""
    import tools
    old_dir   = tools._CHROMA_DIR
    old_cache = dict(tools._vectorstore_cache)
    tools._CHROMA_DIR = str(tmp_path / "chroma_test")
    tools._vectorstore_cache.clear()
    yield
    tools._vectorstore_cache.clear()
    tools._vectorstore_cache.update(old_cache)
    tools._CHROMA_DIR = old_dir


def _make_color_video(path, colors=("black", "white", "blue"), seconds_each=3):
    """Synthesize a video of solid-color scenes — guaranteed hard cuts at known
    timestamps (seconds_each, 2*seconds_each, ...).

    Colors must differ in LUMA: ffmpeg's scene score is luma-only, so a pure-chroma
    cut (e.g. CSS red→green, Y≈81 vs Y≈80) scores ~0 and is undetectable by design."""
    n = len(colors)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={c}:size=160x120:duration={seconds_each}:rate=10"]
    cmd += ["-filter_complex", "".join(f"[{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0",
            str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


# ════════════════════════════════════════════════════════════════════════════════
#  Pure unit tests (no ffmpeg, no models, no network) — fast
# ════════════════════════════════════════════════════════════════════════════════

def test_process_video_registered_as_tool():
    """The agent only sees tools listed in TOOLS — guard the registration."""
    from tools import TOOLS
    assert "process_video" in [t.name for t in TOOLS]


def test_fmt_ts():
    assert vu._fmt_ts(0) == "00:00"
    assert vu._fmt_ts(83) == "01:23"
    assert vu._fmt_ts(3725) == "1:02:05"   # hour-long videos get an hour field


def test_merge_timeline_interleaves_not_concatenates():
    """LOCKED DECISION regression: events must be ordered by timestamp, never
    grouped all-visual-then-all-spoken; at the same second VISUAL comes first."""
    spoken = [(0.5, 2.0, "hello"), (10.0, 12.0, "world")]
    visual = [(0.0, "title card"), (10.0, "slide two"), (20.0, "outro")]
    events = vu._merge_timeline(spoken, visual)
    assert [(k, t) for _, k, t in events] == [
        ("VISUAL", "title card"),
        ("SPOKEN", "hello"),
        ("VISUAL", "slide two"),   # same second as "world" → VISUAL first
        ("SPOKEN", "world"),
        ("VISUAL", "outro"),
    ]


def test_cache_key_is_filesystem_safe_and_stable():
    info = {"extractor_key": "Youtube", "id": "dQw4w9WgXcQ"}
    assert vu._cache_key(info) == "youtube-dQw4w9WgXcQ"
    nasty = vu._cache_key({"extractor_key": "TikTok", "id": 'a/b\\c:d*e?"<>|'})
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", nasty), nasty


def test_render_markdown_format(temp_videos_dir):
    """Output contract: frontmatter, hub wikilink, and '[MM:SS] KIND: text' lines."""
    info = {"title": "Demo clip", "duration": 75, "id": "abc123",
            "uploader": "Tester", "upload_date": "20260101"}
    events = [(0.0, "VISUAL", "a title card reading DEMO"),
              (3.0, "SPOKEN", "welcome to the demo"),
              (63.0, "SPOKEN", "that's all")]
    md = vu._render_markdown("https://example.com/v", info, events, "en", 1, 2)
    assert md.lstrip().startswith("---")                      # YAML frontmatter
    assert "[[Videos]]" in md                                 # hub wikilink
    assert "[00:00] VISUAL: a title card reading DEMO" in md
    assert "[00:03] SPOKEN: welcome to the demo" in md
    assert "[01:03] SPOKEN: that's all" in md
    # interleaved order preserved in the rendered body
    assert md.index("VISUAL: a title card") < md.index("SPOKEN: welcome")


def test_process_video_rejects_non_url():
    result = vu.process_video_impl("not a video url")
    assert result.startswith("Error")


def test_process_video_reports_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(vu, "_ffmpeg_exe", lambda: None)
    result = vu.process_video_impl("https://example.com/watch?v=x")
    assert result.startswith("Error") and "ffmpeg" in result


# ════════════════════════════════════════════════════════════════════════════════
#  ffmpeg scene detection — real subprocess, synthetic video, known cut times
# ════════════════════════════════════════════════════════════════════════════════

@requires_ffmpeg
def test_scene_detection_finds_cuts_at_right_times(tmp_path):
    """
    DISCRIMINATING: a 9s video with hard cuts at exactly 3s and 6s must yield
    exactly 3 keyframes — t≈0 (forced first frame), t≈3, t≈6. A flat-1fps sampler
    would return ~9; a broken showinfo parser returns wrong/zero timestamps.

    Uses the production default threshold. (This test caught two things during
    development: 0.3 was too high a default for partial-frame changes, and ffmpeg's
    scene metric is luma-only — hence luma-distinct scene colors here.)
    """
    video = _make_color_video(tmp_path / "scenes.mp4")
    frames = vu._extract_scene_frames(
        video, str(tmp_path / "frames"), duration=9.0,
        threshold=vu._scene_threshold(), max_frames=40, max_gap=60.0,
    )
    stamps = [ts for ts, _ in frames]
    assert len(frames) == 3, f"expected 3 keyframes (start + 2 cuts), got {stamps}"
    assert stamps[0] < 0.5, f"first frame should be ~0, got {stamps}"
    assert abs(stamps[1] - 3.0) < 0.6, f"first cut should be ~3s, got {stamps}"
    assert abs(stamps[2] - 6.0) < 0.6, f"second cut should be ~6s, got {stamps}"
    assert all(os.path.exists(p) for _, p in frames)


@requires_ffmpeg
def test_static_video_gets_anchor_frames_not_zero(tmp_path):
    """A static (no scene change) video must still get the first frame plus one
    anchor per max_gap seconds — the talking-head case."""
    video = _make_color_video(tmp_path / "static.mp4", colors=("gray",), seconds_each=9)
    frames = vu._extract_scene_frames(
        video, str(tmp_path / "frames"), duration=9.0,
        threshold=vu._scene_threshold(), max_frames=40, max_gap=4.0,
    )
    stamps = [ts for ts, _ in frames]
    assert len(frames) == 2, f"expected first frame + one 4s anchor, got {stamps}"
    assert stamps[0] < 0.5 and 3.0 <= stamps[1] <= 5.0, stamps


@requires_ffmpeg
def test_frame_cap_thins_evenly(tmp_path):
    """Above VIDEO_MAX_FRAMES the set is thinned evenly, always keeping frame 0."""
    video = _make_color_video(
        tmp_path / "many.mp4",
        # ordered for a big luma swing at every cut (see _make_color_video note)
        colors=("red", "yellow", "blue", "white", "green", "cyan", "black", "magenta"),
        seconds_each=1,
    )
    frames = vu._extract_scene_frames(
        video, str(tmp_path / "frames"), duration=8.0,
        threshold=vu._scene_threshold(), max_frames=4, max_gap=60.0,
    )
    assert len(frames) == 4
    assert frames[0][0] < 0.5                     # first frame survives thinning
    assert frames[-1][0] > 5.0                    # coverage reaches the end


# ════════════════════════════════════════════════════════════════════════════════
#  Moondream — real model, discriminating OCR (the vision-saga style guard)
# ════════════════════════════════════════════════════════════════════════════════

@requires_moondream
def test_moondream_reads_rendered_text(tmp_path):
    """
    DISCRIMINATING: Moondream must transcribe an unguessable string rendered into
    an image. A model that can't see — or a caption()-only path that ignores text —
    cannot produce 'ZEBRA 9314'.
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 320), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    draw.text((320, 160), "ZEBRA 9314", fill="black", anchor="mm", font=font)
    path = tmp_path / "ocr.png"
    img.save(path)

    visual = vu._describe_frames([(0.0, str(path))])
    assert len(visual) == 1
    text = visual[0][1].lower()
    assert "zebra" in text, f"Moondream did not read the word; got: {text!r}"
    assert "9314" in text, f"Moondream did not read the digits; got: {text!r}"


# ════════════════════════════════════════════════════════════════════════════════
#  End-to-end — real download, real Whisper, real Moondream (opt-in)
# ════════════════════════════════════════════════════════════════════════════════

# "Me at the zoo" — 19 seconds, English speech, public, stable since 2005.
_E2E_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


@requires_e2e
@requires_ffmpeg
@requires_moondream
def test_process_video_end_to_end(temp_videos_dir, temp_vectorstore):
    """
    DISCRIMINATING e2e: the pipeline must produce a file whose timeline contains
    BOTH spoken and visual lines, interleaved, and the content must reflect what
    the clip actually says/shows ('elephants') — impossible if download, Whisper,
    scene detection, Moondream, or the merge is broken.
    """
    result = vu.process_video_impl(_E2E_URL)
    assert not result.startswith("Error"), result
    m = re.search(r"Saved to: (.+\.md)", result)
    assert m, f"no saved path in result: {result[:300]}"
    path = m.group(1).strip()
    assert os.path.exists(path)

    content = open(path, encoding="utf-8").read()
    assert re.search(r"^\[\d{2}:\d{2}\] SPOKEN: ", content, re.M), "no spoken lines"
    assert re.search(r"^\[\d{2}:\d{2}\] VISUAL: ", content, re.M), "no visual lines"
    assert "elephant" in content.lower(), "transcript/visuals missed the elephants"

    # Cache contract: second call returns instantly from the saved file.
    again = vu.process_video_impl(_E2E_URL)
    assert "cache hit" in again.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
