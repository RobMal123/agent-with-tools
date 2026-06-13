"""
video_understanding.py — turn a social video (YouTube / TikTok / X) into ONE
timestamp-aligned markdown file describing both what is SAID and what is SHOWN,
so the agent can fact-check claims and surface concepts inside its normal ReAct loop.

Pipeline:
    url → yt-dlp download → [audio: faster-whisper] + [keyframes: ffmpeg scene-detect
    → Moondream query()] → merge on shared timeline → cache → file path

Design notes:
  - Frames are selected by SCENE CHANGE (ffmpeg select='gt(scene,T)'), not flat 1fps:
    a talking-head yields few frames, a slide-heavy video yields many. Long static
    stretches still get one anchor frame per VIDEO_FRAME_MAX_GAP seconds.
  - Each kept frame goes through Moondream's query() with a text-extraction prompt
    (not generic caption()) so burned-in captions / slides / overlays are transcribed.
    Moondream's OCR is English-optimised — Swedish on-screen text may be weaker; the
    Whisper audio side is multilingual and unaffected.
  - Whisper segments and frames share the same clock (seconds from video start) and
    are INTERLEAVED in the output, never concatenated by type.
  - Output is cached in KNOWLEDGE_DIR/videos/ keyed by extractor + video ID; the
    same clip is never reprocessed.
  - All heavy deps (yt_dlp, torch, transformers) are imported lazily inside functions:
    importing this module — and therefore tools.py — stays cheap and works even when
    they are not installed. tools is also only imported inside functions, so there is
    no import cycle with the @tool wrapper in tools.py.
  - Like every agent tool in this project, the public entry point returns an
    "Error: ..." string instead of raising — an exception would crash the whole run.
"""

import gc
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime

# Pinned by default — Moondream pushes breaking updates to `main` frequently.
# Swap to Moondream 3 Preview (9B MoE) via MOONDREAM_MODEL=moondream/moondream3-preview
# (same query() interface; needs more VRAM and a HF token while gated).
DEFAULT_MOONDREAM_MODEL    = "vikhyatk/moondream2"
DEFAULT_MOONDREAM_REVISION = "2025-06-21"

# Targeted prompt (locked decision): transcribe on-screen text first, then describe.
FRAME_PROMPT = ("Transcribe any text visible in the image, then briefly "
                "describe what is shown.")


# ── Tunables (read per call so tests/env can adjust without re-import) ────────────

def _scene_threshold() -> float:
    """ffmpeg scene-change score (0–1, luma-weighted frame diff) a frame must exceed
    to be kept. Hard camera cuts score ~0.3–0.8, but slide/caption changes — where
    most of the frame stays identical — often score only ~0.05–0.2, so the default
    is low. Over-firing on shaky footage is harmless: VIDEO_MAX_FRAMES + even
    thinning degrade it to uniform sampling, while static video stays sparse."""
    return float(os.environ.get("VIDEO_SCENE_THRESHOLD", "0.08"))


def _max_frames() -> int:
    """Hard cap on keyframes per video (each one is a Moondream call)."""
    return int(os.environ.get("VIDEO_MAX_FRAMES", "40"))


def _frame_max_gap() -> float:
    """Longest stretch without a keyframe before an anchor frame is grabbed —
    keeps long static (talking-head) sections from going visually unobserved."""
    return float(os.environ.get("VIDEO_FRAME_MAX_GAP", "60"))


def _max_duration() -> float:
    """Refuse videos longer than this many seconds (download + Whisper get slow)."""
    return float(os.environ.get("VIDEO_MAX_DURATION", "3600"))


def _videos_dir() -> str:
    """Cache/output dir — looked up late so tests can repoint tools.VIDEOS_DIR."""
    import tools
    os.makedirs(tools.VIDEOS_DIR, exist_ok=True)
    return tools.VIDEOS_DIR


def _ffmpeg_exe() -> str | None:
    return shutil.which("ffmpeg")


# ── 1) Fetch (yt-dlp) ─────────────────────────────────────────────────────────────

def _probe(url: str) -> dict:
    """Metadata only (no download): id/extractor for the cache key, duration, title."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    # Some share links still resolve to a single-entry playlist wrapper.
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise ValueError("URL resolved to an empty playlist")
        info = entries[0]
    return info


def _cache_key(info: dict) -> str:
    """Stable per-video file stem: <extractor>-<video id>, filesystem-safe."""
    extractor = str(info.get("extractor_key") or info.get("extractor") or "video").lower()
    vid = str(info.get("id") or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{extractor}-{vid}")[:100]


def _download(url: str, tmpdir: str) -> str:
    """Download once (≤720p to keep decode fast); audio and frames share this file,
    which is what guarantees the two halves of the timeline share one clock."""
    import yt_dlp
    opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        "outtmpl": os.path.join(tmpdir, "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        requested = (info.get("requested_downloads") or [{}])[0]
        path = requested.get("filepath") or ydl.prepare_filename(info)
    if not os.path.exists(path):
        # prepare_filename can predict the pre-merge extension; take what's on disk.
        candidates = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
        candidates = [c for c in candidates if os.path.isfile(c)]
        if not candidates:
            raise FileNotFoundError("yt-dlp finished but no media file was found")
        path = max(candidates, key=os.path.getsize)
    return path


# ── 2) Audio → timestamped text (existing Whisper setup) ──────────────────────────

def _transcribe(video_path: str) -> tuple[list[tuple[float, float, str]], str]:
    """Whisper segments [(start, end, text), ...] + detected language. faster-whisper
    decodes the audio track straight out of the mp4 (PyAV), no separate extraction.
    vad_filter skips music/silence-only stretches that otherwise hallucinate text."""
    from tools import _get_whisper_model
    model, err = _get_whisper_model()
    if err:
        raise RuntimeError(err)
    segments, info = model.transcribe(video_path, beam_size=5, vad_filter=True)
    spoken: list[tuple[float, float, str]] = []
    for seg in segments:
        text = " ".join((seg.text or "").split())
        if text:
            spoken.append((float(seg.start), float(seg.end), text))
    return spoken, (getattr(info, "language", "") or "")


# ── 3) Frame selection (scene detection, self-scaling) ────────────────────────────

_SHOWINFO_PTS = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def _extract_scene_frames(video_path: str, frames_dir: str, duration: float,
                          threshold: float, max_frames: int,
                          max_gap: float) -> list[tuple[float, str]]:
    """Keep the first frame plus every frame whose scene-change score exceeds
    `threshold`; backfill one anchor per `max_gap` seconds of static video; thin
    evenly to `max_frames`. Returns [(timestamp_seconds, jpg_path), ...] sorted."""
    ffmpeg = _ffmpeg_exe()
    os.makedirs(frames_dir, exist_ok=True)

    # showinfo logs pts_time for each frame that survives select — that is the
    # frame's position on the shared timeline. Single quotes are consumed by
    # ffmpeg's own filter-graph parser (no shell involved), protecting the commas.
    vf = f"select='eq(n,0)+gt(scene,{threshold})',showinfo"
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-i", video_path,
         "-vf", vf, "-fps_mode", "vfr", "-q:v", "3",
         os.path.join(frames_dir, "scene_%05d.jpg")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed: {(proc.stderr or '')[-400:]}")

    stamps = [float(s) for s in _SHOWINFO_PTS.findall(proc.stderr or "")]
    files = sorted(f for f in os.listdir(frames_dir) if f.startswith("scene_"))
    frames = [(ts, os.path.join(frames_dir, name)) for ts, name in zip(stamps, files)]

    # Backfill: one anchor frame per max_gap seconds where nothing changed enough
    # to trigger the scene filter (e.g. a 10-minute talking head).
    fill_ts: list[float] = []
    known = [ts for ts, _ in frames] or [0.0]
    bounds = known + ([duration] if duration and duration > known[-1] else [])
    for a, b in zip(bounds, bounds[1:]):
        t = a + max_gap
        while t < b - 1.0:
            fill_ts.append(t)
            t += max_gap
    for j, t in enumerate(fill_ts):
        out = os.path.join(frames_dir, f"fill_{j:05d}.jpg")
        grab = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-ss", f"{t:.2f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "3", out],
            capture_output=True,
        )
        if grab.returncode == 0 and os.path.exists(out):
            frames.append((t, out))

    frames.sort(key=lambda f: f[0])
    if len(frames) > max_frames:
        # Thin evenly across the video, always keeping the first frame.
        step = (len(frames) - 1) / (max_frames - 1)
        keep = sorted({round(i * step) for i in range(max_frames)})
        frames = [frames[i] for i in keep]
    return frames


# ── 4) Frame → text (Moondream, local HF transformers) ────────────────────────────

def _load_moondream():
    import torch
    from transformers import AutoModelForCausalLM
    model_id = os.environ.get("MOONDREAM_MODEL", DEFAULT_MOONDREAM_MODEL)
    revision = os.environ.get("MOONDREAM_REVISION", DEFAULT_MOONDREAM_REVISION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval()
    return model


def _describe_frames(frames: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Run Moondream's query() (targeted text-extraction prompt) on every kept frame.
    The model is loaded per video and released afterwards so it doesn't sit on VRAM
    next to the Ollama chat model between calls."""
    if not frames:
        return []
    from PIL import Image
    model = _load_moondream()
    visual: list[tuple[float, str]] = []
    try:
        for ts, path in frames:
            try:
                image = Image.open(path).convert("RGB")
                answer = model.query(image, FRAME_PROMPT)["answer"]
            except Exception as e:        # one bad frame must not sink the video
                answer = f"(frame analysis failed: {e})"
            text = " ".join(str(answer).split())
            if len(text) > 600:
                text = text[:600].rstrip() + "…"
            visual.append((ts, text))
    finally:
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return visual


# ── 5) Merge on the shared timeline + render ──────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _merge_timeline(spoken: list[tuple[float, float, str]],
                    visual: list[tuple[float, str]]) -> list[tuple[float, str, str]]:
    """Interleave spoken segments and frame descriptions by timestamp (never
    concatenated by type). At the same second, VISUAL sorts first — the scene sets
    the context for the words spoken over it."""
    events = [(ts, "VISUAL", text) for ts, text in visual]
    events += [(start, "SPOKEN", text) for start, _end, text in spoken]
    events.sort(key=lambda e: (e[0], 0 if e[1] == "VISUAL" else 1))
    return events


def _render_markdown(url: str, info: dict, events: list[tuple[float, str, str]],
                     language: str, n_frames: int, n_segments: int) -> str:
    from tools import _frontmatter, _hub_link
    title = " ".join(str(info.get("title") or "Untitled video").split())
    duration = float(info.get("duration") or 0)
    uploader = str(info.get("uploader") or info.get("channel") or "").strip()
    upload_date = str(info.get("upload_date") or "")          # YYYYMMDD from yt-dlp
    if re.fullmatch(r"\d{8}", upload_date):
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    processed = datetime.now().strftime("%Y-%m-%d %H:%M")

    extra = {"video_id": str(info.get("id") or ""), "duration": _fmt_ts(duration)}
    if uploader:
        extra["uploader"] = uploader
    if upload_date:
        extra["published"] = upload_date
    head = _frontmatter(title=f"Video: {title}", note_type="video",
                        tags=["video"], source=url, extra=extra)

    meta = [f"**URL:** {url}"]
    if uploader:
        meta.append(f"**Uploader:** {uploader}" + (f" · {upload_date}" if upload_date else ""))
    meta.append(f"**Duration:** {_fmt_ts(duration)} | **Processed:** {processed}")
    meta.append(f"**Audio language:** {language or 'none detected'}")
    meta.append(f"**Timeline:** {n_segments} spoken segment(s) · {n_frames} keyframe(s), "
                "scene-change selected")

    if events:
        timeline = "\n".join(f"[{_fmt_ts(ts)}] {kind}: {text}" for ts, kind, text in events)
    else:
        timeline = "(no speech detected and no distinct visuals found)"

    return (
        head
        + f"# Video: {title}\n\n"
        + f"{_hub_link('video')}\n\n"
        + "  \n".join(meta) + "\n\n"
        + "> SPOKEN = Whisper transcript of the audio track. VISUAL = on-screen text +\n"
        + "> scene description of a keyframe (may duplicate burned-in captions).\n"
        + "> Timestamps are [MM:SS] from the start of the video.\n\n"
        + "## Timeline\n\n"
        + timeline + "\n"
    )


def _truncate(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[... truncated — use read_file on the saved path for the full version]"


# ── Entry point (wrapped by the process_video tool in tools.py) ────────────────────

def process_video_impl(url: str) -> str:
    """url → path to the timestamp-aligned description file (plus its content).
    Cached by video ID: the same clip is never downloaded or analysed twice."""
    url = (url or "").strip().strip('"').strip("'")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "Error: process_video needs an http(s) video URL (YouTube / TikTok / X)."

    if _ffmpeg_exe() is None:
        return ("Error: ffmpeg was not found on PATH — it is required for keyframe "
                "extraction and stream merging. Install ffmpeg, then retry.")
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return "Error: yt-dlp is not installed. Run: pip install yt-dlp"

    # Identity first (no download): cache hit must cost nothing.
    try:
        info = _probe(url)
    except Exception as e:
        return f"Error: could not read video metadata ({e})"

    out_path = os.path.join(_videos_dir(), _cache_key(info) + ".md")
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                cached = f.read()
        except Exception:
            cached = ""
        return (f"Already processed (cache hit). Saved at: {out_path}\n\n---\n\n"
                + _truncate(cached))

    duration = float(info.get("duration") or 0)
    if duration and duration > _max_duration():
        return (f"Error: video is {_fmt_ts(duration)} long, over the "
                f"{_fmt_ts(_max_duration())} limit (set VIDEO_MAX_DURATION to override).")

    tmpdir = tempfile.mkdtemp(prefix="video_understanding_")
    try:
        print(f"[process_video] downloading: {url}")
        video_path = _download(url, tmpdir)

        print("[process_video] transcribing audio (whisper)…")
        spoken, language = _transcribe(video_path)

        print("[process_video] selecting keyframes (ffmpeg scene detection)…")
        frames = _extract_scene_frames(
            video_path, os.path.join(tmpdir, "frames"), duration,
            _scene_threshold(), _max_frames(), _frame_max_gap(),
        )

        print(f"[process_video] describing {len(frames)} keyframe(s) (moondream)…")
        visual = _describe_frames(frames)

        events = _merge_timeline(spoken, visual)
        md = _render_markdown(url, info, events, language, len(visual), len(spoken))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as e:
        return f"Error: video processing failed ({e})"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    try:
        from tools import _auto_index
        _auto_index(out_path, "video")                 # non-fatal, like other tools
    except Exception:
        pass

    return f"Video processed. Saved to: {out_path}\n\n---\n\n" + _truncate(md)
