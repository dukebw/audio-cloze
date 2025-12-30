#!/usr/bin/env python3
"""Evaluate ASR accuracy across backends with visual comparison interface."""

import argparse
import html as html_lib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
import hashlib
from asr_utils import (
    ensure_qwen3_weights,
    load_glm_asr_transformers,
    patch_funasr_load_in_8bit,
    select_asr_device,
)

# Configuration
SAMPLE_DURATION = 45  # seconds per sample
NUM_YOUTUBE_SAMPLES = 18
NUM_PODCAST_SAMPLES = 12
TOTAL_SAMPLES = NUM_YOUTUBE_SAMPLES + NUM_PODCAST_SAMPLES

WHISPER_MODEL_PATH = Path(__file__).parent / "models" / "ggml-large-v3.bin"
FUNASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
GLM_ASR_MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
EVAL_DIR = Path(__file__).parent / "eval_samples"
DEFAULT_AUDIO_SUBDIR = "audio"
EP499_SAMPLE_START = 610  # contains "留下兩個懸念" in EP499 TAXI DRIVER
RESULTS_FILE = EVAL_DIR / "eval_results.json"
HTML_FILE = EVAL_DIR / "eval_compare.html"
WHISPERX_METHOD = "whisperx"
DEFAULT_ALIGNMENT_ORDER = (WHISPERX_METHOD,)

_FUNASR_MODEL = None
_GLM_ASR_MODEL = None
_GLM_ASR_PROCESSOR = None
_WHISPERX_ALIGN_MODEL = None
_WHISPERX_ALIGN_META = None
_WHISPERX_PATCHED = False
_STABLE_TS_MODEL = None


@dataclass
class Sample:
    """An audio sample for evaluation."""
    sample_id: str
    source_type: str  # 'youtube' or 'podcast'
    source_id: str
    title: str
    channel: str
    audio_path: str
    start_time: float
    duration: float


@dataclass
class TranscriptionResult:
    """Result from an ASR backend."""
    backend: str
    text: str
    elapsed_seconds: float
    word_count: int
    audio_duration: float = SAMPLE_DURATION  # For realtime calculation
    error: Optional[str] = None
    alignments: Optional[dict] = None

    @property
    def realtime_speed(self) -> float:
        """Calculate realtime speed multiple (e.g., 10x means 10x faster than realtime)."""
        if self.elapsed_seconds > 0 and not self.error:
            return self.audio_duration / self.elapsed_seconds
        return 0.0


@dataclass(frozen=True)
class BackendSpec:
    """Metadata for a transcription backend."""
    key: str
    label: str
    css_class: str
    transcribe: Callable[[Path], "TranscriptionResult"]


@dataclass
class EvalResult:
    """Complete evaluation result for a sample."""
    sample: Sample
    transcriptions: dict  # backend -> TranscriptionResult


def resolve_audio_dir(audio_dir_arg: str) -> Path:
    audio_dir = Path(audio_dir_arg).expanduser()
    if not audio_dir.is_absolute():
        audio_dir = EVAL_DIR / audio_dir
    return audio_dir


def relative_audio_path(audio_path: Path) -> str:
    try:
        return str(audio_path.relative_to(EVAL_DIR))
    except ValueError:
        return str(audio_path)


def resolve_audio_path(audio_path: str, audio_dir: Path) -> Path:
    path = Path(audio_path).expanduser()
    if path.is_absolute():
        return path
    candidate = EVAL_DIR / path
    if candidate.exists():
        return candidate
    return audio_dir / path.name


def ensure_sample_audio(sample: Sample, audio_path: Path,
                        conn: Optional[sqlite3.Connection]) -> bool:
    """Re-download missing audio for an existing sample."""
    if audio_path.exists():
        return True

    if sample.source_type == "youtube":
        print(f"  Re-downloading YouTube audio: {audio_path.name}")
        return download_youtube_sample(
            sample.source_id, sample.start_time, sample.duration, audio_path
        )

    if sample.source_type == "podcast":
        if conn is None:
            print("  Missing database connection; cannot re-download podcast audio")
            return False
        row = conn.execute(
            "SELECT audio_url FROM episodes WHERE episode_id = ?",
            (sample.source_id,)
        ).fetchone()
        if not row or not row[0]:
            print(f"  Missing audio URL for episode {sample.source_id[:8]}")
            return False
        print(f"  Re-downloading podcast audio: {audio_path.name}")
        return download_podcast_sample(
            row[0], sample.start_time, sample.duration, audio_path
        )

    return False


def add_explicit_ep499_sample(results: list[EvalResult],
                              conn: sqlite3.Connection,
                              audio_dir: Path,
                              backend_specs: list[BackendSpec]) -> None:
    """Ensure EP499 sample is included for side-by-side comparison."""
    row = conn.execute(
        """
        SELECT episode_id, title, channel_name, audio_url, duration
        FROM episodes
        WHERE title LIKE '%EP499%'
        ORDER BY pub_date DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        print("WARNING: EP499 episode not found in database.")
        return

    episode_id, title, channel, audio_url, duration = row
    if not audio_url:
        print("WARNING: EP499 audio URL missing; cannot add sample.")
        return

    start_time = float(EP499_SAMPLE_START)
    sample_id = f"pod_{episode_id[:8]}_{int(start_time)}"
    audio_path = audio_dir / f"{sample_id}.mp3"
    label = f"{title} (explicit @ {int(start_time)}s)"
    sample = Sample(
        sample_id=sample_id,
        source_type="podcast",
        source_id=episode_id,
        title=label,
        channel=channel,
        audio_path=relative_audio_path(audio_path),
        start_time=start_time,
        duration=SAMPLE_DURATION
    )

    existing = next((r for r in results if r.sample.sample_id == sample_id), None)
    if not ensure_sample_audio(sample, audio_path, conn):
        print(f"  Missing audio: {audio_path.name} (skipping EP499 sample)")
        return

    try:
        rel_path = audio_path.relative_to(EVAL_DIR)
        if sample.audio_path != str(rel_path):
            sample.audio_path = str(rel_path)
    except ValueError:
        pass

    transcriptions = run_transcriptions(
        audio_path,
        backend_specs,
        existing=existing.transcriptions if existing else None,
        audio_duration=sample.duration
    )

    if existing:
        existing.sample = sample
        existing.transcriptions = transcriptions
    else:
        results.append(EvalResult(sample=sample, transcriptions=transcriptions))


def get_random_youtube_samples(conn: sqlite3.Connection, n: int) -> list[dict]:
    """Get random YouTube video samples."""
    query = """
    SELECT v.video_id, v.title, v.channel, v.duration
    FROM videos v
    WHERE v.duration > 120
    ORDER BY RANDOM()
    LIMIT ?
    """
    rows = conn.execute(query, (n,)).fetchall()
    samples = []
    for row in rows:
        # Pick a random start time (avoiding first/last 30 seconds)
        duration = row[3]
        max_start = max(30, duration - SAMPLE_DURATION - 30)
        seed = int(hashlib.sha256(str(row[0]).encode("utf-8")).hexdigest(), 16)
        start = 30 + (seed % int(max_start - 30)) if max_start > 30 else 30
        samples.append({
            'video_id': row[0],
            'title': row[1],
            'channel': row[2],
            'duration': duration,
            'start_time': start,
        })
    return samples


def get_random_podcast_samples(conn: sqlite3.Connection, n: int) -> list[dict]:
    """Get random podcast episode samples."""
    query = """
    SELECT episode_id, title, channel_name, audio_url, duration
    FROM episodes
    WHERE audio_url IS NOT NULL AND duration > 300
    ORDER BY RANDOM()
    LIMIT ?
    """
    rows = conn.execute(query, (n,)).fetchall()
    samples = []
    for row in rows:
        duration = row[4]
        # Pick a random start time (avoiding first 2 min, last 1 min)
        max_start = max(120, duration - SAMPLE_DURATION - 60)
        seed = int(hashlib.sha256(str(row[0]).encode("utf-8")).hexdigest(), 16)
        start = 120 + (seed % int(max_start - 120)) if max_start > 120 else 120
        samples.append({
            'episode_id': row[0],
            'title': row[1],
            'channel': row[2],
            'audio_url': row[3],
            'duration': duration,
            'start_time': start,
        })
    return samples


def download_youtube_sample(video_id: str, start: float, duration: float, output_path: Path) -> bool:
    """Download a YouTube audio sample."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--download-sections", f"*{start}-{start + duration}",
        "-o", str(output_path),
        "--no-playlist",
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return output_path.exists()
    except Exception as e:
        print(f"  Error downloading YouTube: {e}")
        return False


def download_podcast_sample(audio_url: str, start: float, duration: float, output_path: Path) -> bool:
    """Download a podcast audio sample."""
    # First try yt-dlp with section download
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--download-sections", f"*{start}-{start + duration}",
        "-o", str(output_path),
        audio_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if output_path.exists():
            return True
    except Exception:
        pass

    # Fallback: download full and extract with ffmpeg
    try:
        import requests
        temp_full = output_path.with_suffix('.full.mp3')
        print(f"  Downloading full podcast...")
        response = requests.get(audio_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(temp_full, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract segment with ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_full),
            "-ss", str(start), "-t", str(duration),
            "-c:a", "libmp3lame", "-q:a", "2",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        temp_full.unlink(missing_ok=True)
        return output_path.exists()
    except Exception as e:
        print(f"  Error downloading podcast: {e}")
        return False


def convert_to_wav(mp3_path: Path) -> Path:
    """Convert MP3 to 16kHz mono WAV."""
    wav_path = mp3_path.with_suffix('.wav')
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return wav_path


# =============================================================================
# Alignment helpers (for eval_compare.html highlighting)
# =============================================================================

def _is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _parse_timestamp(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        parts = value.split(",")
        hms = parts[0].split(":")
        if len(hms) != 3:
            return None
        hours = int(hms[0])
        minutes = int(hms[1])
        seconds = int(hms[2])
        millis = int(parts[1]) if len(parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    except Exception:
        return None


def _parse_whispercpp_json(json_path: Path) -> tuple[str, list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    segments = data.get("transcription", []) or data.get("segments", [])
    text = "".join(seg.get("text", "") for seg in segments).strip()
    tokens: list[dict] = []
    for seg in segments:
        for tok in seg.get("tokens", []) or []:
            token_text = tok.get("text", "")
            if not token_text:
                continue
            stripped = token_text.strip()
            if stripped.startswith("[_") or stripped.startswith("<|") or (stripped.startswith("[") and stripped.endswith("]")):
                continue
            if "�" in stripped:
                continue
            offsets = tok.get("offsets") or {}
            start_ms = offsets.get("from")
            end_ms = offsets.get("to")
            if start_ms is None or end_ms is None:
                timestamps = tok.get("timestamps") or {}
                start = _parse_timestamp(timestamps.get("from"))
                end = _parse_timestamp(timestamps.get("to"))
            else:
                start = start_ms / 1000.0
                end = end_ms / 1000.0
            tokens.append({
                "text": token_text,
                "start": start,
                "end": end,
                "score": tok.get("p"),
            })
    return text, tokens


def _load_audio_mono_16k(audio_path: Path):
    import torch
    import torchaudio
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = audio.get_array_of_samples()
        try:
            import numpy as np
            arr = np.array(samples, dtype="float32") / 32768.0
            waveform = torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
        sr = 16000
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def _init_whisperx():
    global _WHISPERX_PATCHED
    import os as _os
    import numpy as _np
    import torch as _torch
    import typing as _typing
    import collections as _collections
    from omegaconf import listconfig, dictconfig, base, nodes as oc_nodes
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    _os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan
    if not _WHISPERX_PATCHED:
        safe_nodes = [v for v in vars(oc_nodes).values() if isinstance(v, type)]
        try:
            _torch.serialization.add_safe_globals([
                listconfig.ListConfig,
                dictconfig.DictConfig,
                base.ContainerMetadata,
                _typing.Any,
                list,
                dict,
                tuple,
                set,
                _collections.defaultdict,
                int,
                float,
                str,
                bool,
                _torch.torch_version.TorchVersion,
            ] + safe_nodes)
        except Exception:
            pass
        orig_load = _torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        _torch.load = _patched_load

        orig_from_pretrained = Wav2Vec2ForCTC.from_pretrained

        def _safe_from_pretrained(*args, **kwargs):
            kwargs.setdefault("use_safetensors", True)
            return orig_from_pretrained(*args, **kwargs)

        Wav2Vec2ForCTC.from_pretrained = _safe_from_pretrained

        if not hasattr(Wav2Vec2Processor, "sampling_rate"):
            def _sampling_rate(self):
                fe = getattr(self, "feature_extractor", None)
                return getattr(fe, "sampling_rate", None)
            Wav2Vec2Processor.sampling_rate = property(_sampling_rate)
        _WHISPERX_PATCHED = True


def _mms_char_tokens(audio_path: Path, transcript: str) -> list[dict]:
    tokens = [{"text": c, "start": None, "end": None, "score": None} for c in transcript]
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA
    except ImportError as e:
        raise RuntimeError(f"missing deps: {e}")

    waveform = _load_audio_mono_16k(audio_path)
    device = "cpu"
    bundle = MMS_FA
    model = bundle.get_model().to(device)
    with torch.no_grad():
        emission, _ = model(waveform.to(device))
    dictionary = bundle.get_dict()
    unk_id = dictionary.get("<unk>")
    align_indices = []
    token_ids = []
    for i, c in enumerate(transcript):
        if not _is_chinese(c):
            continue
        token_id = dictionary.get(c.lower())
        if token_id is None:
            if unk_id is None or unk_id == 0:
                continue
            token_id = unk_id
        if token_id == 0:
            continue
        align_indices.append(i)
        token_ids.append(token_id)
    if not token_ids:
        return tokens
    targets = torch.tensor([token_ids], dtype=torch.int32, device=device)
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    num_frames = emission.shape[1]
    audio_duration = waveform.shape[1] / 16000
    frame_duration = audio_duration / num_frames if num_frames else 0.0
    alignments = alignments[0].cpu().tolist()
    scores = scores[0].cpu().tolist()
    char_idx = 0
    for i, (token_id, score) in enumerate(zip(alignments, scores)):
        if token_id != 0 and char_idx < len(align_indices):
            idx = align_indices[char_idx]
            tokens[idx]["start"] = i * frame_duration
            tokens[idx]["end"] = (i + 1) * frame_duration
            tokens[idx]["score"] = score
            char_idx += 1
    return tokens


def _mms_pinyin_tokens(audio_path: Path, transcript: str) -> list[dict]:
    tokens = [{"text": c, "start": None, "end": None, "score": None} for c in transcript]
    align_indices = [i for i, c in enumerate(transcript) if _is_chinese(c)]
    align_chars = [transcript[i] for i in align_indices]
    if not align_chars:
        return tokens
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA
        from pypinyin import pinyin, Style
    except ImportError as e:
        raise RuntimeError(f"missing deps: {e}")

    pinyin_tokens = []
    for c in align_chars:
        py = pinyin(c, style=Style.NORMAL, strict=False, errors="ignore")
        if not py:
            pinyin_tokens.append("")
        else:
            pinyin_tokens.append(py[0][0])

    waveform = _load_audio_mono_16k(audio_path)
    device = "cpu"
    bundle = MMS_FA
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    with torch.no_grad():
        emission, _ = model(waveform.to(device))
    tokenized = tokenizer(pinyin_tokens)
    spans = aligner(emission[0], tokenized)
    ratio = waveform.shape[1] / emission.shape[1] / 16000
    for j, span_item in enumerate(spans):
        if j >= len(align_indices):
            break
        if isinstance(span_item, list):
            if not span_item:
                continue
            start_span = span_item[0]
            end_span = span_item[-1]
        else:
            start_span = span_item
            end_span = span_item
        idx = align_indices[j]
        tokens[idx]["start"] = float(start_span.start * ratio)
        tokens[idx]["end"] = float(end_span.end * ratio)
        tokens[idx]["score"] = float(getattr(end_span, "score", 0.0))
    return tokens


def _whisperx_tokens(audio_path: Path, transcript: str) -> list[dict]:
    try:
        _init_whisperx()
        import whisperx
    except Exception as e:
        raise RuntimeError(f"missing deps: {e}")
    global _WHISPERX_ALIGN_MODEL, _WHISPERX_ALIGN_META
    device = "cpu"
    if _WHISPERX_ALIGN_MODEL is None or _WHISPERX_ALIGN_META is None:
        _WHISPERX_ALIGN_MODEL, _WHISPERX_ALIGN_META = whisperx.load_align_model(
            language_code="zh",
            device=device,
        )
    audio = whisperx.load_audio(str(audio_path))
    duration = audio.shape[0] / whisperx.audio.SAMPLE_RATE
    segments = [{"start": 0.0, "end": float(duration), "text": transcript}]
    result = whisperx.align(
        segments,
        _WHISPERX_ALIGN_MODEL,
        _WHISPERX_ALIGN_META,
        audio,
        device,
        return_char_alignments=True,
    )
    tokens: list[dict] = []
    for seg in result.get("segments", []):
        chars = seg.get("chars") or []
        if chars:
            for ch in chars:
                tokens.append({
                    "text": ch.get("char") or ch.get("text") or "",
                    "start": ch.get("start"),
                    "end": ch.get("end"),
                    "score": ch.get("score"),
                })
        else:
            for word in seg.get("words", []) or []:
                tokens.append({
                    "text": word.get("word", ""),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "score": word.get("score") or word.get("probability"),
                })
    return tokens


def _stable_ts_tokens(audio_path: Path, transcript: str) -> list[dict]:
    try:
        import stable_whisper
    except Exception as e:
        raise RuntimeError(f"missing deps: {e}")
    global _STABLE_TS_MODEL
    if _STABLE_TS_MODEL is None:
        _STABLE_TS_MODEL = stable_whisper.load_model("base")
    try:
        result = _STABLE_TS_MODEL.align(str(audio_path), transcript, "Chinese")
    except Exception:
        result = _STABLE_TS_MODEL.align(str(audio_path), transcript, "zh")
    data = result.to_dict() if hasattr(result, "to_dict") else result
    tokens: list[dict] = []
    for seg in data.get("segments", []) if isinstance(data, dict) else []:
        for word in seg.get("words", []) or []:
            tokens.append({
                "text": word.get("word", ""),
                "start": word.get("start"),
                "end": word.get("end"),
                "score": word.get("probability") or word.get("score"),
            })
    return tokens


def _round_tokens(tokens: list[dict], ndigits: int = 3) -> list[dict]:
    rounded = []
    for tok in tokens:
        item = dict(tok)
        if isinstance(item.get("start"), (int, float)):
            item["start"] = round(float(item["start"]), ndigits)
        if isinstance(item.get("end"), (int, float)):
            item["end"] = round(float(item["end"]), ndigits)
        if isinstance(item.get("score"), (int, float)):
            item["score"] = round(float(item["score"]), ndigits)
        rounded.append(item)
    return rounded


def compute_alignments(audio_path: Path, transcript: str) -> dict:
    results: dict = {}
    try:
        tokens = _whisperx_tokens(audio_path, transcript)
        tokens = _round_tokens(tokens)
        if tokens:
            results[WHISPERX_METHOD] = {"status": "ok", "error": None, "tokens": tokens}
        else:
            results[WHISPERX_METHOD] = {"status": "error", "error": "no tokens", "tokens": []}
    except Exception as e:
        results[WHISPERX_METHOD] = {"status": "error", "error": str(e), "tokens": []}
    return results


def choose_default_alignment(alignments: dict) -> Optional[str]:
    for name in DEFAULT_ALIGNMENT_ORDER:
        entry = alignments.get(name)
        if entry and entry.get("status") == "ok" and entry.get("tokens"):
            return name
    for name, entry in alignments.items():
        if entry.get("status") == "ok" and entry.get("tokens"):
            return name
    return None


def transcribe_whisper_cpp(audio_path: Path) -> TranscriptionResult:
    """Transcribe with whisper.cpp."""
    if not WHISPER_MODEL_PATH.exists():
        return TranscriptionResult("whisper-large-v3", "", 0, 0, error="Model not found")

    wav_path = convert_to_wav(audio_path)
    output_base = audio_path.parent / f"wcpp_{audio_path.stem}"

    start = time.time()
    try:
        cmd = [
            "whisper-cli",
            "-m", str(WHISPER_MODEL_PATH),
            "-l", "zh",
            "-t", "8", "-p", "4",
            "-ojf",  # Full JSON output with token timestamps
            "-of", str(output_base),
            str(wav_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        elapsed = time.time() - start

        json_file = Path(str(output_base) + ".json")
        if json_file.exists():
            text, _ = _parse_whispercpp_json(json_file)
            alignments = None
            json_file.unlink()
        else:
            text = ""
            alignments = None

        wav_path.unlink(missing_ok=True)
        return TranscriptionResult("whisper-large-v3", text, elapsed, len(text), alignments=alignments)

    except Exception as e:
        wav_path.unlink(missing_ok=True)
        return TranscriptionResult("whisper-large-v3", "", time.time() - start, 0, error=str(e))


def transcribe_mlx_whisper(audio_path: Path) -> TranscriptionResult:
    """Transcribe with mlx-whisper."""
    try:
        import mlx_whisper
    except ImportError:
        return TranscriptionResult("mlx-whisper", "", 0, 0, error="mlx-whisper not installed")

    start = time.time()
    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="zh",
            word_timestamps=True,
        )
        elapsed = time.time() - start
        text = result.get("text", "").strip()
        return TranscriptionResult("mlx-whisper", text, elapsed, len(text), alignments=None)
    except Exception as e:
        return TranscriptionResult("mlx-whisper", "", time.time() - start, 0, error=str(e))


def transcribe_funasr(audio_path: Path) -> TranscriptionResult:
    """Transcribe with Fun-ASR-Nano."""
    try:
        from funasr import AutoModel
        from funasr.models.fun_asr_nano import model as funasr_nano_model  # noqa: F401
    except ImportError:
        return TranscriptionResult("funasr-nano", "", 0, 0, error="funasr not installed")

    start = time.time()
    try:
        ensure_qwen3_weights()
        patch_funasr_load_in_8bit()
        global _FUNASR_MODEL
        if _FUNASR_MODEL is None:
            device = select_asr_device("FUNASR_DEVICE")
            _FUNASR_MODEL = AutoModel(
                model=FUNASR_MODEL_ID,
                device=device,
                disable_update=True
            )
        res = _FUNASR_MODEL.generate(
            input=[str(audio_path)],
            cache={},
            batch_size=1,
            language="中文",
            itn=True
        )
        elapsed = time.time() - start
        text = res[0]["text"] if res else ""
        return TranscriptionResult("funasr-nano", text, elapsed, len(text))
    except Exception as e:
        return TranscriptionResult("funasr-nano", "", time.time() - start, 0, error=str(e))


def _load_glm_asr_transformers():
    global _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR
    if _GLM_ASR_MODEL is None or _GLM_ASR_PROCESSOR is None:
        _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR = load_glm_asr_transformers(GLM_ASR_MODEL_ID)
    return _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR


def transcribe_glm_asr(audio_path: Path) -> TranscriptionResult:
    """Transcribe with GLM-ASR (transformers)."""
    start = time.time()
    try:
        model, processor = _load_glm_asr_transformers()
        inputs = processor.apply_transcription_request(str(audio_path))
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            inputs = inputs.to(model.device, dtype=dtype)
        else:
            inputs = inputs.to(model.device)
        try:
            import torch
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(os.environ.get("GLM_ASR_MAX_NEW_TOKENS", "512"))
                )
        except Exception:
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(os.environ.get("GLM_ASR_MAX_NEW_TOKENS", "512"))
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        text = decoded[0].strip() if decoded else ""
        return TranscriptionResult("glm-asr", text, time.time() - start, len(text))
    except Exception as e:
        return TranscriptionResult("glm-asr", "", time.time() - start, 0, error=str(e))


BACKEND_SPECS = [
    BackendSpec("whisper-large-v3", "Whisper-Large-V3", "whisper", transcribe_whisper_cpp),
    BackendSpec("funasr-nano", "Fun-ASR-Nano", "funasr", transcribe_funasr),
    BackendSpec("glm-asr", "GLM-ASR", "glm", transcribe_glm_asr),
]
BACKEND_BY_KEY = {spec.key: spec for spec in BACKEND_SPECS}
LEGACY_BACKEND_MAP = {
    "whisper.cpp": "whisper-large-v3",
    "mlx-whisper": "mlx-whisper",
    "funasr-nano": "funasr-nano",
}


def normalize_backend_key(key: str) -> str:
    return LEGACY_BACKEND_MAP.get(key, key)


def load_existing_results() -> list[EvalResult]:
    if not RESULTS_FILE.exists():
        return []
    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    results: list[EvalResult] = []
    for item in data:
        sample = Sample(**item["sample"])
        transcriptions = {}
        for key, tr in item.get("transcriptions", {}).items():
            normalized = normalize_backend_key(key)
            tr_data = dict(tr)
            tr_data["backend"] = normalized
            if tr_data.get("error") is None and not isinstance(tr_data.get("audio_duration", SAMPLE_DURATION), (int, float)):
                tr_data["error"] = tr_data.get("audio_duration")
                tr_data["audio_duration"] = SAMPLE_DURATION
            transcriptions[normalized] = TranscriptionResult(**tr_data)
        results.append(EvalResult(sample=sample, transcriptions=transcriptions))
    return results


def generate_html_report(results: list[EvalResult], backend_specs: list[BackendSpec]) -> str:
    """Generate HTML comparison interface."""

    backend_keys = [spec.key for spec in backend_specs]
    backend_labels = {spec.key: spec.label for spec in backend_specs}
    backend_css = {spec.key: spec.css_class for spec in backend_specs}

    # Calculate aggregate benchmark stats
    benchmark_stats = {}
    for backend in backend_keys:
        valid = [
            r.transcriptions.get(backend)
            for r in results
            if r.transcriptions.get(backend) and not r.transcriptions[backend].error
        ]
        avg_time = sum(t.elapsed_seconds for t in valid) / len(valid) if valid else None
        avg_speed = sum(t.realtime_speed for t in valid) / len(valid) if valid else None
        avg_chars = sum(t.word_count for t in valid) / len(valid) if valid else None
        errors = sum(
            1 for r in results
            if (backend not in r.transcriptions) or r.transcriptions[backend].error
        )
        benchmark_stats[backend] = {
            'avg_time': avg_time,
            'avg_speed': avg_speed,
            'avg_chars': avg_chars,
            'errors': errors,
            'count': len(valid)
        }

    html = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASR Accuracy Evaluation</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1, h2 { text-align: center; color: #333; }
        .section-title { margin-top: 40px; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-card {
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-card h3 { margin: 0; color: #666; font-size: 14px; }
        .stat-card .value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .benchmark-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .benchmark-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .benchmark-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .benchmark-card.whisper { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .benchmark-card.mlx { background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); }
        .benchmark-card.funasr { background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%); }
        .benchmark-card.glm { background: linear-gradient(135deg, #F7971E 0%, #FFD200 100%); }
        .benchmark-card h4 { margin: 0 0 15px 0; font-size: 18px; }
        .speed-value { font-size: 48px; font-weight: bold; }
        .speed-label { font-size: 14px; opacity: 0.9; }
        .benchmark-details { margin-top: 15px; font-size: 13px; opacity: 0.85; }
        .speed-bar-container {
            margin-top: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            overflow: hidden;
            height: 24px;
            position: relative;
        }
        .speed-bar {
            height: 100%;
            background: rgba(255,255,255,0.4);
            transition: width 0.3s ease;
        }
        .speed-bar-label {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
            font-weight: bold;
        }
        .filters {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 8px 16px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .filter-btn:hover { background: #e3f2fd; }
        .filter-btn.active { background: #2196F3; color: white; border-color: #2196F3; }
        .sample {
            background: white;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .sample-header {
            background: #2196F3;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .sample-header h3 { margin: 0; font-size: 16px; }
        .sample-meta { font-size: 12px; opacity: 0.9; }
        .sample-type {
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .audio-player {
            padding: 15px 20px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }
        .audio-player audio { width: 100%; }
        .transcriptions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1px;
            background: #eee;
        }
        .transcription {
            background: white;
            padding: 15px;
        }
        .transcription-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .backend-name {
            font-weight: bold;
            color: #333;
        }
        .backend-time {
            font-size: 12px;
            color: #666;
        }
        .transcription-text {
            font-size: 15px;
            line-height: 1.8;
            color: #333;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .token {
            padding: 0 1px;
            border-radius: 4px;
        }
        .token.active {
            background: #ffe08a;
        }
        .token.no-time {
            color: #777;
        }
        .error { color: #f44336; font-style: italic; }
        .rating-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        .rating-btn {
            padding: 5px 10px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .rating-btn:hover { background: #f0f0f0; }
        .rating-btn.selected { background: #4CAF50; color: white; border-color: #4CAF50; }
        .rating-btn.selected.poor { background: #f44336; border-color: #f44336; }
        .rating-btn.selected.ok { background: #ff9800; border-color: #ff9800; }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-table th, .summary-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .summary-table th { background: #f5f5f5; font-weight: 600; }
        .export-btn {
            display: block;
            margin: 20px auto;
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .export-btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>🎙️ ASR Accuracy Evaluation</h1>

    <div class="stats">
        <div class="stat-card">
            <h3>Total Samples</h3>
            <div class="value">''' + str(len(results)) + '''</div>
        </div>
        <div class="stat-card">
            <h3>YouTube</h3>
            <div class="value">''' + str(sum(1 for r in results if r.sample.source_type == 'youtube')) + '''</div>
        </div>
        <div class="stat-card">
            <h3>Podcast</h3>
            <div class="value">''' + str(sum(1 for r in results if r.sample.source_type == 'podcast')) + '''</div>
        </div>
    </div>

    <h2 class="section-title">Benchmark: Speed Comparison</h2>
    <div class="benchmark-section">
        <p style="text-align:center;color:#666;">Average transcription speed for ''' + str(SAMPLE_DURATION) + '''-second audio samples</p>
        <div class="benchmark-grid">
'''

    # Add benchmark cards
    max_speed = max((s['avg_speed'] or 0) for s in benchmark_stats.values()) if benchmark_stats else 1
    if max_speed <= 0:
        max_speed = 1

    for backend, stats in benchmark_stats.items():
        label = backend_labels.get(backend, backend)
        speed_display = f"{stats['avg_speed']:.1f}x" if stats['avg_speed'] is not None else "N/A"
        avg_time_display = f"{stats['avg_time']:.1f}s" if stats['avg_time'] is not None else "N/A"
        bar_width = (stats['avg_speed'] / max_speed * 100) if stats['avg_speed'] else 0
        html += f'''
            <div class="benchmark-card {backend_css.get(backend, '')}">
                <h4>{label}</h4>
                <div class="speed-value">{speed_display}</div>
                <div class="speed-label">realtime speed</div>
                <div class="benchmark-details">
                    Avg: {avg_time_display} per {SAMPLE_DURATION}s audio<br>
                    {stats['count']} samples, {stats['errors']} errors
                </div>
                <div class="speed-bar-container">
                    <div class="speed-bar" style="width: {bar_width:.0f}%"></div>
                </div>
            </div>
'''

    html += '''
        </div>
    </div>

    <h2 class="section-title">Transcription Comparison</h2>
    <div class="filters">
        <button class="filter-btn active" onclick="filterSamples('all')">All</button>
        <button class="filter-btn" onclick="filterSamples('youtube')">YouTube</button>
        <button class="filter-btn" onclick="filterSamples('podcast')">Podcast</button>
    </div>

    <div id="samples">
'''

    for i, result in enumerate(results):
        sample = result.sample
        safe_title = html_lib.escape(sample.title)
        safe_channel = html_lib.escape(sample.channel)
        html += f'''
        <div class="sample" data-type="{sample.source_type}" id="sample-{i}">
            <div class="sample-header">
                <div>
                    <h3>{safe_title[:60]}{'...' if len(safe_title) > 60 else ''}</h3>
                    <div class="sample-meta">{safe_channel} • {sample.start_time:.0f}s - {sample.start_time + sample.duration:.0f}s</div>
                </div>
                <span class="sample-type">{sample.source_type.upper()}</span>
            </div>
            <div class="audio-player">
                <audio controls preload="none">
                    <source src="{sample.audio_path}" type="audio/mpeg">
                </audio>
            </div>
            <div class="transcriptions">
'''
        for backend in backend_keys:
            tr = result.transcriptions.get(backend)
            if tr is None:
                tr = TranscriptionResult(backend, "", 0, 0, error="Not run")
            alignment_data_html = ""
            text_html = ""
            if tr.error:
                text_html = f'<span class="error">Error: {html_lib.escape(tr.error)}</span>'
            else:
                alignments = tr.alignments or {}
                default_alignment = None
                if alignments:
                    default_alignment = choose_default_alignment(alignments)
                if alignments and default_alignment:
                    payload = {"default": default_alignment, "methods": alignments}
                    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
                    alignment_data_html = f'<script type="application/json" class="alignment-data">{payload_json}</script>'
                    text_html = '<span class="error">Loading alignment…</span>'
                else:
                    safe_text = html_lib.escape(tr.text)
                    text_html = safe_text if safe_text else '<span class="error">No transcription</span>'

            speed_str = f"{tr.realtime_speed:.1f}x RT" if tr.realtime_speed > 0 else "N/A"
            label = backend_labels.get(backend, backend)
            html += f'''
                <div class="transcription" data-backend="{backend}">
                    <div class="transcription-header">
                        <span class="backend-name">{label}</span>
                        <span class="backend-time">{tr.elapsed_seconds:.1f}s ({speed_str}) • {tr.word_count} chars</span>
                    </div>
                    <div class="transcription-text">{text_html}</div>
                    {alignment_data_html}
                    <div class="rating-buttons">
                        <button class="rating-btn" onclick="rate({i}, '{backend}', 'good')">Good</button>
                        <button class="rating-btn" onclick="rate({i}, '{backend}', 'ok')">OK</button>
                        <button class="rating-btn" onclick="rate({i}, '{backend}', 'poor')">Poor</button>
                    </div>
                </div>
'''
        html += '''
            </div>
        </div>
'''

    html += '''
    </div>

    <h2 class="section-title">Your Ratings Summary</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Backend</th>
                <th>Avg Speed</th>
                <th>Avg Chars</th>
                <th>Errors</th>
                <th>Good</th>
                <th>OK</th>
                <th>Poor</th>
            </tr>
        </thead>
        <tbody id="summary-body">
        </tbody>
    </table>'''

    html += '''

    <button class="export-btn" onclick="exportRatings()">Export Ratings (JSON)</button>

    <script>
        const ratings = {};
        const backends = ''' + json.dumps(backend_keys) + ''';
        const backendLabels = ''' + json.dumps(backend_labels) + ''';
        const benchmarkStats = ''' + json.dumps(benchmark_stats) + ''';

        function filterSamples(type) {
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            document.querySelectorAll('.sample').forEach(sample => {
                if (type === 'all' || sample.dataset.type === type) {
                    sample.style.display = 'block';
                } else {
                    sample.style.display = 'none';
                }
            });
        }

        function rate(sampleId, backend, rating) {
            const key = `${sampleId}-${backend}`;
            ratings[key] = rating;

            // Update button styles
            const btns = document.querySelectorAll(`#sample-${sampleId} [data-backend="${backend}"] .rating-btn`);
            btns.forEach(btn => {
                btn.classList.remove('selected', 'poor', 'ok');
                if (btn.textContent.includes(rating === 'good' ? 'Good' : rating === 'ok' ? 'OK' : 'Poor')) {
                    btn.classList.add('selected');
                    if (rating === 'poor') btn.classList.add('poor');
                    if (rating === 'ok') btn.classList.add('ok');
                }
            });

            updateSummary();
        }

        function updateSummary() {
            const counts = {};
            backends.forEach(b => {
                counts[b] = { good: 0, ok: 0, poor: 0 };
            });

            Object.entries(ratings).forEach(([key, rating]) => {
                const backend = key.split('-').slice(1).join('-');
                if (counts[backend]) {
                    counts[backend][rating]++;
                }
            });

            const tbody = document.getElementById('summary-body');
            tbody.innerHTML = backends.map(b => {
                const stats = benchmarkStats[b] || {};
                const avgSpeed = stats.avg_speed ? stats.avg_speed.toFixed(1) + 'x' : '-';
                const avgChars = stats.avg_chars ? Math.round(stats.avg_chars) : '-';
                const errors = stats.errors !== undefined ? stats.errors : '-';
                return `
                <tr>
                    <td>${backendLabels[b] || b}</td>
                    <td>${avgSpeed}</td>
                    <td>${avgChars}</td>
                    <td>${errors}</td>
                    <td>${counts[b].good}</td>
                    <td>${counts[b].ok}</td>
                    <td>${counts[b].poor}</td>
                </tr>
            `}).join('');
        }

        function escapeHtml(text) {
            return String(text || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function buildTokensHtml(tokens) {
            return (tokens || []).map(tok => {
                const text = escapeHtml(tok.text);
                const start = tok.start;
                const end = tok.end;
                if (start !== null && start !== undefined && end !== null && end !== undefined) {
                    return `<span class="token" data-start="${start}" data-end="${end}">${text}</span>`;
                }
                return `<span class="token no-time">${text}</span>`;
            }).join('');
        }

        function renderAlignment(trEl, method) {
            const payload = trEl._alignmentData;
            const textEl = trEl.querySelector('.transcription-text');
            if (!payload || !textEl) return;
            const entry = payload.methods ? payload.methods[method] : null;
            if (!entry || entry.status !== 'ok' || !entry.tokens || entry.tokens.length === 0) {
                textEl.innerHTML = '<span class="error">Alignment not available</span>';
                textEl.dataset.alignEnabled = 'false';
                textEl.dataset.activeIndex = '';
                return;
            }
            textEl.innerHTML = buildTokensHtml(entry.tokens);
            textEl.dataset.alignEnabled = 'true';
            textEl.dataset.activeIndex = '';
        }

        function highlightSample(sampleEl, currentTime) {
            sampleEl.querySelectorAll('.transcription-text[data-align-enabled="true"]').forEach(textEl => {
                const tokens = textEl.querySelectorAll('.token[data-start]');
                if (!tokens.length) return;
                let activeIndex = -1;
                for (let i = 0; i < tokens.length; i++) {
                    const start = parseFloat(tokens[i].dataset.start);
                    const end = parseFloat(tokens[i].dataset.end);
                    if (currentTime >= start && currentTime < end) {
                        activeIndex = i;
                        break;
                    }
                }
                const prev = parseInt(textEl.dataset.activeIndex || '-1', 10);
                if (prev === activeIndex) return;
                if (prev >= 0 && tokens[prev]) {
                    tokens[prev].classList.remove('active');
                }
                if (activeIndex >= 0 && tokens[activeIndex]) {
                    tokens[activeIndex].classList.add('active');
                }
                textEl.dataset.activeIndex = String(activeIndex);
            });
        }

        function initAlignments() {
            document.querySelectorAll('.transcription').forEach(trEl => {
                const dataEl = trEl.querySelector('script.alignment-data');
                if (!dataEl) return;
                let payload = null;
                try {
                    payload = JSON.parse(dataEl.textContent);
                } catch (e) {
                    return;
                }
                trEl._alignmentData = payload;
                const defaultMethod = payload.default;
                if (defaultMethod) {
                    renderAlignment(trEl, defaultMethod);
                }
            });

            document.querySelectorAll('.sample').forEach(sampleEl => {
                const audio = sampleEl.querySelector('audio');
                if (!audio) return;
                const handler = () => highlightSample(sampleEl, audio.currentTime || 0);
                audio.addEventListener('timeupdate', handler);
                audio.addEventListener('seeked', handler);
                audio.addEventListener('play', handler);
                audio.addEventListener('loadedmetadata', handler);
            });
        }

        function exportRatings() {
            const data = JSON.stringify({ratings, benchmarkStats}, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'asr_ratings.json';
            a.click();
        }

        initAlignments();
        // Initial summary
        updateSummary();
    </script>
</body>
</html>
'''
    return html


def run_transcriptions(
    audio_path: Path,
    backend_specs: list[BackendSpec],
    existing: Optional[dict[str, TranscriptionResult]] = None,
    audio_duration: float = SAMPLE_DURATION,
) -> dict[str, TranscriptionResult]:
    transcriptions = dict(existing) if existing else {}
    for spec in backend_specs:
        existing_tr = transcriptions.get(spec.key)
        if existing_tr and not existing_tr.error:
            if existing_tr.text:
                needs_alignment = (
                    not existing_tr.alignments
                    or set(existing_tr.alignments.keys()) != {WHISPERX_METHOD}
                )
                if needs_alignment:
                    existing_tr.alignments = compute_alignments(audio_path, existing_tr.text)
            transcriptions[spec.key] = existing_tr
            continue
        print(f"  Transcribing with {spec.label}...")
        tr = spec.transcribe(audio_path)
        tr.audio_duration = audio_duration
        if tr.error:
            print(f"    -> ERROR: {tr.error}")
        else:
            print(f"    -> {tr.elapsed_seconds:.1f}s, {tr.word_count} chars")
        if not tr.error and tr.text:
            tr.alignments = compute_alignments(audio_path, tr.text)
        transcriptions[spec.key] = tr
    return transcriptions


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR accuracy across backends.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing results and resample")
    parser.add_argument("--num-youtube", type=int, default=NUM_YOUTUBE_SAMPLES,
                        help="Number of YouTube samples to evaluate")
    parser.add_argument("--num-podcast", type=int, default=NUM_PODCAST_SAMPLES,
                        help="Number of podcast samples to evaluate")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_SUBDIR,
                        help="Audio cache directory (relative to eval_samples unless absolute)")
    parser.add_argument("--backends", default=",".join(spec.key for spec in BACKEND_SPECS),
                        help="Comma-separated backend keys to run")
    parser.add_argument("--include-ep499", action="store_true",
                        help="Include explicit EP499 TAXI DRIVER sample for comparison")
    args = parser.parse_args()

    backend_keys = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backend_keys if b not in BACKEND_BY_KEY]
    if unknown:
        print(f"ERROR: Unknown backend(s): {', '.join(unknown)}")
        sys.exit(1)
    backend_specs = [BACKEND_BY_KEY[b] for b in backend_keys]

    print("=" * 60)
    print("ASR Accuracy Evaluation")
    print("=" * 60)

    # Create eval directory
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = resolve_audio_dir(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    existing_results = [] if args.fresh else load_existing_results()

    if existing_results:
        print(f"\nReusing {len(existing_results)} existing samples from {RESULTS_FILE}...")
        results = existing_results
        db_path = Path(__file__).parent / "vocab.db"
        conn = sqlite3.connect(str(db_path)) if db_path.exists() else None
        if conn is None:
            print("WARNING: Database not found; podcast audio cannot be re-downloaded.")
        if conn is not None and args.include_ep499:
            add_explicit_ep499_sample(results, conn, audio_dir, backend_specs)
        for i, result in enumerate(results):
            sample = result.sample
            audio_path = resolve_audio_path(sample.audio_path, audio_dir)
            print(f"\n[{i+1}/{len(results)}] {sample.title[:50]}...")
            if not ensure_sample_audio(sample, audio_path, conn):
                print(f"  Missing audio: {audio_path.name} (skipping)")
                continue
            try:
                rel_path = audio_path.relative_to(EVAL_DIR)
                if sample.audio_path != str(rel_path):
                    sample.audio_path = str(rel_path)
            except ValueError:
                pass
            result.transcriptions = run_transcriptions(
                audio_path,
                backend_specs,
                existing=result.transcriptions,
                audio_duration=sample.duration
            )
        if conn is not None:
            conn.close()
    else:
        # Connect to database
        db_path = Path(__file__).parent / "vocab.db"
        if not db_path.exists():
            print(f"ERROR: Database not found: {db_path}")
            sys.exit(1)

        conn = sqlite3.connect(str(db_path))

        # Get samples
        print(f"\nSelecting {args.num_youtube} YouTube + {args.num_podcast} podcast samples...")
        youtube_samples = get_random_youtube_samples(conn, args.num_youtube)
        podcast_samples = get_random_podcast_samples(conn, args.num_podcast)

        print(f"  YouTube samples: {len(youtube_samples)}")
        print(f"  Podcast samples: {len(podcast_samples)}")

        # Process YouTube samples
        print(f"\n{'='*60}")
        print("Processing YouTube samples...")
        print("=" * 60)

        for i, yt in enumerate(youtube_samples):
            print(f"\n[{i+1}/{len(youtube_samples)}] {yt['title'][:50]}...")
            sample_id = f"yt_{yt['video_id']}_{int(yt['start_time'])}"
            audio_path = audio_dir / f"{sample_id}.mp3"

            if not audio_path.exists():
                print(f"  Downloading {SAMPLE_DURATION}s from {yt['start_time']:.0f}s...")
                if not download_youtube_sample(yt['video_id'], yt['start_time'], SAMPLE_DURATION, audio_path):
                    print("  FAILED to download, skipping")
                    continue

            sample = Sample(
                sample_id=sample_id,
                source_type='youtube',
                source_id=yt['video_id'],
                title=yt['title'],
                channel=yt['channel'],
                audio_path=relative_audio_path(audio_path),
                start_time=yt['start_time'],
                duration=SAMPLE_DURATION
            )

            transcriptions = run_transcriptions(
                audio_path,
                backend_specs,
                audio_duration=sample.duration
            )

            results.append(EvalResult(
                sample=sample,
                transcriptions=transcriptions
            ))

        # Process podcast samples
        print(f"\n{'='*60}")
        print("Processing Podcast samples...")
        print("=" * 60)

        for i, pod in enumerate(podcast_samples):
            print(f"\n[{i+1}/{len(podcast_samples)}] {pod['title'][:50]}...")
            sample_id = f"pod_{pod['episode_id'][:8]}_{int(pod['start_time'])}"
            audio_path = audio_dir / f"{sample_id}.mp3"

            if not audio_path.exists():
                print(f"  Downloading {SAMPLE_DURATION}s from {pod['start_time']:.0f}s...")
                if not download_podcast_sample(pod['audio_url'], pod['start_time'], SAMPLE_DURATION, audio_path):
                    print("  FAILED to download, skipping")
                    continue

            sample = Sample(
                sample_id=sample_id,
                source_type='podcast',
                source_id=pod['episode_id'],
                title=pod['title'],
                channel=pod['channel'],
                audio_path=relative_audio_path(audio_path),
                start_time=pod['start_time'],
                duration=SAMPLE_DURATION
            )

            transcriptions = run_transcriptions(
                audio_path,
                backend_specs,
                audio_duration=sample.duration
            )

            results.append(EvalResult(
                sample=sample,
                transcriptions=transcriptions
            ))

        if args.include_ep499:
            add_explicit_ep499_sample(results, conn, audio_dir, backend_specs)

        conn.close()

    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results...")
    print("=" * 60)

    results_data = []
    for r in results:
        results_data.append({
            'sample': asdict(r.sample),
            'transcriptions': {k: asdict(v) for k, v in r.transcriptions.items()}
        })

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {RESULTS_FILE}")

    # Generate HTML
    html = generate_html_report(results, backend_specs)
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML saved to: {HTML_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total samples: {len(results)}")
    print(f"  YouTube: {sum(1 for r in results if r.sample.source_type == 'youtube')}")
    print(f"  Podcast: {sum(1 for r in results if r.sample.source_type == 'podcast')}")

    for spec in backend_specs:
        times = [
            r.transcriptions[spec.key].elapsed_seconds
            for r in results
            if spec.key in r.transcriptions and not r.transcriptions[spec.key].error
        ]
        chars = [
            r.transcriptions[spec.key].word_count
            for r in results
            if spec.key in r.transcriptions and not r.transcriptions[spec.key].error
        ]
        errors = sum(
            1 for r in results
            if spec.key not in r.transcriptions or r.transcriptions[spec.key].error
        )
        if times:
            print(f"\n  {spec.label}:")
            print(f"    Avg time: {sum(times)/len(times):.1f}s")
            print(f"    Avg chars: {sum(chars)/len(chars):.0f}")
            print(f"    Errors: {errors}")

    print(f"\n{'='*60}")
    print(f"Open {HTML_FILE} in your browser to compare transcriptions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
