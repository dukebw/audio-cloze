#!/usr/bin/env python3
"""Compare alignment methods on a fixed sample.

Default target: EP499 TAXI DRIVER sample at 610s (pod_631ec155_610).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Disable PyTorch weights-only loading for WhisperX checkpoints.
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

from audio_vocab_miner import (
    align_transcript_to_audio,
    localize_word_with_asr,
    to_traditional,
    convert_audio_to_wav,
)

EVAL_RESULTS = Path(__file__).resolve().parents[1] / "eval_samples" / "eval_results.json"
EVAL_AUDIO_DIR = Path(__file__).resolve().parents[1] / "eval_samples" / "audio"
DEFAULT_SAMPLE_ID = "pod_631ec155_610"


@dataclass
class AlignmentResult:
    method: str
    status: str
    error: Optional[str]
    occurrences: list[dict]


def load_eval_sample(sample_id: str) -> dict:
    if not EVAL_RESULTS.exists():
        raise FileNotFoundError(f"Missing eval results: {EVAL_RESULTS}")
    data = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    for item in data:
        if item.get("sample", {}).get("sample_id") == sample_id:
            return item
    raise ValueError(f"Sample {sample_id} not found in eval_results.json")


def choose_word(fun_text: str, whisper_text: str) -> str:
    candidates = ["答案", "覺得", "喜歡", "好笑", "司機", "你覺得", "我覺得"]
    for cand in candidates:
        if cand in fun_text and cand in whisper_text:
            return cand
    return "覺得" if "覺得" in fun_text else "答案"


def mms_char_align(audio_path: Path, transcript: str, word: str) -> AlignmentResult:
    try:
        occ = align_transcript_to_audio(audio_path, transcript, word)
        return AlignmentResult("mms_char", "ok", None, occ)
    except Exception as e:
        return AlignmentResult("mms_char", "error", str(e), [])


def mms_pinyin_align(audio_path: Path, transcript: str, word: str) -> AlignmentResult:
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA
        from pypinyin import pinyin, Style
    except Exception as e:
        return AlignmentResult("mms_pinyin", "error", f"missing deps: {e}", [])

    # Chinese-only stream
    chars = [c for c in transcript if "\u4e00" <= c <= "\u9fff"]
    if not chars:
        return AlignmentResult("mms_pinyin", "error", "no Chinese chars", [])

    pinyin_tokens = []
    for c in chars:
        py = pinyin(c, style=Style.NORMAL, strict=False, errors="ignore")
        if not py:
            pinyin_tokens.append("")
            continue
        pinyin_tokens.append(py[0][0])

    # Load audio (16k mono)
    temp_wav = None
    load_path = audio_path
    if audio_path.suffix.lower() != ".wav":
        temp_wav = convert_audio_to_wav(audio_path)
        load_path = temp_wav

    try:
        waveform, sr = torchaudio.load(str(load_path))
    except Exception:
        # fallback to ffmpeg -> wav
        if temp_wav is None:
            temp_wav = convert_audio_to_wav(audio_path)
            load_path = temp_wav
            waveform, sr = torchaudio.load(str(load_path))
        else:
            raise

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    device = "cpu"
    bundle = MMS_FA
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.no_grad():
        emission, _ = model(waveform.to(device))

    # Align pinyin tokens as "words"
    tokenized = tokenizer(pinyin_tokens)
    try:
        spans = aligner(emission[0], tokenized)
    except Exception as e:
        if temp_wav is not None:
            temp_wav.unlink(missing_ok=True)
        return AlignmentResult("mms_pinyin", "error", str(e), [])

    # Convert frame indices -> seconds
    ratio = waveform.shape[1] / emission.shape[1] / 16000

    # Map word occurrences in Chinese chars
    full_text = "".join(chars)
    word = to_traditional(word)
    full_text = to_traditional(full_text)

    occ = []
    start = 0
    while True:
        idx = full_text.find(word, start)
        if idx == -1:
            break
        end = idx + len(word)
        if end <= len(spans):
            start_span = spans[idx][0] if isinstance(spans[idx], list) else spans[idx]
            end_span = spans[end - 1][-1] if isinstance(spans[end - 1], list) else spans[end - 1]
            span_start = start_span.start * ratio
            span_end = end_span.end * ratio
            occ.append({
                "start": float(span_start),
                "end": float(span_end),
                "text": full_text[idx:end],
                "confidence": float(getattr(end_span, "score", 0.0)),
                "found_variant": word,
                "match_type": "pinyin",
            })
        start = idx + 1

    if temp_wav is not None:
        temp_wav.unlink(missing_ok=True)

    return AlignmentResult("mms_pinyin", "ok", None, occ)


def _occurrences_from_tokens(tokens: list[dict], word: str, match_type: str) -> list[dict]:
    target = to_traditional(word)
    norm_tokens = [to_traditional(t.get("word", "").strip()) for t in tokens]
    occ = []
    for i in range(len(norm_tokens)):
        if not norm_tokens[i]:
            continue
        if norm_tokens[i] == target:
            occ.append({
                "start": tokens[i].get("start"),
                "end": tokens[i].get("end"),
                "text": norm_tokens[i],
                "confidence": tokens[i].get("score") or tokens[i].get("probability"),
                "found_variant": target,
                "match_type": match_type,
            })
            continue
        if len(target) > 1:
            acc = ""
            for j in range(i, len(norm_tokens)):
                if not norm_tokens[j]:
                    break
                acc += norm_tokens[j]
                if not target.startswith(acc):
                    break
                if acc == target:
                    occ.append({
                        "start": tokens[i].get("start"),
                        "end": tokens[j].get("end"),
                        "text": acc,
                        "confidence": tokens[j].get("score") or tokens[j].get("probability"),
                        "found_variant": target,
                        "match_type": match_type,
                    })
                    break
    return occ


def whisperx_align(audio_path: Path, word: str) -> AlignmentResult:
    try:
        import torch
        import typing
        import collections
        from omegaconf import listconfig, dictconfig, base, nodes as oc_nodes
        import whisperx
    except Exception as e:
        return AlignmentResult("whisperx", "error", f"missing deps: {e}", [])

    try:
        safe_nodes = [v for v in vars(oc_nodes).values() if isinstance(v, type)]
        torch.serialization.add_safe_globals([
            listconfig.ListConfig,
            dictconfig.DictConfig,
            base.ContainerMetadata,
            typing.Any,
            list,
            dict,
            tuple,
            set,
            collections.defaultdict,
            int,
            float,
            str,
            bool,
            torch.torch_version.TorchVersion,
        ] + safe_nodes)
        # WhisperX uses torch.load internally; force weights_only=False to avoid safelist errors.
        orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)
        torch.load = _patched_load
        device = "cpu"
        model = whisperx.load_model("small", device, compute_type="int8")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=8)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    except Exception as e:
        return AlignmentResult("whisperx", "error", str(e), [])

    tokens = []
    for seg in result.get("segments", []):
        tokens.extend(seg.get("words", []) or [])
    occ = _occurrences_from_tokens(tokens, word, "whisperx")
    return AlignmentResult("whisperx", "ok", None, occ)


def stable_ts_align(audio_path: Path, transcript: str, word: str) -> AlignmentResult:
    try:
        import stable_whisper
    except Exception as e:
        return AlignmentResult("stable_ts", "error", f"missing deps: {e}", [])

    try:
        model = stable_whisper.load_model("base")
        try:
            result = model.align(str(audio_path), transcript, "Chinese")
        except Exception:
            result = model.align(str(audio_path), transcript, "zh")
        data = result.to_dict() if hasattr(result, "to_dict") else result
    except Exception as e:
        return AlignmentResult("stable_ts", "error", str(e), [])

    tokens = []
    segments = data.get("segments", []) if isinstance(data, dict) else []
    for seg in segments:
        tokens.extend(seg.get("words", []) or [])
    occ = _occurrences_from_tokens(tokens, word, "stable_ts")
    return AlignmentResult("stable_ts", "ok", None, occ)


def mfa_align(audio_path: Path, transcript: str, word: str, work_dir: Path) -> AlignmentResult:
    if shutil.which("mfa") is None:
        return AlignmentResult("mfa", "skip", "mfa binary not found", [])

    # MFA requires a corpus directory, dictionary, and acoustic model.
    # This is a thin wrapper; expects MFA_DICT and MFA_MODEL env vars.
    mfa_dict = os.environ.get("MFA_DICT")
    mfa_model = os.environ.get("MFA_MODEL")
    if not mfa_dict or not mfa_model:
        return AlignmentResult("mfa", "skip", "set MFA_DICT and MFA_MODEL", [])

    corpus = work_dir / "mfa_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    wav_path = corpus / "sample.wav"
    txt_path = corpus / "sample.txt"
    wav_path.write_bytes(convert_audio_to_wav(audio_path).read_bytes())
    txt_path.write_text(transcript, encoding="utf-8")

    out_dir = work_dir / "mfa_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus),
        mfa_dict,
        mfa_model,
        str(out_dir),
        "--clean"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as e:
        return AlignmentResult("mfa", "error", str(e), [])

    # Parse TextGrid if produced
    textgrid = next(out_dir.glob("**/*.TextGrid"), None)
    if not textgrid:
        return AlignmentResult("mfa", "error", "TextGrid not found", [])

    try:
        from textgrid import TextGrid
    except Exception as e:
        return AlignmentResult("mfa", "error", f"missing textgrid lib: {e}", [])

    tg = TextGrid.fromFile(str(textgrid))
    occ = []
    target = to_traditional(word)
    for tier in tg.tiers:
        for interval in tier.intervals:
            token = to_traditional(interval.mark)
            if target in token:
                occ.append({
                    "start": interval.minTime,
                    "end": interval.maxTime,
                    "text": token,
                    "confidence": None,
                    "found_variant": target,
                    "match_type": "mfa",
                })
    return AlignmentResult("mfa", "ok", None, occ)


def whispercpp_native(audio_path: Path, word: str) -> AlignmentResult:
    try:
        occ = localize_word_with_asr(audio_path, word)
        return AlignmentResult("whispercpp_native", "ok", None, occ)
    except Exception as e:
        return AlignmentResult("whispercpp_native", "error", str(e), [])


def main():
    parser = argparse.ArgumentParser(description="Compare alignment methods on a sample.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID, help="Sample ID in eval_results.json")
    parser.add_argument("--word", help="Target word to align")
    parser.add_argument("--output", default="eval_samples/align_ep499_report.json",
                        help="Output JSON path")
    args = parser.parse_args()

    item = load_eval_sample(args.sample_id)
    sample = item["sample"]
    audio_path = EVAL_AUDIO_DIR / Path(sample["audio_path"]).name
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio: {audio_path}")

    fun_text = item["transcriptions"].get("funasr-nano", {}).get("text", "")
    whisper_text = item["transcriptions"].get("whisper-large-v3", {}).get("text", "")

    word = args.word or choose_word(fun_text, whisper_text)

    results = []
    results.append(whispercpp_native(audio_path, word))
    results.append(mms_char_align(audio_path, fun_text, word))
    results.append(mms_pinyin_align(audio_path, fun_text, word))
    results.append(whisperx_align(audio_path, word))
    results.append(stable_ts_align(audio_path, fun_text, word))

    work_dir = Path("/tmp/align_compare")
    results.append(mfa_align(audio_path, fun_text, word, work_dir))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": sample,
        "word": word,
        "funasr_text": fun_text,
        "whisper_text": whisper_text,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Simple markdown summary
    md_path = out_path.with_suffix(".md")
    lines = [
        f"# Alignment Comparison for {sample['title']}",
        f"- sample_id: `{sample['sample_id']}`",
        f"- word: `{word}`",
        "",
        "## Results",
    ]
    for r in results:
        lines.append(f"- **{r.method}**: {r.status}")
        if r.error:
            lines.append(f"  - error: {r.error}")
        if r.occurrences:
            first = r.occurrences[0]
            lines.append(f"  - first: {first.get('start')}s -> {first.get('end')}s, text: {first.get('text')}")
        else:
            lines.append("  - occurrences: 0")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_path} and {md_path}")


if __name__ == "__main__":
    main()
