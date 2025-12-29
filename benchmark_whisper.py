#!/usr/bin/env python3
"""Benchmark ASR backends on Apple Silicon: whisper.cpp, mlx-whisper, Fun-ASR-Nano, GLM-ASR."""

import json, os, subprocess, sys, time
from pathlib import Path
from asr_utils import (
    ensure_qwen3_weights,
    load_glm_asr_transformers,
    patch_funasr_load_in_8bit,
    select_asr_device,
)

AUDIO_FILE = Path(__file__).parent / "benchmark_sample.mp3"
MODEL_PATH = Path(__file__).parent / "models" / "ggml-large-v3.bin"
FUNASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
GLM_ASR_MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
TEST_WORDS = ["護城河", "蘋果", "產品", "市場", "功能", "用戶", "體驗"]
GLM_ASR_ENDPOINT = "http://127.0.0.1:8000/v1"  # Optional OpenAI-compatible endpoint

_FUNASR_MODEL = None
_GLM_ASR_MODEL = None
_GLM_ASR_PROCESSOR = None


def get_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def to_wav(path: Path) -> Path:
    wav = path.with_suffix('.wav')
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav)], capture_output=True)
    return wav


def print_result(name: str, duration: float, elapsed: float, words: list = None, text: str = None):
    """Print benchmark result. Accepts either word list (with timestamps) or plain text."""
    speed = duration / elapsed
    if words:
        found = {w: [x for x in words if w in x["word"]] for w in TEST_WORDS}
        found = {k: v for k, v in found.items() if v}
        word_count = len(words)
    else:
        found = {w: True for w in TEST_WORDS if w in (text or "")}
        word_count = len(text) if text else 0  # Character count for text-only
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"  {duration:.0f}s audio in {elapsed:.1f}s ({speed:.1f}x realtime)")
    if words:
        print(f"  {word_count} words, {len(found)}/{len(TEST_WORDS)} vocab found")
    else:
        print(f"  {word_count} chars, {len(found)}/{len(TEST_WORDS)} vocab found (no timestamps)")
    return {"impl": name, "time": elapsed, "speed": speed, "words": word_count, "vocab": len(found)}


def bench_whisper_cpp():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return None
    wav = to_wav(AUDIO_FILE)
    duration = get_duration(wav)
    out = Path("/tmp/wcpp_bench")

    start = time.time()
    subprocess.run(["whisper-cli", "-m", str(MODEL_PATH), "-l", "zh", "-t", "8", "-p", "4",
                    "-ojf", "-of", str(out), str(wav)], capture_output=True)
    elapsed = time.time() - start

    words = []
    jf = Path(str(out) + ".json")
    if jf.exists():
        data = json.loads(jf.read_bytes().decode('utf-8', errors='replace'))
        for seg in data.get("transcription", []):
            for tok in seg.get("tokens", []):
                if "text" in tok and "offsets" in tok:
                    words.append({"word": tok["text"].strip(),
                                  "start": tok["offsets"]["from"]/1000,
                                  "end": tok["offsets"]["to"]/1000})
        jf.unlink()
    return print_result("whisper.cpp", duration, elapsed, words=words)


def bench_mlx_whisper():
    try:
        import mlx_whisper
    except ImportError:
        print("ERROR: pip install mlx-whisper")
        return None

    duration = get_duration(AUDIO_FILE)
    start = time.time()
    result = mlx_whisper.transcribe(str(AUDIO_FILE), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                                     language="zh", word_timestamps=True)
    elapsed = time.time() - start

    words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
             for seg in result.get("segments", []) for w in seg.get("words", [])]
    return print_result("mlx-whisper", duration, elapsed, words=words)


def bench_funasr_nano():
    """Benchmark Fun-ASR-Nano (no word timestamps, text only)."""
    try:
        from funasr import AutoModel
        from funasr.models.fun_asr_nano import model as funasr_nano_model  # noqa: F401
    except ImportError:
        print("ERROR: pip install funasr")
        return None

    duration = get_duration(AUDIO_FILE)
    print("  Loading Fun-ASR-Nano model...")
    start = time.time()
    global _FUNASR_MODEL
    if _FUNASR_MODEL is None:
        ensure_qwen3_weights()
        patch_funasr_load_in_8bit()
        _FUNASR_MODEL = AutoModel(
            model=FUNASR_MODEL_ID,
            device=select_asr_device("FUNASR_DEVICE"),
            disable_update=True
        )
    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.1f}s")

    start = time.time()
    res = _FUNASR_MODEL.generate(
        input=[str(AUDIO_FILE)],
        cache={},
        batch_size=1,
        language="中文",
        itn=True
    )
    elapsed = time.time() - start

    text = res[0]["text"] if res else ""
    return print_result("funasr-nano", duration, elapsed, text=text)


def bench_glm_asr():
    """Benchmark GLM-ASR (transformers, text only)."""
    global _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR
    if _GLM_ASR_MODEL is None or _GLM_ASR_PROCESSOR is None:
        print("  Loading GLM-ASR model...")
        try:
            _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR = load_glm_asr_transformers(GLM_ASR_MODEL_ID)
        except ImportError:
            print("ERROR: pip install git+https://github.com/huggingface/transformers.git")
            return None

    duration = get_duration(AUDIO_FILE)
    start = time.time()
    inputs = _GLM_ASR_PROCESSOR.apply_transcription_request(str(AUDIO_FILE))
    dtype = getattr(_GLM_ASR_MODEL, "dtype", None)
    if dtype is not None:
        inputs = inputs.to(_GLM_ASR_MODEL.device, dtype=dtype)
    else:
        inputs = inputs.to(_GLM_ASR_MODEL.device)
    try:
        import torch
        with torch.inference_mode():
            outputs = _GLM_ASR_MODEL.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(os.environ.get("GLM_ASR_MAX_NEW_TOKENS", "512"))
            )
    except Exception:
        outputs = _GLM_ASR_MODEL.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(os.environ.get("GLM_ASR_MAX_NEW_TOKENS", "512"))
        )
    elapsed = time.time() - start
    prompt_len = inputs["input_ids"].shape[1]
    decoded = _GLM_ASR_PROCESSOR.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    text = decoded[0].strip() if decoded else ""
    return print_result("glm-asr", duration, elapsed, text=text)


def main():
    print(f"ASR Benchmark - {AUDIO_FILE.name}")
    if not AUDIO_FILE.exists():
        sys.exit(f"ERROR: {AUDIO_FILE} not found")

    benchmarks = [
        bench_whisper_cpp,
        bench_mlx_whisper,
        bench_funasr_nano,
        bench_glm_asr,
    ]
    results = [r for r in [b() for b in benchmarks] if r]
    if not results:
        return

    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    results.sort(key=lambda x: x["speed"], reverse=True)
    for r in results:
        marker = " <-- FASTEST" if r == results[0] else ""
        print(f"  {r['impl']:<15} {r['speed']:.1f}x realtime{marker}")

    Path(__file__).parent.joinpath("benchmark_results.json").write_text(
        json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
