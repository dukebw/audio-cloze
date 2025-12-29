#!/usr/bin/env python3
"""Benchmark whisper.cpp vs mlx-whisper on Apple Silicon."""

import json, subprocess, sys, time
from pathlib import Path

AUDIO_FILE = Path(__file__).parent / "benchmark_sample.mp3"
MODEL_PATH = Path(__file__).parent / "models" / "ggml-large-v3.bin"
TEST_WORDS = ["護城河", "蘋果", "產品", "市場", "功能", "用戶", "體驗"]


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


def print_result(name: str, duration: float, elapsed: float, words: list):
    speed = duration / elapsed
    found = {w: [x for x in words if w in x["word"]] for w in TEST_WORDS}
    found = {k: v for k, v in found.items() if v}
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"  {duration:.0f}s audio in {elapsed:.1f}s ({speed:.1f}x realtime)")
    print(f"  {len(words)} words, {len(found)}/{len(TEST_WORDS)} vocab found")
    return {"impl": name, "time": elapsed, "speed": speed, "words": len(words), "vocab": len(found)}


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
    return print_result("whisper.cpp", duration, elapsed, words)


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
    return print_result("mlx-whisper", duration, elapsed, words)


def main():
    print(f"Whisper Benchmark - {AUDIO_FILE.name}")
    if not AUDIO_FILE.exists():
        sys.exit(f"ERROR: {AUDIO_FILE} not found")

    results = [r for r in [bench_whisper_cpp(), bench_mlx_whisper()] if r]
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
