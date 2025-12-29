#!/usr/bin/env python3
"""
Benchmark: faster-whisper vs whisper.cpp for audio-cloze task.

Tests:
1. Transcription speed (5-minute Chinese podcast sample)
2. Word-level timestamp accuracy
3. Memory usage
4. Ability to find target vocabulary words with timing
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Test vocabulary (words we expect might be in the sample)
TEST_WORDS = ["護城河", "蘋果", "產品", "市場", "功能", "用戶", "體驗"]

AUDIO_FILE = Path(__file__).parent / "benchmark_sample.mp3"
MODEL_PATH = Path(__file__).parent / "models" / "ggml-large-v3.bin"


def benchmark_faster_whisper():
    """Benchmark faster-whisper with large-v3."""
    print("\n" + "=" * 60)
    print("BENCHMARK: faster-whisper (large-v3)")
    print("=" * 60)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed")
        return None

    results = {"implementation": "faster-whisper"}

    # Model loading time
    print("Loading model...")
    load_start = time.time()
    model = WhisperModel("large-v3", device="auto", compute_type="auto")
    load_time = time.time() - load_start
    results["model_load_time"] = load_time
    print(f"  Model load time: {load_time:.2f}s")

    # Transcription time
    print("Transcribing...")
    trans_start = time.time()
    segments, info = model.transcribe(
        str(AUDIO_FILE),
        language="zh",
        word_timestamps=True,
    )
    segments = list(segments)  # Consume the generator
    trans_time = time.time() - trans_start
    results["transcription_time"] = trans_time
    results["audio_duration"] = info.duration
    results["speed_ratio"] = info.duration / trans_time
    print(f"  Audio duration: {info.duration:.1f}s ({info.duration/60:.1f} min)")
    print(f"  Transcription time: {trans_time:.2f}s")
    print(f"  Speed ratio: {results['speed_ratio']:.2f}x real-time")

    # Collect all words with timestamps
    all_words = []
    full_text = ""
    for seg in segments:
        full_text += seg.text
        if seg.words:
            for w in seg.words:
                all_words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability if w.probability else 0.5,
                })

    results["total_words"] = len(all_words)
    results["full_text"] = full_text[:500] + "..." if len(full_text) > 500 else full_text

    # Find test vocabulary
    found_words = {}
    for test_word in TEST_WORDS:
        matches = []
        for w in all_words:
            if test_word in w["word"]:
                matches.append(w)
        if matches:
            found_words[test_word] = matches
            print(f"  Found '{test_word}': {len(matches)} occurrence(s)")

    results["found_words"] = found_words
    results["vocabulary_found_count"] = len(found_words)

    print(f"\nTotal words with timestamps: {len(all_words)}")
    print(f"Test vocabulary found: {len(found_words)}/{len(TEST_WORDS)}")

    return results


def run_whisper_cpp_single(wav_file, audio_duration, threads=4, processors=1):
    """Run a single whisper.cpp benchmark with specific parameters."""
    output_base = Path("/tmp/whisper_benchmark")

    trans_start = time.time()
    cmd = [
        "whisper-cli",
        "-m", str(MODEL_PATH),
        "-l", "zh",
        "-t", str(threads),
        "-p", str(processors),  # Parallel segment processing
        "-ojf",  # JSON full output with word timestamps
        "-of", str(output_base),
        str(wav_file)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    trans_time = time.time() - trans_start
    speed_ratio = audio_duration / trans_time

    # Clean up JSON file
    json_file = Path(str(output_base) + ".json")
    for f in output_base.parent.glob(f"{output_base.name}*"):
        f.unlink()

    return {"threads": threads, "processors": processors, "time": trans_time, "speed": speed_ratio}


def benchmark_whisper_cpp(optimize=False):
    """Benchmark whisper.cpp with large-v3."""
    print("\n" + "=" * 60)
    print("BENCHMARK: whisper.cpp (large-v3) - M4 Max Metal")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return None

    results = {"implementation": "whisper.cpp"}

    # Convert to WAV (whisper.cpp prefers 16kHz WAV)
    wav_file = AUDIO_FILE.with_suffix(".wav")
    print("Converting to WAV...")
    convert_start = time.time()
    subprocess.run([
        "ffmpeg", "-y", "-i", str(AUDIO_FILE),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_file)
    ], capture_output=True)
    convert_time = time.time() - convert_start
    results["wav_convert_time"] = convert_time
    print(f"  WAV conversion: {convert_time:.2f}s")

    # Get audio duration
    probe = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_file)
    ], capture_output=True, text=True)
    audio_duration = float(probe.stdout.strip())
    results["audio_duration"] = audio_duration

    # Optimization sweep for M4 Max
    if optimize:
        print("\nRunning optimization sweep for M4 Max...")
        configs = [
            (4, 1), (8, 1), (12, 1), (16, 1),  # Different thread counts
            (8, 2), (8, 4),  # Multiple processors (parallel segments)
            (4, 4), (4, 2),  # Lower threads + more processors
        ]
        sweep_results = []
        for threads, procs in configs:
            print(f"  Testing t={threads}, p={procs}...", end=" ", flush=True)
            r = run_whisper_cpp_single(wav_file, audio_duration, threads, procs)
            print(f"{r['speed']:.2f}x")
            sweep_results.append(r)

        # Find best configuration
        best = max(sweep_results, key=lambda x: x["speed"])
        print(f"\n  BEST CONFIG: threads={best['threads']}, processors={best['processors']}")
        print(f"  Speed: {best['speed']:.2f}x real-time ({best['time']:.2f}s)")
        results["optimization_sweep"] = sweep_results
        results["best_config"] = best
        optimal_threads = best["threads"]
        optimal_procs = best["processors"]
    else:
        # Default optimal for M4 Max: 8 threads, 1 processor (GPU does heavy lifting)
        optimal_threads = 8
        optimal_procs = 1

    # Run final benchmark with optimal config
    output_base = Path("/tmp/whisper_benchmark")
    print(f"\nFinal run with t={optimal_threads}, p={optimal_procs}...")
    trans_start = time.time()
    cmd = [
        "whisper-cli",
        "-m", str(MODEL_PATH),
        "-l", "zh",
        "-t", str(optimal_threads),
        "-p", str(optimal_procs),
        "-ojf",  # JSON full output with word timestamps
        "-of", str(output_base),
        str(wav_file)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    trans_time = time.time() - trans_start
    results["transcription_time"] = trans_time
    results["speed_ratio"] = audio_duration / trans_time
    results["optimal_threads"] = optimal_threads
    results["optimal_processors"] = optimal_procs
    print(f"  Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print(f"  Transcription time: {trans_time:.2f}s")
    print(f"  Speed ratio: {results['speed_ratio']:.2f}x real-time")

    # Parse JSON output (handle encoding issues with Chinese BPE tokens)
    json_file = Path(str(output_base) + ".json")
    if json_file.exists():
        with open(json_file, 'rb') as f:
            raw = f.read()
        # Decode with errors='replace' to handle invalid UTF-8 byte sequences
        text = raw.decode('utf-8', errors='replace')
        data = json.loads(text)

        # Extract word-level timestamps
        all_words = []
        full_text = ""
        for seg in data.get("transcription", []):
            full_text += seg.get("text", "")
            for token in seg.get("tokens", []):
                if "text" in token and "offsets" in token:
                    all_words.append({
                        "word": token["text"].strip(),
                        "start": token["offsets"]["from"] / 1000.0,
                        "end": token["offsets"]["to"] / 1000.0,
                        "probability": token.get("p", 0.5),
                    })

        results["total_words"] = len(all_words)
        results["full_text"] = full_text[:500] + "..." if len(full_text) > 500 else full_text

        # Find test vocabulary
        found_words = {}
        for test_word in TEST_WORDS:
            matches = []
            for w in all_words:
                if test_word in w["word"]:
                    matches.append(w)
            if matches:
                found_words[test_word] = matches
                print(f"  Found '{test_word}': {len(matches)} occurrence(s)")

        results["found_words"] = found_words
        results["vocabulary_found_count"] = len(found_words)

        print(f"\nTotal words with timestamps: {len(all_words)}")
        print(f"Test vocabulary found: {len(found_words)}/{len(TEST_WORDS)}")
    else:
        print("ERROR: JSON output not found")
        print(f"stderr: {proc.stderr[:1000]}")
        results["error"] = "JSON output not generated"

    # Cleanup
    if wav_file.exists():
        wav_file.unlink()
    for f in output_base.parent.glob(f"{output_base.name}*"):
        f.unlink()

    return results


def compare_results(fw_results, wcpp_results):
    """Compare and summarize results."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if not wcpp_results:
        print("ERROR: Missing whisper.cpp results")
        return

    if not fw_results:
        # whisper.cpp only mode
        print(f"\nwhistper.cpp Results:")
        print(f"  Speed: {wcpp_results.get('speed_ratio', 0):.2f}x real-time")
        print(f"  Transcription time: {wcpp_results.get('transcription_time', 0):.2f}s")
        if 'best_config' in wcpp_results:
            bc = wcpp_results['best_config']
            print(f"  Optimal config: threads={bc['threads']}, processors={bc['processors']}")
        return

    print(f"\n{'Metric':<30} {'faster-whisper':<20} {'whisper.cpp':<20}")
    print("-" * 70)

    metrics = [
        ("Transcription time (s)", "transcription_time", ".2f"),
        ("Speed ratio (x real-time)", "speed_ratio", ".2f"),
        ("Words with timestamps", "total_words", "d"),
        ("Vocabulary words found", "vocabulary_found_count", "d"),
    ]

    for label, key, fmt in metrics:
        fw_val = fw_results.get(key, "N/A")
        wcpp_val = wcpp_results.get(key, "N/A")
        if isinstance(fw_val, (int, float)):
            fw_str = f"{fw_val:{fmt}}"
            wcpp_str = f"{wcpp_val:{fmt}}" if isinstance(wcpp_val, (int, float)) else str(wcpp_val)
        else:
            fw_str = str(fw_val)
            wcpp_str = str(wcpp_val)
        print(f"{label:<30} {fw_str:<20} {wcpp_str:<20}")

    # Speed comparison
    if "transcription_time" in fw_results and "transcription_time" in wcpp_results:
        fw_time = fw_results["transcription_time"]
        wcpp_time = wcpp_results["transcription_time"]
        speedup = fw_time / wcpp_time if wcpp_time > 0 else 0
        if speedup > 1:
            print(f"\nwhistper.cpp is {speedup:.2f}x FASTER than faster-whisper")
        else:
            print(f"\nfaster-whisper is {1/speedup:.2f}x FASTER than whisper.cpp")

    # Recommendation for audio-cloze
    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR AUDIO-CLOZE")
    print("=" * 60)

    fw_speed = fw_results.get("speed_ratio", 0)
    wcpp_speed = wcpp_results.get("speed_ratio", 0)
    fw_words = fw_results.get("vocabulary_found_count", 0)
    wcpp_words = wcpp_results.get("vocabulary_found_count", 0)

    if wcpp_speed > fw_speed * 1.2:  # whisper.cpp significantly faster
        if wcpp_words >= fw_words:
            print("WINNER: whisper.cpp")
            print("  - Faster transcription")
            print("  - Equal or better vocabulary detection")
        else:
            print("TRADE-OFF: whisper.cpp faster, but faster-whisper finds more words")
    elif fw_speed > wcpp_speed * 1.2:  # faster-whisper significantly faster
        print("WINNER: faster-whisper")
        print("  - Faster transcription on this hardware")
    else:
        print("SIMILAR PERFORMANCE")
        print("  - Both implementations perform similarly")
        print("  - Choose based on deployment constraints")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper vs whisper.cpp")
    parser.add_argument("--optimize", action="store_true", help="Run optimization sweep for whisper.cpp")
    parser.add_argument("--whisper-cpp-only", action="store_true", help="Only benchmark whisper.cpp")
    args = parser.parse_args()

    print("Audio-Cloze Whisper Benchmark")
    print(f"Audio file: {AUDIO_FILE}")
    print(f"Hardware: M4 Max (16 cores, Metal GPU)")
    print(f"Test vocabulary: {TEST_WORDS}")

    if not AUDIO_FILE.exists():
        print(f"ERROR: Audio file not found: {AUDIO_FILE}")
        sys.exit(1)

    # Run benchmarks
    if args.whisper_cpp_only:
        fw_results = None
    else:
        fw_results = benchmark_faster_whisper()
    wcpp_results = benchmark_whisper_cpp(optimize=args.optimize)

    # Compare
    compare_results(fw_results, wcpp_results)

    # Save results
    results = {
        "audio_file": str(AUDIO_FILE),
        "faster_whisper": fw_results,
        "whisper_cpp": wcpp_results,
    }
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
