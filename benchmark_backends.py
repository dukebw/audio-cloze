#!/usr/bin/env python3
"""
Benchmark: whisper.cpp backends on M4 Max
- Metal GPU only
- CoreML (ANE) + Metal
- CPU only

Tests on 5-minute Chinese podcast audio with word timestamps.
"""

import json
import subprocess
import time
from pathlib import Path

# Paths
WHISPER_CLI = Path("/Users/bduke/work/whisper-cpp-coreml/build/bin/whisper-cli")
HOMEBREW_CLI = Path("/opt/homebrew/bin/whisper-cli")
MODEL_DIR = Path("/Users/bduke/work/whisper-cpp-coreml/models")
GGML_MODEL = MODEL_DIR / "ggml-large-v3.bin"
AUDIO_FILE = Path("/Users/bduke/work/audio-cloze/benchmark_sample.mp3")

# Optimal thread settings from previous benchmark
THREADS = 8
PROCESSORS = 4


def convert_to_wav(audio_path: Path) -> Path:
    """Convert to 16kHz WAV for whisper.cpp."""
    wav_path = audio_path.with_suffix('.wav')
    if not wav_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path)
        ], capture_output=True)
    return wav_path


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def run_benchmark(name: str, cmd: list, audio_duration: float) -> dict:
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd[:5])}...")

    # First run may be slow for CoreML (ANE compilation)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    speed = audio_duration / elapsed
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {speed:.2f}x real-time")

    # Check for CoreML loading
    if "coreml" in proc.stderr.lower() or "ane" in proc.stderr.lower():
        print("  Note: CoreML/ANE initialized")

    return {
        "name": name,
        "time": elapsed,
        "speed": speed,
        "stderr": proc.stderr[:500] if proc.returncode != 0 else ""
    }


def main():
    print("=" * 60)
    print("WHISPER.CPP BACKEND BENCHMARK - M4 Max")
    print("=" * 60)

    # Check prerequisites
    if not WHISPER_CLI.exists():
        print(f"ERROR: whisper-cli not found at {WHISPER_CLI}")
        return
    if not GGML_MODEL.exists():
        print(f"ERROR: Model not found at {GGML_MODEL}")
        return
    if not AUDIO_FILE.exists():
        print(f"ERROR: Audio file not found at {AUDIO_FILE}")
        return

    # Convert audio
    print("\nPreparing audio...")
    wav_file = convert_to_wav(AUDIO_FILE)
    duration = get_duration(wav_file)
    print(f"Audio: {wav_file.name} ({duration:.1f}s / {duration/60:.1f} min)")

    output_base = Path("/tmp/whisper_backend_test")
    results = []

    # Test 1: Metal only (default, no CoreML encoder)
    print("\n" + "-" * 60)
    print("Test 1: Metal GPU (no CoreML)")
    cmd = [
        str(WHISPER_CLI),
        "-m", str(GGML_MODEL),
        "-l", "zh",
        "-t", str(THREADS),
        "-p", str(PROCESSORS),
        "-ojf",
        "-of", str(output_base),
        str(wav_file)
    ]
    r = run_benchmark("Metal GPU", cmd, duration)
    results.append(r)

    # Clean up
    for f in output_base.parent.glob(f"{output_base.name}*"):
        f.unlink()

    # Test 2: CoreML + Metal (with encoder.mlmodelc)
    # CoreML requires the encoder model to be in same directory as ggml model
    coreml_encoder = MODEL_DIR / "ggml-large-v3-encoder.mlmodelc"
    if coreml_encoder.exists():
        print("\n" + "-" * 60)
        print("Test 2: CoreML (ANE) + Metal GPU")
        print("  Note: First run includes ANE compilation time")

        # First run (cold)
        cmd = [
            str(WHISPER_CLI),
            "-m", str(GGML_MODEL),
            "-l", "zh",
            "-t", str(THREADS),
            "-p", str(PROCESSORS),
            "-ojf",
            "-of", str(output_base),
            str(wav_file)
        ]
        r = run_benchmark("CoreML+Metal (cold)", cmd, duration)
        results.append(r)

        for f in output_base.parent.glob(f"{output_base.name}*"):
            f.unlink()

        # Second run (warm - ANE already compiled)
        print("\n" + "-" * 60)
        print("Test 3: CoreML + Metal (warm)")
        r = run_benchmark("CoreML+Metal (warm)", cmd, duration)
        results.append(r)

        for f in output_base.parent.glob(f"{output_base.name}*"):
            f.unlink()
    else:
        print(f"\nSkipping CoreML: encoder not found at {coreml_encoder}")

    # Test 4: CPU only (no GPU)
    print("\n" + "-" * 60)
    print("Test 4: CPU only (no GPU)")
    cmd = [
        str(WHISPER_CLI),
        "-m", str(GGML_MODEL),
        "-l", "zh",
        "-t", str(THREADS),
        "-p", str(PROCESSORS),
        "-ng",  # no GPU
        "-ojf",
        "-of", str(output_base),
        str(wav_file)
    ]
    r = run_benchmark("CPU only", cmd, duration)
    results.append(r)

    for f in output_base.parent.glob(f"{output_base.name}*"):
        f.unlink()

    # Test 5: Homebrew version (Metal, for comparison)
    if HOMEBREW_CLI.exists():
        print("\n" + "-" * 60)
        print("Test 5: Homebrew whisper-cli (Metal)")
        cmd = [
            str(HOMEBREW_CLI),
            "-m", str(Path("/Users/bduke/work/audio-cloze/models/ggml-large-v3.bin")),
            "-l", "zh",
            "-t", str(THREADS),
            "-p", str(PROCESSORS),
            "-ojf",
            "-of", str(output_base),
            str(wav_file)
        ]
        r = run_benchmark("Homebrew Metal", cmd, duration)
        results.append(r)

        for f in output_base.parent.glob(f"{output_base.name}*"):
            f.unlink()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Backend':<25} {'Time (s)':<12} {'Speed (x RT)':<15}")
    print("-" * 52)

    fastest = min(results, key=lambda x: x["time"])
    for r in results:
        marker = " <-- FASTEST" if r["name"] == fastest["name"] else ""
        print(f"{r['name']:<25} {r['time']:<12.2f} {r['speed']:<15.2f}{marker}")

    # Calculate speedups relative to CPU
    cpu_result = next((r for r in results if "CPU" in r["name"]), None)
    if cpu_result:
        print(f"\nSpeedup vs CPU:")
        for r in results:
            if r != cpu_result:
                speedup = cpu_result["time"] / r["time"]
                print(f"  {r['name']}: {speedup:.2f}x faster")

    # Save results
    with open(Path(__file__).parent / "benchmark_backends.json", "w") as f:
        json.dump({
            "audio_duration": duration,
            "hardware": "Apple M4 Max",
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to benchmark_backends.json")


if __name__ == "__main__":
    main()
