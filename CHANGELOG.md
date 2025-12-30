# Changelog

Completed work and key snapshots.

## 2025-12-30

### Added
- Multi-source indexing: yt-dlp channel crawl, subtitle download + FTS5 indexing, RSS parsing, podcast ASR transcript indexing, unified search, `sources.txt` config.
- Podcast ASR pipeline: chunked ASR, whisper.cpp + CoreML option, mlx-whisper benchmarked, Fun-ASR-Nano + GLM-ASR backends, MMS_FA forced alignment for timestamps, transcript caching in SQLite.
- Clip generation: ffmpeg extraction, cloze silencing, clip merge across runs, HTML review UI.
- Eval report with WhisperX alignment highlighting.

### Fixed / stabilized
- Fun-ASR-Nano: Qwen3 weight bootstrap + transformers `load_in_8bit` patch.
- GLM-ASR: local transformers path via AutoProcessor + AutoModelForSeq2SeqLM (endpoint mode optional).

### Evaluation snapshot (2025-12-30)
Ran ASR accuracy benchmarks on 30 samples (18 YouTube + 12 podcast):

| Backend | Avg Time | Avg Chars | Errors |
|---------|----------|-----------|--------|
| Whisper-Large-V3 | 3.8s | 247 | 0 |
| Fun-ASR-Nano | 4.1s | 291 | 0 |
| GLM-ASR | 4.2s | 285 | 0 |

Results saved to `eval_samples/eval_compare.html` for visual comparison.
