# Roadmap

What's done, what's next.

## Done

### Multi-source indexing
- [x] YouTube channel crawling with yt-dlp
- [x] Subtitle download and FTS5 indexing
- [x] RSS podcast feed parsing
- [x] Full podcast ASR transcript indexing
- [x] Unified search across both sources
- [x] `sources.txt` config format

### ASR for podcasts
- [x] Full transcript index (chunked ASR over episodes)
- [x] whisper.cpp integration (replaces slower faster-whisper)
- [x] CoreML support for Apple Neural Engine
- [x] mlx-whisper benchmarked as alternative (similar speed)
- [x] Smart chunking for long episodes (5-min chunks, early exit)
- [x] Simplified/Traditional Chinese variant matching
- [x] Pluggable ASR backend system (`--asr-backend` CLI arg)
- [x] Fun-ASR-Nano backend
- [x] GLM-ASR backend
- [x] Forced alignment with torchaudio MMS_FA (for backends without native timestamps)
- [x] Transcript caching in SQLite

### Clip generation
- [x] Audio extraction with ffmpeg
- [x] Cloze audio generation (silence target word)
- [x] Clip merging across multiple mining runs
- [x] HTML review interface with approve/reject

---

## Next up

### Review UI improvements
- [ ] Keyboard shortcuts (j/k navigate, a approve, r reject)
- [ ] Filter by status (pending/approved/rejected)
- [ ] Search within clips
- [ ] Dark mode

### Anki export
- [ ] Direct AnkiConnect integration
- [ ] Batch export approved clips
- [ ] Custom note type with audio fields

### Claude fallback
When a word isn't in the index:
- [ ] Ask Claude for synonyms and search suggestions
- [ ] Auto-search YouTube for suggested queries
- [ ] Index new videos on the fly
- [ ] Retry original word

### More sources
- [ ] WenetSpeech corpus support (pre-transcribed, huge)
- [ ] Bilibili videos (different subtitle format)
- [ ] Local audio files with manual transcripts

---

## Maybe later

- **Sentence difficulty scoring** - prioritize clips at user's level
- **Spaced repetition integration** - track which words are "learned"
- **Multi-language** - same approach works for Japanese, Korean, etc.
- **Web UI** - Flask/FastAPI instead of CLI
- **Pre-built index distribution** - share indexed databases

---

## Technical debt

- [ ] Better error handling for network failures
- [ ] Resume interrupted indexing
- [ ] Dedup clips with same audio (different videos, same sentence)
- [ ] Tests

---

## Known Issues

### Fun-ASR-Nano (resolved Dec 2025)

Fun-ASR-Nano now loads correctly via `funasr.AutoModel` using
`trust_remote_code=True` and `remote_code="./model.py"` as specified in the
model card. The backend keeps the forced-alignment fallback for timestamps.

---

### GLM-ASR (resolved Dec 2025)

GLM-ASR now runs locally via transformers 5.x using `AutoProcessor` +
`AutoModelForSeq2SeqLM`, removing the dependency on an external SGLang server.
Endpoint mode remains available for OpenAI-compatible deployments.

---

### Evaluation Results (Dec 29, 2025)

Ran ASR accuracy benchmarks on 30 samples (18 YouTube + 12 podcast):

| Backend | Status | Avg Time | Avg Chars |
|---------|--------|----------|-----------|
| Whisper-Large-V3 | Working | 3.8s | 247 |
| Fun-ASR-Nano | Working | 4.1s | 291 |
| GLM-ASR | Working | 4.2s | 285 |

Results saved to `eval_samples/eval_compare.html` for visual comparison.

---

## Benchmarks

Current numbers on M4 Max:

| Task | Time |
|------|------|
| Index 50 YouTube videos | ~10 min |
| Index 50 podcast episodes | depends on ASR backend (full transcription) |
| Mine 1 word (YouTube hit) | instant |
| Mine 1 word (podcast, needs alignment) | ~20 sec per 5-min chunk |

ASR backends (5-min audio):

| Backend | Speed | Word Timestamps | Status |
|---------|-------|-----------------|--------|
| whisper.cpp + CoreML | 12-20x RT | Native | Working |
| mlx-whisper | 12-15x RT | Native | Working |
| Fun-ASR-Nano | — | Forced alignment | Working |
| GLM-ASR | — | Forced alignment | Working |
