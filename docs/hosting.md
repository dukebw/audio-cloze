# Hosting audio-cloze artifacts

This project publishes **HTML** to `brendanduke.ca` (via the `dukebw/personal-website` repo)
while hosting **audio binaries** in S3 (or behind CloudFront). This keeps the website repo lean
and scales as clips grow.

## Target URLs

- `https://brendanduke.ca/audio-cloze/` → landing page
- `https://brendanduke.ca/audio-cloze/review/` → `audio_clips/review.html`
- `https://brendanduke.ca/audio-cloze/evals/zh/` → Mandarin eval report
- `https://brendanduke.ca/audio-cloze/evals/ja/` → Japanese eval report

Audio is served from a separate origin:

- `https://audio.brendanduke.ca/audio-cloze/...` (recommended) or
- a CloudFront/S3 public URL.

## Prerequisites

- AWS CLI configured (`aws configure`)
- An S3 bucket for audio, e.g. `s3://brendanduke-audio`
- (Optional) CloudFront distribution pointing `audio.brendanduke.ca` to the bucket
- S3 CORS rule that allows `GET` from `https://brendanduke.ca` (for cross-origin audio)
- `rsync` available (macOS default)

## Publish flow

The script builds a small `dist/` folder, rewrites audio URLs to the CDN base,
uploads audio to S3, and syncs HTML into `personal-website`.

```bash
export AUDIO_CLOZE_AUDIO_BASE_URL="https://audio.brendanduke.ca"
export AUDIO_CLOZE_S3_BUCKET="s3://brendanduke-audio"

python scripts/publish_site.py --delete
```

Then commit/push the `personal-website` repo to deploy.

### Options

- `--dry-run` prints actions without running them.
- `--skip-audio-upload` only builds HTML + syncs the website repo.
- `--skip-site-sync` only uploads audio to S3.
- `--site-root ~/work/personal-website/static/audio-cloze` overrides the default target.

## Notes

- The script rewrites relative audio URLs in HTML to the CDN base.
- The JSON comparison files (if present) are copied alongside each eval report.
- Audio uploads exclude non-audio files in `audio_clips/`.
