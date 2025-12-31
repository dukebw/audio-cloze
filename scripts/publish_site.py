#!/usr/bin/env python3
"""Publish audio-cloze HTML + audio to S3 + personal-website."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm")
AUDIO_EXTS_PATTERN = "|".join(ext.lstrip(".") for ext in AUDIO_EXTS)
AUDIO_ATTR_RE = re.compile(
    rf"(src|href)=(['\"])(?!https?://|data:|//)([^'\"]+\.({AUDIO_EXTS_PATTERN}))\2",
    re.IGNORECASE,
)


def _rewrite_audio_urls(html: str, base_url: str, subpath: str) -> str:
    base = base_url.rstrip("/")
    sub = subpath.strip("/")

    def _repl(match: re.Match[str]) -> str:
        attr = match.group(1)
        quote = match.group(2)
        rel = match.group(3)
        rel = rel.lstrip("./")
        rel = rel.lstrip("/")
        url = f"{base}/{sub}/{rel}"
        return f"{attr}={quote}{url}{quote}"

    return AUDIO_ATTR_RE.sub(_repl, html)


def _copy_html(src: Path, dest_dir: Path, audio_base_url: str, audio_subpath: str) -> None:
    if not src.exists():
        print(f"WARN: missing {src}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    html = src.read_text(encoding="utf-8")
    html = _rewrite_audio_urls(html, audio_base_url, audio_subpath)
    (dest_dir / "index.html").write_text(html, encoding="utf-8")


def _copy_optional(src: Path, dest_dir: Path, dest_name: str | None = None) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (dest_name or src.name)
    shutil.copy2(src, target)


def _run(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    if dry_run:
        print(f"DRY RUN: {printable}")
        return
    subprocess.run(cmd, check=True)


def _sync_audio(local_dir: Path, s3_uri: str, dry_run: bool, delete: bool) -> None:
    if not local_dir.exists():
        print(f"WARN: missing audio dir {local_dir}")
        return
    cmd = ["aws", "s3", "sync", str(local_dir), s3_uri, "--exclude", "*"]
    for ext in AUDIO_EXTS:
        cmd.extend(["--include", f"*{ext}"])
    if delete:
        cmd.append("--delete")
    _run(cmd, dry_run=dry_run)


def build_dist(root: Path, dist: Path, audio_base_url: str) -> None:
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True, exist_ok=True)

    # Landing page
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>audio-cloze artifacts</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ margin-bottom: 10px; }}
    ul {{ line-height: 1.8; }}
    .meta {{ color: #666; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>audio-cloze artifacts</h1>
  <div class="meta">Updated {updated}</div>
  <ul>
    <li><a href="review/">Review UI</a></li>
    <li><a href="evals/zh/">Mandarin evals</a></li>
    <li><a href="evals/ja/">Japanese evals</a></li>
  </ul>
</body>
</html>
"""
    (dist / "index.html").write_text(index_html, encoding="utf-8")

    # Review UI
    _copy_html(
        root / "audio_clips" / "review.html",
        dist / "review",
        audio_base_url,
        "audio-cloze/review",
    )

    # Mandarin eval
    _copy_html(
        root / "eval_samples" / "eval_compare.html",
        dist / "evals" / "zh",
        audio_base_url,
        "audio-cloze/evals/zh",
    )
    _copy_optional(
        root / "eval_samples" / "eval_compare_text.json",
        dist / "evals" / "zh",
    )

    # Japanese eval
    _copy_html(
        root / "eval_samples" / "jp_4w0bjx3L_gw" / "eval_compare.html",
        dist / "evals" / "ja",
        audio_base_url,
        "audio-cloze/evals/ja",
    )
    _copy_optional(
        root / "eval_samples" / "jp_4w0bjx3L_gw" / "eval_compare_text.json",
        dist / "evals" / "ja",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish audio-cloze artifacts.")
    parser.add_argument(
        "--site-root",
        default="~/work/personal-website/audio-cloze",
        help="Destination folder in personal-website repo.",
    )
    parser.add_argument(
        "--audio-base-url",
        default=os.environ.get("AUDIO_CLOZE_AUDIO_BASE_URL"),
        help="Base URL for hosted audio (e.g. https://audio.brendanduke.ca).",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("AUDIO_CLOZE_S3_BUCKET"),
        help="S3 bucket URI (e.g. s3://brendanduke-audio).",
    )
    parser.add_argument(
        "--skip-audio-upload",
        action="store_true",
        help="Skip uploading audio to S3.",
    )
    parser.add_argument(
        "--skip-site-sync",
        action="store_true",
        help="Skip syncing HTML into personal-website.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files not present locally during sync.",
    )
    args = parser.parse_args()

    if not args.audio_base_url:
        print("ERROR: --audio-base-url or AUDIO_CLOZE_AUDIO_BASE_URL is required.")
        return 2
    if not args.skip_audio_upload and not args.s3_bucket:
        print("ERROR: --s3-bucket or AUDIO_CLOZE_S3_BUCKET is required for audio upload.")
        return 2

    root = Path(__file__).resolve().parents[1]
    dist = root / "dist" / "site"

    build_dist(root, dist, args.audio_base_url)

    if not args.skip_audio_upload:
        bucket = args.s3_bucket.rstrip("/")
        _sync_audio(root / "audio_clips", f"{bucket}/audio-cloze/review", args.dry_run, args.delete)
        _sync_audio(root / "eval_samples" / "audio", f"{bucket}/audio-cloze/evals/zh/audio", args.dry_run, args.delete)
        _sync_audio(
            root / "eval_samples" / "jp_4w0bjx3L_gw" / "audio",
            f"{bucket}/audio-cloze/evals/ja/audio",
            args.dry_run,
            args.delete,
        )

    if not args.skip_site_sync:
        site_root = Path(args.site_root).expanduser()
        site_root.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-av", "--delete", str(dist) + "/", str(site_root) + "/"]
        _run(cmd, dry_run=args.dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
