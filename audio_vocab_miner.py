#!/usr/bin/env python3
"""
Audio Vocabulary Miner - Inverted Index Architecture

Strategy:
1. INDEX: Crawl YouTube channels → Download subtitles → Build SQLite FTS5 index
2. MINE: Query index (instant) → Download audio slice → Create cloze

No Whisper needed - trust subtitle timing directly.
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Audio processing
from pydub import AudioSegment

# Simplified/Traditional Chinese conversion
from opencc import OpenCC
_s2t = OpenCC('s2t')
_t2s = OpenCC('t2s')

# VTT parsing
import webvtt


# =============================================================================
# Database Schema
# =============================================================================

SCHEMA = """
-- Main captions table with FTS5 for Chinese full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS captions USING fts5(
    video_id,
    channel,
    start_time,
    end_time,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

-- Metadata table for video info
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel TEXT,
    title TEXT,
    duration REAL,
    subtitle_type TEXT,
    indexed_at TEXT
);

-- Track indexing progress
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    channel_url TEXT,
    channel_name TEXT,
    last_crawled TEXT,
    video_count INTEGER DEFAULT 0
);
"""


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize database with schema."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# =============================================================================
# Utility Functions
# =============================================================================

def run_command(cmd: list[str], capture_output=True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, capture_output=capture_output, text=True)


def normalize_zh(s: str) -> str:
    """Normalize Chinese text for matching."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s.lower()


def get_word_variants(word: str) -> list[str]:
    """Get both simplified and traditional variants of a word."""
    traditional = _s2t.convert(word)
    simplified = _t2s.convert(word)
    variants = [word]
    if traditional != word:
        variants.append(traditional)
    if simplified != word and simplified not in variants:
        variants.append(simplified)
    return variants


def timestamp_to_seconds(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = ts.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


# =============================================================================
# Subtitle Functions
# =============================================================================

def download_subtitles(video_id: str, output_dir: Path) -> Optional[Path]:
    """Download Chinese subtitles (prefer manual, fallback to auto)."""
    # Try manual subtitles first
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--write-subs",
        "--sub-lang", "zh,zh-TW,zh-CN,zh-Hans,zh-Hant,zh-HK",
        "--skip-download",
        "--no-warnings",
        "--quiet",
        "-o", str(output_dir / f"{video_id}"),
    ]
    run_command(cmd)

    # Check if downloaded
    for f in output_dir.glob(f"{video_id}*.vtt"):
        return f
    for f in output_dir.glob(f"{video_id}*.srt"):
        return f

    # Fallback to auto subtitles
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--write-auto-subs",
        "--sub-lang", "zh,zh-TW,zh-CN,zh-Hans,zh-Hant,zh-HK",
        "--skip-download",
        "--no-warnings",
        "--quiet",
        "-o", str(output_dir / f"{video_id}"),
    ]
    run_command(cmd)

    for f in output_dir.glob(f"{video_id}*.vtt"):
        return f
    for f in output_dir.glob(f"{video_id}*.srt"):
        return f
    return None


def parse_vtt(vtt_path: Path) -> list[tuple[float, float, str]]:
    """Parse VTT/SRT file into list of (start_sec, end_sec, text)."""
    entries = []
    try:
        for caption in webvtt.read(str(vtt_path)):
            start = timestamp_to_seconds(caption.start)
            end = timestamp_to_seconds(caption.end)
            text = re.sub(r'<[^>]+>', '', caption.text)
            text = text.replace('\n', ' ').strip()
            if text:
                entries.append((start, end, text))
    except Exception as e:
        log.error(f"Error parsing VTT {vtt_path}: {e}")
    return entries


def get_subtitle_type(sub_path: Path) -> str:
    """Determine if subtitle is manual or auto-generated."""
    name = sub_path.name.lower()
    if '.zh-hant.' in name or '.zh-tw.' in name:
        return 'manual'
    if '.zh-hans.' in name or '.zh-cn.' in name:
        return 'manual'
    if '.zh.' in name and 'auto' not in name:
        return 'manual'
    return 'auto'


# =============================================================================
# Channel Crawler (INDEX command)
# =============================================================================

def get_channel_videos(channel_url: str, limit: int = 500) -> list[dict]:
    """Get video info from a channel."""
    log.info(f"Fetching video list from {channel_url}...")
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end", str(limit),
        "--print", "%(id)s\t%(title)s\t%(duration)s",
        "--no-warnings",
        f"{channel_url}/videos"
    ]
    result = run_command(cmd)

    videos = []
    if result.returncode == 0 and result.stdout:
        for line in result.stdout.strip().split('\n'):
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2 and len(parts[0]) == 11:
                    videos.append({
                        'id': parts[0],
                        'title': parts[1],
                        'duration': float(parts[2]) if len(parts) > 2 and parts[2].replace('.','').isdigit() else 0
                    })

    log.info(f"Found {len(videos)} videos")
    return videos


def is_indexed(conn: sqlite3.Connection, video_id: str) -> bool:
    """Check if video is already indexed."""
    result = conn.execute(
        "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    return result is not None


def index_video(conn: sqlite3.Connection, video_id: str, channel: str,
                title: str, duration: float, temp_dir: Path) -> bool:
    """Download subtitles for a video and add to index."""
    sub_path = download_subtitles(video_id, temp_dir)
    if not sub_path:
        return False

    sub_type = get_subtitle_type(sub_path)
    captions = parse_vtt(sub_path)

    if not captions:
        sub_path.unlink(missing_ok=True)
        return False

    # Insert captions
    for start, end, text in captions:
        conn.execute(
            "INSERT INTO captions (video_id, channel, start_time, end_time, text) VALUES (?, ?, ?, ?, ?)",
            (video_id, channel, str(start), str(end), text)
        )

    # Insert video metadata
    conn.execute(
        "INSERT OR REPLACE INTO videos (video_id, channel, title, duration, subtitle_type, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
        (video_id, channel, title, duration, sub_type, datetime.now().isoformat())
    )

    sub_path.unlink(missing_ok=True)
    return True


def crawl_channel(conn: sqlite3.Connection, channel_url: str, channel_name: str,
                  temp_dir: Path, limit: int = 500) -> int:
    """Crawl a channel and index all videos with subtitles."""
    videos = get_channel_videos(channel_url, limit)

    indexed = 0
    skipped = 0
    failed = 0

    for i, video in enumerate(videos):
        video_id = video['id']

        if is_indexed(conn, video_id):
            skipped += 1
            continue

        log.info(f"[{i+1}/{len(videos)}] Indexing {video_id}: {video['title'][:40]}...")

        if index_video(conn, video_id, channel_name, video['title'], video['duration'], temp_dir):
            indexed += 1
            if indexed % 10 == 0:
                conn.commit()
                log.info(f"  Progress: {indexed} indexed, {skipped} skipped, {failed} no subs")
        else:
            failed += 1

    conn.commit()

    # Update channel record
    conn.execute("""
        INSERT OR REPLACE INTO channels (channel_id, channel_url, channel_name, last_crawled, video_count)
        VALUES (?, ?, ?, ?, ?)
    """, (channel_name, channel_url, channel_name, datetime.now().isoformat(), indexed))
    conn.commit()

    log.info(f"Channel complete: {indexed} indexed, {skipped} already indexed, {failed} no subtitles")
    return indexed


def cmd_index(args):
    """INDEX command: Crawl channels and build subtitle index."""
    db_path = Path(args.db)
    conn = init_database(db_path)

    # Read channels from file or use defaults
    if args.channels:
        channels_file = Path(args.channels)
        if channels_file.exists():
            with open(channels_file) as f:
                channels = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Format: URL or "URL NAME"
                        parts = line.split(None, 1)
                        url = parts[0]
                        name = parts[1] if len(parts) > 1 else url.split('@')[-1].split('/')[0]
                        channels.append((url, name))
        else:
            log.error(f"Channels file not found: {channels_file}")
            return
    else:
        # Default channels
        channels = [
            ("https://www.youtube.com/@TVBSNEWS01", "TVBS新聞"),
            ("https://www.youtube.com/@newsebc", "東森新聞"),
            ("https://www.youtube.com/@setmoney", "三立新聞"),
        ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        total_indexed = 0

        for channel_url, channel_name in channels:
            log.info(f"\n{'='*60}")
            log.info(f"Crawling: {channel_name}")
            log.info(f"{'='*60}")

            indexed = crawl_channel(conn, channel_url, channel_name, temp_path, limit=args.limit)
            total_indexed += indexed

    log.info(f"\nTotal indexed: {total_indexed} videos")
    conn.close()


# =============================================================================
# Query Engine (MINE command)
# =============================================================================

def find_word_in_index(conn: sqlite3.Connection, word: str, limit: int = 10) -> list[dict]:
    """Find clips containing word from index."""
    variants = get_word_variants(word)

    results = []
    seen_videos = set()

    for variant in variants:
        # Use LIKE query instead of FTS MATCH
        # FTS5 unicode61 doesn't tokenize Chinese characters properly
        like_pattern = f'%{variant}%'

        try:
            rows = conn.execute("""
                SELECT c.video_id, c.channel, c.start_time, c.end_time, c.text,
                       v.title, v.subtitle_type
                FROM captions c
                JOIN videos v ON c.video_id = v.video_id
                WHERE c.text LIKE ?
                ORDER BY
                    CASE WHEN v.subtitle_type = 'manual' THEN 0 ELSE 1 END,
                    c.start_time
                LIMIT ?
            """, (like_pattern, limit * 2)).fetchall()

            for row in rows:
                video_id = row[0]
                if video_id not in seen_videos:
                    seen_videos.add(video_id)
                    results.append({
                        'video_id': video_id,
                        'channel': row[1],
                        'start': float(row[2]),
                        'end': float(row[3]),
                        'context': row[4],
                        'title': row[5],
                        'subtitle_type': row[6],
                        'found_variant': variant
                    })
        except sqlite3.OperationalError as e:
            log.warning(f"FTS query error for '{variant}': {e}")

    return results[:limit]


def download_audio_slice(video_id: str, start: float, end: float, output_path: Path) -> bool:
    """Download only the specified time range of audio."""
    time_range = f"*{start:.1f}-{end:.1f}"

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--download-sections", time_range,
        "-o", str(output_path),
        "--no-warnings",
        "--quiet",
        "--force-keyframes-at-cuts",
    ]
    run_command(cmd)
    return output_path.exists()


def create_audio_cloze(audio_path: Path, word_start: float, word_end: float,
                       output_path: Path) -> bool:
    """Create a version of the audio with the target word silenced."""
    try:
        audio = AudioSegment.from_file(str(audio_path))

        # Convert to milliseconds
        start_ms = int(word_start * 1000)
        end_ms = int(word_end * 1000)

        # Add buffer for subtitle timing variance
        buffer_ms = 200
        start_ms = max(0, start_ms - buffer_ms)
        end_ms = min(len(audio), end_ms + buffer_ms)

        # Create silence
        silence_duration = end_ms - start_ms
        silence = AudioSegment.silent(duration=silence_duration)

        # Replace word with silence
        cloze_audio = audio[:start_ms] + silence + audio[end_ms:]
        cloze_audio.export(str(output_path), format="mp3")
        return True

    except Exception as e:
        log.error(f"Failed to create cloze audio: {e}")
        return False


@dataclass
class AudioClip:
    """A processed audio clip ready for review."""
    word: str
    video_id: str
    video_title: str
    transcript: str
    audio_full_path: Path
    audio_cloze_path: Path
    word_start: float
    word_end: float
    source_url: str


def create_clip_from_hit(word: str, hit: dict, output_dir: Path) -> Optional[AudioClip]:
    """Download audio slice and create cloze using subtitle timing."""
    video_id = hit['video_id']

    # Add context padding
    context_before = 8.0
    context_after = 8.0
    start = max(0, hit['start'] - context_before)
    end = hit['end'] + context_after

    # Download audio slice
    clip_path = output_dir / f"{word}_{video_id}_clip.mp3"
    log.info(f"  Downloading audio slice: {start:.1f}s - {end:.1f}s")

    if not download_audio_slice(video_id, start, end, clip_path):
        log.warning(f"  Failed to download audio")
        return None

    # Calculate relative word timing within the clip
    word_start_rel = hit['start'] - start
    word_end_rel = hit['end'] - start

    # Create cloze version (silence the word)
    cloze_path = output_dir / f"{word}_{video_id}_cloze.mp3"
    if not create_audio_cloze(clip_path, word_start_rel, word_end_rel, cloze_path):
        clip_path.unlink(missing_ok=True)
        return None

    log.info(f"  Created clip and cloze for '{word}'")

    return AudioClip(
        word=word,
        video_id=video_id,
        video_title=hit['title'],
        transcript=hit['context'],
        audio_full_path=clip_path,
        audio_cloze_path=cloze_path,
        word_start=word_start_rel,
        word_end=word_end_rel,
        source_url=f"https://www.youtube.com/watch?v={video_id}"
    )


def process_word_from_index(conn: sqlite3.Connection, word: str,
                           output_dir: Path, max_clips: int = 2) -> list[AudioClip]:
    """Process a word using the inverted index."""
    log.info(f"\n{'='*60}")
    log.info(f"MINING: {word}")
    log.info(f"{'='*60}")

    # Instant lookup
    start_time = time.time()
    hits = find_word_in_index(conn, word, limit=max_clips * 3)
    query_time = time.time() - start_time

    log.info(f"Found {len(hits)} hits in {query_time*1000:.1f}ms")

    if not hits:
        log.warning(f"Word '{word}' not found in index")
        return []

    clips = []
    for i, hit in enumerate(hits):
        if len(clips) >= max_clips:
            break

        log.info(f"\n--- Hit {i+1}/{len(hits)}: {hit['title'][:50]}...")
        log.info(f"    Channel: {hit['channel']}, Type: {hit['subtitle_type']}")
        log.info(f"    Context: {hit['context'][:60]}...")

        clip = create_clip_from_hit(word, hit, output_dir)
        if clip:
            clips.append(clip)

    log.info(f"\nWord '{word}' complete: {len(clips)} clips")
    return clips


def cmd_mine(args):
    """MINE command: Query index and create audio clips."""
    db_path = Path(args.db)

    if not db_path.exists():
        log.error(f"Database not found: {db_path}")
        log.error("Run 'index' command first to build the subtitle index")
        return

    conn = sqlite3.connect(str(db_path))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_clips = []

    for word in args.words:
        clips = process_word_from_index(conn, word, output_dir, max_clips=args.num_clips)
        all_clips.extend(clips)

    conn.close()

    if all_clips:
        html_path = generate_html_review(all_clips, output_dir)
        metadata_path = save_clips_metadata(all_clips, output_dir)

        print(f"\n{'='*60}")
        print(f"COMPLETE!")
        print(f"{'='*60}")
        print(f"Clips generated: {len(all_clips)}")
        print(f"HTML review: {html_path}")
        print(f"Metadata: {metadata_path}")
    else:
        print("\nNo clips generated. Try indexing more channels.")


# =============================================================================
# Stats Command
# =============================================================================

def cmd_stats(args):
    """STATS command: Show index statistics."""
    db_path = Path(args.db)

    if not db_path.exists():
        log.error(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))

    # Video count
    video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]

    # Caption count
    caption_count = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]

    # Channel stats
    channels = conn.execute("""
        SELECT channel_name, video_count, last_crawled
        FROM channels
        ORDER BY video_count DESC
    """).fetchall()

    # Subtitle type breakdown
    manual = conn.execute("SELECT COUNT(*) FROM videos WHERE subtitle_type = 'manual'").fetchone()[0]
    auto = conn.execute("SELECT COUNT(*) FROM videos WHERE subtitle_type = 'auto'").fetchone()[0]

    # Database size
    db_size = db_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"INDEX STATISTICS")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Size: {db_size:.1f} MB")
    print(f"\nVideos indexed: {video_count}")
    print(f"  - Manual subtitles: {manual}")
    print(f"  - Auto subtitles: {auto}")
    print(f"Caption entries: {caption_count}")

    if channels:
        print(f"\nChannels ({len(channels)}):")
        for name, count, last in channels:
            print(f"  - {name}: {count} videos (crawled: {last[:10] if last else 'never'})")

    # Sample search to test index
    if args.test_word:
        print(f"\n{'='*60}")
        print(f"TEST SEARCH: '{args.test_word}'")
        print(f"{'='*60}")

        hits = find_word_in_index(conn, args.test_word, limit=5)
        if hits:
            for i, hit in enumerate(hits):
                print(f"\n{i+1}. {hit['title'][:50]}...")
                print(f"   Time: {hit['start']:.1f}s - {hit['end']:.1f}s")
                print(f"   Context: {hit['context'][:60]}...")
        else:
            print(f"No results found for '{args.test_word}'")

    conn.close()


# =============================================================================
# HTML Review Generator
# =============================================================================

def generate_html_review(clips: list[AudioClip], output_dir: Path) -> Path:
    """Generate an HTML page for reviewing clips with audio players."""
    html_path = output_dir / "review.html"

    clips_json_data = []
    for clip in clips:
        clips_json_data.append({
            'id': f"{clip.word}_{clip.video_id}",
            'word': clip.word,
            'video_id': clip.video_id,
            'video_title': clip.video_title,
            'transcript': clip.transcript,
            'audio_full': clip.audio_full_path.name,
            'audio_cloze': clip.audio_cloze_path.name,
            'word_start': clip.word_start,
            'word_end': clip.word_end,
            'source_url': clip.source_url,
        })

    clips_json_str = json.dumps(clips_json_data, ensure_ascii=False)

    html_content = '''<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audio Vocabulary Review</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
    .stats { color: #666; margin-bottom: 10px; }
    .approved-summary {
      background: #e8f5e9;
      border: 1px solid #4caf50;
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 25px;
    }
    .approved-summary h3 { margin: 0 0 10px 0; color: #2e7d32; }
    .approved-summary .count { font-size: 1.2em; font-weight: bold; }
    .approved-summary button {
      background: #4caf50;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 10px;
      margin-top: 10px;
    }
    .approved-summary button:hover { background: #45a049; }
    .approved-list { margin-top: 10px; font-size: 0.9em; color: #555; }
    .word-section { margin-bottom: 40px; }
    .word-section h2 {
      font-size: 2em;
      color: #1a1a1a;
      margin-bottom: 15px;
      padding: 10px 15px;
      background: #fff;
      border-left: 4px solid #4a90d9;
    }
    .clip-card {
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 15px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border: 2px solid transparent;
      transition: border-color 0.2s;
    }
    .clip-card.approved { border-color: #4caf50; background: #f9fff9; }
    .clip-card.rejected { border-color: #f44336; background: #fff9f9; opacity: 0.6; }
    .approval-buttons { display: flex; gap: 10px; margin-bottom: 15px; }
    .approval-buttons button {
      padding: 8px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1em;
      transition: all 0.2s;
    }
    .btn-approve { background: #e8f5e9; color: #2e7d32; border: 1px solid #4caf50 !important; }
    .btn-approve:hover, .btn-approve.active { background: #4caf50; color: white; }
    .btn-reject { background: #ffebee; color: #c62828; border: 1px solid #f44336 !important; }
    .btn-reject:hover, .btn-reject.active { background: #f44336; color: white; }
    .audio-row { display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; }
    .audio-item { flex: 1; min-width: 200px; }
    .audio-item label { display: block; font-weight: 600; margin-bottom: 5px; color: #555; }
    .audio-item audio { width: 100%; }
    .transcript {
      background: #f9f9f9;
      padding: 15px;
      border-radius: 4px;
      line-height: 1.8;
      font-size: 1.1em;
    }
    .highlight { background: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .meta { margin-top: 10px; font-size: 0.9em; color: #666; }
    .meta a { color: #4a90d9; }
    .timing { color: #888; font-size: 0.85em; }
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: #333;
      color: white;
      padding: 12px 24px;
      border-radius: 4px;
      display: none;
    }
    .toast.show { display: block; }
  </style>
</head>
<body>
  <h1>Audio Vocabulary Review</h1>
  <p class="stats">Generated: ''' + str(len(clips)) + ''' clips</p>

  <div class="approved-summary">
    <h3>Approved Clips</h3>
    <p><span class="count" id="approved-count">0</span> / ''' + str(len(clips)) + ''' clips approved</p>
    <button onclick="exportApproved()">Copy Approved JSON</button>
    <button onclick="clearAll()">Clear All Selections</button>
    <div class="approved-list" id="approved-list"></div>
  </div>
'''

    current_word = None
    for clip in clips:
        clip_id = f"{clip.word}_{clip.video_id}"

        if clip.word != current_word:
            if current_word is not None:
                html_content += '  </div>\n'
            current_word = clip.word
            html_content += f'  <div class="word-section">\n    <h2>{clip.word}</h2>\n'

        # Highlight target word
        highlighted = clip.transcript
        for variant in get_word_variants(clip.word):
            highlighted = highlighted.replace(variant, f'<span class="highlight">{clip.word}</span>')

        html_content += f'''    <div class="clip-card" id="card-{clip_id}" data-clip-id="{clip_id}">
      <div class="approval-buttons">
        <button class="btn-approve" onclick="setApproval('{clip_id}', 'approved')">Approve</button>
        <button class="btn-reject" onclick="setApproval('{clip_id}', 'rejected')">Reject</button>
      </div>
      <div class="audio-row">
        <div class="audio-item">
          <label>Full Audio</label>
          <audio controls src="{clip.audio_full_path.name}"></audio>
        </div>
        <div class="audio-item">
          <label>Cloze Audio (word silenced)</label>
          <audio controls src="{clip.audio_cloze_path.name}"></audio>
        </div>
      </div>
      <p class="transcript">{highlighted}</p>
      <p class="meta">
        Source: <a href="{clip.source_url}" target="_blank">{clip.video_title[:60]}...</a>
        <span class="timing"> | Word at {clip.word_start:.2f}s - {clip.word_end:.2f}s</span>
      </p>
    </div>
'''

    if current_word is not None:
        html_content += '  </div>\n'

    html_content += '''
  <div class="toast" id="toast"></div>

  <script>
    const CLIPS_DATA = ''' + clips_json_str + ''';
    const STORAGE_KEY = 'audio_vocab_approvals';

    function getApprovals() {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    }

    function saveApprovals(approvals) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(approvals));
    }

    function setApproval(clipId, status) {
      const approvals = getApprovals();
      if (approvals[clipId] === status) {
        delete approvals[clipId];
      } else {
        approvals[clipId] = status;
      }
      saveApprovals(approvals);
      updateUI();
    }

    function updateUI() {
      const approvals = getApprovals();
      document.querySelectorAll('.clip-card').forEach(card => {
        const clipId = card.dataset.clipId;
        const status = approvals[clipId];
        card.classList.remove('approved', 'rejected');
        card.querySelectorAll('.btn-approve, .btn-reject').forEach(btn => btn.classList.remove('active'));
        if (status === 'approved') {
          card.classList.add('approved');
          card.querySelector('.btn-approve').classList.add('active');
        } else if (status === 'rejected') {
          card.classList.add('rejected');
          card.querySelector('.btn-reject').classList.add('active');
        }
      });
      const approvedClips = CLIPS_DATA.filter(c => approvals[c.id] === 'approved');
      document.getElementById('approved-count').textContent = approvedClips.length;
      const listEl = document.getElementById('approved-list');
      listEl.innerHTML = approvedClips.length > 0 ? 'Approved: ' + approvedClips.map(c => c.word).join(', ') : '';
    }

    function exportApproved() {
      const approvals = getApprovals();
      const approvedClips = CLIPS_DATA.filter(c => approvals[c.id] === 'approved');
      navigator.clipboard.writeText(JSON.stringify(approvedClips, null, 2)).then(() => {
        showToast('Copied ' + approvedClips.length + ' approved clips to clipboard');
      });
    }

    function clearAll() {
      localStorage.removeItem(STORAGE_KEY);
      updateUI();
      showToast('Cleared all selections');
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2000);
    }

    updateUI();
  </script>
</body>
</html>
'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path


def save_clips_metadata(clips: list[AudioClip], output_dir: Path) -> Path:
    """Save clip metadata as JSON."""
    metadata_path = output_dir / "clips.json"

    data = []
    for clip in clips:
        data.append({
            'word': clip.word,
            'video_id': clip.video_id,
            'video_title': clip.video_title,
            'transcript': clip.transcript,
            'audio_full': clip.audio_full_path.name,
            'audio_cloze': clip.audio_cloze_path.name,
            'word_start': clip.word_start,
            'word_end': clip.word_end,
            'source_url': clip.source_url,
        })

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return metadata_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with subcommands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio Vocabulary Miner - Index-first architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from default news channels
  python audio_vocab_miner.py index --db vocab.db

  # Build index from custom channels file
  python audio_vocab_miner.py index --db vocab.db --channels channels.txt

  # Mine words (instant query)
  python audio_vocab_miner.py mine 面交 吊飾 放鴿子 --db vocab.db -o audio_clips

  # Check index stats
  python audio_vocab_miner.py stats --db vocab.db --test 面交
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # INDEX command
    index_parser = subparsers.add_parser('index', help='Crawl channels and build subtitle index')
    index_parser.add_argument('--db', default='vocab.db', help='Database path')
    index_parser.add_argument('--channels', help='File with channel URLs (one per line)')
    index_parser.add_argument('--limit', type=int, default=500, help='Max videos per channel')

    # MINE command
    mine_parser = subparsers.add_parser('mine', help='Query index and create audio clips')
    mine_parser.add_argument('words', nargs='+', help='Chinese words to find audio for')
    mine_parser.add_argument('--db', default='vocab.db', help='Database path')
    mine_parser.add_argument('-o', '--output', default='./audio_clips', help='Output directory')
    mine_parser.add_argument('-n', '--num-clips', type=int, default=2, help='Clips per word')

    # STATS command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    stats_parser.add_argument('--db', default='vocab.db', help='Database path')
    stats_parser.add_argument('--test', dest='test_word', help='Test search for a word')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'mine':
        cmd_mine(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
