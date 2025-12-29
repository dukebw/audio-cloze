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

# RSS feed parsing
import feedparser
import hashlib
import html
import requests


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
    video_count INTEGER DEFAULT 0,
    source_type TEXT DEFAULT 'youtube'
);

-- Podcast episodes table
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    feed_url TEXT,
    channel_name TEXT,
    title TEXT,
    description TEXT,
    pub_date TEXT,
    audio_url TEXT,
    duration REAL,
    transcribed INTEGER DEFAULT 0,
    indexed_at TEXT
);

-- Episode show notes FTS for Phase 1 text search
CREATE VIRTUAL TABLE IF NOT EXISTS episode_notes USING fts5(
    episode_id,
    channel,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

-- WenetSpeech pre-transcribed segments
CREATE TABLE IF NOT EXISTS wenetspeech_segments (
    segment_id TEXT PRIMARY KEY,
    audio_path TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    speaker_id TEXT,
    confidence REAL,
    domain TEXT
);

-- WenetSpeech FTS index
CREATE VIRTUAL TABLE IF NOT EXISTS wenetspeech_fts USING fts5(
    segment_id,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

-- Podcast caption segments (from ASR Phase 2)
CREATE TABLE IF NOT EXISTS podcast_captions (
    caption_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    confidence REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

-- Podcast captions FTS index
CREATE VIRTUAL TABLE IF NOT EXISTS podcast_captions_fts USING fts5(
    caption_id,
    episode_id,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
"""

# Migration for existing databases
MIGRATIONS = """
-- Add source_type to channels if not exists
ALTER TABLE channels ADD COLUMN source_type TEXT DEFAULT 'youtube';
"""


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize database with schema and apply migrations."""
    conn = sqlite3.connect(str(db_path))

    # Create tables (IF NOT EXISTS handles existing tables)
    conn.executescript(SCHEMA)

    # Apply migrations for existing databases
    try:
        # Check if source_type column exists in channels
        cursor = conn.execute("PRAGMA table_info(channels)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'source_type' not in columns:
            conn.execute("ALTER TABLE channels ADD COLUMN source_type TEXT DEFAULT 'youtube'")
            log.info("Applied migration: added source_type to channels")
    except sqlite3.OperationalError:
        pass  # Table might not exist yet

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


# =============================================================================
# RSS Feed Indexer (Podcast Support)
# =============================================================================

def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text and decode entities."""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def generate_episode_id(feed_url: str, episode_guid: str) -> str:
    """Generate a stable episode ID from feed URL and GUID."""
    combined = f"{feed_url}:{episode_guid}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_audio_enclosure(entry: dict) -> Optional[str]:
    """Extract audio URL from RSS enclosure."""
    # Check enclosures first
    for enclosure in entry.get('enclosures', []):
        if enclosure.get('type', '').startswith('audio/'):
            return enclosure.get('href')
    # Fallback: check links
    for link in entry.get('links', []):
        if link.get('type', '').startswith('audio/'):
            return link.get('href')
    return None


def parse_duration(entry: dict) -> float:
    """Parse episode duration from iTunes duration tag or similar."""
    duration_str = entry.get('itunes_duration', '')
    if not duration_str:
        return 0.0

    # Handle formats: "HH:MM:SS", "MM:SS", or seconds
    parts = str(duration_str).split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(duration_str)
    except (ValueError, TypeError):
        return 0.0


def is_episode_indexed(conn: sqlite3.Connection, episode_id: str) -> bool:
    """Check if episode is already in database."""
    result = conn.execute(
        "SELECT 1 FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    return result is not None


def crawl_podcast_feed(conn: sqlite3.Connection, feed_url: str,
                       channel_name: str, limit: int = 100) -> int:
    """Crawl a podcast RSS feed and index episode show notes.

    Phase 1 indexing: Only indexes show notes text for fast searching.
    ASR transcription (Phase 2) happens on-demand during mining.
    """
    log.info(f"Fetching RSS feed: {feed_url}")

    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        log.error(f"Failed to parse feed: {e}")
        return 0

    if feed.bozo and not feed.entries:
        log.error(f"Invalid feed: {feed.bozo_exception}")
        return 0

    indexed = 0
    skipped = 0

    for entry in feed.entries[:limit]:
        # Generate stable episode ID
        episode_guid = entry.get('id', entry.get('link', entry.get('title', '')))
        episode_id = generate_episode_id(feed_url, episode_guid)

        if is_episode_indexed(conn, episode_id):
            skipped += 1
            continue

        # Extract episode metadata
        title = entry.get('title', 'Untitled')
        description = entry.get('description', entry.get('summary', ''))
        pub_date = entry.get('published', '')
        audio_url = get_audio_enclosure(entry)
        duration = parse_duration(entry)

        if not description:
            log.debug(f"  Skipping {title[:30]}... (no description)")
            continue

        # Clean show notes
        clean_description = strip_html_tags(description)

        # Check if description has Chinese characters (filter out English-only)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', clean_description))
        if not has_chinese:
            log.debug(f"  Skipping {title[:30]}... (no Chinese)")
            continue

        # Insert episode metadata
        conn.execute("""
            INSERT INTO episodes
            (episode_id, feed_url, channel_name, title, description,
             pub_date, audio_url, duration, transcribed, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (episode_id, feed_url, channel_name, title, clean_description,
              pub_date, audio_url, duration, datetime.now().isoformat()))

        # Index show notes in FTS table
        conn.execute("""
            INSERT INTO episode_notes (episode_id, channel, text)
            VALUES (?, ?, ?)
        """, (episode_id, channel_name, clean_description))

        indexed += 1
        if indexed % 10 == 0:
            log.info(f"  Indexed {indexed} episodes...")

    conn.commit()

    # Update channel record
    conn.execute("""
        INSERT OR REPLACE INTO channels
        (channel_id, channel_url, channel_name, last_crawled, video_count, source_type)
        VALUES (?, ?, ?, ?, ?, 'podcast')
    """, (channel_name, feed_url, channel_name, datetime.now().isoformat(), indexed))
    conn.commit()

    log.info(f"Podcast complete: {indexed} episodes indexed, {skipped} already indexed")
    return indexed


# =============================================================================
# WenetSpeech Corpus Indexer
# =============================================================================

def index_wenetspeech_corpus(conn: sqlite3.Connection, manifest_path: Path,
                              corpus_name: str, limit: int = 100000) -> int:
    """Index WenetSpeech corpus from manifest file.

    WenetSpeech provides pre-transcribed segments with verified timing.
    Manifest format expected: JSON with 'segments' array.
    """
    log.info(f"Indexing WenetSpeech corpus: {manifest_path}")

    if not manifest_path.exists():
        log.warning(f"WenetSpeech manifest not found: {manifest_path}")
        log.info("Skipping WenetSpeech indexing - download corpus first")
        return 0

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        log.error(f"Failed to load manifest: {e}")
        return 0

    segments = manifest.get('segments', manifest.get('data', []))
    if not segments:
        log.warning("No segments found in manifest")
        return 0

    indexed = 0

    for segment in segments[:limit]:
        segment_id = segment.get('id', segment.get('segment_id', str(indexed)))

        # Check if already indexed
        existing = conn.execute(
            "SELECT 1 FROM wenetspeech_segments WHERE segment_id = ?",
            (segment_id,)
        ).fetchone()
        if existing:
            continue

        audio_path = segment.get('audio_path', segment.get('audio', ''))
        text = segment.get('text', segment.get('transcript', ''))

        if not text:
            continue

        # Insert segment
        conn.execute("""
            INSERT INTO wenetspeech_segments
            (segment_id, audio_path, start_time, end_time, text,
             speaker_id, confidence, domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment_id,
            audio_path,
            segment.get('start', segment.get('start_time', 0)),
            segment.get('end', segment.get('end_time', 0)),
            text,
            segment.get('speaker', segment.get('speaker_id', '')),
            segment.get('confidence', 1.0),
            segment.get('domain', corpus_name)
        ))

        # Index in FTS
        conn.execute("""
            INSERT INTO wenetspeech_fts (segment_id, text)
            VALUES (?, ?)
        """, (segment_id, text))

        indexed += 1
        if indexed % 10000 == 0:
            conn.commit()
            log.info(f"  Indexed {indexed} segments...")

    conn.commit()
    log.info(f"WenetSpeech complete: {indexed} segments indexed")
    return indexed


# =============================================================================
# ASR Localizer (Whisper large-v3 for Podcast Timing)
# =============================================================================

# Lazy load whisper to avoid import overhead when not needed
_whisper_model = None

def get_whisper_model():
    """Get or load the Whisper large-v3 model (lazy loading)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            log.info("Loading Whisper large-v3 model (this may take a moment)...")
            _whisper_model = whisper.load_model("large-v3")
            log.info("Whisper model loaded successfully")
        except ImportError:
            log.error("Whisper not installed. Run: pip install openai-whisper")
            return None
    return _whisper_model


def download_podcast_audio(audio_url: str, output_path: Path,
                           start_time: float = None, end_time: float = None) -> bool:
    """Download podcast audio, optionally with time range.

    If start_time and end_time are provided, downloads only that segment.
    """
    try:
        if start_time is not None and end_time is not None:
            # Use yt-dlp for time-range downloads (works with many podcast CDNs)
            cmd = [
                "yt-dlp",
                "-x", "--audio-format", "mp3",
                "--download-sections", f"*{start_time}-{end_time}",
                "-o", str(output_path),
                audio_url
            ]
            result = run_command(cmd)
            if result.returncode == 0 and output_path.exists():
                return True

        # Fallback: download full file with requests
        log.info(f"  Downloading audio from {audio_url[:50]}...")
        response = requests.get(audio_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path.exists()

    except Exception as e:
        log.error(f"Failed to download audio: {e}")
        return False


def localize_word_with_asr(audio_path: Path, word: str) -> list[dict]:
    """Use Whisper large-v3 to find precise word timing in audio.

    Returns list of occurrences with:
    - start: start time in seconds
    - end: end time in seconds
    - text: surrounding segment text
    - confidence: word probability
    """
    model = get_whisper_model()
    if model is None:
        return []

    log.info(f"  Running Whisper ASR on {audio_path.name}...")

    try:
        result = model.transcribe(
            str(audio_path),
            language="zh",
            word_timestamps=True,
        )
    except Exception as e:
        log.error(f"Whisper transcription failed: {e}")
        return []

    # Get word variants (simplified/traditional)
    variants = get_word_variants(word)
    occurrences = []

    for segment in result.get('segments', []):
        words = segment.get('words', [])

        for i, word_info in enumerate(words):
            word_text = word_info.get('word', '').strip()

            # Check if any variant appears in this word or surrounding context
            found_variant = None
            for variant in variants:
                if variant in word_text:
                    found_variant = variant
                    break
                # Check surrounding words for multi-character matches
                context = ''.join(w.get('word', '') for w in words[max(0, i-2):i+3])
                if variant in context:
                    found_variant = variant
                    break

            if found_variant:
                occurrences.append({
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0),
                    'text': segment.get('text', ''),
                    'confidence': word_info.get('probability', 0.5),
                    'found_variant': found_variant,
                })

    log.info(f"  ASR found {len(occurrences)} occurrences of '{word}'")
    return occurrences


def smart_asr_localize(audio_url: str, word: str, duration: float,
                        temp_dir: Path) -> list[dict]:
    """Smart ASR: chunk long episodes to avoid full transcription.

    For short episodes (<15 min): transcribe full audio
    For long episodes: process in 5-minute chunks, stop when found
    """
    if duration <= 0:
        duration = 3600  # Assume 1 hour if unknown

    # Short episodes: transcribe full
    if duration < 900:  # 15 minutes
        audio_path = temp_dir / "podcast_full.mp3"
        if download_podcast_audio(audio_url, audio_path):
            return localize_word_with_asr(audio_path, word)
        return []

    # Long episodes: chunk-based search
    chunk_duration = 300  # 5-minute chunks
    all_occurrences = []

    for chunk_idx, chunk_start in enumerate(range(0, int(duration), chunk_duration)):
        chunk_end = min(chunk_start + chunk_duration, duration)
        chunk_path = temp_dir / f"chunk_{chunk_idx}.mp3"

        log.info(f"  Processing chunk {chunk_idx+1}: {chunk_start/60:.1f}-{chunk_end/60:.1f} min")

        # Try to download just the chunk
        if not download_podcast_audio(audio_url, chunk_path, chunk_start, chunk_end):
            continue

        occurrences = localize_word_with_asr(chunk_path, word)

        # Adjust timing to absolute position
        for occ in occurrences:
            occ['start'] += chunk_start
            occ['end'] += chunk_start
        all_occurrences.extend(occurrences)

        # Clean up chunk
        chunk_path.unlink(missing_ok=True)

        # Early exit if we have enough hits
        if len(all_occurrences) >= 3:
            log.info(f"  Found enough occurrences, stopping chunk processing")
            break

    return all_occurrences


# =============================================================================
# Sources File Parser
# =============================================================================

def parse_sources_file(path: Path) -> list[dict]:
    """Parse multi-source configuration file.

    Supports formats:
    - Legacy: URL NAME (assumes YouTube)
    - New: TYPE URL NAME (youtube/podcast/wenetspeech)

    Examples:
        youtube https://www.youtube.com/@shasha77 志祺七七
        podcast https://feeds.buzzsprout.com/1974862.rss 百靈果News
        wenetspeech /data/wenetspeech/manifest.json WenetSpeech
    """
    sources = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(None, 2)
            if len(parts) < 2:
                continue

            # Check if first part is a source type
            if parts[0] in ('youtube', 'podcast', 'wenetspeech'):
                sources.append({
                    'type': parts[0],
                    'url': parts[1],
                    'name': parts[2] if len(parts) > 2 else parts[1].split('/')[-1]
                })
            else:
                # Legacy format: URL NAME (assume YouTube)
                sources.append({
                    'type': 'youtube',
                    'url': parts[0],
                    'name': parts[1] if len(parts) > 1 else parts[0].split('@')[-1].split('/')[0]
                })

    return sources


def cmd_index(args):
    """INDEX command: Crawl sources and build subtitle index."""
    db_path = Path(args.db)
    conn = init_database(db_path)

    # Parse sources file (supports new multi-source format and legacy channels.txt)
    sources_file = Path(args.sources) if hasattr(args, 'sources') and args.sources else None
    channels_file = Path(args.channels) if args.channels else None

    sources = []
    if sources_file and sources_file.exists():
        sources = parse_sources_file(sources_file)
    elif channels_file and channels_file.exists():
        sources = parse_sources_file(channels_file)
    else:
        # Default YouTube channels
        sources = [
            {'type': 'youtube', 'url': "https://www.youtube.com/@TVBSNEWS01", 'name': "TVBS新聞"},
            {'type': 'youtube', 'url': "https://www.youtube.com/@newsebc", 'name': "東森新聞"},
            {'type': 'youtube', 'url': "https://www.youtube.com/@setmoney", 'name': "三立新聞"},
        ]

    # Filter by source type if specified
    source_type_filter = getattr(args, 'type', 'all')
    if source_type_filter != 'all':
        sources = [s for s in sources if s['type'] == source_type_filter]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        total_indexed = 0

        for source in sources:
            source_type = source['type']
            source_url = source['url']
            source_name = source['name']

            log.info(f"\n{'='*60}")
            log.info(f"[{source_type.upper()}] Crawling: {source_name}")
            log.info(f"{'='*60}")

            if source_type == 'youtube':
                indexed = crawl_channel(conn, source_url, source_name, temp_path, limit=args.limit)
                total_indexed += indexed
            elif source_type == 'podcast':
                indexed = crawl_podcast_feed(conn, source_url, source_name, limit=args.limit)
                total_indexed += indexed
            elif source_type == 'wenetspeech':
                indexed = index_wenetspeech_corpus(conn, Path(source_url), source_name, limit=args.limit)
                total_indexed += indexed
            else:
                log.warning(f"Unknown source type: {source_type}")

    log.info(f"\nTotal indexed: {total_indexed} items")
    conn.close()


# =============================================================================
# Query Engine (MINE command) - Multi-Source Search
# =============================================================================

def search_youtube_captions(conn: sqlite3.Connection, word: str, limit: int = 10) -> list[dict]:
    """Search YouTube captions (Phase 1: subtitle timing, high confidence)."""
    variants = get_word_variants(word)
    results = []
    seen_videos = set()

    for variant in variants:
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
                        'found_variant': variant,
                        'source_type': 'youtube',
                        'timing_confidence': 'subtitle',
                        'requires_asr': False,
                    })
        except sqlite3.OperationalError as e:
            log.warning(f"YouTube caption query error: {e}")

    return results[:limit]


def search_podcast_notes(conn: sqlite3.Connection, word: str, limit: int = 10) -> list[dict]:
    """Search podcast show notes (Phase 1: text hit, needs ASR for timing)."""
    variants = get_word_variants(word)
    results = []
    seen_episodes = set()

    for variant in variants:
        like_pattern = f'%{variant}%'
        try:
            rows = conn.execute("""
                SELECT en.episode_id, en.channel, en.text,
                       e.title, e.audio_url, e.duration
                FROM episode_notes en
                JOIN episodes e ON en.episode_id = e.episode_id
                WHERE en.text LIKE ?
                LIMIT ?
            """, (like_pattern, limit * 2)).fetchall()

            for row in rows:
                episode_id = row[0]
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    results.append({
                        'video_id': episode_id,  # Use video_id for compatibility
                        'episode_id': episode_id,
                        'channel': row[1],
                        'context': row[2][:200],  # Show notes snippet
                        'title': row[3],
                        'audio_url': row[4],
                        'duration': row[5] or 0,
                        'found_variant': variant,
                        'source_type': 'podcast',
                        'timing_confidence': 'text_only',
                        'requires_asr': True,
                        # Timing will be filled in by ASR
                        'start': None,
                        'end': None,
                    })
        except sqlite3.OperationalError as e:
            log.warning(f"Podcast notes query error: {e}")

    return results[:limit]


def search_wenetspeech(conn: sqlite3.Connection, word: str, limit: int = 10) -> list[dict]:
    """Search WenetSpeech corpus (pre-transcribed, high confidence timing)."""
    variants = get_word_variants(word)
    results = []
    seen_segments = set()

    for variant in variants:
        like_pattern = f'%{variant}%'
        try:
            rows = conn.execute("""
                SELECT ws.segment_id, ws.audio_path, ws.start_time, ws.end_time,
                       ws.text, ws.speaker_id, ws.confidence, ws.domain
                FROM wenetspeech_segments ws
                WHERE ws.text LIKE ?
                ORDER BY ws.confidence DESC
                LIMIT ?
            """, (like_pattern, limit * 2)).fetchall()

            for row in rows:
                segment_id = row[0]
                if segment_id not in seen_segments:
                    seen_segments.add(segment_id)
                    results.append({
                        'video_id': segment_id,  # Use video_id for compatibility
                        'segment_id': segment_id,
                        'audio_path': row[1],
                        'start': float(row[2]),
                        'end': float(row[3]),
                        'context': row[4],
                        'channel': row[7] or 'WenetSpeech',  # domain as channel
                        'title': f"WenetSpeech: {row[4][:30]}...",
                        'confidence': row[6],
                        'found_variant': variant,
                        'source_type': 'wenetspeech',
                        'timing_confidence': 'asr_verified',
                        'requires_asr': False,
                    })
        except sqlite3.OperationalError as e:
            log.warning(f"WenetSpeech query error: {e}")

    return results[:limit]


def find_word_in_index(conn: sqlite3.Connection, word: str, limit: int = 10,
                       sources: list[str] = None) -> list[dict]:
    """Unified search across all source types.

    Args:
        conn: Database connection
        word: Word to search for
        limit: Maximum results to return
        sources: List of source types to search ('youtube', 'podcast', 'wenetspeech', 'all')

    Returns:
        List of hits sorted by timing confidence
    """
    if sources is None:
        sources = ['all']
    if 'all' in sources:
        sources = ['youtube', 'podcast', 'wenetspeech']

    results = []

    # Search each source type
    if 'youtube' in sources:
        youtube_results = search_youtube_captions(conn, word, limit)
        results.extend(youtube_results)
        if youtube_results:
            log.info(f"  YouTube: {len(youtube_results)} hits")

    if 'podcast' in sources:
        podcast_results = search_podcast_notes(conn, word, limit)
        results.extend(podcast_results)
        if podcast_results:
            log.info(f"  Podcast: {len(podcast_results)} hits (need ASR for timing)")

    if 'wenetspeech' in sources:
        wenetspeech_results = search_wenetspeech(conn, word, limit)
        results.extend(wenetspeech_results)
        if wenetspeech_results:
            log.info(f"  WenetSpeech: {len(wenetspeech_results)} hits")

    # Sort by timing confidence (prefer sources with verified timing)
    confidence_order = {
        'subtitle': 0,        # YouTube subtitles
        'asr_verified': 1,    # WenetSpeech pre-transcribed
        'asr_estimated': 2,   # Podcast with ASR done
        'text_only': 3,       # Podcast show notes only
    }
    results.sort(key=lambda x: confidence_order.get(x.get('timing_confidence', 'text_only'), 99))

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
    video_id: str  # Can be video_id (YouTube) or episode_id (podcast) or segment_id (WenetSpeech)
    video_title: str
    transcript: str
    audio_full_path: Path
    audio_cloze_path: Path
    word_start: float
    word_end: float
    source_url: str
    # Multi-source fields
    source_type: str = 'youtube'  # 'youtube', 'podcast', 'wenetspeech'
    timing_confidence: str = 'subtitle'  # 'subtitle', 'asr_verified', 'text_only'


def create_clip_from_hit(word: str, hit: dict, output_dir: Path,
                         temp_dir: Path = None) -> Optional[AudioClip]:
    """Create audio clip from a search hit (supports multiple source types)."""
    source_type = hit.get('source_type', 'youtube')
    video_id = hit['video_id']

    # Handle different source types
    if source_type == 'youtube':
        return _create_youtube_clip(word, hit, output_dir)
    elif source_type == 'podcast':
        return _create_podcast_clip(word, hit, output_dir, temp_dir)
    elif source_type == 'wenetspeech':
        return _create_wenetspeech_clip(word, hit, output_dir)
    else:
        log.warning(f"Unknown source type: {source_type}")
        return None


def _create_youtube_clip(word: str, hit: dict, output_dir: Path) -> Optional[AudioClip]:
    """Create clip from YouTube source (existing logic)."""
    video_id = hit['video_id']

    # Add context padding
    context_before = 8.0
    context_after = 8.0
    start = max(0, hit['start'] - context_before)
    end = hit['end'] + context_after

    # Download audio slice
    clip_path = output_dir / f"{word}_{video_id}_clip.mp3"
    log.info(f"  Downloading YouTube audio slice: {start:.1f}s - {end:.1f}s")

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
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        source_type='youtube',
        timing_confidence='subtitle'
    )


def _create_podcast_clip(word: str, hit: dict, output_dir: Path,
                         temp_dir: Path = None) -> Optional[AudioClip]:
    """Create clip from podcast source (requires ASR for timing)."""
    episode_id = hit['episode_id']
    audio_url = hit.get('audio_url')
    duration = hit.get('duration', 0)

    if not audio_url:
        log.warning(f"  No audio URL for episode {episode_id}")
        return None

    log.info(f"  Processing podcast episode: {hit['title'][:40]}...")

    # Use temp directory for ASR processing
    if temp_dir is None:
        temp_dir = output_dir

    # Run ASR to find precise timing
    log.info(f"  Running ASR to locate '{word}' in audio...")
    occurrences = smart_asr_localize(audio_url, word, duration, temp_dir)

    if not occurrences:
        log.warning(f"  Word '{word}' not found in podcast audio (ASR)")
        return None

    # Use the first (best confidence) occurrence
    occ = occurrences[0]
    log.info(f"  Found '{word}' at {occ['start']:.1f}s - {occ['end']:.1f}s")

    # Add context padding
    context_before = 8.0
    context_after = 8.0
    start = max(0, occ['start'] - context_before)
    end = occ['end'] + context_after

    # Download the clip segment
    clip_path = output_dir / f"{word}_{episode_id[:8]}_clip.mp3"
    log.info(f"  Downloading podcast clip: {start:.1f}s - {end:.1f}s")

    if not download_podcast_audio(audio_url, clip_path, start, end):
        log.warning(f"  Failed to download podcast clip")
        return None

    # Calculate relative word timing within the clip
    word_start_rel = occ['start'] - start
    word_end_rel = occ['end'] - start

    # Create cloze version
    cloze_path = output_dir / f"{word}_{episode_id[:8]}_cloze.mp3"
    if not create_audio_cloze(clip_path, word_start_rel, word_end_rel, cloze_path):
        clip_path.unlink(missing_ok=True)
        return None

    log.info(f"  Created podcast clip and cloze for '{word}'")

    return AudioClip(
        word=word,
        video_id=episode_id,
        video_title=hit['title'],
        transcript=occ.get('text', hit['context']),
        audio_full_path=clip_path,
        audio_cloze_path=cloze_path,
        word_start=word_start_rel,
        word_end=word_end_rel,
        source_url=audio_url,
        source_type='podcast',
        timing_confidence='asr_verified'
    )


def _create_wenetspeech_clip(word: str, hit: dict, output_dir: Path) -> Optional[AudioClip]:
    """Create clip from WenetSpeech corpus (pre-transcribed)."""
    segment_id = hit['segment_id']
    audio_path = hit.get('audio_path')

    if not audio_path or not Path(audio_path).exists():
        log.warning(f"  WenetSpeech audio not found: {audio_path}")
        return None

    log.info(f"  Using WenetSpeech segment: {segment_id}")

    # WenetSpeech already has precise timing
    start = hit['start']
    end = hit['end']

    # Add context padding if audio file supports it
    context_before = 2.0  # Less padding for pre-segmented clips
    context_after = 2.0

    # Copy the relevant segment
    clip_path = output_dir / f"{word}_{segment_id[:8]}_clip.mp3"
    try:
        audio = AudioSegment.from_file(audio_path)
        # For WenetSpeech, the timing is relative to the segment file
        # So we use the segment directly with minimal padding
        start_ms = max(0, int(start * 1000) - int(context_before * 1000))
        end_ms = min(len(audio), int(end * 1000) + int(context_after * 1000))
        clip_audio = audio[start_ms:end_ms]
        clip_audio.export(str(clip_path), format="mp3")
    except Exception as e:
        log.warning(f"  Failed to extract WenetSpeech clip: {e}")
        return None

    # Calculate relative word timing within the clip
    word_start_rel = context_before
    word_end_rel = context_before + (end - start)

    # Create cloze version
    cloze_path = output_dir / f"{word}_{segment_id[:8]}_cloze.mp3"
    if not create_audio_cloze(clip_path, word_start_rel, word_end_rel, cloze_path):
        clip_path.unlink(missing_ok=True)
        return None

    log.info(f"  Created WenetSpeech clip and cloze for '{word}'")

    return AudioClip(
        word=word,
        video_id=segment_id,
        video_title=hit['title'],
        transcript=hit['context'],
        audio_full_path=clip_path,
        audio_cloze_path=cloze_path,
        word_start=word_start_rel,
        word_end=word_end_rel,
        source_url=f"file://{audio_path}",
        source_type='wenetspeech',
        timing_confidence='asr_verified'
    )


def process_word_from_index(conn: sqlite3.Connection, word: str,
                           output_dir: Path, max_clips: int = 2,
                           sources: list[str] = None) -> list[AudioClip]:
    """Process a word using the unified multi-source index."""
    log.info(f"\n{'='*60}")
    log.info(f"MINING: {word}")
    log.info(f"{'='*60}")

    # Unified search across all sources
    start_time = time.time()
    hits = find_word_in_index(conn, word, limit=max_clips * 3, sources=sources)
    query_time = time.time() - start_time

    log.info(f"Found {len(hits)} hits in {query_time*1000:.1f}ms")

    if not hits:
        log.warning(f"Word '{word}' not found in index")
        return []

    clips = []

    # Use temp directory for podcast ASR processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, hit in enumerate(hits):
            if len(clips) >= max_clips:
                break

            source_type = hit.get('source_type', 'youtube')
            timing_conf = hit.get('timing_confidence', 'unknown')

            log.info(f"\n--- Hit {i+1}/{len(hits)}: {hit['title'][:50]}...")
            log.info(f"    Source: {source_type}, Timing: {timing_conf}")
            log.info(f"    Channel: {hit['channel']}")
            log.info(f"    Context: {hit['context'][:60]}...")

            clip = create_clip_from_hit(word, hit, output_dir, temp_path)
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
    """STATS command: Show multi-source index statistics."""
    db_path = Path(args.db)

    if not db_path.exists():
        log.error(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))

    # YouTube stats
    video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    caption_count = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    manual = conn.execute("SELECT COUNT(*) FROM videos WHERE subtitle_type = 'manual'").fetchone()[0]
    auto = conn.execute("SELECT COUNT(*) FROM videos WHERE subtitle_type = 'auto'").fetchone()[0]

    # Podcast stats
    try:
        episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        episode_notes_count = conn.execute("SELECT COUNT(*) FROM episode_notes").fetchone()[0]
    except sqlite3.OperationalError:
        episode_count = 0
        episode_notes_count = 0

    # WenetSpeech stats
    try:
        wenetspeech_count = conn.execute("SELECT COUNT(*) FROM wenetspeech_segments").fetchone()[0]
    except sqlite3.OperationalError:
        wenetspeech_count = 0

    # Channel stats with source type
    try:
        channels = conn.execute("""
            SELECT channel_name, video_count, last_crawled, source_type
            FROM channels
            ORDER BY source_type, video_count DESC
        """).fetchall()
    except sqlite3.OperationalError:
        channels = conn.execute("""
            SELECT channel_name, video_count, last_crawled
            FROM channels
            ORDER BY video_count DESC
        """).fetchall()
        channels = [(c[0], c[1], c[2], 'youtube') for c in channels]

    # Database size
    db_size = db_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"MULTI-SOURCE INDEX STATISTICS")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Size: {db_size:.1f} MB")

    # YouTube section
    print(f"\n--- YOUTUBE ---")
    print(f"Videos indexed: {video_count}")
    print(f"  - Manual subtitles: {manual}")
    print(f"  - Auto subtitles: {auto}")
    print(f"Caption entries: {caption_count}")

    # Podcast section
    if episode_count > 0:
        print(f"\n--- PODCASTS ---")
        print(f"Episodes indexed: {episode_count}")
        print(f"Show notes entries: {episode_notes_count}")

    # WenetSpeech section
    if wenetspeech_count > 0:
        print(f"\n--- WENETSPEECH ---")
        print(f"Segments indexed: {wenetspeech_count}")

    # Sources by type
    if channels:
        youtube_channels = [c for c in channels if c[3] == 'youtube']
        podcast_feeds = [c for c in channels if c[3] == 'podcast']

        if youtube_channels:
            print(f"\nYouTube Channels ({len(youtube_channels)}):")
            for name, count, last, _ in youtube_channels:
                print(f"  - {name}: {count} videos")

        if podcast_feeds:
            print(f"\nPodcast Feeds ({len(podcast_feeds)}):")
            for name, count, last, _ in podcast_feeds:
                print(f"  - {name}: {count} episodes")

    # Sample search to test index
    if args.test_word:
        print(f"\n{'='*60}")
        print(f"TEST SEARCH: '{args.test_word}'")
        print(f"{'='*60}")

        hits = find_word_in_index(conn, args.test_word, limit=5)
        if hits:
            for i, hit in enumerate(hits):
                source = hit.get('source_type', 'youtube')
                timing = hit.get('timing_confidence', 'unknown')
                print(f"\n{i+1}. [{source.upper()}] {hit['title'][:45]}...")
                if hit.get('start') is not None:
                    print(f"   Time: {hit['start']:.1f}s - {hit['end']:.1f}s ({timing})")
                else:
                    print(f"   Timing: needs ASR")
                print(f"   Context: {hit['context'][:55]}...")
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
            'source_type': clip.source_type,
            'timing_confidence': clip.timing_confidence,
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
    index_parser = subparsers.add_parser('index', help='Crawl sources and build index')
    index_parser.add_argument('--db', default='vocab.db', help='Database path')
    index_parser.add_argument('--sources', help='Sources file (supports youtube/podcast/wenetspeech)')
    index_parser.add_argument('--channels', help='Legacy: channel URLs file (assumes YouTube)')
    index_parser.add_argument('--type', choices=['all', 'youtube', 'podcast', 'wenetspeech'],
                             default='all', help='Only index specific source type')
    index_parser.add_argument('--limit', type=int, default=500, help='Max items per source')

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
