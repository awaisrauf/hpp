"""
Subtitle utilities for REPL environment.

Loading & normalization:
  load_subtitles(path)                       - parse SRT or JSON, return normalized list

REPL functions (all take subtitle list as first arg):
  get_subtitles(subs, start, end)            - subtitles within a time range
  search_subtitles(subs, pattern, start, end)- regex search, optional time constraint
  find_times(subs, pattern)                  - timestamps where pattern matches
  summarize_subtitles(subs, max_lines)       - condensed overview for long subtitle lists

Normalized format: [{"start": 1.5, "end": 3.2, "text": "Hello"}, ...]
All times are float seconds, matching hierarchy shot/scene timestamps.
"""

import os
import re
import json


# ── Timestamp conversion ────────────────────────────────────────────────────

def _ts_to_seconds(ts):
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' or 'HH:MM:SS' to float seconds."""
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_srt(path):
    """Parse SRT file into normalized subtitle list."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    subs = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue

        text = ' '.join(lines[2:])
        text = re.sub(r'<[^>]+>', '', text).strip()
        if not text or text in ('♪', '♪ ♪'):
            continue

        subs.append({
            "start": round(_ts_to_seconds(time_match.group(1)), 2),
            "end": round(_ts_to_seconds(time_match.group(2)), 2),
            "text": text,
        })

    return subs


def _parse_json(path):
    """Parse JSON subtitle file into normalized subtitle list."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    subs = []
    for entry in data:
        text = entry.get('line', entry.get('text', '')).strip()
        if not text or text in ('[Music]', '[Applause]'):
            continue

        start = entry.get('start', '0')
        end = entry.get('end', start)

        # Handle both string timestamps and numeric values
        if isinstance(start, str):
            start = _ts_to_seconds(start)
        if isinstance(end, str):
            end = _ts_to_seconds(end)

        subs.append({
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "text": text,
        })

    return subs


def load_subtitles(path):
    """Load subtitles from SRT or JSON file. Auto-detects format.

    Returns: list of {"start": float, "end": float, "text": str}
             Empty list if file not found or parse fails.
    """
    if not path or not os.path.exists(path):
        return []

    try:
        if path.endswith('.srt'):
            return _parse_srt(path)
        elif path.endswith('.json'):
            return _parse_json(path)
        else:
            # Try SRT first, fall back to JSON
            try:
                return _parse_srt(path)
            except Exception:
                return _parse_json(path)
    except Exception:
        return []


# ── REPL functions ───────────────────────────────────────────────────────────

def get_subtitles(subs, start=None, end=None):
    """Get subtitles within a time range.

    Args:
        subs: subtitle list (from load_subtitles)
        start: start time in seconds (None = beginning)
        end: end time in seconds (None = end of video)

    Returns: list of {"start", "end", "text"} dicts
    """
    if start is None and end is None:
        return subs

    results = []
    for s in subs:
        if start is not None and s["end"] < start:
            continue
        if end is not None and s["start"] > end:
            continue
        results.append(s)

    print(f"{len(results)} subtitles between {start}s - {end}s")
    return results


def search_subtitles(subs, pattern, start=None, end=None):
    """Search subtitles with a regex pattern, optionally within a time range.

    Args:
        subs: subtitle list
        pattern: regex pattern (e.g. r"fire|explo", r"\\bdog\\b")
        start: optional start time in seconds
        end: optional end time in seconds

    Returns: list of matching {"start", "end", "text"} dicts
    """
    try:
        regex = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        print(f"Invalid regex: {e}")
        return []

    # Filter by time range first
    pool = get_subtitles(subs, start, end) if (start is not None or end is not None) else subs

    results = [s for s in pool if regex.search(s["text"])]

    print(f"'{pattern}' matched {len(results)} subtitles"
          + (f" between {start}s-{end}s" if start is not None or end is not None else ""))
    for r in results[:10]:
        print(f"  [{r['start']:.1f}s] {r['text']}")
    if len(results) > 10:
        print(f"  ... {len(results) - 10} more")
    return results


def find_times(subs, pattern):
    """Find timestamps where a regex pattern matches in subtitles.

    Args:
        subs: subtitle list
        pattern: regex pattern

    Returns: list of {"time": float, "text": str} — one per match
    """
    try:
        regex = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        print(f"Invalid regex: {e}")
        return []

    results = []
    for s in subs:
        if regex.search(s["text"]):
            results.append({"time": s["start"], "text": s["text"]})

    print(f"'{pattern}' found at {len(results)} timestamps:")
    for r in results[:10]:
        print(f"  {r['time']:.1f}s — {r['text']}")
    if len(results) > 10:
        print(f"  ... {len(results) - 10} more")
    return results


def summarize_subtitles(subs, chunk_seconds=60, max_lines=50):
    """Condensed overview of subtitles grouped by time chunks.

    Groups subtitles into time windows and merges text per window.
    Useful when full subtitle list is too long for the LLM to scan.

    Args:
        subs: subtitle list
        chunk_seconds: group subtitles into chunks of this many seconds
        max_lines: maximum number of chunks to return

    Returns: str — printable summary
    """
    if not subs:
        return "No subtitles available."

    total_dur = subs[-1]["end"] - subs[0]["start"]
    chunks = []
    chunk_start = subs[0]["start"]

    current_texts = []
    for s in subs:
        if s["start"] >= chunk_start + chunk_seconds and current_texts:
            chunks.append((chunk_start, ' '.join(current_texts)))
            chunk_start = s["start"]
            current_texts = []
        current_texts.append(s["text"])

    if current_texts:
        chunks.append((chunk_start, ' '.join(current_texts)))

    # Trim if too many
    if len(chunks) > max_lines:
        step = len(chunks) / max_lines
        chunks = [chunks[int(i * step)] for i in range(max_lines)]

    lines = [f"Subtitles summary ({len(subs)} entries, {total_dur:.0f}s):"]
    for t, text in chunks:
        # Truncate long chunks
        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"  [{t:.0f}s] {text}")

    summary = '\n'.join(lines)
    print(summary)
    return summary
