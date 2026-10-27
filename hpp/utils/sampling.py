"""
Frame sampling over the video hierarchy. Frames are written as JPEGs and their paths
returned, ready to pass to vlm_query().

  - get_keyframes(items)                    — keyframe paths from shot/scene/cluster dicts
  - get_video_overview(n)                   — n keyframes spread over all shots
  - get_frame_at(timestamp)                 — single frame at exact time
  - sample_uniform_frames(start, end, n)    — uniformly spaced frames
  - zoom(target, context, n)                — dense frames around a timestamp, shot, or scene
"""

import bisect
import os
import subprocess
import tempfile

from hpp.utils.hierarchy import get_shots, get_scenes


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_video_duration(video_path):
    """Get video duration in seconds using ffmpeg."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def _extract_frames_at_times(video_path, timestamps, output_dir, prefix="frame"):
    """Extract frames at specific timestamps using ffmpeg.

    Args:
        video_path: path to video file
        timestamps: list of seconds
        output_dir: directory to save frames
        prefix: filename prefix

    Returns:
        list[str] — paths to saved frames
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for ts in timestamps:
        out_path = os.path.join(output_dir, f"{prefix}_{ts:.2f}s.jpg")
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", str(ts), "-i", video_path,
            "-frames:v", "1", out_path
        ], check=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            paths.append(out_path)

    return paths


# ── Public API ───────────────────────────────────────────────────────────

def find_shot_or_scene_at_time(shots, time):
    """Binary search for the shot containing a timestamp."""
    if not shots:
        return None
    starts = [s["start"] for s in shots]
    idx = bisect.bisect_right(starts, time) - 1
    idx = max(0, min(idx, len(shots) - 1))
    return shots[idx]

def get_keyframes(items):
    """Return keyframe paths from one or more shot/scene/cluster dicts.

    Args:
        items: a single dict or list of dicts, each having a "keyframe" key.

    Returns: list[str] — keyframe file paths
    """
    if isinstance(items, dict):
        items = [items]
    paths = [it["keyframe"] for it in items if "keyframe" in it]
    print(f"get_keyframes() → {len(paths)} keyframes")
    return paths

def get_video_overview(video_path, n=10, output_dir=None):
    """Return n uniformly sampled shot keyframes as a video overview.

    Retrieves precomputed shot keyframes from the video hierarchy and picks
    n of them evenly spaced across the full shot list.

    Args:
        video_path: path to video file
        n: number of keyframes to return (default 10)
        output_dir: passed to get_shots for keyframe extraction

    Returns: list[str] — keyframe file paths
    """
    shots = get_shots(video_path, output_dir)
    paths = sample_uniform_frames_from_shots(shots, n)
    print(f"get_video_overview(n={n}) → {len(paths)} keyframes from {len(shots)} shots")
    return paths


def sample_uniform_frames_from_shots(shots, n):
    """
    Uniformly sample n frames from shots.
    Each shot is treated as 1 unit.
    Assumes each shot has a representative frame at shot["key_frame"].
    
    Args:
        shots (list): list of shot objects
        n (int): number of frames to sample
        
    Returns:
        list: sampled key frames
    """
    total = len(shots)
    if total == 0:
        return []
    
    if n >= total:
        return [shot["keyframe"] for shot in shots]
    
    step = total / n
    indices = [int(i * step) for i in range(n)]
    
    return [shots[i]["keyframe"] for i in indices]

def get_frame_at(video_path, timestamp, output_dir=None):
    """Extract a single frame at an exact timestamp.

    Args:
        video_path: path to video file
        timestamp: time in seconds

    Returns: str — path to saved frame
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hpp_frames_")

    paths = _extract_frames_at_times(video_path, [timestamp], output_dir,
                                     prefix="frame")
    if not paths:
        raise ValueError(f"Failed to extract frame at {timestamp}s")

    print(f"Frame at {timestamp:.1f}s → {paths[0]}")
    return paths[0]


def sample_uniform_frames(video_path, start=None, end=None, n=5, output_dir=None):
    """Return uniformly sampled frames within a timeframe.

    Args:
        video_path: path to video file
        start: start time in seconds (default: 0)
        end: end time in seconds (default: video duration)
        n: number of frames to sample (default 5)
        output_dir: where to save frames

    Returns: list[str] — frame paths
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hpp_frames_")

    duration = _get_video_duration(video_path)
    if start is None:
        start = 0.0
    if end is None:
        end = duration

    start = max(0.0, start)
    end = min(duration, end)
    if start >= end:
        raise ValueError(f"Invalid range: {start:.1f}s to {end:.1f}s")

    if n <= 1:
        timestamps = [(start + end) / 2]
    else:
        step = (end - start) / (n - 1)
        timestamps = [start + i * step for i in range(n)]

    paths = _extract_frames_at_times(video_path, timestamps, output_dir,
                                     prefix="uniform")

    print(f"sample_uniform_frames({start:.1f}s - {end:.1f}s, n={n}) → {len(paths)} frames")
    return paths

def zoom(video_path, target, context=0, n=5, output_dir=None):
    """Dense frames around a target — timestamp, shot dict, or scene dict.

    Args:
        video_path: path to video file
        target: what to zoom into:
            - float/int: timestamp in seconds → finds containing shot
            - dict with "shot_id": zooms into that shot (± context shots)
            - dict with "scene_id": zooms into that scene's full time range
        context: neighboring shots to include on each side (default 0)
        n: max frames to return (default 5)
        output_dir: where to save frames

    Returns: list[str] — frame paths
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hpp_frames_")

    n = int(n)
    context = int(context)  # cast to int — float context causes list index crash

    shots = get_shots(video_path, output_dir)

    if isinstance(target, (int, float)) and not isinstance(target, bool):
        # Timestamp → find containing shot
        shot = find_shot_or_scene_at_time(shots, float(target))
        center_id = shot["shot_id"] if shot else 0
    elif isinstance(target, dict):
        if "shot_id" in target:
            center_id = target["shot_id"]
        elif "scene_id" in target:
            # Scene → extract frames from its full time range
            scenes = get_scenes(video_path, output_dir)
            scene = next((s for s in scenes if s["scene_id"] == target["scene_id"]), None)
            if scene is None:
                raise ValueError(f"Scene {target['scene_id']} not found")
            paths = sample_uniform_frames(video_path, scene["start"], scene["end"],
                                          n=n, output_dir=output_dir)
            print(f"zoom(scene {target['scene_id']}, "
                  f"{scene['start']:.1f}s-{scene['end']:.1f}s) → {len(paths)} frames")
            return paths
        else:
            raise ValueError(f"Target dict must have 'shot_id' or 'scene_id', "
                             f"got: {list(target.keys())}")
    else:
        raise ValueError(f"Target must be a timestamp (float/int) or dict with "
                         f"'shot_id'/'scene_id'. Got: {type(target).__name__}")

    # Build shot range: center ± context (both are ints now)
    first = max(0, center_id - context)
    last = min(len(shots) - 1, center_id + context)

    start = shots[first]["start"]
    end = shots[last]["end"]

    paths = sample_uniform_frames(video_path, start, end, n=n, output_dir=output_dir)
    print(f"zoom(shot {center_id}, ±{context}) → {len(paths)} frames | "
          f"{start:.1f}s-{end:.1f}s")
    return paths
