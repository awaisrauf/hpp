"""
Codec-based hierarchical video structuring: shots -> scenes -> clusters.

  - get_shots()       — I-frame shot boundaries with action scores + merging
  - get_scenes()      — Temporal/spatial merge via RADIO (≤ half of shots)
  - get_clusters()    — Global semantic clustering (≤ 1/10 of scenes)
  - get_clip(start, end)              — clip between timestamps

Cache: only shot, scene, and cluster dicts are stored.  When get_scenes()
is called both scenes and clusters are computed and embeddings are freed.

Speed modes for get_shots():
  action_mode="packet" (default) — full packet scan via ffprobe, accurate action scores
  action_mode="keyframe"         — keyframe-only scan, action from shot duration (3-5x faster)
"""

import json
import os
import subprocess
import tempfile
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from decord import VideoReader, cpu
from sklearn.cluster import AgglomerativeClustering


# ── Config ───────────────────────────────────────────────────────────────

_DECODE_WIDTH = 320       # decord decode resolution (smaller = faster)
_DECODE_HEIGHT = 240
_JPEG_QUALITY = 75        # lower = faster saves, smaller files
_SAVE_WORKERS = 8         # parallel JPEG save threads

# ── Cache ────────────────────────────────────────────────────────────────

_radio_model = None
_cache = {}  # video_path -> {shots, scenes, clusters, output_dir}


def clear_cache(video_path=None):
    """Clear cached results.  None -> clear all."""
    if video_path:
        _cache.pop(video_path, None)
    else:
        _cache.clear()


# ── RADIO Model ──────────────────────────────────────────────────────────

def _load_radio():
    """Load and cache NVIDIA C-RADIO model on CUDA."""
    global _radio_model
    if _radio_model is None:
        _radio_model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version='c-radio_v3-b',
            progress=False, skip_validation=True, verbose=False
        )
        _radio_model = _radio_model.cuda().eval()
    return _radio_model


def _unload_radio():
    """Free RADIO model from GPU after scenes/clusters are computed."""
    global _radio_model
    if _radio_model is not None:
        del _radio_model
        _radio_model = None
        torch.cuda.empty_cache()


def _embed_images(image_paths, batch_size=64, target_res=(224, 224)):
    """Compute RADIO summary + spatial embeddings for a list of images."""
    model = _load_radio()
    all_summary, all_spatial = [], []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = [pil_to_tensor(Image.open(p).convert('RGB')).float() / 255.0
                   for p in batch_paths]

        batch = torch.stack(tensors)
        batch = F.interpolate(batch, size=target_res, mode='bilinear', align_corners=False)
        batch = batch.to(dtype=torch.bfloat16, device='cuda')

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            summary, spatial = model(batch)

        all_summary.append(summary.float().cpu())
        all_spatial.append(spatial.float().mean(dim=1).cpu())

    return torch.cat(all_summary), torch.cat(all_spatial)


# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_output_dir(video_path, output_dir):
    """Return output_dir (creating if needed), using cached dir as fallback."""
    if output_dir is None:
        output_dir = _cache.get(video_path, {}).get("output_dir")
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hpp_hierarchy_")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _ffprobe_duration(video_path):
    """Get video duration from container header (fast, no scanning)."""
    cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True
    )
    try:
        return float(json.loads(cmd.stdout)["format"]["duration"])
    except Exception:
        return None


def _ffprobe_packets(video_path):
    """Full packet scan — I-frame timestamps + all packet sizes for action scores.

    Returns: (iframe_times, packet_sizes, total_duration)
    """
    total_duration = _ffprobe_duration(video_path)

    pkt_cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_packets", "-show_entries", "packet=pts_time,size,flags",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )

    iframe_times = []
    packet_sizes = []

    for line in pkt_cmd.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            ts = float(parts[0])
        except (ValueError, IndexError):
            continue
        size = int(parts[1]) if parts[1] else 0
        flags = parts[2] if len(parts) > 2 else ""

        packet_sizes.append((ts, np.log(size + 1)))
        if "K" in flags:
            iframe_times.append(ts)

    iframe_times.sort()
    return iframe_times, packet_sizes, total_duration


def _ffprobe_keyframes_only(video_path):
    """Keyframe-only scan — much faster, skips non-keyframe packets entirely.

    Returns: (iframe_times, total_duration)
    """
    total_duration = _ffprobe_duration(video_path)

    cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-skip_frame", "nokey",
         "-show_frames", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )

    iframe_times = []
    for line in cmd.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            iframe_times.append(float(line))
        except ValueError:
            continue

    iframe_times.sort()
    return iframe_times, total_duration


def _action_scores_from_packets(iframe_times, packet_sizes, total_duration):
    """Compute per-shot action scores from packet sizes (accurate)."""
    iframe_arr = np.array(iframe_times)
    ends = np.append(iframe_arr[1:], total_duration)
    pkt_ts = np.array([t for t, _ in packet_sizes])
    pkt_sz = np.array([s for _, s in packet_sizes])

    shot_actions = np.zeros(len(iframe_times))
    for i in range(len(iframe_times)):
        mask = (pkt_ts >= iframe_arr[i]) & (pkt_ts < ends[i])
        if mask.any():
            shot_actions[i] = pkt_sz[mask].mean()

    max_action = shot_actions.max()
    if max_action > 0:
        shot_actions /= max_action
    return shot_actions, ends


def _action_scores_from_duration(iframe_times, total_duration):
    """Approximate action scores from shot duration (fast, no packet scan).

    Shorter shots → higher action (more cuts = more dynamic content).
    """
    iframe_arr = np.array(iframe_times)
    ends = np.append(iframe_arr[1:], total_duration)
    durations = ends - iframe_arr

    # Inverse duration: shorter = higher action
    inv_dur = 1.0 / np.maximum(durations, 0.01)
    max_inv = inv_dur.max()
    if max_inv > 0:
        shot_actions = inv_dur / max_inv
    else:
        shot_actions = np.zeros(len(iframe_times))
    return shot_actions, ends


def _save_jpeg(args):
    """Save a single (RGB numpy array, path) pair. Used by ThreadPoolExecutor."""
    rgb, path = args
    Image.fromarray(rgb).save(path, quality=_JPEG_QUALITY)
    return path


def _hist_corr(h1, h2):
    """Pearson correlation between two flat histograms."""
    h1 = h1 - h1.mean()
    h2 = h2 - h2.mean()
    denom = np.sqrt((h1 * h1).sum() * (h2 * h2).sum())
    return float((h1 * h2).sum() / denom) if denom > 0 else 0.0


def _compute_hist(rgb_frame, resize=(64, 64)):
    """Compute normalised HSV histogram from an RGB numpy array."""
    img = Image.fromarray(rgb_frame).convert("HSV").resize(resize, Image.BILINEAR)
    arr = np.asarray(img)
    h = np.histogramdd(arr.reshape(-1, 3).astype(np.float32),
                       bins=[8, 8, 8],
                       range=[(0, 256), (0, 256), (0, 256)])[0]
    h = h / (h.sum() + 1e-8)
    return h.ravel()


def _merge_similar_shots(shots, hists, threshold=0.80):
    """Merge adjacent shots whose keyframe HSV histograms are correlated >= threshold.

    Args:
        shots: list of shot dicts (with keyframe paths already set)
        hists: list of pre-computed histogram vectors (or None per shot)

    Returns: (merged_shots, surviving_indices)
        merged_shots: new list of shot dicts with renumbered shot_ids
        surviving_indices: set of original indices whose keyframes are used
    """
    if len(shots) <= 1:
        return list(shots), set(range(len(shots)))

    # Greedy forward merge
    groups = [[0]]
    for i in range(1, len(shots)):
        prev_idx = groups[-1][-1]
        if hists[prev_idx] is not None and hists[i] is not None:
            corr = _hist_corr(hists[prev_idx], hists[i])
        else:
            corr = 0.0
        if corr >= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])

    # Build merged shot list
    merged = []
    surviving = set()
    for new_id, idxs in enumerate(groups):
        first = shots[idxs[0]]
        last = shots[idxs[-1]]
        mid_idx = idxs[len(idxs) // 2]
        mid = shots[mid_idx]
        surviving.add(mid_idx)
        avg_action = np.mean([shots[j].get("action", 0.0) for j in idxs])
        merged.append({
            "shot_id": new_id,
            "start": first["start"],
            "end": last["end"],
            "duration": round(last["end"] - first["start"], 2),
            "keyframe": mid["keyframe"],
            "action": round(float(avg_action), 2),
        })

    return merged, surviving


# ══════════════════════════════════════════════════════════════════════════
# Level 1: Shots — I-frame based with action scores + merging
# ══════════════════════════════════════════════════════════════════════════

def get_shots(video_path, output_dir=None, merge_threshold=0.99,
              action_mode="packet"):
    """I-frame shot boundaries with action scores and optional merging.

    Args:
        video_path: path to video file
        output_dir: where to save keyframes
        merge_threshold: HSV histogram correlation for merging.
            None = no merging.  e.g. 0.80 for moderate merging.
        action_mode: how to compute action scores:
            "packet"   — full packet scan, accurate (default)
            "keyframe" — keyframe-only scan, duration-based action (3-5x faster)

    Returns:
        list[dict]: [{shot_id, start, end, duration, keyframe, action}, ...]
    """
    if video_path in _cache and "shots" in _cache[video_path]:
        cached = _cache[video_path]["shots"]
        if cached and os.path.exists(cached[0]["keyframe"]):
            return cached

    output_dir = _ensure_output_dir(video_path, output_dir)

    # ── Pass 1: ffprobe — get I-frame times (+ optionally packet sizes) ──
    if action_mode == "keyframe":
        iframe_times, total_duration = _ffprobe_keyframes_only(video_path)
        packet_sizes = None
    else:
        iframe_times, packet_sizes, total_duration = _ffprobe_packets(video_path)

    if not iframe_times:
        # Fallback: grab first frame with decord
        kf = os.path.join(output_dir, "shot_0_0.00s.jpg")
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            Image.fromarray(vr[0].asnumpy()).save(kf)
            if total_duration is None:
                total_duration = len(vr) / vr.get_avg_fps()
            del vr
        except Exception:
            pass
        dur = total_duration or 0.0
        shots = [{"shot_id": 0, "start": 0.0, "end": round(dur, 2),
                  "duration": round(dur, 2), "keyframe": kf, "action": 0.0}]
        c = _cache.setdefault(video_path, {})
        c["shots"] = shots
        c["output_dir"] = output_dir
        return shots

    if total_duration is None:
        total_duration = iframe_times[-1] + 1.0

    # ── Action scores ──
    if action_mode == "keyframe" or packet_sizes is None:
        shot_actions, ends = _action_scores_from_duration(iframe_times, total_duration)
    else:
        shot_actions, ends = _action_scores_from_packets(
            iframe_times, packet_sizes, total_duration)

    # ── Pass 2: decord batch read of keyframes (reduced resolution) ──
    vr = VideoReader(video_path, ctx=cpu(0),
                     width=_DECODE_WIDTH, height=_DECODE_HEIGHT)
    fps = vr.get_avg_fps()
    n_frames = len(vr)

    iframe_arr = np.array(iframe_times)
    kf_indices = np.clip((iframe_arr * fps).astype(int), 0, n_frames - 1)

    # Batch read all keyframes at once
    kf_frames = vr.get_batch(kf_indices.tolist()).asnumpy()  # (N, H, W, 3) RGB
    del vr

    # ── Build paths + compute merge histograms in-memory ──
    kf_paths = []
    hists = []
    for idx in range(len(iframe_times)):
        ts = iframe_times[idx]
        kf_paths.append(os.path.join(output_dir, f"shot_{idx}_{ts:.2f}s.jpg"))
        if merge_threshold is not None:
            hists.append(_compute_hist(kf_frames[idx]))
        else:
            hists.append(None)

    # ── Build raw shots ──
    raw_shots = []
    for i in range(len(iframe_times)):
        raw_shots.append({
            "shot_id": i,
            "start": round(iframe_times[i], 2),
            "end": round(float(ends[i]), 2),
            "duration": round(float(ends[i] - iframe_times[i]), 2),
            "keyframe": kf_paths[i],
            "action": round(float(shot_actions[i]), 4),
        })

    # ── Merge, then save only surviving keyframes ──
    if merge_threshold is not None:
        shots, surviving = _merge_similar_shots(raw_shots, hists, threshold=merge_threshold)
        merge_info = f"{len(raw_shots)} raw → {len(shots)} (merged, thresh={merge_threshold})"
    else:
        shots = raw_shots
        surviving = set(range(len(raw_shots)))
        merge_info = f"{len(shots)} shots"

    # Only save keyframes that survived merging (skip ~50% that got merged away)
    save_jobs = [(kf_frames[i], kf_paths[i]) for i in surviving]
    with ThreadPoolExecutor(max_workers=_SAVE_WORKERS) as pool:
        list(pool.map(_save_jpeg, save_jobs))

    c = _cache.setdefault(video_path, {})
    c["shots"] = shots
    c["output_dir"] = output_dir

    avg_dur = sum(s["duration"] for s in shots) / len(shots) if shots else 0
    print(f"{merge_info} | "
          f"{shots[0]['start']:.0f}s - {shots[-1]['end']:.0f}s | "
          f"avg {avg_dur:.1f}s/shot | mode={action_mode}")
    return shots


# ══════════════════════════════════════════════════════════════════════════
# Level 2: Scenes — merge adjacent shots by cosine similarity
# ══════════════════════════════════════════════════════════════════════════

def get_scenes(video_path, output_dir=None, threshold=0.6):
    """Merge adjacent similar shots into scenes using RADIO spatial features.

    Computes cosine similarity between adjacent shot embeddings.  If
    sim(z_i, z_{i+1}) >= threshold the shots belong to the same scene.

    Args:
        video_path: path to video file
        output_dir: where to save keyframes
        threshold: cosine similarity above which adjacent shots merge (0-1)

    Returns:
        list[dict]: [{scene_id, start, end, duration, shot_ids, keyframe, action}, ...]
    """
    if video_path in _cache and "scenes" in _cache[video_path]:
        cached = _cache[video_path]["scenes"]
        if cached and os.path.exists(cached[0]["keyframe"]):
            return cached

    shots = get_shots(video_path, output_dir)

    if len(shots) <= 1:
        scenes = [_shot_to_scene(shots, [0], scene_id=0)]
        c = _cache.setdefault(video_path, {})
        c["scenes"] = scenes
        c["clusters"] = [{
            "cluster_id": 0, "scene_ids": [0],
            "start": scenes[0]["start"], "end": scenes[0]["end"],
            "duration": scenes[0]["duration"], "keyframe": scenes[0]["keyframe"],
        }]
        return scenes

    # Compute RADIO spatial embeddings
    paths = [s["keyframe"] for s in shots]
    summary, spatial = _embed_images(paths)

    # Cosine similarity between adjacent shots
    norm = F.normalize(spatial, dim=1)
    sims = (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()

    # Merge: if similarity >= threshold, keep in same scene
    groups = [[0]]
    for i in range(len(sims)):
        if sims[i] >= threshold:
            groups[-1].append(i + 1)
        else:
            groups.append([i + 1])

    scenes = []
    for sid, shot_ids in enumerate(groups):
        scenes.append(_shot_to_scene(shots, shot_ids, scene_id=sid))

    _cache.setdefault(video_path, {})["scenes"] = scenes
    # Also compute clusters (same GPU pass), then free embeddings and model
    _compute_clusters(video_path, scenes, summary)
    _cache[video_path].pop("_embeddings", None)
    _unload_radio()

    print(f"{len(scenes)} scenes from {len(shots)} shots (sim threshold={threshold})")
    return scenes


def _shot_to_scene(shots, shot_ids, scene_id):
    """Build a scene dict from a group of shot indices."""
    first = shots[shot_ids[0]]
    last = shots[shot_ids[-1]]
    mid = shots[shot_ids[len(shot_ids) // 2]]
    avg_action = np.mean([shots[j].get("action", 0.0) for j in shot_ids])
    return {
        "scene_id": scene_id,
        "start": first["start"],
        "end": last["end"],
        "duration": round(last["end"] - first["start"], 2),
        "shot_ids": shot_ids,
        "keyframe": mid["keyframe"],
        "action": round(float(avg_action), 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# Level 3: Clusters — global semantic grouping
# ══════════════════════════════════════════════════════════════════════════

def _compute_clusters(video_path, scenes, summary_embs, distance_threshold=0.7):
    """Cluster scenes using global RADIO summary embeddings.

    Uses agglomerative clustering with a distance threshold — the number
    of clusters emerges naturally from the data.

    Args:
        distance_threshold: cosine distance cutoff for merging (lower = more clusters)
    """
    if len(scenes) <= 1:
        clusters = [{
            "cluster_id": 0,
            "scene_ids": [0],
            "start": scenes[0]["start"],
            "end": scenes[0]["end"],
            "duration": scenes[0]["duration"],
            "keyframe": scenes[0]["keyframe"],
        }]
        _cache.setdefault(video_path, {})["clusters"] = clusters
        return clusters

    # Mean summary embedding per scene
    scene_embs = []
    for sc in scenes:
        scene_embs.append(summary_embs[sc["shot_ids"]].mean(dim=0))
    scene_embs = torch.stack(scene_embs)

    # Pairwise cosine distance matrix
    norm = F.normalize(scene_embs, dim=1)
    dist = 1 - (norm @ norm.T).numpy()
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='complete',
    )
    labels = clustering.fit_predict(dist)

    cluster_map = {}
    for scene_id, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(scene_id)

    clusters = []
    for cid, scene_ids in sorted(cluster_map.items()):
        mid = scene_ids[len(scene_ids) // 2]
        starts = [scenes[s]["start"] for s in scene_ids]
        ends = [scenes[s]["end"] for s in scene_ids]
        avg_action = np.mean([scenes[s].get("action", 0.0) for s in scene_ids])
        clusters.append({
            "cluster_id": cid,
            "scene_ids": scene_ids,
            "start": min(starts),
            "end": max(ends),
            "duration": round(sum(scenes[s]["duration"] for s in scene_ids), 2),
            "keyframe": scenes[mid]["keyframe"],
            "action": round(float(avg_action), 2),
        })

    _cache.setdefault(video_path, {})["clusters"] = clusters

    sizes = [len(c["scene_ids"]) for c in clusters]
    print(f"{len(clusters)} clusters (dist_thresh={distance_threshold}) | "
          f"scenes per cluster: {sizes}")
    return clusters


def get_clusters(video_path, output_dir=None):
    """Get clusters. Triggers get_scenes() if not cached.

    Returns:
        list[dict]: [{cluster_id, scene_ids, start, end, duration, keyframe, action}, ...]
    """
    if video_path in _cache and "clusters" in _cache[video_path]:
        cached = _cache[video_path]["clusters"]
        if cached and os.path.exists(cached[0]["keyframe"]):
            return cached

    # This will also compute clusters
    get_scenes(video_path, output_dir)
    return _cache[video_path]["clusters"]


# ══════════════════════════════════════════════════════════════════════════
# Video metadata
# ══════════════════════════════════════════════════════════════════════════

def get_video_metadata(video_path):
    """Basic video metadata plus cached hierarchy counts (no computation triggered).

    Args:
        video_path: path to video file

    Returns:
        dict with keys:
            duration  — total seconds (float)
            fps       — frames per second (float)
            width     — frame width in pixels (int)
            height    — frame height in pixels (int)
            n_shots   — number of shots if already computed, else None
            n_scenes  — number of scenes if already computed, else None
            n_clusters— number of clusters if already computed, else None
    """
    cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
         "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True
    )
    try:
        info = json.loads(cmd.stdout)
        stream = info.get("stream") or (info.get("streams") or [{}])[0]
        fmt = info.get("format", {})
        duration = float(fmt.get("duration", 0))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps_str = stream.get("avg_frame_rate", "0/1")
        num, den = (fps_str.split("/") + ["1"])[:2]
        fps = round(float(num) / max(float(den), 1e-6), 3)
    except Exception:
        duration, width, height, fps = 0.0, 0, 0, 0.0

    cached = _cache.get(video_path, {})
    shots = cached.get("shots")
    scenes = cached.get("scenes")
    clusters = cached.get("clusters")

    meta = {
        "duration": round(duration, 2),
        "fps": fps,
        "width": width,
        "height": height,
        "n_shots": len(shots) if shots is not None else None,
        "n_scenes": len(scenes) if scenes is not None else None,
        "n_clusters": len(clusters) if clusters is not None else None,
    }

    parts = [f"duration={meta['duration']}s", f"fps={fps}",
             f"{width}x{height}"]
    if meta["n_shots"] is not None:
        parts.append(f"shots={meta['n_shots']}")
    if meta["n_scenes"] is not None:
        parts.append(f"scenes={meta['n_scenes']}")
    if meta["n_clusters"] is not None:
        parts.append(f"clusters={meta['n_clusters']}")
    print("video_metadata:", " | ".join(parts))
    return meta


# ══════════════════════════════════════════════════════════════════════════
# Clip extraction
# ══════════════════════════════════════════════════════════════════════════

def get_clip(video_path, start, end, output_dir=None):
    """Extract a video clip between two timestamps (ffmpeg stream copy).

    Args:
        video_path: path to video file
        start: start time in seconds
        end: end time in seconds

    Returns: str — path to .mp4 clip
    """
    output_dir = _ensure_output_dir(video_path, output_dir)
    out_path = os.path.join(output_dir, f"clip_{start:.1f}s_{end:.1f}s.mp4")
    duration = end - start

    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-c", "copy", out_path,
    ], check=True, capture_output=True)

    print(f"Clip: {start:.1f}s - {end:.1f}s ({duration:.1f}s) → {out_path}")
    return out_path
