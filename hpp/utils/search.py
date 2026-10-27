"""
Motion-aware semantic search over shots/scenes with late-interaction (MaxSim) scoring.

  - Each shot gets a frame budget proportional to its codec motion energy
    (the `action` score from hierarchy.get_shots), between _MIN_FRAMES and _MAX_FRAMES.
  - Frames are encoded per-frame with VideoCLIP-XL-v2; query tokens are encoded
    with its text encoder.
  - score = a * MaxSim(query tokens, shot frames) + (1 - a) * cos(mean query, mean shot),
    with a = SEARCH_MAXSIM_WEIGHT (default 0.5).
  - Scene embeddings are pooled from their shots, so scenes cost no extra encoding.

Frame embeddings are cached in RAM per video: the first query encodes the video,
later queries only encode text.

  search_in_shots(video_path, query, n=5, threshold=None)
  search_in_scenes(video_path, query, n=5, threshold=None)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hpp.utils.hierarchy import get_shots, get_scenes

# ── Config ───────────────────────────────────────────────────────────────

# Local copy of https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2 (scripts/download_models.sh)
_VIDEOCLIP_DIR = os.environ.get(
    "HPP_VIDEOCLIP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "VideoCLIP-XL-v2"),
)
_MIN_FRAMES    = 2     # minimum frames per shot/part
_MAX_FRAMES    = 8    # maximum frames per shot/part
_KF_WEIGHT     = 1     # keyframe appended this many extra times in the frame pool
_ENCODE_BATCH  = 64    # frames per GPU batch during encoding
_MAXSIM_WEIGHT = float(os.environ.get("SEARCH_MAXSIM_WEIGHT", "0.5"))
_SEARCH_ENCODER = os.environ.get("SEARCH_ENCODER", "videoclip")  # videoclip | clip | siglip

_V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_V_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

# ── Caches ───────────────────────────────────────────────────────────────

_videoclip_model = None
_text_tokenizer  = None
_hf_model        = None
_hf_processor    = None
_cache = {}  # video_path -> {key -> data}  (embeddings on CPU)


# ── Model loading ────────────────────────────────────────────────────────

def check_search_model():
    """Fail fast if the VideoCLIP-XL-v2 files are missing (otherwise every search call
    fails inside the REPL and the LLM silently loses semantic search)."""
    if _SEARCH_ENCODER != "videoclip":
        return
    needed = ["modeling.py", "utils/text_encoder/text_encoder.py", "VideoCLIP-XL-v2.bin"]
    missing = [f for f in needed if not os.path.exists(os.path.join(_VIDEOCLIP_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"VideoCLIP-XL-v2 not found in {os.path.abspath(_VIDEOCLIP_DIR)} (missing: {', '.join(missing)}). "
            "Run `bash scripts/download_models.sh`, or set HPP_VIDEOCLIP_DIR.")


def _load_videoclip():
    global _videoclip_model, _text_tokenizer
    if _videoclip_model is not None:
        return _videoclip_model, _text_tokenizer

    check_search_model()
    if _VIDEOCLIP_DIR not in sys.path:
        sys.path.insert(0, _VIDEOCLIP_DIR)

    from modeling import VideoCLIP_XL
    from utils.text_encoder import text_encoder

    model = VideoCLIP_XL()
    weights = os.path.join(_VIDEOCLIP_DIR, "VideoCLIP-XL-v2.bin")
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    _videoclip_model = model
    _text_tokenizer  = text_encoder
    return model, text_encoder


def _load_hf_encoder():
    global _hf_model, _hf_processor
    if _hf_model is not None:
        return _hf_model, _hf_processor
    from transformers import AutoModel, AutoProcessor
    names = {"clip":   "openai/clip-vit-large-patch14",
             "siglip": "google/siglip-so400m-patch14-384"}
    name = names[_SEARCH_ENCODER]
    _hf_processor = AutoProcessor.from_pretrained(name)
    _hf_model = AutoModel.from_pretrained(name).cuda().eval()
    return _hf_model, _hf_processor


def _unload_vision_encoder():
    """Move vision encoder to CPU after frame embeddings are cached in RAM.

    Text encoder stays on GPU for fast per-query text encoding.
    This frees ~1.2GB VRAM, leaving room for RADIO during get_scenes().
    Vision encoder is moved back to GPU automatically when _encode_frames is called.
    """
    if _SEARCH_ENCODER != "videoclip":
        return  # HF models stay on GPU; no offloading needed
    if _videoclip_model is not None and _videoclip_model.vision_model is not None:
        if next(_videoclip_model.vision_model.parameters()).is_cuda:
            _videoclip_model.vision_model = _videoclip_model.vision_model.cpu()
            torch.cuda.empty_cache()


def _ensure_vision_on_gpu():
    """Move vision encoder back to GPU if it was offloaded. No-op if already there."""
    if _SEARCH_ENCODER != "videoclip":
        return  # HF models stay on GPU permanently
    if _videoclip_model is not None and _videoclip_model.vision_model is not None:
        if not next(_videoclip_model.vision_model.parameters()).is_cuda:
            _videoclip_model.vision_model = _videoclip_model.vision_model.cuda()
            print("VideoCLIP vision encoder moved back to GPU")


# ── Frame budget based on motion ─────────────────────────────────────────

def _motion_frame_counts(action_scores, video_duration, n_segments):
    """Distribute frame budget across segments proportional to action scores.

    Total budget = n_segments * base_frames, where base_frames scales with
    average segment duration: max(4, min(16, avg_dur // 4)).

    Each segment gets at least _MIN_FRAMES and at most _MAX_FRAMES frames.
    High-action segments get proportionally more frames than low-action ones.

    Args:
        action_scores: array-like of float, one per segment (0–1 normalised)
        video_duration: total video duration in seconds
        n_segments: number of segments

    Returns:
        list[int] — frame count per segment
    """
    if n_segments == 0:
        return []

    avg_dur    = video_duration / n_segments
    base       = max(_MIN_FRAMES, min(_MAX_FRAMES, int(avg_dur / 4)))
    total      = n_segments * base

    actions = np.array(action_scores, dtype=float)
    actions = np.clip(actions, 0.01, None)          # avoid zero weights
    weights = actions / actions.sum()

    raw    = weights * total
    counts = np.round(raw).astype(int)
    counts = np.clip(counts, _MIN_FRAMES, _MAX_FRAMES)

    # Adjust total to match budget (add/remove from highest-action items)
    diff = total - int(counts.sum())
    if diff != 0:
        order = np.argsort(-actions)                # highest action first
        step  = int(np.sign(diff))
        for idx in order:
            if diff == 0:
                break
            new_val = counts[idx] + step
            if _MIN_FRAMES <= new_val <= _MAX_FRAMES:
                counts[idx] = new_val
                diff -= step

    return counts.tolist()


def _part_action_scores(segments, shots):
    """Derive action scores for equal-duration parts from shot overlap.

    For each segment, average the action scores of all shots that overlap it.
    Falls back to 0.5 for parts with no shot coverage.

    Returns: np.ndarray of float, one per segment.
    """
    scores = np.empty(len(segments))
    for i, seg in enumerate(segments):
        overlapping = [
            sh["action"] for sh in shots
            if sh["start"] < seg["end"] and sh["end"] > seg["start"]
        ]
        scores[i] = float(np.mean(overlapping)) if overlapping else 0.5
    return scores


# ── Frame sampling ───────────────────────────────────────────────────────

def _get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (n / fps) if fps > 0 else 0.0


def _sample_frames(video_path, segments, frame_counts):
    """Sample frames per segment. Single cap.open for all segments.

    Args:
        segments:     list of dicts with "start"/"end" keys (seconds)
        frame_counts: list[int] of per-segment frame counts

    Returns:
        list of (n_frames_i, 3, 224, 224) float32 numpy arrays
    """
    cap    = cv2.VideoCapture(video_path)
    result = []

    for seg, n_frames in zip(segments, frame_counts):
        start = seg["start"]
        end   = seg["end"]
        dur   = max(end - start, 0.1)
        frames = []

        for i in range(n_frames):
            t = start + (i / max(n_frames - 1, 1)) * dur
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (224, 224))
                f = ((frame / 255.0 - _V_MEAN) / _V_STD).astype(np.float32)
                frames.append(f.transpose(2, 0, 1))   # (3, 224, 224)

        # Pad if any reads failed
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames
                          else np.zeros((3, 224, 224), dtype=np.float32))

        result.append(np.stack(frames[:n_frames]))    # (n_frames, 3, 224, 224)

    cap.release()
    return result


# ── Encoding ─────────────────────────────────────────────────────────────

def _encode_frames_hf(all_frames_np, batch_size=_ENCODE_BATCH):
    """Encode frames through a HF CLIP/SigLIP vision encoder.

    Args:
        all_frames_np: list of (n_frames_i, 3, 224, 224) float32 numpy arrays
                       in VideoCLIP normalisation space (ImageNet mean/std, CHW)
    Returns:
        list of (n_frames_i, D) CPU float32 tensors — one per segment
    """
    from PIL import Image
    model, processor = _load_hf_encoder()

    flat_pil  = []
    boundaries = []
    for seg_frames in all_frames_np:
        s = len(flat_pil)
        # Undo VideoCLIP (ImageNet) normalisation → raw uint8 HWC RGB
        hwc = seg_frames.transpose(0, 2, 3, 1)                             # (n, H, W, 3)
        uint8 = ((hwc * _V_STD + _V_MEAN) * 255).clip(0, 255).astype(np.uint8)
        for f in uint8:
            flat_pil.append(Image.fromarray(f))
        boundaries.append((s, len(flat_pil)))

    all_embs = []
    for i in range(0, len(flat_pil), batch_size):
        inputs = processor(images=flat_pil[i:i + batch_size], return_tensors="pt")
        pixel_values = inputs["pixel_values"].cuda()
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values)
        all_embs.append(F.normalize(feats.float(), dim=-1).cpu())

    all_cat = torch.cat(all_embs, dim=0)  # (N_total, D)
    return [all_cat[s:e] for s, e in boundaries]


def _encode_frames(all_frames_np, batch_size=_ENCODE_BATCH):
    """Encode all frames per-frame through VideoCLIP-XL.

    Args:
        all_frames_np: list of (n_frames_i, 3, 224, 224) numpy arrays
    Returns:
        list of (n_frames_i, D) CPU float32 tensors — one per segment
    """
    if _SEARCH_ENCODER != "videoclip":
        return _encode_frames_hf(all_frames_np, batch_size)
    model, _ = _load_videoclip()
    _ensure_vision_on_gpu()

    # Flatten all frames, track segment boundaries
    flat       = []
    boundaries = []
    for seg_frames in all_frames_np:
        s = len(flat)
        flat.extend(seg_frames)
        boundaries.append((s, len(flat)))

    stacked   = np.stack(flat)   # (N_total, 3, 224, 224)
    all_embs  = []

    for i in range(0, len(stacked), batch_size):
        batch = (torch.from_numpy(stacked[i:i + batch_size])
                 .unsqueeze(1).float().cuda())   # (B, 1, 3, 224, 224)
        with torch.no_grad():
            feats = model.vision_model.get_vid_features(batch).float()
        all_embs.append(F.normalize(feats, dim=-1).cpu())

    all_cat = torch.cat(all_embs, dim=0)  # (N_total, D) on CPU

    return [all_cat[s:e] for s, e in boundaries]


def _encode_keyframes(shots, batch_size=_ENCODE_BATCH):
    """Encode each shot's precomputed keyframe image file.

    Returns:
        list of (1, D) CPU float32 tensors, one per shot.
        Entry is None if the keyframe file is missing or unreadable.
    """
    if _SEARCH_ENCODER != "videoclip":
        return [None] * len(shots)  # keyframe boosting disabled for HF encoders
    model, _ = _load_videoclip()
    _ensure_vision_on_gpu()

    frames_np     = []
    valid_indices = []

    for i, shot in enumerate(shots):
        kf_path = shot.get("keyframe")
        if kf_path and os.path.exists(kf_path):
            frame = cv2.imread(kf_path)
            if frame is not None:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (224, 224))
                f = ((frame / 255.0 - _V_MEAN) / _V_STD).astype(np.float32)
                frames_np.append(f.transpose(2, 0, 1))  # (3, 224, 224)
                valid_indices.append(i)

    result = [None] * len(shots)
    if not frames_np:
        return result

    stacked  = np.stack(frames_np)   # (N_valid, 3, 224, 224)
    all_embs = []

    for i in range(0, len(stacked), batch_size):
        batch = (torch.from_numpy(stacked[i:i + batch_size])
                 .unsqueeze(1).float().cuda())   # (B, 1, 3, 224, 224)
        with torch.no_grad():
            feats = model.vision_model.get_vid_features(batch).float()
        all_embs.append(F.normalize(feats, dim=-1).cpu())

    encoded = torch.cat(all_embs, dim=0)  # (N_valid, D)

    for j, shot_idx in enumerate(valid_indices):
        result[shot_idx] = encoded[j:j + 1]   # (1, D)

    return result


def _encode_text_hf(query):
    """Encode query via HF CLIP/SigLIP → (pooled (1,D), per-token (M,D)) on GPU float32."""
    model, processor = _load_hf_encoder()
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        pooled      = model.get_text_features(**inputs)               # (1, D)
        text_out    = model.text_model(**inputs)
        last_hidden = text_out.last_hidden_state                      # (1, M, H)

        mask        = inputs["attention_mask"][0].bool()              # (M,)
        valid_h     = last_hidden[0][mask]                            # (M_valid, H)

        if hasattr(model, "text_projection"):
            # CLIP: top-level text_projection Linear
            token_feats = model.text_projection(valid_h)              # (M_valid, D)
        else:
            # SigLIP: projection is model.text_model.head
            token_feats = model.text_model.head(valid_h)              # (M_valid, D)

    return (F.normalize(pooled.float(),       dim=-1),
            F.normalize(token_feats.float(),  dim=-1))


def _encode_text(query):
    """Encode query → (pooled (1,D), per-token (M,D)) on GPU float32."""
    if _SEARCH_ENCODER != "videoclip":
        return _encode_text_hf(query)
    model, tokenizer = _load_videoclip()
    tokens = tokenizer.tokenize([query], truncate=True).cuda()

    with torch.no_grad():
        full_feats = model.text_model.encode_text_full(tokens)  # (1, n_ctx, D)
        proj       = model.text_model.text_projection

        eot_idx = tokens.argmax(dim=-1)
        pooled  = full_feats[torch.arange(1), eot_idx] @ proj    # (1, D)
        all_tok = full_feats @ proj                               # (1, n_ctx, D)

    mask        = tokens[0] != 0
    token_feats = all_tok[0][mask]                                # (M, D)

    return (F.normalize(pooled.float(),      dim=-1),
            F.normalize(token_feats.float(), dim=-1))


# ── MaxSim scoring ───────────────────────────────────────────────────────

def _score_maxsim(text_tok, text_pooled, per_seg_embs_cpu, kf_embs_cpu=None):
    """Compute MaxSim + cosine hybrid score for all segments in one GPU pass.

    MaxSim is normalised by dividing by M (number of text tokens), keeping
    scores in [-1, 1] regardless of query length. This makes absolute
    thresholds meaningful across queries of different lengths.

    Score = _MAXSIM_WEIGHT * mean_MaxSim + (1 - _MAXSIM_WEIGHT) * cosine
      where mean_MaxSim = mean over M tokens of max-over-frames cosine sim.

    If kf_embs_cpu is provided, each shot's keyframe embedding is appended
    _KF_WEIGHT times to its frame pool before scoring, giving it extra weight.

    CPU embeddings are moved to GPU only for the matmul, then freed.

    Args:
        text_tok:          (M, D) GPU — per-token text embeddings
        text_pooled:       (1, D) GPU — pooled text embedding
        per_seg_embs_cpu:  list of (K_i, D) CPU tensors
        kf_embs_cpu:       optional list of (1, D) or None entries, one per segment

    Returns: (N,) GPU float tensor of scores in [-1, 1]
    """
    boundaries = []
    offset     = 0
    all_parts  = []

    for i, embs in enumerate(per_seg_embs_cpu):
        parts = [embs]
        if (kf_embs_cpu is not None
                and i < len(kf_embs_cpu)
                and kf_embs_cpu[i] is not None):
            parts.append(kf_embs_cpu[i].expand(_KF_WEIGHT, -1))  # (_KF_WEIGHT, D)
        seg = torch.cat(parts, dim=0)
        boundaries.append((offset, offset + len(seg)))
        offset += len(seg)
        all_parts.append(seg)

    all_embs   = torch.cat(all_parts, dim=0).float().cuda()  # (F_total, D)
    sim_matrix = text_tok @ all_embs.T                       # (M, F_total)

    n_items = len(per_seg_embs_cpu)
    scores  = torch.empty(n_items, device=text_tok.device)

    for i, (s, e) in enumerate(boundaries):
        maxsim = sim_matrix[:, s:e].max(dim=1).values.mean()  # mean over M tokens → [-1, 1]
        mean_f = F.normalize(all_embs[s:e].mean(dim=0, keepdim=True), dim=-1)
        cosine = (text_pooled @ mean_f.T).squeeze()
        scores[i] = _MAXSIM_WEIGHT * maxsim + (1 - _MAXSIM_WEIGHT) * cosine

    return scores


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def _shot_embeddings(video_path):
    """Per-shot frame embeddings (motion-based frame budget) + keyframe embeddings, cached in RAM."""
    c = _cache.setdefault(video_path, {})

    if "shot_embs" not in c:
        shots    = get_shots(video_path)
        duration = _get_video_duration(video_path)

        action_scores = np.array([s.get("action", 0.5) for s in shots])
        frame_counts  = _motion_frame_counts(action_scores, duration, len(shots))

        frames_list   = _sample_frames(video_path, shots, frame_counts)
        per_shot_embs = _encode_frames(frames_list)    # list of (K_i, D) CPU
        kf_embs       = _encode_keyframes(shots)       # list of (1, D) or None

        c["shots"]      = shots
        c["shot_embs"]  = per_shot_embs
        c["shot_kf"]    = kf_embs

        total = sum(frame_counts)
        avg   = total / len(shots) if shots else 0
        print(f"Embedded {len(shots)} shots "
              f"(motion-based: {total} total, avg {avg:.1f}/shot) + keyframes → RAM")
        _unload_vision_encoder()

    return c["shots"], c["shot_embs"], c["shot_kf"]


def search_in_shots(video_path, query, n=5, threshold=None):
    """Find shots most relevant to a text query using MaxSim scoring.

    Frame count per shot is proportional to its action score (motion),
    with total budget based on video duration. Each shot gets at least
    _MIN_FRAMES and at most _MAX_FRAMES frames. High-action shots get more frames.

    Embeddings cached in RAM after first call; subsequent queries are fast.

    Args:
        threshold: if set, only results with score >= threshold are returned
                   (may return fewer than n). Default None returns exactly n.

    Returns:
        list of dicts sorted by score (highest first):
        [{shot_id, start, end, duration, score, keyframe}, ...]
    """
    shots, per_shot_embs, kf_embs = _shot_embeddings(video_path)

    text_pooled, text_tok = _encode_text(query)
    scores = _score_maxsim(text_tok, text_pooled, per_shot_embs, kf_embs_cpu=kf_embs)

    n    = min(n, len(scores))
    topk = torch.topk(scores, n)

    results = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        s = shots[idx]
        results.append({
            "shot_id":  s["shot_id"],
            "start":    s["start"],
            "end":      s["end"],
            "duration": s["duration"],
            "score":    round(score, 3),
            "keyframe": s["keyframe"],
        })

    if threshold is not None:
        results = [r for r in results if r['score'] >= threshold]

    label = f"top {len(results)}" if threshold is None else f"{len(results)} (threshold={threshold})"
    print(f"search_in_shots '{query}' in {len(shots)} shots → {label}:")
    for r in results:
        print(f"  {r['start']:.1f}s - {r['end']:.1f}s  score={r['score']}")
    return results


def search_in_scenes(video_path, query, n=5, threshold=None):
    """Find scenes most relevant to a text query, reusing cached shot embeddings.

    Each scene's frame pool is built by concatenating the embeddings of its
    constituent shots (identified via scene["shot_ids"]).  No extra GPU encoding
    is needed if shot embeddings are already cached.

    Returns:
        list of dicts sorted by score (highest first):
        [{scene_id, start, end, duration, score, keyframe, shot_ids}, ...]
    """
    # Shares the shot-embedding cache with search_in_shots
    shots, per_shot_embs, _ = _shot_embeddings(video_path)

    # Build shot_id → index map
    shot_id_to_idx = {s["shot_id"]: i for i, s in enumerate(shots)}

    scenes = get_scenes(video_path)

    # Build per-scene embedding pools from shot embeddings
    scene_embs = []
    valid_scenes = []
    for scene in scenes:
        ids = scene.get("shot_ids", [])
        parts = [per_shot_embs[shot_id_to_idx[sid]]
                 for sid in ids if sid in shot_id_to_idx]
        if not parts:
            # Fallback: use all shot embs as a single pool
            parts = per_shot_embs
        scene_embs.append(torch.cat(parts, dim=0))  # (K_scene, D)
        valid_scenes.append(scene)

    if not valid_scenes:
        return []

    text_pooled, text_tok = _encode_text(query)
    scores = _score_maxsim(text_tok, text_pooled, scene_embs)

    n    = min(n, len(scores))
    topk = torch.topk(scores, n)

    results = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        sc = valid_scenes[idx]
        results.append({
            "scene_id": sc["scene_id"],
            "start":    sc["start"],
            "end":      sc["end"],
            "duration": sc["duration"],
            "score":    round(score, 3),
            "keyframe": sc.get("keyframe"),
            "shot_ids": sc.get("shot_ids", []),
        })

    if threshold is not None:
        results = [r for r in results if r['score'] >= threshold]

    label = f"top {len(results)}" if threshold is None else f"{len(results)} (threshold={threshold})"
    print(f"search_in_scenes '{query}' in {len(valid_scenes)} scenes → {label}:")
    for r in results:
        print(f"  {r['start']:.1f}s - {r['end']:.1f}s  score={r['score']}")
    return results


# ── Cache management ─────────────────────────────────────────────────────

def clear_cache(video_path=None):
    """Clear cached embeddings. None -> clear all."""
    if video_path:
        _cache.pop(video_path, None)
    else:
        _cache.clear()
