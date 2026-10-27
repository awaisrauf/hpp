"""
Export HPP reasoning traces for the web trajectory viewer (tools/viewer/).

For every trace this writes <out>/<id>.json (question, steps, code, REPL output,
VLM calls, and the video regions each step probed) and <out>/<id>/*.jpg (the frames
the VLM actually looked at, re-extracted from the video at the logged timestamps,
plus a filmstrip of the whole video). <out>/manifest.json lists all exports.

  python tools/export_trajectory.py runs/lvb/traces/lvb_3600_xyz_trace.json --out viewer_data
  python tools/export_trajectory.py --selection selection.json --out viewer_data

A selection file is a JSON list of {"trace": path, "video": optional override,
"id": ..., "title": ..., "note": ..., "thumb": optional card-image timestamp} entries.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

THUMB_W = 360          # px width of VLM frame thumbnails
STRIP_W = 160          # px width of filmstrip thumbnails
STRIP_N = 24           # filmstrip frames across the whole video
MAX_FRAMES_PER_CALL = 12
MAX_STDOUT = 3000
MAX_VLM_TEXT = 3000

FUNCS = ["get_video_info", "get_shots", "get_scenes", "get_clusters", "get_shot_detail",
         "search_in_shots", "search_in_scenes",
         "summarize_subtitles", "find_times", "search_subtitles", "get_subtitles",
         "get_overview", "sample", "sample_shot", "get_frames", "get_frame_at", "zoom",
         "get_clip", "vlm_query"]
FUNC_RE = re.compile(r"(?<![\w.])(" + "|".join(FUNCS) + r")\s*\(")

TS_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)s\.(?:jpg|png)")
CLIP_NAME = re.compile(r"clip_(\d+(?:\.\d+)?)s_(\d+(?:\.\d+)?)s\.mp4")
CLIP_IN_PROMPT = re.compile(r"\bfrom\s+(\d+(?:\.\d+)?)\s*s\s+to\s+(\d+(?:\.\d+)?)\s*s\b", re.I)
TMP_PATH = re.compile(r"/tmp/[\w/.\-]*?repl_env_\w+/")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The trace logger echoes each VLM call into the REPL output; the viewer shows those calls separately.
VLM_ECHO = re.compile(r"\n?  🔍 SUB-LLM CALL #\d+\n(?:  .*\n?)*")

# Lines printed by the probing functions themselves (hpp/utils/*.py). Hit lines are
# only read directly under their header, so the LLM's own prints are never mistaken for hits.
SEARCH_HDR = re.compile(r"^search_in_(shots|scenes) '(.*)' in \d+")
SEARCH_HIT = re.compile(r"^\s+(?:(?:shot|scene) \d+: )?(\d+(?:\.\d+)?)s ?- ?(\d+(?:\.\d+)?)s(?:\s+score=([\d.\-]+))?\s*$")
FIND_HDR = re.compile(r"^'(.*)' found at \d+ timestamps:")
FIND_HIT = re.compile(r"^\s+(\d+(?:\.\d+)?)s — (.*)")
SUBS_HDR = re.compile(r"^'(.*)' matched \d+ subtitles")
SUBS_HIT = re.compile(r"^\s+\[(\d+(?:\.\d+)?)s\] (.*)")
UNIFORM = re.compile(r"sample_uniform_frames\((\d+(?:\.\d+)?)s - (\d+(?:\.\d+)?)s, n=(\d+)\)")
ZOOM = re.compile(r"^zoom\(.*\| (\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s")
FRAME_AT = re.compile(r"^Frame at (\d+(?:\.\d+)?)s")
CLIP = re.compile(r"^Clip: (\d+(?:\.\d+)?)s - (\d+(?:\.\d+)?)s")


def video_info(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s, fmt = d["streams"][0], d["format"]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    fps = num / den if den else 0
    dur = float(fmt.get("duration", 0))
    n = int(s.get("nb_frames") or 0) or int(dur * fps)
    return {"duration": round(dur, 2), "fps": round(fps, 3), "frames": n,
            "width": s["width"], "height": s["height"]}


def extract_frame(video, t, out_path, width):
    if os.path.exists(out_path):
        return
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{max(t, 0):.3f}", "-i", video, "-frames:v", "1",
                    "-vf", f"scale={width}:-2", "-q:v", "6", out_path], check=False)


def parse_question(query):
    """Split the benchmark prompt into (note, question, options)."""
    q = re.sub(r"\s*Answer with just the option letter.*$", "", query.strip(), flags=re.S)
    note = ""
    if q.startswith("c in this question is camera wearer."):
        note = "C is the camera wearer."
        q = q[len("c in this question is camera wearer."):]
    m = re.search(r"(?:^|\s)A[\)\.]\s", q)
    if not m:
        return note, q.strip(), []
    stem, rest = q[:m.start()].strip(), q[m.start():]
    parts = re.split(r"(?:^|\s)([A-F])[\)\.]\s", " " + rest)
    options = [parts[i + 1].strip().rstrip(".").strip() for i in range(1, len(parts) - 1, 2)]
    return note, stem, options


def parse_spans(stdout):
    """Video regions probed in one REPL block, from the functions' own log lines."""
    spans, mode, label = [], None, None
    for line in stdout.splitlines():
        if m := SEARCH_HDR.match(line):
            mode, label = "search", m.group(2)
            continue
        if m := FIND_HDR.match(line) or SUBS_HDR.match(line):
            mode, label = "subtitle", m.group(1)
            continue
        if mode == "search" and (m := SEARCH_HIT.match(line)):
            spans.append({"kind": "search", "t0": float(m[1]), "t1": float(m[2]),
                          "score": float(m[3]) if m[3] else None, "label": label})
            continue
        if mode == "subtitle" and (m := FIND_HIT.match(line) or SUBS_HIT.match(line)):
            spans.append({"kind": "subtitle", "t0": float(m[1]), "t1": float(m[1]),
                          "label": f"'{label}': {m[2][:100]}"})
            continue
        mode = None
        if m := UNIFORM.search(line):
            spans.append({"kind": "frames", "t0": float(m[1]), "t1": float(m[2]), "label": f"{m[3]} frames"})
        elif m := ZOOM.match(line):
            spans.append({"kind": "zoom", "t0": float(m[1]), "t1": float(m[2]), "label": "zoom"})
        elif m := FRAME_AT.match(line):
            spans.append({"kind": "frames", "t0": float(m[1]), "t1": float(m[1]), "label": "frame"})
        elif m := CLIP.match(line):
            spans.append({"kind": "clip", "t0": float(m[1]), "t1": float(m[2]), "label": "clip"})
    # zoom() logs its own sampling line too; keep one span per region
    zooms = {(s["t0"], s["t1"]) for s in spans if s["kind"] == "zoom"}
    out, seen = [], set()
    for s in spans:
        key = (s["kind"], s["t0"], s["t1"])
        if key in seen or (s["kind"] == "frames" and (s["t0"], s["t1"]) in zooms):
            continue
        seen.add(key)
        out.append(s)
    return out


def clean(text, limit):
    text = TMP_PATH.sub("…/", ANSI.sub("", text or ""))
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


def export(trace_path, out_dir, video=None, tid=None, title=None, note=None, thumb_t=None, pool=None):
    tr = json.load(open(trace_path))
    md = tr["metadata"]
    qm = md.get("question_metadata") or {}
    video = video or md["video_path"]
    tid = tid or md.get("file_name") or os.path.basename(trace_path).replace("_trace.json", "")
    fdir = os.path.join(out_dir, tid)
    os.makedirs(fdir, exist_ok=True)
    info = video_info(video)
    jobs = []

    def frame_ref(t):
        name = f"f_{t:09.2f}.jpg"
        jobs.append((video, t, os.path.join(fdir, name), THUMB_W))
        return {"t": round(t, 2), "src": f"{tid}/{name}"}

    steps, seen = [], set()
    for it in tr["iterations"]:
        raw = it["model_response"].get("raw_text") or ""
        text = re.sub(r"```repl.*?```", "", raw, flags=re.S).strip()
        blocks = []
        for ex in it.get("repl_executions") or []:
            stdout = VLM_ECHO.sub("\n", ANSI.sub("", ex.get("stdout") or ""))
            vlm = []
            stdout_clips = [{"t0": float(m[1]), "t1": float(m[2])} for m in CLIP_NAME.finditer(stdout)]
            n_clips = 0
            for call in ex.get("sub_llm_calls") or []:
                ts = [float(m.group(1)) for p in call.get("image_paths") or [] if (m := TS_IN_NAME.search(p))]
                seen.update(round(t, 2) for t in ts)
                clip = None
                for src in [call.get("video_path") or ""] + [call.get("prompt") or ""]:
                    if m := CLIP_NAME.search(src):
                        clip = {"t0": float(m[1]), "t1": float(m[2])}
                if clip is None and not ts:  # the clip path was not logged: use the window the prompt names, else clips in printed order
                    if m := CLIP_IN_PROMPT.search(call.get("prompt") or ""):
                        clip = {"t0": float(m[1]), "t1": float(m[2])}
                    elif n_clips < len(stdout_clips):
                        clip = stdout_clips[n_clips]
                if clip:
                    n_clips += 1
                entry = {"prompt": clean(call.get("prompt"), 1500),
                         "response": clean(call.get("response"), MAX_VLM_TEXT),
                         "n_images": len(call.get("image_paths") or []),
                         "frames": [frame_ref(t) for t in ts[:MAX_FRAMES_PER_CALL]]}
                if clip:
                    span = clip["t1"] - clip["t0"]
                    entry["clip"] = {**clip, "frames": [frame_ref(clip["t0"] + span * (i + .5) / 4) for i in range(4)]}
                vlm.append(entry)
            code = ex.get("code") or ""
            blocks.append({
                "code": code,
                "calls": sorted(set(FUNC_RE.findall(code)), key=FUNCS.index),
                "stdout": clean(stdout, MAX_STDOUT),
                "stderr": clean(ex.get("stderr"), 800),
                "seconds": round(ex.get("duration") or 0, 1),
                "spans": parse_spans(stdout),
                "vlm": vlm,
            })
        steps.append({"iteration": it["iteration"], "text": text, "blocks": blocks,
                      "seconds": round(it.get("duration") or 0, 1)})

    strip = []
    for i in range(STRIP_N):
        t = info["duration"] * (i + .5) / STRIP_N
        name = f"s_{i:02d}.jpg"
        jobs.append((video, t, os.path.join(fdir, name), STRIP_W))
        strip.append({"t": round(t, 2), "src": f"{tid}/{name}"})

    list((pool or ThreadPoolExecutor(8)).map(lambda j: extract_frame(*j), jobs))

    # card thumbnail: a frame from the last VLM call, i.e. the evidence the answer rests on
    calls = [v for st in steps for b in st["blocks"] for v in b["vlm"]]
    pool_frames = [f for v in calls[::-1] for f in ((v.get("clip") or {}).get("frames") or v["frames"])]
    thumb = (pool_frames[len(pool_frames[:4]) // 2] if pool_frames else strip[len(strip) // 2])["src"]
    if thumb_t is not None:  # explicit choice from the selection file
        refs = [f for v in calls for f in v["frames"] + ((v.get("clip") or {}).get("frames") or [])] + strip
        thumb = min(refs, key=lambda f: abs(f["t"] - thumb_t))["src"]

    q_note, question, options = parse_question(md.get("query") or "")
    usage = tr.get("usage") or {}
    final = str(tr.get("final_answer") or "").strip()
    final = (re.search(r"[A-F]", final.replace("FINAL", "")) or [final])[0]
    gt = md.get("ground_truth")
    data = {
        "id": tid, "title": title or question[:80], "note": note or "",
        "dataset": qm.get("dataset"),
        "category": qm.get("question_type") or qm.get("task_type") or qm.get("question_category") or "",
        "level": qm.get("level"),
        "question": question, "question_note": q_note, "options": options,
        "ground_truth": gt, "prediction": final, "correct": final == gt,
        "models": {"llm": md.get("main_model"), "vlm": md.get("sub_model")},
        "video": {**info, "name": os.path.basename(video)},
        "filmstrip": strip,
        "stats": {
            "iterations": len(steps),
            "repl_blocks": sum(len(s["blocks"]) for s in steps),
            "vlm_calls": sum(len(b["vlm"]) for s in steps for b in s["blocks"]),
            "frames_seen": len(seen),
            "llm_tokens": (usage.get("main_llm") or {}).get("total_tokens"),
            "vlm_tokens": (usage.get("sub_llm") or {}).get("total_tokens"),
            "seconds": round(md.get("total_time") or 0, 1),
        },
        "steps": steps,
    }
    with open(os.path.join(out_dir, f"{tid}.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return {"id": tid, "title": data["title"], "note": data["note"], "dataset": data["dataset"],
            "category": data["category"], "duration": info["duration"], "correct": data["correct"],
            "iterations": data["stats"]["iterations"], "vlm_calls": data["stats"]["vlm_calls"],
            "thumb": thumb}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="*", help="*_trace.json files")
    p.add_argument("--selection", help="JSON list of {trace, video?, id?, title?, note?}")
    p.add_argument("--out", required=True)
    p.add_argument("--no-viewer", action="store_true", help="only write the data, not the viewer page")
    args = p.parse_args()

    entries = json.load(open(args.selection)) if args.selection else []
    entries += [{"trace": t} for t in args.traces]
    os.makedirs(args.out, exist_ok=True)
    pool = ThreadPoolExecutor(16)
    manifest = []
    for e in entries:
        m = export(e["trace"], args.out, video=e.get("video"), tid=e.get("id"), title=e.get("title"),
                   note=e.get("note"), thumb_t=e.get("thumb"), pool=pool)
        manifest.append(m)
        print(f"exported {m['id']}: {m['iterations']} steps, {m['vlm_calls']} VLM calls")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    if not args.no_viewer:
        viewer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
        for name in os.listdir(viewer):
            shutil.copy(os.path.join(viewer, name), args.out)
        print(f"viewer: cd {args.out} && python -m http.server  ->  http://localhost:8000/")


if __name__ == "__main__":
    main()
