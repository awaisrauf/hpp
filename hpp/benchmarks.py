"""
Benchmark loaders. Each loader yields `Item`s with a uniform interface; question
formatting and answer parsing match the paper's evaluation exactly.

Expected layouts (as downloaded from HuggingFace):

  egoschema/       Subset/test-00000-of-00001.parquet, videos/<video_idx>.mp4
  longvideobench/  lvb_val.json, videos/, subtitles/
  videomme/        videomme/test-00000-of-00001.parquet, data/<videoID>.mp4, subtitle/<videoID>.srt
  mlvu/            json/<n>_<task>.json, any subfolder with the .mp4 files (e.g. video/<task>/)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

DEFAULT_CONTEXT = "Analyze the video to answer the question."


@dataclass
class Item:
    id: str
    video_path: str
    query: str
    ground_truth: str
    num_options: int
    group: str                       # used for per-split accuracy (e.g. lvb duration group)
    subtitle_path: Optional[str] = None
    context: str = DEFAULT_CONTEXT
    ground_truth_text: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def file_name(self):
        return re.sub(r"[^\w.\-]", "_", f"{self.group}_{self.id}")


def letter(i: int) -> str:
    return chr(65 + i)


def normalize_answer(answer, valid: str = "ABCDEF") -> str:
    """Extract the option letter from the agent's final answer."""
    if answer is None:
        return ""
    a = str(answer).strip().upper()
    for prefix in ["THE ANSWER IS", "ANSWER:", "FINAL ANSWER:", "FINAL(", "FINAL"]:
        if a.startswith(prefix):
            a = a[len(prefix):].strip()
    a = a.rstrip(")").replace('"', "").replace("'", "").replace("`", "")
    return a[0] if a and a[0] in valid else a.strip()


def _lettered_question(question, candidates):
    opts = " ".join(f"{letter(i)}) {c}" for i, c in enumerate(candidates))
    letters = ", ".join(letter(i) for i in range(len(candidates)))
    return f"{question} {opts}. Answer with just the option letter ({letters})."


# ── EgoSchema (500-question subset) ──────────────────────────────────────

def load_egoschema(root) -> Iterator[Item]:
    import pandas as pd
    df = pd.read_parquet(os.path.join(root, "Subset", "test-00000-of-00001.parquet"))
    for _, row in df.iterrows():
        options = list(row["option"])
        yield Item(
            id=row["question_idx"],
            video_path=os.path.join(root, "videos", f"{row['video_idx']}.mp4"),
            query=("c in this question is camera wearer." + row["question"] + " " + " ".join(options)
                   + " Answer with just the option letter (A, B, C, D, or E)."),
            ground_truth=letter(int(row["answer"])),
            num_options=5,
            group="ego",
            metadata={"dataset": "EgoSchema", "question_idx": str(row["question_idx"]),
                      "video_idx": row["video_idx"]},
        )


# ── LongVideoBench (validation set, with subtitles) ──────────────────────

LVB_DURATION_GROUPS = [15, 60, 600, 3600]


def load_longvideobench(root, duration_groups=None) -> Iterator[Item]:
    import json
    with open(os.path.join(root, "lvb_val.json")) as f:
        data = json.load(f)
    groups = duration_groups or LVB_DURATION_GROUPS
    for dg in groups:
        for it in data:
            if it.get("duration_group") != dg:
                continue
            sub = os.path.join(root, "subtitles", it["subtitle_path"]) if it.get("subtitle_path") else None
            yield Item(
                id=it["id"],
                video_path=os.path.join(root, "videos", it["video_path"]),
                subtitle_path=sub if sub and os.path.exists(sub) else None,
                query=_lettered_question(it["question"], it["candidates"]),
                ground_truth=letter(it["correct_choice"]),
                ground_truth_text=it["candidates"][it["correct_choice"]],
                num_options=len(it["candidates"]),
                group=f"lvb_{dg}",
                metadata={"dataset": "LongVideoBench", "duration": it.get("duration"),
                          "duration_group": dg, "level": it.get("level"),
                          "topic_category": it.get("topic_category"),
                          "question_category": it.get("question_category"), "item_id": it["id"]},
            )


# ── Video-MME (with subtitles) ───────────────────────────────────────────

VMME_DURATIONS = ["short", "medium", "long"]


def _parse_srt(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    subs = []
    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3 or not lines[0].strip().isdigit():
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", lines[1].strip())
        if not m:
            continue
        text = re.sub(r"<[^>]+>", "", " ".join(lines[2:])).strip()
        if text and text not in ("♪ ♪", "♪"):
            subs.append({"start": m.group(1), "text": text})
    return subs


def _subtitle_context(subs, max_chars=4000):
    """Short transcript placed in the REPL `context` variable (as in the paper runs)."""
    lines, total = [], 0
    for s in subs:
        line = f"[{s['start'][:8]}] {s['text']}"
        if total + len(line) > max_chars:
            lines.append("... (subtitles truncated)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def load_videomme(root, durations=None) -> Iterator[Item]:
    import pandas as pd
    df = pd.read_parquet(os.path.join(root, "videomme", "test-00000-of-00001.parquet"))
    for dur in durations or VMME_DURATIONS:
        for _, row in df[df["duration"] == dur].iterrows():
            vid = row["videoID"]
            srt = os.path.join(root, "subtitle", f"{vid}.srt")
            subs = _parse_srt(srt) if os.path.exists(srt) else []
            context = DEFAULT_CONTEXT
            if subs:
                context += f"\n\nVideo subtitles/transcript (with timestamps):\n{_subtitle_context(subs)}"
            yield Item(
                id=row["question_id"],
                video_path=os.path.join(root, "data", f"{vid}.mp4"),
                subtitle_path=srt if subs else None,
                query=row["question"] + " " + " ".join(list(row["options"]))
                      + " Answer with just the option letter (A, B, C, or D).",
                ground_truth=row["answer"],
                num_options=4,
                group=f"vmme_{dur}",
                context=context,
                metadata={"dataset": "VideoMME", "duration": dur, "domain": row["domain"],
                          "sub_category": row["sub_category"], "task_type": row["task_type"],
                          "question_id": row["question_id"], "video_id": vid},
            )


# ── MLVU (dev set, 7 multiple-choice tasks, M-Avg) ───────────────────────

MLVU_TASKS = ["plotQA", "needle", "ego", "count", "order", "anomaly_reco", "topic_reasoning"]
_MLVU_TYPE = {"needle": "findNeedle"}


def load_mlvu(root, tasks=None) -> Iterator[Item]:
    import json
    vid_map = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(".mp4"):
                vid_map[f] = os.path.join(r, f)

    json_dir = os.path.join(root, "json")
    for task in tasks or MLVU_TASKS:
        qtype = _MLVU_TYPE.get(task, task)
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(json_dir, fname)) as f:
                items = json.load(f)
            if not items or "candidates" not in items[0] or items[0].get("question_type") != qtype:
                continue
            stem = fname[:-5]
            for idx, it in enumerate(items):
                cands = it["candidates"]
                gt = next((letter(i) for i, c in enumerate(cands) if c == it["answer"]), None) or \
                     next((letter(i) for i, c in enumerate(cands)
                           if str(c).strip().lower() == str(it["answer"]).strip().lower()), "?")
                qid = f"{stem}_{idx:04d}"
                yield Item(
                    id=qid,
                    video_path=vid_map.get(it["video"], os.path.join(root, "video", it["video"])),
                    query=_lettered_question(it["question"], cands),
                    ground_truth=gt,
                    ground_truth_text=str(it["answer"]),
                    num_options=len(cands),
                    group=f"mlvu_{task}",
                    metadata={"dataset": "MLVU", "question_type": it.get("question_type"),
                              "duration": it.get("duration"), "question_id": qid, "video": it["video"]},
                )


LOADERS = {
    "egoschema": load_egoschema,
    "longvideobench": load_longvideobench,
    "videomme": load_videomme,
    "mlvu": load_mlvu,
}

# Letters accepted by the answer parser, per benchmark (as in the original scripts).
VALID_LETTERS = {"egoschema": "ABCDE", "longvideobench": "ABCDE", "videomme": "ABCD", "mlvu": "ABCDEF"}
