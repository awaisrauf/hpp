"""
Evaluate HPP on a long-video benchmark.

  python eval.py --benchmark longvideobench --data /data/LongVideoBench --out runs/lvb
  python eval.py --benchmark videomme --data /data/Video-MME --splits long --shard 0/3
  python eval.py --benchmark mlvu --data /data/MLVU --splits count needle --limit 20

Results are appended to <out>/results.jsonl (safe to interrupt and re-run: finished
ids are skipped), one reasoning trace per question goes to <out>/traces/, and
<out>/summary.json holds accuracy overall and per split.
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()  # before argparse: defaults read HPP_LLM / HPP_VLM / VLM_GPU

from hpp.benchmarks import LOADERS, VALID_LETTERS, normalize_answer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    p.add_argument("--data", required=True, help="dataset root (see hpp/benchmarks.py for the layout)")
    p.add_argument("--out", default=None, help="output dir (default: runs/<benchmark>)")
    p.add_argument("--splits", nargs="*", default=None,
                   help="longvideobench: 15 60 600 3600 | videomme: short medium long | mlvu: task names")
    p.add_argument("--model", default=os.getenv("HPP_LLM", "gpt-5-mini"), help="reasoning LLM")
    p.add_argument("--vlm", default=os.getenv("HPP_VLM", "Qwen/Qwen3-VL-2B-Instruct"), help="perception VLM")
    p.add_argument("--gpu", type=int, default=int(os.getenv("VLM_GPU", "0")))
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke tests)")
    p.add_argument("--shard", default=None, help="i/n: run every n-th question starting at i")
    return p.parse_args()


def load_items(args):
    loader = LOADERS[args.benchmark]
    if args.splits and args.benchmark == "longvideobench":
        items = loader(args.data, [int(s) for s in args.splits])
    elif args.splits:
        items = loader(args.data, args.splits)
    else:
        items = loader(args.data)
    items = list(items)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        items = items[i::n]
    if args.limit:
        items = items[:args.limit]
    return items


def clear_video_caches(video_path):
    from hpp.utils import hierarchy
    hierarchy.clear_cache(video_path)
    if "hpp.utils.search" in sys.modules:
        sys.modules["hpp.utils.search"].clear_cache(video_path)
    import torch
    torch.cuda.empty_cache()


def summarize(records, path):
    def acc(rs):
        c = sum(r["is_correct"] for r in rs)
        return {"total": len(rs), "correct": c, "accuracy": round(100 * c / len(rs), 2) if rs else 0.0}

    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)
    summary = {"overall": acc(records), "by_split": {g: acc(rs) for g, rs in sorted(by_group.items())}}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    out = args.out or os.path.join("runs", args.benchmark)
    trace_dir = os.path.join(out, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    results_path = os.path.join(out, "results.jsonl")

    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r

    items = load_items(args)
    todo = [it for it in items if it.id not in done]
    print(f"{args.benchmark}: {len(items)} questions, {len(items) - len(todo)} already done, {len(todo)} to run")
    print(f"LLM: {args.model} | VLM: {args.vlm} (cuda:{args.gpu}) | output: {out}")

    from hpp.agent import HPP
    from hpp.utils.search import check_search_model
    from hpp.utils.videos import get_video_metadata
    from hpp.utils.vlm import HFVLMClient
    if todo:
        check_search_model()
    vlm_client = HFVLMClient(model=args.vlm, gpu=args.gpu) if todo else None

    t0 = time.time()
    for k, item in enumerate(todo, 1):
        record = {"id": item.id, "group": item.group, "video_path": item.video_path,
                  "ground_truth": item.ground_truth, **item.metadata}
        if not os.path.exists(item.video_path):
            record.update(model_answer="ERROR: video not found", prediction="", is_correct=False, status="failed")
        else:
            meta = get_video_metadata(item.video_path)
            meta["subtitle"] = "Available" if item.subtitle_path else "Not Available"
            agent = HPP(
                model=args.model, recursive_model=args.vlm,
                video_path=item.video_path, frames_metadata=meta, subtitle_path=item.subtitle_path,
                max_iterations=args.max_iterations, num_options=item.num_options,
                enable_logging=True, output_file_path=trace_dir, file_name=item.file_name,
                ground_truth=item.ground_truth, ground_truth_other=item.ground_truth_text,
                extra_metadata=item.metadata, vlm_client=vlm_client,
            )
            try:
                answer, status, error = agent.completion(context=item.context, query=item.query), "success", None
            except Exception as e:
                answer, status, error = f"ERROR: {e}", "failed", str(e)
                agent.trace_logger.end_session(error=error, status="failed", usage=agent.usage_tracker.to_dict())
            pred = normalize_answer(answer, VALID_LETTERS[args.benchmark])
            record.update(model_answer=answer, prediction=pred, status=status, error=error,
                          is_correct=status == "success" and pred == item.ground_truth,
                          usage=agent.usage_tracker.to_dict())
            clear_video_caches(item.video_path)

        done[item.id] = record
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        n_correct = sum(r["is_correct"] for r in done.values())
        rate = k / (time.time() - t0) * 3600
        print(f"[{k}/{len(todo)}] {item.id}: pred={record['prediction'] or '-'} gt={item.ground_truth} "
              f"| acc {100 * n_correct / len(done):.1f}% ({n_correct}/{len(done)}) | ~{rate:.0f} q/h")

    ids = {it.id for it in items}
    summary = summarize([r for i, r in done.items() if i in ids], os.path.join(out, "summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
