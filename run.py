"""
Ask HPP a question about one video.

  python run.py --video demo.mp4 --question "What does the chef add after the onions?" \
      --options "Garlic" "Tomatoes" "Salt" "Butter"

  python run.py --video lecture.mp4 --subtitles lecture.srt --question "..." --options ...

The reasoning trace (every LLM turn, REPL execution and VLM call) is written to
<traces>/<name>_trace.json; turn it into a web page with tools/export_trajectory.py.
"""
import argparse
import os
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()  # before argparse: defaults read HPP_LLM / HPP_VLM / VLM_GPU


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--options", nargs="+", required=True, help="answer options, in order (A, B, C, ...)")
    p.add_argument("--subtitles", default=None, help=".srt or LongVideoBench-style .json subtitles")
    p.add_argument("--model", default=os.getenv("HPP_LLM", "gpt-5-mini"))
    p.add_argument("--vlm", default=os.getenv("HPP_VLM", "Qwen/Qwen3-VL-2B-Instruct"))
    p.add_argument("--gpu", type=int, default=int(os.getenv("VLM_GPU", "0")))
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--traces", default="traces")
    p.add_argument("--name", default="demo")
    args = p.parse_args()

    from hpp.agent import HPP
    from hpp.benchmarks import letter, normalize_answer
    from hpp.utils.search import check_search_model
    from hpp.utils.videos import get_video_metadata
    from hpp.utils.vlm import HFVLMClient

    if not os.path.exists(args.video):
        p.error(f"video not found: {args.video}")
    check_search_model()

    opts = " ".join(f"{letter(i)}) {o}" for i, o in enumerate(args.options))
    letters = ", ".join(letter(i) for i in range(len(args.options)))
    query = f"{args.question} {opts}. Answer with just the option letter ({letters})."

    meta = get_video_metadata(args.video)
    meta["subtitle"] = "Available" if args.subtitles else "Not Available"

    agent = HPP(
        model=args.model, recursive_model=args.vlm,
        video_path=args.video, frames_metadata=meta, subtitle_path=args.subtitles,
        max_iterations=args.max_iterations, num_options=len(args.options),
        enable_logging=True, output_file_path=args.traces, file_name=args.name,
        vlm_client=HFVLMClient(model=args.vlm, gpu=args.gpu),
    )
    answer = normalize_answer(agent.completion(query=query))
    idx = ord(answer) - 65 if len(answer) == 1 else -1
    print(f"\nAnswer: {answer}" + (f") {args.options[idx]}" if 0 <= idx < len(args.options) else ""))
    print(f"Trace:  {os.path.join(args.traces, args.name + '_trace.json')}")


if __name__ == "__main__":
    main()
