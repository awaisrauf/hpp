"""
Replay a saved trace through the agent with a scripted LLM (no API key needed).

The LLM turns come from the trace; everything else (REPL, probing functions,
search, VLM) runs for real. Checks that every code block executes without new
errors and that the agent reaches the same FINAL answer. Useful as an install check.

  python tests/replay_trace.py examples/traces/<name>_trace.json [--fake-vlm]
"""
import argparse
import warnings
import json
import os
import re
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ScriptedLLM:
    def __init__(self, turns, final):
        self.turns = list(turns)
        self.final = final  # used if the agent asks for a forced final answer
        self.last_input_tokens = self.last_output_tokens = 0

    def completion(self, messages, **kwargs):
        return self.turns.pop(0) if self.turns else self.final


class FakeVLM:
    last_input_tokens = last_output_tokens = 0

    def completion(self, messages, images_paths=None, video_path=None, **kwargs):
        return f"[fake VLM: {len(images_paths or [])} images]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--video", default=None, help="override the video path stored in the trace")
    p.add_argument("--subtitles", default=None)
    p.add_argument("--fake-vlm", action="store_true", help="skip loading the VLM")
    p.add_argument("--vlm", default="Qwen/Qwen3-VL-2B-Instruct")
    args = p.parse_args()

    trace = json.load(open(args.trace))
    md = trace["metadata"]
    turns = [it["model_response"]["raw_text"] for it in trace["iterations"]]
    orig_errors = sum(bool((ex.get("stderr") or "").strip())
                      for it in trace["iterations"] for ex in it.get("repl_executions") or [])

    from hpp.agent import HPP
    from hpp.benchmarks import normalize_answer
    from hpp.utils.videos import get_video_metadata
    if "Subtitles: Available" in (md.get("system_prompt") or "") and not args.subtitles:
        print("warning: this trace used subtitles; pass --subtitles or subtitle-based blocks will fail")
    video = args.video or md["video_path"]
    meta = get_video_metadata(video)
    meta["subtitle"] = "Available" if args.subtitles else "Not Available"
    if args.fake_vlm:
        vlm = FakeVLM()
    else:
        from hpp.utils.vlm import HFVLMClient
        vlm = HFVLMClient(args.vlm)

    m = re.search(r"option letter \(([^)]*)\)", md["query"])
    num_options = len(re.findall(r"\b[A-F]\b", m.group(1))) if m else 5
    agent = HPP(video_path=video, frames_metadata=meta, subtitle_path=args.subtitles,
                vlm_client=vlm, num_options=num_options, max_iterations=md.get("max_iterations", 10),
                api_key="unused", verbose=False)
    agent.llm = ScriptedLLM(turns, f"FINAL({normalize_answer(trace['final_answer'])})")

    import hpp.utils.utils as U
    stderrs, run = [], U.execute_code

    def recording_execute(repl_env, code, *a, **k):
        formatted, result = run(repl_env, code, *a, **k)
        stderrs.append(result.stderr.strip() if result is not None else formatted)
        return formatted, result
    U.execute_code = recording_execute

    answer = agent.completion(query=md["query"])
    new_errors = [e for e in stderrs if e]
    n_exec = len(stderrs)
    ok = normalize_answer(answer) == normalize_answer(trace["final_answer"])
    print(f"{os.path.basename(args.trace)}: executed {n_exec} blocks | errors original={orig_errors} "
          f"replay={len(new_errors)} | answer {answer!r} vs original {trace['final_answer']!r} -> "
          f"{'OK' if ok and len(new_errors) <= orig_errors else 'MISMATCH'}")
    for e in new_errors[:3]:
        print("   ", e[-300:].replace("\n", " | "))
    sys.exit(0 if ok and len(new_errors) <= orig_errors else 1)


if __name__ == "__main__":
    main()
