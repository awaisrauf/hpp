"""
HPP agent loop: a coding LLM iteratively writes ```repl``` code that probes the
video (hierarchy, search, sampling) and calls a small VLM for perception, until it
commits to FINAL(<letter>).
"""

import re
import signal
import threading
import time
from typing import Optional

import hpp.utils.utils as utils
from hpp.prompt import DEFAULT_QUERY, build_system_prompt, next_action_prompt
from hpp.repl import REPLEnv
from hpp.utils.llm import OpenAIClient
from hpp.utils.trace_logger import TraceLogger
from hpp.utils.usage_tracker import UsageTracker

ITERATION_BUDGET = 120  # seconds of REPL execution allowed per iteration


def _is_main_thread():
    return threading.current_thread() is threading.main_thread()


class _IterationTimeout(BaseException):
    """Raised by SIGALRM when an iteration exceeds its budget. Subclasses
    BaseException so the catch-all in REPLEnv.code_execution does not swallow it."""


def _timeout_handler(signum, frame):
    raise _IterationTimeout()


class HPP:
    """Hierarchical Programmatic Probing over one video."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-5-mini",
                 recursive_model: str = "Qwen/Qwen3-VL-2B-Instruct",
                 video_path: Optional[str] = None,
                 frames_metadata: Optional[dict] = None,
                 subtitle_path: Optional[str] = None,
                 max_iterations: int = 10,
                 enable_logging: bool = False,
                 output_file_path: str = None,
                 file_name: str = None,
                 ground_truth: str = None,
                 ground_truth_other: str = None,
                 vlm_client=None,
                 num_options: int = 4,
                 gpu_lock=None,
                 extra_metadata: dict = None,
                 search_threshold: float = None,
                 base_url: Optional[str] = None,
                 verbose: bool = True):
        self.model = model
        self.recursive_model = recursive_model
        self.num_options = num_options
        self.llm = OpenAIClient(api_key, base_url=base_url, model=model)
        self.vlm_client = vlm_client
        self.gpu_lock = gpu_lock
        self.search_threshold = search_threshold
        self.extra_metadata = extra_metadata or {}
        self.ground_truth = ground_truth
        self.ground_truth_other = ground_truth_other
        self._max_iterations = max_iterations

        self.video_path = video_path
        self.frame_metadata = frames_metadata
        self.subtitle_path = subtitle_path

        self.repl_env = None
        self.messages = []
        self.query = None
        self.usage_tracker = UsageTracker()

        # verbose=True is what the paper runs used. Note that the logger's per-VLM-call summary is
        # printed while the REPL captures stdout, so it also becomes part of the LLM's REPL output.
        self.trace_logger = TraceLogger(
            output_dir=output_file_path or "./traces",
            file_name=file_name or f"run_{int(time.time())}",
            enabled=enable_logging,
            terminal_output=verbose,
        )

    def setup_context(self, context, query: Optional[str] = None):
        self.query = query = query or DEFAULT_QUERY
        self.messages = build_system_prompt(video_metadata=self.frame_metadata)

        self.trace_logger.start_session({
            "query": query,
            "main_model": self.model,
            "sub_model": self.recursive_model,
            "video_path": self.video_path,
            "system_prompt": "\n".join(m.get("content", "") for m in self.messages),
            "max_iterations": self._max_iterations,
            "ground_truth": self.ground_truth,
            "ground_truth_other": self.ground_truth_other,
            "question_metadata": self.extra_metadata or None,
        })

        context_data, context_str = utils.convert_context_for_repl(context)
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            video_path=self.video_path,
            frame_metadata=self.frame_metadata,
            subtitle_path=self.subtitle_path,
            recursive_model=self.recursive_model,
            trace_logger=self.trace_logger,
            vlm_client=self.vlm_client,
            usage_tracker=self.usage_tracker,
            gpu_lock=self.gpu_lock,
            search_threshold=self.search_threshold,
        )
        return self.messages

    def completion(self, context="Analyze the video to answer the question.", query: Optional[str] = None) -> str:
        """Answer `query` about the video. Returns the option letter (or raw text on failure)."""
        self.usage_tracker = UsageTracker()
        self.messages = self.setup_context(context, query)

        for iteration in range(self._max_iterations):
            iter_start = time.time()
            self.trace_logger.start_iteration(iteration)
            self.trace_logger._current_iteration = iteration

            self.messages.append(next_action_prompt(query, iteration, video_metadata=self.frame_metadata,
                                                    num_options=self.num_options))
            self.trace_logger.log_input_messages(iteration, self.messages)

            response_w_reasoning = self.llm.completion(self.messages)
            self.usage_tracker.log_main_llm(self.llm.last_input_tokens, self.llm.last_output_tokens)
            response = re.sub(r'<think>.*?</think>', '', response_w_reasoning, flags=re.DOTALL).strip()

            code_blocks = utils.find_code_blocks(response)
            has_code = bool(code_blocks)
            self.trace_logger.log_model_response(iteration, response, has_code, code_blocks,
                                                 raw_with_reasoning=response_w_reasoning)
            self.messages.append({"role": "assistant", "content": response})

            if has_code:
                self._run_code_blocks(iteration, iter_start, code_blocks)

            final_answer = utils.check_for_final_answer(response, self.repl_env)
            is_valid = self._is_answer_correct_format(final_answer)
            self.trace_logger.end_iteration(iteration, final_answer is not None and is_valid)

            action, message = self._check_answer_status(final_answer, is_valid, iteration)
            if action == "return":
                self.trace_logger.end_session(final_answer, status="completed", usage=self.usage_tracker.to_dict())
                return final_answer
            if action == "prompt" and message:
                self.messages.append({"role": "assistant", "content": message})

        # Out of iterations: ask once more for a final answer.
        self.messages.append(next_action_prompt(query, iteration, final_answer=True, num_options=self.num_options))
        response = self.llm.completion(self.messages)
        self.usage_tracker.log_main_llm(self.llm.last_input_tokens, self.llm.last_output_tokens)

        extracted = utils.check_for_final_answer(response, self.repl_env)
        final_answer = extracted if extracted and self._is_answer_correct_format(extracted) else response
        self.trace_logger.end_session(final_answer, status="completed", usage=self.usage_tracker.to_dict())
        return final_answer

    def _run_code_blocks(self, iteration, iter_start, code_blocks):
        use_alarm = _is_main_thread()  # signals only work in the main thread
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(ITERATION_BUDGET)
        exec_num = 0
        try:
            for code in code_blocks:
                execution = self.trace_logger.start_repl_execution(iteration, code)
                exec_num = execution.get("execution_number", 0)
                self.trace_logger._current_execution = exec_num

                formatted_result, result = utils.execute_code(self.repl_env, code)
                self.messages = utils.add_execution_result_to_messages(self.messages, code, formatted_result)
                self.usage_tracker.log_repl_execution(result.execution_time if result else 0)

                if result is not None:
                    self.trace_logger.end_repl_execution(iteration, exec_num, result.stdout, result.stderr,
                                                         result.execution_time)
                    if result.stderr:
                        self.trace_logger.log_repl_error(iteration, exec_num, code, result.stderr)
                else:
                    self.trace_logger.end_repl_execution(iteration, exec_num, "", formatted_result, 0)
                    self.trace_logger.log_repl_error(iteration, exec_num, code, formatted_result)
        except _IterationTimeout:
            elapsed = time.time() - iter_start
            msg = (f"TIMEOUT: Iteration {iteration} exceeded {ITERATION_BUDGET}s budget "
                   f"({elapsed:.0f}s). Too many calls to VLM.")
            print(msg)
            self.messages = utils.add_execution_result_to_messages(self.messages, "# timed out", msg)
            self.trace_logger.log_repl_error(iteration, exec_num, "# timed out", msg)
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    # ------------------------------------------------------------------
    def _option_letters(self):
        return [chr(65 + i) for i in range(self.num_options)]

    def _options_str(self):
        parts = [f'FINAL("{l}")' for l in self._option_letters()]
        return ", ".join(parts[:-1]) + ", or " + parts[-1]

    def _is_answer_correct_format(self, final_answer):
        if not final_answer:
            return False
        answer = str(final_answer).strip().strip('"').strip("'").strip().upper()
        return answer in self._option_letters()

    def _check_answer_status(self, final_answer, is_valid, iteration):
        """Return (action, message) with action in {"return", "prompt", "continue"}."""
        is_second_last = iteration == self._max_iterations - 2
        is_last = iteration == self._max_iterations - 1

        if final_answer and is_valid:
            return "return", None
        if is_last and final_answer:
            return "return", None
        if is_second_last:
            return "prompt", ("IMPORTANT: Next iteration is your LAST. Aggregate all your analysis "
                              f"and respond with {self._options_str()}.")
        if final_answer and not is_valid:
            letters = "/".join(self._option_letters())
            return "prompt", f"Wrong format. Select {letters} based on your analysis and use FINAL(\"X\")."
        return "continue", None
