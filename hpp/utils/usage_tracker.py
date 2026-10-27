"""
Token usage tracker for one question: main LLM tokens, sub LLM (VLM) tokens/calls,
search calls, REPL executions (count + total time).
"""


class UsageTracker:
    """Accumulates usage stats for a single question."""

    def __init__(self):
        self.main_llm_input_tokens = 0
        self.main_llm_output_tokens = 0
        self.main_llm_calls = 0

        self.sub_llm_input_tokens = 0
        self.sub_llm_output_tokens = 0
        self.sub_llm_calls = 0

        self.search_calls = 0

        self.repl_executions = 0
        self.repl_total_time = 0.0

    def log_main_llm(self, input_tokens, output_tokens):
        self.main_llm_input_tokens += input_tokens
        self.main_llm_output_tokens += output_tokens
        self.main_llm_calls += 1

    def log_sub_llm(self, input_tokens, output_tokens):
        self.sub_llm_input_tokens += input_tokens
        self.sub_llm_output_tokens += output_tokens
        self.sub_llm_calls += 1

    def log_search(self):
        self.search_calls += 1

    def log_repl_execution(self, execution_time):
        self.repl_executions += 1
        self.repl_total_time += execution_time

    def to_dict(self):
        return {
            "main_llm": {
                "input_tokens": self.main_llm_input_tokens,
                "output_tokens": self.main_llm_output_tokens,
                "total_tokens": self.main_llm_input_tokens + self.main_llm_output_tokens,
                "calls": self.main_llm_calls,
            },
            "sub_llm": {
                "input_tokens": self.sub_llm_input_tokens,
                "output_tokens": self.sub_llm_output_tokens,
                "total_tokens": self.sub_llm_input_tokens + self.sub_llm_output_tokens,
                "calls": self.sub_llm_calls,
            },
            "search_calls": self.search_calls,
            "repl": {
                "executions": self.repl_executions,
                "total_time_s": round(self.repl_total_time, 2),
            },
        }
