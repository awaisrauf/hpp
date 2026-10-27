"""
Client for the reasoning LLM: any OpenAI-compatible chat-completions endpoint
(OpenAI, MiniMax, a local vLLM / SGLang server, ...). Text only — the LLM never
sees pixels; perception goes through the VLM inside the REPL.

Configure with env vars (or a .env file):
  OPENAI_API_KEY   API key for the endpoint ("EMPTY" is fine for local servers)
  MAIN_LLM_URL     base URL, default https://api.openai.com/v1
"""
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _is_minimax_model(model: str) -> bool:
    return model.lower().startswith("minimax")


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = "gpt-5-mini"):
        self.model = model
        self.base_url = base_url or os.getenv("MAIN_LLM_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
                             base_url=self.base_url)
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def completion(self, messages: list[dict[str, str]] | str, max_tokens: Optional[int] = None,
                   timeout: Optional[float] = None, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        req = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if timeout is not None:
            req["timeout"] = timeout
        if _is_minimax_model(self.model):
            req["tool_choice"] = "none"  # plain text responses only
            req["tools"] = []

        response = self.client.chat.completions.create(**req)

        usage = getattr(response, "usage", None)
        self.last_input_tokens = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        self.last_output_tokens = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        if not response.choices:
            return "[LLM returned no response]"
        content = response.choices[0].message.content
        if content is None:
            return "[LLM returned empty response]"
        if "Thinking" in self.model and "</think>" in content:
            return content.split("</think>")[1].strip()
        return content
