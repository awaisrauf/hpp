"""
In-process VLM via HuggingFace transformers (no server needed).
Used for localized perception on a handful of frames or a short clip.

    client = HFVLMClient("Qwen/Qwen3-VL-2B-Instruct")
    client.completion("Describe this image", images_paths=["frame.jpg"])
    client.completion("Describe this clip", video_path="clip.mp4")
"""

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Loaded once per process and shared by every REPL.
_model = None
_processor = None
_model_key = None


def _load_model(model_name: str, gpu: int = 0):
    global _model, _processor, _model_key
    if _model is not None and _model_key == (model_name, gpu):
        return _model, _processor

    print(f"Loading VLM {model_name} on cuda:{gpu} ...")
    _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map=f"cuda:{gpu}", trust_remote_code=True,
    ).eval()
    _model_key = (model_name, gpu)
    return _model, _processor


def _text_from_messages(messages) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        return messages.get("content", "")
    if isinstance(messages, list):
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else " ".join(p["text"] for p in c if p.get("type") == "text")
        return ""
    return str(messages)


class HFVLMClient:
    def __init__(self, model: str = "Qwen/Qwen3-VL-2B-Instruct", max_new_tokens: int = 1024, gpu: int = 0):
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.gpu = gpu
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        _load_model(model, gpu)

    def completion(self, messages, max_tokens: Optional[int] = None, images_paths=None, video_path=None,
                   **kwargs) -> str:
        model, processor = _load_model(self.model_name, self.gpu)

        if isinstance(images_paths, str):
            images_paths = [images_paths]

        # Media first, text last (Qwen-VL convention).
        content = [{"type": "image", "image": p} for p in (images_paths or [])]
        if video_path:
            content.append({"type": "video", "video": video_path})
        content.append({"type": "text", "text": _text_from_messages(messages)})

        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens or self.max_new_tokens)
        generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]

        self.last_input_tokens = inputs.input_ids.shape[-1]
        self.last_output_tokens = len(generated[0]) if generated else 0
        return processor.batch_decode(generated, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
