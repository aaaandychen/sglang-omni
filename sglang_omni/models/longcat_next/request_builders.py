# SPDX-License-Identifier: Apache-2.0
"""StagePayload ↔ SGLang request adapters for LongCat-Next text backbone."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


def _build_suppress_tokens(tokenizer: object) -> list[int]:
    """Return token ids to suppress during text-only generation.

    LongCat-Next uses token ids in [text_vocab_size,
    text_vocab_plus_multimodal_special_token_size) for visual/audio tokens.
    Suppressing them prevents the model from emitting modality tokens when
    only text output is expected.
    """
    try:
        text_vocab = int(tokenizer.vocab_size)  # typically 131 072
    except (AttributeError, TypeError):
        return []
    try:
        full_vocab = int(getattr(tokenizer, "vocab_size", text_vocab))
    except (AttributeError, TypeError):
        return []
    if full_vocab <= text_vocab:
        return []
    # Suppress everything beyond the text vocabulary.
    return list(range(text_vocab, full_vocab))


def _extract_text(payload: StagePayload) -> str:
    """Pull the input text from a StagePayload.

    Accepts ``payload.request.inputs`` as a plain string, or a dict with
    any of the keys ``"text"`` / ``"prompt"`` / ``"messages"``.
    """
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("text", "prompt"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value
        messages = inputs.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            return "\n".join(parts)
    if isinstance(inputs, list):
        parts = []
        for item in inputs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("content", ""))
        return "\n".join(parts)
    return str(inputs) if inputs else ""


def make_longcat_next_text_adapters(
    *,
    tokenizer: object,
    max_new_tokens: int = 4096,
) -> tuple[
    Callable[[StagePayload], SGLangARRequestData],
    Callable[[SGLangARRequestData], StagePayload],
]:
    """Build (request_builder, result_adapter) for text-only generation.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer for the LongCat-Next model.
    max_new_tokens:
        Default per-request generation budget.
    """
    eos_token_id: int = int(tokenizer.eos_token_id)
    pad_token_id: int = int(getattr(tokenizer, "pad_token_id", eos_token_id) or eos_token_id)
    vocab_size: int = int(tokenizer.vocab_size)
    suppress_tokens: list[int] = _build_suppress_tokens(tokenizer)

    # ── request_builder ────────────────────────────────────────────────

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        params: dict[str, Any] = dict(payload.request.params or {})
        text: str = _extract_text(payload)

        # Tokenize
        messages = [{"role": "user", "content": text}]
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # BatchEncoding is not a dict subclass in transformers 5.x
            if hasattr(tokenized, "input_ids"):
                input_ids = tokenized.input_ids
            else:
                input_ids = tokenized
        except (AttributeError, ValueError):
            input_ids = tokenizer.encode(text)
        input_ids = list(map(int, input_ids))

        # Sampling params
        temperature: float = float(params.get("temperature", 0.7))
        request_max_new_tokens: int = int(
            params.get("max_new_tokens", max_new_tokens)
        )
        stop_token_ids: list[int] = [eos_token_id]
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=float(params.get("top_p", 0.8)),
            top_k=int(params.get("top_k", 20)),
            stop_token_ids=stop_token_ids,
        )
        sampling_params.normalize(tokenizer=None)

        # SGLang request object
        req = Req(
            rid=payload.request_id,
            origin_input_text=text,
            origin_input_ids=list(input_ids),
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )
        if suppress_tokens:
            req._codec_suppress_tokens = list(suppress_tokens)

        return SGLangARRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            stage_payload=payload,
        )

    # ── result_adapter ─────────────────────────────────────────────────

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids: list[int] = list(data.output_ids or [])
        text: str = tokenizer.decode(output_ids, skip_special_tokens=True)

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "modality": "text",
                "usage": {
                    "output_tokens": len(output_ids),
                },
            },
        )

    return request_builder, result_adapter


__all__ = ["make_longcat_next_text_adapters"]
