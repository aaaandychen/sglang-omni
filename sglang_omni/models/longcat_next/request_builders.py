# SPDX-License-Identifier: Apache-2.0
"""StagePayload ↔ SGLang request adapters for LongCat-Next."""

from __future__ import annotations

from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.longcat_next.payload_types import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    IMAGE_STAGE,
    LongcatNextPipelineState,
    PREPROCESSING_STAGE,
    payload_with_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


def _build_suppress_tokens(tokenizer: object) -> list[int]:
    try:
        text_vocab = int(tokenizer.vocab_size)
    except (AttributeError, TypeError):
        return []
    full_vocab = int(
        getattr(
            tokenizer,
            "text_vocab_plus_multimodal_special_token_size",
            text_vocab,
        )
    )
    if full_vocab <= text_vocab:
        return []
    return list(range(text_vocab, full_vocab))


def _extract_text(payload: StagePayload) -> str:
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
            return _messages_to_text(messages)
    if isinstance(inputs, list):
        return _messages_to_text(inputs)
    return str(inputs) if inputs else ""


def _messages_to_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for item in messages:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for piece in content:
                    if isinstance(piece, str):
                        parts.append(piece)
                    elif isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        parts.append(piece["text"])
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _payload_state(payload: StagePayload) -> LongcatNextPipelineState:
    return LongcatNextPipelineState.from_dict(payload.data)


def _active_encoder_stages(encoder_inputs: dict[str, dict[str, Any]]) -> list[str]:
    return [stage for stage in (IMAGE_STAGE, AUDIO_STAGE) if encoder_inputs.get(stage)]


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    state = _payload_state(output)
    return [*_active_encoder_stages(state.encoder_inputs), AGGREGATE_STAGE]


def resolve_mm_aggregate_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    del request_id
    if from_stage != PREPROCESSING_STAGE:
        return None
    state = _payload_state(payload)
    return [PREPROCESSING_STAGE, *_active_encoder_stages(state.encoder_inputs)]


def project_to_image_encoder(payload: StagePayload) -> StagePayload:
    return _project_to_encoder(payload, IMAGE_STAGE)


def project_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return _project_to_encoder(payload, AUDIO_STAGE)


def project_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = _payload_state(payload)
    projected = LongcatNextPipelineState(
        prompt=state.prompt,
        mm_inputs=state.mm_inputs,
        encoder_inputs={
            k: {"present": True}
            for k in _active_encoder_stages(state.encoder_inputs)
        },
    )
    return payload_with_state(payload, projected)


def project_encoder_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = _payload_state(payload)
    projected = LongcatNextPipelineState(encoder_outs=state.encoder_outs)
    return payload_with_state(payload, projected)


def _project_to_encoder(payload: StagePayload, stage_name: str) -> StagePayload:
    state = _payload_state(payload)
    inputs = state.encoder_inputs.get(stage_name) or {}
    projected = LongcatNextPipelineState(encoder_inputs={stage_name: inputs})
    return payload_with_state(payload, projected)


def _make_sampling_params(
    params: dict[str, Any],
    *,
    max_new_tokens: int,
    eos_token_id: int,
) -> tuple[SamplingParams, int, float]:
    temperature = float(params.get("temperature", 0.7))
    request_max_new_tokens = int(params.get("max_new_tokens", max_new_tokens))
    sampling_params = SamplingParams(
        max_new_tokens=request_max_new_tokens,
        temperature=temperature,
        top_p=float(params.get("top_p", 0.8)),
        top_k=int(params.get("top_k", 20)),
        stop_token_ids=[eos_token_id],
    )
    sampling_params.normalize(tokenizer=None)
    return sampling_params, request_max_new_tokens, temperature


def make_longcat_next_text_adapters(
    *,
    tokenizer: object,
    max_new_tokens: int = 4096,
) -> tuple[
    Callable[[StagePayload], SGLangARRequestData],
    Callable[[SGLangARRequestData], StagePayload],
]:
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    suppress_tokens = _build_suppress_tokens(tokenizer)

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        params: dict[str, Any] = dict(payload.request.params or {})
        state = _payload_state(payload)
        text_ar_inputs = state.text_ar_inputs or {}

        if text_ar_inputs.get("input_ids") is not None:
            input_ids = torch.as_tensor(text_ar_inputs["input_ids"], dtype=torch.long).flatten().tolist()
            origin_text = str(state.prompt.get("text") or _extract_text(payload))
        else:
            origin_text = _extract_text(payload)
            messages = [{"role": "user", "content": origin_text}]
            try:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
            except (AttributeError, ValueError):
                input_ids = tokenizer.encode(origin_text)
            input_ids = list(map(int, input_ids))

        sampling_params, request_max_new_tokens, temperature = _make_sampling_params(
            params,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        req = Req(
            rid=payload.request_id,
            origin_input_text=origin_text,
            origin_input_ids=list(map(int, input_ids)),
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )
        if suppress_tokens:
            req._codec_suppress_tokens = list(suppress_tokens)

        req_data = SGLangARRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            stage_payload=payload,
        )
        mm_inputs = text_ar_inputs.get("longcat_mm_inputs")
        if mm_inputs:
            req_data.longcat_mm_inputs = mm_inputs
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "modality": "text",
                "usage": {"output_tokens": len(output_ids)},
            },
        )

    return request_builder, result_adapter


__all__ = [
    "make_longcat_next_text_adapters",
    "resolve_preprocessing_next_stages",
    "resolve_mm_aggregate_wait_sources",
    "project_to_image_encoder",
    "project_to_audio_encoder",
    "project_to_mm_aggregate",
    "project_encoder_to_mm_aggregate",
]
