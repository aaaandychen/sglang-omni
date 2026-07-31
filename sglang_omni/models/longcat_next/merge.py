# SPDX-License-Identifier: Apache-2.0
"""Fan-in merge functions for LongCat-Next Phase 2."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.longcat_next.payload_types import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    LongcatNextPipelineState,
    PREPROCESSING_STAGE,
    longcat_timing,
    payload_with_state,
)
from sglang_omni.pipeline.tensor_ref import is_tensor_ref_dict, tensor_ref_numel
from sglang_omni.proto import StagePayload


def _state(payload: StagePayload | None) -> LongcatNextPipelineState:
    return LongcatNextPipelineState.from_dict(payload.data if payload else {})


def _non_empty(value: Any) -> bool:
    """True when *value* is a non-empty tensor or TensorRef."""
    if value is None:
        return False
    if is_tensor_ref_dict(value):
        return tensor_ref_numel(value) > 0
    if hasattr(value, "numel"):
        return value.numel() > 0
    return bool(value)



def merge_for_text_ar(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate preprocessing + encoder outputs into text_ar payload."""
    with longcat_timing("merge_for_text_ar", num_sources=len(payloads)):
        return _merge_for_text_ar(payloads)


def _merge_for_text_ar(payloads: dict[str, StagePayload]) -> StagePayload:
    pre_payload = payloads[PREPROCESSING_STAGE]
    pre = _state(pre_payload)
    image = _state(payloads.get(IMAGE_STAGE))
    audio = _state(payloads.get(AUDIO_STAGE))

    input_ids = pre.prompt.get("input_ids")
    if input_ids is None:
        input_ids = pre.mm_inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("LongCat-Next merge missing input_ids from preprocessing")

    # encoder states are projected via project_encoder_to_mm_aggregate, which
    # wraps encoder_outs in {IMAGE_STAGE/AUDIO_STAGE: {...}}, so the path is
    # image.encoder_outs[IMAGE_STAGE] — the outer key matches the inner key.
    image_out = image.encoder_outs.get(IMAGE_STAGE, {})
    audio_out = audio.encoder_outs.get(AUDIO_STAGE, {})
    _dbg_ve = image_out.get("visual_embeds")
    _logger = __import__("logging").getLogger(__name__)
    _logger.warning(
        "merge_for_text_ar image_out.visual_embeds type=%s is_tensor_ref=%s numel=%s",
        type(_dbg_ve).__name__,
        is_tensor_ref_dict(_dbg_ve),
        tensor_ref_numel(_dbg_ve) if is_tensor_ref_dict(_dbg_ve) else getattr(_dbg_ve, "numel", lambda: "N/A")(),
    )

    image_positions = pre.mm_inputs.get("image_positions")
    audio_positions = pre.mm_inputs.get("audio_positions")

    longcat_mm_inputs: dict[str, Any] = {}
    if image_out.get("visual_embeds") is not None and image_positions is not None:
        longcat_mm_inputs["image_embeds"] = image_out["visual_embeds"]
        longcat_mm_inputs["image_positions"] = image_positions
    if audio_out.get("audio_embeds") is not None and audio_positions is not None:
        longcat_mm_inputs["audio_embeds"] = audio_out["audio_embeds"]
        longcat_mm_inputs["audio_positions"] = audio_positions

    # Keep these fields explicit for future audio-text generation compatibility.
    if pre.mm_inputs.get("audiotext_start_positions") is not None:
        longcat_mm_inputs["audio_text_positions"] = pre.mm_inputs.get(
            "audiotext_start_positions"
        )
    if pre.mm_inputs.get("audiotext_pad_positions") is not None:
        longcat_mm_inputs["audio_text_zero_positions"] = pre.mm_inputs.get(
            "audiotext_pad_positions"
        )

    text_ar_inputs = {
        "input_ids": input_ids,
        "sampling_params": dict(pre.prompt.get("sampling_params") or {}),
        "longcat_mm_inputs": longcat_mm_inputs,
    }

    merged = LongcatNextPipelineState(
        prompt=pre.prompt,
        mm_inputs=pre.mm_inputs,
        text_ar_inputs=text_ar_inputs,
    )
    return payload_with_state(pre_payload, merged)
