# SPDX-License-Identifier: Apache-2.0
"""Typed payload helpers for LongCat-Next Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
AGGREGATE_STAGE = "mm_aggregate"
TEXT_AR_STAGE = "text_ar"
PREPROCESSING_STAGE = "preprocessing"


@dataclass
class LongcatNextPipelineState:
    """Serializable-ish state carried between LongCat-Next stages.

    Tensors are intentionally kept as Python objects.  The relay backend is
    responsible for moving them between processes; Phase 2 MVP prioritizes
    correctness and keeps the structure explicit.
    """

    prompt: dict[str, Any] = field(default_factory=dict)
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, dict[str, Any]] = field(default_factory=dict)
    text_ar_inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.prompt:
            data["prompt"] = self.prompt
        if self.mm_inputs:
            data["mm_inputs"] = self.mm_inputs
        if self.encoder_inputs:
            data["encoder_inputs"] = self.encoder_inputs
        if self.encoder_outs:
            data["encoder_outs"] = self.encoder_outs
        if self.text_ar_inputs:
            data["text_ar_inputs"] = self.text_ar_inputs
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "LongcatNextPipelineState":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            prompt=dict(data.get("prompt") or {}),
            mm_inputs=dict(data.get("mm_inputs") or {}),
            encoder_inputs=dict(data.get("encoder_inputs") or {}),
            encoder_outs=dict(data.get("encoder_outs") or {}),
            text_ar_inputs=dict(data.get("text_ar_inputs") or {}),
        )


def payload_with_state(payload: Any, state: LongcatNextPipelineState):
    from sglang_omni.proto import StagePayload

    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )
