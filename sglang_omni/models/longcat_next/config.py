# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LongCat-Next text-only AR backbone (Phase 1).

Architecture: ``LongcatNextTextForCausalLM``, a thin wrapper around SGLang's
``LongcatFlashForCausalLM`` that handles the lm_head vocab-size asymmetry
(131125 output tokens vs 131072 input tokens) so the complete backbone can
serve text while leaving headroom for later multimodal stages.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.longcat_next"


class LongcatNextTextPipelineConfig(PipelineConfig):
    """Single-stage batched text generation pipeline for LongCat-Next backbone."""

    architecture: ClassVar[str] = "LongcatNextTextForCausalLM"

    model_path: str
    entry_stage: str = "text"
    stages: list[StageConfig] = [
        StageConfig(
            name="text",
            process="text",
            factory=f"{_PKG}.stages.create_longcat_next_text_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
            },
            gpu=0,
            tp_size=1,
            terminal=True,
        )
    ]


EntryClass = LongcatNextTextPipelineConfig
