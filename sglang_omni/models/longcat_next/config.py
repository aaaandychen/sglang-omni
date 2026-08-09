# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LongCat-Next.

Phase 1 keeps a single text-only AR stage.  Phase 2 adds multimodal input
understanding with separate image/audio encoder stages and a lightweight
aggregate stage before the AR backbone.
"""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.longcat_next"


def _audio_output_enabled() -> bool:
    """Return ``True`` when Phase 3 audio output is enabled."""
    return os.getenv(
        "SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT", ""
    ).lower() in ("1", "true", "yes", "on")


def _text_ar_stage(
    *,
    name: str = "text",
    gpu: list[int] | None = None,
    terminal: bool = True,
    next: list[str] | None = None,
    stream_to: list[str] | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        process=name,
        factory=f"{_PKG}.stages.create_longcat_next_text_executor",
        factory_args={
            "device": "cuda:0",
            "max_running_requests": 32,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
        },
        gpu=[0, 1, 2, 3] if gpu is None else gpu,
        tp_size=4,
        terminal=terminal,
        **(dict(next=next) if next else {}),
        **(dict(stream_to=stream_to) if stream_to else {}),
    )


class LongcatNextTextPipelineConfig(PipelineConfig):
    """Single-stage batched text generation pipeline for LongCat-Next backbone."""

    architecture: ClassVar[str] = "LongcatNextTextForCausalLM"

    model_path: str
    entry_stage: str = "text"
    stages: list[StageConfig] = Field(
        default_factory=lambda: [_text_ar_stage()]
    )


class LongcatNextPipelineConfig(PipelineConfig):
    """Phase 2 multimodal-input pipeline: image/audio/text in, text out."""

    architecture: ClassVar[str] = "LongcatNextTextForCausalLM"

    model_path: str
    entry_stage: str = "preprocessing"

    # TensorRef lazy relay: visual_embeds / audio_embeds bypass mm_aggregate
    # so the CPU-side aggregate stage never materializes large tensors.
    # Refs are created on the encoder→mm_aggregate hop with
    # consumer_stage=text_ar; only text_ar resolves them to GPU tensors.
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: {
            "SGLANG_OMNI_ENABLE_TENSOR_REFS": "1",
            "SGLANG_OMNI_TENSOR_REF_EDGES": (
                "image_encoder:mm_aggregate:text_ar,"
                "audio_encoder:mm_aggregate:text_ar"
            ),
            "SGLANG_OMNI_TENSOR_REF_PATHS": "visual_embeds,audio_embeds",
        }
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="preprocessing",
                process="preprocessing",
                factory=f"{_PKG}.stages.create_preprocessing_executor",
                next=["image_encoder", "audio_encoder", "mm_aggregate"],
                route_fn=(
                    f"{_PKG}.request_builders."
                    "resolve_preprocessing_next_stages"
                ),
                project_payload={
                    "image_encoder": (
                        f"{_PKG}.request_builders.project_to_image_encoder"
                    ),
                    "audio_encoder": (
                        f"{_PKG}.request_builders.project_to_audio_encoder"
                    ),
                    "mm_aggregate": (
                        f"{_PKG}.request_builders.project_to_mm_aggregate"
                    ),
                },
            ),
            # Encoder stages micro-batch across requests by default
            # (one merged ViT/audio forward per batch window).  Tune via
            # factory_args or env: SGLANG_OMNI_LONGCAT_{IMAGE,AUDIO}_ENCODER_
            # MAX_BATCH_SIZE / MAX_BATCH_WAIT_MS / MAX_BATCH_PATCHES|FRAMES;
            # MAX_BATCH_SIZE=1 restores the legacy serial behavior.
            StageConfig(
                name="image_encoder",
                process="image_encoder",
                factory=f"{_PKG}.stages.create_image_encoder_executor",
                factory_args={"device": "cuda", "dtype": "bfloat16"},
                gpu=0,
                next="mm_aggregate",
                project_payload={
                    "mm_aggregate": (
                        f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                    ),
                },
            ),
            StageConfig(
                name="audio_encoder",
                process="audio_encoder",
                factory=f"{_PKG}.stages.create_audio_encoder_executor",
                factory_args={"device": "cuda", "dtype": "bfloat16"},
                gpu=1,
                next="mm_aggregate",
                project_payload={
                    "mm_aggregate": (
                        f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                    ),
                },
            ),
            StageConfig(
                name="mm_aggregate",
                process="mm_aggregate",
                factory=f"{_PKG}.stages.create_aggregate_executor",
                wait_for=["preprocessing", "image_encoder", "audio_encoder"],
                wait_for_fn=(
                    f"{_PKG}.request_builders."
                    "resolve_mm_aggregate_wait_sources"
                ),
                merge_fn=f"{_PKG}.merge.merge_for_text_ar",
                next="text_ar",
            ),
            _text_ar_stage(
                name="text_ar",
                gpu=[2, 3, 4, 5],
                terminal=not _audio_output_enabled(),
                next=(
                    ["code2wav"] if _audio_output_enabled() else None
                ),
                stream_to=(
                    ["code2wav"] if _audio_output_enabled() else None
                ),
            ),
        ]
        + (
            [
                StageConfig(
                    name="code2wav",
                    process="code2wav",
                    factory=f"{_PKG}.stages.create_code2wav_executor",
                    factory_args={"device": "cuda", "dtype": "bfloat16"},
                    gpu=6,
                    terminal=True,
                    can_accept_stream_before_payload=True,
                ),
            ]
            if _audio_output_enabled()
            else []
        )
    )


EntryClass = LongcatNextTextPipelineConfig

Variants = {
    "text": LongcatNextTextPipelineConfig,
    "multimodal": LongcatNextPipelineConfig,
}
