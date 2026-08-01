# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LongCat-Next pipelines."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang_omni.models.longcat_next.payload_types import longcat_log_timing, longcat_timing
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

# Env var to enable Phase 3 audio output (audio_head + code2wav).
# Default off — the model acts as a text-only AR backbone until explicitly enabled.
_ENV_ENABLE_AUDIO = "SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT"


def _audio_output_enabled() -> bool:
    return os.getenv(_ENV_ENABLE_AUDIO, "").lower() in ("1", "true", "yes", "on")

_ENCODER_CACHE_MAX_SIZE = 256
_ENCODER_CACHE_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB


def create_preprocessing_executor(model_path: str):
    from sglang_omni.models.longcat_next.components.preprocessor import (
        LongcatNextPreprocessor,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    with longcat_timing("preprocessing_executor_init"):
        preprocessor = LongcatNextPreprocessor(model_path)

    def _preprocess(payload):
        with longcat_timing("preprocessing_request", request_id=payload.request_id):
            return preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload):
        with longcat_timing("mm_aggregate_request", request_id=payload.request_id):
            return payload

    return SimpleScheduler(_identity)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = "bfloat16",
):
    from sglang_omni.models.longcat_next.components.encoders import (
        LongcatNextImageEncoder,
    )
    from sglang_omni.models.longcat_next.payload_types import (
        IMAGE_STAGE,
        LongcatNextPipelineState,
        payload_with_state,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    with longcat_timing("image_encoder_executor_init", device=device, dtype=dtype):
        model = LongcatNextImageEncoder(model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=_ENCODER_CACHE_MAX_SIZE,
        max_bytes=_ENCODER_CACHE_MAX_BYTES,
        cache_device=None,
    )

    def _encode(payload):
        with longcat_timing("image_encoder_request", request_id=payload.request_id):
            state = LongcatNextPipelineState.from_dict(payload.data)
            inputs = state.encoder_inputs.get(IMAGE_STAGE) or {}
            if not inputs or inputs.get("pixel_values") is None:
                state.encoder_outs[IMAGE_STAGE] = {}
                return payload_with_state(payload, state)

            cache_key = inputs.get("cache_key")
            if cache_key:
                cached = cache.get(cache_key)
                if cached is not None:
                    longcat_log_timing(
                        "encoder_cache",
                        stage=IMAGE_STAGE,
                        action="hit",
                        cache_key=cache_key,
                        request_id=payload.request_id,
                    )
                    state.encoder_outs[IMAGE_STAGE] = cached
                    return payload_with_state(payload, state)
                longcat_log_timing(
                    "encoder_cache",
                    stage=IMAGE_STAGE,
                    action="miss",
                    cache_key=cache_key,
                    request_id=payload.request_id,
                )

            result = model(
                pixel_values=inputs["pixel_values"],
                visual_grid_thw=inputs["visual_grid_thw"],
            )
            state.encoder_outs[IMAGE_STAGE] = result

            if cache_key:
                cache.put(cache_key, result)
                longcat_log_timing(
                    "encoder_cache",
                    stage=IMAGE_STAGE,
                    action="store",
                    cache_key=cache_key,
                    request_id=payload.request_id,
                )

            return payload_with_state(payload, state)

    return SimpleScheduler(_encode)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = "bfloat16",
):
    from sglang_omni.models.longcat_next.components.encoders import (
        LongcatNextAudioEncoder,
    )
    from sglang_omni.models.longcat_next.payload_types import (
        AUDIO_STAGE,
        LongcatNextPipelineState,
        payload_with_state,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    with longcat_timing("audio_encoder_executor_init", device=device, dtype=dtype):
        model = LongcatNextAudioEncoder(model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=_ENCODER_CACHE_MAX_SIZE,
        max_bytes=_ENCODER_CACHE_MAX_BYTES,
        cache_device=None,
    )

    def _encode(payload):
        with longcat_timing("audio_encoder_request", request_id=payload.request_id):
            state = LongcatNextPipelineState.from_dict(payload.data)
            inputs = state.encoder_inputs.get(AUDIO_STAGE) or {}
            if not inputs:
                state.encoder_outs[AUDIO_STAGE] = {}
                return payload_with_state(payload, state)

            cache_key = inputs.get("cache_key")
            if cache_key:
                cached = cache.get(cache_key)
                if cached is not None:
                    longcat_log_timing(
                        "encoder_cache",
                        stage=AUDIO_STAGE,
                        action="hit",
                        cache_key=cache_key,
                        request_id=payload.request_id,
                    )
                    state.encoder_outs[AUDIO_STAGE] = cached
                    return payload_with_state(payload, state)
                longcat_log_timing(
                    "encoder_cache",
                    stage=AUDIO_STAGE,
                    action="miss",
                    cache_key=cache_key,
                    request_id=payload.request_id,
                )

            result = model(
                audio=inputs["audio"],
                encoder_length=inputs["encoder_length"],
                bridge_length=inputs["bridge_length"],
            )
            state.encoder_outs[AUDIO_STAGE] = result

            if cache_key:
                cache.put(cache_key, result)
                longcat_log_timing(
                    "encoder_cache",
                    stage=AUDIO_STAGE,
                    action="store",
                    cache_key=cache_key,
                    request_id=payload.request_id,
                )

            return payload_with_state(payload, state)

    return SimpleScheduler(_encode)


def create_longcat_next_text_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 4096,
    context_length: int = 131072,
    mem_fraction_static: float | None = None,
    enable_torch_compile: bool = True,
    tp_size: int = 1,
    tp_rank: int = 0,
    server_args_overrides: dict[str, Any] | None = None,
    nccl_port: int | None = None,
):
    """Create an OmniScheduler for the LongCat-Next AR backbone."""
    from transformers import AutoTokenizer

    from sglang_omni.models.longcat_next.model_runner import LongcatNextModelRunner
    from sglang_omni.models.longcat_next.request_builders import (
        make_longcat_next_text_adapters,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure_defer_cuda_graph,
    )
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    gpu_id = int(device.split(":")[-1]) if ":" in device else tp_rank
    with longcat_timing("text_ar_tokenizer_init", tp_rank=tp_rank, device=device):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        disable_overlap_schedule=False,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=16384,
        chunked_prefill_size=16384,
        sampling_backend="pytorch",
        dtype=dtype,
    )

    with longcat_timing("text_ar_server_args_build", tp_rank=tp_rank, device=device):
        server_args = build_sglang_server_args(
            model_path,
            context_length=context_length,
            tp_size=tp_size,
            **overrides,
        )
    server_args.moe_runner_backend = "flashinfer_cutlass"

    validate_generation_batch_policy(
        model_name="LongCat-Next Text",
        server_args=server_args,
    )

    with longcat_timing("text_ar_infrastructure_init", tp_rank=tp_rank, gpu_id=gpu_id):
        want_cuda_graph, (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            prefill_mgr,
            decode_mgr,
            model_config,
        ) = create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            gpu_id,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            model_arch_override="LongcatNextTextForCausalLM",
        )

    # After model init, the effective vocab size may differ from config.json
    # due to ngram embedding expansion.  Update model_config AND the model's
    # LogitsProcessor so the CUDA graph runner and logits path agree.
    _model = model_worker.model_runner.model
    _actual_vocab = int(_model.lm_head.weight.shape[0]) * getattr(_model.lm_head, "tp_size", 1)
    model_config.vocab_size = _actual_vocab
    _model.logits_processor.vocab_size = _actual_vocab

    # ── Phase 3: audio head (gated by env var) ────────────────────────
    if _audio_output_enabled():
        with longcat_timing("text_ar_audio_head_init", tp_rank=tp_rank, gpu_id=gpu_id):
            _attach_audio_head(model_path, _model, gpu_id=gpu_id, dtype=dtype)
    else:
        logger.info(
            "LongCat-Next audio output disabled (set %s=1 to enable)",
            _ENV_ENABLE_AUDIO,
        )

    if want_cuda_graph:
        with longcat_timing("text_ar_cuda_graph_init", tp_rank=tp_rank, gpu_id=gpu_id):
            model_worker.model_runner.init_device_graphs()

    with longcat_timing("text_ar_output_processor_init", tp_rank=tp_rank):
        output_proc = SGLangOutputProcessor(
            capture_hidden=False,
            capture_hidden_layers=None,
            model=model_worker.model_runner.model,
        )

    with longcat_timing("text_ar_request_adapters_init", tp_rank=tp_rank):
        request_builder, result_adapter = make_longcat_next_text_adapters(
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=LongcatNextModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        enable_async_decode=True,
    )


def _attach_audio_head(
    model_path: str,
    model: Any,
    *,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
) -> None:
    """Create and attach a LongcatNextAudioHead to the AR model (Phase 3)."""
    from transformers import AutoConfig

    from sglang_omni.models.longcat_next.components.audio_head import (
        LongcatNextAudioHead,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    audio_cfg = config.audio_config
    vq_cfg = audio_cfg.get("vq_config", {}) if isinstance(audio_cfg, dict) else getattr(audio_cfg, "vq_config", {})
    codebook_sizes = (
        list(vq_cfg.get("codebook_sizes", []))
        if isinstance(vq_cfg, dict)
        else list(getattr(vq_cfg, "codebook_sizes", []))
    )
    if not codebook_sizes:
        raise ValueError(
            "LongCat-Next config.audio_config.vq_config.codebook_sizes "
            "is required for Phase 3 audio head."
        )

    audio_head = LongcatNextAudioHead(
        model_path,
        hidden_size=int(config.hidden_size),
        codebook_sizes=codebook_sizes,
        transformer_ffn_scale=(
            audio_cfg.get("audio_head_transformer_ffn_scale", 0)
            if isinstance(audio_cfg, dict)
            else getattr(audio_cfg, "audio_head_transformer_ffn_scale", 0)
        ),
        transformer_dims=(
            audio_cfg.get("audio_head_transformer_dims", 0)
            if isinstance(audio_cfg, dict)
            else getattr(audio_cfg, "audio_head_transformer_dims", 0)
        ),
        transformer_layers=(
            audio_cfg.get("audio_head_transformer_layers", 0)
            if isinstance(audio_cfg, dict)
            else getattr(audio_cfg, "audio_head_transformer_layers", 0)
        ),
        audio_offset=int(config.audio_offset),
        device=f"cuda:{gpu_id}",
        dtype=dtype,
    )
    model.set_audio_head(audio_head)


def create_code2wav_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = "bfloat16",
):
    """Create a SimpleScheduler for the audio de-tokenizer + vocoder (Phase 3)."""
    from sglang_omni.models.longcat_next.components.code2wav import (
        LongcatNextCode2Wav,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    with longcat_timing("code2wav_executor_init", device=device, dtype=dtype):
        code2wav = LongcatNextCode2Wav(
            model_path, device=device, dtype=dtype,
        )

    def _decode(payload):
        import torch

        with longcat_timing("code2wav_request", request_id=payload.request_id):
            state = payload.data if isinstance(payload.data, dict) else {}
            audio_codes = state.get("audio_codes")

            result = dict(state)  # pass-through text, usage, etc.
            if audio_codes is None:
                result["audio_waveforms"] = []
                return result
            if isinstance(audio_codes, list):
                audio_codes = (
                    torch.stack(audio_codes) if audio_codes
                    else torch.empty((0, 8), dtype=torch.long)
                )
            waveforms = code2wav.decode(audio_codes)
            result["audio_waveforms"] = waveforms
            return result

    return SimpleScheduler(_decode)


def create_longcat_next_executor(*args: Any, **kwargs: Any):
    """Alias kept for backward compatibility."""
    return create_longcat_next_text_executor(*args, **kwargs)


__all__ = [
    "create_preprocessing_executor",
    "create_aggregate_executor",
    "create_image_encoder_executor",
    "create_audio_encoder_executor",
    "create_longcat_next_text_executor",
    "create_longcat_next_executor",
    "create_code2wav_executor",
]
