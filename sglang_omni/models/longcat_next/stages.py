# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LongCat-Next pipelines."""

from __future__ import annotations

from typing import Any


def create_preprocessing_executor(model_path: str):
    from sglang_omni.models.longcat_next.components.preprocessor import (
        LongcatNextPreprocessor,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = LongcatNextPreprocessor(model_path)

    def _preprocess(payload):
        return preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload):
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

    model = LongcatNextImageEncoder(model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(IMAGE_STAGE) or {}
        if not inputs:
            state.encoder_outs[IMAGE_STAGE] = {}
            return payload_with_state(payload, state)
        result = model(
            pixel_values=inputs["pixel_values"],
            visual_grid_thw=inputs["visual_grid_thw"],
        )
        state.encoder_outs[IMAGE_STAGE] = result
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

    model = LongcatNextAudioEncoder(model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(AUDIO_STAGE) or {}
        if not inputs:
            state.encoder_outs[AUDIO_STAGE] = {}
            return payload_with_state(payload, state)
        result = model(
            audio=inputs["audio"],
            encoder_length=inputs["encoder_length"],
            bridge_length=inputs["bridge_length"],
        )
        state.encoder_outs[AUDIO_STAGE] = result
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
    enable_torch_compile: bool = False,
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=16384,
        chunked_prefill_size=16384,
        sampling_backend="pytorch",
        dtype=dtype,
    )

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

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

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
    )


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
]
