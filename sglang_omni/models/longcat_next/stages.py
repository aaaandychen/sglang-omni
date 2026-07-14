# SPDX-License-Identifier: Apache-2.0
"""Stage factory for LongCat-Next text-only AR backbone."""

from __future__ import annotations

from typing import Any


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
    """Create an OmniScheduler for the LongCat-Next text backbone.

    Parameters
    ----------
    model_path:
        HF model id or local path to the **full** LongCat-Next checkpoint.
        Visual/audio weights are present on disk but skipped at load time.
    device:
        ``"cuda:0"``-style device string.
    dtype:
        Computation dtype (``"bfloat16"`` / ``"float16"``).
    max_running_requests:
        Maximum concurrent requests admitted to the batch.
    max_new_tokens:
        Per-request token budget passed to :class:`SamplingParams`.
    context_length:
        SGLang context length.  LongCat-Next supports up to 131 072.
    mem_fraction_static:
        Fraction of GPU memory reserved for KV cache (SGLang
        ``mem_fraction_static``).  ``None`` lets SGLang auto-select.
    enable_torch_compile:
        Enable ``torch.compile`` for the decode path.
    tp_size:
        Tensor-parallel size (passed through to SGLang ServerArgs).
        sglang-omni injects this from ``StageConfig.tp_size``.
    tp_rank:
        Tensor-parallel rank within this stage (injected by the runtime).
    server_args_overrides:
        Extra SGLang ServerArgs forwarded to ``build_sglang_server_args``.
    """
    from transformers import AutoTokenizer

    from sglang_omni.model_runner.base import ModelRunner
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

    # ── GPU id ─────────────────────────────────────────────────────────
    gpu_id = int(device.split(":")[-1]) if ":" in device else tp_rank

    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # ── SGLang ServerArgs ──────────────────────────────────────────────
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
    # SGLang 0.5.12 Triton MoE kernel (3.5.1) crashes on H800 SM90.
    # Force flashinfer_cutlass MoE backend to bypass Triton entirely.
    # flashinfer_cutlass is in SGLang's MOE_RUNNER_BACKEND_CHOICES.
    server_args.moe_runner_backend = "flashinfer_cutlass"

    validate_generation_batch_policy(
        model_name="LongCat-Next Text",
        server_args=server_args,
    )

    # ── SGLang infrastructure (model worker, pools, managers) ─────────
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

    # ── CUDA graph capture ─────────────────────────────────────────────
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    # ── Output processor ───────────────────────────────────────────────
    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

    # ── Request / result adapters ──────────────────────────────────────
    request_builder, result_adapter = make_longcat_next_text_adapters(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # ── Assemble OmniScheduler ─────────────────────────────────────────
    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_longcat_next_executor(*args: Any, **kwargs: Any):
    """Alias kept for backward compatibility."""
    return create_longcat_next_text_executor(*args, **kwargs)


__all__ = [
    "create_longcat_next_text_executor",
    "create_longcat_next_executor",
]
