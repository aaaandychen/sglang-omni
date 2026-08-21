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

# Phase 4 §1: optionally offload the encoder-output cache to pinned host memory
# to free encoder-GPU memory. Default off (keeps tensors on the producing GPU).
#   SGLANG_OMNI_LONGCAT_ENCODER_CACHE_DEVICE = gpu|cpu   (default gpu)
#   SGLANG_OMNI_LONGCAT_ENCODER_CACHE_MAX_BYTES = <bytes>  (offload budget)
_ENV_ENCODER_CACHE_DEVICE = "SGLANG_OMNI_LONGCAT_ENCODER_CACHE_DEVICE"
_ENV_ENCODER_CACHE_MAX_BYTES = "SGLANG_OMNI_LONGCAT_ENCODER_CACHE_MAX_BYTES"


def _encoder_cache_device() -> str | None:
    """Return "cpu" when offload is requested, else None (keep on GPU)."""
    value = os.getenv(_ENV_ENCODER_CACHE_DEVICE, "").strip().lower()
    return "cpu" if value == "cpu" else None


def _encoder_cache_max_bytes() -> int:
    """Byte budget for the encoder cache.

    When offloading to CPU the host budget can be larger than encoder-GPU
    memory, so honor an override; otherwise fall back to the GPU default.
    """
    raw = os.getenv(_ENV_ENCODER_CACHE_MAX_BYTES, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "Invalid %s=%r; falling back to default", _ENV_ENCODER_CACHE_MAX_BYTES, raw
            )
    return _ENCODER_CACHE_MAX_BYTES


def _build_encoder_cache() -> StageOutputCache:
    """Construct the per-stage encoder-output cache, honoring offload env vars."""
    device = _encoder_cache_device()
    return StageOutputCache(
        max_size=_ENCODER_CACHE_MAX_SIZE,
        max_bytes=_encoder_cache_max_bytes(),
        cache_device=device,
        pin_memory=device == "cpu",
    )


# ---------------------------------------------------------------------------
# Encoder micro-batching helpers
#
# Both encoder stages run on a single GPU and used to process requests
# strictly one at a time (SimpleScheduler defaults).  The scheduler already
# supports cross-request micro-batching via ``batch_compute_fn``; the helpers
# below implement the LongCat-specific pieces:
#   * partition a batch into ready results (empty inputs / cache hit) and
#     pending compute items;
#   * merge pending items into ONE encoder forward (Qwen2-VL-style packed
#     pixel_values + grid_thw for images, zero-padded feature stacking for
#     audio) and split the outputs back per request;
#   * fall back to per-request serial encoding if the merged forward fails.
#
# Expected per-request token counts come from the pad-token positions the
# preprocessor already computed (image_positions / bridge_length), i.e. the
# ground truth the AR injection path enforces downstream — no dependency on
# tokenizer-internal merge factors.  A count mismatch aborts the merged path
# and triggers the serial fallback, so batching never corrupts a request.
# ---------------------------------------------------------------------------


def _resolve_batch_param(value: int | None, env_name: str, default: int) -> int:
    """Resolve a batching knob: explicit factory arg > env var > default."""
    if value is not None:
        return int(value)
    raw = os.getenv(env_name, "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("invalid %s=%r, using default %s", env_name, raw, default)
    return default


class _PendingEncode:
    """One batch item that needs a real encoder forward."""

    __slots__ = ("index", "payload", "state", "inputs", "expected_tokens")

    def __init__(self, index, payload, state, inputs, expected_tokens):
        self.index = index
        self.payload = payload
        self.state = state
        self.inputs = inputs
        self.expected_tokens = expected_tokens


def _partition_encoder_batch(payloads, *, stage: str, cache: "StageOutputCache"):
    """Split a batch into (ready: dict[int, payload], pending: list[_PendingEncode]).

    Items with no encoder inputs or with an encoder-cache hit are answered
    immediately; everything else becomes a pending compute item carrying the
    expected output token count used later for split validation.
    """
    from sglang_omni.models.longcat_next.payload_types import (
        LongcatNextPipelineState,
        payload_with_state,
    )

    ready: dict[int, object] = {}
    pending: list[_PendingEncode] = []
    for index, payload in enumerate(payloads):
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(stage) or {}
        if not inputs or not _has_encoder_media(stage, inputs):
            state.encoder_outs[stage] = {}
            ready[index] = payload_with_state(payload, state)
            continue

        cache_key = inputs.get("cache_key")
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                longcat_log_timing(
                    "encoder_cache",
                    stage=stage,
                    action="hit",
                    cache_key=cache_key,
                    request_id=payload.request_id,
                )
                state.encoder_outs[stage] = cached
                ready[index] = payload_with_state(payload, state)
                continue
            longcat_log_timing(
                "encoder_cache",
                stage=stage,
                action="miss",
                cache_key=cache_key,
                request_id=payload.request_id,
            )

        pending.append(
            _PendingEncode(
                index, payload, state, inputs, _expected_encoder_tokens(stage, inputs)
            )
        )
    return ready, pending


def _has_encoder_media(stage: str, inputs: dict) -> bool:
    from sglang_omni.models.longcat_next.payload_types import IMAGE_STAGE

    if stage == IMAGE_STAGE:
        return inputs.get("pixel_values") is not None
    return inputs.get("audio") is not None


def _expected_encoder_tokens(stage: str, inputs: dict) -> int:
    """Expected per-request output token count for split validation.

    Image: number of image_pad placeholder positions.  Audio: sum of
    bridge_length (falling back to audio pad positions).  Returns 0 when the
    count cannot be determined, which disables the merged path.
    """
    from sglang_omni.models.longcat_next.payload_types import IMAGE_STAGE

    if stage == IMAGE_STAGE:
        positions = inputs.get("image_positions")
        return int(positions.numel()) if positions is not None else 0
    bridge_length = inputs.get("bridge_length")
    if bridge_length is not None:
        return int(bridge_length.sum().item())
    positions = inputs.get("audio_positions")
    return int(positions.numel()) if positions is not None else 0


def _finalize_pending(
    item: _PendingEncode, result, ready: dict, *, stage: str, cache
) -> None:
    """Attach an encoder result to its request state, cache it, mark ready."""
    from sglang_omni.models.longcat_next.payload_types import payload_with_state

    item.state.encoder_outs[stage] = result
    cache_key = item.inputs.get("cache_key")
    if cache_key:
        cache.put(cache_key, result)
        longcat_log_timing(
            "encoder_cache",
            stage=stage,
            action="store",
            cache_key=cache_key,
            request_id=item.payload.request_id,
        )
    ready[item.index] = payload_with_state(item.payload, item.state)


def _image_request_cost(payload) -> int:
    """Batch cost = number of pixel patches (drives max_batch_cost)."""
    from sglang_omni.models.longcat_next.payload_types import (
        IMAGE_STAGE,
        LongcatNextPipelineState,
    )

    try:
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(IMAGE_STAGE) or {}
        pixel_values = inputs.get("pixel_values")
        return int(pixel_values.shape[0]) if pixel_values is not None else 0
    except Exception:  # pragma: no cover - defensive; cost never blocks correctness
        return 1


def _audio_request_cost(payload) -> int:
    """Batch cost = padded feature frames (batch_rows x frame_len)."""
    from sglang_omni.models.longcat_next.payload_types import (
        AUDIO_STAGE,
        LongcatNextPipelineState,
    )

    try:
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(AUDIO_STAGE) or {}
        audio = inputs.get("audio")
        if audio is None:
            return 0
        if audio.dim() == 2:
            return int(audio.shape[0])
        return int(audio.shape[0] * audio.shape[1])
    except Exception:  # pragma: no cover - defensive
        return 1


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
    max_batch_size: int | None = None,
    max_batch_wait_ms: int | None = None,
    max_batch_cost: int | None = None,
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
    cache = _build_encoder_cache()

    # Micro-batching knobs (factory arg > env > default).
    batch_size = _resolve_batch_param(
        max_batch_size, "SGLANG_OMNI_LONGCAT_IMAGE_ENCODER_MAX_BATCH_SIZE", 4
    )
    batch_wait_ms = _resolve_batch_param(
        max_batch_wait_ms, "SGLANG_OMNI_LONGCAT_IMAGE_ENCODER_MAX_BATCH_WAIT_MS", 10
    )
    batch_cost = _resolve_batch_param(
        max_batch_cost, "SGLANG_OMNI_LONGCAT_IMAGE_ENCODER_MAX_BATCH_PATCHES", 16384
    )

    def _encode_one(payload):
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

    def _encode_pending_merged(pending: list[_PendingEncode], ready: dict) -> None:
        """Run ONE ViT forward for all pending requests and split the output.

        pixel_values is packed across images as [num_patches, patch_dim] with
        per-image rows in visual_grid_thw, so concatenating both along dim 0
        yields a valid merged input (same packing the tokenizer already uses
        for multi-image requests).
        """
        import torch

        counts = [item.expected_tokens for item in pending]
        total_tokens = sum(counts)
        with longcat_timing(
            "image_encoder_batch",
            batch_size=len(pending),
            total_tokens=total_tokens,
        ):
            pixel_values = torch.cat(
                [item.inputs["pixel_values"] for item in pending], dim=0
            )
            visual_grid_thw = torch.cat(
                [item.inputs["visual_grid_thw"] for item in pending], dim=0
            )
            merged = model(
                pixel_values=pixel_values, visual_grid_thw=visual_grid_thw
            )
        for key, tensor in merged.items():
            if tensor.shape[0] != total_tokens:
                raise RuntimeError(
                    f"image encoder merged output '{key}' has {tensor.shape[0]} "
                    f"tokens, expected {total_tokens} from image_positions"
                )
        offset = 0
        for item, count in zip(pending, counts):
            result = {key: tensor[offset : offset + count] for key, tensor in merged.items()}
            offset += count
            _finalize_pending(item, result, ready, stage=IMAGE_STAGE, cache=cache)

    def _encode_batch(payloads):
        ready, pending = _partition_encoder_batch(
            payloads, stage=IMAGE_STAGE, cache=cache
        )
        mergeable = len(pending) > 1 and all(
            item.expected_tokens > 0 for item in pending
        )
        if mergeable:
            try:
                _encode_pending_merged(pending, ready)
            except Exception:
                logger.exception(
                    "image_encoder merged batch failed (%d requests); "
                    "falling back to serial encoding",
                    len(pending),
                )
                pending = [item for item in pending if item.index not in ready]
                for item in pending:
                    ready[item.index] = _encode_one(item.payload)
        else:
            for item in pending:
                ready[item.index] = _encode_one(item.payload)
        return [ready[index] for index in range(len(payloads))]

    return SimpleScheduler(
        _encode_one,
        batch_compute_fn=_encode_batch,
        max_batch_size=batch_size,
        max_batch_wait_ms=batch_wait_ms,
        request_cost_fn=_image_request_cost,
        max_batch_cost=batch_cost if batch_cost > 0 else None,
    )


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = "bfloat16",
    max_batch_size: int | None = None,
    max_batch_wait_ms: int | None = None,
    max_batch_cost: int | None = None,
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
    cache = _build_encoder_cache()

    # Micro-batching knobs (factory arg > env > default).
    batch_size = _resolve_batch_param(
        max_batch_size, "SGLANG_OMNI_LONGCAT_AUDIO_ENCODER_MAX_BATCH_SIZE", 8
    )
    batch_wait_ms = _resolve_batch_param(
        max_batch_wait_ms, "SGLANG_OMNI_LONGCAT_AUDIO_ENCODER_MAX_BATCH_WAIT_MS", 10
    )
    batch_cost = _resolve_batch_param(
        max_batch_cost, "SGLANG_OMNI_LONGCAT_AUDIO_ENCODER_MAX_BATCH_FRAMES", 0
    )

    def _encode_one(payload):
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

    def _encode_pending_merged(pending: list[_PendingEncode], ready: dict) -> None:
        """Run ONE audio-encoder forward for all pending requests and split.

        Each request's audio features are [B_i, L_i, D] (already padded
        within the request).  Cross-request batching zero-pads to the max
        frame length and stacks along the batch dim; the encoder masks on
        encoder_length, and the VQ bridger emits exactly sum(bridge_length)
        tokens, so outputs are split by per-request bridge_length sums.
        """
        import torch

        def _as_batched(audio):
            return audio.unsqueeze(0) if audio.dim() == 2 else audio

        counts = [item.expected_tokens for item in pending]
        total_tokens = sum(counts)
        with longcat_timing(
            "audio_encoder_batch",
            batch_size=len(pending),
            total_tokens=total_tokens,
        ):
            audios = [_as_batched(item.inputs["audio"]) for item in pending]
            max_len = max(int(audio.shape[1]) for audio in audios)
            total_rows = sum(int(audio.shape[0]) for audio in audios)
            merged_audio = audios[0].new_zeros(
                (total_rows, max_len, int(audios[0].shape[2]))
            )
            row = 0
            for audio in audios:
                rows, frames = int(audio.shape[0]), int(audio.shape[1])
                merged_audio[row : row + rows, :frames] = audio
                row += rows
            encoder_length = torch.cat(
                [item.inputs["encoder_length"] for item in pending]
            )
            bridge_length = torch.cat([item.inputs["bridge_length"] for item in pending])
            merged = model(
                audio=merged_audio,
                encoder_length=encoder_length,
                bridge_length=bridge_length,
            )
        for key, tensor in merged.items():
            if tensor.shape[0] != total_tokens:
                raise RuntimeError(
                    f"audio encoder merged output '{key}' has {tensor.shape[0]} "
                    f"tokens, expected {total_tokens} from bridge_length"
                )
        offset = 0
        for item, count in zip(pending, counts):
            result = {key: tensor[offset : offset + count] for key, tensor in merged.items()}
            offset += count
            _finalize_pending(item, result, ready, stage=AUDIO_STAGE, cache=cache)

    def _encode_batch(payloads):
        ready, pending = _partition_encoder_batch(
            payloads, stage=AUDIO_STAGE, cache=cache
        )
        mergeable = len(pending) > 1 and all(
            item.expected_tokens > 0 and item.inputs.get("bridge_length") is not None
            for item in pending
        )
        if mergeable:
            try:
                _encode_pending_merged(pending, ready)
            except Exception:
                logger.exception(
                    "audio_encoder merged batch failed (%d requests); "
                    "falling back to serial encoding",
                    len(pending),
                )
                pending = [item for item in pending if item.index not in ready]
                for item in pending:
                    ready[item.index] = _encode_one(item.payload)
        else:
            for item in pending:
                ready[item.index] = _encode_one(item.payload)
        return [ready[index] for index in range(len(payloads))]

    return SimpleScheduler(
        _encode_one,
        batch_compute_fn=_encode_batch,
        max_batch_size=batch_size,
        max_batch_wait_ms=batch_wait_ms,
        request_cost_fn=_audio_request_cost,
        max_batch_cost=batch_cost if batch_cost > 0 else None,
    )


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

    # ── Phase 3: streaming audio output builder ──────────────────────────
    def _audio_stream_builder(
        rid: str, sched_req_data: Any, req_output: Any
    ) -> list:
        """Emit one OutgoingMessage per decode step carrying the latest audio codes.

        Returns an ``OutgoingMessage(type="stream")`` so that the stage runtime
        routes it to the ``stream_to`` targets (code2wav).
        """
        sgl_req = getattr(sched_req_data, "req", None)
        if sgl_req is None:
            return []
        codes = getattr(sgl_req, "_longcat_latest_audio_codes", None)
        if codes is None:
            return []
        # Prevent duplicate emission: mark as emitted.
        sgl_req._longcat_latest_audio_codes = None
        from sglang_omni.scheduling.messages import OutgoingMessage

        return [OutgoingMessage(request_id=rid, type="stream", data=codes.cpu())]

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
        stream_output_builder=_audio_stream_builder,
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
    """Create a streaming scheduler that decodes audio incrementally every N frames."""
    import collections
    import asyncio

    import torch

    from sglang_omni.models.longcat_next.components.code2wav import (
        LongcatNextCode2Wav,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.messages import OutgoingMessage
    from sglang_omni.scheduling.streaming_simple_scheduler import (
        StreamingSimpleScheduler,
    )

    with longcat_timing("code2wav_executor_init", device=device, dtype=dtype):
        code2wav = LongcatNextCode2Wav(
            model_path, device=device, dtype=dtype,
        )

    # Per-request ring buffers: list of [8] code tensors.
    _buffers: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    _STREAM_FRAMES = 20  # decode every 20 audio frames

    def _wav_to_result(wav: torch.Tensor | None) -> dict[str, Any]:
        result: dict[str, Any] = {"text": "", "modality": "audio", "usage": {}}
        if wav is not None and wav.numel() > 0:
            arr = wav.detach().cpu().numpy()
            result["audio_waveform"] = arr.tobytes()
            result["audio_waveform_dtype"] = str(arr.dtype)
            result["audio_waveform_shape"] = list(arr.shape)
        return result

    def _decode_buffer(rid: str) -> torch.Tensor | None:
        """Decode all buffered frames for *rid*, clear the buffer."""
        frames = _buffers.pop(rid, [])
        if not frames:
            return None
        codes = torch.stack(frames).unsqueeze(0).to(device=device)
        wavs = code2wav.decode(codes)
        return wavs[0] if wavs else None

    def _decode_batch(payload: StagePayload) -> StagePayload:
        """Batch decode — falls back to non-streaming path.

        For streaming requests (identified by buffered frames), the audio
        codes have already been decoded incrementally via ``on_stream_chunk``;
        just flush remaining frames and skip re-decoding.
        """
        rid = payload.request_id
        if rid in _buffers:
            # Streaming request — flush remaining buffered frames.
            remaining = _buffers.pop(rid, [])
            wav = None
            if remaining:
                codes = torch.stack(remaining).unsqueeze(0).to(device=device)
                wavs = code2wav.decode(codes)
                wav = wavs[0] if wavs else None
            r = _wav_to_result(wav)
            return StagePayload(
                request_id=rid, request=payload.request, data=r,
            )

        # Non-streaming path — identical to original.
        with longcat_timing("code2wav_request", request_id=rid):
            state = payload.data if isinstance(payload.data, dict) else {}
            audio_codes = state.get("audio_codes")
            result: dict[str, Any] = {
                "text": state.get("text", ""),
                "modality": state.get("modality", "text"),
                "usage": state.get("usage", {}),
            }
            if audio_codes is None:
                return StagePayload(
                    request_id=rid, request=payload.request, data=result,
                )
            if isinstance(audio_codes, list):
                audio_codes = (
                    torch.stack(audio_codes) if audio_codes
                    else torch.empty((0, 8), dtype=torch.long)
                )
            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)
            audio_codes = audio_codes.to(device=device)
            waveforms = code2wav.decode(audio_codes)
            wav = waveforms[0] if waveforms else None
            if wav is not None and wav.numel() > 0:
                arr = wav.detach().cpu().numpy()
                result["audio_waveform"] = arr.tobytes()
                result["audio_waveform_dtype"] = str(arr.dtype)
                result["audio_waveform_shape"] = list(arr.shape)
            return StagePayload(
                request_id=rid, request=payload.request, data=result,
            )

    class _StreamingCode2WavScheduler(StreamingSimpleScheduler):
        def __init__(self):
            super().__init__(
                compute_fn=_decode_batch,
                batch_compute_fn=None,
                max_batch_size=1,
                max_batch_wait_ms=0,
            )

        def on_stream_chunk(self, request_id: str, item) -> list[OutgoingMessage]:
            """Handle each audio frame: buffer, decode every N frames."""
            codes = item.data if not isinstance(item, (str, bytes)) else item
            if not isinstance(codes, torch.Tensor):
                return []
            _buffers[request_id].append(codes.cpu() if codes.is_cuda else codes)
            if len(_buffers[request_id]) < _STREAM_FRAMES:
                return []

            wav = _decode_buffer(request_id)
            if wav is None:
                return []
            r = _wav_to_result(wav)
            return [OutgoingMessage(request_id=request_id, type="stream", data=r)]

        def is_streaming_payload(self, payload: StagePayload) -> bool:
            return False  # Never use built-in streaming path; we handle it ourselves.

    return _StreamingCode2WavScheduler()


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
