# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for LongCat-Next encoder micro-batching.

The encoder models are replaced with fakes that mimic the real output
contract (token count derived from grid_thw / bridge_length, outputs marked
with arange so split boundaries are verifiable).  No GPU required.

Run:  pytest tests/unit_test/test_longcat_encoder_batch.py -q
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - exercised on GPU dev machines
    torch = None

from sglang_omni.models.longcat_next import stages as longcat_stages
from sglang_omni.models.longcat_next.payload_types import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    LongcatNextPipelineState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest


def _install_fake_flash_attn() -> None:
    """Stub heavy deps so components.encoders imports on CPU-only machines.

    components/dynamic.py bridges the flash-attn v4 API at import time, and
    models/weight_loader.py imports transformers.utils.hub at module level.
    The fake encoders below never call either, so inert stubs are enough.
    """
    import sys
    import types

    if "flash_attn" not in sys.modules:

        def _stub(*args, **kwargs):
            raise NotImplementedError("stub — not used by encoder fakes")

        flash_attn = types.ModuleType("flash_attn")
        cute = types.ModuleType("flash_attn.cute")
        testing = types.ModuleType("flash_attn.cute.testing")
        testing.index_first_axis = _stub
        testing.pad_input = _stub
        testing.unpad_input = _stub
        cute.testing = testing
        cute.flash_attn_func = _stub
        cute.flash_attn_varlen_func = _stub
        flash_attn.cute = cute
        sys.modules["flash_attn"] = flash_attn
        sys.modules["flash_attn.cute"] = cute
        sys.modules["flash_attn.cute.testing"] = testing

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        tf_utils = types.ModuleType("transformers.utils")
        tf_hub = types.ModuleType("transformers.utils.hub")

        def _cached_file(*args, **kwargs):
            raise NotImplementedError("transformers stub — not used by fakes")

        tf_hub.cached_file = _cached_file
        tf_utils.hub = tf_hub
        transformers.utils = tf_utils
        sys.modules["transformers"] = transformers
        sys.modules["transformers.utils"] = tf_utils
        sys.modules["transformers.utils.hub"] = tf_hub


_install_fake_flash_attn()

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeImageEncoder:
    """Mimics LongcatNextImageEncoder: tokens = sum(t * (h//2) * (w//2))."""

    instances: list = []
    fail_when_patches_above: int | None = None

    def __init__(self, model_path, *, device="cuda", dtype=None):
        self.calls = []
        FakeImageEncoder.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.fail_when_patches_above = None

    def __call__(self, pixel_values, visual_grid_thw):
        self.calls.append((pixel_values, visual_grid_thw))
        if (
            self.fail_when_patches_above is not None
            and pixel_values.shape[0] > self.fail_when_patches_above
        ):
            raise RuntimeError("simulated merged-forward failure")
        grid = visual_grid_thw.long()
        tokens = int((grid[:, 0] * (grid[:, 1] // 2) * (grid[:, 2] // 2)).sum().item())
        marker = torch.arange(tokens, dtype=torch.float32)
        return {
            "visual_ids": torch.arange(tokens, dtype=torch.long)
            .unsqueeze(1)
            .expand(tokens, 8)
            .contiguous(),
            "visual_embeds": marker.unsqueeze(1).expand(tokens, 4).contiguous(),
        }


class FakeAudioEncoder:
    """Mimics LongcatNextAudioEncoder: tokens = sum(bridge_length)."""

    instances: list = []

    def __init__(self, model_path, *, device="cuda", dtype=None):
        self.calls = []
        FakeAudioEncoder.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []

    def __call__(self, audio, encoder_length, bridge_length):
        self.calls.append((audio, encoder_length, bridge_length))
        tokens = int(bridge_length.sum().item())
        marker = torch.arange(tokens, dtype=torch.float32)
        return {
            "audio_ids": torch.arange(tokens, dtype=torch.long)
            .unsqueeze(1)
            .expand(tokens, 8)
            .contiguous(),
            "audio_embeds": marker.unsqueeze(1).expand(tokens, 4).contiguous(),
        }


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _image_payload(request_id, grids, *, cache_key=None, with_positions=True):
    """grids: list of (t, h, w); patches = t*h*w, tokens = t*(h//2)*(w//2)."""
    grid = torch.tensor(grids, dtype=torch.long)
    patches = int(sum(t * h * w for t, h, w in grids))
    tokens = int(sum(t * (h // 2) * (w // 2) for t, h, w in grids))
    inputs = {
        "pixel_values": torch.zeros(patches, 1176),
        "visual_grid_thw": grid,
    }
    if with_positions:
        inputs["image_positions"] = torch.arange(tokens)
    if cache_key:
        inputs["cache_key"] = cache_key
    state = LongcatNextPipelineState(encoder_inputs={IMAGE_STAGE: inputs})
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def _audio_payload(request_id, frames, bridge, *, cache_key=None, feature_dim=3):
    audio = torch.zeros(1, frames, feature_dim)
    inputs = {
        "audio": audio,
        "encoder_length": torch.tensor([frames]),
        "bridge_length": torch.tensor([bridge]),
        "audio_positions": torch.arange(bridge),
    }
    if cache_key:
        inputs["cache_key"] = cache_key
    state = LongcatNextPipelineState(encoder_inputs={AUDIO_STAGE: inputs})
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def _make_image_executor():
    FakeImageEncoder.reset()
    from sglang_omni.models.longcat_next.components import encoders

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(encoders, "LongcatNextImageEncoder", FakeImageEncoder)
        executor = longcat_stages.create_image_encoder_executor("fake-model-path")
    return executor, FakeImageEncoder.instances[0]


def _make_audio_executor():
    FakeAudioEncoder.reset()
    from sglang_omni.models.longcat_next.components import encoders

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(encoders, "LongcatNextAudioEncoder", FakeAudioEncoder)
        executor = longcat_stages.create_audio_encoder_executor("fake-model-path")
    return executor, FakeAudioEncoder.instances[0]


def _outs(payload, stage):
    return payload.data["encoder_outs"][stage]


# ---------------------------------------------------------------------------
# Pure-python knob tests (no torch needed)
# ---------------------------------------------------------------------------


def test_resolve_batch_param_precedence(monkeypatch):
    env = "SGLANG_OMNI_TEST_BATCH_PARAM"
    monkeypatch.delenv(env, raising=False)
    assert longcat_stages._resolve_batch_param(None, env, 7) == 7
    monkeypatch.setenv(env, "11")
    assert longcat_stages._resolve_batch_param(None, env, 7) == 11
    # Explicit factory arg wins over env.
    assert longcat_stages._resolve_batch_param(3, env, 7) == 3
    # Garbage env falls back to default.
    monkeypatch.setenv(env, "not-an-int")
    assert longcat_stages._resolve_batch_param(None, env, 7) == 7


def test_request_cost_fns_without_torch_objects():
    payload = SimpleNamespace(
        data={
            "encoder_inputs": {
                IMAGE_STAGE: {"pixel_values": SimpleNamespace(shape=(42, 1176))},
                AUDIO_STAGE: {"audio": SimpleNamespace(dim=lambda: 3, shape=(2, 100, 80))},
            }
        }
    )
    assert longcat_stages._image_request_cost(payload) == 42
    assert longcat_stages._audio_request_cost(payload) == 200
    # Missing media costs nothing.
    empty = SimpleNamespace(data={"encoder_inputs": {}})
    assert longcat_stages._image_request_cost(empty) == 0
    assert longcat_stages._audio_request_cost(empty) == 0


# ---------------------------------------------------------------------------
# Image encoder batching
# ---------------------------------------------------------------------------


@needs_torch
def test_image_batch_merges_into_single_forward_and_splits():
    executor, model = _make_image_executor()
    p1 = _image_payload("r1", [(1, 4, 6)], cache_key="k1")  # 6 tokens
    p2 = _image_payload("r2", [(2, 4, 4)], cache_key="k2")  # 8 tokens

    results = executor._batch_fn([p1, p2])

    assert len(model.calls) == 1, "batch should run ONE merged ViT forward"
    merged_pixels, merged_grid = model.calls[0]
    assert merged_grid.shape == (2, 3)  # one grid row per request (t folded in)

    out1, out2 = _outs(results[0], IMAGE_STAGE), _outs(results[1], IMAGE_STAGE)
    assert out1["visual_embeds"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert out2["visual_embeds"][:, 0].tolist() == [6, 7, 8, 9, 10, 11, 12, 13]
    assert out1["visual_ids"].shape == (6, 8)
    assert [r.request_id for r in results] == ["r1", "r2"]


@needs_torch
def test_image_batch_cache_hit_skips_compute():
    executor, model = _make_image_executor()
    p1 = _image_payload("r1", [(1, 4, 6)], cache_key="shared-key")
    executor._batch_fn([p1])
    assert len(model.calls) == 1

    # Same cache key → answered from cache, no new forward.
    p2 = _image_payload("r2", [(1, 4, 6)], cache_key="shared-key")
    results = executor._batch_fn([p2])
    assert len(model.calls) == 1
    assert _outs(results[0], IMAGE_STAGE)["visual_embeds"].shape[0] == 6


@needs_torch
def test_image_batch_mixed_with_cache_hit_and_passthrough():
    executor, model = _make_image_executor()
    warm = _image_payload("warm", [(1, 4, 6)], cache_key="warm-key")
    executor._batch_fn([warm])
    assert len(model.calls) == 1

    empty = _image_payload("empty", [(1, 4, 6)])
    empty.data["encoder_inputs"][IMAGE_STAGE]["pixel_values"] = None
    hit = _image_payload("hit", [(1, 4, 6)], cache_key="warm-key")
    miss = _image_payload("miss", [(1, 4, 4)], cache_key="new-key")  # 4 tokens

    results = executor._batch_fn([empty, hit, miss])

    assert len(model.calls) == 2, "only the cache-miss item needs a forward"
    assert _outs(results[0], IMAGE_STAGE) == {}
    assert _outs(results[1], IMAGE_STAGE)["visual_embeds"][:, 0].tolist() == [0] * 0 + [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert _outs(results[2], IMAGE_STAGE)["visual_embeds"][:, 0].tolist() == [0, 1, 2, 3]
    assert [r.request_id for r in results] == ["empty", "hit", "miss"]


@needs_torch
def test_image_batch_falls_back_to_serial_on_merged_failure():
    executor, model = _make_image_executor()
    # Merged forward (48 patches) fails; per-request forwards (24 patches) pass.
    FakeImageEncoder.fail_when_patches_above = 24
    p1 = _image_payload("r1", [(1, 4, 6)], cache_key="f1")
    p2 = _image_payload("r2", [(1, 4, 6)], cache_key="f2")

    results = executor._batch_fn([p1, p2])

    assert len(model.calls) == 3, "1 failed merged + 2 serial retries"
    for result in results:
        assert _outs(result, IMAGE_STAGE)["visual_embeds"][:, 0].tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]


@needs_torch
def test_image_batch_falls_back_when_positions_missing():
    executor, model = _make_image_executor()
    # Without image_positions the expected token count is unknown → serial.
    p1 = _image_payload("r1", [(1, 4, 6)], with_positions=False)
    p2 = _image_payload("r2", [(1, 4, 6)], with_positions=False)

    results = executor._batch_fn([p1, p2])

    assert len(model.calls) == 2, "unverifiable split must not use merged path"
    assert all(len(r.data["encoder_outs"]) == 1 for r in results)


# ---------------------------------------------------------------------------
# Audio encoder batching
# ---------------------------------------------------------------------------


@needs_torch
def test_audio_batch_pads_stacks_and_splits():
    executor, model = _make_audio_executor()
    p1 = _audio_payload("a1", frames=5, bridge=2, cache_key="ak1")
    p2 = _audio_payload("a2", frames=8, bridge=3, cache_key="ak2")

    results = executor._batch_fn([p1, p2])

    assert len(model.calls) == 1
    merged_audio, merged_enc_len, merged_bridge = model.calls[0]
    assert merged_audio.shape == (2, 8, 3), "stacked batch dim, padded to max frames"
    assert merged_audio[0, 5:, :].abs().sum().item() == 0, "short request zero-padded"
    assert merged_enc_len.tolist() == [5, 8]
    assert merged_bridge.tolist() == [2, 3]

    out1, out2 = _outs(results[0], AUDIO_STAGE), _outs(results[1], AUDIO_STAGE)
    assert out1["audio_embeds"][:, 0].tolist() == [0, 1]
    assert out2["audio_embeds"][:, 0].tolist() == [2, 3, 4]
    assert out1["audio_ids"].shape == (2, 8)
    assert [r.request_id for r in results] == ["a1", "a2"]


@needs_torch
def test_audio_single_item_uses_serial_path():
    executor, model = _make_audio_executor()
    p1 = _audio_payload("a1", frames=5, bridge=2)

    results = executor._batch_fn([p1])

    assert len(model.calls) == 1
    audio, _, _ = model.calls[0]
    assert audio.shape == (1, 5, 3), "single request must not be padded/merged"
    assert _outs(results[0], AUDIO_STAGE)["audio_embeds"][:, 0].tolist() == [0, 1]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
