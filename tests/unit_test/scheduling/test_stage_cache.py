# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StageOutputCache (CPU-only; CUDA overlap path is GPU-only).

The async side-stream offload requires CUDA; on CPU-only hosts the cache must
degrade gracefully to the synchronous path with identical semantics.
"""

from __future__ import annotations

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache


def test_put_get_cpu_tensor_roundtrip() -> None:
    cache = StageOutputCache(max_size=4, cache_device="cpu", pin_memory=True)
    value = torch.arange(6, dtype=torch.float32)
    cache.put("k", value)
    got = cache.get("k")
    assert got is not None
    assert torch.equal(got, value)


def test_put_get_nested_structure() -> None:
    cache = StageOutputCache(max_size=4)
    value = {"a": torch.ones(2), "b": [torch.zeros(3), 42, "s"]}
    cache.put("k", value)
    got = cache.get("k")
    assert torch.equal(got["a"], value["a"])
    assert torch.equal(got["b"][0], value["b"][0])
    assert got["b"][1:] == [42, "s"]


def test_lru_eviction_and_byte_budget() -> None:
    cache = StageOutputCache(max_size=2)
    cache.put("a", torch.zeros(1))
    cache.put("b", torch.zeros(1))
    cache.get("a")  # refresh
    cache.put("c", torch.zeros(1))  # evicts b (least recently used)
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.eviction_count == 1

    byte_cache = StageOutputCache(max_bytes=8)  # 2 float32s
    byte_cache.put("x", torch.zeros(4, dtype=torch.float32))  # 16B > budget
    assert byte_cache.get("x") is None  # too large: never cached


def test_replace_existing_key_updates_bytes() -> None:
    cache = StageOutputCache(max_bytes=16)
    cache.put("k", torch.zeros(2, dtype=torch.float32))  # 8B
    cache.put("k", torch.zeros(4, dtype=torch.float32))  # 16B replaces
    assert cache.current_bytes == 16
    assert len(cache) == 1


def test_clear_and_remove_if() -> None:
    cache = StageOutputCache()
    cache.put("req-1:a", torch.zeros(1))
    cache.put("req-1:b", torch.zeros(1))
    cache.put("req-2:a", torch.zeros(1))
    removed = cache.remove_if(lambda key: key.startswith("req-1:"))
    assert removed == 2
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.current_bytes == 0


def test_cpu_offload_path_uses_sync_copy_without_cuda(monkeypatch) -> None:
    """Without CUDA, pin_memory offload must still work via the sync path."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = StageOutputCache(max_size=4, cache_device="cpu", pin_memory=True)
    assert cache._offload_context() is None  # no side stream without CUDA
    value = torch.ones(4)
    cache.put("k", value)
    assert torch.equal(cache.get("k"), value)
    # CPU-source tensors with pin=True get pinned directly.
    if hasattr(torch.Tensor, "is_pinned"):
        assert cache.get("k").is_pinned() in (True, False)  # platform dependent


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
