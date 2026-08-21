# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _CacheEntry:
    data: Any
    size_bytes: int


def _detach_value(
    value: Any,
    *,
    device: torch.device | None,
    pin: bool = False,
) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            # Offload to pinned host memory so the later relay H2D can overlap.
            to_cpu = device.type == "cpu"
            if to_cpu and pin and value.device.type == "cuda":
                staging = torch.empty_like(
                    value, device=device, pin_memory=True
                )
                # Synchronous D2H: put() is a sync call, so we cannot return
                # before the copy completes. A side stream here would need an
                # event waited on at get() time to actually overlap; until that
                # exists, copy on the current stream and let it complete.
                staging.copy_(value)
                value = staging
            else:
                value = value.to(device=device)
        elif pin and value.device.type == "cpu" and not value.is_pinned():
            value = value.pin_memory()
        return value
    if isinstance(value, dict):
        return {
            key: _detach_value(item, device=device, pin=pin)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _detach_value(item, device=device, pin=pin) for item in value
        )
    return value


def _value_size_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_value_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_size_bytes(item) for item in value)
    return 0


class StageOutputCache:
    """Small in-memory LRU cache for non-AR stage outputs.

    With ``cache_device="cpu"`` and ``pin_memory=True``, cached tensors are
    offloaded to pinned host memory to free the producing GPU.
    """

    def __init__(
        self,
        max_size: int | None = None,
        max_bytes: int | None = None,
        cache_device: torch.device | str | None = None,
        size_fn: Callable[[Any], int] | None = None,
        pin_memory: bool = False,
    ) -> None:
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache_device = cache_device
        self.current_bytes = 0
        self.eviction_count = 0
        self._size_fn = size_fn or _value_size_bytes
        # Pin only makes sense when offloading to host memory.
        self.pin_memory = bool(
            pin_memory and cache_device is not None and cache_device.type == "cpu"
        )

    def get(self, key: str | None) -> Any | None:
        if key is None:
            return None
        key = str(key)
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return entry.data

    def put(self, key: str | None, data: Any) -> None:
        if key is None:
            return
        key = str(key)
        size_bytes = self._size_fn(data)
        old_entry = self._cache.pop(key, None)
        if old_entry is not None:
            self.current_bytes -= old_entry.size_bytes
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            return
        self._cache[key] = _CacheEntry(
            data=_detach_value(
                data,
                device=self.cache_device,
                pin=self.pin_memory,
            ),
            size_bytes=size_bytes,
        )
        self.current_bytes += size_bytes
        self._cache.move_to_end(key)
        self._evict_over_budget()

    def clear(self) -> None:
        self._cache.clear()
        self.current_bytes = 0

    def remove_if(self, predicate: Callable[[str], bool]) -> int:
        removed = 0
        for key in list(self._cache):
            if not predicate(key):
                continue
            entry = self._cache.pop(key)
            self.current_bytes -= entry.size_bytes
            removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._cache)

    def _evict_over_budget(self) -> None:
        while self.max_size is not None and len(self._cache) > self.max_size:
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self._cache:
                self.current_bytes = 0
                return
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
