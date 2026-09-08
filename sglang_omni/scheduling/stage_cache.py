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
    # Set when host tensors were filled by an async D2H issued on the offload
    # side stream. Must be waited on before the data is read (get / evict /
    # clear). ``None`` for the synchronous copy path.
    ready_event: torch.cuda.Event | None = None


class _OffloadContext:
    """Per-``put()`` bookkeeping for async D2H on the side stream."""

    __slots__ = ("stream", "used")

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        # True once at least one tensor copy was issued on the side stream.
        self.used = False


def _detach_value(
    value: Any,
    *,
    device: torch.device | None,
    pin: bool = False,
    offload: _OffloadContext | None = None,
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
                if offload is not None:
                    # Async D2H on the side stream: put() returns immediately
                    # and the producer keeps computing; the entry's
                    # ``ready_event`` is waited on when the value is consumed
                    # (get) or dropped (evict/clear), which in practice happens
                    # long after the copy finished — true overlap.
                    current = torch.cuda.current_stream(value.device)
                    offload.stream.wait_stream(current)
                    with torch.cuda.stream(offload.stream):
                        staging.copy_(value, non_blocking=True)
                    # Keep the source's memory from being reused by the
                    # producer stream while the side stream still reads it.
                    value.record_stream(offload.stream)
                    offload.used = True
                else:
                    staging.copy_(value)
                value = staging
            else:
                value = value.to(device=device)
        elif pin and value.device.type == "cpu" and not value.is_pinned():
            value = value.pin_memory()
        return value
    if isinstance(value, dict):
        return {
            key: _detach_value(item, device=device, pin=pin, offload=offload)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _detach_value(item, device=device, pin=pin, offload=offload)
            for item in value
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
    offloaded to pinned host memory to free the producing GPU. The D2H copy is
    issued asynchronously on a shared side stream, so ``put()`` never blocks
    the producer; a per-entry CUDA event guards consumption — ``get()`` (and
    eviction) waits on it before the host bytes are read.
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
        # Shared side stream for async D2H offloads; created lazily on the
        # first CUDA → pinned-host copy.
        self._offload_stream: torch.cuda.Stream | None = None

    def _offload_context(self) -> _OffloadContext | None:
        """Return an async-offload context when pinned CUDA → host copies apply."""
        if not self.pin_memory or not torch.cuda.is_available():
            return None
        if self._offload_stream is None:
            self._offload_stream = torch.cuda.Stream()
        return _OffloadContext(self._offload_stream)

    @staticmethod
    def _wait_ready(entry: _CacheEntry) -> None:
        """Block until the entry's async D2H (if any) has completed."""
        if entry.ready_event is not None:
            entry.ready_event.synchronize()
            entry.ready_event = None

    def get(self, key: str | None) -> Any | None:
        if key is None:
            return None
        key = str(key)
        entry = self._cache.get(key)
        if entry is None:
            return None
        # In practice get() happens long after put(), so this wait is nearly
        # always a no-op — the overlap win is that put() never blocked.
        self._wait_ready(entry)
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
            # The staging buffer being dropped may still be the target of an
            # in-flight async D2H; wait before releasing it.
            self._wait_ready(old_entry)
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            return
        offload = self._offload_context()
        detached = _detach_value(
            data,
            device=self.cache_device,
            pin=self.pin_memory,
            offload=offload,
        )
        ready_event: torch.cuda.Event | None = None
        if offload is not None and offload.used:
            ready_event = torch.cuda.Event()
            ready_event.record(offload.stream)
        self._cache[key] = _CacheEntry(
            data=detached,
            size_bytes=size_bytes,
            ready_event=ready_event,
        )
        self.current_bytes += size_bytes
        self._cache.move_to_end(key)
        self._evict_over_budget()

    def clear(self) -> None:
        for entry in self._cache.values():
            self._wait_ready(entry)
        self._cache.clear()
        self.current_bytes = 0

    def remove_if(self, predicate: Callable[[str], bool]) -> int:
        removed = 0
        for key in list(self._cache):
            if not predicate(key):
                continue
            entry = self._cache.pop(key)
            self.current_bytes -= entry.size_bytes
            self._wait_ready(entry)
            removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._cache)

    def _evict_over_budget(self) -> None:
        while self.max_size is not None and len(self._cache) > self.max_size:
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self._wait_ready(entry)
            self.eviction_count += 1
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self._cache:
                self.current_bytes = 0
                return
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self._wait_ready(entry)
            self.eviction_count += 1
