# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import uuid
from multiprocessing import shared_memory as _shm
from typing import Any

import numpy as np
import torch

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

# Per-process registry of ShmPutOperation handles that must survive GC
# until the consumer stage reads and unlinks the SHM segment.
# Keyed by shm_name, removed by ShmGetOperation after successful unlink.
_PENDING_PUTS: dict[str, ShmPutOperation] = {}


def _register_shm_put(op: "ShmPutOperation") -> None:
    shm_name = op._shm_obj.name
    _PENDING_PUTS[shm_name] = op


def _unregister_shm_put(shm_name: str) -> None:
    _PENDING_PUTS.pop(shm_name, None)


def shm_create_from_tensor(tensor: torch.Tensor) -> tuple[_shm.SharedMemory, _shm.SharedMemory]:
    """Creates a SHM block, writes tensor data, returns (creator, keeper).

    Returns TWO SharedMemory handles for the same segment:
    - creator: the original ``create=True`` handle (used for writing)
    - keeper:  a second ``name=...`` handle that pins the segment in the
      resource tracker so that GC of the creator handle in this process
      does not trigger premature shm_unlink before the consumer opens it.
    """
    t_cpu = tensor.cpu() if tensor.is_cuda else tensor
    t_np = t_cpu.numpy().reshape(-1)
    size = t_np.nbytes

    shm = _shm.SharedMemory(create=True, size=size)
    keeper = _shm.SharedMemory(name=shm.name)

    shm_view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    shm_view[:] = t_np[:]

    return shm, keeper


class ShmOperation(RelayOperation):
    """Base class implementation for SHM operations."""

    def __init__(self, metadata: Any):
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> Any:
        return self._metadata

    # wait_for_completion is implemented by subclasses


class ShmPutOperation(ShmOperation):
    """
    Handle for Put.
    In this simplified SHM model, writing is synchronous during creation,
    so the operation is effectively complete immediately.
    """

    def __init__(self, metadata: Any, shm_obj: _shm.SharedMemory, keeper: _shm.SharedMemory | None = None):
        super().__init__(metadata)
        self._shm_obj = shm_obj
        self._keeper = keeper

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        # Close the creator fd so we don't leak it.  The _keeper handle
        # (a second shm_open on the same segment) stays alive inside
        # this op object and pins the segment in the resource tracker,
        # preventing premature shm_unlink before the consumer opens it.
        if not self._completed:
            self._shm_obj.close()
            self._completed = True
        return


class ShmGetOperation(ShmOperation):
    """
    Handle for Get.
    Performs copy from SHM to destination tensor and unlinks the shared memory.
    """

    def __init__(self, metadata: Any, dest_tensor: torch.Tensor):
        super().__init__(metadata)
        self._transfer_info = metadata["transfer_info"]
        self._dest_tensor = dest_tensor

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        shm_name = self._transfer_info["shm_name"]
        size = self._transfer_info["size"]

        try:
            # 1. Open SHM
            try:
                existing_shm = _shm.SharedMemory(name=shm_name)
            except FileNotFoundError:
                raise RuntimeError(f"SHM block {shm_name} not found.")

            try:
                # 2. Zero-copy Read -> Copy to Dest
                shm_array = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
                src_tensor = torch.from_numpy(shm_array)

                dest_view = self._dest_tensor.view(torch.uint8).reshape(-1)
                copy_len = min(dest_view.numel(), size)
                dest_view[:copy_len].copy_(src_tensor[:copy_len])

                if self._dest_tensor.is_cuda:
                    torch.cuda.synchronize(self._dest_tensor.device)

            finally:
                # 3. Cleanup (Receiver owns lifecycle)
                existing_shm.close()
                existing_shm.unlink()

        finally:
            self._completed = True


@register_relay("shm")
class ShmRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cpu",
    ):
        self.engine_id = engine_id
        self.device = device
        # Semaphore mimics the 'credits' flow control
        self._sem = asyncio.Semaphore(credits)
        self._slot_size_bytes = slot_size_mb * 1024 * 1024

    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> RelayOperation:
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Flow control
        await self._sem.acquire()

        try:
            # 1. Create SHM and write data
            shm, keeper = shm_create_from_tensor(tensor)
            size_bytes = shm.size

            # 2. Construct Metadata
            metadata = {
                "engine_id": self.engine_id,
                "transfer_info": {
                    "shm_name": shm.name,
                    "size": size_bytes,
                    "req_id": request_id,
                },
            }

            # 3. Release semaphore immediately (Fire-and-Forget model)
            self._sem.release()

            op = ShmPutOperation(metadata, shm, keeper)
            _register_shm_put(op)
            return op

        except Exception as e:
            self._sem.release()
            raise e

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> RelayOperation:
        # Note: metadata validation is implicit here based on usage in test
        return ShmGetOperation(metadata=metadata, dest_tensor=dest_tensor)

    def cleanup(self, request_id: str) -> None:
        # In this pattern, cleanup is handled inside wait_for_completion (unlink)
        # or via garbage collection if the process dies.
        pass

    def close(self) -> None:
        pass

    # Optional hook for tests
    def reset_pool(self):
        pass
