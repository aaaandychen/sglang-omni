# SPDX-License-Identifier: Apache-2.0
"""Model runner hooks for LongCat-Next Phase 2 multimodal input injection."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class LongcatNextModelRunner(ModelRunner):
    """Attach multimodal replacement tensors to ForwardBatch during prefill."""

    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del requests
        if not schedule_batch.forward_mode.is_extend():
            return

        device = forward_batch.input_ids.device
        raw_extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if raw_extend_lens is None:
            extend_lens = []
        elif hasattr(raw_extend_lens, "tolist"):
            extend_lens = list(raw_extend_lens.tolist())
        else:
            extend_lens = list(raw_extend_lens)
        offsets: list[int] = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += int(length)

        replace_embeds_parts: list[torch.Tensor] = []
        replace_positions_parts: list[torch.Tensor] = []

        for i, req in enumerate(schedule_batch.reqs):
            data = getattr(req, "_omni_data", None)
            mm = getattr(data, "longcat_mm_inputs", None) if data is not None else None
            if not mm:
                continue
            batch_start = offsets[i] if i < len(offsets) else 0
            chunk_len = int(extend_lens[i]) if i < len(extend_lens) else 0
            # Global start in the original request sequence for this EXTEND
            # chunk.  In non-chunked prefill this is 0; in chunked prefill it
            # equals the prefix length already resident in KV cache.
            _prefix = getattr(req, "prefix_indices", None)
            chunk_global_start = len(_prefix) if _prefix is not None and _prefix.numel() > 0 else 0
            chunk_global_end = chunk_global_start + chunk_len
            _consumed = getattr(req, "_longcat_mm_consumed", None)
            consumed = _consumed if _consumed is not None else {}

            for key in ("image", "audio"):
                embeds = mm.get(f"{key}_embeds")
                positions = mm.get(f"{key}_positions")
                if embeds is None or positions is None:
                    continue
                positions = torch.as_tensor(positions, dtype=torch.long)
                if positions.numel() <= 0:
                    continue
                in_chunk = (positions >= chunk_global_start) & (positions < chunk_global_end)
                if not bool(in_chunk.any()):
                    continue

                selected_positions = positions[in_chunk]
                selected_count = int(selected_positions.numel())
                offset = int(consumed.get(key, 0))
                chunk = embeds[offset : offset + selected_count]
                if int(chunk.shape[0]) != selected_count:
                    raise ValueError(
                        f"LongCat-Next {key} embedding count mismatch for "
                        f"request {getattr(req, 'rid', '<unknown>')}: "
                        f"need {selected_count}, got {int(chunk.shape[0])}"
                    )
                local_positions = selected_positions - chunk_global_start + int(batch_start)
                replace_embeds_parts.append(chunk.to(device=device))
                replace_positions_parts.append(local_positions.to(device=device))
                consumed[key] = offset + selected_count

            req._longcat_mm_consumed = consumed
            if getattr(req, "is_chunked", 0) == 0:
                req._longcat_mm_consumed = None

        if replace_embeds_parts:
            forward_batch.longcat_replace_embeds = torch.cat(replace_embeds_parts, dim=0)
            forward_batch.longcat_replace_positions = torch.cat(replace_positions_parts, dim=0)
        else:
            forward_batch.longcat_replace_embeds = None
            forward_batch.longcat_replace_positions = None
