# SPDX-License-Identifier: Apache-2.0
"""Model runner hooks for LongCat-Next Phase 2 multimodal + Phase 3 audio output."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.longcat_next.payload_types import longcat_log_timing, longcat_timing

# ── Phase 3: audio generation special tokens ──────────────────────────────
# From config.json → audio_config.
_AUDIOGEN_START_TOKEN_ID: int = 131123
_AUDIOGEN_END_TOKEN_ID: int = 131124


class LongcatNextModelRunner(ModelRunner):
    """Phase 2 multimodal prefill + Phase 3 audio generation hooks."""

    def execute(self, scheduler_output: Any):
        batch = getattr(scheduler_output, "batch_data", None)
        forward_mode = getattr(batch, "forward_mode", None)
        phase = "prefill" if forward_mode is not None and forward_mode.is_extend() else "decode"
        reqs = getattr(batch, "reqs", []) if batch is not None else []
        with longcat_timing(
            "text_ar_execute",
            phase=phase,
            batch_size=len(reqs),
        ):
            return super().execute(scheduler_output)

    # ── Phase 2: multimodal prefill ──────────────────────────────────────

    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del requests
        if not schedule_batch.forward_mode.is_extend():
            forward_batch.longcat_replace_embeds = None
            forward_batch.longcat_replace_positions = None
            return

        with longcat_timing(
            "text_ar_before_prefill_mm_injection",
            batch_size=len(getattr(schedule_batch, "reqs", []) or []),
        ):
            self._attach_multimodal_replacements(forward_batch, schedule_batch)

    def _attach_multimodal_replacements(self, forward_batch: Any, schedule_batch: Any) -> None:
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
            longcat_log_timing(
                "text_ar_mm_replacements_attached",
                replace_tokens=int(forward_batch.longcat_replace_positions.numel()),
                device=device,
            )
        else:
            forward_batch.longcat_replace_embeds = None
            forward_batch.longcat_replace_positions = None

    # ── Phase 3: audio generation hooks ──────────────────────────────────

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Inject previous-step audio codes for LLM input embedding fusion.

        For requests in audio generation mode, sets
        ``forward_batch.longcat_audio_codes`` with the codes produced at the
        previous decode step (or zeros for the very first audio step).  The
        model's :meth:`forward` reads this tensor to fuse audio embeddings.
        """
        del is_lookahead
        reqs = getattr(schedule_batch, "reqs", [])
        if not reqs:
            forward_batch.longcat_audio_codes = None
            return

        device = forward_batch.input_ids.device
        model = self.tp_worker.model_runner.model
        num_codebooks = (
            model.audio_head.num_codebooks
            if model.audio_head is not None
            else 8
        )
        zeros = torch.zeros(num_codebooks, dtype=torch.long, device=device)
        codes_parts: list[torch.Tensor] = []
        has_audio = False

        for req in reqs:
            audio_state = getattr(req, "_longcat_audio_state", None)
            if audio_state and audio_state.get("mode") == "audio":
                prev = audio_state.get("prev_codes")
                codes_parts.append(
                    prev.to(device=device) if prev is not None else zeros.clone()
                )
                has_audio = True
            else:
                codes_parts.append(zeros.clone())

        if has_audio:
            forward_batch.longcat_audio_codes = torch.stack(codes_parts)
            longcat_log_timing(
                "text_ar_before_decode_audio_fusion",
                batch_size=len(reqs),
                audio_requests=sum(
                    1 for r in reqs
                    if getattr(r, "_longcat_audio_state", {}).get("mode") == "audio"
                ),
            )
        else:
            forward_batch.longcat_audio_codes = None

        # Always clear the side-channel from the previous forward.
        forward_batch.longcat_new_audio_codes = None

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Consume audio codes produced by the model forward and update state.

        Reads ``forward_batch.longcat_new_audio_codes`` (set by the model's
        :meth:`forward` as a side channel), stores them for the next decode
        step, and appends to the per-request accumulated code list.

        State transitions (text ↔ audio mode) are driven by text token
        detection and run regardless of whether audio codes were produced.
        """
        output_ids = getattr(schedule_batch, "output_ids", None)
        if output_ids is None:
            return
        reqs = getattr(schedule_batch, "reqs", [])
        if not reqs:
            return

        new_audio_codes = getattr(forward_batch, "longcat_new_audio_codes", None)
        model = self.tp_worker.model_runner.model
        num_codebooks = (
            model.audio_head.num_codebooks
            if model.audio_head is not None
            else 8
        )

        for i, req in enumerate(reqs):
            text_token = int(output_ids[i])
            codes_i = (
                new_audio_codes[i].cpu()
                if new_audio_codes is not None
                else torch.zeros(num_codebooks, dtype=torch.long)
            )

            # Initialize or read per-request audio state.
            audio_state = getattr(req, "_longcat_audio_state", None)
            if audio_state is None:
                audio_state = {"mode": "text", "prev_codes": None}
            mode = audio_state.get("mode", "text")

            if mode == "audio":
                # Still in audio mode — store codes for next step.
                audio_state["prev_codes"] = codes_i
                req._longcat_audio_codes_list.append(codes_i)
                if text_token == _AUDIOGEN_END_TOKEN_ID:
                    audio_state["mode"] = "text"
                    audio_state["prev_codes"] = None
                    longcat_log_timing(
                        "audio_gen_end",
                        request_id=getattr(req, "rid", str(i)),
                        total_audio_steps=len(req._longcat_audio_codes_list),
                    )
            elif text_token == _AUDIOGEN_START_TOKEN_ID:
                # Enter audio generation mode.
                # prev_codes stays None — before_decode at the next step
                # will supply zeros for the first audio forward.
                audio_state["mode"] = "audio"
                audio_state["prev_codes"] = None
                req._longcat_audio_codes_list = []
                longcat_log_timing(
                    "audio_gen_start",
                    request_id=getattr(req, "rid", str(i)),
                )
            # else: text mode, no transition — nothing to do.

            req._longcat_audio_state = audio_state

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        """Attach accumulated audio codes and waveform placeholder to outputs."""
        for sched_req in scheduler_output.requests:
            codes = getattr(sched_req.data.req, "_longcat_audio_codes_list", None)
            if codes and len(codes) > 0:
                stacked = torch.stack(codes)  # [N_steps, 8]
                rid = sched_req.request_id
                if rid in outputs:
                    outputs[rid].extra["audio_codes"] = stacked
