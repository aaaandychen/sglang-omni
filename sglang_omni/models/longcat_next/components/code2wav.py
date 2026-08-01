# SPDX-License-Identifier: Apache-2.0
"""LongCat-Next code2wav: audio codebook tokens → PCM waveform (Phase 3).

Decoding chain
--------------
8× codebook tokens  →  flow matching de-tokenizer  →  mel spectrogram
mel spectrogram     →  HiFi-GAN vocoder             →  PCM waveform 24 kHz

Both components are loaded from the LongCat-Next checkpoint:
* Flow matching de-tokenizer: ``model.audio_tokenizer.*`` (1740 keys in
  safetensors).  Reuses the same ``LongcatNextAudioTokenizer`` class that
  Phase 2 loads for the audio encoder, but calls its ``.decode()`` method.
* HiFi-GAN vocoder: ``cosy24k_vocoder/hift.pt`` (a standalone ``.pt`` file
  in the checkpoint directory).

Control
-------
Set the environment variable ``SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1``
to enable audio output.  When disabled (the default), the model behaves as
a text-only AR backbone (Phase 1/2 behaviour), and code2wav is never built.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import nn

from sglang_omni.models.longcat_next.components.dynamic import (
    get_longcat_config,
    get_longcat_remote_class,
)
from sglang_omni.models.longcat_next.payload_types import longcat_timing
from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

_ENV_ENABLE = "SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT"


def audio_output_enabled() -> bool:
    """Return ``True`` when the audio output path is active."""
    return os.getenv(_ENV_ENABLE, "").lower() in ("1", "true", "yes", "on")


class LongcatNextCode2Wav(nn.Module):
    """Audio de-tokenizer: discrete codebook codes → PCM waveform."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        self._dtype = torch_dtype
        self.model_path = model_path

        # ── audio de-tokenizer (flow matching) ──────────────────────────
        with longcat_timing("code2wav_config_init"):
            config = get_longcat_config(model_path)
        audio_cfg = config.audio_config
        vq_cfg = (
            audio_cfg.get("vq_config", {})
            if isinstance(audio_cfg, dict)
            else getattr(audio_cfg, "vq_config", {})
        )
        codebook_sizes_raw = (
            list(vq_cfg.get("codebook_sizes", []))
            if isinstance(vq_cfg, dict)
            else list(getattr(vq_cfg, "codebook_sizes", []))
        )
        self.codebook_sizes: list[int] = [int(s) for s in codebook_sizes_raw]

        with longcat_timing("code2wav_audio_tokenizer_class_init"):
            audio_cls = get_longcat_remote_class(
                model_path,
                "modular_longcat_next_audio.LongcatNextAudioTokenizer",
            )
        self.audio_tokenizer = audio_cls(config)
        with longcat_timing("code2wav_audio_tokenizer_load", device=device):
            load_module(
                self.audio_tokenizer,
                model_path,
                prefix="model.audio_tokenizer.",
                dtype=torch_dtype,
                device=device,
                strict=True,
            )

        # ── HiFi-GAN vocoder ────────────────────────────────────────────
        with longcat_timing("code2wav_vocoder_load"):
            self.vocoder = _load_vocoder(model_path, device, torch_dtype)

        self.eval()

    @torch.no_grad()
    def decode(
        self, audio_codes: torch.Tensor
    ) -> list[torch.Tensor | None]:
        """Decode audio codebook tokens to waveform tensors.

        Args:
            audio_codes: ``[N, 8]`` codebook token ids on GPU.

        Returns:
            List of ``N`` waveform tensors (each ``[1, samples]`` float32 on
            CPU), or ``None`` for empty segments.
        """
        # Detect valid lengths: codebook 0 token == codebook_sizes[0] (8192)
        # signals end-of-audio.
        if audio_codes.shape[0] == 0:
            return []

        response_len = (
            (audio_codes[:, :, 0] == self.codebook_sizes[0])
            .long()
            .argmax(dim=1)
        )
        # argmax returns 0 when no match is found; treat those as full length.
        response_len = torch.where(
            (audio_codes[:, :, 0] == self.codebook_sizes[0]).any(dim=1),
            response_len,
            torch.full_like(response_len, audio_codes.shape[1]),
        )
        valid_mask = response_len > 0
        valid_len = response_len[valid_mask]

        if valid_len.numel() == 0:
            return [None] * audio_codes.shape[0]

        # Flatten valid segments for batch flow-matching decode.
        valid_codes_list = [
            audio_codes[i, : int(valid_len[j]), :]
            for j, i in enumerate(valid_mask.nonzero(as_tuple=True)[0])
        ]
        flatten_codes = torch.cat(valid_codes_list, dim=0).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=self._dtype):
            ret = self.audio_tokenizer.decode(
                flatten_codes.view(-1, audio_codes.shape[-1]),
                bridge_length=valid_len,
            )

            # Reconstruct per-sample waveforms.
            results: list[torch.Tensor | None] = []
            valid_idx = 0
            for i in range(audio_codes.shape[0]):
                if not valid_mask[i]:
                    results.append(None)
                    continue
                mel = ret.flow_matching_mel[valid_idx][
                    : int(ret.flow_matching_mel_lengths[valid_idx]), :
                ]
                wav = self.vocoder.decode(
                    mel.transpose(0, 1).unsqueeze(0),
                )
                results.append(wav.cpu())
                valid_idx += 1

        return results


def _load_vocoder(
    model_path: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the HiFi-GAN vocoder from the checkpoint directory."""
    vocoder_pt = os.path.join(
        str(resolve_model_path(model_path)),
        "cosy24k_vocoder",
        "hift.pt",
    )
    if not os.path.exists(vocoder_pt):
        raise FileNotFoundError(
            f"LongCat-Next vocoder checkpoint not found at {vocoder_pt}"
        )
    cosy24k_cls = get_longcat_remote_class(
        model_path,
        "cosy24k_vocoder.Cosy24kVocoder",
    )
    vocoder = cosy24k_cls.from_pretrained(vocoder_pt)
    return vocoder.to(device=device, dtype=dtype).eval()
