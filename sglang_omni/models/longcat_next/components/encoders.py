# SPDX-License-Identifier: Apache-2.0
"""LongCat-Next image/audio encoder wrappers for Phase 2."""

from __future__ import annotations

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
    load_weights_by_prefix,
    resolve_dtype,
)


def _codebook_sizes(config_obj: Any) -> list[int]:
    if isinstance(config_obj, dict):
        sizes = config_obj.get("codebook_sizes")
        if sizes is None and isinstance(config_obj.get("vq_config"), dict):
            sizes = config_obj["vq_config"].get("codebook_sizes")
    else:
        sizes = getattr(config_obj, "codebook_sizes", None)
        if sizes is None and hasattr(config_obj, "vq_config"):
            sizes = getattr(config_obj.vq_config, "codebook_sizes", None)
    if sizes is None:
        raise AttributeError("LongCat-Next config is missing vq_config.codebook_sizes")
    return list(map(int, sizes))


class _OffsetCodebookEmbedding(nn.Module):
    """Sum per-codebook embeddings sliced from model.embed_tokens.weight."""

    def __init__(
        self,
        *,
        model_path: str,
        offset: int,
        codebook_sizes: list[int],
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        with longcat_timing("codebook_embedding_load", device=device, offset=offset):
            state = load_weights_by_prefix(model_path, prefix="model.embed_tokens.")
            weight = state["weight"]
            layers = []
            start = int(offset)
            for size in codebook_sizes:
                stop = start + int(size)
                emb_weight = weight[start:stop].to(dtype=dtype)
                layers.append(nn.Embedding.from_pretrained(emb_weight, freeze=True))
                start = stop
            self.layers = nn.ModuleList(layers).to(device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long()
        out = None
        for i, layer in enumerate(self.layers):
            part = layer(ids[..., i].to(layer.weight.device))
        out = part if out is None else out + part.to(out.device)
        assert out is not None
        return out


class LongcatNextImageEncoder(nn.Module):
    """Image input encoder: pixels -> AR hidden-size visual embeddings."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        self.model_path = model_path
        with longcat_timing("image_encoder_config_init", device=device, dtype=torch_dtype):
            self.config = get_longcat_config(model_path)
        with longcat_timing("image_encoder_remote_class_init"):
            visual_cls = get_longcat_remote_class(
                model_path,
                "modular_longcat_next_visual.LongcatNextVisualTokenizer",
            )
        self.visual_tokenizer = visual_cls(self.config)
        with longcat_timing("image_encoder_load_module", device=device, dtype=torch_dtype):
            self.visual_tokenizer = load_module(
                self.visual_tokenizer,
                model_path,
                prefix="model.visual_tokenizer.",
                dtype=torch_dtype,
                device=device,
                strict=True,
            )
        self.codebook_embedding = _OffsetCodebookEmbedding(
            model_path=model_path,
            offset=int(self.config.visual_offset),
            codebook_sizes=_codebook_sizes(self.config.visual_config.vq_config),
            device=device,
            dtype=torch_dtype,
        )
        self.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, visual_grid_thw: torch.Tensor) -> dict[str, torch.Tensor]:
        device = next(self.visual_tokenizer.parameters()).device
        with longcat_timing("image_encoder_h2d", device=device):
            pixel_values = pixel_values.to(device=device)
            visual_grid_thw = visual_grid_thw.to(device=device)
        with longcat_timing("image_encoder_tokenize", device=device):
            visual_ids = self.visual_tokenizer.encode(pixel_values, visual_grid_thw)
        with longcat_timing("image_encoder_codebook_embedding", device=device):
            visual_embeds = self.codebook_embedding(visual_ids)
        with longcat_timing("image_encoder_projection", device=device):
            visual_embeds = self.visual_tokenizer.visual_embedding_layer(visual_embeds)
        return {
            "visual_ids": visual_ids.detach(),
            "visual_embeds": visual_embeds.reshape(-1, visual_embeds.shape[-1]).detach(),
        }


class LongcatNextAudioEncoder(nn.Module):
    """Audio input encoder: fbank features -> AR hidden-size audio embeddings."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        self.model_path = model_path
        with longcat_timing("audio_encoder_config_init", device=device, dtype=torch_dtype):
            self.config = get_longcat_config(model_path)
        with longcat_timing("audio_encoder_remote_class_init"):
            audio_cls = get_longcat_remote_class(
                model_path,
                "modular_longcat_next_audio.LongcatNextAudioTokenizer",
            )
        self.audio_tokenizer = audio_cls(self.config)
        with longcat_timing("audio_encoder_load_module", device=device, dtype=torch_dtype):
            self.audio_tokenizer = load_module(
                self.audio_tokenizer,
                model_path,
                prefix="model.audio_tokenizer.",
                dtype=torch_dtype,
                device=device,
                strict=True,
            )
        self.codebook_embedding = _OffsetCodebookEmbedding(
            model_path=model_path,
            offset=int(self.config.audio_offset),
            codebook_sizes=_codebook_sizes(self.config.audio_config.vq_config),
            device=device,
            dtype=torch_dtype,
        )
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        encoder_length: torch.Tensor,
        bridge_length: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = next(self.audio_tokenizer.parameters()).device
        with longcat_timing("audio_encoder_h2d", device=device):
            audio = audio.to(device=device)
            encoder_length = encoder_length.to(device=device)
            bridge_length = bridge_length.to(device=device)
        with longcat_timing("audio_encoder_tokenize", device=device):
            audio_ids = self.audio_tokenizer.encode(audio, encoder_length, bridge_length)
        with longcat_timing("audio_encoder_codebook_embedding", device=device):
            audio_embeds = self.codebook_embedding(audio_ids)
        return {
            "audio_ids": audio_ids.detach(),
            "audio_embeds": audio_embeds.reshape(-1, audio_embeds.shape[-1]).detach(),
        }
