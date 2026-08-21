# SPDX-License-Identifier: Apache-2.0
"""LongCat-Next audio head for Phase 3 speech output.

The official OmniAudioHead (modules/image_head.py in the LongCat-Next inference
repo) is a standalone nn.Module with optional TP support.  We re-implement the
non-TP path here to avoid a runtime dependency on the inference repo, with the
same flash-attention-based DepthTransformer architecture.

Reference
---------
* Official implementation:
  LongCat-Next-inference/modules/image_head.py  (OmniAudioHead,
  CasualDepthTransformerLayer, FlashVarLenAttention, RMSNorm)
* Official usage:
  LongCat-Next-inference/modules/output_processor.py  (depth_transformer_forward)
"""

from __future__ import annotations

import os

import torch
from torch import nn

from sglang_omni.models.longcat_next.components.encoders import _OffsetCodebookEmbedding
from sglang_omni.models.longcat_next.payload_types import longcat_timing
from sglang_omni.models.weight_loader import load_module, load_weights_by_prefix, resolve_dtype

# flash_attn v4 compat bridge — ensure flash_attn_func is importable from the
# top-level ``flash_attn`` namespace (moved to ``flash_attn.cute`` in v4).
# ``components/dynamic.py`` also installs this bridge; the guard avoids a
# redundant re-assignment when both modules are loaded.
import flash_attn
if not hasattr(flash_attn, "flash_attn_func"):
    import flash_attn.cute
    flash_attn.flash_attn_func = flash_attn.cute.flash_attn_func


class _RMSNorm(nn.Module):
    """RMSNorm matching the official implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class _FlashVarLenAttention(nn.Module):
    """Self-attention over the codebook-depth dimension with flash-attn.

    When ``enable_tp=False`` (our path), uses plain ``nn.Linear`` projections.
    The forward reshapes ``[B*depth, dims]`` → ``[B*depth, heads, head_dim]``
    and calls ``flash_attn_func`` with ``causal=True``.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, causal: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(-1, 8, self.num_heads, self.head_dim)
        key = key.view(-1, 8, self.num_heads, self.head_dim)
        value = value.view(-1, 8, self.num_heads, self.head_dim)

        attn = flash_attn.flash_attn_func(
            query, key, value, causal=self.causal,
        )
        attn = attn.reshape(B, self.embed_dim)
        return self.out_proj(attn)


class _CasualDepthTransformerLayer(nn.Module):
    """One layer of the codebook-depth causal transformer.

    Attention operates over the 8 codebook positions (causal mask so codebook
    ``i`` can only attend to ``0..i``).  FFN uses an einsum-based reshape when
    ``depth > 1`` (the default for audio with 8 codebooks).
    """

    def __init__(
        self,
        llm_hidden_size: int,
        transformer_dims: int,
        transformer_ffn_scale: int | float,
        depth: int,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.llm_hidden_size = llm_hidden_size
        self.transformer_dims = transformer_dims
        self.transformer_ffn_scale = transformer_ffn_scale

        num_heads = transformer_dims // 128
        assert transformer_dims % 128 == 0
        assert transformer_dims % depth == 0

        self.self_attention = _FlashVarLenAttention(
            transformer_dims, num_heads, causal=True,
        )
        self.layernorm1 = _RMSNorm(transformer_dims)
        self.layernorm2 = _RMSNorm(transformer_dims)

        ffn_intermediate = int(transformer_ffn_scale * transformer_dims)
        self.linear1 = nn.Linear(transformer_dims, ffn_intermediate, bias=True)
        self.linear2 = nn.Linear(ffn_intermediate, transformer_dims, bias=True)

    def forward(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        res = x
        x = self.layernorm1(x)
        _x = self.self_attention(x.reshape(-1, self.transformer_dims))
        _x = _x.view(bsz, self.depth, self.transformer_dims)
        _res = _x + res

        res = self.layernorm2(_res)
        if self.depth > 1:
            # einsum path: reshape weights for depth-aware FFN
            x_out = torch.einsum(
                "bld,tld->blt",
                res,
                self.linear1.weight.reshape(
                    self.transformer_ffn_scale * self.transformer_dims // self.depth,
                    self.depth,
                    self.transformer_dims,
                ),
            )
            x_out = torch.nn.functional.gelu(x_out)
            x_out = torch.einsum(
                "blt,dlt->bld",
                x_out,
                self.linear2.weight.reshape(
                    self.transformer_dims,
                    self.depth,
                    self.transformer_ffn_scale * self.transformer_dims // self.depth,
                ),
            )
        else:
            x_out = self.linear1(res)
            x_out = torch.nn.functional.gelu(x_out)
            x_out = self.linear2(x_out)

        return _res + x_out


class _OmniAudioHead(nn.Module):
    """Minimal non-TP reimplementation of the official OmniAudioHead.

    One forward call produces the logits for a SINGLE codebook level.
    The caller iterates ``codebook_id`` from 0..num_codebooks-1, updating
    ``audio_tokens`` in place for the causal depth dependency.
    """

    def __init__(
        self,
        hidden_size: int,
        codebook_sizes: list[int],
        transformer_ffn_scale: int | float,
        transformer_dims: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = hidden_size
        self.codebook_sizes = list(codebook_sizes)
        self.num_codebooks = len(codebook_sizes)
        self.transformer_dims = transformer_dims

        self.hidden_norm = _RMSNorm(hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, transformer_dims, bias=False)

        self.transformer_layers = nn.ModuleList([
            _CasualDepthTransformerLayer(
                hidden_size, transformer_dims, transformer_ffn_scale,
                self.num_codebooks,
            )
            for _ in range(transformer_layers)
        ])
        self.headnorm = _RMSNorm(transformer_dims)
        # codebook_size + 1 : the extra logit signals "no audio / stop"
        self.heads = nn.ModuleList([
            nn.Linear(transformer_dims, vq_size + 1)
            for vq_size in codebook_sizes
        ])

    def forward(
        self,
        x: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_emb_layers: nn.ModuleList,
        batch_size: int,
        codebook_id: int,
    ) -> torch.Tensor:
        """Predict codebook ``codebook_id`` logits from the LLM hidden state.

        Args:
            x: LLM last-layer hidden state  ``[B, hidden_size]``.
            audio_tokens: codebook tokens so far  ``[B, num_codebooks]``
                (entries ≥ codebook_id may be uninitialised / zero).
            audio_emb_layers: 7 ``nn.Embedding`` layers (one per codebook
                0..num_codebooks-2) sliced from ``embed_tokens``.
            batch_size: ``B``.
            codebook_id: which codebook level to predict (0..7).

        Returns:
            Logits ``[B, codebook_size + 1]`` for this level.
        """
        # Cumulative embedding of previous codebooks (codebooks 0..6)
        cumsum_audio_embed = torch.stack([
            audio_emb_layers[i](audio_tokens[..., i])
            for i in range(self.num_codebooks - 1)
        ], dim=1)
        cumsum_audio_embed = torch.cumsum(cumsum_audio_embed, dim=1)

        # [B, num_codebooks, hidden_size] — pos 0 is LLM hidden, pos 1.. are cumsum embeds
        hidden_states = torch.cat([
            x.reshape(-1, 1, self.llm_hidden_size),
            cumsum_audio_embed,
        ], dim=1)
        assert hidden_states.size(1) == self.num_codebooks, (
            f"depth mismatch: {hidden_states.size(1)} vs {self.num_codebooks}"
        )

        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = self.hidden_proj(hidden_states)

        for tlayer in self.transformer_layers:
            hidden_states = tlayer(hidden_states, batch_size)

        hidden_states = self.headnorm(hidden_states)
        return self.heads[codebook_id](hidden_states[:, codebook_id])


class LongcatNextAudioHead(nn.Module):
    """Phase 3 audio head: 8-step causal codebook prediction.

    Wraps ``_OmniAudioHead`` with weight loading and the 8-step decode loop.
    """

    def __init__(
        self,
        model_path: str,
        *,
        hidden_size: int,
        codebook_sizes: list[int],
        transformer_ffn_scale: int | float,
        transformer_dims: int,
        transformer_layers: int,
        audio_offset: int,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        self.num_codebooks = len(codebook_sizes)
        self.codebook_sizes = list(codebook_sizes)

        with longcat_timing("audio_head_init", device=device, dtype=torch_dtype):
            self.audio_head = _OmniAudioHead(
                hidden_size=hidden_size,
                codebook_sizes=codebook_sizes,
                transformer_ffn_scale=transformer_ffn_scale,
                transformer_dims=transformer_dims,
                transformer_layers=transformer_layers,
            )

        with longcat_timing("audio_head_load_weights", device=device):
            load_module(
                self.audio_head,
                model_path,
                prefix="audio_head.",
                dtype=torch_dtype,
                device=device,
                strict=True,
            )

        with longcat_timing("audio_head_emb_layers_load"):
            # audio_emb_layers are 7 (num_codebooks-1) embedding slices from
            # model.embed_tokens.weight, starting at audio_offset.  The last
            # codebook (index 7) does not need its own embedding because the
            # cumulative context only covers codebooks 0..6.
            self.audio_emb_layers = _build_codebook_emb_layers(
                model_path=model_path,
                offset=audio_offset,
                codebook_sizes=codebook_sizes[:-1],
                device=device,
                dtype=torch_dtype,
            )

        # input_codebook_embedding sums ALL 8 codebook embeddings from
        # embed_tokens.weight (like Phase 2 encoder _OffsetCodebookEmbedding).
        # This provides the audio contribution fused into the LLM input
        # embedding at the next decode step.
        with longcat_timing("audio_head_input_emb_load"):
            self.input_codebook_embedding = _OffsetCodebookEmbedding(
                model_path=model_path,
                offset=audio_offset,
                codebook_sizes=codebook_sizes,
                device=device,
                dtype=torch_dtype,
            )

        self.eval()

        # Optional CUDA-graph capture of the 8-step decode loop, keyed by batch
        # size. Default off; enable via SGLANG_OMNI_LONGCAT_AUDIO_HEAD_CUDA_GRAPH.
        self._graph_enabled = os.getenv(
            "SGLANG_OMNI_LONGCAT_AUDIO_HEAD_CUDA_GRAPH", ""
        ).lower() in ("1", "true", "yes", "on")
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_in: dict[int, torch.Tensor] = {}
        self._graph_out: dict[int, torch.Tensor] = {}

    def _decode_loop(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pure 8-step causal codebook prediction (graph-capturable).

        Depends only on ``hidden_state``: causal masking makes codebook ``k``
        attend to cols ``0..k-1`` (written earlier in this loop), so the codes
        buffer is zero-initialised and never reads external state.
        """
        bs = hidden_state.shape[0]
        codes = torch.zeros(
            bs, self.num_codebooks, dtype=torch.long, device=hidden_state.device,
        )
        for codebook_id in range(self.num_codebooks):
            logits = self.audio_head(
                hidden_state, codes, self.audio_emb_layers, bs, codebook_id,
            )
            codes[:, codebook_id] = torch.argmax(logits, dim=-1)
        return codes

    def _decode_loop_graphed(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run :meth:`_decode_loop` through a per-batch-size CUDA graph."""
        bs = hidden_state.shape[0]
        graph = self._graphs.get(bs)
        if graph is None:
            # Warm up on a side stream, then capture (standard PyTorch recipe).
            static_in = hidden_state.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._decode_loop(static_in)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._decode_loop(static_in)
            self._graphs[bs] = graph
            self._graph_in[bs] = static_in
            self._graph_out[bs] = static_out

        self._graph_in[bs].copy_(hidden_state)
        graph.replay()
        # Clone so callers own the result independent of the static buffer.
        return self._graph_out[bs].clone()

    @torch.no_grad()
    def forward(
        self,
        hidden_state: torch.Tensor,
        prev_audio_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the 8-step causal codebook prediction loop.

        Args:
            hidden_state: LLM last-layer output  ``[B, hidden_size]``.
            prev_audio_codes: unused (kept for API compatibility); the loop is
                self-contained given ``hidden_state`` thanks to causal masking.

        Returns:
            Predicted codes  ``[B, num_codebooks]`` (int64, on the same device).
        """
        del prev_audio_codes
        if self._graph_enabled and hidden_state.is_cuda:
            return self._decode_loop_graphed(hidden_state)
        return self._decode_loop(hidden_state)

    @torch.no_grad()
    def build_input_embedding(
        self, audio_codes: torch.Tensor
    ) -> torch.Tensor:
        """Sum 8 codebook embeddings for LLM input fusion.

        Args:
            audio_codes: ``[B, 8]`` codebook token ids (int64).

        Returns:
            ``[B, hidden_size]`` summed embedding, same dtype as the model.
        """
        return self.input_codebook_embedding(audio_codes)


def _build_codebook_emb_layers(
    *,
    model_path: str,
    offset: int,
    codebook_sizes: list[int],
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.ModuleList:
    """Build per-codebook embedding layers from sliced ``embed_tokens.weight``."""
    state = load_weights_by_prefix(model_path, prefix="model.embed_tokens.")
    weight = state["weight"]
    layers = []
    start = int(offset)
    for size in codebook_sizes:
        stop = start + int(size)
        emb_weight = weight[start:stop].to(dtype=dtype)
        layers.append(nn.Embedding.from_pretrained(emb_weight, freeze=True))
        start = stop
    return nn.ModuleList(layers).to(device=device, dtype=dtype)
