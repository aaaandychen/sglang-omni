# SPDX-License-Identifier: Apache-2.0
"""SGLang-native model wrapper for LongCat-Next text-only AR backbone.

LongCat-Next's backbone (LongCat-Flash-Lite MoE A3B) is architecturally
identical to LongCat-Flash at the layer level — MLA attention, dense + MoE
FFN blocks, RMSNorm — differing only in scale (14 layers vs 28, 3072-dim vs
6144, 256 experts vs 512).

This class inherits SGLang's ``LongcatFlashForCausalLM`` and overrides two
things:

1. **Config field-name mapping** — LongCat-Next HF config uses
   ``ffn_hidden_size`` / ``expert_ffn_hidden_size`` / ``num_layers``,
   but the SGLang model expects ``intermediate_size`` /
   ``moe_intermediate_size`` / ``num_hidden_layers``.  We map them
   before ``super().__init__()``.

2. **load_weights** — filters out multimodal component weights (visual /
   audio tokenizer and head layers) before delegating to the parent, and
   applies a defensive ``[:full_vocab]`` truncation to ``embed_tokens``
   and ``lm_head`` weights (same pattern as the official LongCat-Next
   inference backend).

Reference
---------
* Official LongCat-Next inference: ``modules/nmm_flash.py``
  (``NmmFlashForCausalLM`` inherits ``FLASHForCausalLM`` from
  ``sglang.srt.models.longcat_flash`` — same parent class).
* SGLang upstream: ``sglang/srt/models/longcat_flash.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Tuple

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM, LongcatFlashMoE

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


# ── zero-expert / non-EP illegal-memory-access fix ───────────────────────
# LongCat-Next's MoE has 256 routed experts + 128 ``identity`` zero-experts;
# the router emits 384-wide logits and top-k picks 12.  SGLang's
# ``LongcatFlashMoE.forward`` calls ``zero_experts_compute_triton`` which
# rewrites every zero-expert slot in ``topk_idx`` to ``-1`` in place (and
# zeroes its combine weight), then hands ``topk_idx`` to a plain FusedMoE.
#
# The Triton ``fused_moe_kernel`` only skips ``off_experts == -1`` blocks
# when ``filter_expert`` is true, and ``filter_expert`` is
# ``num_experts != num_local_experts`` — which is *false* under pure TP
# (every rank owns all 256 experts).  So the ``-1`` slots are not skipped
# and the kernel indexes ``w13_weight[-1]`` → CUDA illegal memory access.
#
# Fix: clamp the ``-1`` slots to expert 0 before they reach ``self.experts``.
# Their combine weight is already 0 (set by ``zero_experts_compute_triton``),
# so the down-projection multiplies their contribution by 0 — numerically
# identical to skipping them, but without the out-of-bounds index.
def _longcat_moe_forward_zero_expert_safe(
    self: LongcatFlashMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    from sglang.srt.models.longcat_flash import (
        StandardTopKOutput,
        tensor_model_parallel_all_reduce,
        zero_experts_compute_triton,
    )

    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.router(hidden_states)
    topk_weights, topk_idx, _ = self.topk(hidden_states, router_logits)

    zero_expert_result = None
    if self.zero_expert_type is not None:
        # Rewrites zero-expert slots in topk_idx to -1 and zeroes their weight.
        zero_expert_result = zero_experts_compute_triton(
            expert_indices=topk_idx,
            expert_scales=topk_weights,
            num_experts=self.num_experts,
            zero_expert_type=self.zero_expert_type,
            hidden_states=hidden_states,
        )
        # Clamp -1 (zero-expert / filtered) slots to a valid expert index so
        # the FusedMoE kernel never dereferences w[-1].  These slots carry a
        # zero combine weight, so their contribution stays exactly 0.
        topk_idx = topk_idx.masked_fill(topk_idx < 0, 0)

    topk_output = StandardTopKOutput(topk_weights, topk_idx, _)

    final_hidden_states = self.experts(hidden_states, topk_output)
    final_hidden_states *= self.routed_scaling_factor

    if zero_expert_result is not None and hidden_states.shape[0] > 0:
        final_hidden_states += zero_expert_result.to(final_hidden_states.device)

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states.view(num_tokens, hidden_dim)


# Install the patched forward once, at import time.
LongcatFlashMoE.forward = _longcat_moe_forward_zero_expert_safe

# Multimodal components present in the full LongCat-Next checkpoint but
# unused by the text-only AR backbone.  Their weights are filtered out in
# ``load_weights`` before the parent processes them.
_MULTIMODAL_SKIP_PREFIXES: tuple[str, ...] = (
    "audio_head.",
    "model.audio_tokenizer.",
    "visual_head.",
    "model.visual_tokenizer.",
    "model.audio_embed_layers.",
)


class LongcatNextTextForCausalLM(LongcatFlashForCausalLM):
    """LongCat-Next AR backbone for sglang-omni (text-only, Phase 1)."""

    # With config.vocab_size = full_vocab (131 125), both embed_tokens
    # and lm_head have the same shape.  The parent's default weight tying
    # (_tied_weights_keys = ["lm_head.weight"]) correctly ties them, so
    # we no longer need to override.
    _tied_weights_keys: list[str] = ["lm_head.weight"]

    # ── __init__ ────────────────────────────────────────────────────────

    def __init__(
        self,
        config: object,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Resolve the two vocabulary sizes from the HF config.
        text_vocab: int = int(
            getattr(config, "text_vocab_size", config.vocab_size)
        )
        full_vocab: int = int(
            getattr(
                config,
                "text_vocab_plus_multimodal_special_token_size",
                text_vocab,
            )
        )

        # ── Ngram embedding config derivation ────────────────────────
        # LongCat-Next's input embedding is NgramEmbedding (word embedding
        # plus 12 n-gram projection terms, meaned over 13 tensors).  The
        # stock LongcatFlashModel.__init__ builds it when config.use_ngram_
        # embedding is True and reads config.ngram_embedding_{m,k,n}.
        #
        # SGLang's own LongcatFlashConfig derives these from
        # ngram_vocab_size_ratio / emb_neighbor_num / emb_split_num.  But
        # LongCat-Next ships a LongcatNextConfig (trust_remote_code) that
        # carries only the RAW fields, not the derived ones — so we derive
        # them here, mirroring configs/longcat_flash.py.
        #
        # CRITICAL: ngram_embedding_m (the n-gram hash modulus base) must be
        # derived from the TEXT vocab (131072), NOT the full vocab (131125)
        # which includes 53 multimodal special tokens.  Verified against the
        # checkpoint: embedders.0.weight has 10223617 = int(78*131072)+1 rows.
        # Using full_vocab would break the load-weight shape assertion.
        _ngram_ratio = getattr(config, "ngram_vocab_size_ratio", None)
        if _ngram_ratio is not None and _ngram_ratio > 0:
            config.use_ngram_embedding = True
            config.ngram_embedding_m = int(_ngram_ratio * text_vocab)
            config.ngram_embedding_n = int(getattr(config, "emb_neighbor_num", 4))
            config.ngram_embedding_k = int(getattr(config, "emb_split_num", 4))
            logger.info(
                "LongCat-Next: ngram embedding enabled "
                "(m=%d, n=%d, k=%d, word_table=%d, hash_base=%d)",
                config.ngram_embedding_m,
                config.ngram_embedding_n,
                config.ngram_embedding_k,
                full_vocab,
                text_vocab,
            )
        else:
            config.use_ngram_embedding = False

        # ── Config field-name compatibility ──────────────────────────
        # LongCat-Next HF config uses different field names than the
        # sglang 0.5.10 longcat_flash.py expects.  ModelConfig may
        # already have these attributes with wrong defaults, so we
        # unconditionally override from the HF config source fields.
        if not hasattr(config, "use_ngram_embedding"):
            config.use_ngram_embedding = False
        config.intermediate_size = getattr(config, "ffn_hidden_size", None) or 6144
        config.moe_intermediate_size = getattr(config, "expert_ffn_hidden_size", None) or 1024
        config.num_hidden_layers = getattr(config, "num_layers", None) or 14
        if not hasattr(config, "hidden_act") or not getattr(config, "hidden_act", None):
            config.hidden_act = "silu"
        if not hasattr(config, "rope_parameters"):
            config.rope_parameters = {"rope_theta": config.rope_theta}
        if not hasattr(config, "router_bias") or getattr(config, "router_bias", None) is None:
            config.router_bias = False
        if not hasattr(config, "rounter_params_dtype"):
            config.rounter_params_dtype = "bfloat16"

        # ── Create parent with full vocab (embed_tokens + lm_head) ──
        # Use full_vocab (131125) so that embed_tokens.org_vocab_size
        # matches the checkpoint weight shape.  The extra multimodal
        # embedding rows are never activated by text-only inputs.
        saved_vocab = int(config.vocab_size)
        config.vocab_size = full_vocab
        try:
            super().__init__(
                config, quant_config=quant_config, prefix=prefix
            )
        finally:
            config.vocab_size = saved_vocab

        # lm_head is already created by the parent with
        # config.vocab_size = full_vocab (131125) — no replacement needed.

        # ── Fix ngram hash base (text_vocab, not word-table size) ─────
        # The parent built NgramEmbedding with num_embeddings=full_vocab
        # (131125) so word_embeder has the right row count.  But the venv
        # NgramEmbedding uses that SAME num_embeddings as the n-gram hash
        # base in pow(num_embeddings, delta, mod) (n_gram_embedding.py),
        # whereas the model was trained with the TEXT vocab (131072) as the
        # base — the two differ by the 53 multimodal special tokens.  A
        # wrong base yields entirely different oe_weights → wrong n-gram ids
        # → garbage output.  Recompute oe_weights (a plain, non-persistent
        # CUDA buffer) with base=text_vocab.  Mirrors the official fork's
        # FusedOverEmbedding num_embeddings_text parameter (over_embedding.py).
        if getattr(config, "use_ngram_embedding", False):
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            for ng in self.modules():
                if not isinstance(ng, NgramEmbedding):
                    continue
                N = ng.over_embedding_n
                K = ng.over_embedding_k
                M = ng.over_embedding_m
                for n in range(2, N + 1):
                    for k in range(K):
                        mod = M + 2 * ((n - 2) * K + k) + 1
                        ng.oe_mods[n - 2][k] = mod
                        for delta in range(N):
                            ng.oe_weights[n - 2][k][delta] = pow(
                                text_vocab, delta, mod
                            )
                logger.info(
                    "LongCat-Next: recomputed oe_weights with hash_base=%d",
                    text_vocab,
                )

    # ── multimodal prefill forward ─────────────────────────────────────

    def _build_longcat_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Compute embeddings with LongCat n-gram support, then patch pads.

        SGLang versions differ in the exact NgramEmbedding call signature.  Try
        the forward-batch aware forms first so Phase 1's token-table path stays
        active; fall back to the plain embedding call for non-ngram configs.
        """
        embed_ids = input_ids
        replace_positions = getattr(forward_batch, "longcat_replace_positions", None)
        if replace_positions is not None and len(replace_positions) > 0:
            embed_ids = input_ids.clone()
            embed_ids[replace_positions.to(embed_ids.device)] = 0

        embed_tokens = self.model.embed_tokens
        try:
            input_embeds = embed_tokens(embed_ids, forward_batch)
        except TypeError:
            try:
                input_embeds = embed_tokens(embed_ids, forward_batch=forward_batch)
            except TypeError:
                input_embeds = embed_tokens(embed_ids)

        replace_embeds = getattr(forward_batch, "longcat_replace_embeds", None)
        if replace_positions is not None and replace_embeds is not None:
            pos = replace_positions.to(input_embeds.device)
            input_embeds[pos] = replace_embeds.to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        replace_embeds = getattr(forward_batch, "longcat_replace_embeds", None)
        if replace_embeds is None:
            return super().forward(input_ids, positions, forward_batch)

        input_embeds = self._build_longcat_input_embeds(input_ids, forward_batch)
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # ── load_weights ────────────────────────────────────────────────────

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        """Filter multimodal weights, then delegate to the parent.

        Mirror of the official ``NmmFlashForCausalLM.load_weights``
        (modules/nmm_flash.py in the LongCat-Next inference repo).
        The checkpoint contains both ``q_a_proj`` and ``kv_a_proj_with_mqa``
        weights → SGLang's native Q/KV fusion works correctly.
        """
        full_vocab: int = int(
            getattr(
                self.config,
                "text_vocab_plus_multimodal_special_token_size",
                self.config.vocab_size,
            )
        )

        _total_ckpt = 0
        _skipped_mm = 0
        filtered: list[Tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            _total_ckpt += 1
            if any(
                name.startswith(prefix)
                for prefix in _MULTIMODAL_SKIP_PREFIXES
            ):
                _skipped_mm += 1
                continue

            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                weight = weight[:full_vocab]

            filtered.append((name, weight))

        # ── Before delegating: snapshot model params for post-load diff ──
        _before = {n: p.sum().item() for n, p in self.named_parameters()}

        super().load_weights(iter(filtered))

        # ── After delegating: detect params that stayed at their init value ──
        _unchanged = []
        _loaded_keys = set()
        for n, p in self.named_parameters():
            before_sum = _before.get(n)
            after_sum = p.sum().item()
            if before_sum is not None and abs(after_sum - before_sum) < 1e-12:
                _unchanged.append(n)
            elif after_sum != 0:
                _loaded_keys.add(n)

        logger.info(
            "LongCat-Next load_weights: ckpt_keys=%d mm_skipped=%d "
            "filtered=%d unchanged=%d loaded=%d",
            _total_ckpt, _skipped_mm, len(filtered),
            len(_unchanged), len(_loaded_keys),
        )
        if _unchanged:
            logger.warning(
                "LongCat-Next: %d params UNCHANGED after weight load "
                "(possibly not loaded): %s",
                len(_unchanged),
                ", ".join(_unchanged[:10]),
            )
            if len(_unchanged) > 10:
                logger.warning(
                    "  ... and %d more unchanged params", len(_unchanged) - 10
                )


EntryClass = LongcatNextTextForCausalLM
