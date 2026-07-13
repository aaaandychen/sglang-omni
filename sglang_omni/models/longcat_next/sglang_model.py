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
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

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

        # ── Ngram config compatibility ───────────────────────────────
        # SGLang's LongcatFlashModel.__init__ reads:
        #   config.use_ngram_embedding        (→ whether to use NgramEmbedding)
        #   config.ngram_embedding_{m,k,n}    (→ NgramEmbedding constructor params)
        #
        # SGLang's own LongcatFlashConfig computes these from
        # ngram_vocab_size_ratio / emb_neighbor_num / emb_split_num,
        # which are also the field names used by LongCat-Next's HF config.
        # When the HF config is loaded via trust_remote_code=True,
        # LongcatFlashNgramConfig exposes them as derived properties, so
        # ngram should work out of the box.
        #
        # If it doesn't (e.g. stripped config or name mismatch), disable
        # ngram before super().__init__() — once the model is created the
        # embedding type is baked in and cannot be changed.
        _want_ngram = bool(getattr(config, "use_ngram_embedding", False))
        _has_ngram_params = all(
            hasattr(config, attr)
            for attr in (
                "ngram_embedding_m",
                "ngram_embedding_k",
                "ngram_embedding_n",
            )
        )
        if _want_ngram and not _has_ngram_params:
            logger.warning(
                "LongCat-Next: ngram embedding requested but config is "
                "missing ngram_embedding_{m,k,n}.  Falling back to "
                "standard VocabParallelEmbedding."
            )
            # Approach 1: set use_ngram_embedding=False on ModelConfig
            # (it's a plain instance attribute, not a property).
            try:
                config.use_ngram_embedding = False
            except AttributeError:
                pass
            # Approach 2: if use_ngram_embedding is derived from
            # ngram_vocab_size_ratio (as in LongcatFlashConfig),
            # zeroing the ratio also disables it.
            if getattr(config, "use_ngram_embedding", False):
                try:
                    config.ngram_vocab_size_ratio = 0
                except AttributeError:
                    pass
            # If both approaches failed, let super().__init__() raise —
            # the error message will point to the missing field.

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
