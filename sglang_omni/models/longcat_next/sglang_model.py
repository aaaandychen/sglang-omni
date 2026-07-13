# SPDX-License-Identifier: Apache-2.0
"""SGLang-native model wrapper for LongCat-Next text-only AR backbone.

LongCat-Next's backbone (LongCat-Flash-Lite MoE A3B) is architecturally
identical to LongCat-Flash at the layer level — MLA attention, dense + MoE
FFN blocks, RMSNorm — differing only in scale (14 layers vs 28, 3072-dim vs
6144, 256 experts vs 512).

This class inherits SGLang's ``LongcatFlashForCausalLM`` and overrides two
things:

1. **lm_head vocab size** — the parent creates both ``embed_tokens`` and
   ``lm_head`` from ``config.vocab_size``.  We set ``config.vocab_size =
   text_vocab_size`` before ``super().__init__`` so the embedding table
   uses the text-only vocabulary (131 072), then replace ``self.lm_head``
   with a head sized to ``text_vocab_plus_multimodal_special_token_size``
   (131 125) so the full checkpoint can be loaded without truncation.

2. **load_weights** — filters out multimodal component weights (visual /
   audio tokenizer and head layers) before delegating to the parent, and
   applies a defensive ``[:131125]`` truncation to ``embed_tokens`` and
   ``lm_head`` weights (same pattern as the official LongCat-Next
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
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
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

    # Parent declares ``_tied_weights_keys = ["lm_head.weight"]``.
    # embed_tokens has 131 072 rows, lm_head has 131 125 rows — they
    # cannot be tied.  Override to an empty list so ``post_init()`` /
    # ``tie_weights()`` is a no-op.
    _tied_weights_keys: list[str] = []

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
        hidden_size: int = int(config.hidden_size)

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

        # ── Create parent with text-only vocab ───────────────────────
        saved_vocab = int(config.vocab_size)
        config.vocab_size = text_vocab
        try:
            super().__init__(
                config, quant_config=quant_config, prefix=prefix
            )
        finally:
            config.vocab_size = saved_vocab

        # ── Replace lm_head with full-vocab version ──────────────────
        if full_vocab != text_vocab:
            from sglang.srt.server_args import get_global_server_args

            use_attn_tp = bool(
                getattr(get_global_server_args(), "enable_dp_lm_head", False)
            )
            head_prefix = f"{prefix}lm_head" if prefix else "lm_head"
            self.lm_head = ParallelLMHead(
                full_vocab,
                hidden_size,
                quant_config=quant_config,
                prefix=head_prefix,
                use_attn_tp_group=use_attn_tp,
            )

    # ── load_weights ────────────────────────────────────────────────────

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        """Filter multimodal weights, then delegate to the parent.

        Mirror of the official ``NmmFlashForCausalLM.load_weights``
        (modules/nmm_flash.py in the LongCat-Next inference repo).
        """
        full_vocab: int = int(
            getattr(
                self.config,
                "text_vocab_plus_multimodal_special_token_size",
                self.config.vocab_size,
            )
        )

        filtered: list[Tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            # Drop multimodal component weights.
            if any(
                name.startswith(prefix)
                for prefix in _MULTIMODAL_SKIP_PREFIXES
            ):
                continue

            # Defensive truncation: the official code applies
            # ``weight[:131125]`` to both embed_tokens and lm_head.
            # For embed_tokens (131 072 rows) this is a no-op; for
            # lm_head it guards against unexpected padding rows.
            if name == "model.embed_tokens.weight":
                weight = weight[:full_vocab]
            elif name == "lm_head.weight":
                weight = weight[:full_vocab]

            filtered.append((name, weight))

        # Parent handles: MLA (fused QKV), MoE (stacked params +
        # expert mapping), ngram embeddings (routes
        # ``model.ngram_embeddings.*`` → ``NgramEmbedding.load_weight``),
        # and all standard backbone layers.
        super().load_weights(iter(filtered))


EntryClass = LongcatNextTextForCausalLM
