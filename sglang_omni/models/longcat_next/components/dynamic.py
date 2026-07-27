# SPDX-License-Identifier: Apache-2.0
"""Dynamic imports for LongCat-Next HuggingFace remote code."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# ── flash-attn v4 compatibility bridge ──────────────────────────────
# sglang 0.5.12.post1 pins flash-attn-4 (CUTE rewrite), which moves the
# v2-era top-level API and bert_padding utilities into sub-modules.
# LongCat-Next's upstream code still imports from the old locations, so
# we re-expose those symbols before any model code is loaded via
# ``get_class_from_dynamic_module`` or ``AutoProcessor.from_pretrained``.
import sys
import types

import flash_attn
import flash_attn.cute as _cute
from flash_attn.cute.testing import (
    index_first_axis,
    pad_input,
    unpad_input,
)

# Top-level functions (were flash_attn.flash_attn_varlen_func etc.)
flash_attn.flash_attn_func = _cute.flash_attn_func
flash_attn.flash_attn_varlen_func = _cute.flash_attn_varlen_func

# flash_attn.bert_padding (was a sub-module in v2, moved to .cute.testing in v4)
_bert_padding = types.ModuleType("flash_attn.bert_padding")
_bert_padding.index_first_axis = index_first_axis
_bert_padding.pad_input = pad_input
_bert_padding.unpad_input = unpad_input
flash_attn.bert_padding = _bert_padding
sys.modules["flash_attn.bert_padding"] = _bert_padding

# ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def get_longcat_remote_class(model_path: str, class_ref: str):
    """Load a class from the model repo's trust_remote_code files.

    ``class_ref`` is e.g. ``"modular_longcat_next_visual.LongcatNextVisualTokenizer"``.
    Using Transformers' dynamic-module loader avoids vendoring the sizeable
    official visual/audio tokenizer implementation into sglang-omni.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception as exc:  # pragma: no cover - import-time env dependent
        raise RuntimeError(
            "Transformers dynamic module utilities are required for LongCat-Next "
            "Phase 2 encoder loading."
        ) from exc

    return get_class_from_dynamic_module(
        class_ref,
        model_path,
        trust_remote_code=True,
    )


def get_longcat_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def get_longcat_processor(model_path: str) -> Any:
    """Return official LongCatNextProcessor when available."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
