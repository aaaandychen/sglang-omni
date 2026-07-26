# SPDX-License-Identifier: Apache-2.0
"""Dynamic imports for LongCat-Next HuggingFace remote code."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


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
