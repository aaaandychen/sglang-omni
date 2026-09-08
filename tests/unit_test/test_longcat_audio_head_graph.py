# SPDX-License-Identifier: Apache-2.0
"""Unit tests for audio-head CUDA-graph batch-size bucketing.

The graph path itself needs CUDA; the bucket-mapping logic is pure Python and
tested here without instantiating weights (``__init__`` is bypassed).
"""

from __future__ import annotations

import pytest

pytest.importorskip("flash_attn", reason="audio_head module requires flash_attn")

from sglang_omni.models.longcat_next.components.audio_head import (
    _GRAPH_BS_LADDER,
    LongcatNextAudioHead,
)


def _bare_head(max_bs: int = 128) -> LongcatNextAudioHead:
    head = object.__new__(LongcatNextAudioHead)
    head._graph_max_bs = max_bs
    return head


def test_bucket_maps_to_next_ladder_step() -> None:
    head = _bare_head()
    assert head._graph_bucket(1) == 1
    assert head._graph_bucket(2) == 2
    assert head._graph_bucket(3) == 4
    assert head._graph_bucket(5) == 8
    assert head._graph_bucket(128) == 128


def test_bucket_beyond_cap_falls_back_to_eager() -> None:
    head = _bare_head(max_bs=32)
    assert head._graph_bucket(16) == 16
    assert head._graph_bucket(33) is None
    assert head._graph_bucket(128) is None


def test_ladder_is_bounded() -> None:
    # The whole point: at most len(ladder) graphs ever exist.
    assert len(_GRAPH_BS_LADDER) <= 8
    assert list(_GRAPH_BS_LADDER) == sorted(_GRAPH_BS_LADDER)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
