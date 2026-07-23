# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the consolidated weka prompt-synthesis primitives."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.hash_ids_synthesis import HashIdsPromptSynthesisMixin
from aiperf.dataset.loader.weka_synth_buf import compose_weka_prompt_tokens

BLOCK_SIZE = 64


def _block_stub(hids: list[int]) -> list[int]:
    """Return BLOCK_SIZE deterministic, distinct tokens per hash_id."""
    return [10000 + (h * 1000) + i for h in hids for i in range(BLOCK_SIZE)]


def _tail_stub(n: int, seed: str) -> list[int]:
    """Position-keyed: same (n, seed) -> same bytes; here independent of seed"""
    return [99000 + i for i in range(n)]


def test_compose_empty_hash_ids_uses_full_tail():
    out = compose_weka_prompt_tokens(
        hash_ids=[],
        input_length=10,
        decode_block_tokens=_block_stub,
        sample_partial_tail_tokens=_tail_stub,
        seed="s",
    )
    assert out == [99000 + i for i in range(10)]


def test_compose_exact_tile_no_tail():
    """input_length == M * block_size -> hashed prefix only, no tail."""
    out = compose_weka_prompt_tokens(
        hash_ids=[1, 2],
        input_length=2 * BLOCK_SIZE,
        decode_block_tokens=_block_stub,
        sample_partial_tail_tokens=_tail_stub,
        seed="s",
    )
    assert out == _block_stub([1, 2])


def test_compose_last_block_partial_truncates_prefix():
    """input_length < M * block_size -> truncate the hashed prefix."""
    out = compose_weka_prompt_tokens(
        hash_ids=[1, 2, 3],
        input_length=130,
        decode_block_tokens=_block_stub,
        sample_partial_tail_tokens=_tail_stub,
        seed="s",
    )
    assert len(out) == 130
    assert out == _block_stub([1, 2, 3])[:130]


def test_compose_prefix_only_appends_tail():
    """input_length > M * block_size -> append sha256-keyed partial tail"""
    out = compose_weka_prompt_tokens(
        hash_ids=[1, 2],
        input_length=200,
        decode_block_tokens=_block_stub,
        sample_partial_tail_tokens=_tail_stub,
        seed="s",
    )
    assert len(out) == 200
    assert out[:128] == _block_stub([1, 2])
    assert out[128:] == [99000 + i for i in range(72)]


def test_compose_zero_length_with_empty_hash_ids():
    """Edge: input_length=0 -> empty result."""
    out = compose_weka_prompt_tokens(
        hash_ids=[],
        input_length=0,
        decode_block_tokens=_block_stub,
        sample_partial_tail_tokens=_tail_stub,
        seed="s",
    )
    assert out == []


def _mixin_with_corpus(size: int = 1000) -> HashIdsPromptSynthesisMixin:
    """Construct a HashIdsPromptSynthesisMixin instance with a deterministic"""

    class _Holder(HashIdsPromptSynthesisMixin):
        pass

    m = _Holder()
    pg = MagicMock()
    pg._corpus_size = size
    pg._tokenized_corpus = list(range(10000, 10000 + size))
    m.prompt_generator = pg
    return m


def test_partial_tail_same_seed_same_bytes():
    m = _mixin_with_corpus()
    a = m.sample_partial_tail_tokens(50, "trace-A:turn_0:prompt_tail")
    b = m.sample_partial_tail_tokens(50, "trace-A:turn_0:prompt_tail")
    assert a == b


def test_partial_tail_different_seed_different_bytes():
    m = _mixin_with_corpus()
    a = m.sample_partial_tail_tokens(50, "trace-A:turn_0:prompt_tail")
    b = m.sample_partial_tail_tokens(50, "trace-A:turn_1:prompt_tail")
    c = m.sample_partial_tail_tokens(50, "trace-B:turn_0:prompt_tail")
    assert a != b
    assert a != c
    assert b != c


def test_partial_tail_zero_length_is_empty():
    m = _mixin_with_corpus()
    assert m.sample_partial_tail_tokens(0, "any-seed") == []


@pytest.mark.parametrize("n", [1, 50, 256, 999])
def test_partial_tail_returns_exact_length(n):
    m = _mixin_with_corpus(size=1000)
    out = m.sample_partial_tail_tokens(n, "seed")
    assert len(out) == n
