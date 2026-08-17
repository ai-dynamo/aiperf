# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared builders for the segment trie, pool, and store tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from aiperf.dataset.graph.segment_trie.trie_content import (
    ReconCallbacks,
    TrieNode,
    TrieRequest,
)

DYNAMO_NESTED_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


def trie_node(
    node_id: str,
    *,
    hash_ids: Sequence[int] = (),
    input_length: int = 0,
    output_length: int = 0,
    t: float = 0.0,
    api_time: float = 1.0,
    order: int = 0,
    async_ancestors: frozenset[str] | None = None,
    warped_start: float | None = None,
) -> TrieNode:
    """A TrieNode wrapping one recorded request, with everything defaulted."""
    node = TrieNode(
        node_id=node_id,
        request=TrieRequest(
            hash_ids=list(hash_ids),
            input_length=input_length,
            output_length=output_length,
            t=t,
            api_time=api_time,
        ),
        order=order,
        async_ancestors=async_ancestors or frozenset(),
    )
    if warped_start is not None:
        node.warped_start = warped_start
    return node


def stub_recon_callbacks(
    *,
    tokens_per_block: int = 4,
    block_exact: bool = True,
    decode_block_tokens: Callable[[list[int]], list[int]] | None = None,
) -> ReconCallbacks:
    """Deterministic collision-free recon stubs: each block decodes to ``[hash_id] * tokens_per_block``."""
    return ReconCallbacks(
        decode_block_tokens=decode_block_tokens
        or (lambda hids: [t for h in hids for t in [h] * tokens_per_block]),
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
        block_exact=block_exact,
    )
