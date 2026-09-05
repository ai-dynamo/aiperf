# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone weka prompt-token composer (hash blocks + sha256-keyed tail)."""

from __future__ import annotations

from collections.abc import Callable


def compose_weka_prompt_tokens(
    *,
    hash_ids: list[int],
    input_length: int,
    decode_block_tokens: Callable[[list[int]], list[int]],
    sample_partial_tail_tokens: Callable[[int, str], list[int]],
    seed: str,
) -> list[int]:
    """Build the prompt token sequence for a weka turn.

    Replaces the ``synthesize_prompts_from_hash_ids`` parent-side phase: the
    same hash-id-seeded RNG used by :class:`ConversationReconstructor` for
    LCP segments is reused here for the prompt itself, so workers can
    produce both byte-deterministically without a separate ``parallel_decode``
    pool.

    Three layouts:

    - ``hash_ids`` empty: prompt is entirely a sha256-keyed sample of length
      ``input_length``.
    - ``input_length <= len(hash_ids) * block_size``: exact-tile or
      last-block-partial; truncate the hashed prefix to ``input_length``.
      Byte-identical to ``_build_token_sequence``'s last-block-partial path
      because ``sample_tokens_from_corpus`` calls ``randrange`` exactly once
      regardless of block size, so prefix truncation matches.
    - ``input_length > len(hash_ids) * block_size``: prefix-only — append a
      sha256-keyed partial tail. Byte content of the tail differs from
      ``_build_token_sequence``'s order-dependent ``_corpus_rng`` path; the
      sha256-keyed seed makes the tail position-deterministic and
      reproducible across processes.
    """
    if not hash_ids:
        return sample_partial_tail_tokens(input_length, seed)
    block_tokens = decode_block_tokens(hash_ids)
    if input_length <= len(block_tokens):
        return block_tokens[:input_length]
    tail = input_length - len(block_tokens)
    return block_tokens + sample_partial_tail_tokens(tail, seed)
