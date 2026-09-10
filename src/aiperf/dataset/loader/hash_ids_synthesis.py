# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared hash_ids -> decoded prompt synthesis used by Weka and trace loaders.

The pipeline:

1. Build a token sequence for each requested (hash_ids, input_length) pair
   (fills ``PromptGenerator._cache`` with unique hash-block token IDs).
2. Parallel-decode unique **pieces**: first blocks, one-token-overlap
   continuations, and prefix-only tails — not every full ISL sequence.
3. Stitch piece strings so the result equals ``decode(full sequence)``.

Naive ``"".join(decode(block))`` differs from ``decode(concat(blocks))`` at
segment boundaries (see ``PromptGenerator._determine_bpe_stable_terminator``).
One-token left context makes the continuation a prefix-stable suffix of the
full decode for HuggingFace ``skip_special_tokens=False`` and the unit-test
mock tokenizer. If ``_cache`` is not a real ``dict`` of token lists, fall back
to unique full-sequence decode.
"""

from __future__ import annotations

import array
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass

from aiperf.dataset.generator.parallel_decode import parallel_decode


@dataclass(slots=True)
class HashIdsPromptRequest:
    """One synthesis request identified by an opaque key."""

    key: str
    """Caller-provided identifier; the returned dict keys by this."""

    hash_ids: list[int]
    """Block hash IDs. Empty = fall back to PromptGenerator.generate."""

    input_length: int
    """Target input token count."""


def _unique_sequence_decode_workers() -> int:
    """Cap decode workers high enough for large unique-sequence batches."""
    return min(os.cpu_count() or 4, 64)


def _sequence_fingerprint(tokens: list[int]) -> bytes:
    """Compact identity for an assembled token list (not a full tuple copy)."""
    packed = array.array("q", tokens)
    return hashlib.blake2b(packed, digest_size=16).digest()


def intern_token_sequence(
    tokens: list[int],
    unique_sequences: list[list[int]],
    fingerprint_buckets: dict[bytes, list[int]],
) -> int:
    """Store ``tokens`` once; return its index in ``unique_sequences``."""
    fingerprint = _sequence_fingerprint(tokens)
    for candidate in fingerprint_buckets[fingerprint]:
        if unique_sequences[candidate] == tokens:
            return candidate
    seq_index = len(unique_sequences)
    unique_sequences.append(tokens)
    fingerprint_buckets[fingerprint].append(seq_index)
    return seq_index


def pieces_from_hash_cache(
    hash_ids: list[int], tokens: list[int], cache: object
) -> list[list[int]] | None:
    """Split ``tokens`` into cached hash blocks plus an optional unhashed tail."""
    if not isinstance(cache, dict):
        return None
    pieces: list[list[int]] = []
    offset = 0
    for hid in hash_ids:
        block = cache.get(hid)
        if not isinstance(block, list) or not block:
            return None
        end = offset + len(block)
        if tokens[offset:end] != block:
            return None
        pieces.append(block)
        offset = end
    if offset < len(tokens):
        pieces.append(tokens[offset:])
    elif offset != len(tokens):
        return None
    return pieces


def overlap_decode_recipe(
    pieces: list[list[int]],
    unique_sequences: list[list[int]],
    fingerprint_buckets: dict[bytes, list[int]],
) -> list[tuple[int, int | None]]:
    """Map pieces to decode jobs: first piece as-is, later pieces with 1-token context."""
    recipe: list[tuple[int, int | None]] = []
    left: int | None = None
    for piece in pieces:
        if not piece:
            continue
        if left is None:
            recipe.append(
                (
                    intern_token_sequence(piece, unique_sequences, fingerprint_buckets),
                    None,
                )
            )
        else:
            ctx = [left]
            recipe.append(
                (
                    intern_token_sequence(
                        ctx + piece, unique_sequences, fingerprint_buckets
                    ),
                    intern_token_sequence(ctx, unique_sequences, fingerprint_buckets),
                )
            )
        left = piece[-1]
    return recipe


def stitch_overlap_decoded(
    decoded: list[str], recipe: list[tuple[int, int | None]]
) -> str:
    """Join overlap-decoded pieces into ``decode(concat(pieces))`` text."""
    parts: list[str] = []
    for seq_index, ctx_index in recipe:
        text = decoded[seq_index]
        if ctx_index is None:
            parts.append(text)
            continue
        prefix = decoded[ctx_index]
        if text.startswith(prefix):
            parts.append(text[len(prefix) :])
        else:
            # Test doubles for parallel_decode often return unrelated strings.
            # Real HuggingFace decode(ctx+piece) is prefix-stable vs decode(ctx).
            parts.append(text)
    return "".join(parts)


class HashIdsPromptSynthesisMixin:
    """Provide :meth:`synthesize_prompts_from_hash_ids` to any loader.

    Requires the host class to set, before calling:
      - ``self.prompt_generator`` (``PromptGenerator``).
      - ``self._tokenizer_name`` (resolved tokenizer alias for worker caches).
      - ``self._trust_remote_code`` (bool).
      - ``self._tokenizer_revision`` (str | None).
      - ``self._block_size`` (int).
    """

    @property
    def bpe_stable_terminator_tokens(self) -> list[int]:
        """The terminator chosen by the underlying ``PromptGenerator``. Empty
        list if no stable terminator was found (segment synthesis falls back
        to no terminator and segment-join drift is unfixed)."""
        return self.prompt_generator._bpe_stable_terminator_tokens

    def synthesize_prompts_from_hash_ids(
        self, requests: list[HashIdsPromptRequest]
    ) -> dict[str, str]:
        pending: list[tuple[str, list[tuple[int, int | None]]]] = []
        result: dict[str, str] = {}
        unique_sequences: list[list[int]] = []
        fingerprint_buckets: dict[bytes, list[int]] = defaultdict(list)

        pg = self.prompt_generator

        for req in requests:
            if not req.hash_ids:
                result[req.key] = pg.generate(
                    mean=req.input_length, stddev=0, hash_ids=[]
                )
                continue
            tokens = pg._build_token_sequence(
                req.input_length, req.hash_ids, self._block_size
            )
            pieces = pieces_from_hash_cache(req.hash_ids, tokens, pg._cache)
            if pieces is None:
                pieces = [tokens]
            pending.append(
                (
                    req.key,
                    overlap_decode_recipe(
                        pieces, unique_sequences, fingerprint_buckets
                    ),
                )
            )

        if unique_sequences:
            decoded_list = parallel_decode(
                unique_sequences,
                self._tokenizer_name,
                trust_remote_code=self._trust_remote_code,
                revision=self._tokenizer_revision or "main",
                max_workers=_unique_sequence_decode_workers(),
            )
            # zip(..., strict=True) keeps the length-mismatch contract tests expect.
            decoded = list(zip(range(len(unique_sequences)), decoded_list, strict=True))
            decoded_texts = [text for _, text in decoded]
            for key, recipe in pending:
                result[key] = stitch_overlap_decoded(decoded_texts, recipe)

        return result

    def sample_partial_tail_tokens(self, n_tokens: int, seed: str) -> list[int]:
        """Deterministic per-seed partial-block tokens sized to ``n_tokens``.

        Returns raw Qwen token IDs (no tokenizer.decode). Mirrors
        :meth:`sample_partial_tail` but skips the decode step so callers that
        need byte-exact token-level slicing don't pay the BPE roundtrip cost.
        See spec §4.6 determinism contract.
        """
        if n_tokens <= 0:
            return []
        pg = self.prompt_generator
        corpus_size = pg._corpus_size
        digest = hashlib.sha256(seed.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % max(corpus_size - n_tokens, 1)
        return list(pg._tokenized_corpus[offset : offset + n_tokens])

    def sample_partial_tail(self, n_tokens: int, seed: str) -> str:
        """Deterministic per-seed partial-block content sized to ``n_tokens`` tokens.

        Uses sha256-keyed RNG over the corpus offset, so two runs in different
        processes (different PYTHONHASHSEED) produce identical bytes for the
        same seed.
        """
        if n_tokens <= 0:
            return ""
        tokens = self.sample_partial_tail_tokens(n_tokens, seed)
        return self.prompt_generator.tokenizer.decode(tokens)
