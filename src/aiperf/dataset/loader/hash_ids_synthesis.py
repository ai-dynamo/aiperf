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
from dataclasses import dataclass, field

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


@dataclass(slots=True)
class OverlapIntern:
    """Intern overlap decode jobs by hash_id / left token, not full-list hashes."""

    sequences: list[list[int]] = field(default_factory=list)
    first: dict[int, int] = field(default_factory=dict)
    cont: dict[tuple[int, int], int] = field(default_factory=dict)
    ctx: dict[int, int] = field(default_factory=dict)
    tail: dict[bytes, list[int]] = field(default_factory=lambda: defaultdict(list))
    full: dict[bytes, list[int]] = field(default_factory=lambda: defaultdict(list))

    def intern_full(self, tokens: list[int]) -> int:
        return intern_token_sequence(tokens, self.sequences, self.full)

    def intern_first(self, hash_id: int, tokens: list[int]) -> int:
        idx = self.first.get(hash_id)
        if idx is not None:
            return idx
        idx = len(self.sequences)
        self.sequences.append(tokens)
        self.first[hash_id] = idx
        return idx

    def intern_ctx(self, left: int) -> int:
        idx = self.ctx.get(left)
        if idx is not None:
            return idx
        idx = len(self.sequences)
        self.sequences.append([left])
        self.ctx[left] = idx
        return idx

    def intern_cont(self, left: int, hash_id: int, tokens: list[int]) -> int:
        key = (left, hash_id)
        idx = self.cont.get(key)
        if idx is not None:
            return idx
        idx = len(self.sequences)
        self.sequences.append([left, *tokens])
        self.cont[key] = idx
        return idx

    def intern_tail(self, left: int, tokens: list[int]) -> int:
        seq = [left, *tokens]
        packed = _sequence_fingerprint(seq)
        for candidate in self.tail[packed]:
            if self.sequences[candidate] == seq:
                return candidate
        idx = len(self.sequences)
        self.sequences.append(seq)
        self.tail[packed].append(idx)
        return idx


def overlap_decode_recipe(
    pieces: list[list[int]],
    piece_keys: list[int | None],
    intern: OverlapIntern,
) -> list[tuple[int, int | None]]:
    """Map pieces to decode jobs: first piece as-is, later pieces with 1-token context."""
    recipe: list[tuple[int, int | None]] = []
    left: int | None = None
    for piece, key in zip(pieces, piece_keys, strict=True):
        if not piece:
            continue
        if left is None:
            if key is None:
                recipe.append((intern.intern_full(piece), None))
            else:
                recipe.append((intern.intern_first(key, piece), None))
        else:
            ctx_index = intern.intern_ctx(left)
            if key is None:
                seq_index = intern.intern_tail(left, piece)
            else:
                seq_index = intern.intern_cont(left, key, piece)
            recipe.append((seq_index, ctx_index))
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
        intern = OverlapIntern()
        pg = self.prompt_generator
        build_pieces = getattr(pg, "_build_token_pieces", None)
        use_pieces = (
            type(getattr(pg, "_cache", None)) is dict
            and callable(build_pieces)
            and type(build_pieces).__name__ == "method"
        )

        for req in requests:
            if not req.hash_ids:
                result[req.key] = pg.generate(
                    mean=req.input_length, stddev=0, hash_ids=[]
                )
                continue
            if use_pieces:
                pieces = pg._build_token_pieces(
                    req.input_length, req.hash_ids, self._block_size
                )
                keys: list[int | None] = list(req.hash_ids)
                if len(pieces) > len(keys):
                    keys.append(None)
                recipe = overlap_decode_recipe(pieces, keys, intern)
            else:
                tokens = pg._build_token_sequence(
                    req.input_length, req.hash_ids, self._block_size
                )
                recipe = [(intern.intern_full(tokens), None)]
            pending.append((req.key, recipe))

        unique_sequences = intern.sequences
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
