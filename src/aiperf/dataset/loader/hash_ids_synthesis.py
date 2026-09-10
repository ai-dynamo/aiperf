# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared hash_ids -> decoded prompt synthesis used by Weka and trace loaders.

The pipeline:

1. Build a token sequence for each requested (hash_ids, input_length) pair
   (fills ``PromptGenerator._cache`` with unique hash-block token IDs).
2. Parallel-decode each unique **complete** token sequence once.
3. Map decoded strings back onto caller keys.

Block token IDs are reused across requests; ``tokenizer.decode`` always runs on
the assembled sequence. Joining per-block decode strings would reintroduce
segment-join BPE drift (see ``PromptGenerator._determine_bpe_stable_terminator``).
Identical assembled sequences share one decode call.
"""

from __future__ import annotations

import hashlib
import os
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
        pending: list[tuple[str, tuple[int, ...]]] = []
        result: dict[str, str] = {}
        unique_sequences: dict[tuple[int, ...], list[int]] = {}

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
            seq_key = tuple(tokens)
            unique_sequences.setdefault(seq_key, tokens)
            pending.append((req.key, seq_key))

        if unique_sequences:
            seq_order = list(unique_sequences)
            decoded_list = parallel_decode(
                [unique_sequences[k] for k in seq_order],
                self._tokenizer_name,
                trust_remote_code=self._trust_remote_code,
                revision=self._tokenizer_revision or "main",
                max_workers=_unique_sequence_decode_workers(),
            )
            decoded = dict(zip(seq_order, decoded_list, strict=True))
            for key, seq_key in pending:
                result[key] = decoded[seq_key]

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
