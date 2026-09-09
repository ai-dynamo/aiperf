# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared hash_ids -> decoded prompt synthesis used by Weka and trace loaders.

The pipeline:

1. Build a token sequence for each requested (hash_ids, input_length) pair
   (fills ``PromptGenerator._cache`` with unique hash blocks).
2. Parallel-decode each unique hash block (and any unhashed tail) once.
3. Concatenate the decoded block strings per request.

Mooncake/Bailian traces reuse the same block hashes across tens of thousands
of requests. Decoding every full prompt repeats tokenizer work proportional to
``requests * ISL``. Decoding unique blocks is proportional to
``unique_blocks * block_size``.
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


def _unique_block_decode_workers() -> int:
    """Cap decode workers high enough for large unique-block batches."""
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
        pending: list[tuple[str, list[tuple[str, object]]]] = []
        result: dict[str, str] = {}
        piece_tokens: dict[tuple[str, object], list[int]] = {}

        pg = self.prompt_generator
        cache = getattr(pg, "_cache", None)
        cache_is_map = isinstance(cache, dict)

        for req in requests:
            if not req.hash_ids:
                result[req.key] = pg.generate(
                    mean=req.input_length, stddev=0, hash_ids=[]
                )
                continue
            tokens = pg._build_token_sequence(
                req.input_length, req.hash_ids, self._block_size
            )
            piece_keys = _piece_keys_for_request(
                hash_ids=req.hash_ids,
                tokens=tokens,
                cache=cache if cache_is_map else None,
                piece_tokens=piece_tokens,
            )
            pending.append((req.key, piece_keys))

        if piece_tokens:
            piece_order = list(piece_tokens)
            decoded_list = parallel_decode(
                [piece_tokens[k] for k in piece_order],
                self._tokenizer_name,
                trust_remote_code=self._trust_remote_code,
                revision=self._tokenizer_revision or "main",
                max_workers=_unique_block_decode_workers(),
            )
            decoded = dict(zip(piece_order, decoded_list, strict=True))
            for key, piece_keys in pending:
                result[key] = "".join(decoded[k] for k in piece_keys)

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


def _piece_keys_for_request(
    *,
    hash_ids: list[int],
    tokens: list[int],
    cache: dict[int, list[int]] | None,
    piece_tokens: dict[tuple[str, object], list[int]],
) -> list[tuple[str, object]]:
    """Map one request onto unique decode pieces (hash blocks + optional tail).

    Falls back to decoding the full token sequence when ``_cache`` is missing
    entries (tests that stub ``_build_token_sequence`` without filling the
    cache).
    """
    if cache is not None and all(hid in cache for hid in hash_ids):
        hashed_len = 0
        keys: list[tuple] = []
        for hid in hash_ids:
            key = ("h", hid)
            if key not in piece_tokens:
                piece_tokens[key] = list(cache[hid])
            keys.append(key)
            hashed_len += len(cache[hid])
        tail = tokens[hashed_len:]
        if tail:
            tkey = ("t", tuple(tail))
            piece_tokens.setdefault(tkey, list(tail))
            keys.append(tkey)
        return keys

    skey = ("s", tuple(tokens))
    piece_tokens.setdefault(skey, list(tokens))
    return [skey]
