# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic ``dynamo.request.trace.v1`` capture generator for corpus-scale build-plane memory measurement."""

# tests/unit/dataset/graph/adapters/test_dynamo_corpus_scale_memory.py needs a
# corpus-scale capture reproducing the two amplifiers real captures exhibit,
# without committing a multi-GB fixture:
#
# 1. Hash-id-slot amplification. Dynamo re-lists the FULL prompt block-hash list
#    on every request_end, so T turns adding G fresh blocks each list
#    H = G*T(T+1)/2 slots while introducing only U = T*G unique blocks. After
#    JSON decode every slot starts as a distinct u64 int object (above the
#    small-int cache); without interning the persistent TrieRequest.hash_ids
#    lists would be the largest build tier at corpus scale (~24.8 GB @1M nodes).
#    _collect_records interns them read-time, and this generator's H-vs-U gap is
#    exactly what exercises that intern table.
# 2. Unique-block growth. The T*G fresh blocks per session are globally distinct
#    (seeded 64-bit ids never recur), so resolve_content_parents' resolution
#    automaton (one int state per unique position) and the dynamo decode cache
#    (one entry per unique hash) both grow with the unique-block count.
#
# Output is byte-deterministic given seed, and every record is block-consistent
# per trie_lowering._assert_block_aligned ((n-1)*bs < input_length <= n*bs):
# input_length is n_hashes * block_size so every recorded block is fully covered
# and the covered-count ISL gate reconstructs exactly input_length tokens.
# Records parse through from_dynamo_trace unchanged.

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import orjson

# Epoch-ish base so timestamps look like real Unix-ms captures; the absolute
# value is irrelevant (the trie lowering works trace-relative), only per-session
# monotonicity matters.
_BASE_UNIX_MS = 1_700_000_000_000
# Per-turn start stride (a turn begins this many ms after the previous turn of
# its session). Sessions are offset by a 1 ms per-session jitter so no two
# records share an ``event_time`` (the interval-order rank tie-breaks on
# ``node_id`` regardless, but distinct times keep the sort stable).
_TURN_STRIDE_MS = 1_000
# Every request's recorded interval is made to SPAN the whole capture window
# (:func:`_capture_duration_ms`), so all requests overlap pairwise and NOTHING
# finishes before anything else starts. The interval-order edge pass
# (``build_interval_edges``, O(candidates^2) per node) then sees an empty
# finished-before frontier at every node -- each node roots at START or
# start-anchors to its in-flight causal parent -- keeping edge derivation off
# the memory measurement's critical path. Edge STRUCTURE is irrelevant to the
# four memory tiers (hash-id ints, resolution trie, decode cache, content pool),
# which is all this capture is built to size.
_OVERLAP_MULTIPLIER = 4
# Small fixed recorded completion; the response segment synthesizes this many
# tokens per node (a cheap partial-tail slice), so it is kept low to keep the
# generator's amplification concentrated in the prompt hash lists.
_OUTPUT_TOKENS = 8


def _capture_duration_ms(sessions: int, turns_per_session: int) -> int:
    """Recorded per-request duration that makes every interval overlap."""
    # Larger than the whole capture's start span, so the earliest request only
    # ends well after the latest one starts -- the full-overlap condition that
    # keeps the interval-order pass cheap.
    span = turns_per_session * _TURN_STRIDE_MS + sessions
    return span * _OVERLAP_MULTIPLIER + 1_000


@dataclass(frozen=True, slots=True)
class SyntheticCorpusShape:
    """Derived element counts for one synthetic capture's parameters."""

    # All four are pure functions of the generator parameters, so the memory
    # measurement's analytic per-tier model is computed from THESE, never from
    # hardcoded byte totals.

    n_nodes: int
    """One trie node per ``request_end`` (``sessions * turns``)."""

    n_hash_slots: int
    """Total ``TrieRequest.hash_ids`` entries; the hash-id-int tier scales with this."""

    n_unique_blocks: int
    """Globally distinct block hashes; the resolution-trie and decode-cache tiers scale with this."""

    n_unique_segments: int
    """Globally distinct content-addressed segments; the content pool and sid-string tiers scale with this."""


def synthetic_corpus_shape(
    *, sessions: int, turns_per_session: int, new_blocks_per_turn: int
) -> SyntheticCorpusShape:
    """Element counts a capture with these parameters will produce."""
    t = turns_per_session
    return SyntheticCorpusShape(
        n_nodes=sessions * t,
        n_hash_slots=sessions * new_blocks_per_turn * (t * (t + 1) // 2),
        n_unique_blocks=sessions * t * new_blocks_per_turn,
        n_unique_segments=sessions * (2 * t - 1) + sessions * t,
    )


def _fresh_hash(rng: random.Random) -> int:
    """One positive 64-bit block hash (u64; never 0, never negative)."""
    # Dynamo records input_sequence_hashes as Rust u64 and the reader rejects
    # negatives (they collide with the virtual negative-id namespace).
    return rng.getrandbits(64) or 1


def write_synthetic_dynamo_capture(
    path: str | Path,
    *,
    sessions: int,
    turns_per_session: int,
    new_blocks_per_turn: int,
    block_size: int = 16,
    seed: int,
) -> Path:
    """Write a deterministic chain-heavy ``dynamo.request.trace.v1`` JSONL file and return its path."""
    # Emits sessions * turns_per_session request_end records. Within a session,
    # turn k carries the full previous prefix plus new_blocks_per_turn fresh
    # 64-bit hashes, so every earlier block is re-listed (the slot amplification)
    # while fresh hashes stay globally unique. input_length is
    # len(hashes) * block_size, satisfying the block-alignment gate and making
    # the covered-count ISL target equal input_length. Timestamps are monotone
    # within each session. Records stream out one line at a time so a
    # corpus-scale capture never materializes in memory.
    out = Path(path)
    rng = random.Random(seed)
    duration_ms = _capture_duration_ms(sessions, turns_per_session)
    with out.open("wb") as f:
        first = True
        for s in range(sessions):
            session_id = f"synth-{seed}-{s:08d}"
            prefix: list[int] = []
            for k in range(1, turns_per_session + 1):
                for _ in range(new_blocks_per_turn):
                    prefix.append(_fresh_hash(rng))
                n = len(prefix)
                input_length = n * block_size
                # +s jitter keeps event_times distinct across sessions; the
                # (k-1) stride keeps a session's turns strictly increasing.
                received_ms = _BASE_UNIX_MS + (k - 1) * _TURN_STRIDE_MS + s
                record = {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": received_ms + duration_ms,
                    "event_source": "dynamo",
                    "agent_context": {"session_id": session_id},
                    "request": {
                        "request_id": f"{session_id}-r{k}",
                        "model": "synthetic-model",
                        "input_tokens": input_length,
                        "output_tokens": _OUTPUT_TOKENS,
                        "cached_tokens": 0,
                        "request_received_ms": received_ms,
                        "total_time_ms": duration_ms,
                        "replay": {
                            "trace_block_size": block_size,
                            "input_length": input_length,
                            # copy() so each record owns its own list snapshot at
                            # this turn's length (the prefix keeps growing).
                            "input_sequence_hashes": prefix.copy(),
                        },
                    },
                }
                if not first:
                    f.write(b"\n")
                f.write(orjson.dumps(record))
                first = False
    return out


__all__ = [
    "SyntheticCorpusShape",
    "synthetic_corpus_shape",
    "write_synthetic_dynamo_capture",
]
