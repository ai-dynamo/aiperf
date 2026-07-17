# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic ``dynamo.request.trace.v1`` capture generator.

The build-plane memory measurement (``tests/unit/dataset/graph/adapters/
test_dynamo_corpus_scale_memory.py``) needs a corpus-scale dynamo capture whose
shape reproduces the two amplifiers real captures exhibit, WITHOUT committing a
multi-GB fixture:

1. **Hash-id-slot amplification.** Dynamo records the FULL prompt block-hash
   list on every ``request_end`` -- turn ``k`` re-lists all of turn ``k-1``'s
   blocks plus its own fresh ones. A chain of ``T`` turns adding ``G`` fresh
   blocks per turn therefore lists ``G * T(T+1)/2`` hash-id slots even though it
   only introduces ``T * G`` unique blocks. After JSON decode every slot starts
   as a distinct Python ``int`` object (u64 values, above the small-int cache),
   which without interning would make the persistent ``TrieRequest.hash_ids``
   lists the single
   largest build tier at corpus scale (~24.8 GB @1M nodes).
   ``_collect_records`` interns those recorded ints read-time, so
   the resident slot cost is one list pointer plus an amortized single canonical
   ``int`` per UNIQUE value; this generator's ``H``-vs-``U`` amplification
   (``H = G*T(T+1)/2`` slots over ``U = T*G`` unique blocks) is exactly what
   exercises that intern table.
2. **Unique-block growth.** The ``T * G`` fresh blocks per session are globally
   distinct (seeded 64-bit ids never recur across turns or sessions), so the
   resolution automaton built by
   :func:`~aiperf.dataset.graph.segment_ir.trie_content.resolve_content_parents`
   (one int state per unique position) and the dynamo decode cache (one entry
   per unique hash) both grow with the unique-block count.

The generated capture is byte-deterministic given ``seed`` and every record is
block-consistent per
:func:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering._assert_block_aligned`
(``(n-1)*bs < input_length <= n*bs``): ``input_length`` is set to
``n_hashes * block_size`` so every recorded block is fully covered and the
covered-count ISL gate reconstructs exactly ``input_length`` tokens. Records
parse through :func:`~aiperf.dataset.graph.adapters.dynamo.trace.from_dynamo_trace`
unchanged.
"""

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
    """Recorded per-request duration that makes every interval overlap.

    Larger than the whole capture's start span (``turns * stride + sessions``),
    so the earliest request only ends well after the latest request starts --
    the full-overlap condition the interval-order pass needs to stay cheap.
    """
    span = turns_per_session * _TURN_STRIDE_MS + sessions
    return span * _OVERLAP_MULTIPLIER + 1_000


@dataclass(frozen=True, slots=True)
class SyntheticCorpusShape:
    """Derived element counts for one synthetic capture's parameters.

    All three are pure functions of the generator parameters, so the memory
    measurement's analytic per-tier model is computed from THESE (never from
    hardcoded byte totals):

    * ``n_nodes`` -- one trie node per ``request_end`` (``sessions * turns``).
    * ``n_hash_slots`` -- total ``TrieRequest.hash_ids`` entries across all
      nodes = ``sessions * new_blocks_per_turn * T(T+1)/2``; the hash-id-int
      tier scales with this.
    * ``n_unique_blocks`` -- globally distinct block hashes =
      ``sessions * turns * new_blocks_per_turn``; the resolution-trie and
      decode-cache tiers scale with this.
    * ``n_unique_segments`` -- globally distinct content-addressed message/response
      segments = ``sessions * (2*turns - 1) + n_nodes`` (one prompt + one response
      segment per turn, minus the shared per-session prompt root, plus the response
      leaves); the content pool (``Segment`` objects) and the canonical
      sid-string addressing tier scale with this.
    """

    n_nodes: int
    n_hash_slots: int
    n_unique_blocks: int
    n_unique_segments: int


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
    """One positive 64-bit block hash (u64; never 0, never negative).

    Dynamo records ``input_sequence_hashes`` as Rust ``u64`` and the reader
    rejects negatives (they collide with the virtual negative-id namespace), so
    the generator draws from ``[1, 2**64)``.
    """
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
    """Write a deterministic chain-heavy ``dynamo.request.trace.v1`` JSONL file.

    Emits ``sessions * turns_per_session`` ``request_end`` records shaped like
    ``tests/unit/dataset/test_dynamo_streaming_store_parity.py``'s
    ``_dynamo_record``. Within a session, turn ``k`` (1-indexed) carries the
    full previous prefix plus ``new_blocks_per_turn`` fresh 64-bit hashes, so
    ``input_sequence_hashes`` grows by ``new_blocks_per_turn`` each turn and
    every earlier block is re-listed -- the hash-id-slot amplification. Fresh
    hashes are globally unique (one seeded RNG stream, no recurrence), so the
    unique-block count is exactly ``sessions * turns_per_session *
    new_blocks_per_turn``.

    ``input_length`` is ``len(hashes) * block_size`` (fully block-covered), which
    satisfies the block-alignment gate and makes the covered-count ISL target
    equal to ``input_length``. Timestamps are monotone within each session.

    Records are streamed to disk one JSON line at a time so a corpus-scale
    capture (millions of records) never materializes in memory.

    Returns the written ``path``.
    """
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
