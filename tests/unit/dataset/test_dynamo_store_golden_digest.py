# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden store-digest pin: the uniform-drift detector the parity suites cannot be, since a change that shifts BOTH sides of a parity comparison equally still passes them."""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.segment_trie.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
)

# The four finalized unified-store files, in the fixed order finalize() writes
# them. blake2b(digest_size=16) is a 128-bit content fingerprint: collision-free
# for a drift detector, compact enough to read in a diff.
_STORE_FILES = ("content.blob", "content.idx", "nodes.blob", "nodes.idx")

# Committed golden digests, pinned at this tree. Regenerating them is DELIBERATE:
# run this test, copy the reported ``got`` values here, and state in the commit
# WHY the store bytes moved (a corpus/trie/envelope change). An UNEXPLAINED drift
# is a frozen-behavior break -- investigate, do not re-pin blindly.
#
# nodes.blob / nodes.idx re-pinned for the dynamo multi-graph restructure: this
# fixture's ``s1`` + ``s2`` sessions are two independent session-trees, so the
# adapter now emits TWO per-tree ``TraceRecord``s (``s1`` with nodes s1_a1/s1_a2
# at ordinals 0/1; ``s2`` with node s2_a1 at ordinal 0) instead of ONE union
# trace ``s1`` carrying all three nodes at ordinals 0/1/2. The per-node manifest
# region is thus re-keyed by (trace_id, ordinal) -- a LAYOUT-only shift. The
# content pool is byte-UNCHANGED (content.blob / content.idx below are identical
# to the pre-restructure pins), proving no node/segment CONTENT changed; the
# determinism guard still holds (two independent builds agree).
#
# nodes.blob / nodes.idx re-pinned again 2026-07-07 for the recorded-output
# pinning unification: every dynamo node envelope now carries
# ``dispatch_overrides["max_output_tokens"]`` (recorded output_tokens, weka
# parity; recorded 0 upgrades to 1). Content pool digests unchanged once more.
#
# nodes.blob / nodes.idx re-pinned once more 2026-07-07: model / stream / the
# cap all moved to NATIVE LlmNode fields (Turn naming); the redundant
# ``dispatch_overrides["stream"]`` entry is gone (the envelope's top-level
# ``stream`` was always authoritative) and the envelope folds the native
# model + cap back in, so overrides became {model, max_output_tokens} -- same
# effective wire values, envelope bytes only.
#
# content.blob / content.idx re-pinned 2026-07-07 for the data-inherent node-id
# scheme ({session_id}:{k}): response/tiny synthesis seeds are node-id-derived
# and now trace-scoped, so those pool entries' bytes moved. Hash-block prompt
# content is unchanged (bare-hash-id namespace untouched); nodes.* digests are
# unchanged because manifests key by ordinal, not node id.
GOLDEN_DIGESTS: dict[str, str] = {
    "content.blob": "24bf57403c7920ec1e769f21f365612c",
    "content.idx": "6533bb19bdf9c1ec99e78ca704d564f3",
    "nodes.blob": "da6297b1a2346fbed7922efb534db79b",
    # Re-pinned when the manifest inner key dropped its phase-variant suffix
    # ("<ordinal>:<variant>" -> "<ordinal>"). Keys only: content.blob,
    # content.idx and nodes.blob are all byte-identical across that change.
    "nodes.idx": "58a1ce8388a9a1b1a21b38fdf656ec5f",
}

_REPIN = (
    "an INTENTIONAL trie/corpus/envelope change re-pins this digest deliberately "
    "(regenerate the literal and explain the byte shift in the commit); an "
    "unexplained drift is a frozen-behavior break -- do not delete the check"
)


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    """A single ``dynamo.request.trace.v1`` record."""
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": {
            "request_id": f"r{ts}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": 8,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_tokens,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _write_dynamo_fixture(path: Path) -> Path:
    """Write the 3-record dynamo trace fixture, literals COPIED (not imported) from ``test_dynamo_streaming_store_parity`` so the golden pin stays self-contained against fixture edits elsewhere."""
    records = [
        _dynamo_record(1000, "s1", 32, [111, 222]),
        _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
        _dynamo_record(3000, "s2", 48, [555, 666, 777]),
    ]
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return path


async def _build_store_digests(
    fixture: Path, tmp_path: Path, bid: str
) -> dict[str, str]:
    """Parse the fixture fresh, drain it interned into a store keyed by ``bid``, and return ``{filename: blake2b-hexdigest}`` for the four finalized files -- each call re-parses from disk, so two calls are fully independent builds and can serve as the determinism guard."""
    parsed = from_dynamo_trace(
        fixture, content_root_seed=1234, content_tokenizer="builtin"
    )
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    await build_unified_trie_store_interned(parsed, store)
    store_dir = tmp_path / f"aiperf_graph_segments_{bid}"
    return {
        name: hashlib.blake2b(
            (store_dir / name).read_bytes(), digest_size=16
        ).hexdigest()
        for name in _STORE_FILES
    }


@pytest.mark.asyncio
async def test_golden_store_digest_pin(tmp_path: Path) -> None:
    """The finalized unified-store files digest to their committed golden values, and two independent builds agree -- a pin over a nondeterministic build would gate nothing."""
    fixture = _write_dynamo_fixture(tmp_path / "dyn_golden.jsonl")

    first = await _build_store_digests(fixture, tmp_path, "golden-a")
    second = await _build_store_digests(fixture, tmp_path, "golden-b")

    assert first == second, (
        "nondeterministic store build: two independent in-test builds produced "
        f"different digests ({first} vs {second}). A golden pin over a "
        "nondeterministic build cannot gate anything downstream -- STOP and fix "
        "the ordering nondeterminism before pinning."
    )

    for name in _STORE_FILES:
        assert first[name] == GOLDEN_DIGESTS[name], (
            f"golden store-digest drift for {name!r}: got {first[name]!r}, "
            f"pinned {GOLDEN_DIGESTS[name]!r}. {_REPIN}"
        )
