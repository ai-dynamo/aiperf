# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""segment_pool survives the directory merge and the cross-process msgpack codec.

Regression guards for the directory/parallel weka trie-IR data-loss bug: the
merge dropped ``ParsedGraph.segment_pool`` (no ``segment_pool=`` arg) and the
codec decoded the (then ``Any``-typed) field to a bare ``dict`` instead of a
``SegmentPool``. Either drop knocks the trie graph off the trie ordinal scheme
and yields zero traces.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.codecs import (
    decode_parsed_graph_msgpack,
    encode_parsed_graph_msgpack,
)
from aiperf.dataset.graph.merge import (
    GraphMergeError,
    merge_parsed_graphs,
)
from aiperf.dataset.graph.models import ParsedGraph, TraceRecord
from aiperf.dataset.graph.segment_ir.pool import Segment, SegmentPool


def _pool_with(entries: list[tuple[str, str, list[int], str | None]]) -> SegmentPool:
    pool = SegmentPool()
    for role, content, tokens, parent_id in entries:
        pool.add(role=role, content=content, tokens=tokens, parent_id=parent_id)
    return pool


def test_merge_unions_per_file_segment_pools() -> None:
    # A shared root segment (same role/content/tokens/parent -> same content id)
    # plus a file-distinct leaf in each file.
    shared = ("system", "sys", [1, 2], None)
    pool_a = _pool_with([shared, ("user", "alpha", [3], "ignored")])
    pool_b = _pool_with([shared, ("user", "beta", [4], "ignored")])

    pg_a = ParsedGraph(
        traces=[TraceRecord(id="t-a")],
        segment_pool=pool_a,
    )
    pg_b = ParsedGraph(
        traces=[TraceRecord(id="t-b")],
        segment_pool=pool_b,
    )

    merged = merge_parsed_graphs([pg_a, pg_b])

    assert isinstance(merged.segment_pool, SegmentPool)
    expected_ids = set(pool_a._by_id) | set(pool_b._by_id)
    assert set(merged.segment_pool._by_id) == expected_ids
    # Shared id present exactly once; union is strictly larger than either input.
    assert len(merged.segment_pool._by_id) == len(expected_ids)
    assert len(merged.segment_pool._by_id) > len(pool_a._by_id)

    # Every file's path still materializes against the merged pool.
    for pool in (pool_a, pool_b):
        for sid, seg in pool._by_id.items():
            assert merged.segment_pool.materialize([sid]) == [
                {"role": seg.role, "content": seg.content}
            ]


def test_merge_divergent_content_same_id_raises() -> None:
    # Ids are content-addressed (blake2b over parent_id/role/tokens), so the SAME
    # id across pools MUST carry identical content. Two pools mapping one id to
    # divergent segments can only mean a content-addressing / hash break, so the
    # merge must fail loud rather than silently keep whichever entry wins.
    seg_a = Segment(id="collide", role="user", content="alpha", parent_id=None)
    seg_b = Segment(id="collide", role="user", content="beta", parent_id=None)
    pool_a = SegmentPool(_by_id={"collide": seg_a})
    pool_b = SegmentPool(_by_id={"collide": seg_b})

    pg_a = ParsedGraph(traces=[TraceRecord(id="t-a")], segment_pool=pool_a)
    pg_b = ParsedGraph(traces=[TraceRecord(id="t-b")], segment_pool=pool_b)

    with pytest.raises(GraphMergeError):
        merge_parsed_graphs([pg_a, pg_b])


def test_merge_identical_content_same_id_dedups() -> None:
    # The normal case: two pools share an id whose content is identical (Segment
    # is a frozen dataclass, so == is a value comparison). The union dedups to a
    # single entry and never raises.
    expected = Segment(id="shared", role="user", content="alpha", parent_id=None)
    pool_a = SegmentPool(
        _by_id={
            "shared": Segment(id="shared", role="user", content="alpha", parent_id=None)
        }
    )
    pool_b = SegmentPool(
        _by_id={
            "shared": Segment(id="shared", role="user", content="alpha", parent_id=None)
        }
    )

    pg_a = ParsedGraph(traces=[TraceRecord(id="t-a")], segment_pool=pool_a)
    pg_b = ParsedGraph(traces=[TraceRecord(id="t-b")], segment_pool=pool_b)

    merged = merge_parsed_graphs([pg_a, pg_b])

    assert isinstance(merged.segment_pool, SegmentPool)
    assert set(merged.segment_pool._by_id) == {"shared"}
    assert merged.segment_pool._by_id["shared"] == expected


def test_merge_without_pool_leaves_segment_pool_none() -> None:
    pg = ParsedGraph(traces=[TraceRecord(id="t-a")])
    merged = merge_parsed_graphs([pg])
    assert merged.segment_pool is None


def test_codec_round_trips_real_segment_pool() -> None:
    pool = _pool_with(
        [
            ("system", "sys", [1, 2], None),
            ("user", "hello", [3, 4, 5], "x"),
        ]
    )
    pg = ParsedGraph(segment_pool=pool)

    decoded = decode_parsed_graph_msgpack(encode_parsed_graph_msgpack(pg))

    assert type(decoded.segment_pool).__name__ == "SegmentPool"
    assert set(decoded.segment_pool._by_id) == set(pool._by_id)
    for sid, seg in pool._by_id.items():
        assert decoded.segment_pool.materialize([sid]) == [
            {"role": seg.role, "content": seg.content}
        ]


def test_codec_round_trips_none_pool_unchanged() -> None:
    pg = ParsedGraph()
    decoded = decode_parsed_graph_msgpack(encode_parsed_graph_msgpack(pg))
    assert decoded.segment_pool is None
