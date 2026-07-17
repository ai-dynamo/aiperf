# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verbatim raw-wire-JSON segments through the pool and the unified store.

Authored dag messages are arbitrary dicts the legacy path forwards verbatim
(key order and extra keys included). These tests pin the raw-segment variant:
``SegmentPool.add_raw_message`` interns ``orjson.dumps(message)`` verbatim and
the unified store persists that blob byte-for-byte instead of re-serializing a
normalized ``{"role", "content"}`` dict.
"""

import asyncio

import orjson
import pytest

from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


@pytest.fixture
def pool() -> SegmentPool:
    return SegmentPool()


def test_add_raw_message_preserves_key_order_and_extra_keys(pool):
    msg = {"content": "hi", "role": "user", "name": "alice"}
    sid = pool.add_raw_message(message=msg, parent_id=None)
    seg = pool.get(sid)
    assert seg.wire_json == orjson.dumps(msg).decode()
    assert orjson.loads(seg.wire_json) == msg
    # key order survives round-trip
    assert list(orjson.loads(seg.wire_json)) == ["content", "role", "name"]


def test_add_raw_message_dedups_by_content_and_prefix(pool):
    m = {"role": "user", "content": "x"}
    a = pool.add_raw_message(message=m, parent_id=None)
    b = pool.add_raw_message(message=m, parent_id=None)
    c = pool.add_raw_message(message=m, parent_id=a)
    assert a == b
    assert a != c  # prefix-dependent id


def test_raw_and_text_ids_never_alias(pool):
    m = {"role": "user", "content": "x"}
    raw = pool.add_raw_message(message=m, parent_id=None)
    text = pool.add_text(role="user", content="x", parent_id=None)
    assert raw != text


def test_materialize_returns_verbatim_dict_for_raw(pool):
    m = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    sid = pool.add_raw_message(message=m, parent_id=None)
    assert pool.materialize([sid]) == [m]


def test_store_round_trips_raw_and_text_segments(tmp_path):
    """A raw segment persists verbatim; a role/content segment normalizes as before."""
    msg = {"content": "hi", "role": "user", "name": "alice"}
    store = GraphSegmentUnifiedBackingStore(tmp_path, "raw")
    raw_h = store.put_segment(
        "raw_seg", "user", "", wire_json=orjson.dumps(msg).decode()
    )
    text_h = store.put_segment("text_seg", "system", "SYS")
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "raw").open() as c:
        # materialize returns the verbatim dict for the raw segment
        assert c.materialize_handles([raw_h]) == [msg]
        assert c.materialize_handles([text_h]) == [{"role": "system", "content": "SYS"}]
        # the request body embeds the exact orjson.dumps(msg) bytes verbatim
        body = c.build_request_body_handles([raw_h], b"")
        assert body == b'{"messages":[' + orjson.dumps(msg) + b"]}"
