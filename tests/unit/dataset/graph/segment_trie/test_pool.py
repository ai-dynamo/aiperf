# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw wire-JSON segments stay byte-verbatim through the pool and the unified store."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)

# Authored dag messages are arbitrary dicts that must be forwarded to the
# endpoint untouched, so add_raw_message interns the orjson.dumps bytes verbatim
# instead of round-tripping through a normalized role/content pair.


@pytest.fixture
def pool() -> SegmentPool:
    return SegmentPool()


def test_add_raw_message_preserves_key_order_and_extra_keys(pool: SegmentPool) -> None:
    """A raw message keeps its non-canonical key order and its extra keys."""
    msg = {"content": "hi", "role": "user", "name": "alice"}
    seg = pool.get(pool.add_raw_message(message=msg, parent_id=None))
    assert seg.wire_json == orjson.dumps(msg).decode()
    assert list(orjson.loads(seg.wire_json)) == ["content", "role", "name"]


def test_add_raw_message_dedups_by_content_and_prefix(pool: SegmentPool) -> None:
    """Identical raw messages intern to one id, but a different parent yields a distinct prefix-dependent id."""
    m = {"role": "user", "content": "x"}
    a = pool.add_raw_message(message=m, parent_id=None)
    b = pool.add_raw_message(message=m, parent_id=None)
    c = pool.add_raw_message(message=m, parent_id=a)
    assert a == b
    assert a != c


def test_raw_and_text_ids_never_alias(pool: SegmentPool) -> None:
    """A raw message and an equivalent role/content segment get different ids, so verbatim bytes are never swapped for normalized ones."""
    m = {"role": "user", "content": "x"}
    raw = pool.add_raw_message(message=m, parent_id=None)
    text = pool.add_text(role="user", content="x", parent_id=None)
    assert raw != text


def test_materialize_returns_verbatim_dict_for_raw(pool: SegmentPool) -> None:
    """Materializing a raw segment returns the original dict, including structured content parts."""
    m = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    sid = pool.add_raw_message(message=m, parent_id=None)
    assert pool.materialize([sid]) == [m]


def test_store_round_trips_raw_and_text_segments(tmp_path: Path) -> None:
    """Through the unified store a raw segment persists verbatim while a role/content segment normalizes, and the request body embeds the exact raw bytes."""
    msg = {"content": "hi", "role": "user", "name": "alice"}
    store = GraphSegmentUnifiedBackingStore(tmp_path, "raw")
    raw_h = store.put_segment(
        "raw_seg", "user", "", wire_json=orjson.dumps(msg).decode()
    )
    text_h = store.put_segment("text_seg", "system", "SYS")
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "raw").open() as c:
        assert c.materialize_handles([raw_h]) == [msg]
        assert c.materialize_handles([text_h]) == [{"role": "system", "content": "SYS"}]
        body = c.build_request_body_handles([raw_h], b"")
        assert body == b'{"messages":[' + orjson.dumps(msg) + b"]}"
