# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified segment store's write, finalize, abort, and client-read faces."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.exceptions import MemoryMapSerializationError
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)

_TWO_SEGMENT_MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "hi"},
]
_TWO_SEGMENT_BODY = (
    b'{"messages":[{"role":"system","content":"SYS"},{"role":"user","content":"hi"}]}'
)


def _two_segment_store(
    tmp_path: Path, bid: str
) -> tuple[GraphSegmentUnifiedBackingStore, int, int]:
    """Build an unfinalized store holding the two canonical segments plus one interned node manifest, returning it with both handles."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, bid)
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned("t0", 0, [ha, hb], {}, True)
    return store, ha, hb


def _finalized_two_segment_store(
    tmp_path: Path, bid: str
) -> tuple[GraphSegmentUnifiedBackingStore, int, int]:
    """``_two_segment_store`` driven all the way through ``finalize()``."""
    store, ha, hb = _two_segment_store(tmp_path, bid)
    asyncio.run(store.finalize())
    return store, ha, hb


def test_unified_backing_store_writes_four_files(tmp_path: Path) -> None:
    """A finalized store persists exactly the four files the client addresses by base path and benchmark id."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b1")
    store.put_segment("a", "system", "SYS")
    store.put_segment("b", "user", "hi")
    env = orjson.dumps({"prompt_segment_ids": ["a", "b"], "stream": True})
    store.add_node_manifest("t0", 0, env)
    asyncio.run(store.finalize())

    d = tmp_path / "aiperf_graph_segments_b1"
    assert (d / "content.blob").exists()
    assert (d / "content.idx").exists()
    assert (d / "nodes.blob").exists()
    assert (d / "nodes.idx").exists()

    client = GraphSegmentUnifiedClient(tmp_path, "b1")
    assert client.data_dir == d


def test_unified_client_duck_types_both_stores(tmp_path: Path) -> None:
    """One client serves both faces: node-envelope addressing and segment-content materialization."""
    _store, ha, hb = _finalized_two_segment_store(tmp_path, "b2")

    with GraphSegmentUnifiedClient(tmp_path, "b2").open() as c:
        # envelope addressing face:
        raw = c.get_node_envelope("t0", 0)
        assert orjson.loads(raw) == {
            "handles": [ha, hb],
            "dispatch_overrides": {},
            "stream": True,
        }
        assert c.get_node_envelope("t0", 9) is None
        # Segment-content face:
        assert c.materialize_handles([ha, hb]) == _TWO_SEGMENT_MESSAGES
        assert c.build_request_body_handles([ha, hb], b"") == _TWO_SEGMENT_BODY


def test_endpoint_extra_applied_flag_round_trips_when_set(tmp_path: Path) -> None:
    """A manifest written with ``endpoint_extra_applied=True`` reads back with the flag; the key is written only when True."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "eea1")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned(
        "t0", 0, [ha], {}, True, endpoint_extra_applied=True
    )
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "eea1").open() as c:
        env = orjson.loads(c.get_node_envelope("t0", 0))
        assert env["endpoint_extra_applied"] is True


def test_endpoint_extra_applied_flag_omitted_leaves_envelope_bytes_unchanged(
    tmp_path: Path,
) -> None:
    """Default (unset) leaves the manifest envelope byte-identical to a store written without the parameter at all -- byte-neutrality when the flag is off is the whole contract."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "eea2")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned("t0", 0, [ha], {}, True)
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "eea2").open() as c:
        raw = c.get_node_envelope("t0", 0)
        assert b"endpoint_extra_applied" not in raw
        assert orjson.loads(raw) == {
            "handles": [ha],
            "dispatch_overrides": {},
            "stream": True,
        }


def test_put_segment_returns_stable_insertion_index_handles(tmp_path: Path) -> None:
    """Handles are insertion indices, stable across a dedup re-put and re-derivable via ``segment_handle`` (``None`` for an unknown id)."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "h1")
    assert store.put_segment("a", "system", "SYS") == 0
    assert store.put_segment("b", "user", "hi") == 1
    assert store.put_segment("a", "system", "SYS") == 0  # dedup -> same handle
    assert store.segment_handle("b") == 1
    assert store.segment_handle("missing") is None


def test_unified_client_empty_pool_and_unknown_handle(tmp_path: Path) -> None:
    """An empty pool opens without raising and yields an empty message body, while an out-of-range handle on a non-empty pool fails loud."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b3")
    asyncio.run(store.finalize())
    with GraphSegmentUnifiedClient(tmp_path, "b3").open() as c:
        assert c._content_mv is None  # empty pool leaves mapping unset, no raise
        assert c.build_request_body_handles([], b"") == b'{"messages":[]}'
    store2 = GraphSegmentUnifiedBackingStore(tmp_path, "b4")
    store2.put_segment("a", "system", "SYS")
    asyncio.run(store2.finalize())
    with (
        GraphSegmentUnifiedClient(tmp_path, "b4").open() as c,
        pytest.raises(MemoryMapSerializationError),
    ):
        c.materialize_handles([1])


def test_a1_json_content_idx_is_rejected_with_reparse_error(tmp_path: Path) -> None:
    """Pre-retirement A1 (non-interned JSON content.idx) stores fail loud."""
    # Hand-rolled A1-shaped store: the JSON content.idx the retired
    # non-interned builder used to write ({"ids": hex->handle, "spans": ...}).
    d = tmp_path / "aiperf_graph_segments_a1"
    d.mkdir()
    seg = orjson.dumps({"role": "user", "content": "hi"})
    (d / "content.blob").write_bytes(seg)
    (d / "content.idx").write_bytes(
        orjson.dumps({"ids": {"a": 0}, "spans": [[0, len(seg)]]})
    )
    (d / "nodes.blob").write_bytes(b"")
    (d / "nodes.idx").write_bytes(orjson.dumps({}))

    with pytest.raises(ValueError, match="re-parse"):
        GraphSegmentUnifiedClient(tmp_path, "a1").open()


def test_finalize_releases_write_side_buffers(tmp_path: Path) -> None:
    """``finalize()`` drops every write-side buffer at its END, while ``build_stats`` -- snapshotted at its ENTRY -- survives the release."""
    store, _ha, _hb = _two_segment_store(tmp_path, "rel1")

    # Before finalize: content is already spilled to disk (tracked by the running
    # counter, never accumulated in _content_buf); the index buffers hold state.
    assert store._content_bytes_written > 0
    assert store._content_buf == bytearray()  # spill: content never sits in RAM
    assert len(store._nodes_buf) > 0
    assert store._ids and store._spans and store._node_offsets

    asyncio.run(store.finalize())

    assert store._content_buf == bytearray()
    assert store._nodes_buf == bytearray()
    assert store._ids == {}
    assert store._spans == []
    assert store._node_offsets == {}
    # The snapshot survives the release -- it is taken at finalize ENTRY, before
    # the writes, so post-finalize log sites keep reading real numbers.
    assert store.build_stats is not None
    assert store.build_stats.segment_count == 2


def test_store_fully_readable_after_finalize_buffer_release(tmp_path: Path) -> None:
    """The persisted store stays fully readable AFTER finalize releases the write buffers."""
    store, ha, hb = _finalized_two_segment_store(tmp_path, "rel2")

    assert store._content_buf == bytearray()  # writer state is gone

    with GraphSegmentUnifiedClient(tmp_path, "rel2").open() as c:
        assert orjson.loads(c.get_node_envelope("t0", 0)) == {
            "handles": [ha, hb],
            "dispatch_overrides": {},
            "stream": True,
        }
        assert c.materialize_handles([ha, hb]) == _TWO_SEGMENT_MESSAGES


@pytest.mark.parametrize(
    "call",
    [
        param(lambda s: s.put_segment("b", "user", "hi"), id="put_segment"),
        param(
            lambda s: s.add_node_manifest("t0", 0, b"{}"),
            id="add_node_manifest",
        ),
        param(lambda s: s.segment_handle("a"), id="segment_handle"),
    ],
)  # fmt: skip
def test_write_and_handle_reads_raise_after_finalize(
    tmp_path: Path, call: Callable[[GraphSegmentUnifiedBackingStore], object]
) -> None:
    """Post-finalize writes AND handle lookups fail loud, not silently no-op."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "rel3")
    store.put_segment("a", "system", "SYS")
    asyncio.run(store.finalize())

    with pytest.raises(RuntimeError, match="after finalize"):
        call(store)


def test_double_finalize_fails_loud(tmp_path: Path) -> None:
    """A second ``finalize()`` names the double-finalize case rather than re-writing the store."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "rel4")
    store.put_segment("a", "system", "SYS")
    asyncio.run(store.finalize())

    with pytest.raises(RuntimeError, match="finalize called twice"):
        asyncio.run(store.finalize())


def test_interned_content_idx_is_packed_and_client_reads_int_handles(
    tmp_path: Path,
) -> None:
    """The interned ``content.idx`` is packed binary (not JSON) and the client reads int handles off it, raising on an unknown one."""
    _store, ha, hb = _finalized_two_segment_store(tmp_path, "h2")

    # content.idx is packed binary (8-byte 'Q' pairs), not JSON:
    raw = (tmp_path / "aiperf_graph_segments_h2" / "content.idx").read_bytes()
    assert len(raw) == 2 * 2 * 8  # 2 segments * (off,size) * 8 bytes
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(raw)  # not JSON

    with GraphSegmentUnifiedClient(tmp_path, "h2").open() as c:
        assert c.materialize_handles([ha, hb]) == _TWO_SEGMENT_MESSAGES
        assert c.build_request_body_handles([ha, hb], b"") == _TWO_SEGMENT_BODY
        with pytest.raises(MemoryMapSerializationError):
            c.materialize_handles([999])  # unknown handle raises


def test_content_bytes_counter_equals_blob_size_and_span_sum(tmp_path: Path) -> None:
    """The incremental-spill counter is the single source of truth for content size: it equals the span sum, the reported ``build_stats``, and the on-disk blob length even with a dedup re-put."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "spill-eq")
    store.put_segment("a", "system", "SYS")
    store.put_segment("b", "user", "hi")
    store.put_segment("c", "assistant", "there")
    store.put_segment("a", "system", "SYS")  # dedup: counted/written once

    counter = store._content_bytes_written
    assert counter == sum(size for _off, size in store._spans)

    asyncio.run(store.finalize())

    assert store.build_stats is not None
    assert store.build_stats.content_bytes == counter
    blob = (tmp_path / "aiperf_graph_segments_spill-eq" / "content.blob").read_bytes()
    assert len(blob) == counter


def test_abort_removes_partial_store_files_and_is_idempotent(tmp_path: Path) -> None:
    """A store that errors before finalize leaves no half-written files, and a second ``abort()`` is a no-op rather than a raise."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "abort1")
    store.put_segment("a", "user", "hi")
    d = tmp_path / "aiperf_graph_segments_abort1"
    assert (d / "content.blob").exists()  # eager-opened + written at put time

    store.abort()

    assert not (d / "content.blob").exists()
    assert not (d / "content.idx").exists()
    assert not (d / "nodes.blob").exists()
    assert not (d / "nodes.idx").exists()

    store.abort()  # idempotent second call must not raise


def test_abort_after_finalize_preserves_the_finalized_store(tmp_path: Path) -> None:
    """``abort()`` after a SUCCESSFUL finalize is a safe no-op: the files stay put and the client still materializes."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "abort2")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned("t0", 0, [ha], {}, True)
    asyncio.run(store.finalize())

    store.abort()

    d = tmp_path / "aiperf_graph_segments_abort2"
    assert (d / "content.blob").exists()
    assert (d / "content.idx").exists()
    with GraphSegmentUnifiedClient(tmp_path, "abort2").open() as c:
        assert c.materialize_handles([ha]) == [{"role": "user", "content": "hi"}]
