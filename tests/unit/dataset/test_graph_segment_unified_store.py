import asyncio

import orjson
import pytest

from aiperf.common.exceptions import MemoryMapSerializationError
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


def test_unified_backing_store_writes_four_files(tmp_path):
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b1")
    store.put_segment("a", "system", "SYS")
    store.put_segment("b", "user", "hi")
    env = orjson.dumps({"prompt_segment_ids": ["a", "b"], "stream": True})
    store.add_node_manifest("t0", 0, "profiling", env)
    asyncio.run(store.finalize())

    d = tmp_path / "aiperf_graph_segments_b1"
    assert (d / "content.blob").exists()
    assert (d / "content.idx").exists()
    assert (d / "nodes.blob").exists()
    assert (d / "nodes.idx").exists()

    client = GraphSegmentUnifiedClient(tmp_path, "b1")
    assert client.data_dir == d


def test_unified_client_duck_types_both_stores(tmp_path):
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b2")
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha, hb], {}, True)
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "b2").open() as c:
        # envelope addressing face:
        raw = c.get_node_envelope("t0", 0, "profiling")
        assert orjson.loads(raw) == {
            "handles": [ha, hb],
            "dispatch_overrides": {},
            "stream": True,
        }
        assert c.get_node_envelope("t0", 9, "profiling") is None
        # Segment-content face:
        assert c.materialize_handles([ha, hb]) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ]
        assert c.build_request_body_handles([ha, hb], b"") == (
            b'{"messages":[{"role":"system","content":"SYS"},'
            b'{"role":"user","content":"hi"}]}'
        )


def test_endpoint_extra_applied_flag_round_trips_when_set(tmp_path):
    """A manifest written with ``endpoint_extra_applied=True`` reads back with the
    flag; the key is written only when True."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "eea1")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned(
        "t0", 0, "profiling", [ha], {}, True, endpoint_extra_applied=True
    )
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "eea1").open() as c:
        env = orjson.loads(c.get_node_envelope("t0", 0, "profiling"))
        assert env["endpoint_extra_applied"] is True


def test_endpoint_extra_applied_flag_omitted_leaves_envelope_bytes_unchanged(tmp_path):
    """Default (unset) leaves the manifest envelope byte-identical to a store
    written without the parameter at all -- byte-neutrality when the flag is
    off is the whole contract."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "eea2")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha], {}, True)
    asyncio.run(store.finalize())

    with GraphSegmentUnifiedClient(tmp_path, "eea2").open() as c:
        raw = c.get_node_envelope("t0", 0, "profiling")
        assert b"endpoint_extra_applied" not in raw
        assert orjson.loads(raw) == {
            "handles": [ha],
            "dispatch_overrides": {},
            "stream": True,
        }


def test_put_segment_returns_stable_insertion_index_handles(tmp_path):
    store = GraphSegmentUnifiedBackingStore(tmp_path, "h1")
    assert store.put_segment("a", "system", "SYS") == 0
    assert store.put_segment("b", "user", "hi") == 1
    assert store.put_segment("a", "system", "SYS") == 0  # dedup -> same handle
    assert store.segment_handle("b") == 1
    assert store.segment_handle("missing") is None


def test_unified_client_empty_pool_and_unknown_handle(tmp_path):
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


def test_a1_json_content_idx_is_rejected_with_reparse_error(tmp_path):
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


def test_finalize_releases_write_side_buffers(tmp_path):
    """finalize() drops every write-side buffer at its END.

    The store object outlives finalize through the structural-merge / sidecar /
    prefix-cache build tail, where NOTHING reads these accumulation buffers
    (``build_stats`` is already snapshotted at finalize ENTRY). Holding the
    load-bearing ones (``_ids`` dedup map, ``_spans`` content.idx source,
    ``_node_offsets``, ``_nodes_buf``) just pins RAM for zero readers. Content
    is spilled to disk at ``put`` time, so ``_content_buf`` is already empty
    (its clear is a formality) and the running counter tracks the footprint.
    This pins the memory-release contract: an edit that stops clearing a buffer
    trips here.
    """
    store = GraphSegmentUnifiedBackingStore(tmp_path, "rel1")
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha, hb], {}, True)

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


def test_store_fully_readable_after_finalize_buffer_release(tmp_path):
    """The persisted store stays fully readable AFTER finalize releases the write
    buffers: open with the client and materialize a node envelope + its content
    end-to-end. Byte parity is proven by the dynamo/dag_jsonl parity suites; here
    we prove readability survives the release, since readers work off the flushed
    files, never the writer's dropped buffers."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "rel2")
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha, hb], {}, True)
    asyncio.run(store.finalize())

    assert store._content_buf == bytearray()  # writer state is gone

    with GraphSegmentUnifiedClient(tmp_path, "rel2").open() as c:
        assert orjson.loads(c.get_node_envelope("t0", 0, "profiling")) == {
            "handles": [ha, hb],
            "dispatch_overrides": {},
            "stream": True,
        }
        assert c.materialize_handles([ha, hb]) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ]


def test_write_and_handle_reads_raise_after_finalize(tmp_path):
    """Post-finalize writes AND handle lookups fail loud, not silently no-op.

    ``put_segment`` / ``add_node_manifest`` already guarded; ``segment_handle``
    now guards too because its ``_ids`` map is released at finalize END -- an
    unguarded lookup would silently return ``None`` (misresolving a handle) where
    it previously returned the real handle. Double-finalize stays a hard error.
    """
    store = GraphSegmentUnifiedBackingStore(tmp_path, "rel3")
    store.put_segment("a", "system", "SYS")
    asyncio.run(store.finalize())

    with pytest.raises(RuntimeError, match="after finalize"):
        store.put_segment("b", "user", "hi")
    with pytest.raises(RuntimeError, match="after finalize"):
        store.add_node_manifest("t0", 0, "profiling", b"{}")
    with pytest.raises(RuntimeError, match="after finalize"):
        store.segment_handle("a")
    with pytest.raises(RuntimeError, match="finalize called twice"):
        asyncio.run(store.finalize())


def test_interned_content_idx_is_packed_and_client_reads_int_handles(tmp_path):
    store = GraphSegmentUnifiedBackingStore(tmp_path, "h2")
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha, hb], {}, True)
    asyncio.run(store.finalize())

    # content.idx is packed binary (8-byte 'Q' pairs), not JSON:
    raw = (tmp_path / "aiperf_graph_segments_h2" / "content.idx").read_bytes()
    assert len(raw) == 2 * 2 * 8  # 2 segments * (off,size) * 8 bytes
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(raw)  # not JSON

    with GraphSegmentUnifiedClient(tmp_path, "h2").open() as c:
        assert c.materialize_handles([ha, hb]) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ]
        assert c.build_request_body_handles([ha, hb], b"") == (
            b'{"messages":[{"role":"system","content":"SYS"},'
            b'{"role":"user","content":"hi"}]}'
        )
        with pytest.raises(MemoryMapSerializationError):
            c.materialize_handles([999])  # unknown handle raises


def test_content_bytes_counter_equals_blob_size_and_span_sum(tmp_path):
    """The incremental-spill counter is the single source of truth for content
    size: ``_content_bytes_written`` == sum of span lengths (pre-finalize) ==
    the finalized ``content.blob`` size == ``build_stats.content_bytes``.

    This is the spill's core invariant -- a divergence would mean a segment's
    bytes were counted but not written (or vice versa), silently corrupting
    every span offset after it.
    """
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


def test_abort_removes_partial_store_files_and_is_idempotent(tmp_path):
    """A store that errors before finalize leaves no half-written files.

    put_segment streams straight to ``content.blob``, so an aborted build would
    otherwise leave a partial blob for a later open to trip on. ``abort()``
    closes the write handle and unlinks the four files; it is idempotent and
    non-raising (safe to call twice, and after finalize).
    """
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


def test_abort_after_finalize_preserves_the_finalized_store(tmp_path):
    """``abort()`` after a SUCCESSFUL finalize is a safe no-op: the complete
    on-disk store is never unlinked (the ``_finalized`` guard skips the unlink),
    so a stray abort on the builder's success path cannot delete a good store."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "abort2")
    ha = store.put_segment("a", "user", "hi")
    store.add_node_manifest_interned("t0", 0, "profiling", [ha], {}, True)
    asyncio.run(store.finalize())

    store.abort()

    d = tmp_path / "aiperf_graph_segments_abort2"
    assert (d / "content.blob").exists()
    assert (d / "content.idx").exists()
    with GraphSegmentUnifiedClient(tmp_path, "abort2").open() as c:
        assert c.materialize_handles([ha]) == [{"role": "user", "content": "hi"}]
