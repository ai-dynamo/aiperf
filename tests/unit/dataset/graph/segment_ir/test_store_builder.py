# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming-split gates and unified-store read hardening."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest

from aiperf.common.exceptions import MemoryMapSerializationError
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph.segment_ir.store_builder import (
    _trie_envelope,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)

# --- S4: streaming split rejects slot-carrying nodes loudly ------------------


def _parsed_with_trie_meta(extra_trie_meta: dict) -> ParsedGraph:
    pool = SegmentPool()
    sid = pool.add(role="user", content="hi", tokens=[1, 2], parent_id=None)
    node = LlmNode(
        prompt=[],
        output="n0_out",
        arrival_offset_us=0,
        metadata={"trie": {"prompt_segment_ids": [sid], **extra_trie_meta}},
    )
    return ParsedGraph(
        graph=GraphRecord(nodes={"n0": node}),
        traces=[TraceRecord(id="t0")],
        segment_pool=pool,
    )


@pytest.mark.parametrize(
    "extra_trie_meta",
    [
        pytest.param({"assembly": [{"s": {"src": "n0"}}]}, id="assembly_items"),
        pytest.param({"capture": True}, id="capture"),
    ],
)  # fmt: skip
def test_streaming_split_rejects_slot_carrying_node(extra_trie_meta: dict) -> None:
    """Assembly items / capture never reach the streamed envelope (it carries
    neither), so the streaming split must fail loud naming the node instead of
    silently persisting a manifest missing the node's dynamic slots."""
    parsed = _parsed_with_trie_meta(extra_trie_meta)
    with pytest.raises(NotImplementedError, match="'n0'"):
        list(iter_trace_segment_payloads(parsed))


def test_streaming_split_accepts_slotless_node() -> None:
    parsed = _parsed_with_trie_meta({})
    payloads = list(iter_trace_segment_payloads(parsed))
    assert len(payloads) == 1 and payloads[0].envelopes


# --- endpoint_extra_applied: adapter-owned extras precedence flag -------------


def test_trie_envelope_omits_endpoint_extra_applied_when_unset() -> None:
    """A node WITHOUT the adapter stamp yields an envelope with NO
    ``endpoint_extra_applied`` key and byte-identical ``orjson.dumps`` output to
    before -- weka/dynamo/native corpora envelopes stay bit-for-bit unchanged."""
    node = LlmNode(prompt=[], output="o", arrival_offset_us=0, streaming=False)
    env = _trie_envelope(node, ["a", "b"])
    assert "endpoint_extra_applied" not in env
    assert orjson.dumps(env) == orjson.dumps(
        {
            "prompt_segment_ids": ["a", "b"],
            "dispatch_overrides": {},
            "stream": False,
        }
    )


def test_trie_envelope_carries_endpoint_extra_applied_when_stamped() -> None:
    """A node stamped ``metadata["dispatch"]["endpoint_extra_applied"] = True``
    (adapter folded the run's ``--extra-inputs`` into ``dispatch_overrides`` at
    parse) carries the flag in its envelope so the worker skips re-merging."""
    node = LlmNode(
        prompt=[],
        output="o",
        arrival_offset_us=0,
        streaming=False,
        metadata={"dispatch": {"endpoint_extra_applied": True}},
    )
    env = _trie_envelope(node, ["a"])
    assert env["endpoint_extra_applied"] is True


# --- native body-field fold-in --------------------------------------------------


def test_trie_envelope_folds_native_body_fields() -> None:
    """The native Turn-named fields (``model`` / ``max_tokens`` / ``raw_tools``)
    fold into the envelope's wire-body dict after the ``extra_body`` vendor
    keys; ``extra_headers`` rides the envelope top level, never the body."""
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    node = LlmNode(
        prompt=[],
        output="o",
        arrival_offset_us=0,
        streaming=False,
        model="m",
        max_tokens=25,
        raw_tools=tools,
        extra_headers={"x-dynamo-session-id": "s1"},
        extra_body={"temperature": 0.5},
    )
    env = _trie_envelope(node, ["a"])
    assert env["dispatch_overrides"] == {
        "temperature": 0.5,
        "model": "m",
        "max_output_tokens": 25,
        "tools": tools,
    }
    assert env["extra_headers"] == {"x-dynamo-session-id": "s1"}
    assert "extra_headers" not in env["dispatch_overrides"]
    assert env["stream"] is False


def test_trie_envelope_fold_skips_hand_authored_extra_body_entry() -> None:
    """A hand-authored ``extra_body`` entry naming a foldable key wins: the
    fold never duplicates or overwrites it."""
    node = LlmNode(
        prompt=[],
        output="o",
        arrival_offset_us=0,
        streaming=True,
        model="native-model",
        max_tokens=7,
        extra_body={
            "model": "override-model",
            "max_output_tokens": 3,
            "top_p": 0.9,
        },
    )
    env = _trie_envelope(node, ["a"])
    assert env["dispatch_overrides"] == {
        "model": "override-model",
        "max_output_tokens": 3,
        "top_p": 0.9,
    }


def test_trie_envelope_no_folds_when_fields_unset() -> None:
    node = LlmNode(prompt=[], output="o", arrival_offset_us=0, streaming=False)
    env = _trie_envelope(node, ["a"])
    assert env["dispatch_overrides"] == {}
    assert "extra_headers" not in env


# --- S5: content-region spans bounds-checked at open --------------------------


def test_truncated_content_blob_fails_loud_on_open(tmp_path: Path) -> None:
    """A span past the end of content.blob (stale/partial store) must raise at
    open instead of Python slice clamping returning truncated bytes."""
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b")
    handle = store.put_segment("seg_a", "user", "hello world, plenty of content")
    store.add_node_manifest_interned("t0", 0, "profiling", [handle], {}, True)
    asyncio.run(store.finalize())

    blob = tmp_path / "aiperf_graph_segments_b" / "content.blob"
    blob.write_bytes(blob.read_bytes()[:-4])

    with pytest.raises(MemoryMapSerializationError, match="content.blob"):
        GraphSegmentUnifiedClient(tmp_path, "b").open()
