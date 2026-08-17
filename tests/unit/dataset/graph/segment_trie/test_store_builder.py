# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming-split gates, trie-envelope body folding, and unified-store read hardening."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.exceptions import MemoryMapSerializationError
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.dataset.graph.segment_trie.store_builder import (
    _trie_envelope,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


def parsed_with_trie_meta(extra_trie_meta: dict) -> ParsedGraph:
    """A one-node, one-trace graph whose node carries ``extra_trie_meta`` alongside its segment ids."""
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


def envelope_for(node: LlmNode, segment_ids: list[str] | None = None) -> dict:
    """The trie envelope the streaming store would persist for ``node``."""
    return _trie_envelope(node, segment_ids or ["a"])


class TestStreamingSplitSlotGate:
    """S4: the streaming split refuses nodes whose dynamic slots the streamed envelope cannot carry."""

    @pytest.mark.parametrize(
        "extra_trie_meta",
        [
            param({"assembly": [{"s": {"src": "n0"}}]}, id="assembly_items"),
            param({"capture": True}, id="capture"),
        ],
    )  # fmt: skip
    def test_slot_carrying_node_rejected_by_name(self, extra_trie_meta: dict) -> None:
        """A node with assembly items or capture fails loud, naming the node."""
        # Silently persisting would yield a manifest missing the node's dynamic
        # slots, which only surfaces much later as a wrong-payload dispatch.
        with pytest.raises(NotImplementedError, match="'n0'"):
            list(iter_trace_segment_payloads(parsed_with_trie_meta(extra_trie_meta)))

    def test_slotless_node_accepted(self) -> None:
        """A node with only segment ids streams out as a single payload with envelopes."""
        payloads = list(iter_trace_segment_payloads(parsed_with_trie_meta({})))
        assert len(payloads) == 1 and payloads[0].envelopes


class TestEndpointExtraAppliedFlag:
    """The adapter-owned ``endpoint_extra_applied`` stamp rides the envelope only when actually set."""

    def test_omitted_when_unset(self) -> None:
        """An unstamped node's envelope has no such key and stays byte-identical to the historical shape."""
        # Byte equality is the guarantee that existing weka/dynamo/native corpora
        # envelopes are unchanged by the flag's introduction.
        env = envelope_for(
            LlmNode(prompt=[], output="o", arrival_offset_us=0, streaming=False),
            ["a", "b"],
        )
        assert "endpoint_extra_applied" not in env
        assert orjson.dumps(env) == orjson.dumps(
            {
                "prompt_segment_ids": ["a", "b"],
                "dispatch_overrides": {},
                "stream": False,
            }
        )

    def test_carried_when_stamped(self) -> None:
        """A node stamped by the adapter carries the flag so the worker skips re-merging ``--extra-inputs``."""
        env = envelope_for(
            LlmNode(
                prompt=[],
                output="o",
                arrival_offset_us=0,
                streaming=False,
                metadata={"dispatch": {"endpoint_extra_applied": True}},
            )
        )
        assert env["endpoint_extra_applied"] is True


class TestNativeBodyFieldFold:
    """Native Turn-named fields fold into the envelope's wire body, after any authored ``extra_body`` keys."""

    def test_model_max_tokens_and_tools_fold_into_body(self) -> None:
        """``model`` / ``max_tokens`` / ``raw_tools`` land in dispatch_overrides while ``extra_headers`` rides the top level."""
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        env = envelope_for(
            LlmNode(
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
        )
        assert env["dispatch_overrides"] == {
            "temperature": 0.5,
            "model": "m",
            "max_output_tokens": 25,
            "tools": tools,
        }
        assert env["extra_headers"] == {"x-dynamo-session-id": "s1"}
        assert "extra_headers" not in env["dispatch_overrides"]
        assert env["stream"] is False

    def test_hand_authored_extra_body_entry_wins(self) -> None:
        """A hand-authored ``extra_body`` entry naming a foldable key beats the native field."""
        env = envelope_for(
            LlmNode(
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
        )
        assert env["dispatch_overrides"] == {
            "model": "override-model",
            "max_output_tokens": 3,
            "top_p": 0.9,
        }

    def test_nothing_folds_when_fields_unset(self) -> None:
        """A node with no body fields set produces empty overrides and no headers key."""
        env = envelope_for(
            LlmNode(prompt=[], output="o", arrival_offset_us=0, streaming=False)
        )
        assert env["dispatch_overrides"] == {}
        assert "extra_headers" not in env


class TestContentRegionBoundsCheck:
    """S5: content-region spans are bounds-checked when the store is opened."""

    def test_truncated_content_blob_fails_loud_on_open(self, tmp_path: Path) -> None:
        """A span past the end of content.blob raises at open instead of silently returning clamped bytes."""
        store = GraphSegmentUnifiedBackingStore(tmp_path, "b")
        handle = store.put_segment("seg_a", "user", "hello world, plenty of content")
        store.add_node_manifest_interned("t0", 0, [handle], {}, True)
        asyncio.run(store.finalize())

        blob = tmp_path / "aiperf_graph_segments_b" / "content.blob"
        blob.write_bytes(blob.read_bytes()[:-4])

        with pytest.raises(MemoryMapSerializationError, match=r"content\.blob"):
            GraphSegmentUnifiedClient(tmp_path, "b").open()
