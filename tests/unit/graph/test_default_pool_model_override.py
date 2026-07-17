# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: a weka trie node honors a run-level ``--model`` override on the wire.

The segment-trie IR (the sole weka path) carries each node's RECORDED ``model``
verbatim in its build-time ``dispatch_overrides`` manifest -- there is NO build-
time model map (the legacy ``build_weka_model_map`` remapped recorded models to
``endpoint.model_names``; the trie path does not). The user's ``--model`` override
is layered run-level at the worker via
:func:`~aiperf.graph.worker_materialize.apply_run_level_payload_options`
(``endpoint.extra`` carrying ``("model", ...)`` CLOBBERS the per-node key, agentx
``payload.update(endpoint.extra)`` precedence), so the recorded model NEVER reaches
the wire when an override is configured.

These tests build the real interned unified trie store from a weka fixture,
materialize a node, and assert the override wins on the wire while the recorded
model leaks only when no override is configured.
"""

from pathlib import Path

import pytest

from aiperf.common.models import EndpointInfo
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import (
    apply_run_level_payload_options,
    materialize_graph_request_unified,
)

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"

# The single recorded model in the fixture -- the value the trie manifest carries
# and that must NOT reach the wire once an override is configured.
_RECORDED_MODEL = "claude-opus-4-5-20251101"
# The user's run-level ``--model`` override -- the value that MUST reach the wire.
_OVERRIDE_MODEL = "my-served-model"


async def _build_trie_store(
    tmp_path: Path, benchmark_id: str
) -> tuple[str, int, GraphSegmentUnifiedClient]:
    """Build the interned unified trie store for the fixture's sole trace.

    Returns ``(trace_id, first_node_ordinal, client)`` with the client opened;
    the caller closes it.
    """
    parsed = from_weka_trace(str(FIX))
    assert parsed.segment_pool is not None, "weka now always builds the trie IR"

    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=benchmark_id
    )
    catalog = await build_unified_trie_store_interned(parsed, store)

    trace_id = parsed.traces[0].id
    node_ordinals = catalog[trace_id]
    assert node_ordinals, "expected per-node trie ordinals"
    first_ordinal = min(node_ordinals.values())

    client = GraphSegmentUnifiedClient(
        base_path=tmp_path, benchmark_id=benchmark_id
    ).open()
    return trace_id, first_ordinal, client


@pytest.mark.asyncio
async def test_trie_envelope_carries_recorded_model(tmp_path):
    """With no override, the recorded model stands on the materialized payload."""
    trace_id, ordinal, client = await _build_trie_store(tmp_path, "rec")
    try:
        payload = materialize_graph_request_unified(
            client, trace_id, ordinal, "profiling"
        )
        assert payload is not None
        assert payload["model"] == _RECORDED_MODEL
    finally:
        client.close()


@pytest.mark.asyncio
async def test_run_level_model_override_reaches_wire(tmp_path):
    """A run-level ``--model`` (``endpoint.extra``) clobbers the recorded model.

    The trie carries the recorded model per-node; the worker layers the run-level
    override last, so the override reaches the wire and the recorded model never
    leaks. This is the trie-path equivalent of the legacy build-time model map.
    """
    trace_id, ordinal, client = await _build_trie_store(tmp_path, "ovr")
    try:
        payload = materialize_graph_request_unified(
            client, trace_id, ordinal, "profiling"
        )
        assert payload is not None
        # Precondition: before the run-level layer the recorded model is present.
        assert payload["model"] == _RECORDED_MODEL

        endpoint = EndpointInfo(
            type="chat",
            streaming=True,
            use_server_token_count=False,
            extra=[("model", _OVERRIDE_MODEL)],
        )
        apply_run_level_payload_options(payload, endpoint)

        assert payload["model"] == _OVERRIDE_MODEL, "run-level --model wins on the wire"
        assert payload["model"] != _RECORDED_MODEL, "recorded model must not leak"
    finally:
        client.close()
