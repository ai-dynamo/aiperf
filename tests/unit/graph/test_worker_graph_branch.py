# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker routing of a graph credit to the mmap materializer (D1).

A credit carrying ``trace_id``/``node_ordinal`` must NOT take the linear
session-cache path: the worker opens the unified store reader from the same
(base_path, benchmark_id) the build wrote to, materializes the request, and sends
the materialized payload verbatim through the existing inference client.
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import RequestRecord
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import materialize_graph_request_unified
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import FakeTokenizer

FIX_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


def _graph_client_metadata(
    tmp_path: Path, benchmark_id: str
) -> GraphSegmentClientMetadata:
    """Broadcast-shaped store location the worker opens (Task 5: no env fallback)."""
    return GraphSegmentClientMetadata(
        store_base_path=tmp_path,
        benchmark_id=benchmark_id,
        sidecar_path=tmp_path
        / f"aiperf_graph_meta_{benchmark_id}"
        / "graph_meta.msgpack",
    )


@pytest.fixture
async def mock_worker(
    benchmark_run, fake_tokenizer: FakeTokenizer, skip_service_registration
):
    worker = Worker(run=benchmark_run, service_id="mock-service-id")
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


async def _build_store_for_worker(worker: Worker, tmp_path: Path):
    """Build the interned unified trie store the worker will open.

    Mirrors ``dataset_manager._build_graph_trie_stores``: the content-addressed
    pool AND every node's ``prompt_segment_ids`` manifest land in the ONE
    unified store. The worker opens it from the graph-typed dataset broadcast
    (``GraphSegmentClientMetadata``), which we set directly here.
    """
    parsed = from_weka_trace(str(FIX_MIN))
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=worker.run.benchmark_id
    )
    addr = await build_unified_trie_store_interned(parsed, store)
    worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, worker.run.benchmark_id
    )
    return parsed, addr


@pytest.mark.asyncio
async def test_graph_credit_routes_to_materializer(mock_worker, tmp_path, monkeypatch):
    """A graph credit sends the materialized payload via the inference client."""
    parsed, addr = await _build_store_for_worker(mock_worker, tmp_path)
    t0 = parsed.traces[0].id
    leaf_ordinal = addr[t0]["trace_03_n3:2"]

    captured = {}

    async def fake_send_request(request_info, first_token_callback=None):
        captured["request_info"] = request_info
        return RequestRecord(
            request_info=request_info,
            timestamp_ns=time.time_ns(),
            start_perf_ns=time.perf_counter_ns(),
            end_perf_ns=time.perf_counter_ns(),
        )

    monkeypatch.setattr(
        mock_worker.inference_client,
        "send_request",
        AsyncMock(side_effect=fake_send_request),
    )
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=7,
        phase=CreditPhase.PROFILING,
        conversation_id=t0,
        x_correlation_id="x-7",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id=t0,
        node_ordinal=leaf_ordinal,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())

    await mock_worker._process_credit(ctx)

    # Inference client was called exactly once with the materialized payload.
    mock_worker.inference_client.send_request.assert_awaited_once()
    request_info = captured["request_info"]
    assert len(request_info.turns) == 1
    # The interned unified store default: the graph credit takes the BYTES
    # path -- raw_payload is None and raw_payload_bytes carries the
    # materialized body verbatim.
    raw_payload_bytes = request_info.turns[0].raw_payload_bytes
    assert raw_payload_bytes is not None
    wire = orjson.loads(raw_payload_bytes)

    # Reconstruct the expected materialized payload through the same unified
    # reader the worker opened; the dict-path materializer walks the same
    # interned handles, so the messages match the bytes path's wire body.
    expected_client = GraphSegmentUnifiedClient(
        base_path=tmp_path, benchmark_id=mock_worker.run.benchmark_id
    ).open()
    expected = materialize_graph_request_unified(
        expected_client,
        t0,
        leaf_ordinal,
        "profiling",
        use_legacy_max_tokens=mock_worker.model_endpoint.endpoint.use_legacy_max_tokens,
    )
    expected_client.close()
    assert wire["messages"] == expected["messages"]
    assert wire["messages"]
    # No linear session was created for the graph credit.
    assert mock_worker.session_manager.get("x-7") is None


@pytest.mark.asyncio
async def test_missing_graph_envelope_records_error_no_send(
    mock_worker, tmp_path, monkeypatch
):
    """A graph credit with no stored delta records an error and sends nothing."""
    parsed, addr = await _build_store_for_worker(mock_worker, tmp_path)
    t0 = parsed.traces[0].id

    monkeypatch.setattr(mock_worker.inference_client, "send_request", AsyncMock())
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=8,
        phase=CreditPhase.PROFILING,
        conversation_id=t0,
        x_correlation_id="x-8",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id=t0,
        node_ordinal=9999,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())

    await mock_worker._process_credit(ctx)

    mock_worker.inference_client.send_request.assert_not_awaited()
    assert ctx.error is not None
    assert ctx.error.type == "GraphEnvelopeMissing"


@pytest.mark.asyncio
async def test_pre_v3_unified_store_rejection_surfaces_in_fatal_error(
    mock_worker, tmp_path, monkeypatch
):
    """An on-disk pre-v3 (A1 JSON index) unified store surfaces its A2-strict
    rejection in the fatal error instead of claiming no store exists."""
    mock_worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, mock_worker.run.benchmark_id
    )
    store_dir = tmp_path / f"aiperf_graph_segments_{mock_worker.run.benchmark_id}"
    store_dir.mkdir(parents=True)
    # Legacy A1 index: a JSON object (starts with b'{') that the A2-strict
    # reader rejects with "re-parse required".
    (store_dir / "content.idx").write_bytes(b'{"ab12": [0, 4]}')

    monkeypatch.setattr(mock_worker.inference_client, "send_request", AsyncMock())
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=10,
        phase=CreditPhase.PROFILING,
        conversation_id="t-x",
        x_correlation_id="x-10",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id="t-x",
        node_ordinal=0,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())

    await mock_worker._process_credit(ctx)

    mock_worker.inference_client.send_request.assert_not_awaited()
    assert ctx.error is not None
    assert ctx.error.type == "GraphStoreUnavailable"
    assert "rejected" in ctx.error.message
    assert "re-parse required" in ctx.error.message


@pytest.mark.asyncio
async def test_non_graph_credit_still_uses_session_path(mock_worker, monkeypatch):
    """A non-graph credit (no trace_id) must NOT touch the graph materializer."""
    # Force the session path to a controlled early return: no dataset client and
    # not stopping triggers the dataset-manager fallback we can assert against.
    called = {"graph": False}
    orig = mock_worker._process_graph_credit

    async def spy(*a, **k):
        called["graph"] = True
        return await orig(*a, **k)

    monkeypatch.setattr(mock_worker, "_process_graph_credit", spy)
    monkeypatch.setattr(
        mock_worker,
        "_retrieve_conversation",
        AsyncMock(side_effect=RuntimeError("stop")),
    )

    credit = Credit(
        id=9,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-9",
        x_correlation_id="x-9",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())

    await mock_worker._process_credit(ctx)
    assert called["graph"] is False
