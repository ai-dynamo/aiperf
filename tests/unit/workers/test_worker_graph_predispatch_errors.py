# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-dispatch graph credit failures must emit a synthetic error record (WK2).

RecordsManager's completion barrier counts success+error RECORDS against the
credit-side ``final_requests_completed``; a graph credit that errors BEFORE
dispatch (missing store, missing envelope, missing pool entry) returns its
credit with an error but -- without the fix -- produced no ``RequestRecord``,
starving the barrier and hanging the run at "please wait for the results".
These tests drive the real ``Worker._process_graph_credit`` /
``Worker._send_graph_error_record`` bound onto a mock self and assert every
pre-dispatch error path pushes exactly one error ``InferenceResultsMessage``
record while preserving the credit-return error attribution.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.dynamic_pool import GraphDynamicPool, GraphPoolMissingError
from aiperf.plugin.enums import EndpointType
from aiperf.workers.worker import Worker

TRACE = "t-1"
# Instance id is ``{template}::{nonce}``; the worker strips the ``::{nonce}``
# back to the base template id for catalog / store lookups.
INSTANCE = "t-1::inst0"


def _model_endpoint() -> ModelEndpointInfo:
    """A real (validating) ModelEndpointInfo for the synthetic RequestInfo."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000/v1",
        ),
    )


def _credit_context(node_ordinal: int = 0) -> CreditContext:
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id=TRACE,
            x_correlation_id="t-1::corr0",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
            trace_id=INSTANCE,
            node_ordinal=node_ordinal,
        ),
        drop_perf_ns=0,
    )


def _mock_worker(store_reader) -> MagicMock:
    """Mock self carrying the REAL pre-dispatch error methods under test."""
    self = MagicMock()
    self._graph_store_reader = store_reader
    self._graph_dynamic_pool = GraphDynamicPool(max_bytes=1024 * 1024)
    self.model_endpoint = _model_endpoint()
    self._send_inference_result_message = AsyncMock()
    self.inference_client.send_request = AsyncMock()
    # Bind the real methods under test onto the mock self.
    self._process_graph_credit = types.MethodType(Worker._process_graph_credit, self)
    self._send_graph_error_record = types.MethodType(
        Worker._send_graph_error_record, self
    )
    self._set_graph_envelope_missing = types.MethodType(
        Worker._set_graph_envelope_missing, self
    )
    return self


def _sole_sent_record(self: MagicMock) -> RequestRecord:
    assert self._send_inference_result_message.await_count == 1
    record = self._send_inference_result_message.await_args.args[0]
    assert isinstance(record, RequestRecord)
    return record


async def _client_with_node(
    tmp_path: Path, envelope_extra: dict | None = None
) -> GraphSegmentUnifiedClient:
    """A real finalized unified store carrying ONE node manifest at ordinal 0."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="bench")
    handle = store.put_segment("seg0", "user", "hello")
    store.add_node_manifest(
        TRACE,
        0,
        "profiling",
        orjson.dumps({"handles": [handle], **(envelope_extra or {})}),
    )
    await store.finalize()
    return GraphSegmentUnifiedClient(tmp_path, "bench").open()


@pytest.mark.asyncio
async def test_store_missing_credit_emits_error_record() -> None:
    """A missing graph store sends a synthetic error InferenceResults record."""
    store_error = ErrorDetails(
        type="GraphStoreUnavailable", message="no store could be opened"
    )

    def _reader(credit_context: CreditContext) -> None:
        # Mirrors the real _graph_store_reader failure contract: attribute the
        # error on the context and return None.
        credit_context.error = store_error
        return None

    self = _mock_worker(MagicMock(side_effect=_reader))
    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-1", None)

    record = _sole_sent_record(self)
    assert record.error is store_error
    assert record.valid is False
    assert record.request_info.x_request_id == "x-req-1"
    assert record.request_info.x_correlation_id == ctx.credit.x_correlation_id
    # Credit-return semantics preserved: the context still carries the error.
    assert ctx.error is store_error
    # Nothing was dispatched to the inference server.
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_envelope_missing_credit_emits_error_record(tmp_path: Path) -> None:
    """An unaddressable node ordinal sends a GraphEnvelopeMissing error record."""
    client = await _client_with_node(tmp_path)
    self = _mock_worker(MagicMock(return_value=client))
    ctx = _credit_context(node_ordinal=7)
    await self._process_graph_credit(ctx, "x-req-2", None)

    record = _sole_sent_record(self)
    assert record.error is not None
    assert record.error.type == "GraphEnvelopeMissing"
    assert record.valid is False
    assert ctx.error is record.error
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_missing_credit_emits_error_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dynamic-pool entry sends an error record carrying the sniffable prefix."""
    import aiperf.graph.worker_materialize as wm

    client = await _client_with_node(tmp_path, envelope_extra={"items": []})

    def _raise_pool_missing(*args, **kwargs):
        raise GraphPoolMissingError(3)

    monkeypatch.setattr(wm, "materialize_graph_request_unified", _raise_pool_missing)
    self = _mock_worker(MagicMock(return_value=client))
    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-3", None)

    record = _sole_sent_record(self)
    assert record.error is not None
    assert record.valid is False
    assert "aiperf.graph.pool_missing:" in record.error.message
    # The context keeps the raw prefixed string the dispatch adapter sniffs.
    assert isinstance(ctx.error, str)
    assert ctx.error.startswith("aiperf.graph.pool_missing:")
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_dispatch_sends_single_record(tmp_path: Path) -> None:
    """Control: the happy path still sends exactly one (dispatch-built) record."""
    client = await _client_with_node(tmp_path)
    self = _mock_worker(MagicMock(return_value=client))
    self._dispatch_graph_request = AsyncMock()
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-4", None)

    self._dispatch_graph_request.assert_awaited_once()
    # No synthetic pre-dispatch record on the happy path.
    self._send_inference_result_message.assert_not_awaited()
    assert ctx.error is None


@pytest.mark.asyncio
async def test_unexpected_exception_emits_error_record_and_credit_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An UNANTICIPATED raiser inside ``_process_graph_credit`` (here a corrupt
    envelope failing ``orjson.loads`` in ``read_node_envelope``) must still emit
    exactly one error record AND attribute the error on the CreditReturn --
    without the catch-all the credit returns error=None (counted as a success)
    while no record flows, starving the RecordsManager barrier."""
    import aiperf.graph.worker_materialize as wm

    def _raise_decode_error(*args, **kwargs):
        raise orjson.JSONDecodeError("corrupt envelope bytes", "", 0)

    monkeypatch.setattr(wm, "read_node_envelope", _raise_decode_error)
    self = _mock_worker(MagicMock(return_value=MagicMock()))
    self._prefill_concurrency_enabled = False
    self.credit_return_push_client.send = AsyncMock()
    # Drive the full credit task so the CreditReturn built in its finally is
    # observable alongside the record.
    self._process_credit = types.MethodType(Worker._process_credit, self)
    self._on_credit_drop_message_task = types.MethodType(
        Worker._on_credit_drop_message_task, self
    )
    ctx = _credit_context()

    await self._on_credit_drop_message_task(ctx)

    record = _sole_sent_record(self)
    assert record.error is not None
    assert record.valid is False
    assert "corrupt envelope bytes" in record.error.message
    # Credit-return attribution: the barrier-side counterpart must see an error.
    assert ctx.error is not None
    credit_return = self.credit_return_push_client.send.await_args.args[0]
    assert credit_return.error is not None, (
        "an escaped exception must not be counted as a completed request"
    )
    self.inference_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_after_dispatch_record_does_not_double_emit(
    tmp_path: Path,
) -> None:
    """A raiser AFTER the dispatch path resolved (here the pool bracket close)
    attributes the error but must NOT emit a second, synthetic record."""
    client = await _client_with_node(tmp_path)
    self = _mock_worker(MagicMock(return_value=client))
    self._dispatch_graph_request = AsyncMock()
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    self._graph_dynamic_pool.credit_finished = MagicMock(
        side_effect=RuntimeError("post-record boom")
    )
    ctx = _credit_context()

    await self._process_graph_credit(ctx, "x-req-5", None)

    self._dispatch_graph_request.assert_awaited_once()
    # The dispatch path owns the record; the catch-all must not add another.
    self._send_inference_result_message.assert_not_awaited()
    assert ctx.error is not None
