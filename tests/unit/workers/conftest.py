# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for worker unit tests."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import RequestRecord
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
from aiperf.graph.dynamic_pool import GraphDynamicPool
from aiperf.plugin.enums import EndpointType
from aiperf.workers.worker import Worker

GRAPH_TRACE = "t-1"
# Instance id is ``{template}::{nonce}``; the worker strips the ``::{nonce}``
# back to the base template id for catalog / store lookups.
GRAPH_INSTANCE = "t-1::inst0"


def make_model_endpoint(
    base_url: str = "http://localhost:8000/v1",
) -> ModelEndpointInfo:
    """A real (validating) single-model chat ModelEndpointInfo."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=EndpointType.CHAT, base_url=base_url),
    )


def make_graph_credit_context(node_ordinal: int = 0) -> CreditContext:
    """Graph credit context addressing one node ordinal of the test trace instance."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id=GRAPH_TRACE,
            x_correlation_id="t-1::corr0",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
            trace_id=GRAPH_INSTANCE,
            node_ordinal=node_ordinal,
        ),
        drop_perf_ns=0,
    )


def make_graph_worker(store_reader) -> MagicMock:
    """Mock worker self carrying the REAL pre-dispatch graph error methods."""
    self = MagicMock()
    self._graph_store_reader = store_reader
    self._graph_dynamic_pool = GraphDynamicPool(max_bytes=1024 * 1024)
    self.model_endpoint = make_model_endpoint()
    self._send_inference_result_message = AsyncMock()
    self.inference_client.send_request = AsyncMock()
    self._process_graph_credit = types.MethodType(Worker._process_graph_credit, self)
    self._send_graph_error_record = types.MethodType(
        Worker._send_graph_error_record, self
    )
    self._fail_graph_credit = types.MethodType(Worker._fail_graph_credit, self)
    self._set_graph_envelope_missing = types.MethodType(
        Worker._set_graph_envelope_missing, self
    )
    return self


def sole_sent_record(self: MagicMock) -> RequestRecord:
    """The one and only RequestRecord the mock worker emitted."""
    assert self._send_inference_result_message.await_count == 1
    record = self._send_inference_result_message.await_args.args[0]
    assert isinstance(record, RequestRecord)
    return record


async def graph_client_with_node(
    tmp_path: Path, envelope_extra: dict | None = None
) -> GraphSegmentUnifiedClient:
    """A real finalized unified store carrying ONE node manifest at ordinal 0."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="bench")
    handle = store.put_segment("seg0", "user", "hello")
    store.add_node_manifest(
        GRAPH_TRACE,
        0,
        orjson.dumps({"handles": [handle], **(envelope_extra or {})}),
    )
    await store.finalize()
    return GraphSegmentUnifiedClient(tmp_path, "bench").open()
