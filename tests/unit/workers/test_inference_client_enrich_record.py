# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CreditPhase
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import (
    RecordContext,
    RequestInfo,
    RequestRecord,
)
from aiperf.workers.inference_client import InferenceClient


def _make_request_info(**overrides) -> RequestInfo:
    defaults = dict(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="c",
        turn_index=0,
        x_request_id="r",
        x_correlation_id="x",
        agent_depth=2,
        parent_correlation_id="root",
        model_endpoint=ModelEndpointInfo.model_construct(),
        turns=[],
    )
    defaults.update(overrides)
    return RequestInfo(**defaults)


class TestEnrichRequestRecord:
    def test_record_context_replaces_request_info_on_record(self):
        ri = _make_request_info()
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        assert enriched.request_info is not None
        # Pure RecordContext, NOT a RequestInfo subclass instance.
        assert type(enriched.request_info) is RecordContext

    def test_dag_fields_propagate(self):
        ri = _make_request_info(agent_depth=3, parent_correlation_id="p")
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        assert enriched.request_info.agent_depth == 3
        assert enriched.request_info.parent_correlation_id == "p"

    def test_transport_extras_dropped(self):
        ri = _make_request_info()
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        # downcast strips model_endpoint / endpoint_headers / drop_perf_ns,
        # but ``turns``, ``system_message``, ``user_context_message`` were
        # hoisted onto RecordContext (records pipeline reads them) and
        # therefore survive the downcast.
        dump = enriched.request_info.model_dump()
        assert "model_endpoint" not in dump
        assert "endpoint_headers" not in dump
        assert "drop_perf_ns" not in dump
        assert "turns" in dump
