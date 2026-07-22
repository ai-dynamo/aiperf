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
        phase_index=2,
        profiling_index=1,
        phase_name="recovery",
        phase_kind="profiling",
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
        ri = _make_request_info(
            agent_depth=3, parent_correlation_id="p", root_correlation_id="root-1"
        )
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        assert enriched.request_info.agent_depth == 3
        assert enriched.request_info.parent_correlation_id == "p"
        assert enriched.request_info.root_correlation_id == "root-1"

    def test_phase_identity_fields_propagate(self):
        ri = _make_request_info(
            phase_index=3,
            profiling_index=2,
            phase_name="storm",
            phase_kind="profiling",
        )
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        assert enriched.request_info.phase_index == 3
        assert enriched.request_info.profiling_index == 2
        assert enriched.request_info.phase_name == "storm"
        assert enriched.request_info.phase_kind == "profiling"

    def test_transport_extras_dropped(self):
        ri = _make_request_info()
        record = RequestRecord()
        enriched = InferenceClient._enrich_request_record(record, ri)
        # Downcast strips RequestInfo-only transport/payload-builder fields while
        # keeping the hoisted scalar inputs the records pipeline reads.
        dump = enriched.request_info.model_dump()
        assert "model_endpoint" not in dump
        assert "endpoint_headers" not in dump
        assert "drop_perf_ns" not in dump
        assert "turns" not in dump
        assert "system_message" not in dump
        assert "user_context_message" not in dump
        assert "payload_bytes" in dump
        assert "max_tokens" in dump
        assert "audio_duration_seconds" in dump
