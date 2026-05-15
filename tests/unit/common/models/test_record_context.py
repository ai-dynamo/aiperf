# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CreditPhase
from aiperf.common.models.record_models import (
    RecordContext,
    RequestInfo,
    RequestRecord,
)


def _make_record_context(**overrides) -> RecordContext:
    defaults = dict(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="c",
        turn_index=0,
        x_request_id="r",
        x_correlation_id="x",
    )
    defaults.update(overrides)
    return RecordContext(**defaults)


class TestRecordContext:
    def test_default_dag_fields(self):
        ctx = _make_record_context()
        assert ctx.agent_depth == 0
        assert ctx.parent_correlation_id is None
        assert ctx.payload_bytes is None
        assert ctx.max_tokens is None
        assert ctx.audio_duration_seconds is None

    def test_explicit_dag_fields(self):
        ctx = _make_record_context(
            agent_depth=3,
            parent_correlation_id="root",
        )
        assert ctx.agent_depth == 3
        assert ctx.parent_correlation_id == "root"


class TestRequestInfoIsRecordContext:
    def test_request_info_inherits_record_context(self):
        assert issubclass(RequestInfo, RecordContext)

    def test_request_info_has_transport_extras(self):
        ri_fields = set(RequestInfo.model_fields.keys())
        ctx_fields = set(RecordContext.model_fields.keys())
        extras = ri_fields - ctx_fields
        # ``turns``, ``system_message``, ``user_context_message`` were hoisted
        # onto RecordContext because the records pipeline reads them
        # post-transport. The remaining transport-only extras are the
        # endpoint/URL/timing fields.
        assert {"model_endpoint", "endpoint_headers", "drop_perf_ns"}.issubset(extras)
        assert "turns" not in extras
        assert "system_message" not in extras
        assert "user_context_message" not in extras


class TestRequestRecordHoldsRecordContext:
    def test_record_context_assignable_to_request_info_field(self):
        ctx = _make_record_context(agent_depth=2)
        rr = RequestRecord(request_info=ctx)
        assert rr.request_info is ctx
        assert rr.request_info.agent_depth == 2

    def test_request_info_subclass_assignable(self):
        ctx = _make_record_context()
        rr = RequestRecord(request_info=ctx)
        dumped = rr.model_dump()
        rebuilt = RequestRecord.model_validate(dumped)
        assert rebuilt.request_info is not None
        assert rebuilt.request_info.x_correlation_id == "x"
