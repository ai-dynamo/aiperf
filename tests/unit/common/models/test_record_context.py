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

    def test_phase_identity_fields(self):
        ctx = _make_record_context(
            phase_index=2,
            profiling_index=1,
            phase_name="recovery",
            phase_kind="profiling",
        )

        assert ctx.phase_index == 2
        assert ctx.profiling_index == 1
        assert ctx.phase_name == "recovery"
        assert ctx.phase_kind == "profiling"


class TestRequestInfoIsRecordContext:
    def test_request_info_inherits_record_context(self):
        assert issubclass(RequestInfo, RecordContext)

    def test_request_info_has_transport_extras(self):
        ri_fields = set(RequestInfo.model_fields.keys())
        ctx_fields = set(RecordContext.model_fields.keys())
        extras = ri_fields - ctx_fields
        # ``turns``, ``system_message``, ``user_context_message`` live ONLY on
        # RequestInfo (worker-side) so the full Turn list never crosses the ZMQ
        # hop to the record processor — only the canonical ``payload_bytes`` (a
        # RecordContext field) travels. They are transport-only extras here.
        assert {"model_endpoint", "endpoint_headers", "drop_perf_ns"}.issubset(extras)
        assert "turns" in extras
        assert "system_message" in extras
        assert "user_context_message" in extras
        # The hoisted scalars stay on RecordContext (they cross the wire).
        assert "max_tokens" not in extras
        assert "audio_duration_seconds" not in extras
        assert "payload_bytes" not in extras
        # Phase identity fields also live on RecordContext.
        assert "phase_index" not in extras
        assert "profiling_index" not in extras
        assert "phase_name" not in extras
        assert "phase_kind" not in extras


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
