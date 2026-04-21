# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the records envelopes that now carry msgspec
payloads (RequestRecord, MetricResult, ProfileResults, ProcessRecordsResult,
WorkerProcessingStats, PhaseRecordsStats) via PydanticStructMixin.

Covers the four envelopes in the records spec scope (R5):
- InferenceResultsMessage
- RealtimeMetricsMessage
- ProfileResultsMessage
- ProcessRecordsResultMessage

RecordsProcessingStatsMessage and AllRecordsReceivedMessage are exercised
by the credit-spec envelope tests since they share the same PhaseRecordsStats
payload shape.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.messages.inference_messages import (
    InferenceResultsMessage,
    RealtimeMetricsMessage,
)
from aiperf.common.messages.progress_messages import (
    ProcessRecordsResultMessage,
    ProfileResultsMessage,
)
from aiperf.common.models import (
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
    RequestRecord,
    TextResponse,
)


def _metric_result() -> MetricResult:
    return MetricResult(
        tag="request_latency",
        header="Request Latency",
        unit="ms",
        count=10,
        avg=1.5,
        p50=1.2,
        p99=3.4,
        min=0.5,
        max=4.1,
    )


def _profile_results() -> ProfileResults:
    return ProfileResults(
        completed=10,
        start_ns=100,
        end_ns=200,
        records=[_metric_result()],
    )


def _request_record() -> RequestRecord:
    return RequestRecord(
        model_name="test-model",
        timestamp_ns=10,
        start_perf_ns=11,
        end_perf_ns=25,
        status=200,
        responses=[
            TextResponse(perf_ns=16, text='{"result":"ok"}'),
        ],
    )


@pytest.mark.parametrize(
    "message_factory",
    [
        param(
            lambda: InferenceResultsMessage(
                service_id="worker-1", record=_request_record()
            ),
            id="InferenceResultsMessage",
        ),
        param(
            lambda: RealtimeMetricsMessage(
                service_id="records", metrics=[_metric_result()]
            ),
            id="RealtimeMetricsMessage",
        ),
        param(
            lambda: ProfileResultsMessage(
                service_id="records", profile_results=_profile_results()
            ),
            id="ProfileResultsMessage",
        ),
        param(
            lambda: ProcessRecordsResultMessage(
                service_id="records",
                results=ProcessRecordsResult(results=_profile_results()),
            ),
            id="ProcessRecordsResultMessage",
        ),
    ],
)
def test_records_envelope_roundtrips(message_factory) -> None:
    """Envelope with msgspec payload must round-trip through Pydantic JSON."""
    message = message_factory()

    payload = message.model_dump_json()
    decoded = type(message).model_validate_json(payload)

    assert decoded == message


def test_inference_results_discriminates_response_union_from_dict() -> None:
    """RequestRecord.responses dispatch via msgspec tagged-union on decode."""
    payload = {
        "service_id": "worker-1",
        "request_ns": 1,
        "message_type": "inference_results",
        "record": {
            "model_name": "test-model",
            "timestamp_ns": 10,
            "start_perf_ns": 11,
            "end_perf_ns": 25,
            "status": 200,
            "responses": [
                {
                    "response_type": "text",
                    "perf_ns": 16,
                    "text": '{"ok":true}',
                },
                {
                    "response_type": "sse",
                    "perf_ns": 17,
                    "packets": [{"name": "data", "value": "chunk-1"}],
                },
            ],
        },
    }

    msg = InferenceResultsMessage.model_validate(payload)

    from aiperf.common.models import SSEMessage
    from aiperf.common.models import TextResponse as _TR

    assert len(msg.record.responses) == 2
    assert isinstance(msg.record.responses[0], _TR)
    assert isinstance(msg.record.responses[1], SSEMessage)
    assert msg.record.responses[0].text == '{"ok":true}'
    assert msg.record.responses[1].packets[0].value == "chunk-1"
