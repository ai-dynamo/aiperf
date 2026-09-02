# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.accuracy.models import (
    AccuracyRecordsData,
    AccuracySummary,
    ProcessAccuracyResult,
    TaskAccuracyStats,
)
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import (
    Message,
    MetricRecordsData,
    ProcessAccuracyResultMessage,
    RecordsMessage,
)
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.models.trace_models import AioHttpTraceData


def _make_record(**overrides) -> AccuracyRecordsData:
    defaults = dict(
        session_num=0,
        worker_id="worker-1",
        benchmark_phase=CreditPhase.PROFILING,
        timestamp_ns=123456789,
        task="mmlu.astronomy",
        grader_name="exact_match",
        passed=True,
        unparsed=False,
        confidence=0.9,
        expected="B",
        actual="B",
        explanation="matched",
    )
    defaults.update(overrides)
    return AccuracyRecordsData(**defaults)


def test_accuracy_record_type_is_classvar_constant() -> None:
    a = _make_record(session_num=1)
    b = _make_record(session_num=2)
    assert AccuracyRecordsData.model_fields["record_type"].default == "accuracy"
    assert a.record_type == "accuracy"
    assert b.record_type == "accuracy"
    # record_type is now a SERIALIZED discriminator field (needed for wire
    # reconstruction on the generic RecordsMessage), so it IS dumped.
    assert a.model_dump()["record_type"] == "accuracy"


def test_accuracy_record_task_defaults_to_none() -> None:
    record = _make_record(task=None)
    assert record.task is None


def test_accuracy_record_round_trip_preserves_fields() -> None:
    record = _make_record()
    dumped = record.model_dump()
    rebuilt = AccuracyRecordsData(**dumped)
    assert rebuilt == record
    assert rebuilt.passed is True
    assert rebuilt.benchmark_phase == CreditPhase.PROFILING
    assert rebuilt.expected == "B"


def test_task_accuracy_stats_round_trip() -> None:
    stats = TaskAccuracyStats(
        total=4,
        passed=3,
        unparsed=1,
        accuracy_rate=0.75,
        unparsed_rate=0.25,
    )
    assert TaskAccuracyStats(**stats.model_dump()) == stats


def _summary() -> AccuracySummary:
    return AccuracySummary(
        total_evaluated=5,
        total_passed=3,
        accuracy_rate=0.6,
        overall_unparsed=1,
        grader_name="exact_match",
        per_task={
            "mmlu.b": TaskAccuracyStats(
                total=2, passed=2, unparsed=0, accuracy_rate=1.0, unparsed_rate=0.0
            ),
            "mmlu.a": TaskAccuracyStats(
                total=3, passed=1, unparsed=1, accuracy_rate=1 / 3, unparsed_rate=1 / 3
            ),
        },
    )


def test_process_accuracy_result_defaults_none() -> None:
    assert ProcessAccuracyResult().results is None
    wrapped = ProcessAccuracyResult(results=_summary())
    assert wrapped.results is not None
    assert wrapped.results.total_evaluated == 5


def test_process_accuracy_result_message_type() -> None:
    assert (
        ProcessAccuracyResultMessage.model_fields["message_type"].default
        == MessageType.PROCESS_ACCURACY_RESULT
    )


def test_records_message_serializes_accuracy_records() -> None:
    """Accuracy records ride the generic RecordsMessage envelope and serialize
    with their own record_type-carrying fields intact."""
    metadata = MetricRecordMetadata(
        session_num=0,
        request_start_ns=1_000,
        request_end_ns=2_000,
        worker_id="worker-1",
        record_processor_id="rp",
        benchmark_phase=CreditPhase.PROFILING,
    )
    message = RecordsMessage(
        service_id="records-manager",
        metadata=metadata,
        records=[_make_record(session_num=1), _make_record(session_num=2)],
    )
    assert message.message_type == MessageType.RECORDS
    dumped = message.model_dump()
    assert len(dumped["records"]) == 2
    assert dumped["records"][0]["session_num"] == 1
    assert dumped["records"][0]["record_type"] == "accuracy"


def test_records_message_wire_roundtrip_reconstructs_concrete_types() -> None:
    """The generic RecordsMessage must survive the ZMQ encode/decode boundary
    with each heterogeneous record rebuilt as its CONCRETE RecordData subclass.

    This is the regression guard for the serialization bug: a ``list[Any]``
    would decode the records back into bare dicts (no record_type routing), so
    the records manager would fail to dispatch them. Serializing via the real
    ``to_json_bytes`` / ``Message.from_json`` path proves the discriminator
    reconstruction works.
    """
    metadata = MetricRecordMetadata(
        session_num=7,
        request_start_ns=1_000,
        request_end_ns=2_000,
        worker_id="worker-1",
        record_processor_id="rp",
        benchmark_phase=CreditPhase.PROFILING,
    )
    trace = AioHttpTraceData(
        trace_type="aiohttp",
        reference_time_ns=1_700_000_000_000_000_000,
        reference_perf_ns=1_000_000_000,
        request_send_start_perf_ns=1_000_000_000,
    )
    metric_record = MetricRecordsData(
        metadata=metadata,
        metrics={"request_latency": 250_000_000},
        trace_data=trace,
    )
    accuracy_record = _make_record(session_num=7)

    message = RecordsMessage(
        service_id="record-processor",
        metadata=metadata,
        records=[metric_record, accuracy_record],
        error=None,
    )

    decoded = Message.from_json(message.to_json_bytes())

    assert isinstance(decoded, RecordsMessage)
    assert len(decoded.records) == 2

    round_metric, round_accuracy = decoded.records
    assert isinstance(round_metric, MetricRecordsData)
    assert round_metric.record_type == "metric_records"
    assert round_metric.metrics["request_latency"] == 250_000_000
    # trace_data must survive with its concrete discriminated subtype intact.
    assert isinstance(round_metric.trace_data, AioHttpTraceData)
    assert round_metric.trace_data.trace_type == "aiohttp"

    assert isinstance(round_accuracy, AccuracyRecordsData)
    assert round_accuracy.record_type == "accuracy"
    assert round_accuracy.session_num == 7


def test_process_accuracy_result_message_carries_summary() -> None:
    message = ProcessAccuracyResultMessage(
        service_id="records-manager",
        accuracy_result=ProcessAccuracyResult(results=_summary()),
    )
    assert message.message_type == MessageType.PROCESS_ACCURACY_RESULT
    assert message.accuracy_result.results.total_evaluated == 5
