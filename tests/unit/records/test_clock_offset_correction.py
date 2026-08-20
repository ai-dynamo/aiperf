# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clock-offset correction is applied where worker time becomes controller time.

Measuring the offset and stamping it on a record is provenance, not correction.
These tests pin the two sites that actually convert: the metric-record metadata
anchor (from which every exported wall-clock timestamp is derived) and
``ParsedResponseRecord.timestamp_ns`` (which benchmark start/end and replay lag
fold across pods).
"""

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ParsedResponseRecord, RecordContext, RequestRecord


@pytest.mark.parametrize(
    "offset_ns, expected",
    [
        param(None, 1_000_000_500, id="unmeasured-passes-through"),
        param(0, 1_000_000_500, id="no-skew"),
        param(500, 1_000_000_000, id="worker-ahead-subtracts"),
        param(-500, 1_000_001_000, id="worker-behind-adds"),
    ],
)  # fmt: skip
def test_controller_timestamp_applies_the_offset(
    offset_ns: int | None, expected: int
) -> None:
    record = RequestRecord(timestamp_ns=1_000_000_500, clock_offset_ns=offset_ns)
    assert record.controller_timestamp_ns == expected
    # Raw provenance survives the correction.
    assert record.timestamp_ns == 1_000_000_500


def test_parsed_record_timestamp_is_corrected() -> None:
    """The value every cross-pod metric folds must already be controller-frame."""
    request = RequestRecord(timestamp_ns=2_000_000_000, clock_offset_ns=7_000_000)
    parsed = ParsedResponseRecord(request=request, responses=[])

    assert parsed.timestamp_ns == 1_993_000_000
    assert parsed.request.timestamp_ns == 2_000_000_000


def test_metric_metadata_anchor_is_corrected() -> None:
    """request_start/ack/end all derive from one corrected anchor."""
    from aiperf.records.record_processor_service import RecordProcessor

    record = RequestRecord(
        request_info=RecordContext(
            conversation_id="conv-1",
            turn_index=0,
            credit_num=1,
            x_request_id="req-7f2a",
            x_correlation_id="corr-1",
            credit_phase=CreditPhase.PROFILING,
        ),
        timestamp_ns=5_000_000_000,
        clock_offset_ns=3_000_000,
        start_perf_ns=1_000,
        end_perf_ns=1_500,
    )
    processor = object.__new__(RecordProcessor)
    processor.service_id = "record-processor-1"
    metadata = RecordProcessor._create_metric_record_metadata(
        processor, record, "worker-7f2a"
    )

    assert metadata.request_start_ns == 4_997_000_000
    # Perf deltas ride on the corrected anchor, so the whole timeline shifts
    # together rather than the endpoints drifting apart.
    assert metadata.request_end_ns == 4_997_000_500
