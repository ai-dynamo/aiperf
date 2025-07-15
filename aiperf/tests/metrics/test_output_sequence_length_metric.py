#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
)
from aiperf.services.records_manager.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)


def make_record(token_counts):
    responses = [
        ResponseData(
            perf_ns=0,
            raw_text=["test"],
            parsed_text=["test"],
            token_count=count,
            metadata={},
        )
        for count in token_counts
    ]
    request = RequestRecord(
        start_perf_ns=1, timestamp_ns=2, end_perf_ns=3, has_error=False
    )
    return ParsedResponseRecord(worker_id="w1", request=request, responses=responses)


def test_osl_metric_with_multiple_records():
    osl_metric = OutputSequenceLengthMetric()
    record1 = make_record([3, 5])
    osl_metric.update_value(record=record1)
    record2 = make_record([7])
    osl_metric.update_value(record=record2)
    assert osl_metric.values() == [8, 7]


def test_osl_metric_invalid_record():
    osl_metric = OutputSequenceLengthMetric()
    with pytest.raises(ValueError):
        osl_metric.update_value(record=None)
