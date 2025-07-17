#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from aiperf.common.record_models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
)
from aiperf.services.records_manager.metrics.types.input_sequence_length_metric import (
    InputSequenceLengthMetric,
)


def make_record(input_token_count: int) -> ParsedResponseRecord:
    request = RequestRecord(
        request={},
        start_perf_ns=1,
        timestamp_ns=2,
        end_perf_ns=3,
    )
    response = ResponseData(
        perf_ns=2, raw_text=["test"], parsed_text=["test"], token_count=1, metadata={}
    )
    return ParsedResponseRecord(
        worker_id="worker",
        request=request,
        responses=[response],
        input_token_count=input_token_count,
    )


def test_isl_metric_with_multiple_records():
    isl = InputSequenceLengthMetric()
    record1 = make_record(5)
    record2 = make_record(7)

    isl.update_value(record1)
    isl.update_value(record2)

    assert isl.values() == [5, 7]
