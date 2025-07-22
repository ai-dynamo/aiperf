#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models.record_models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
)
from aiperf.services.records_manager.metrics.types.inter_token_latency_metric import (
    InterTokenLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


def sample_record(
    request_ns: int, end_ns: int, perf_ns: int, output_tokens: int
) -> ParsedResponseRecord:
    req = RequestRecord(start_perf_ns=request_ns, end_perf_ns=end_ns, has_error=False)
    resp = ResponseData(
        perf_ns=perf_ns,
        raw_text=["hi"],
        parsed_text=["hi"],
        token_count=output_tokens,
        metadata={},
    )
    return ParsedResponseRecord(
        worker_id="w1", request=req, responses=[resp], output_token_count=output_tokens
    )


def test_inter_token_latency_multiple_records():
    itl_metric = InterTokenLatencyMetric()
    req_latency = RequestLatencyMetric()
    ttft_metric = TTFTMetric()

    record1 = sample_record(
        0, 100, 40, output_tokens=6
    )  # latency = 100-0 = 100ns, ttft = 40ns
    record2 = sample_record(
        0, 200, 60, output_tokens=3
    )  # latency = 200-0 = 200ns, ttft = 60ns

    req_latency.metric = [100, 200]
    ttft_metric.metric = [40, 60]

    itl_metric.update_value(
        record=record1, metrics={"request_latency": req_latency, "ttft": ttft_metric}
    )

    itl_metric.update_value(
        record=record2, metrics={"request_latency": req_latency, "ttft": ttft_metric}
    )

    expected_1 = (100 - 40) / (6 - 1)  # = 60 / 5 = 12
    expected_2 = (200 - 60) / (3 - 1)  # = 140 / 2 = 70

    assert itl_metric.values() == [expected_1, expected_2]


def test_itl_with_token_count_le_one_raises():
    itl = InterTokenLatencyMetric()
    record = sample_record(0, 100, 50, output_tokens=1)
    with pytest.raises(ValueError, match="greater than 1"):
        itl.update_value(
            record=record,
            metrics={"request_latency": RequestLatencyMetric(), "ttft": TTFTMetric()},
        )
