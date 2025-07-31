# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import ParsedResponseRecord, RequestRecord, ResponseData
from aiperf.metrics.types import (
    OutputTokenCountMetric,
)


@pytest.fixture
def sample_record():
    return ParsedResponseRecord(
        worker_id="worker-1",
        request=RequestRecord(
            conversation_id="c1",
            turn_index=0,
            model_name="model",
            start_perf_ns=0,
            timestamp_ns=0,
        ),
        responses=[
            ResponseData(
                perf_ns=100,
                raw_text=["hello"],
                parsed_text=["hello"],
                token_count=5,
            )
        ],
        output_token_count=5,
    )


def test_output_token_count_metric(sample_record):
    metric = OutputTokenCountMetric()
    metric.update_value(record=sample_record)
    assert metric.values() == [5]
