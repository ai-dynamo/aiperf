# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.services.records_manager.metrics.types.inter_token_latency_metric import (
    InterTokenLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.output_token_count_metric import (
    OutputTokenCountMetric,
)
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


@pytest.fixture
def sample_metrics():
    req_latency = RequestLatencyMetric()
    ttft = TTFTMetric()
    output_tokens = OutputTokenCountMetric()

    req_latency.metric = [100, 200]
    ttft.metric = [40, 60]
    output_tokens.metric = [6, 3]

    return {
        "request_latency": req_latency,
        "ttft": ttft,
        "output_token_count": output_tokens,
    }


def test_inter_token_latency_metric_computes_correctly(sample_metrics):
    metric = InterTokenLatencyMetric()
    metric.update_value(metrics=sample_metrics)

    expected_1 = (100 - 40) / (6 - 1)  # 60 / 5 = 12
    expected_2 = (200 - 60) / (3 - 1)  # 140 / 2 = 70

    assert metric.values() == [expected_1, expected_2]
