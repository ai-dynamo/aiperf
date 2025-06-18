#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.inter_token_latency_metric import (
    InterTokenLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.output_token_metric import (
    OutputTokenMetric,
)
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric
from aiperf.tests.utils.metric_test_utils import MockMetric


@pytest.fixture
def metrics():
    # Example: 3 records
    return {
        RequestLatencyMetric: MockMetric([100, 200, 300]),
        TTFTMetric.tag: MockMetric([10, 20, 30]),
        OutputTokenMetric.tag: MockMetric([9, 18, 27]),
    }


def test_inter_token_latency_basic(metrics):
    metric = InterTokenLatencyMetric()
    metric.update_value(metrics=metrics)
    expected = [
        (100 - 10) / 9,
        (200 - 20) / 18,
        (300 - 30) / 27,
    ]
    assert metric.values() == expected


def test_inter_token_latency_zero_tokens(metrics):
    metrics[OutputTokenMetric.tag] = MockMetric([9, 0, 27])
    metric = InterTokenLatencyMetric()
    metric.update_value(metrics=metrics)
    expected = [
        (100 - 10) / 9,
        0,
        (300 - 30) / 27,
    ]
    assert metric.values() == expected


def test_inter_token_latency_mismatched_lengths():
    metrics = {
        RequestLatencyMetric: MockMetric([100, 200]),
        TTFTMetric.tag: MockMetric([10]),
        OutputTokenMetric.tag: MockMetric([9, 18]),
    }
    metric = InterTokenLatencyMetric()
    with pytest.raises(ValueError):
        metric.update_value(metrics=metrics)
