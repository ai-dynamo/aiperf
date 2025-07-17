# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.services.records_manager.metrics.types.output_token_throughput_per_user_metric import (
    OutputTokenThroughputPerUserMetric,
)


@pytest.fixture
def mock_inter_token_latencies():
    class MockInterTokenLatency:
        tag = "inter_token_latency"

        def values(self):
            return [500_000_000, 250_000_000]

    return MockInterTokenLatency()


def test_add_multiple_records(mock_inter_token_latencies):
    metrics = {"inter_token_latency": mock_inter_token_latencies}
    output_token_throughput_per_user_metric = OutputTokenThroughputPerUserMetric()
    output_token_throughput_per_user_metric.update_value(record=None, metrics=metrics)
    assert output_token_throughput_per_user_metric.values() == [2.0, 4.0]
