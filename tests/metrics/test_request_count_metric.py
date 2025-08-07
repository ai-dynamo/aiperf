# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.utils import close_enough
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestRequestCountMetric:
    def test_request_count_no_records(self):
        """Test that no metric is returned when no records are provided (as opposed to a default value or None)"""
        metric_results = run_simple_metrics_pipeline(
            [],
            RequestCountMetric.tag,
        )
        assert RequestCountMetric.tag not in metric_results

    @pytest.mark.parametrize("num_records", [1, 3, 10, 100, 1_000, 10_000])
    def test_request_count_multiple_records(self, num_records: int):
        """Test request count aggregation across multiple records"""
        records = [create_record(start_ns=100 * i) for i in range(num_records)]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestCountMetric.tag,
        )
        # num_records total records, each contributing 1 to the count
        assert close_enough(metric_results[RequestCountMetric.tag], num_records)
