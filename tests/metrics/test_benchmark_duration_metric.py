# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestBenchmarkDurationMetric:
    def test_benchmark_duration_calculation(self):
        """Test benchmark duration: max_response_time - min_request_time"""
        records = [
            create_record(start_ns=10, responses=[15]),  # min request is 10
            create_record(start_ns=20, responses=[25]),
            create_record(start_ns=30, responses=[40]),  # max response is 40
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            BenchmarkDurationMetric.tag,
        )

        # benchmark duration is 40 - 10 = 30
        assert metric_results[BenchmarkDurationMetric.tag] == 30

    def test_benchmark_duration_single_record(self):
        """Test benchmark duration with single record"""
        record = create_record(start_ns=100, responses=[150])
        metric_results = run_simple_metrics_pipeline(
            [record],
            BenchmarkDurationMetric.tag,
        )

        # benchmark duration is 150 - 100 = 50
        assert metric_results[BenchmarkDurationMetric.tag] == 50
