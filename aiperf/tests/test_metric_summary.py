#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.benchmark_duration_metric import (
    BenchmarkDurationMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary


def test_metric_summary_process_and_get_metrics():
    ms = MetricSummary()

    # Prepare records
    records = [
        RequestRecord(
            start_perf_ns=1,
            responses=[SSEMessage(perf_ns=10), SSEMessage(perf_ns=20)],
        )
    ]

    ms.process(records)

    for tag, metric in ms.get_metrics_summary().items():
        if tag == TTFTMetric.tag:
            assert metric == [9]
        elif tag == BenchmarkDurationMetric.tag:
            assert metric == 19
