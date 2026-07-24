# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx, param

from aiperf.common.enums import MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.session_count_metric import SessionCountMetric
from aiperf.metrics.types.session_throughput_metric import SessionThroughputMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_records_data,
)

NANOS_PER_HOUR = 3_600_000_000_000


class TestSessionCountMetric:
    def test_counts_only_final_turn_root_records(self):
        records = [
            create_record(start_ns=100),
            create_record(start_ns=200),
            create_record(start_ns=300),
            create_record(start_ns=400),
        ]
        records[0].request.request_info.is_final_turn = True
        records[0].request.request_info.agent_depth = 0
        records[1].request.request_info.is_final_turn = False
        records[1].request.request_info.agent_depth = 0
        records[2].request.request_info.is_final_turn = True
        records[2].request.request_info.agent_depth = 1
        records[3].request.request_info.is_final_turn = True
        records[3].request.request_info.agent_depth = 0

        results = run_simple_metrics_pipeline(records, SessionCountMetric.tag)

        assert results[SessionCountMetric.tag] == 2

    def test_no_records(self):
        results = run_simple_metrics_pipeline([], SessionCountMetric.tag)
        assert SessionCountMetric.tag not in results


class TestSessionThroughputMetric:
    def test_derive_value_sessions_per_hour(self):
        results = MetricResultsDict()
        results[SessionCountMetric.tag] = 180
        results[BenchmarkDurationMetric.tag] = 2 * NANOS_PER_HOUR

        assert SessionThroughputMetric().derive_value(results) == approx(90.0)

    def test_derive_value_uses_explicit_window(self):
        results = MetricResultsDict()
        results[SessionCountMetric.tag] = 45
        results[BenchmarkDurationMetric.tag] = 10 * NANOS_PER_HOUR
        results.window_start_ns = 0
        results.window_end_ns = NANOS_PER_HOUR // 2

        assert SessionThroughputMetric().derive_value(results) == approx(90.0)

    @pytest.mark.parametrize(
        "duration",
        [
            param(0, id="zero"),
            param(None, id="none"),
        ],
    )  # fmt: skip
    def test_derive_value_invalid_duration_raises(self, duration):
        results = MetricResultsDict()
        results[SessionCountMetric.tag] = 10
        results[BenchmarkDurationMetric.tag] = duration
        with pytest.raises(NoMetricValue):
            SessionThroughputMetric().derive_value(results)

    def test_hour_unit_conversion(self):
        assert MetricTimeUnit.NANOSECONDS.convert_to(
            MetricTimeUnit.HOURS, NANOS_PER_HOUR
        ) == approx(1.0)

    @pytest.mark.asyncio
    async def test_column_store_accumulator_exports_count_and_rate(self, benchmark_run):
        accumulator = create_accumulator_with_metrics(
            benchmark_run,
            SessionCountMetric,
            MinRequestTimestampMetric,
            MaxResponseTimestampMetric,
            BenchmarkDurationMetric,
            SessionThroughputMetric,
        )
        accumulator._derive_funcs = {
            BenchmarkDurationMetric.tag: BenchmarkDurationMetric().derive_value,
            SessionThroughputMetric.tag: SessionThroughputMetric().derive_value,
        }

        for idx, session_value in enumerate((1, 0, 1)):
            start_ns = idx * (NANOS_PER_HOUR // 2)
            end_ns = start_ns + 1
            await accumulator.process_record(
                create_metric_records_data(
                    session_num=idx,
                    request_start_ns=start_ns,
                    request_end_ns=end_ns,
                    results=[
                        {
                            SessionCountMetric.tag: session_value,
                            MinRequestTimestampMetric.tag: start_ns,
                            MaxResponseTimestampMetric.tag: end_ns,
                        }
                    ],
                )
            )

        results = accumulator._compute_results()
        assert results[SessionCountMetric.tag].avg == 2
        assert results[SessionThroughputMetric.tag].avg == approx(2.0)
