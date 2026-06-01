# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx, param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.session_count_metric import SessionCountMetric
from aiperf.metrics.types.session_throughput_metric import SessionThroughputMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.metrics.conftest import run_simple_metrics_pipeline

NANOS_PER_HOUR = 3_600_000_000_000


def _record(
    *, start_ns: int = 100, is_final_turn: bool = True, agent_depth: int = 0
) -> ParsedResponseRecord:
    """Build a valid single-response record with controllable session fields."""
    request = RequestRecord(
        request_info=RequestInfo(
            model_endpoint=ModelEndpointInfo(
                models=ModelListInfo(
                    models=[ModelInfo(name="test-model")],
                    model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                ),
                endpoint=EndpointInfo(
                    type=EndpointType.CHAT,
                    base_url="http://localhost:8000/v1/test",
                ),
            ),
            turns=[],
            turn_index=0,
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-request-id",
            x_correlation_id="test-correlation-id",
            conversation_id="test-conversation",
            is_final_turn=is_final_turn,
            agent_depth=agent_depth,
        ),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 50,
    )
    return ParsedResponseRecord(
        request=request,
        responses=[
            ParsedResponse(perf_ns=start_ns + 50, data=TextResponseData(text="test"))
        ],
        token_counts=TokenCounts(input=10, output=1, reasoning=None),
    )


class TestSessionCountMetric:
    def test_counts_only_final_turn_root_records(self):
        """Only final-turn records of root sessions (agent_depth == 0) count."""
        records = [
            _record(start_ns=100, is_final_turn=True, agent_depth=0),  # counted
            _record(start_ns=200, is_final_turn=False, agent_depth=0),  # mid-turn
            _record(start_ns=300, is_final_turn=True, agent_depth=1),  # DAG child
            _record(start_ns=400, is_final_turn=True, agent_depth=0),  # counted
        ]
        metric_results = run_simple_metrics_pipeline(records, SessionCountMetric.tag)
        assert metric_results[SessionCountMetric.tag] == 2

    def test_no_records(self):
        """No metric is returned when no records are provided."""
        metric_results = run_simple_metrics_pipeline([], SessionCountMetric.tag)
        assert SessionCountMetric.tag not in metric_results


class TestSessionThroughputMetric:
    def test_derive_value_sessions_per_hour(self):
        metric = SessionThroughputMetric()
        metric_results = MetricResultsDict()
        metric_results[SessionCountMetric.tag] = 180  # sessions
        metric_results[BenchmarkDurationMetric.tag] = 2 * NANOS_PER_HOUR  # 2 hours
        # 180 sessions / 2 hours = 90 sessions/hour
        assert metric.derive_value(metric_results) == approx(90.0)

    @pytest.mark.parametrize(
        "duration",
        [
            param(0, id="zero"),
            param(None, id="none"),
        ],
    )  # fmt: skip
    def test_derive_value_invalid_duration_raises(self, duration):
        metric = SessionThroughputMetric()
        metric_results = MetricResultsDict()
        metric_results[SessionCountMetric.tag] = 10
        metric_results[BenchmarkDurationMetric.tag] = duration
        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)
