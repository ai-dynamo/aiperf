# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.post_processors.strategies.metric_results import MetricResultsStrategy
from tests.unit.post_processors.conftest import create_metric_records_message


@dataclass
class _FakeHistogramInstrument:
    records: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def record(self, value: float, attributes: dict[str, Any]) -> None:
        self.records.append((value, attributes))


class _MetricStrategyContext:
    def __init__(
        self,
        *,
        attributes: dict[str, Any],
        coerced_values: dict[str, list[float]],
    ) -> None:
        self.attributes = attributes
        self.coerced_values = coerced_values
        self.histograms: dict[str, _FakeHistogramInstrument] = {}
        self.record_attributes_calls: list[Any] = []
        self.coerce_calls: list[tuple[str, Any]] = []
        self.get_histogram_calls: list[str] = []
        self._user_config: Any = None

    @property
    def user_config(self) -> Any:
        if self._user_config is None:
            from aiperf.common.config import EndpointConfig, UserConfig
            from aiperf.plugin.enums import EndpointType

            self._user_config = UserConfig(
                endpoint=EndpointConfig(
                    model_names=["test-model"],
                    type=EndpointType.CHAT,
                ),
            )
        return self._user_config

    async def get_or_create_histogram(
        self, metric_name: str, **kwargs: Any
    ) -> _FakeHistogramInstrument:
        self.get_histogram_calls.append(metric_name)
        return self.histograms.setdefault(metric_name, _FakeHistogramInstrument())

    def build_record_attributes(self, record: Any) -> dict[str, Any]:
        self.record_attributes_calls.append(record)
        return self.attributes

    def coerce_metric_values(self, metric_name: str, metric_value: Any) -> list[float]:
        self.coerce_calls.append((metric_name, metric_value))
        return self.coerced_values.get(metric_name, [])


class TestMetricResultsStrategy:
    def test_supports_only_metric_record_data(self) -> None:
        strategy = MetricResultsStrategy(
            _MetricStrategyContext(attributes={}, coerced_values={})
        )
        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 1}]
        ).to_data()
        timing_stats = CreditPhaseStats(phase=CreditPhase.PROFILING)

        assert strategy.supports(metric_record) is True
        assert strategy.supports(timing_stats) is False

    @pytest.mark.asyncio
    async def test_process_records_numeric_metric_values_into_histograms(self) -> None:
        context = _MetricStrategyContext(
            attributes={"aiperf.worker.id": "worker-1"},
            coerced_values={
                "request_latency": [123_000_000.0],
                "tokens_per_response": [1.0, 2.0, 3.0],
                "ignored_metric": [],
            },
        )
        strategy = MetricResultsStrategy(context)
        metric_record = create_metric_records_message(
            results=[
                {
                    "request_latency": 123_000_000,
                    "tokens_per_response": [1, 2, 3],
                    "ignored_metric": [],
                }
            ]
        ).to_data()

        await strategy.process(metric_record)

        assert context.record_attributes_calls == [metric_record]
        assert context.coerce_calls == [
            ("request_latency", 123_000_000),
            ("tokens_per_response", [1, 2, 3]),
            ("ignored_metric", []),
        ]
        # request_latency is translated to gen_ai.client.operation.duration by GenAI semconv
        assert "gen_ai.client.operation.duration" in context.get_histogram_calls
        assert "tokens_per_response" in context.get_histogram_calls
        assert len(context.get_histogram_calls) == 2
        # GenAI semconv histogram records use translated attributes, not raw aiperf ones
        assert "gen_ai.client.operation.duration" in context.histograms
        assert len(context.histograms["gen_ai.client.operation.duration"].records) == 1
        assert context.histograms["tokens_per_response"].records == [
            (1.0, {"aiperf.worker.id": "worker-1"}),
            (2.0, {"aiperf.worker.id": "worker-1"}),
            (3.0, {"aiperf.worker.id": "worker-1"}),
        ]
        assert "ignored_metric" not in context.histograms

    @pytest.mark.asyncio
    async def test_process_ignores_non_metric_record_inputs(self) -> None:
        context = _MetricStrategyContext(attributes={}, coerced_values={})
        strategy = MetricResultsStrategy(context)

        await strategy.process(CreditPhaseStats(phase=CreditPhase.PROFILING))

        assert context.record_attributes_calls == []
        assert context.coerce_calls == []
        assert context.get_histogram_calls == []
