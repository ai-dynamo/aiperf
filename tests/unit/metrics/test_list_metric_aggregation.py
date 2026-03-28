# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.metrics.list_metric_aggregation import (
    ExactListMetricAggregator,
    TDigestListMetricAggregator,
    build_list_metric_aggregator,
)
from aiperf.metrics.metric_dicts import MetricArray

SAMPLE_VALUES = [1.0, 2.0, 3.0, 10.0, 20.0, 21.0, 22.0, 50.0, 100.0]
ACCURACY_SAMPLE_VALUES = [float(value) for value in range(1, 101)]


@pytest.mark.parametrize(
    ("mode", "aggregator_type"),
    [
        pytest.param(ListMetricAggregationMode.EXACT, ExactListMetricAggregator),
        pytest.param(ListMetricAggregationMode.TDIGEST, TDigestListMetricAggregator),
    ],
)
def test_build_list_metric_aggregator_preserves_metric_result_shape(
    mode: ListMetricAggregationMode,
    aggregator_type: type[ExactListMetricAggregator | TDigestListMetricAggregator],
) -> None:
    """Build aggregators that emit the existing MetricResult summary shape."""
    aggregator = build_list_metric_aggregator(mode)

    aggregator.extend(SAMPLE_VALUES)
    result = aggregator.to_result("latency", "Latency", "ms")

    assert isinstance(aggregator, aggregator_type)
    assert result.tag == "latency"
    assert result.header == "Latency"
    assert result.unit == "ms"
    assert result.count == len(SAMPLE_VALUES)
    assert result.sum == pytest.approx(sum(SAMPLE_VALUES))
    assert result.min == min(SAMPLE_VALUES)
    assert result.max == max(SAMPLE_VALUES)
    assert result.avg is not None
    assert result.std is not None
    assert result.p1 is not None
    assert result.p5 is not None
    assert result.p10 is not None
    assert result.p25 is not None
    assert result.p50 is not None
    assert result.p75 is not None
    assert result.p90 is not None
    assert result.p95 is not None
    assert result.p99 is not None


def test_tdigest_percentiles_stay_close_to_exact_on_fixed_sample_set() -> None:
    """T-digest summaries should stay close to the exact percentile results."""
    exact = build_list_metric_aggregator(ListMetricAggregationMode.EXACT)
    tdigest = build_list_metric_aggregator(ListMetricAggregationMode.TDIGEST)

    exact.extend(ACCURACY_SAMPLE_VALUES)
    tdigest.extend(ACCURACY_SAMPLE_VALUES)

    exact_result = exact.to_result("latency", "Latency", "ms")
    tdigest_result = tdigest.to_result("latency", "Latency", "ms")

    assert tdigest_result.count == exact_result.count
    assert tdigest_result.sum == pytest.approx(exact_result.sum)
    assert tdigest_result.min == exact_result.min
    assert tdigest_result.max == exact_result.max
    assert tdigest_result.avg == pytest.approx(exact_result.avg)
    assert tdigest_result.std == pytest.approx(exact_result.std)
    assert tdigest_result.p1 == pytest.approx(exact_result.p1, abs=2.0)
    assert tdigest_result.p5 == pytest.approx(exact_result.p5, abs=2.0)
    assert tdigest_result.p10 == pytest.approx(exact_result.p10, abs=2.0)
    assert tdigest_result.p25 == pytest.approx(exact_result.p25, abs=2.0)
    assert tdigest_result.p50 == pytest.approx(exact_result.p50, abs=2.0)
    assert tdigest_result.p75 == pytest.approx(exact_result.p75, abs=2.0)
    assert tdigest_result.p90 == pytest.approx(exact_result.p90, abs=2.0)
    assert tdigest_result.p95 == pytest.approx(exact_result.p95, abs=2.0)
    assert tdigest_result.p99 == pytest.approx(exact_result.p99, abs=2.0)


@pytest.mark.parametrize(
    "aggregator",
    [
        pytest.param(ExactListMetricAggregator(), id="exact"),
        pytest.param(TDigestListMetricAggregator(), id="tdigest"),
    ],
)
def test_list_metric_aggregator_combines_append_and_extend_ingest_paths(
    aggregator: ExactListMetricAggregator | TDigestListMetricAggregator,
) -> None:
    """Aggregators should preserve summaries across mixed ingest calls."""
    aggregator.extend([1.0, 2.0])
    aggregator.append(3.0)
    aggregator.extend([4.0, 5.0])

    result = aggregator.to_result("latency", "Latency", "ms")

    assert result.count == 5
    assert result.sum == pytest.approx(15.0)
    assert result.min == 1.0
    assert result.max == 5.0
    assert result.avg == pytest.approx(3.0)


def test_exact_list_metric_aggregator_extend_uses_metric_array_bulk_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact aggregation should keep the bulk MetricArray.extend ingest path."""
    aggregator = ExactListMetricAggregator()
    values = [1.0, 2.0, 3.0]
    extend_calls: list[list[float]] = []
    append_calls: list[float] = []

    def record_extend(self: MetricArray, batch: list[float]) -> None:
        extend_calls.append(batch)

    def record_append(self: MetricArray, value: float) -> None:
        append_calls.append(value)

    monkeypatch.setattr(MetricArray, "extend", record_extend)
    monkeypatch.setattr(MetricArray, "append", record_append)

    aggregator.extend(values)

    assert extend_calls == [values]
    assert append_calls == []


@pytest.mark.parametrize(
    "aggregator",
    [
        pytest.param(ExactListMetricAggregator(), id="exact"),
        pytest.param(TDigestListMetricAggregator(), id="tdigest"),
    ],
)
def test_list_metric_aggregator_to_result_raises_consistent_error_when_empty(
    aggregator: ExactListMetricAggregator | TDigestListMetricAggregator,
) -> None:
    """Both implementations should reject empty summaries the same way."""
    with pytest.raises(
        IndexError,
        match="Cannot summarize an empty list metric aggregator",
    ):
        aggregator.to_result("latency", "Latency", "ms")
