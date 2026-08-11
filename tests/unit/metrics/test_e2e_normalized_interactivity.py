# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the E2E Normalized Interactivity metrics (AgentX Pareto x-axis).

The family is injected post-aggregation from the ``request_latency`` and
``output_sequence_length`` columns (see
``inject_e2e_normalized_interactivity_metrics``), so the tests build a
:class:`ColumnStore` and assert on the injected :class:`MetricResult` values.

A golden-parity test pins the injected value to InferenceX's exact derivation
(``1 / quantile(request_latency_s / OSL)`` with numpy-linear interpolation, from
``derived-agentic-metrics.ts``), so the port is provably equivalent for
identical inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.e2e_normalized_interactivity_analyzer import (
    inject_e2e_normalized_interactivity_metrics,
)
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.e2e_normalized_interactivity_metrics import (
    E2ENormalizedInteractivityP75Metric,
    E2ENormalizedInteractivityP90Metric,
)

P90 = E2ENormalizedInteractivityP90Metric.tag
P75 = E2ENormalizedInteractivityP75Metric.tag


def _infx_quantile(sorted_asc: list[float], q: float) -> float:
    """InferenceX's ``quantile`` (agentic-shared.ts): numpy-linear interpolation."""
    if not sorted_asc:
        return float("nan")
    if len(sorted_asc) == 1:
        return sorted_asc[0]
    pos = (len(sorted_asc) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return sorted_asc[lo]
    return sorted_asc[lo] + (sorted_asc[hi] - sorted_asc[lo]) * (pos - lo)


def _store(
    *,
    latency_s: list[float],
    osl: list[float],
    ttft_s: list[float] | None = None,
    isl: list[float] | None = None,
) -> ColumnStore:
    """ColumnStore of per-request columns as the accumulator ingests them.

    ``latency_s`` / ``ttft_s`` are given in seconds for readability and stored as
    nanoseconds (the metric's native column unit).
    """
    n = len(latency_s)
    ttft_s = ttft_s if ttft_s is not None else [0.5] * n
    isl = isl if isl is not None else [100.0] * n
    store = ColumnStore(initial_capacity=max(n, 1))
    for idx in range(n):
        store.ingest(
            idx=idx,
            record_metrics={
                "request_latency": latency_s[idx] * NANOS_PER_SECOND,
                "output_sequence_length": osl[idx],
                "time_to_first_token": ttft_s[idx] * NANOS_PER_SECOND,
                "input_sequence_length": isl[idx],
            },
            start_ns=float(idx),
            end_ns=float(idx),
            generation_start_ns=None,
        )
    return store


def _inject(store: ColumnStore, mask: np.ndarray | None = None) -> dict:
    results: dict = {}
    inject_e2e_normalized_interactivity_metrics(store, results, mask=mask)
    return results


def test_inject_matches_inverse_of_ratio_percentile() -> None:
    """p90/p75 == 1 / percentile(request_latency_s / OSL), the InferenceX way."""
    latency_s = [1.0, 2.0, 3.0, 4.0, 10.0]
    osl = [100.0] * 5
    results = _inject(_store(latency_s=latency_s, osl=osl))

    ratio = np.array(latency_s) / 100.0
    assert results[P90].avg == pytest.approx(1.0 / np.percentile(ratio, 90))
    assert results[P75].avg == pytest.approx(1.0 / np.percentile(ratio, 75))
    assert results[P90].count == 5
    assert results[P90].unit == "tokens/sec/user"


def test_golden_parity_with_inferencex_quantile() -> None:
    """Injected value is byte-for-byte InferenceX's ``1 / quantile(ratio)``."""
    latency_s = [0.7, 1.3, 2.9, 3.1, 5.5, 8.2, 0.4]
    osl = [50.0, 120.0, 200.0, 30.0, 512.0, 64.0, 8.0]
    results = _inject(_store(latency_s=latency_s, osl=osl))

    ratios = sorted(latency_s[i] / osl[i] for i in range(len(osl)))
    assert results[P90].avg == pytest.approx(1.0 / _infx_quantile(ratios, 0.90))
    assert results[P75].avg == pytest.approx(1.0 / _infx_quantile(ratios, 0.75))


def test_slow_tail_convention_is_not_percentile_of_rate() -> None:
    """Locks 1/p(ratio) != p(OSL/latency): the naive per-request-rate percentile
    would report the fast tail and must not be what we emit."""
    latency_s = [1.0, 2.0, 3.0, 4.0, 10.0]
    osl = [100.0] * 5
    results = _inject(_store(latency_s=latency_s, osl=osl))

    rate = np.array(osl) / np.array(latency_s)  # per-request OSL/latency
    naive_p90 = float(np.percentile(rate, 90))  # p(1/x): the WRONG convention
    # p90 slow-tail interactivity is the pessimistic (smaller) value.
    assert results[P90].avg < naive_p90


@pytest.mark.parametrize(
    "bad",
    [
        param({"latency_s": [1.0, 2.0, 0.0], "osl": [100.0, 100.0, 100.0]}, id="zero_latency"),
        param({"latency_s": [1.0, 2.0, 3.0], "osl": [100.0, 100.0, 0.0]}, id="zero_osl"),
    ],
)  # fmt: skip
def test_filter_drops_nonpositive_core_fields(bad: dict) -> None:
    """Requests with non-positive latency or OSL are excluded from the sample."""
    results = _inject(_store(**bad))
    # Third request dropped -> two survive.
    assert results[P90].count == 2


def test_filter_drops_nonpositive_ttft_and_isl() -> None:
    """TTFT<=0 or ISL<=0 excludes the request, matching InferenceX's filter."""
    store = _store(
        latency_s=[1.0, 2.0, 3.0],
        osl=[100.0, 100.0, 100.0],
        ttft_s=[0.5, 0.0, 0.5],  # 2nd dropped on TTFT
        isl=[100.0, 100.0, 0.0],  # 3rd dropped on ISL
    )
    results = _inject(store)
    assert results[P90].count == 1


def test_noop_when_required_columns_absent() -> None:
    """No request_latency / OSL columns -> nothing injected."""
    store = ColumnStore(initial_capacity=2)
    for idx in range(2):
        store.ingest(
            idx=idx,
            record_metrics={"time_to_first_token": 1.0},
            start_ns=float(idx),
            end_ns=float(idx),
            generation_start_ns=None,
        )
    results = _inject(store)
    assert P90 not in results and P75 not in results


def test_mask_restricts_sample() -> None:
    """Only masked-in requests contribute to the percentile."""
    store = _store(latency_s=[1.0, 2.0, 3.0, 100.0], osl=[100.0] * 4)
    mask = np.array([True, True, True, False])  # drop the 100s outlier
    results = _inject(store, mask=mask)
    assert results[P90].count == 3


def test_deferred_derive_raises() -> None:
    """The scalar summarize path defers; values come only from injection."""
    metric = E2ENormalizedInteractivityP90Metric()
    with pytest.raises(NoMetricValue):
        metric._derive_value(MetricResultsDict())


def test_extreme_values_never_inject_nonfinite() -> None:
    """Extreme latency/OSL must not inject nan/inf (NaN/Inf discipline): a tiny
    latency over a large OSL underflows the ratio so the reciprocal would
    overflow to inf; the guard must drop it rather than emit a non-finite value."""
    store = _store(latency_s=[1e-300, 1e-300, 1e-300], osl=[1e9, 1e9, 1e9])
    results = _inject(store)
    for tag in (P90, P75):
        assert tag not in results or math.isfinite(results[tag].avg)


def test_metric_metadata() -> None:
    """Percentile identity, unit, and larger-is-better flag are as specified."""
    assert E2ENormalizedInteractivityP90Metric.percentile == 90.0
    assert E2ENormalizedInteractivityP75Metric.percentile == 75.0
    assert E2ENormalizedInteractivityP90Metric.flags & MetricFlags.LARGER_IS_BETTER
