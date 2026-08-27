# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for derived latency metrics computed from the column store."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.finite import is_finite_value
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.derived_latency import (
    compute_adjusted_effective_latency,
    compute_credit_to_start_latency,
    compute_effective_latency,
    inject_derived_latency_metrics,
)

_NS_PER_MS = 1_000_000.0


def _ingest_record(
    store: ColumnStore,
    idx: int,
    *,
    start_ns: float,
    credit_issued_ns: float,
    end_ns: float,
    has_error: bool = False,
) -> None:
    """Write one record plus its ``credit_issued_ns`` metadata to ``store``."""
    store.ingest(
        idx,
        record_metrics={},
        start_ns=start_ns,
        end_ns=end_ns,
        generation_start_ns=None,
    )
    store.ingest_metadata(
        idx,
        metadata_numeric={"credit_issued_ns": credit_issued_ns},
        metadata_string={},
        metadata_bool={"has_error": has_error},
    )


# Two 10 ms successes and two 500 ms failures, all issued 1 ms before dispatch.
# The failures are 50x slower so any leak into the success distribution moves
# max/avg far outside the tolerance of an assertion on the clean values.
_T0 = 1_751_800_000_000_000_000
_CREDIT_LEAD_NS = 1_000_000
_SUCCESS_SPAN_NS = 10_000_000
_ERROR_SPAN_NS = 500_000_000


def _mixed_store(n_success: int = 2, n_error: int = 2) -> ColumnStore:
    """Store holding ``n_success`` fast successes and ``n_error`` slow failures."""
    store = ColumnStore(initial_capacity=8)
    idx = 0
    for kind, count, span in (
        ("ok", n_success, _SUCCESS_SPAN_NS),
        ("err", n_error, _ERROR_SPAN_NS),
    ):
        for _ in range(count):
            start = _T0 + idx * 1_000_000
            _ingest_record(
                store,
                idx,
                start_ns=start,
                credit_issued_ns=start - _CREDIT_LEAD_NS,
                end_ns=start + span,
                has_error=kind == "err",
            )
            idx += 1
    return store


@pytest.mark.parametrize(
    "start_ns,credit_issued_ns,expect_nonneg_only",
    [
        param(
            1_751_800_000_000_000_000,
            # +200 ns rounds to a higher float64 grid point (~256 ns ULP at
            # this magnitude) than start_ns, so the raw delta is genuinely
            # negative (-256 ns). The spec's +50 ns example rounds to the same
            # float64 value (delta 0.0) and would not exercise the clamp.
            1_751_800_000_000_000_200,
            True,
            id="negative-delta-clamped",
        ),
        param(
            1_751_800_000_000_000_000,
            1_751_800_000_000_000_000 - 5_000_000,
            False,
            id="positive-delta-preserved",
        ),
    ],
)  # fmt: skip
def test_compute_credit_to_start_latency_delta_clamping(
    start_ns: float, credit_issued_ns: float, expect_nonneg_only: bool
) -> None:
    """Negative raw deltas (credit_issued_ns > start_ns from ns quantization /
    clock skew) clamp to zero, while genuine positive deltas are preserved."""
    store = ColumnStore(initial_capacity=8)
    end_ns = start_ns + 10_000_000
    # Two records so percentile computation has more than a single point.
    _ingest_record(
        store, 0, start_ns=start_ns, credit_issued_ns=credit_issued_ns, end_ns=end_ns
    )
    _ingest_record(
        store,
        1,
        start_ns=start_ns + 1_000,
        credit_issued_ns=credit_issued_ns + 1_000,
        end_ns=end_ns + 1_000,
    )

    result = compute_credit_to_start_latency(store)
    assert result is not None

    for value in (
        result.min,
        result.avg,
        result.max,
        result.p1,
        result.p50,
        result.p99,
    ):
        assert is_finite_value(value)
        assert value >= 0.0

    if not expect_nonneg_only:
        # The ~5 ms positive delta must survive the clamp (np.maximum is a
        # no-op on positive values). float64 epoch-ns quantization (~256 ns
        # ULP at this magnitude) shifts the stored delta by tens of ns, so we
        # compare with a ns-scale absolute tolerance rather than exact equality.
        expected_ms = (start_ns - credit_issued_ns) / _NS_PER_MS
        assert result.min == pytest.approx(expected_ms, abs=1e-3)
        assert result.avg == pytest.approx(expected_ms, abs=1e-3)
        # Genuinely positive: not zeroed out by the clamp.
        assert result.min > 4.0


def test_compute_effective_latency_excludes_errored_records() -> None:
    """Failed requests must not contribute to the success-only distribution."""
    result = compute_effective_latency(_mixed_store())
    assert result is not None

    expected_ms = (_SUCCESS_SPAN_NS + _CREDIT_LEAD_NS) / _NS_PER_MS
    assert result.count == 2
    assert result.max == pytest.approx(expected_ms, abs=1e-3)
    assert result.avg == pytest.approx(expected_ms, abs=1e-3)
    assert result.p99 == pytest.approx(expected_ms, abs=1e-3)


def test_compute_effective_latency_all_errors_returns_none() -> None:
    """A run where every request failed leaves no success distribution to report."""
    assert compute_effective_latency(_mixed_store(n_success=0)) is None


def test_compute_effective_latency_honors_export_context_mask() -> None:
    """The error filter composes with the caller's window/phase mask."""
    import numpy as np

    store = _mixed_store(n_success=3, n_error=1)
    # Select the last two successes and the failure; the failure must still drop.
    mask = np.array([False, True, True, True], dtype=np.bool_)
    result = compute_effective_latency(store, mask=mask)
    assert result is not None
    assert result.count == 2


def test_compute_adjusted_effective_latency_inflates_with_infinity() -> None:
    """Each failure enters the adjusted view as one ``+inf`` sample."""
    result = compute_adjusted_effective_latency(_mixed_store(n_success=3, n_error=1))
    assert result is not None

    assert result.count == 4
    assert result.avg == float("inf")
    assert result.max == float("inf")
    assert result.p99 == float("inf")
    # 3 of 4 samples are finite, so the median stays on the success side.
    assert is_finite_value(result.p50)
    assert is_finite_value(result.min)
    # std is undefined in a distribution containing inf.
    assert result.std is None


def test_compute_adjusted_effective_latency_median_fails_over_at_half_errors() -> None:
    """At a 50% failure rate the median request itself failed, so p50 is inf."""
    result = compute_adjusted_effective_latency(_mixed_store(n_success=2, n_error=2))
    assert result is not None
    assert result.p50 == float("inf")
    assert is_finite_value(result.p25)


def test_compute_adjusted_effective_latency_returns_none_without_errors() -> None:
    """Nothing to inflate on a clean run — no adjusted variant is emitted."""
    assert compute_adjusted_effective_latency(_mixed_store(n_error=0)) is None


def test_compute_adjusted_effective_latency_returns_none_without_successes() -> None:
    """No success distribution to inflate when every request failed."""
    assert compute_adjusted_effective_latency(_mixed_store(n_success=0)) is None


def test_inject_derived_latency_metrics_emits_effective_pair() -> None:
    """A mixed run surfaces both the success-only and the failure-aware view."""
    results: dict = {}
    inject_derived_latency_metrics(_mixed_store(), results)

    assert results["effective_latency"].count == 2
    assert results["adj_effective_latency"].count == 4
    assert results["adj_effective_latency"].header == (
        "Effective Latency (CO-aware) (error-adjusted)"
    )
    assert results["adj_effective_latency"].unit == "ms"
