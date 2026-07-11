# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for derived latency metrics computed from the column store."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.finite import is_finite_value
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.derived_latency import compute_credit_to_start_latency

_NS_PER_MS = 1_000_000.0


def _ingest_record(
    store: ColumnStore,
    idx: int,
    *,
    start_ns: float,
    credit_issued_ns: float,
    end_ns: float,
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
    )


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
