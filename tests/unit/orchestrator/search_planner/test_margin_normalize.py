# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_margin_normalize`` (sigma-normalized multi-SLO aggregation)."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.orchestrator.search_planner._margin_normalize import normalize_margins


def test_no_sigmas_falls_back_to_raw_max() -> None:
    margins = {"ttft.p95": 50.0, "tpot.p99": 30.0}
    thresholds = {"ttft.p95": 200.0, "tpot.p99": 100.0}
    binding, key = normalize_margins(margins, sigmas=None, thresholds=thresholds)
    assert binding == 50.0
    assert key == "ttft.p95"


def test_empty_sigmas_falls_back_to_raw_max() -> None:
    margins = {"a": 5.0, "b": 10.0}
    binding, key = normalize_margins(
        margins, sigmas={}, thresholds={"a": 1.0, "b": 1.0}
    )
    assert binding == 10.0
    assert key == "b"


def test_with_sigmas_normalizes() -> None:
    # raw max says tpot binds (500 > 50); normalized max says ttft binds (50/5=10 vs 500/80~6.25).
    margins = {"ttft.p95": 50.0, "tpot.p99": 500.0}
    sigmas = {"ttft.p95": 5.0, "tpot.p99": 80.0}
    thresholds = {"ttft.p95": 200.0, "tpot.p99": 1000.0}
    binding, key = normalize_margins(margins, sigmas=sigmas, thresholds=thresholds)
    assert key == "ttft.p95"
    assert binding == pytest.approx(10.0, abs=1e-6)


def test_sigma_floor_prevents_blowup() -> None:
    # sigma=0 for one constraint would blow up to inf; floored at 0.01*|threshold|=2.0.
    margins = {"a": 50.0, "b": 10.0}
    sigmas = {"a": 0.0, "b": 5.0}
    thresholds = {"a": 200.0, "b": 100.0}
    binding, key = normalize_margins(margins, sigmas=sigmas, thresholds=thresholds)
    # a: 50 / max(0, 0.01*200) = 50/2 = 25; b: 10/5 = 2.
    assert key == "a"
    assert binding == pytest.approx(25.0, abs=1e-6)


@pytest.mark.parametrize(
    ("margins", "sigmas", "thresholds", "expected_key"),
    [
        param(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            None,
            {"a": 10.0, "b": 10.0, "c": 10.0},
            "c",
            id="raw_argmax_picks_largest",
        ),
        param(
            {"a": 100.0, "b": 1.0},
            {"a": 50.0, "b": 0.1},
            {"a": 100.0, "b": 10.0},
            "b",
            id="normalized_argmax_picks_tighter_snr",
        ),
    ],
)  # fmt: skip
def test_binding_key_is_argmax(
    margins: dict[str, float],
    sigmas: dict[str, float] | None,
    thresholds: dict[str, float],
    expected_key: str,
) -> None:
    _, key = normalize_margins(margins, sigmas=sigmas, thresholds=thresholds)
    assert key == expected_key


def test_empty_margins_raises() -> None:
    with pytest.raises(ValueError, match="at least one constraint"):
        normalize_margins({}, sigmas=None, thresholds={})


def test_zero_sigma_zero_threshold_uses_raw_margin() -> None:
    margins = {"a": 5.0, "b": 1.0}
    sigmas = {"a": 0.0, "b": 0.0}
    thresholds = {"a": 0.0, "b": 0.0}
    _, key = normalize_margins(margins, sigmas=sigmas, thresholds=thresholds)
    assert key == "a"
