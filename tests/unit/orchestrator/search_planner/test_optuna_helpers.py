# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pure Optuna helper functions that do not require optional deps."""

from __future__ import annotations

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import Objective
from aiperf.orchestrator.search_planner._optuna_helpers import derive_reference_point


def test_reference_point_uses_explicit_thresholds_when_set() -> None:
    objectives = [
        Objective(
            metric="a",
            direction=OptimizationDirection.MAXIMIZE,
            threshold=10.0,
        ),
        Objective(
            metric="b",
            direction=OptimizationDirection.MINIMIZE,
            threshold=200.0,
        ),
    ]
    rp = derive_reference_point(objectives, observed=[])
    assert rp == [10.0, 200.0]


def test_reference_point_auto_derives_from_observations_when_threshold_none() -> None:
    objectives = [
        Objective(metric="a", direction=OptimizationDirection.MAXIMIZE),
        Objective(metric="b", direction=OptimizationDirection.MINIMIZE),
    ]
    observed = [[10.0, 50.0], [20.0, 100.0], [15.0, 75.0]]
    rp = derive_reference_point(objectives, observed=observed)
    assert rp[0] == pytest.approx(9.5)
    assert rp[1] == pytest.approx(105.0)


def test_reference_point_mixed_explicit_and_auto() -> None:
    objectives = [
        Objective(
            metric="a",
            direction=OptimizationDirection.MAXIMIZE,
            threshold=5.0,
        ),
        Objective(metric="b", direction=OptimizationDirection.MINIMIZE),
    ]
    observed = [[10.0, 50.0], [20.0, 100.0]]
    rp = derive_reference_point(objectives, observed=observed)
    assert rp[0] == 5.0
    assert rp[1] == pytest.approx(105.0)


def test_reference_point_raises_when_no_threshold_and_no_observations() -> None:
    objectives = [
        Objective(metric="a", direction=OptimizationDirection.MAXIMIZE),
    ]
    with pytest.raises(ValueError, match="no threshold set"):
        derive_reference_point(objectives, observed=[])
