# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-field validation tests for SteadyStateConfig manual-window overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.steady_state import SteadyStateConfig


class TestSteadyStateManualWindow:
    @pytest.mark.parametrize(
        "kwargs",
        [
            param({}, id="neither_set"),
            param({"start_pct": 10.0, "end_pct": 90.0}, id="both_set_valid"),
            param({"start_pct": 0.0, "end_pct": 100.0}, id="full_range"),
            param({"enabled": True}, id="enabled_auto_detection"),
        ],
    )  # fmt: skip
    def test_validate_manual_window_valid_combinations_pass(self, kwargs: dict) -> None:
        config = SteadyStateConfig(**kwargs)
        assert config.start_pct == kwargs.get("start_pct")
        assert config.end_pct == kwargs.get("end_pct")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            param(
                {"start_pct": 10.0},
                "without --steady-state-end-pct",
                id="start_only",
            ),
            param(
                {"end_pct": 90.0},
                "without --steady-state-start-pct",
                id="end_only",
            ),
        ],
    )  # fmt: skip
    def test_validate_manual_window_half_set_raises(
        self, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            SteadyStateConfig(**kwargs)

    @pytest.mark.parametrize(
        "start_pct, end_pct",
        [
            param(90.0, 10.0, id="inverted"),
            param(50.0, 50.0, id="empty_window"),
        ],
    )  # fmt: skip
    def test_validate_manual_window_non_positive_span_raises(
        self, start_pct: float, end_pct: float
    ) -> None:
        with pytest.raises(ValidationError, match="must be < end_pct"):
            SteadyStateConfig(start_pct=start_pct, end_pct=end_pct)
