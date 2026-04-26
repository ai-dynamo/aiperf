# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_profiling converter (v1 UserConfig -> profiling phase dict)."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.phases import PhaseType
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_profiling import build_profiling


@pytest.mark.parametrize(
    "loadgen,input_cfg,expected_type",
    [
        param(
            {"concurrency": 100, "request_count": 1000},
            {},
            PhaseType.CONCURRENCY,
            id="concurrency",
        ),
        param(
            {"request_rate": 100.0, "request_count": 1000},
            {},
            PhaseType.POISSON,
            id="request-rate-default-poisson",
        ),
        param(
            {
                "request_rate": 100.0,
                "arrival_pattern": "gamma",
                "arrival_smoothness": 2.0,
                "request_count": 1000,
            },
            {},
            PhaseType.GAMMA,
            id="request-rate-gamma",
        ),
        param(
            {
                "request_rate": 100.0,
                "arrival_pattern": "constant",
                "request_count": 1000,
            },
            {},
            PhaseType.CONSTANT,
            id="request-rate-constant",
        ),
        param(
            {"user_centric_rate": 5.0, "num_users": 50, "request_count": 1000},
            {"conversation": {"turn": {"mean": 3}}},
            PhaseType.USER_CENTRIC,
            id="user-centric",
        ),
        param(
            {},
            {"fixed_schedule": True, "fixed_schedule_auto_offset": True},
            PhaseType.FIXED_SCHEDULE,
            id="fixed-schedule",
        ),
    ],
)  # fmt: skip
def test_build_profiling_picks_phase_type(
    loadgen: dict, input_cfg: dict, expected_type: PhaseType
) -> None:
    payload: dict = {}
    if loadgen:
        payload["loadgen"] = loadgen
    if input_cfg:
        payload["input"] = input_cfg
    user = UserConfig.model_validate(payload)
    out = build_profiling(user)
    assert out["type"] == expected_type


def test_build_profiling_includes_ramp() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "concurrency": 100,
                "request_count": 1000,
                "concurrency_ramp_duration": 30.0,
            }
        }
    )
    out = build_profiling(user)
    assert out["concurrency_ramp"] == {"duration": 30.0}


def test_build_profiling_request_cancellation() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "concurrency": 100,
                "request_count": 1000,
                "request_cancellation_rate": 0.1,
                "request_cancellation_delay": 5.0,
            }
        }
    )
    out = build_profiling(user)
    assert out["cancellation"] == {"rate": 0.1, "delay": 5.0}


def test_build_profiling_defaults_request_count_when_missing() -> None:
    user = UserConfig.model_validate({"loadgen": {"concurrency": 100}})
    out = build_profiling(user)
    assert out.get("requests") == 10  # default added by _validate_profiling


def test_build_profiling_user_centric_requires_session_turns_ge_2() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "user_centric_rate": 5.0,
                "num_users": 50,
                "request_count": 1000,
            },
            "input": {"conversation": {"turn": {"mean": 1}}},
        }
    )
    with pytest.raises(ValueError, match="session-turns-mean"):
        build_profiling(user)
