# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_warmup converter (v1 UserConfig.loadgen -> warmup phase dict)."""

from __future__ import annotations

from aiperf.config.phases import PhaseType
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_warmup import build_warmup


def test_build_warmup_returns_none_when_no_warmup_fields() -> None:
    user = UserConfig.model_validate(
        {"loadgen": {"concurrency": 1, "request_count": 1}}
    )
    assert build_warmup(user) is None


def test_build_warmup_returns_none_when_loadgen_missing() -> None:
    user = UserConfig.model_validate({})
    assert build_warmup(user) is None


def test_build_warmup_with_request_count() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_concurrency": 10,
                "warmup_request_count": 50,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["exclude_from_results"] is True
    assert out["type"] == PhaseType.CONCURRENCY
    assert out["concurrency"] == 10
    assert out["requests"] == 50


def test_build_warmup_with_request_rate_uses_poisson() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_rate": 50.0,
                "warmup_request_count": 100,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["type"] == PhaseType.POISSON
    assert out["rate"] == 50.0


def test_build_warmup_with_duration() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_concurrency": 10,
                "warmup_duration": 30.0,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["duration"] == 30.0


def test_build_warmup_falls_back_to_profiling_concurrency() -> None:
    """If warmup_concurrency unset, fall back to profiling concurrency."""
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "concurrency": 50,
                "warmup_request_count": 25,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["concurrency"] == 50  # falls back from cli.concurrency


def test_build_warmup_with_num_sessions() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_num_sessions": 5,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["sessions"] == 5


def test_build_warmup_with_gamma_pattern() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_rate": 25.0,
                "warmup_arrival_pattern": "gamma",
                "arrival_smoothness": 1.5,
                "warmup_request_count": 100,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["type"] == PhaseType.GAMMA
    assert out["smoothness"] == 1.5


def test_build_warmup_with_constant_pattern() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_rate": 10.0,
                "warmup_arrival_pattern": "constant",
                "warmup_request_count": 50,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["type"] == PhaseType.CONSTANT


def test_build_warmup_with_ramps() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_count": 50,
                "warmup_concurrency_ramp_duration": 10.0,
                "warmup_prefill_concurrency_ramp_duration": 5.0,
                "warmup_request_rate_ramp_duration": 7.0,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["concurrency_ramp"] == {"duration": 10.0}
    assert out["prefill_ramp"] == {"duration": 5.0}
    assert out["rate_ramp"] == {"duration": 7.0}


def test_build_warmup_ramps_fallback_to_profiling() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_count": 50,
                "concurrency_ramp_duration": 8.0,
                "prefill_concurrency_ramp_duration": 4.0,
                "request_rate_ramp_duration": 6.0,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["concurrency_ramp"] == {"duration": 8.0}
    assert out["prefill_ramp"] == {"duration": 4.0}
    assert out["rate_ramp"] == {"duration": 6.0}


def test_build_warmup_with_prefill_concurrency_and_grace_period() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_count": 50,
                "warmup_prefill_concurrency": 4,
                "warmup_grace_period": 12.5,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["prefill_concurrency"] == 4
    assert out["grace_period"] == 12.5


def test_build_warmup_prefill_concurrency_falls_back_to_profiling() -> None:
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "warmup_request_count": 50,
                "prefill_concurrency": 7,
            }
        }
    )
    out = build_warmup(user)
    assert out is not None
    assert out["prefill_concurrency"] == 7
