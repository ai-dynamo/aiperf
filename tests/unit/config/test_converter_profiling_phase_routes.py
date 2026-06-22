# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for phase-specific route gating in build_profiling.

Covers former bugs in ``aiperf.config.flags._converter_profiling``:

1. ``--arrival-smoothness`` outside ``--arrival-pattern gamma`` previously
   silently routed ``smoothness`` onto a non-Gamma phase config and crashed
   v2 ``PhaseConfig`` with ``extra_forbidden``. Should raise a clear error.
2. ``--fixed-schedule-{auto,start,end}-offset`` without ``--fixed-schedule``
   previously either silently dropped or crashed with ``extra_forbidden``.
   Should raise a clear error.
3. ``--benchmark-grace-period`` without ``--benchmark-duration`` previously
   silently dropped the user's flag. Should raise.
4. ``--num-users`` without ``--user-centric-rate`` and
   ``--request-rate-ramp-duration`` without ``--request-rate`` previously
   surfaced as generic Pydantic ``extra_forbidden`` errors.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import ArrivalPattern, PhaseType


def _make_user(
    *,
    loadgen: CLIConfig | None = None,
    input_cfg: CLIConfig | None = None,
) -> CLIConfig:
    endpoint = CLIConfig(url="http://localhost:8000/test", model_names=["test-model"])
    extra = loadgen.model_dump(exclude_unset=True) if loadgen is not None else {}
    inp_extra = (
        input_cfg.model_dump(exclude_unset=True) if input_cfg is not None else {}
    )
    return CLIConfig(**endpoint.model_dump(exclude_unset=True), **extra, **inp_extra)


# ---------------------------------------------------------------------------
# BUG 1 — --arrival-smoothness outside gamma must error
# ---------------------------------------------------------------------------


class TestArrivalSmoothnessGating:
    def test_smoothness_without_gamma_raises_clear_error(self):
        """--arrival-smoothness with default poisson pattern must error."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_smoothness=1.5,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="--arrival-smoothness"):
            build_profiling(user)

    def test_smoothness_with_constant_pattern_raises(self):
        """--arrival-smoothness with --arrival-pattern constant must error."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.CONSTANT,
            arrival_smoothness=2.0,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="arrival-pattern gamma"):
            build_profiling(user)

    def test_smoothness_without_request_rate_raises(self):
        """Concurrency-mode (no rate) with --arrival-smoothness must error."""
        loadgen = CLIConfig(
            arrival_smoothness=1.5,
            concurrency=4,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="--arrival-smoothness"):
            build_profiling(user)

    def test_smoothness_with_gamma_succeeds(self):
        """Valid combination: --arrival-pattern gamma + --arrival-smoothness."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.GAMMA,
            arrival_smoothness=1.5,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA
        assert prof["smoothness"] == 1.5
        assert prof["rate"] == 100.0

    def test_gamma_without_smoothness_succeeds(self):
        """--arrival-pattern gamma without --arrival-smoothness is allowed
        (smoothness is optional on GammaPhase)."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.GAMMA,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA
        assert "smoothness" not in prof


# ---------------------------------------------------------------------------
# BUG 2 — --fixed-schedule-*-offset without --fixed-schedule must error
# ---------------------------------------------------------------------------


class TestFixedScheduleOffsetGating:
    def test_start_offset_without_fixed_schedule_raises(self):
        loadgen = CLIConfig(request_rate=100.0, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_start_offset=1000)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_end_offset_without_fixed_schedule_raises(self):
        loadgen = CLIConfig(request_rate=100.0, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_end_offset=2000)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_auto_offset_without_fixed_schedule_raises(self):
        loadgen = CLIConfig(concurrency=4, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_auto_offset=True)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_offsets_in_concurrency_mode_raises(self):
        loadgen = CLIConfig(concurrency=2, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_start_offset=500)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(
            ValueError, match=r"--fixed-schedule-\{auto,start,end\}-offset"
        ):
            build_profiling(user)

    def test_offsets_with_fixed_schedule_succeed(self):
        """Valid combination: --fixed-schedule + offsets all together."""
        loadgen = CLIConfig(concurrency=4)
        input_cfg = CLIConfig(
            fixed_schedule=True,
            fixed_schedule_start_offset=100,
            fixed_schedule_end_offset=5000,
        )
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.FIXED_SCHEDULE
        assert prof["start_offset"] == 100
        assert prof["end_offset"] == 5000
        # Existing convention: start_offset present => auto_offset defaults False.
        assert prof["auto_offset"] is False

    def test_fixed_schedule_without_offsets_succeeds(self):
        """--fixed-schedule alone (no offsets) is fine."""
        loadgen = CLIConfig(concurrency=4)
        input_cfg = CLIConfig(fixed_schedule=True)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.FIXED_SCHEDULE
        assert "start_offset" not in prof
        assert "end_offset" not in prof


# ---------------------------------------------------------------------------
# BUG 3 — --benchmark-grace-period without --benchmark-duration
# ---------------------------------------------------------------------------


class TestGracePeriodRequiresDuration:
    def test_grace_period_without_duration_raises(self):
        loadgen = CLIConfig(benchmark_grace_period=30, request_count=10, concurrency=1)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--benchmark-grace-period requires --benchmark-duration"
        ):
            build_profiling(user)

    def test_grace_period_with_duration_succeeds(self):
        loadgen = CLIConfig(
            benchmark_duration=60.0, benchmark_grace_period=30, concurrency=1
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["duration"] == 60.0
        assert prof["grace_period"] == 30


# ---------------------------------------------------------------------------
# BUG 4a — --num-users without --user-centric-rate
# ---------------------------------------------------------------------------


class TestNumUsersRequiresUserCentric:
    def test_num_users_with_concurrency_mode_raises(self):
        loadgen = CLIConfig(num_users=5, request_count=10, concurrency=1)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--num-users requires --user-centric-rate"
        ):
            build_profiling(user)

    def test_num_users_with_request_rate_raises(self):
        loadgen = CLIConfig(num_users=5, request_rate=100.0, request_count=10)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--num-users requires --user-centric-rate"
        ):
            build_profiling(user)

    def test_num_users_with_user_centric_succeeds(self):
        """``--user-centric-rate`` resolves to USER_CENTRIC; --num-users flows through."""
        loadgen = CLIConfig(
            user_centric_rate=10.0,
            num_users=5,
            request_count=20,
            conversation_turn_mean=2,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.USER_CENTRIC
        assert prof["users"] == 5


# ---------------------------------------------------------------------------
# BUG 4b — --request-rate-ramp-duration without --request-rate
# ---------------------------------------------------------------------------


class TestRateRampRequiresRequestRate:
    def test_rate_ramp_with_concurrency_mode_raises(self):
        loadgen = CLIConfig(
            request_rate_ramp_duration=30, request_count=10, concurrency=1
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--request-rate-ramp-duration.*rate-controlled"
        ):
            build_profiling(user)

    def test_rate_ramp_with_request_rate_succeeds(self):
        loadgen = CLIConfig(
            request_rate=100.0, request_rate_ramp_duration=30, request_count=10
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof.get("rate_ramp") == {"duration": 30}


class TestAdaptiveScaleRoutes:
    def test_adaptive_scale_cli_fields_route_to_profiling_phase(self):
        loadgen = CLIConfig(
            adaptive_scale=True,
            adaptive_sustain_duration=120.0,
            adaptive_assessment_period=30.0,
            adaptive_scale_sla=["request_latency:p95:le:30000"],
            benchmark_duration=600.0,
            concurrency=200,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)

        assert prof["type"] == PhaseType.CONCURRENCY
        assert prof["adaptive_scale"] is True
        assert prof["adaptive_sustain_duration"] == 120.0
        assert prof["adaptive_assessment_period"] == 30.0
        assert prof["sla"] == [
            {
                "metric_tag": "request_latency",
                "stat": "p95",
                "op": "le",
                "threshold": 30000.0,
            }
        ]

    def test_adaptive_scale_requires_concurrency(self):
        loadgen = CLIConfig(
            adaptive_scale=True,
            adaptive_sustain_duration=120.0,
            benchmark_duration=600.0,
            request_rate=10.0,
        )
        user = _make_user(loadgen=loadgen)

        with pytest.raises(ValueError, match="--adaptive-scale requires --concurrency"):
            build_profiling(user)

    def test_adaptive_scale_rejects_search_sla(self):
        loadgen = CLIConfig(
            adaptive_scale=True,
            adaptive_sustain_duration=120.0,
            benchmark_duration=600.0,
            concurrency=200,
            search_sla=["request_latency:p95:le:30000"],
        )
        user = _make_user(loadgen=loadgen)

        with pytest.raises(ValueError, match="--adaptive-scale-sla"):
            build_profiling(user)

    def test_adaptive_scale_rejects_concurrency_ramp(self):
        from aiperf.config.phases import ConcurrencyPhase

        with pytest.raises(
            ValueError, match="adaptive_scale cannot be combined with concurrency_ramp"
        ):
            ConcurrencyPhase.model_validate(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 600,
                    "concurrency": 200,
                    "concurrency_ramp": 30,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 120,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 30000,
                        }
                    ],
                }
            )

    def test_nested_adaptive_scale_yaml_lowers_to_flat_phase_fields(self):
        from aiperf.config.phases import ConcurrencyPhase

        phase = ConcurrencyPhase.model_validate(
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 600,
                "concurrency": 200,
                "sla": {"request_latency": {"p95": {"lt": 30000}}},
                "adaptive_scale": {
                    "enabled": True,
                    "min_concurrency": 2,
                    "window": 30,
                    "minCompletedRequests": 3,
                    "sustain_duration": 120,
                    "strategy": {
                        "type": "ramp_until_fail",
                        "step_policy": "sla_margin",
                        "base_step": 10,
                        "max_step_multiplier": 4,
                    },
                },
            }
        )

        assert phase.adaptive_scale is True
        assert phase.adaptive_scale_min_concurrency == 2
        assert phase.adaptive_assessment_period == 30
        assert phase.adaptive_min_completed_requests == 3
        assert phase.adaptive_sustain_duration == 120
        assert phase.adaptive_scale_strategy_type == "ramp_until_fail"
        assert phase.adaptive_scale_step_policy == "sla_margin"
        assert phase.adaptive_scale_base_step == 10
        assert phase.adaptive_scale_max_step_multiplier == 4

    def test_nested_adaptive_scale_string_false_disables_phase(self):
        from aiperf.config.phases import ConcurrencyPhase

        phase = ConcurrencyPhase.model_validate(
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 600,
                "concurrency": 200,
                "adaptive_scale": {"enabled": "false"},
            }
        )

        assert phase.adaptive_scale is False
