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

from pathlib import Path
from typing import ClassVar

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
    def test_smoothness_without_explicit_pattern_auto_promotes_to_gamma(self):
        """v1 parity: --request-rate + --arrival-smoothness (or --vllm-burstiness)
        with NO explicit --arrival-pattern auto-promotes to gamma instead of
        falling through to poisson and being hard-rejected. The cutover dropped
        this auto-promote, making --vllm-burstiness unusable on its own."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_smoothness=1.5,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA
        assert prof["smoothness"] == 1.5
        assert prof["rate"] == 100.0

    def test_explicit_poisson_pattern_with_smoothness_raises(self):
        """An EXPLICIT non-gamma pattern + smoothness still errors clearly (the
        auto-promote only fires when the pattern was not user-supplied)."""
        loadgen = CLIConfig(
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.POISSON,
            arrival_smoothness=1.5,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="arrival-pattern gamma"):
            build_profiling(user)

    def test_smoothness_with_constant_pattern_raises(self) -> None:
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

    def test_smoothness_without_request_rate_raises(self) -> None:
        """Concurrency-mode (no rate) with --arrival-smoothness must error."""
        loadgen = CLIConfig(
            arrival_smoothness=1.5,
            concurrency=4,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="--arrival-smoothness"):
            build_profiling(user)

    def test_smoothness_with_gamma_succeeds(self) -> None:
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

    def test_gamma_without_smoothness_succeeds(self) -> None:
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
    def test_start_offset_without_fixed_schedule_raises(self) -> None:
        loadgen = CLIConfig(request_rate=100.0, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_start_offset=1000)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_end_offset_without_fixed_schedule_raises(self) -> None:
        loadgen = CLIConfig(request_rate=100.0, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_end_offset=2000)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_auto_offset_without_fixed_schedule_raises(self) -> None:
        loadgen = CLIConfig(concurrency=4, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_auto_offset=True)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(ValueError, match="--fixed-schedule"):
            build_profiling(user)

    def test_offsets_in_concurrency_mode_raises(self) -> None:
        loadgen = CLIConfig(concurrency=2, request_count=10)
        input_cfg = CLIConfig(fixed_schedule_start_offset=500)
        user = _make_user(loadgen=loadgen, input_cfg=input_cfg)
        with pytest.raises(
            ValueError, match=r"--fixed-schedule-\{auto,start,end\}-offset"
        ):
            build_profiling(user)

    def test_offsets_with_fixed_schedule_succeed(self) -> None:
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

    def test_fixed_schedule_without_offsets_succeeds(self) -> None:
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
    def test_grace_period_without_duration_raises(self) -> None:
        loadgen = CLIConfig(benchmark_grace_period=30, request_count=10, concurrency=1)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--benchmark-grace-period requires --benchmark-duration"
        ):
            build_profiling(user)

    def test_grace_period_with_duration_succeeds(self) -> None:
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
    def test_num_users_with_concurrency_mode_raises(self) -> None:
        loadgen = CLIConfig(num_users=5, request_count=10, concurrency=1)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--num-users requires --user-centric-rate"
        ):
            build_profiling(user)

    def test_num_users_with_request_rate_raises(self) -> None:
        loadgen = CLIConfig(num_users=5, request_rate=100.0, request_count=10)
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match="--num-users requires --user-centric-rate"
        ):
            build_profiling(user)

    def test_num_users_with_user_centric_succeeds(self) -> None:
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
    def test_rate_ramp_with_concurrency_mode_raises(self) -> None:
        loadgen = CLIConfig(
            request_rate_ramp_duration=30, request_count=10, concurrency=1
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(
            ValueError, match=r"--request-rate-ramp-duration.*rate-controlled"
        ):
            build_profiling(user)

    def test_rate_ramp_with_request_rate_succeeds(self) -> None:
        loadgen = CLIConfig(
            request_rate=100.0, request_rate_ramp_duration=30, request_count=10
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof.get("rate_ramp") == {"duration": 30}


# ---------------------------------------------------------------------------
# AGENTIC_REPLAY auto-warmup grace routing
# ---------------------------------------------------------------------------


class TestAgenticWarmupGracePeriodRouting:
    def test_agentic_warmup_grace_routes_onto_profiling_phase(self):
        """--agentic-warmup-grace-period is an AGENTIC_REPLAY route: it lands on
        the profiling phase dict (the agentic auto-warmup reads it from there),
        unlike --warmup-grace-period which feeds the user-declared warmup phase
        and requires --warmup-duration."""
        loadgen = CLIConfig(
            concurrency=8,
            request_count=10,
            agentic_warmup_grace_period=30.0,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["agentic_warmup_grace_period"] == 30.0

    def test_agentic_warmup_grace_absent_when_unset(self):
        """Unset --agentic-warmup-grace-period leaves the profiling phase dict
        without the key (so the warmup barrier defaults to infinite)."""
        loadgen = CLIConfig(concurrency=8, request_count=10)
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert "agentic_warmup_grace_period" not in prof

    def test_agentic_warmup_grace_does_not_require_duration(self):
        """Unlike grace_period (the profiling tail), the agentic warmup grace is
        not duration-gated -- it applies to a CONCURRENCY_BURST warmup with no
        duration, so it must route without a duration set."""
        loadgen = CLIConfig(
            concurrency=8,
            request_count=10,
            agentic_warmup_grace_period=0.0,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["agentic_warmup_grace_period"] == 0.0


class TestSystemIdleGapCapRouting:
    def test_system_idle_gap_cap_routes_onto_profiling_phase(self) -> None:
        loadgen = CLIConfig(
            concurrency=8,
            request_count=10,
            system_idle_gap_cap_seconds=10.0,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["system_idle_gap_cap_seconds"] == 10.0

    def test_system_idle_gap_cap_absent_when_unset(self) -> None:
        loadgen = CLIConfig(concurrency=8, request_count=10)
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert "system_idle_gap_cap_seconds" not in prof


class TestAdaptiveScaleCliRemoval:
    REMOVED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "adaptive_scale",
            "adaptive_sustain_duration",
            "adaptive_assessment_period",
            "adaptive_scale_control",
            "adaptive_control_variable",
            "adaptive_control_min",
            "adaptive_control_max",
            "adaptive_scale_sla",
        }
    )

    def test_removed_adaptive_scale_cli_fields_are_not_on_cli_config(self) -> None:
        assert self.REMOVED_FIELDS.isdisjoint(CLIConfig.model_fields)

    def test_removed_adaptive_scale_cli_fields_are_not_loadgen_routes(self) -> None:
        from aiperf.config.flags._section_fields import LOADGEN_FIELDS

        assert self.REMOVED_FIELDS.isdisjoint(LOADGEN_FIELDS)

    def test_build_profiling_does_not_emit_adaptive_scale_from_cli(self) -> None:
        user = _make_user(
            loadgen=CLIConfig(
                concurrency=8,
                benchmark_duration=60,
                request_count=100,
            )
        )

        prof = build_profiling(user)

        assert self.REMOVED_FIELDS.isdisjoint(prof)
        assert "adaptive_scale" not in prof


class TestAdaptiveScaleValidation:
    def test_adaptive_scale_rejects_concurrency_ramp(
        self: TestAdaptiveScaleValidation,
    ) -> None:
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

    def test_nested_adaptive_scale_yaml_lowers_to_flat_phase_fields(
        self: TestAdaptiveScaleValidation,
    ) -> None:
        from aiperf.config.phases import ConcurrencyPhase

        phase = ConcurrencyPhase.model_validate(
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 600,
                "concurrency": 200,
                "sla": {
                    "request_latency": {"p95": {"lt": 30000}},
                    "itl": {"p95": {"le": 100}},
                    "goodput": {"avg": {"ge": 20}},
                },
                "adaptive_scale": {
                    "enabled": True,
                    "min_concurrency": 2,
                    "max_concurrency": 200,
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
        assert phase.adaptive_control_min == 2
        assert phase.adaptive_control_max == 200
        assert phase.adaptive_assessment_period == 30
        assert phase.adaptive_min_completed_requests == 3
        assert phase.adaptive_sustain_duration == 120
        assert phase.adaptive_scale_strategy_type == "ramp_until_fail"
        assert phase.adaptive_scale_step_policy == "sla_margin"
        assert phase.adaptive_scale_base_step == 10
        assert phase.adaptive_scale_max_step_multiplier == 4
        assert [sla.metric_tag for sla in phase.sla] == [
            "request_latency",
            "itl",
            "goodput",
        ]

    @pytest.mark.parametrize(
        ("phase_data", "match"),
        [
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 10,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                },
                "adaptive_scale requires duration",
                id="missing-duration",
            ),
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                },
                "adaptive_scale requires adaptive_sustain_duration",
                id="missing-sustain",
            ),
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 10,
                },
                "adaptive_scale requires sla filters",
                id="missing-sla",
            ),
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 10,
                    "adaptive_control_min": 8,
                    "adaptive_control_max": 8,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                },
                "control.max must be > control.min",
                id="bad-bounds",
            ),
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 10,
                    "adaptive_control_min": 1.5,
                    "adaptive_control_max": 8,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                },
                "control.min must be an integer",
                id="non-integer-min",
            ),
            pytest.param(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": True,
                    "adaptive_sustain_duration": 10,
                    "adaptive_control_min": 9,
                    "adaptive_control_max": 10,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                },
                "control.min must be <= concurrency",
                id="min-exceeds-concurrency",
            ),
        ],
    )
    def test_adaptive_scale_validation_errors(
        self: TestAdaptiveScaleValidation, phase_data: dict, match: str
    ) -> None:
        from aiperf.config.phases import ConcurrencyPhase

        with pytest.raises(ValueError, match=match):
            ConcurrencyPhase.model_validate(phase_data)

    @pytest.mark.parametrize(
        ("block", "match"),
        [
            pytest.param(
                {"enabled": "maybe"}, "enabled must be a boolean", id="bad-enabled"
            ),
            pytest.param(
                {"control": "bad"}, "control must be a mapping", id="bad-control"
            ),
            pytest.param(
                {"strategy": "bad"}, "strategy must be a mapping", id="bad-strategy"
            ),
            pytest.param({"sla": "bad"}, "sla must be a mapping or list", id="bad-sla"),
        ],
    )
    def test_nested_adaptive_scale_rejects_invalid_blocks(
        self: TestAdaptiveScaleValidation, block: dict, match: str
    ) -> None:
        from aiperf.config.phases import ConcurrencyPhase

        with pytest.raises(ValueError, match=match):
            ConcurrencyPhase.model_validate(
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 8,
                    "adaptive_scale": block,
                    "adaptive_sustain_duration": 10,
                    "sla": [
                        {
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 100,
                        }
                    ],
                }
            )

    def test_nested_adaptive_scale_string_false_disables_phase(
        self: TestAdaptiveScaleValidation,
    ) -> None:
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


class TestRateSeries:
    def test_rate_series_without_request_rate_succeeds(self, tmp_path: Path) -> None:
        json_path = tmp_path / "rate.json"
        json_path.write_text(
            '{"points":[{"time_s":0,"qps":1},{"time_s":60,"qps":7},{"time_s":120,"qps":40}]}',
            encoding="utf-8",
        )
        loadgen = CLIConfig(
            request_rate_series=json_path,
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)

        prof = build_profiling(user)

        assert prof["type"] == PhaseType.CONSTANT
        assert "rate" not in prof
        assert prof["rate_series"]["points"][1] == {"time_s": 60.0, "qps": 7.0}

    def test_rate_series_with_request_rate_raises(self, tmp_path: Path) -> None:
        json_path = tmp_path / "rate.json"
        json_path.write_text(
            '{"points":[{"time_s":0,"qps":5},{"time_s":60,"qps":10}]}',
            encoding="utf-8",
        )
        loadgen = CLIConfig(
            request_rate=100.0,
            request_rate_series=json_path,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)

        with pytest.raises(ValueError, match=r"request-rate.*request-rate-series"):
            build_profiling(user)

    def test_rate_series_with_user_centric_rate_raises(self, tmp_path: Path) -> None:
        json_path = tmp_path / "rate.json"
        json_path.write_text(
            '{"points":[{"time_s":0,"qps":5},{"time_s":60,"qps":10}]}',
            encoding="utf-8",
        )
        loadgen = CLIConfig(
            user_centric_rate=100.0,
            request_rate_series=json_path,
            num_users=4,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)

        with pytest.raises(ValueError, match="user-centric-rate"):
            build_profiling(user)


# ---------------------------------------------------------------------------
# BUG 5 (NVBugs 6656707) — --search-space keywords must auto-infer phase shape
# ---------------------------------------------------------------------------


class TestSearchSpacePhaseShapeInference:
    def test_bare_rate_keyword_infers_poisson_phase(self) -> None:
        """--search-space 'rate:...' with no --request-rate must not crash;
        it should auto-switch to a rate-controlled (poisson default) phase."""
        loadgen = CLIConfig(
            search_space=["rate:1,100:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.POISSON
        assert "rate" not in prof  # value comes from the search planner, not here

    def test_dotted_rate_path_infers_poisson_phase(self) -> None:
        """Full dotted path form ('phases.profiling.rate') resolves the same
        as the bare alias."""
        loadgen = CLIConfig(
            search_space=["phases.profiling.rate:1,100:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.POISSON

    def test_rate_search_space_respects_explicit_arrival_pattern(self) -> None:
        """--arrival-pattern gamma + --search-space 'rate:...' should still
        pick GAMMA, not the POISSON default."""
        loadgen = CLIConfig(
            search_space=["rate:1,100:real"],
            arrival_pattern=ArrivalPattern.GAMMA,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA

    def test_rate_ramp_keyword_infers_rate_controlled_phase(self) -> None:
        loadgen = CLIConfig(
            search_space=["rate_ramp:1,60:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.POISSON

    def test_rate_series_keyword_infers_rate_controlled_phase(self) -> None:
        loadgen = CLIConfig(
            search_space=["rate_series:1,100:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.POISSON

    def test_smoothness_keyword_infers_gamma_phase(self) -> None:
        """--search-space 'smoothness:...' alone (no --request-rate, no
        --arrival-pattern) must auto-switch to gamma, not crash."""
        loadgen = CLIConfig(
            search_space=["smoothness:0.5,2.0:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA

    def test_users_keyword_infers_user_centric_phase(self) -> None:
        loadgen = CLIConfig(
            search_space=["users:1,50:int"],
            conversation_turn_mean=4,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.USER_CENTRIC

    def test_users_and_rate_together_raises_clear_conflict_error(self) -> None:
        """A benchmark has exactly one shape; searching 'users' and 'rate' at
        once is a genuine conflict, not something to silently resolve."""
        loadgen = CLIConfig(
            search_space=["users:1,50:int", "rate:1,100:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="only have one shape"):
            build_profiling(user)

    def test_users_and_smoothness_together_raises_clear_conflict_error(self) -> None:
        loadgen = CLIConfig(
            search_space=["users:1,50:int", "smoothness:0.5,2.0:real"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        with pytest.raises(ValueError, match="only have one shape"):
            build_profiling(user)

    def test_explicit_request_rate_still_wins_without_search_space(self) -> None:
        """Regression guard: explicit --request-rate path (no search-space)
        is unaffected by the new inference logic."""
        loadgen = CLIConfig(request_rate=50.0, request_count=10)
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.POISSON
        assert prof["rate"] == 50.0

    def test_concurrency_search_space_keyword_unaffected(self) -> None:
        """'concurrency' is already valid on every phase type incl. the
        default -- must keep resolving to PhaseType.CONCURRENCY."""
        loadgen = CLIConfig(
            search_space=["concurrency:1,1000:int"],
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.CONCURRENCY

    def test_smoothness_search_space_with_gamma_and_explicit_smoothness_flag(
        self,
    ) -> None:
        """--search-space 'smoothness:...' combined with an explicit
        --arrival-smoothness flag under gamma still succeeds normally (the
        two features are independent; this just proves no interaction bug)."""
        loadgen = CLIConfig(
            search_space=["smoothness:0.5,2.0:real"],
            arrival_pattern=ArrivalPattern.GAMMA,
            arrival_smoothness=1.0,
            request_rate=50.0,
            request_count=10,
        )
        user = _make_user(loadgen=loadgen)
        prof = build_profiling(user)
        assert prof["type"] == PhaseType.GAMMA
        assert prof["smoothness"] == 1.0
