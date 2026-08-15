# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase config helpers in ``aiperf.config.phases``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.phases import (
    BasePhaseConfig,
    ConcurrencyPhase,
    ConstantPhase,
    FixedSchedulePhase,
    GammaPhase,
    PoissonPhase,
    UserCentricPhase,
    get_phase_rate,
    resolve_graph_tstar_window,
)
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import PhaseType


class TestGetPhaseRate:
    """get_phase_rate is the single accessor for the phase ``rate`` field.

    It must return the configured rate for genuine RatePhaseConfig subclasses
    and None for every other phase type -- gated by isinstance, not by
    attribute probing, so a future field rename fails fast in one place.
    """

    @pytest.mark.parametrize(
        ("phase", "expected"),
        [
            param(
                ConstantPhase(
                    name="profiling", type=PhaseType.CONSTANT, rate=5.0, requests=10
                ),
                5.0,
                id="constant",
            ),
            param(
                PoissonPhase(
                    name="profiling", type=PhaseType.POISSON, rate=2.5, requests=10
                ),
                2.5,
                id="poisson",
            ),
            param(
                GammaPhase(
                    name="profiling", type=PhaseType.GAMMA, rate=1.5, requests=10
                ),
                1.5,
                id="gamma",
            ),
            param(
                UserCentricPhase(
                    name="profiling",
                    type=PhaseType.USER_CENTRIC,
                    rate=8.0,
                    users=2,
                    requests=10,
                ),
                8.0,
                id="user_centric",
            ),
        ],
    )  # fmt: skip
    def test_get_phase_rate_rate_phase_returns_rate(
        self, phase: BasePhaseConfig, expected: float
    ) -> None:
        assert get_phase_rate(phase) == expected

    @pytest.mark.parametrize(
        "phase",
        [
            param(
                ConcurrencyPhase(
                    name="profiling",
                    type=PhaseType.CONCURRENCY,
                    concurrency=4,
                    requests=10,
                ),
                id="concurrency",
            ),
            param(
                FixedSchedulePhase(name="profiling", type=PhaseType.FIXED_SCHEDULE),
                id="fixed_schedule",
            ),
        ],
    )  # fmt: skip
    def test_get_phase_rate_non_rate_phase_returns_none(
        self, phase: BasePhaseConfig
    ) -> None:
        assert get_phase_rate(phase) is None

    def test_get_phase_rate_stray_rate_attr_on_non_rate_phase_returns_none(
        self,
    ) -> None:
        """A rate-shaped attribute on a non-rate phase must not leak through.

        The pre-helper ``getattr(phase, "rate", None)`` probes would have
        returned it; the isinstance gate must not.
        """
        phase = ConcurrencyPhase(
            name="profiling", type=PhaseType.CONCURRENCY, concurrency=4, requests=10
        )
        object.__setattr__(phase, "rate", 7.5)
        assert get_phase_rate(phase) is None


class TestAdaptiveScalePhaseValidation:
    def test_adaptive_scale_enabled_rejects_non_boolean_values(self) -> None:
        with pytest.raises(ValidationError, match=r"adaptive_scale\.enabled"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=2,
                duration=10,
                adaptive_scale={
                    "enabled": 2,
                    "sustain_duration": 1,
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )

    def test_flat_adaptive_scale_rejects_non_boolean_values(self) -> None:
        with pytest.raises(ValidationError, match=r"adaptive_scale\.enabled"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=2,
                duration=10,
                adaptive_scale=1,
                adaptive_sustain_duration=1,
                sla=[
                    SLAFilter(
                        metric_tag="request_latency",
                        stat="p95",
                        op="le",
                        threshold=1000,
                    )
                ],
            )

    def test_adaptive_scale_accepts_camel_case_nested_block(self) -> None:
        phase = ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=20,
            duration=10,
            adaptiveScale={
                "enabled": True,
                "control": {"variable": "concurrency", "min": 2, "max": 20},
                "assessmentPeriod": 5,
                "sustainDuration": 1,
                "sla": {"request_latency": {"p95": {"le": 1000}}},
            },
        )

        assert phase.adaptive_scale is True
        assert phase.adaptive_control_variable == "concurrency"
        assert phase.adaptive_control_min == 2
        assert phase.adaptive_control_max == 20
        assert phase.adaptive_assessment_period == 5
        assert phase.adaptive_sustain_duration == 1
        assert phase.sla == [
            SLAFilter(
                metric_tag="request_latency",
                stat="p95",
                op="le",
                threshold=1000,
            )
        ]

    def test_adaptive_scale_field_takes_precedence_over_stale_alias(self) -> None:
        phase = ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=20,
            duration=10,
            adaptive_scale=False,
            adaptiveScale={
                "enabled": True,
                "control": {"variable": "concurrency", "min": 2, "max": 20},
                "sustainDuration": 1,
                "sla": {"request_latency": {"p95": {"le": 1000}}},
            },
        )

        assert phase.adaptive_scale is False

    @pytest.mark.parametrize(
        "adaptive_scale",
        [
            pytest.param({"enabled": True, "typo": 1}, id="top-level-unknown"),
            pytest.param(
                {"enabled": True, "control": {"variable": "concurrency", "typo": 1}},
                id="control-unknown",
            ),
            pytest.param(
                {"enabled": True, "strategy": {"type": "ramp_until_fail", "typo": 1}},
                id="strategy-unknown",
            ),
        ],
    )  # fmt: skip
    def test_nested_adaptive_scale_rejects_unknown_keys(
        self, adaptive_scale: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError, match="unsupported field"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=20,
                duration=10,
                adaptive_scale=adaptive_scale,
                adaptive_sustain_duration=1,
                sla=[
                    SLAFilter(
                        metric_tag="request_latency",
                        stat="p95",
                        op="le",
                        threshold=1000,
                    )
                ],
            )

    def test_fixed_schedule_accepts_disabled_adaptive_scale_block(self) -> None:
        phase = FixedSchedulePhase(
            name="profiling",
            type=PhaseType.FIXED_SCHEDULE,
            duration=10,
            requests=1,
            adaptive_scale={"enabled": False},
        )

        assert phase.adaptive_scale is False

    def test_fixed_schedule_rejects_adaptive_scale(self) -> None:
        with pytest.raises(ValidationError, match="fixed_schedule"):
            FixedSchedulePhase(
                name="profiling",
                type=PhaseType.FIXED_SCHEDULE,
                duration=10,
                requests=1,
                adaptive_scale={
                    "enabled": True,
                    "sustain_duration": 1,
                    "control": {"min": 1, "max": 2},
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )

    def test_request_rate_adaptive_control_rejects_rate_series(self) -> None:
        with pytest.raises(ValidationError, match="rate_series"):
            PoissonPhase(
                name="profiling",
                type=PhaseType.POISSON,
                duration=10,
                requests=10,
                rate_series={
                    "points": [
                        {"time_s": 0, "qps": 5},
                        {"time_s": 10, "qps": 15},
                    ]
                },
                adaptive_scale={
                    "enabled": True,
                    "sustain_duration": 1,
                    "control": {
                        "variable": "request_rate",
                        "min": 1,
                        "max": 20,
                    },
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )


class TestResolveGraphTstarWindow:
    """The agent-graph t* resolver: unset (None) means OFF, a value counts."""

    def _phase(
        self,
        *,
        trajectory_start_min_ratio: float | None = None,
        trajectory_start_max_ratio: float | None = None,
    ) -> ConcurrencyPhase:
        # Only forward the ratios the caller actually passed: an unset ratio has
        # different resolver semantics than one explicitly set to a value.
        ratios = {
            name: value
            for name, value in (
                ("trajectory_start_min_ratio", trajectory_start_min_ratio),
                ("trajectory_start_max_ratio", trajectory_start_max_ratio),
            )
            if value is not None
        }
        return ConcurrencyPhase(
            name="profiling", type=PhaseType.CONCURRENCY, concurrency=1, **ratios
        )

    def test_unset_is_none_and_resolves_to_closed_window(self) -> None:
        phase = self._phase()
        assert phase.trajectory_start_min_ratio is None
        assert phase.trajectory_start_max_ratio is None
        assert resolve_graph_tstar_window(phase) == (0.0, 0.0)

    def test_none_phase_resolves_to_closed_window(self) -> None:
        # A run with no profiling phase must not arm the window.
        assert resolve_graph_tstar_window(None) == (0.0, 0.0)

    @pytest.mark.parametrize(
        ("min_ratio", "max_ratio", "expected"),
        [
            param(0.5, None, (0.5, 0.5), id="min_only_is_a_point_window"),
            param(None, 0.7, (0.0, 0.7), id="max_only_starts_at_zero"),
            param(0.2, 0.8, (0.2, 0.8), id="both_authored"),
        ],
    )  # fmt: skip
    def test_half_set_pair_still_resolves_ordered(
        self,
        min_ratio: float | None,
        max_ratio: float | None,
        expected: tuple[float, float],
    ) -> None:
        """A half-set pair must not produce an inverted window.

        validate_trajectory_start_range deliberately skips pairs where either
        bound is None, so the resolver is the only thing standing between
        ``--trajectory-start-min-ratio 0.5`` alone and a (0.5, 0.0) window that
        crashes in AgentGraphConversationSource at dispatch time.
        """
        window = resolve_graph_tstar_window(
            self._phase(
                trajectory_start_min_ratio=min_ratio,
                trajectory_start_max_ratio=max_ratio,
            )
        )
        assert window == expected
        assert window[0] <= window[1]

    def test_explicit_value_resolves_to_itself(self) -> None:
        phase = self._phase(trajectory_start_max_ratio=0.9)
        assert resolve_graph_tstar_window(phase) == (0.0, 0.9)

    def test_assignment_after_construction_counts(self) -> None:
        # Assigning a ratio IS authoring the window: the value stops being None,
        # which is the whole signal -- no provenance flag to keep in sync.
        phase = self._phase()
        phase.trajectory_start_max_ratio = 1.0
        assert phase._trajectory_start_max_ratio_explicitly_set is False
        assert resolve_graph_tstar_window(phase) == (0.0, 1.0)

    @pytest.mark.parametrize(
        "authored",
        [
            param(None, id="never_authored"),
            param(0.9, id="authored"),
        ],
    )  # fmt: skip
    def test_unset_survives_dump_validate_round_trip(
        self, authored: float | None
    ) -> None:
        """The sweep orchestrator's round-trip must not invent a t* window.

        ``model_fields_set`` is per-instance and does not survive
        ``model_dump`` -> ``model_validate`` (every dumped key returns marked
        "set"), and the sweep writes ``run_config.json`` per cell for the
        subprocess to re-validate. ``None`` is a VALUE, so it round-trips: an
        unauthored phase stays unauthored in the cell instead of resolving to
        the AGENTIC_REPLAY full-trace default and silently engaging a chop.
        """
        phase = self._phase(trajectory_start_max_ratio=authored)
        expected = (0.0, authored if authored is not None else 0.0)
        assert resolve_graph_tstar_window(phase) == expected

        # Mirror local_executor._prepare_run_artifacts exactly.
        revalidated = ConcurrencyPhase.model_validate(
            phase.model_dump(mode="json", exclude_none=True)
        )
        assert resolve_graph_tstar_window(revalidated) == expected

    def test_explicit_zero_leaves_window_off(self) -> None:
        # A deliberate 0.0 must survive as 0.0 rather than being confused with
        # unset -- both close the window here, but only one is authored.
        phase = self._phase(
            trajectory_start_min_ratio=0.0, trajectory_start_max_ratio=0.0
        )
        assert phase.trajectory_start_max_ratio == 0.0
        assert resolve_graph_tstar_window(phase) == (0.0, 0.0)
