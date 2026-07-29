# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AIPerfConfig sweep cross-field validators.

Post-redesign, the only envelope-level cross-field validator that remains
on AIPerfConfig is ``validate_sweep_no_dashboard_ui``; the ex-parameter
sweep field validators (same-seed-needs-seed, cooldown-non-neg,
flags-require-sweep) all moved when their fields moved off MultiRunConfig
onto SweepConfig sub-objects, where Pydantic's per-field constraints
(``ge=0``) and the discriminated SweepConfig union enforce them
structurally.
"""

from __future__ import annotations

import pytest

from aiperf.config.config import AIPerfConfig

_BASE_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


_ENVELOPE_KEYS = {"sweep", "multi_run", "variables", "random_seed"}


def _make(**overrides) -> AIPerfConfig:
    env_kwargs = {k: overrides.pop(k) for k in list(overrides) if k in _ENVELOPE_KEYS}
    body = {**_BASE_KWARGS, **overrides}
    return AIPerfConfig(benchmark=body, **env_kwargs)


# ---------------------------------------------------------------------------
# validate_sweep_no_dashboard_ui — only AIPerfConfig-scope validator left
# ---------------------------------------------------------------------------


def test_sweep_with_dashboard_ui_rejected() -> None:
    with pytest.raises(ValueError, match="Dashboard UI is incompatible"):
        _make(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [10, 20]},
            },
            runtime={"ui": "dashboard"},
        )


def test_sweep_with_simple_ui_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep is not None


# ---------------------------------------------------------------------------
# _reject_scenario_with_sweep — a fixed-spec scenario lock forbids sweeps.
# A scenario locks ONE configuration; a sweep would fan it into N diverging
# variations, each individually satisfying the lock (the falsification the v1
# list-shaped --concurrency rejection prevented). In v2 magic-list flags are
# hoisted to a sweep block before the config is built, so the rejection lives
# here at the envelope level.
# ---------------------------------------------------------------------------


def test_scenario_with_sweep_rejected() -> None:
    with pytest.raises(ValueError, match="does not support parameter sweeps"):
        _make(
            scenario="inferencex-agentx-mvp",
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
            },
            runtime={"ui": "simple"},
        )


def test_scenario_without_sweep_accepted() -> None:
    cfg = _make(scenario="inferencex-agentx-mvp")
    assert cfg.benchmark.scenario == "inferencex-agentx-mvp"
    assert cfg.sweep is None


def test_sweep_without_scenario_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
        },
        runtime={"ui": "simple"},
    )
    assert cfg.benchmark.scenario is None
    assert cfg.sweep is not None


def test_scenario_with_sweep_unsafe_override_warns_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        cfg = _make(
            scenario="inferencex-agentx-mvp",
            unsafe_override=True,
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
            },
            runtime={"ui": "simple"},
        )
    assert cfg.sweep is not None
    assert any(
        "does not support parameter sweeps" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


# ---------------------------------------------------------------------------
# Cooldown / same_seed / iteration_order moved to GridSweep — verified
# structurally there. AIPerfConfig no longer enforces these.
# ---------------------------------------------------------------------------


def test_grid_sweep_negative_cooldown_rejected_by_field_constraint() -> None:
    """``GridSweep.cooldown_seconds`` carries ``ge=0``; bare AIPerfConfig
    construction surfaces the Pydantic field error directly."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _make(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [10, 20]},
                "cooldown_seconds": -1.0,
            },
            runtime={"ui": "simple"},
        )


def test_grid_sweep_zero_cooldown_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "cooldown_seconds": 0.0,
        },
        runtime={"ui": "simple"},
    )
    assert isinstance(cfg.sweep, type(cfg.sweep))
    assert cfg.sweep.cooldown_seconds == 0.0


def test_grid_sweep_positive_cooldown_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "cooldown_seconds": 5.0,
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep.cooldown_seconds == 5.0


def test_grid_sweep_same_seed_field_round_trips() -> None:
    """``same_seed`` lives on GridSweep; envelope wiring is structural."""
    cfg = _make(
        random_seed=42,
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "same_seed": True,
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep.same_seed is True


# ---------------------------------------------------------------------------
# validate_agentic_cache_warmup — ``--agentic-cache-warmup-duration`` is
# consumed only on the AGENTIC_REPLAY path. On any other run the value is
# silently dropped, so the guard hard-raises rather than accept a no-op flag.
# A scenario governs the timing_mode (stamped post-construction by
# apply_scenario, which never re-runs this gate), so the validator resolves the
# scenario's declared timing_mode from its ScenarioSpec and rejects the flag
# when the scenario is not agentic_replay. A no-scenario config is final and
# resolved from the phases directly.
# ---------------------------------------------------------------------------


def _agentic_phase(**phase_overrides) -> list[dict]:
    return [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
            **phase_overrides,
        }
    ]


def test_agentic_cache_warmup_without_scenario_non_agentic_rejected() -> None:
    with pytest.raises(ValueError, match="requires the agentic_replay"):
        _make(phases=_agentic_phase(agentic_cache_warmup_duration=30.0))


def test_agentic_cache_warmup_with_agentic_scenario_accepted() -> None:
    """The agentic scenario locks AGENTIC_REPLAY, so the flag is accepted."""
    cfg = _make(
        scenario="inferencex-agentx-mvp",
        phases=_agentic_phase(agentic_cache_warmup_duration=30.0),
    )
    assert cfg.benchmark.scenario == "inferencex-agentx-mvp"


def test_agentic_cache_warmup_with_non_agentic_scenario_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-agentic scenario must reject the flag, not silently drop it.

    The validator resolves the scenario's declared timing_mode rather than
    deferring; ``apply_scenario`` stamps the mode post-construction and never
    re-runs this gate, so a blanket deferral would let a no-op flag through.
    """
    from aiperf.common.scenario import registry
    from aiperf.plugin.enums import TimingMode

    non_agentic = registry.SCENARIOS["inferencex-agentx-mvp"].model_copy(
        update={"name": "test-non-agentic", "timing_mode": TimingMode.REQUEST_RATE}
    )
    monkeypatch.setitem(registry.SCENARIOS, non_agentic.name, non_agentic)

    with pytest.raises(ValueError, match="requires the agentic_replay"):
        _make(
            scenario="test-non-agentic",
            phases=_agentic_phase(agentic_cache_warmup_duration=30.0),
        )


def test_agentic_cache_warmup_with_explicit_agentic_timing_mode_accepted() -> None:
    cfg = _make(
        phases=_agentic_phase(
            agentic_cache_warmup_duration=30.0,
            timing_mode="agentic_replay",
        ),
    )
    assert cfg.benchmark.phases[0].agentic_cache_warmup_duration == 30.0


def test_agentic_cache_warmup_request_budget_requires_duration() -> None:
    with pytest.raises(ValueError, match="requires.*duration"):
        _make(
            phases=_agentic_phase(
                agentic_cache_warmup_requests_per_lane=10,
                timing_mode="agentic_replay",
            )
        )


def test_agentic_cache_warmup_request_budget_with_duration_accepted() -> None:
    cfg = _make(
        phases=_agentic_phase(
            agentic_cache_warmup_duration=30.0,
            agentic_cache_warmup_requests_per_lane=10,
            timing_mode="agentic_replay",
        )
    )
    phase = cfg.benchmark.phases[0]
    assert phase.agentic_cache_warmup_requests_per_lane == 10


def test_no_agentic_cache_warmup_duration_accepted() -> None:
    cfg = _make(phases=_agentic_phase())
    assert cfg.benchmark.phases[0].agentic_cache_warmup_duration is None
