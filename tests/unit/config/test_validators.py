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
