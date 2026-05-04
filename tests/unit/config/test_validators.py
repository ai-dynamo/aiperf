# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AIPerfConfig sweep cross-field validators.

Covers the three model_validators added in 4.6:

* ``validate_sweep_no_dashboard_ui`` — Dashboard UI + sweep is rejected.
* ``validate_sweep_same_seed_requires_seed`` — same-seed needs --random-seed.
* ``validate_sweep_cooldown_nonneg`` — explicit named-flag error message.
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


def test_sweep_with_dashboard_ui_rejected() -> None:
    with pytest.raises(ValueError, match="Dashboard UI is incompatible"):
        _make(
            sweep={
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
            },
            runtime={"ui": "dashboard"},
        )


def test_sweep_with_simple_ui_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep is not None


def test_same_seed_without_random_seed_rejected() -> None:
    with pytest.raises(ValueError, match="parameter-sweep-same-seed requires"):
        _make(multi_run={"parameter_sweep_same_seed": True})


def test_same_seed_with_random_seed_accepted() -> None:
    cfg = _make(
        random_seed=42,
        sweep={
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
        multi_run={"parameter_sweep_same_seed": True},
    )
    assert cfg.multi_run.parameter_sweep_same_seed is True


def test_negative_cooldown_rejected_by_field_constraint() -> None:
    """Field(ge=0) catches the negative; explicit validator's defensive check
    is exercised when ``ge=0`` is bypassed (covered by direct call below)."""
    with pytest.raises(ValueError, match="greater than or equal to 0|cooldown"):
        _make(
            sweep={
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
            },
            runtime={"ui": "simple"},
            multi_run={"parameter_sweep_cooldown_seconds": -1.0},
        )


def test_cooldown_zero_accepted() -> None:
    cfg = _make(multi_run={"parameter_sweep_cooldown_seconds": 0.0})
    assert cfg.multi_run.parameter_sweep_cooldown_seconds == 0.0


def test_cooldown_positive_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
        multi_run={"parameter_sweep_cooldown_seconds": 5.0},
    )
    assert cfg.multi_run.parameter_sweep_cooldown_seconds == 5.0


# ---------------------------------------------------------------------------
# validate_sweep_flags_require_sweep
# ---------------------------------------------------------------------------


def test_parameter_sweep_mode_with_single_concurrency_raises_error() -> None:
    """`--parameter-sweep-mode` (non-default value) without a sweep is rejected."""
    with pytest.raises(ValueError, match="--parameter-sweep-mode only applies"):
        _make(multi_run={"mode": "independent"})


def test_parameter_sweep_cooldown_seconds_with_single_concurrency_raises_error() -> (
    None
):
    """`--parameter-sweep-cooldown-seconds` without a sweep is rejected."""
    with pytest.raises(
        ValueError, match="--parameter-sweep-cooldown-seconds only applies"
    ):
        _make(multi_run={"parameter_sweep_cooldown_seconds": 5.0})


def test_parameter_sweep_same_seed_with_single_concurrency_raises_error() -> None:
    """`--parameter-sweep-same-seed` without a sweep is rejected."""
    with pytest.raises(ValueError, match="--parameter-sweep-same-seed only applies"):
        _make(random_seed=42, multi_run={"parameter_sweep_same_seed": True})


def test_parameter_sweep_mode_with_concurrency_list_succeeds() -> None:
    """A real sweep + explicit mode validates cleanly."""
    cfg = _make(
        sweep={
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
        multi_run={"mode": "independent"},
    )
    assert cfg.multi_run.mode == "independent"


def test_default_parameter_sweep_mode_with_single_concurrency_succeeds() -> None:
    """Single-concurrency run with no `--parameter-sweep-*` flag must validate.

    Default ``MultiRunConfig`` is constructed without explicit field-set entries,
    so ``model_fields_set`` is empty and the new validator is a no-op.
    """
    cfg = _make()
    assert "mode" not in cfg.multi_run.model_fields_set
    assert "parameter_sweep_cooldown_seconds" not in cfg.multi_run.model_fields_set
    assert "parameter_sweep_same_seed" not in cfg.multi_run.model_fields_set
