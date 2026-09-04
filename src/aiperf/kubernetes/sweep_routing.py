# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing helpers for Kubernetes parameter-sweep and multi-run workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig
    from aiperf.config.sweep.multi_run import MultiRunConfig


def requires_sweep_controller(config: AIPerfConfig) -> bool:
    """Return whether a Config-v2 envelope needs cluster-side orchestration."""
    return config.sweep is not None or requires_multiple_trials(config.multi_run)


def requires_multiple_trials(multi_run: MultiRunConfig) -> bool:
    """Return whether multi-run settings request repeats or convergence."""
    return multi_run.num_runs > 1 or multi_run.convergence is not None


def one_cell_sweep(*, convergence: bool) -> dict[str, Any]:
    """Return a no-op scenario sweep that preserves one base variation.

    ``AIPerfSweepSpec`` requires a sweep block even when only ``multiRun``
    needs the sweep controller. A named empty scenario deep-merges no values,
    so the canonical planner emits its ordinary ``base`` config exactly once.
    Convergence needs variation-outer execution because trial-outer repeated
    mode cannot decide that a cell has converged before scheduling its next
    trial.
    """
    sweep: dict[str, Any] = {
        "type": "scenarios",
        "runs": [{"name": "base"}],
    }
    if convergence:
        sweep["iterationOrder"] = "independent"
    return sweep
