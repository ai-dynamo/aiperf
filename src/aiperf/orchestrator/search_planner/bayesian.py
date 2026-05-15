# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curated BO preset (`--search-planner=bayesian`).

Thin subclass of :class:`OptunaSearchPlanner` that tries the optional
``BoTorchSampler`` first and sets ``optuna_acquisition`` to ``"qlognei"``
(single-objective) or ``"qlognehvi"`` (multi-objective). If the optional
BoTorch stack is unavailable, it warns and falls back to Optuna's core TPE
sampler.
For multi-objective runs the ``qlognehvi`` candidates_func is *not*
installed at sampler construction — :func:`build_sampler` deliberately
passes ``candidates_func=None`` when ``n_obj > 1`` and the planner
installs it later via ``_maybe_install_qnehvi_candidates_func`` once
``n_initial_points`` feasible probes have accumulated (so a reference
point can be derived from the observed feasible front). Surfaces a
high-level "just give me sane Bayesian optimization defaults" entry
point so users don't have to know which ``--optuna-sampler`` /
``--optuna-acquisition`` strings to pass.

Power users who want to pick a different sampler (TPE, GP) or acquisition
should use ``--search-planner=optuna`` instead. Explicit BoTorch requests on
that planner raise when the optional stack is unavailable; only this curated
preset performs the TPE fallback.

The Hvarfner-DSP kernel (Hvarfner et al. ICML 2024, arXiv:2402.02229) is
applied to this preset's qLogNEI / qLogNEHVI fits via
:mod:`_optuna_helpers`'s candidates_func builders; nothing
preset-specific is needed here for that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.orchestrator.search_planner.optuna_planner import OptunaSearchPlanner

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.sweep import AdaptiveSearchSweep


__all__ = ["BayesianSearchPlanner"]


class BayesianSearchPlanner(OptunaSearchPlanner):
    """Curated Optuna preset for the default Bayesian planner.

    Tries `optuna_sampler="botorch"` with qLogNEI or qLogNEHVI first. If the
    optional BoTorch stack is unavailable, falls back to core Optuna TPE. Use
    `OptunaSearchPlanner` directly when caller-provided sampler/acquisition values
    must be preserved.
    """

    def __init__(self, base_config: BenchmarkConfig, cfg: AdaptiveSearchSweep) -> None:
        n_objectives = len(cfg.objectives)
        curated_acquisition = "qlognehvi" if n_objectives > 1 else "qlognei"
        curated_cfg = cfg.model_copy(
            update={
                "optuna_sampler": "botorch",
                "optuna_acquisition": curated_acquisition,
            }
        )
        super().__init__(base_config, curated_cfg, allow_implicit_botorch_fallback=True)
