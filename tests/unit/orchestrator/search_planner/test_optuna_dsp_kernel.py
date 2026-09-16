# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that the Optuna+BoTorch candidates_funcs build their GP with the
Hvarfner-DSP kernel (Matern 5/2 + sqrt(D)-scaled LogNormal prior), not
BoTorch's default RBF.
"""

from __future__ import annotations

import math

import pytest
from pytest import param

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
botorch = pytest.importorskip("botorch")
gpytorch = pytest.importorskip("gpytorch")

from aiperf.orchestrator.search_planner._optuna_helpers import (  # noqa: E402
    build_qlognei_candidates_func,
    build_qnehvi_candidates_func,
)


def _capture_built_models(monkeypatch, captured: list):
    """Patch SingleTaskGP so every constructed model lands in `captured`.

    A subclass rather than a wrapper function: the call sites reach through
    the patched symbol for classmethods/staticmethods too (e.g.
    ``SingleTaskGP.get_batch_dimensions``), which a plain function does not
    carry.
    """
    real_cls = botorch.models.SingleTaskGP

    class _Spy(real_cls):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(botorch.models, "SingleTaskGP", _Spy)
    # Also patch the symbol bound inside _optuna_helpers' candidates_func
    # closure — the `from botorch.models import SingleTaskGP` binding is
    # captured at builder construction time inside the closure, so monkeypatch
    # must hit that namespace too.
    import aiperf.orchestrator.search_planner._optuna_helpers as helpers_mod

    if hasattr(helpers_mod, "SingleTaskGP"):
        monkeypatch.setattr(helpers_mod, "SingleTaskGP", _Spy)


def test_qlognei_candidates_func_fits_dsp_kernel(monkeypatch):
    captured: list = []
    _capture_built_models(monkeypatch, captured)

    func = build_qlognei_candidates_func()
    train_x = torch.rand(8, 3, dtype=torch.float64)
    train_obj = torch.rand(8, 1, dtype=torch.float64)
    bounds = torch.stack(
        [torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)]
    )
    func(train_x, train_obj, None, bounds, None)

    assert len(captured) >= 1, "qlognei builder did not construct any SingleTaskGP"
    gp = captured[-1]
    base_kernel = gp.covar_module.base_kernel
    assert isinstance(base_kernel, gpytorch.kernels.MaternKernel)
    assert base_kernel.nu == 2.5
    assert base_kernel.ard_num_dims == 3
    prior = base_kernel.lengthscale_prior
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)
    expected_loc = math.sqrt(2.0) + 0.5 * math.log(3)
    assert math.isclose(prior.loc.item(), expected_loc, rel_tol=1e-9)


@pytest.mark.parametrize(
    "n_sla_filters",
    [
        param(1, id="one-filter"),
        param(2, id="two-filters"),
    ],
)  # fmt: skip
def test_qlognei_candidates_func_fits_with_sla_constraints(n_sla_filters: int) -> None:
    """Regression: the constrained path cat-stacks objective + one column per
    SLA filter, which makes the GP a batched multi-output model. The call site
    must derive the matching kernel batch_shape or the fit raises
    ``shape '[m, 1]' is invalid for input of size 1``.

    Goes through ``build_qlognei_candidates_func`` rather than reconstructing
    the model inline, so it exercises the wiring that actually regressed.
    """
    func = build_qlognei_candidates_func()
    d = 3
    torch.manual_seed(0)
    train_x = torch.rand(8, d, dtype=torch.float64)
    train_obj = torch.rand(8, 1, dtype=torch.float64)
    # BoTorch treats con <= 0 as feasible. Pin the signs rather than relying
    # on random draws: even rows strictly feasible, odd rows strictly not.
    train_con = torch.rand(8, n_sla_filters, dtype=torch.float64) + 0.5
    train_con[::2] *= -1.0
    bounds = torch.stack(
        [torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)]
    )

    candidates = func(train_x, train_obj, train_con, bounds, None)

    assert candidates.shape == (1, d)


@pytest.mark.parametrize(
    "n_sla_filters",
    [param(0, id="no-sla"), param(1, id="one-sla")],
)  # fmt: skip
def test_qlognei_candidates_func_supports_repeated_fits(n_sla_filters: int) -> None:
    """Keep successive qLogNEI candidates finite and in bounds as data grows."""
    torch.manual_seed(0)
    func = build_qlognei_candidates_func()
    train_x = torch.linspace(0, 1, 6, dtype=torch.float64).unsqueeze(-1)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

    for _ in range(3):
        train_obj = torch.sin(2 * math.pi * train_x)
        train_con = (
            torch.cat([train_x - 0.65, 0.1 - train_x], dim=-1)[:, :n_sla_filters]
            if n_sla_filters
            else None
        )
        candidates = func(train_x, train_obj, train_con, bounds, None)

        assert candidates.shape == (1, 1)
        assert torch.isfinite(candidates).all()
        assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
        train_x = torch.cat([train_x, candidates], dim=0)


def test_qnehvi_candidates_func_fits_dsp_kernel_per_objective(monkeypatch):
    captured: list = []
    _capture_built_models(monkeypatch, captured)

    func = build_qnehvi_candidates_func(reference_point=[-1e9, -1e9])
    d = 4
    train_x = torch.rand(8, d, dtype=torch.float64)
    train_obj = torch.rand(8, 2, dtype=torch.float64)  # two objectives
    bounds = torch.stack(
        [torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)]
    )
    func(train_x, train_obj, None, bounds, None)

    # Two GPs (one per objective).
    assert len(captured) == 2, f"expected 2 SingleTaskGPs, got {len(captured)}"
    expected_loc = math.sqrt(2.0) + 0.5 * math.log(d)
    for gp in captured:
        base_kernel = gp.covar_module.base_kernel
        assert isinstance(base_kernel, gpytorch.kernels.MaternKernel)
        assert base_kernel.nu == 2.5
        assert base_kernel.ard_num_dims == d
        assert math.isclose(
            base_kernel.lengthscale_prior.loc.item(), expected_loc, rel_tol=1e-9
        )
