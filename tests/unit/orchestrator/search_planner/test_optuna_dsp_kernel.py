# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that the Optuna+BoTorch candidates_funcs build their GP with the
Hvarfner-DSP kernel (Matern 5/2 + sqrt(D)-scaled LogNormal prior), not
BoTorch's default RBF.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
botorch = pytest.importorskip("botorch")
gpytorch = pytest.importorskip("gpytorch")

from aiperf.orchestrator.search_planner._optuna_helpers import (  # noqa: E402
    build_qlognei_candidates_func,
    build_qnehvi_candidates_func,
)


def _capture_built_models(monkeypatch, captured: list):
    """Patch SingleTaskGP so every constructed model lands in `captured`."""
    real_cls = botorch.models.SingleTaskGP

    # A subclass, not a wrapper function: production code also reaches for
    # class-level API such as SingleTaskGP.get_batch_dimensions, which a plain
    # function stand-in would not carry.
    class _Spy(real_cls):
        def __init__(self, *args, **kwargs):
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


def test_qlognei_constrained_path_fits_batched_gp() -> None:
    """Regression: constraints make train_y multi-column -> batched GP.

    Exercises ``build_qlognei_candidates_func`` end to end with a non-None
    ``train_con``. Without a batched kernel the fit dies inside BoTorch's
    scipy path with ``shape '[m, 1]' is invalid for input of size 1``. Also
    smoke-tests the ``GenericMCObjective`` + ``constraints`` wiring downstream
    of the fit.
    """
    func = build_qlognei_candidates_func()
    train_x = torch.rand(8, 3, dtype=torch.float64)
    train_obj = torch.rand(8, 1, dtype=torch.float64)
    train_con = torch.rand(8, 2, dtype=torch.float64) - 0.5  # two SLA filters
    bounds = torch.stack(
        [torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)]
    )
    candidates = func(train_x, train_obj, train_con, bounds, None)

    assert candidates.shape == (1, 3)


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
