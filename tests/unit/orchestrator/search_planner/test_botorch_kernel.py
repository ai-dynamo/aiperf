# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hvarfner DSP kernel factory."""

from __future__ import annotations

import math

import pytest
from pytest import param

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
# Module level, not inside the test: a per-test importorskip leaves the CI job
# green (exit 0) with the regression silently skipped. Collected-nothing here
# exits 5 instead, which is loud.
botorch = pytest.importorskip("botorch")

from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Standardize  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

from aiperf.orchestrator.search_planner._botorch_kernel import (  # noqa: E402
    make_dsp_kernel,
)


def test_dsp_kernel_uses_matern_5_2_with_ard():
    kernel = make_dsp_kernel(d=4)
    assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
    base = kernel.base_kernel
    assert isinstance(base, gpytorch.kernels.MaternKernel)
    assert base.nu == 2.5
    assert base.ard_num_dims == 4


def test_dsp_kernel_lengthscale_prior_shifts_with_sqrt_d():
    """Hvarfner 2024: prior is LogNormal(loc=√2 + 0.5*log(D), scale=√3)."""
    d = 9
    kernel = make_dsp_kernel(d=d)
    prior = kernel.base_kernel.lengthscale_prior
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)
    expected_loc = math.sqrt(2.0) + 0.5 * math.log(d)
    expected_scale = math.sqrt(3.0)
    assert math.isclose(prior.loc.item(), expected_loc, rel_tol=1e-9)
    assert math.isclose(prior.scale.item(), expected_scale, rel_tol=1e-9)


class TestDspKernelBatchShape:
    """Regression: the DSP kernel must batch to match a multi-output GP.

    A ``SingleTaskGP`` built over an ``m``-column ``train_y`` is a batched
    multi-output model with ``batch_shape == Size([m])``. Constrained qLogNEI
    reaches that case whenever SLA filters are configured, because
    ``_qlognei_constraint_kwargs`` cat-stacks the objective with one column per
    filter. A kernel without the matching leading dim has size-1 parameters,
    and the fit dies deep inside BoTorch's scipy path with
    ``shape '[m, 1]' is invalid for input of size 1``.

    This crashed cluster adaptive search on every run that set ``slaFilters``.
    It went unnoticed because the runtime image shipped without the [botorch]
    extra, so BayesianSearchPlanner silently fell back to Optuna's TPE sampler
    and this code path never executed.
    """

    def test_batch_shape_is_applied_to_both_kernels(self) -> None:
        kernel = make_dsp_kernel(d=3, batch_shape=torch.Size([2]))
        assert kernel.batch_shape == torch.Size([2])
        assert kernel.base_kernel.batch_shape == torch.Size([2])
        assert kernel.base_kernel.lengthscale.shape[0] == 2

    def test_omitting_batch_shape_stays_unbatched(self) -> None:
        kernel = make_dsp_kernel(d=3)
        assert kernel.batch_shape == torch.Size([])

    @pytest.mark.parametrize("n_outputs", [1, 2, 3])
    def test_gp_fit_succeeds_for_each_output_count(self, n_outputs: int) -> None:
        """m=1 is unconstrained; m>1 is one column per SLA filter."""
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(0)
        train_x = torch.rand(6, 1, dtype=torch.float64)
        train_y = torch.rand(6, n_outputs, dtype=torch.float64)

        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=make_dsp_kernel(
                d=1,
                batch_shape=torch.Size([n_outputs]) if n_outputs > 1 else None,
            ),
            outcome_transform=Standardize(m=n_outputs),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))


class TestDspKernelLengthscaleBounds:
    @pytest.mark.parametrize(
        "d",
        [param(1, id="one-dimension"), param(4, id="four-dimensions")],
    )  # fmt: skip
    @pytest.mark.parametrize(
        "batch_shape",
        [param(torch.Size([]), id="unbatched"), param(torch.Size([2]), id="batched")],
    )  # fmt: skip
    def test_lengthscale_initializes_at_prior_mode(
        self, d: int, batch_shape: torch.Size
    ) -> None:
        base = make_dsp_kernel(d=d, batch_shape=batch_shape).double().base_kernel

        torch.testing.assert_close(
            base.lengthscale,
            base.lengthscale_prior.mode.expand_as(base.lengthscale),
        )

    @pytest.mark.parametrize(
        "dtype",
        [param(torch.float32, id="float32"), param(torch.float64, id="float64")],
    )  # fmt: skip
    @pytest.mark.parametrize(
        "batch_shape",
        [param(torch.Size([]), id="unbatched"), param(torch.Size([2]), id="batched")],
    )  # fmt: skip
    def test_prior_is_finite_at_optimizer_lower_bound(
        self, dtype: torch.dtype, batch_shape: torch.Size
    ) -> None:
        base = make_dsp_kernel(d=2, batch_shape=batch_shape).to(dtype).base_kernel
        _, bounds = get_parameters_and_bounds(base)
        lower_bound, _ = bounds["raw_lengthscale"]

        # Positive()/Softplus exposes -inf to SciPy, where the transformed
        # lengthscale becomes zero and falls outside LogNormalPrior support.
        with torch.no_grad():
            base.raw_lengthscale.fill_(lower_bound)
        log_prob = base.lengthscale_prior.log_prob(base.lengthscale)
        log_prob.sum().backward()

        assert math.isfinite(lower_bound) and lower_bound > 0
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(base.raw_lengthscale.grad).all()

    @pytest.mark.parametrize(
        "n_outputs",
        [
            param(1, id="single-output"),
            param(2, id="two-outputs"),
            param(3, id="three-outputs"),
        ],
    )  # fmt: skip
    def test_gp_fit_projects_out_of_bounds_lengthscale(self, n_outputs: int) -> None:
        torch.manual_seed(0)
        train_x = torch.rand(8, 2, dtype=torch.float64)
        train_y = torch.cat(
            [
                torch.sin((i + 1) * train_x[:, :1] * 3) + train_x[:, 1:]
                for i in range(n_outputs)
            ],
            dim=-1,
        )
        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=make_dsp_kernel(
                d=2,
                batch_shape=torch.Size([n_outputs]) if n_outputs > 1 else None,
            ),
            outcome_transform=Standardize(m=n_outputs),
        )
        base = model.covar_module.base_kernel
        # A controlled bad starting point, not the original optimizer trace.
        # SciPy must project it into bounds before evaluating the GP/prior.
        with torch.no_grad():
            base.raw_lengthscale.fill_(-1000.0)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, max_attempts=1)

        assert not mll.training
        assert (base.lengthscale >= base.raw_lengthscale_constraint.lower_bound).all()
        assert torch.isfinite(base.lengthscale_prior.log_prob(base.lengthscale)).all()
        posterior = model.posterior(train_x)
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()
