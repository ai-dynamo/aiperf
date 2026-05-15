# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hvarfner DSP kernel factory."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

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
