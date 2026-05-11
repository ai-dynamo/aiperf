# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hvarfner-DSP kernel factory for BoTorch GPs.

Per Hvarfner, Hellsten, Nardi 2024, "Vanilla Bayesian Optimization
Performs Great in High Dimensions" (ICML 2024, arXiv:2402.02229), a
LogNormal lengthscale prior whose location parameter scales with
``1/2 log(D)`` (equivalent to scaling lengthscales by sqrt(D) in linear
space) makes vanilla GP-EI competitive up to D > 6000 *without* any
structural restriction on the objective. The paper's prior is
``LogNormal(sqrt(2) + log(D)/2, sqrt(3))``; this implementation
evaluates ``loc`` for the run-time ``d``.

Combined with the canonical Matern-5/2 kernel and ARD lengthscales (one
per input dimension), this is the closest-to-paper-faithful BoTorch GP
prior available without HMC-fully-Bayesian inference (which would be
SAASBO; out of scope for this migration).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpytorch.kernels import ScaleKernel


def make_dsp_kernel(d: int) -> ScaleKernel:
    """Return a Hvarfner-DSP-scaled Matern 5/2 kernel for ``d`` input dims.

    The output ``ScaleKernel`` wraps an ARD ``MaternKernel(nu=2.5)`` with a
    ``LogNormal(loc = sqrt(2) + 1/2 log(d), scale = sqrt(3))`` prior on each
    lengthscale -- the form verified by Hvarfner et al. 2024 section 3.3.
    The output-scale prior is the BoTorch default ``Gamma(2, 0.15)``,
    which is well-behaved for objectives normalized via ``Standardize``.
    """
    import torch
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior, LogNormalPrior

    if d < 1:
        raise ValueError(f"d must be >= 1; got {d}")
    loc = torch.tensor(math.sqrt(2.0) + 0.5 * math.log(d), dtype=torch.float64)
    scale = torch.tensor(math.sqrt(3.0), dtype=torch.float64)
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_prior=LogNormalPrior(loc=loc, scale=scale),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
