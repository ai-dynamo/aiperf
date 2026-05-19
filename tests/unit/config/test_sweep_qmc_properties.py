# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis property tests for QMC sweep expansion.

Mirrors the existing `tests/unit/orchestrator/test_parameter_sweep_properties.py`
pattern in this repo.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep import SamplingDimension
from aiperf.config.sweep.expand_qmc import expand_qmc_sweep


def _base_data():
    return {"benchmark": {"model": "m"}}


def _expand_sobol(data, *, samples, seed, dimensions):
    if samples & (samples - 1) == 0:
        return expand_qmc_sweep(
            data,
            sweep_type="sobol",
            samples=samples,
            seed=seed,
            dimensions=dimensions,
            options={"scramble": True},
        )

    with pytest.warns(UserWarning, match="Sobol balance is best at powers of 2"):
        return expand_qmc_sweep(
            data,
            sweep_type="sobol",
            samples=samples,
            seed=seed,
            dimensions=dimensions,
            options={"scramble": True},
        )


@given(
    samples=st.integers(min_value=2, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    lo=st.floats(min_value=1.0, max_value=10.0),
    hi=st.floats(min_value=100.0, max_value=1000.0),
)
@settings(max_examples=30, deadline=None)
def test_qmc_count_invariant_sobol(samples, seed, lo, hi):
    dims = [SamplingDimension(path="x", lo=lo, hi=hi)]
    out = _expand_sobol(
        _base_data(),
        samples=samples,
        seed=seed,
        dimensions=dims,
    )
    assert len(out) == samples


@given(
    samples=st.integers(min_value=2, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    lo=st.floats(min_value=1.0, max_value=10.0),
    hi=st.floats(min_value=100.0, max_value=1000.0),
)
@settings(max_examples=30, deadline=None)
def test_qmc_values_within_bounds_sobol(samples, seed, lo, hi):
    dims = [SamplingDimension(path="x", lo=lo, hi=hi)]
    out = _expand_sobol(
        _base_data(),
        samples=samples,
        seed=seed,
        dimensions=dims,
    )
    for _, var in out:
        v = var.values["x"]
        assert lo <= v <= hi, f"value {v} outside [{lo}, {hi}]"


@given(
    samples=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_qmc_purity_sobol(samples, seed):
    """Same inputs => same outputs."""
    dims = [SamplingDimension(path="x", lo=1.0, hi=100.0)]
    a = _expand_sobol(
        _base_data(),
        samples=samples,
        seed=seed,
        dimensions=dims,
    )
    b = _expand_sobol(
        _base_data(),
        samples=samples,
        seed=seed,
        dimensions=dims,
    )
    assert [v.values for _, v in a] == [v.values for _, v in b]


@given(
    samples=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_qmc_purity_lhs(samples, seed):
    dims = [SamplingDimension(path="x", lo=1.0, hi=100.0)]
    a = expand_qmc_sweep(
        _base_data(),
        sweep_type="latin_hypercube",
        samples=samples,
        seed=seed,
        dimensions=dims,
        options={"optimization": "random-cd"},
    )
    b = expand_qmc_sweep(
        _base_data(),
        sweep_type="latin_hypercube",
        samples=samples,
        seed=seed,
        dimensions=dims,
        options={"optimization": "random-cd"},
    )
    assert [v.values for _, v in a] == [v.values for _, v in b]
