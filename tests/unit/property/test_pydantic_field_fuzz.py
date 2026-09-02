# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis fuzzing for important Pydantic config models.

Property: every model either validates cleanly OR raises a clean
``pydantic.ValidationError`` / ``aiperf.common.exceptions.ConfigurationError``
/ ``ValueError`` (the only exceptions Pydantic conventionally bubbles
out of validators in this codebase). Any other exception type --
``AttributeError``, ``TypeError`` from a missing-key bug, ``KeyError``,
``RecursionError``, ``IndexError`` -- means a validator crashed on the
adversarial input rather than rejecting it cleanly.

This protects against the class of bugs found in round-2 (e.g. NaN
silently passing a finite-bounds check, an unhashable choices entry
crashing inside ``set()`` build, the ``mean``-only ambiguous-distribution
path raising the wrong error type).
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from pydantic import BaseModel, TypeAdapter, ValidationError

from aiperf.common.exceptions import ConfigurationError
from aiperf.config.distributions import (
    EmpiricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    SamplingDistribution,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    GridSweep,
    LatinHypercubeSweep,
    Objective,
    SamplingDimension,
    ScenarioSweep,
    SobolSweep,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig
from tests.unit.property._strategies import (
    adaptive_objective_inputs,
    adaptive_search_sweep_inputs,
    empirical_distribution_inputs,
    endpoint_inputs,
    fixed_distribution_inputs,
    grid_sweep_inputs,
    lognormal_distribution_inputs,
    multi_run_inputs,
    multimodal_distribution_inputs,
    normal_distribution_inputs,
    sampling_dimension_inputs,
    scenario_sweep_inputs,
    search_space_dimension_inputs,
    sla_filter_inputs,
    sobol_sweep_inputs,
)

# Exceptions a validator is allowed to raise. Anything else means an unhandled
# crash inside a validator -- a bug.
ALLOWED = (ValidationError, ConfigurationError, ValueError)
PROFILE = settings(max_examples=150)


def _check_no_unhandled(model: type[BaseModel], data: Any) -> None:
    """Validate ``data`` and assert any failure is a handled error type."""
    try:
        model.model_validate(data)
    except ALLOWED:
        pass
    except Exception as e:  # we want the exception type
        raise AssertionError(
            f"{model.__name__}.model_validate raised unhandled "
            f"{type(e).__name__}: {e!r}\nfor input: {data!r}"
        ) from e


# ----------------------------------------------------------------------------
# Sweep dimension fuzzers
# ----------------------------------------------------------------------------


@PROFILE
@given(data=sampling_dimension_inputs())
def test_sampling_dimension_never_unhandled(data: dict) -> None:
    _check_no_unhandled(SamplingDimension, data)


@PROFILE
@given(data=search_space_dimension_inputs())
def test_search_space_dimension_never_unhandled(data: dict) -> None:
    _check_no_unhandled(SearchSpaceDimension, data)


@PROFILE
@given(data=sla_filter_inputs())
def test_sla_filter_never_unhandled(data: dict) -> None:
    _check_no_unhandled(SLAFilter, data)


@PROFILE
@given(data=adaptive_objective_inputs())
def test_adaptive_objective_never_unhandled(data: dict) -> None:
    _check_no_unhandled(Objective, data)


# ----------------------------------------------------------------------------
# Sweep top-level fuzzers
# ----------------------------------------------------------------------------


@PROFILE
@given(data=grid_sweep_inputs())
def test_grid_sweep_never_unhandled(data: dict) -> None:
    _check_no_unhandled(GridSweep, data)


@PROFILE
@given(data=scenario_sweep_inputs())
def test_scenario_sweep_never_unhandled(data: dict) -> None:
    _check_no_unhandled(ScenarioSweep, data)


@PROFILE
@given(data=sobol_sweep_inputs())
def test_sobol_sweep_never_unhandled(data: dict) -> None:
    _check_no_unhandled(SobolSweep, data)


@PROFILE
@given(data=sobol_sweep_inputs())
def test_latin_hypercube_sweep_never_unhandled(data: dict) -> None:
    # Same input shape; the type discriminator picks one or the other.
    _check_no_unhandled(LatinHypercubeSweep, data)


@PROFILE
@given(data=adaptive_search_sweep_inputs())
def test_adaptive_search_sweep_never_unhandled(data: dict) -> None:
    _check_no_unhandled(AdaptiveSearchSweep, data)


# ----------------------------------------------------------------------------
# Multi-run / convergence fuzzers
# ----------------------------------------------------------------------------


@PROFILE
@given(data=multi_run_inputs())
def test_multi_run_config_never_unhandled(data: dict) -> None:
    _check_no_unhandled(MultiRunConfig, data)


@PROFILE
@given(
    data=multi_run_inputs().map(
        lambda d: d.get("convergence") if isinstance(d, dict) else None
    )
)
def test_convergence_config_never_unhandled(data: Any) -> None:
    if data is None:
        return
    _check_no_unhandled(ConvergenceConfig, data)


# ----------------------------------------------------------------------------
# Distribution fuzzers (each subclass + the discriminated union)
# ----------------------------------------------------------------------------


@PROFILE
@given(data=fixed_distribution_inputs())
def test_fixed_distribution_never_unhandled(data: Any) -> None:
    if isinstance(data, (int, float)):
        # Scalar shorthand goes through the discriminator-level path,
        # which the union TypeAdapter handles. Hit it via the union below.
        return
    _check_no_unhandled(FixedDistribution, data)


@PROFILE
@given(data=normal_distribution_inputs())
def test_normal_distribution_never_unhandled(data: dict) -> None:
    _check_no_unhandled(NormalDistribution, data)


@PROFILE
@given(data=lognormal_distribution_inputs())
def test_lognormal_distribution_never_unhandled(data: dict) -> None:
    _check_no_unhandled(LogNormalDistribution, data)


@PROFILE
@given(data=multimodal_distribution_inputs())
def test_multimodal_distribution_never_unhandled(data: dict) -> None:
    _check_no_unhandled(MultimodalDistribution, data)


@PROFILE
@given(data=empirical_distribution_inputs())
def test_empirical_distribution_never_unhandled(data: dict) -> None:
    _check_no_unhandled(EmpiricalDistribution, data)


_DIST_ADAPTER = TypeAdapter(SamplingDistribution)


@PROFILE
@given(
    data=fixed_distribution_inputs()
    | normal_distribution_inputs()
    | lognormal_distribution_inputs()
    | multimodal_distribution_inputs()
    | empirical_distribution_inputs()
)
def test_sampling_distribution_union_never_unhandled(data: Any) -> None:
    """The discriminated union picks a subclass from structure -- the
    discriminator itself can raise; that's the user-facing failure mode.
    """
    try:
        _DIST_ADAPTER.validate_python(data)
    except ALLOWED:
        pass
    except Exception as e:
        raise AssertionError(
            f"SamplingDistribution discriminator raised unhandled "
            f"{type(e).__name__}: {e!r}\nfor input: {data!r}"
        ) from e


# ----------------------------------------------------------------------------
# Endpoint fuzzer
# ----------------------------------------------------------------------------


@PROFILE
@given(data=endpoint_inputs())
def test_endpoint_config_never_unhandled(data: dict) -> None:
    _check_no_unhandled(CLIConfig, data)


# ----------------------------------------------------------------------------
# Smoke: well-formed inputs DO validate cleanly
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            SamplingDimension,
            {"path": "phases.profiling.concurrency", "lo": 1.0, "hi": 64.0},
        ),
        (
            SearchSpaceDimension,
            {"path": "phases.profiling.concurrency", "lo": 1.0, "hi": 64.0},
        ),
        (
            SLAFilter,
            {"metric_tag": "ttft", "op": "lt", "threshold": 200.0},
        ),
        (
            Objective,
            {"metric": "throughput", "direction": "maximize"},
        ),
        (
            GridSweep,
            {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 4]},
            },
        ),
        (
            MultiRunConfig,
            {"num_runs": 3},
        ),
        (
            FixedDistribution,
            {"value": 512.0},
        ),
        (
            NormalDistribution,
            {"mean": 100.0, "stddev": 10.0},
        ),
    ],
)
def test_well_formed_input_validates(model: type[BaseModel], payload: dict) -> None:
    """Sanity check that the strategy generators aren't masking validation."""
    instance = model.model_validate(payload)
    assert instance is not None
