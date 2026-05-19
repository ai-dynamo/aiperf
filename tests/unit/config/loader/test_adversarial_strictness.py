# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate-time strictness regressions surfaced by the v2 chaos harness.

Pins the load-time error paths that previously slipped through:

* Orphan ``{{`` / ``{%`` markers in any string field.
* Unterminated ``${`` env-var openers.
* Empty / NaN / malformed sweep dimension values on grid + zip sweeps.

These checks live in the loader / sweep models because the cheap pre-flight
``aiperf config validate`` should catch them; deferring to ``aiperf profile``
buys the user a real-benchmark startup cost just to discover a typo.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from aiperf.config.loader.env_vars import substitute_env_vars
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.jinja import render_jinja2_templates
from aiperf.config.sweep.config import GridSweep, ZipSweep

# ---------- jinja: orphan delimiters ----------


def test_orphan_double_open_brace_raises() -> None:
    """``{{`` with no closing ``}}`` must surface as a load-time error."""
    with pytest.raises(ConfigurationError) as exc:
        render_jinja2_templates({"model": "mock-{{ unclosed"}, context={})
    assert (
        "{{" in str(exc.value.message) or "unbalanced" in str(exc.value.message).lower()
    )


def test_orphan_block_open_raises() -> None:
    """``{% ... %}`` block without a closing ``%}`` must error at load time."""
    with pytest.raises(ConfigurationError) as exc:
        render_jinja2_templates({"model": "mock-{% if true %"}, context={})
    msg = str(exc.value.message).lower()
    assert "{%" in str(exc.value.message) or "unbalanced" in msg


# ---------- env-vars: unterminated brace ----------


def test_unterminated_env_var_brace_raises() -> None:
    """``${VAR`` with no closing ``}`` must surface as ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc:
        substitute_env_vars("mock-${UNTERMINATED_VAR_NAME")
    assert "Unterminated" in str(exc.value)


def test_well_formed_env_var_default_passes() -> None:
    """A valid ``${VAR:default}`` should not trip the new orphan check."""
    assert substitute_env_vars("${THIS_DOES_NOT_EXIST_X:fallback}") == "fallback"


# ---------- sweep: dotted path strictness on grid + zip ----------


def test_grid_sweep_rejects_empty_path() -> None:
    with pytest.raises(ValidationError, match=r"non-empty"):
        GridSweep(parameters={"": [1, 2, 3]})


def test_grid_sweep_rejects_leading_dot() -> None:
    with pytest.raises(ValidationError, match=r"must not start with '\.'"):
        GridSweep(parameters={".phases.profiling.rate": [1.0, 2.0]})


def test_grid_sweep_rejects_double_dot() -> None:
    with pytest.raises(ValidationError, match=r"consecutive dots"):
        GridSweep(parameters={"phases..profiling.rate": [1.0, 2.0]})


def test_grid_sweep_rejects_benchmark_prefix() -> None:
    with pytest.raises(ValidationError, match=r"redundant"):
        GridSweep(parameters={"benchmark.phases.profiling.rate": [1.0, 2.0]})


def test_grid_sweep_rejects_envelope_field() -> None:
    with pytest.raises(ValidationError, match=r"non-sweepable"):
        GridSweep(parameters={"random_seed": [1, 2, 3]})


def test_zip_sweep_rejects_double_dot() -> None:
    with pytest.raises(ValidationError, match=r"consecutive dots"):
        ZipSweep(parameters={"phases..profiling.rate": [1.0, 2.0]})


# ---------- sweep: value-list strictness ----------


def test_grid_sweep_rejects_empty_values() -> None:
    with pytest.raises(ValidationError, match=r"non-empty"):
        GridSweep(parameters={"phases.profiling.rate": []})


def test_grid_sweep_rejects_nan_value() -> None:
    with pytest.raises(ValidationError, match=r"not\s+finite|NaN"):
        GridSweep(parameters={"phases.profiling.rate": [1.0, math.nan, 3.0]})


def test_grid_sweep_rejects_inf_value() -> None:
    with pytest.raises(ValidationError, match=r"not\s+finite|inf"):
        GridSweep(parameters={"phases.profiling.rate": [1.0, math.inf]})


def test_zip_sweep_rejects_nan_value() -> None:
    with pytest.raises(ValidationError, match=r"not\s+finite|NaN"):
        ZipSweep(
            parameters={
                "phases.profiling.rate": [1.0, math.nan],
                "phases.profiling.concurrency": [1, 2],
            }
        )


def test_grid_sweep_accepts_well_formed_paths() -> None:
    """Sanity: the validator does not reject legitimate sweep paths."""
    sweep = GridSweep(
        parameters={
            "phases.profiling.concurrency": [1, 2, 4],
            "variables.isl": [128, 256],
        }
    )
    assert len(sweep.parameters) == 2
