# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""search_history.json must survive numpy scalars reaching the exporter.

``SearchIteration`` is a plain dataclass with ``variation_values:
dict[str, Any]``, so nothing between a planner and the exporter coerces
scalar types. Planners are expected to hand back native Python numbers,
but the export boundary must not be the thing that fails when one
doesn't -- ``scrub_non_finite`` is the guard that makes that true.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    SearchSpaceDimension,
)
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.search_planner.base import SearchIteration


def _real_dim_cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="endpoint.timeout", lo=1.0, hi=100.0, kind="real")
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=10,
    )


def test_write_search_history_serializes_numpy_variation_values(tmp_path: Path) -> None:
    """A numpy.float64 in variation_values must not abort the export.

    Regression for nvbugs 6683575: orjson rejects numpy.float64 outright
    ("Type is not JSON serializable: numpy.float64"), which failed the whole
    run rather than just the export.
    """
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"endpoint.timeout": np.float64(12.5)},
            objective_value=np.float64(10.0),
            objective_values=[np.float64(10.0)],
            feasible=True,
        )
    ]

    write_search_history(tmp_path, history, _real_dim_cfg())

    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert payload["iterations"][0]["variation_values"][
        "endpoint.timeout"
    ] == pytest.approx(12.5)


def test_write_search_history_maps_numpy_nan_objective_to_null(tmp_path: Path) -> None:
    """A numpy NaN must land as JSON null, not as a coerced-but-unscrubbed value.

    The guard has to preserve the module's NaN discipline: converting numpy
    scalars without re-checking finiteness would let orjson turn a NaN into
    ``null`` by its own silent coercion, which is right by accident here but
    wrong for the "not scored" contract the exporter relies on.
    """
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"endpoint.timeout": np.float64(12.5)},
            objective_value=np.float64("nan"),
            objective_values=[np.float64("nan")],
            feasible=False,
        )
    ]

    write_search_history(tmp_path, history, _real_dim_cfg())

    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert payload["iterations"][0]["objective_values"] == [None]
