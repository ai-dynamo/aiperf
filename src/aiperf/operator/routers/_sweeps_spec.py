# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep spec summary helpers for the operator UI API."""

from __future__ import annotations

from typing import Any

from aiperf.operator.routers.sweeps_models import DimensionInfo


def _dimension_display_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def dimensions_from_sweep_model(sweep: Any) -> list[DimensionInfo]:
    from aiperf.config.sweep import (
        AdaptiveSearchSweep,
        GridSweep,
        LatinHypercubeSweep,
        ScenarioSweep,
        SobolSweep,
        ZipSweep,
    )

    if isinstance(sweep, (GridSweep, ZipSweep)):
        return [
            DimensionInfo(name=_dimension_display_name(name), values=list(values))
            for name, values in sweep.variables.items()
        ]
    if isinstance(sweep, AdaptiveSearchSweep):
        return [
            DimensionInfo(
                name=_dimension_display_name(dim.path), values=[dim.lo, dim.hi]
            )
            for dim in sweep.search_space
        ]
    if isinstance(sweep, (SobolSweep, LatinHypercubeSweep)):
        return [
            DimensionInfo(
                name=_dimension_display_name(dim.path),
                values=list(dim.choices)
                if dim.choices is not None
                else [dim.lo, dim.hi],
            )
            for dim in sweep.dimensions
        ]
    if isinstance(sweep, ScenarioSweep):
        return [
            DimensionInfo(
                name="scenario",
                values=[
                    run.get("name", idx) if isinstance(run, dict) else idx
                    for idx, run in enumerate(sweep.runs)
                ],
            )
        ]
    return []
