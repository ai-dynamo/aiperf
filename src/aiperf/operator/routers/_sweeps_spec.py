# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep spec summary helpers for the operator UI API."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from aiperf.operator.models import AIPerfSweepSpec
from aiperf.operator.routers.sweeps_models import DimensionInfo, SpecSummary

logger = logging.getLogger("aiperf.operator.ui")


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


def spec_summary_from_record(rec: Any) -> SpecSummary:
    """Build a SpecSummary from whichever side of the union is available.

    Legacy-shape CRs that fail ``AIPerfSweepSpec.model_validate`` fall back to
    the archived ``aggregate_doc`` path rather than 422'ing the whole route.
    """
    if rec.raw_spec:
        try:
            spec = AIPerfSweepSpec.model_validate(rec.raw_spec)
            multi_run = spec.multi_run.model_dump(mode="json", by_alias=True)
            convergence = (
                spec.multi_run.convergence.model_dump(mode="json", by_alias=True)
                if spec.multi_run.convergence is not None
                else None
            )
            return SpecSummary(
                sweep_type=spec.sweep.type,
                dimensions=dimensions_from_sweep_model(spec.sweep),
                multi_run=multi_run,
                convergence=convergence,
            )
        except ValidationError as exc:
            logger.warning(
                "AIPerfSweep %s/%s raw_spec rejected; falling back to aggregate. %s",
                rec.namespace,
                rec.name,
                exc.errors(include_url=False),
            )
    if rec.aggregate_doc is not None:
        snap = rec.aggregate_doc.get("spec_snapshot") or {}
        dims_raw = snap.get("dimensions") or []
        dims = [
            DimensionInfo(name=d["name"], values=list(d.get("values") or []))
            for d in dims_raw
            if isinstance(d, dict) and isinstance(d.get("name"), str)
        ]
        return SpecSummary(
            sweep_type=str(snap.get("sweep_type") or "grid"),  # type: ignore[arg-type]
            dimensions=dims,
            multi_run=snap.get("multi_run"),
            convergence=snap.get("convergence"),
        )
    return SpecSummary(
        sweep_type="grid", dimensions=[], multi_run=None, convergence=None
    )
