# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep spec summary helpers for the operator UI API.

Also owns the archived-snapshot contract shared with the sweep-controller:
:func:`spec_summary_snapshot` builds the purpose-built summary dict that
``sweep_controller.main._write_sweep_parent_aggregate`` persists under
:data:`SPEC_SUMMARY_KEY` in ``aggregate.json``, and
:func:`spec_summary_from_record` consumes exactly that shape back. Keeping
the producer and the consumer in one module is what keeps the two sides of
the contract from drifting apart again.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from aiperf.operator.models import AIPerfSweepSpec
from aiperf.operator.routers.sweeps_models import DimensionInfo, SpecSummary

logger = logging.getLogger("aiperf.operator.ui")

# aggregate.json key carrying the purpose-built spec summary written by
# spec_summary_snapshot. camelCase to match the doc's other top-level keys
# (totalVariations, completedRuns, specSnapshot, ...).
SPEC_SUMMARY_KEY = "specSummary"
# Older archives carry only the FULL AIPerfSweepSpec dump under this key;
# the reader derives the summary from it via AIPerfSweepSpec.model_validate.
LEGACY_SPEC_SNAPSHOT_KEY = "specSnapshot"


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


def _snapshot_from_parts(sweep: Any, multi_run: Any) -> dict[str, Any]:
    """Build the snapshot dict from validated ``sweep`` + ``multi_run`` models."""
    return {
        "sweep_type": str(sweep.type),
        "dimensions": [
            dim.model_dump(mode="json") for dim in dimensions_from_sweep_model(sweep)
        ],
        "multi_run": multi_run.model_dump(mode="json", by_alias=True),
        "convergence": (
            multi_run.convergence.model_dump(mode="json", by_alias=True)
            if multi_run.convergence is not None
            else None
        ),
    }


def spec_summary_snapshot(spec: AIPerfSweepSpec) -> dict[str, Any]:
    """Build the purpose-built spec summary persisted in ``aggregate.json``.

    This is the producer half of the archived-snapshot contract: the
    sweep-controller writes this dict under :data:`SPEC_SUMMARY_KEY` when it
    archives a finished sweep, and :func:`spec_summary_from_record` reads the
    exact same shape back after the CR has been TTL-reaped. The shape mirrors
    :class:`SpecSummary` (``sweep_type`` / ``dimensions`` / ``multi_run`` /
    ``convergence``) so the reader needs no re-validation of the full spec.

    Example:
        >>> spec = AIPerfSweepSpec.model_validate(cr["spec"])
        >>> spec_summary_snapshot(spec)["sweep_type"]
        'grid'
    """
    return _snapshot_from_parts(spec.sweep, spec.multi_run)


def _summary_from_snapshot(snap: dict[str, Any]) -> SpecSummary:
    """Materialize a SpecSummary from a snapshot-shaped dict, tolerantly.

    Field-by-field extraction (rather than ``SpecSummary.model_validate``) so
    an archive written by a newer build with extra keys still renders.
    """
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


def _summary_from_legacy_spec_dump(
    rec: Any, legacy: dict[str, Any]
) -> dict[str, Any] | None:
    """Derive a snapshot-shaped dict from a full-spec dump in an old archive.

    Old archives persisted ``spec.model_dump(mode="json")`` (the entire
    workload spec) under :data:`LEGACY_SPEC_SNAPSHOT_KEY`. Only the ``sweep``
    and ``multi_run`` sub-blocks are re-validated here — a full
    ``AIPerfSweepSpec.model_validate`` is deliberately avoided because the
    archived dump does not round-trip perfectly (serialized ``None`` on
    constrained deployment fields raises), and the summary never needs those
    parts. Returns ``None`` when the sweep block is absent or unparsable.
    """
    from pydantic import TypeAdapter

    from aiperf.config.sweep import MultiRunConfig
    from aiperf.config.sweep.config import SweepConfig

    sweep_block = legacy.get("sweep")
    if not isinstance(sweep_block, dict) or not sweep_block:
        return None
    try:
        sweep = TypeAdapter(SweepConfig).validate_python(sweep_block)
        multi_run = MultiRunConfig.model_validate(legacy.get("multi_run") or {})
    except (ValidationError, TypeError) as exc:
        logger.warning(
            "AIPerfSweep %s/%s archived specSnapshot rejected; "
            "degrading to empty summary. %s",
            rec.namespace,
            rec.name,
            exc,
        )
        return None
    return _snapshot_from_parts(sweep, multi_run)


def _summary_snapshot_from_archive(
    rec: Any, aggregate_doc: dict[str, Any]
) -> dict[str, Any] | None:
    """Extract a snapshot-shaped dict from an archived ``aggregate.json``.

    Tries the purpose-built :data:`SPEC_SUMMARY_KEY` first; old archives that
    predate it carry only the full spec dump under
    :data:`LEGACY_SPEC_SNAPSHOT_KEY`, which is summarized via
    :func:`_summary_from_legacy_spec_dump`. Returns ``None`` when neither key
    yields a usable summary.
    """
    snap = aggregate_doc.get(SPEC_SUMMARY_KEY)
    if isinstance(snap, dict) and snap:
        return snap
    legacy = aggregate_doc.get(LEGACY_SPEC_SNAPSHOT_KEY)
    if isinstance(legacy, dict) and legacy:
        return _summary_from_legacy_spec_dump(rec, legacy)
    return None


def spec_summary_from_record(rec: Any) -> SpecSummary:
    """Build a SpecSummary from whichever side of the union is available.

    Legacy-shape CRs that fail ``AIPerfSweepSpec.model_validate`` fall back to
    the archived ``aggregate_doc`` path rather than 422'ing the whole route.
    Archived docs are read via :data:`SPEC_SUMMARY_KEY` (with a
    :data:`LEGACY_SPEC_SNAPSHOT_KEY` fallback for old archives); when nothing
    usable exists the summary degrades to grid/no-dimensions.
    """
    if rec.raw_spec:
        try:
            spec = AIPerfSweepSpec.model_validate(rec.raw_spec)
            return _summary_from_snapshot(spec_summary_snapshot(spec))
        except ValueError as exc:
            # pydantic.ValidationError subclasses ValueError, but a malformed
            # distribution value makes model_validate raise a BARE ValueError.
            # `except ValidationError` alone would miss it and let it 500 the
            # summary route; catch both and fall back to the archived aggregate.
            # Only ValidationError carries structured `.errors()`.
            detail = (
                exc.errors(include_url=False)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            logger.warning(
                "AIPerfSweep %s/%s raw_spec rejected; falling back to aggregate. %s",
                rec.namespace,
                rec.name,
                detail,
            )
    if rec.aggregate_doc is not None:
        snap = _summary_snapshot_from_archive(rec, rec.aggregate_doc)
        if snap is not None:
            return _summary_from_snapshot(snap)
    return SpecSummary(
        sweep_type="grid", dimensions=[], multi_run=None, convergence=None
    )
