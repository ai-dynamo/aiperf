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

from aiperf.common.endpoint_credentials import (
    redact_sweep_display_label,
    redact_sweep_public_data,
)
from aiperf.kubernetes.crd_models import AIPerfSweepSpec
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


def _dimension_info(path: str, values: list[Any]) -> DimensionInfo:
    return DimensionInfo(
        name=_dimension_display_name(path),
        values=[redact_sweep_public_data(value, path=path) for value in values],
    )


def _scenario_display_name(run: Any, index: int) -> str | int:
    if not isinstance(run, dict):
        return index
    name = run.get("name")
    return redact_sweep_display_label(name) if isinstance(name, str) else index


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
            _dimension_info(name, list(values))
            for name, values in sweep.parameters.items()
        ]
    if isinstance(sweep, AdaptiveSearchSweep):
        return [
            _dimension_info(dim.path, [dim.lo, dim.hi]) for dim in sweep.search_space
        ]
    if isinstance(sweep, (SobolSweep, LatinHypercubeSweep)):
        return [
            _dimension_info(
                dim.path,
                list(dim.choices) if dim.choices is not None else [dim.lo, dim.hi],
            )
            for dim in sweep.dimensions
        ]
    if isinstance(sweep, ScenarioSweep):
        return [
            DimensionInfo(
                name="scenario",
                values=[
                    _scenario_display_name(run, idx)
                    for idx, run in enumerate(sweep.runs)
                ],
            )
        ]
    return []


def _dump_list(value: Any) -> list[dict[str, Any]] | None:
    """Dump an optional list of pydantic models to JSON-able dicts.

    ``objectives`` and ``sla_filters`` only exist on the adaptive-search side of
    the sweep union, so every other generator type returns ``None`` here.
    """
    if not value:
        return None
    return [item.model_dump(mode="json", by_alias=True) for item in value]


def _snapshot_from_parts(sweep: Any, multi_run: Any) -> dict[str, Any]:
    """Build the snapshot dict from validated ``sweep`` + ``multi_run`` models."""
    return redact_sweep_public_data(
        {
            "sweep_type": str(sweep.type),
            "dimensions": [
                dim.model_dump(mode="json")
                for dim in dimensions_from_sweep_model(sweep)
            ],
            "multi_run": multi_run.model_dump(mode="json", by_alias=True),
            "convergence": (
                multi_run.convergence.model_dump(mode="json", by_alias=True)
                if multi_run.convergence is not None
                else None
            ),
            # Carried so the UI can name the winner by the sweep's own objective and
            # exclude SLA-infeasible variations, instead of ranking by whichever
            # metric the chart selector happens to be on.
            "objectives": _dump_list(getattr(sweep, "objectives", None)),
            "sla_filters": _dump_list(getattr(sweep, "sla_filters", None)),
        }
    )


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
    snap = redact_sweep_public_data(snap)
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
        # Absent from archives written before these were snapshotted; the UI
        # falls back to its metric-ranked winner when they are None.
        objectives=snap.get("objectives"),
        sla_filters=snap.get("sla_filters"),
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


# Keys that were added to the snapshot AFTER the first archives were written.
# A snapshot produced by an older sweep-controller is structurally valid and
# non-empty but silently missing these, so preferring it wholesale strands the
# very fields the UI needs to name a winner.
_LATE_ADDED_SNAPSHOT_KEYS = ("objectives", "sla_filters")


def _backfill_late_added_keys(
    rec: Any, snap: dict[str, Any], legacy: Any
) -> dict[str, Any]:
    """Fill ``objectives`` / ``sla_filters`` from the legacy full-spec dump.

    The two keys were added to :func:`spec_summary_snapshot` after archives had
    already been written, so a real adaptive-search archive can carry a
    four-key ``specSummary`` (``sweep_type`` / ``dimensions`` / ``multi_run`` /
    ``convergence``) alongside a ``specSnapshot`` whose ``sweep`` block still
    holds the full ``objectives`` and ``sla_filters``. Preferring the newer key
    unconditionally therefore returned ``None`` for both on exactly the sweeps
    that declare an objective, and the UI fell back to ranking by whichever
    metric its chart selector was on -- the bug these fields exist to remove.

    Only the missing keys are copied over: everything else on the purpose-built
    snapshot stays authoritative, because it was written by the producer that
    owns the contract.
    """
    if all(snap.get(key) for key in _LATE_ADDED_SNAPSHOT_KEYS):
        return snap
    if not isinstance(legacy, dict) or not legacy:
        return snap
    derived = _summary_from_legacy_spec_dump(rec, legacy)
    if derived is None:
        return snap
    merged = dict(snap)
    for key in _LATE_ADDED_SNAPSHOT_KEYS:
        if not merged.get(key) and derived.get(key):
            merged[key] = derived[key]
    return merged


def _summary_snapshot_from_archive(
    rec: Any, aggregate_doc: dict[str, Any]
) -> dict[str, Any] | None:
    """Extract a snapshot-shaped dict from an archived ``aggregate.json``.

    Tries the purpose-built :data:`SPEC_SUMMARY_KEY` first; old archives that
    predate it carry only the full spec dump under
    :data:`LEGACY_SPEC_SNAPSHOT_KEY`, which is summarized via
    :func:`_summary_from_legacy_spec_dump`. Both keys are usually present at
    once, so a snapshot written before ``objectives`` / ``sla_filters`` existed
    is topped up from the legacy dump rather than accepted as-is -- see
    :func:`_backfill_late_added_keys`. Returns ``None`` when neither key yields
    a usable summary.
    """
    snap = aggregate_doc.get(SPEC_SUMMARY_KEY)
    legacy = aggregate_doc.get(LEGACY_SPEC_SNAPSHOT_KEY)
    if isinstance(snap, dict) and snap:
        return _backfill_late_added_keys(rec, snap, legacy)
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
