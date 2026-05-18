# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-level aggregate JSON writer.

The sweep-controller calls :func:`write_sweep_aggregate` exactly once when the
parent ``AIPerfSweep`` enters a terminal phase. It writes
``<base>/<ns>/sweeps/<name>/<sweep_run_epoch>/aggregate.json`` (and optionally
``conditions.json``) atomically via a sibling ``.tmp`` + ``os.replace`` so a
torn read on the operator HTTP API side surfaces as ``JSONDecodeError`` rather
than a half-decoded dict. ``latest.txt`` is updated *last* so a partial-state
crash never shadows a prior good epoch.

This file is the durable anchor of the dual-backed sweep API: the operator
reads from the CR while live and from the per-epoch directory once the sweep
has finished and the controller pod is gone.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import orjson

__all__ = ["write_children_manifest", "write_sweep_aggregate"]


def write_sweep_aggregate(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
    doc: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
    update_latest: bool = True,
) -> None:
    """Atomic write of ``<base>/<ns>/sweeps/<name>/<epoch>/{aggregate,conditions}.json``.

    By default, writes ``latest.txt`` last so a torn read on the operator side
    sees the prior epoch (or nothing) but never a half-written current epoch.
    Callers that write additional required siblings after ``aggregate.json`` can
    pass ``update_latest=False`` and advance the pointer after their full bundle
    is durable. ``conditions.json`` is only written when ``conditions`` is not
    ``None`` (callers that have not yet collected the conditions list pass
    ``None`` and the file is omitted).

    Args:
        base_dir: Results-server root, typically ``/results``.
        namespace: Parent sweep namespace.
        sweep_name: Parent sweep name.
        sweep_run_epoch: Sweep epoch identifier (decimal seconds since epoch).
        doc: Pre-assembled aggregate document. Shape is owned by the caller —
            this writer is intentionally schema-agnostic.
        conditions: Optional list of CR-style condition dicts; wrapped under a
            ``{"conditions": [...]}`` envelope on disk.
        update_latest: Whether to advance ``latest.txt`` after this writer's own
            files are durable.
    """
    from aiperf.operator.results_layout import write_sweep_latest

    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name / sweep_run_epoch
    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target_dir / "aggregate.json", doc)
    if conditions is not None:
        _atomic_write_json(target_dir / "conditions.json", {"conditions": conditions})
    if update_latest:
        write_sweep_latest(Path(base_dir), namespace, sweep_name, sweep_run_epoch)


def write_children_manifest(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
    children: list[dict[str, Any]],
) -> None:
    """Atomic write of ``<base>/<ns>/sweeps/<name>/<epoch>/children.json``.

    The manifest is the authoritative ``(epoch -> child name + child epoch)``
    linkage. Read by ``sweep_union`` to resolve archived sweeps after the
    parent CR has been TTL-reaped. Children list is sorted by
    ``variation_index`` then ``trial_index`` for deterministic diffs;
    ``trial_index=None`` sorts before any explicit trial.
    """
    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name / sweep_run_epoch
    target_dir.mkdir(parents=True, exist_ok=True)
    sorted_children = sorted(
        children,
        key=lambda c: (
            int(c.get("variation_index") or 0),
            int(c.get("trial_index") or 0) if c.get("trial_index") is not None else -1,
        ),
    )
    payload = {
        "sweep_run_epoch": sweep_run_epoch,
        "children": sorted_children,
    }
    _atomic_write_json(target_dir / "children.json", payload)


def _atomic_write_json(path: Path, payload: Any) -> None:
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
