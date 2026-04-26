# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-level aggregate JSON writer.

The sweep-controller calls :func:`write_sweep_aggregate` exactly once when the
parent ``AIPerfSweep`` enters a terminal phase. It writes
``<base>/<ns>/sweeps/<name>/aggregate.json`` (and optionally
``conditions.json``) atomically via a sibling ``.tmp`` + ``os.replace`` so a
torn read on the operator HTTP API side surfaces as ``JSONDecodeError`` rather
than a half-decoded dict.

This file is the durable anchor of the dual-backed sweep API: the operator
reads from the CR while live and from this file once the sweep has finished
and the controller pod is gone.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import orjson

__all__ = ["write_sweep_aggregate"]


def write_sweep_aggregate(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    doc: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
) -> None:
    """Atomic write of ``<base>/<ns>/sweeps/<name>/{aggregate.json,conditions.json}``.

    Called by the sweep-controller exactly once when the parent enters a
    terminal phase. Uses ``*.tmp`` + ``os.replace`` so a torn read on the
    operator side surfaces as ``JSONDecodeError`` rather than a half-decoded
    dict. ``conditions.json`` is only written when ``conditions`` is not
    ``None`` (callers that have not yet collected the conditions list pass
    ``None`` and the file is omitted).

    Args:
        base_dir: Results-server root, typically ``/results``.
        namespace: Parent sweep namespace.
        sweep_name: Parent sweep name.
        doc: Pre-assembled aggregate document. Shape is owned by the caller —
            this writer is intentionally schema-agnostic.
        conditions: Optional list of CR-style condition dicts; wrapped under a
            ``{"conditions": [...]}`` envelope on disk.
    """
    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target_dir / "aggregate.json", doc)
    if conditions is not None:
        _atomic_write_json(target_dir / "conditions.json", {"conditions": conditions})


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
