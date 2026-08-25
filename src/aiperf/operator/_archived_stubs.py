# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub builders for incomplete archived runs.

A historical epoch directory may exist on disk (and be enumerated by
``GET /api/v1/jobs/{ns}/{name}/epochs``) before any
``profile_export_aiperf.json`` is written — typically a run that failed or
was cancelled mid-flight. The full ``_archived_from_summary`` path can't be
used because it requires a parsed summary; this module supplies a minimal
``AIPerfJobInfo`` that ``find_any_job`` returns instead of ``None``, so the
run-detail page renders an "Unknown / archived" stub instead of 404ing the
SPA into the "Operator API unreachable" banner.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from aiperf.kubernetes.models import AIPerfJobInfo


def archived_stub(
    namespace: str,
    name: str,
    *,
    run_dir: Path,
    name_dir: Path,
) -> AIPerfJobInfo:
    """Build a minimal archived ``AIPerfJobInfo`` for a run dir with no summary.

    ``name_dir`` is the per-name parent (one above the epoch dir); the sweep
    linkage marker — when present — is read from there so child jobs of a
    sweep keep their breadcrumb wiring even when the controller never wrote
    a summary. ``run_dir.stat().st_mtime`` populates ``created`` for
    deterministic ordering on archived list pages.
    """
    # Defer the marker reader import so this module stays a leaf and does
    # not pull job_union back into its own import graph.
    from aiperf.operator.job_union import _sweep_linkage_from_marker

    mtime_iso = (
        _dt.datetime.fromtimestamp(run_dir.stat().st_mtime, tz=_dt.UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    sweep_name, variation_index, variation_label = _sweep_linkage_from_marker(name_dir)
    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase="Unknown",
        job_id=name,
        created=mtime_iso,
        source="archived",
        sweep_name=sweep_name,
        variation_index=variation_index,
        variation_label=variation_label,
    )
