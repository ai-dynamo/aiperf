# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize ``artifacts.user_files`` inside a Kubernetes service container.

Local runs materialize user files from ``ArtifactDirResolver``, which is part of
the pre-bootstrap resolver chain. Kubernetes service containers deliberately
skip that chain (``aiperf service`` boots from the controller-rendered
``BenchmarkRun`` so seeds, synthesized defaults, and artifact identity are
byte-identical across pods), so this module renders the same files from the
serialized run instead of re-resolving anything.

Only the container that owns the run directory calls this. Worker pods mount
their own private ``/results`` ``emptyDir``; letting every service container
write would produce N unharvested copies of each file.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from aiperf.config.user_files import (
    RunMeta,
    build_user_file_context,
    materialize_user_files,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

logger = logging.getLogger(__name__)

__all__ = ["materialize_serialized_run_user_files", "resolve_run_meta"]


def resolve_run_meta(run: BenchmarkRun) -> RunMeta:
    """Return the run identity to render ``artifacts.user_files`` against.

    Prefers the ``RunMeta`` the launcher froze into the serialized run: in
    operator mode that carries the AIPerfJob name and the epoch key of the PVC
    directory the artifacts land in. Deriving it in-pod is not an option because
    every container's ``artifact_dir`` is the fixed ``/results`` mount, whose
    basename would render ``job_name`` as the literal ``"results"``.

    The fallback covers direct mode (``--no-operator``), which has no CR and so
    no epoch: it uses the pod's ``AIPERF_JOB_ID`` / ``AIPERF_NAMESPACE`` and
    wall-clock seconds, matching what the local path does for a non-epoch
    artifact dir.
    """
    if run.run_meta is not None:
        return run.run_meta
    return RunMeta(
        epoch=str(int(time.time())),
        job_name=os.environ.get("AIPERF_JOB_ID") or run.benchmark_id,
        namespace=os.environ.get("AIPERF_NAMESPACE", ""),
    )


def materialize_serialized_run_user_files(run: BenchmarkRun) -> None:
    """Render and write ``run.cfg.artifacts.user_files`` into the run directory.

    A no-op when the run declares no user files. Raises ``UserFileError`` on a
    render, path-escape, or write failure so the container exits before the
    benchmark starts — the same fatal semantics the local path has.
    """
    files = run.cfg.artifacts.user_files
    if not files:
        return

    run_dir = run.artifact_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    context = build_user_file_context(
        run.cfg,
        resolve_run_meta(run),
        run_dir=run_dir,
        variables=run.variables,
    )
    materialize_user_files(files, run_dir=run_dir, context=context)
    logger.info(
        "Materialized %d artifacts.user_files entr%s into %s",
        len(files),
        "y" if len(files) == 1 else "ies",
        run_dir,
    )
