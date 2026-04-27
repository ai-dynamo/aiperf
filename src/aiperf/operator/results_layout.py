# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk results layout owner for the AIPerf operator.

Encapsulates the ``<base>/<namespace>/<name>/<epoch>/`` directory scheme,
the ``latest.txt`` pointer file, and retention pruning.

Run key is the decimal epoch-seconds string parsed from
``metadata.creationTimestamp`` on the AIPerfJob body, matching the
legacy dynamo ``EPOCH=$(date +%s)`` convention.

Example
-------
>>> from pathlib import Path
>>> base = Path("/data/aiperf")
>>> epoch = epoch_key_from_body({"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}})
>>> epoch
'1714069323'
>>> run_dir(base, "bench", "warmup-7f2a", epoch)
PosixPath('/data/aiperf/bench/warmup-7f2a/1714069323')
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LATEST_POINTER = "latest.txt"
# 9-11 digits covers epoch-seconds from 1973 (10^9) through 5138 (10^11),
# which comfortably brackets any realistic AIPerfJob creation timestamp.
EPOCH_RE = re.compile(r"^\d{9,11}$")

__all__ = [
    "EPOCH_RE",
    "LATEST_POINTER",
    "RunEntry",
    "enforce_retention",
    "epoch_key_from_body",
    "job_dir",
    "list_run_epochs",
    "list_runs",
    "resolve_latest",
    "resolve_run_dir",
    "resolve_sweep_dir",
    "run_dir",
    "write_latest",
]


@dataclass(slots=True)
class RunEntry:
    """One run directory with summary metadata.

    Example:
        >>> entry = RunEntry(epoch="1714150923", mtime_epoch=1714150925,
        ...                  file_count=7, total_size_bytes=4823912,
        ...                  is_latest=True)
    """

    epoch: str
    mtime_epoch: int
    file_count: int
    total_size_bytes: int
    is_latest: bool


def list_runs(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """Enumerate all run dirs under ``<base>/<ns>/<name>/``, newest first.

    Returns an empty list if no run dirs exist. The entry flagged
    ``is_latest=True`` matches ``latest.txt`` when the pointer is present
    and its target exists on disk.

    Example:
        >>> list_runs(Path("/data"), "bench", "warmup-7f2a")
        [RunEntry(epoch='1714150923', mtime_epoch=1714150925, file_count=7,
                  total_size_bytes=4823912, is_latest=True)]
    """
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    latest = resolve_latest(base, namespace, name)
    runs: list[RunEntry] = []
    for p in parent.iterdir():
        if not p.is_dir() or not EPOCH_RE.match(p.name):
            continue
        files = [f for f in p.iterdir() if f.is_file()]
        runs.append(
            RunEntry(
                epoch=p.name,
                mtime_epoch=int(p.stat().st_mtime),
                file_count=len(files),
                total_size_bytes=sum(f.stat().st_size for f in files),
                is_latest=(p.name == latest),
            )
        )
    runs.sort(key=lambda r: r.mtime_epoch, reverse=True)
    return runs


def job_dir(base: Path, namespace: str, name: str) -> Path:
    """Return ``<base>/<namespace>/<name>`` — the per-job root.

    Example:
        >>> job_dir(Path("/data"), "bench", "warmup-7f2a")
        PosixPath('/data/bench/warmup-7f2a')
    """
    return Path(base) / namespace / name


def run_dir(base: Path, namespace: str, name: str, epoch: str) -> Path:
    """Return ``<base>/<namespace>/<name>/<epoch>`` — one benchmark run.

    Example:
        >>> run_dir(Path("/data"), "bench", "warmup-7f2a", "1714069323")
        PosixPath('/data/bench/warmup-7f2a/1714069323')
    """
    return job_dir(base, namespace, name) / epoch


def write_latest(base: Path, namespace: str, name: str, epoch: str) -> None:
    """Atomically record ``epoch`` as the current run for a job.

    Writes to ``<job_dir>/latest.txt.tmp`` first then ``os.replace`` onto
    the final path so concurrent readers never observe a partial write.

    Example:
        >>> write_latest(Path("/data"), "bench", "warmup-7f2a", "1714069323")
    """
    target = job_dir(base, namespace, name)
    target.mkdir(parents=True, exist_ok=True)
    pointer = target / LATEST_POINTER
    staged = target / f"{LATEST_POINTER}.tmp"
    staged.write_text(epoch)
    os.replace(staged, pointer)


def resolve_latest(base: Path, namespace: str, name: str) -> str | None:
    """Return the epoch recorded in ``latest.txt`` or ``None`` if absent.

    Example:
        >>> resolve_latest(Path("/data"), "bench", "warmup-7f2a")
        '1714069323'
    """
    pointer = job_dir(base, namespace, name) / LATEST_POINTER
    if not pointer.is_file():
        return None
    value = pointer.read_text().strip()
    return value or None


def resolve_run_dir(
    base: Path,
    namespace: str,
    name: str,
    epoch: str | None = None,
) -> Path | None:
    """Resolve a run directory, defaulting to the latest-pointer target.

    If ``epoch`` is ``None`` or ``"latest"``, reads ``latest.txt`` to
    pick the run. Returns ``None`` when the resolved directory does not
    exist on disk — callers should treat this as "no results yet".

    Example:
        >>> resolve_run_dir(Path("/data"), "bench", "warmup-7f2a")
        PosixPath('/data/bench/warmup-7f2a/1714069323')
    """
    if epoch is None or epoch == "latest":
        resolved = resolve_latest(base, namespace, name)
        if resolved is None:
            return None
        epoch = resolved
    candidate = run_dir(base, namespace, name, epoch)
    if not candidate.is_dir():
        return None
    return candidate


def resolve_sweep_dir(base: Path, namespace: str, name: str) -> Path | None:
    """Return the persisted sweep directory ``<base>/<ns>/sweeps/<name>``, or None.

    The directory is the durable parent manifest location for AIPerfSweep CRs:
    sweep-controllers write ``aggregate.json`` + ``conditions.json`` here at
    terminal phase. The dual-backed sweep API uses this to render archived
    sweeps after the parent CR has been TTL-reaped.

    Example
    -------
    >>> resolve_sweep_dir(Path("/data"), "bench", "saturation-sweep")
    PosixPath('/data/bench/sweeps/saturation-sweep')
    """
    candidate = base / namespace / "sweeps" / name
    if not candidate.is_dir():
        return None
    return candidate


def list_run_epochs(base: Path, namespace: str, name: str) -> list[str]:
    """Return every epoch-shaped subdirectory of the job dir.

    Example:
        >>> list_run_epochs(Path("/data"), "bench", "warmup-7f2a")
        ['1714064523', '1714069323', '1714150923']
    """
    root = job_dir(base, namespace, name)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and EPOCH_RE.match(child.name)
    )


def enforce_retention(
    base: Path,
    namespace: str,
    name: str,
    *,
    keep: int,
    protect_epoch: str,
    retain_days: int = 0,
    dry_run: bool = False,
) -> list[str]:
    """Prune old run dirs by count and optionally age (conservative intersection).

    A run is kept iff BOTH policies agree to keep it (or ``protect_epoch``
    overrides):

    - Count policy keeps the ``keep`` newest by mtime.
    - Age policy keeps runs whose mtime is within ``retain_days`` days.

    A run is deleted only when BOTH policies would reap it. ``retain_days=0``
    disables the age policy (treated as "always keep"), so behavior falls
    back to count-only. ``protect_epoch`` is always retained regardless of
    either policy — the active run must never be deleted out from under
    the writer.

    When ``dry_run=True``, the function performs the same policy evaluation
    and returns the list of epochs that WOULD be deleted, but touches no
    files on disk. This powers the ``aiperf kube results list-runs --preview``
    CLI flow so operators can see the reap plan before enabling retention.

    Returns the list of deleted (or would-be-deleted, if ``dry_run``) epoch
    strings. I/O failures on individual deletions are logged and swallowed
    so one corrupt dir never blocks retention on the rest.

    Example:
        >>> enforce_retention(Path("/data"), "bench", "warmup-7f2a", keep=10, protect_epoch="1714069323")
        ['1714000000', '1714000060']
        >>> enforce_retention(Path("/data"), "bench", "warmup-7f2a", keep=10, protect_epoch="1714069323", dry_run=True)
        ['1714000000', '1714000060']
    """
    root = job_dir(base, namespace, name)
    if not root.is_dir():
        return []

    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and EPOCH_RE.match(child.name)
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    count_keepers = {p.name for p in candidates[:keep]}
    age_cutoff = time.time() - retain_days * 86400 if retain_days > 0 else None

    deleted: list[str] = []
    for child in candidates:
        if child.name == protect_epoch:
            continue
        count_keep = child.name in count_keepers
        age_keep = age_cutoff is None or child.stat().st_mtime >= age_cutoff
        if count_keep and age_keep:
            continue
        if dry_run:
            deleted.append(child.name)
            continue
        try:
            shutil.rmtree(child)
            deleted.append(child.name)
        except OSError as exc:
            logger.warning(
                "retention: failed to remove %s/%s/%s: %s",
                namespace,
                name,
                child.name,
                exc,
            )
    return deleted


def epoch_key_from_body(body: dict) -> str:
    """Parse ``metadata.creationTimestamp`` into a decimal epoch-seconds string.

    Matches the legacy dynamo ``EPOCH=$(date +%s)`` convention so run
    directories sort chronologically and collide only when two jobs are
    created in the same second (rare in practice, and safe because
    Kubernetes names are also unique per-job).

    Example:
        >>> epoch_key_from_body({"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}})
        '1714069323'
    """
    ts = body["metadata"]["creationTimestamp"]
    # ``fromisoformat`` on 3.10 rejects trailing Z; swap for the explicit
    # UTC offset so the parse works across supported Python versions.
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return str(int(dt.timestamp()))
