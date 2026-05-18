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

import asyncio
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
# 9-17 digits covers legacy epoch-seconds directories plus fractional-second
# run keys encoded as <epoch-seconds><microseconds:06d> for same-name resubmits.
EPOCH_RE = re.compile(r"^\d{9,17}$")

__all__ = [
    "EPOCH_RE",
    "LATEST_POINTER",
    "RunEntry",
    "enforce_retention",
    "epoch_key_from_body",
    "job_dir",
    "list_run_epochs",
    "list_runs",
    "list_runs_async",
    "list_sweep_epochs",
    "list_sweep_epochs_async",
    "resolve_latest",
    "resolve_run_dir",
    "resolve_sweep_dir",
    "resolve_sweep_latest",
    "run_dir",
    "write_latest",
    "write_sweep_latest",
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

    When called from an async context (a running loop is detected), each
    discovered run epoch is also handed off to ``runs_index.lazy_backfill_run``
    via ``asyncio.create_task`` so the SQLite index converges on the disk
    truth without blocking this read. Pure-sync callers (CLI, retention)
    skip the backfill — the next operator restart's bootstrap pass picks
    up the leftover.

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
    _schedule_lazy_backfill_runs(base, namespace, name, runs)
    return runs


async def list_runs_async(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """Index-first variant of :func:`list_runs` for async callers.

    Reads the SQLite ``runs_index`` first; on a hit, the index rows are
    returned directly without touching the PVC. On a miss (empty rows)
    AND when the job dir exists on disk, falls back to the legacy
    disk-walk and fires a ``lazy_backfill_run`` task per epoch so the
    next call lands in the index. This is the path used by the
    operator's FastAPI handlers.

    When ``runs_index.open()`` has not been called (unit tests, results
    sidecar processes that don't manage the index) the function silently
    falls through to the disk walk — the index is treated as a cache,
    not a hard dependency.

    Example:
        >>> entries = await list_runs_async(Path("/data"), "bench", "warmup-7f2a")
    """
    from aiperf.operator import runs_index as _runs_index

    try:
        rows = await _runs_index.list_runs_for_job(namespace, name)
    except RuntimeError:
        rows = []

    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []

    disk_runs = list_runs(base, namespace, name)
    if not rows:
        return disk_runs

    combined: dict[str, RunEntry] = {}
    for r in rows:
        if not (parent / r.epoch).is_dir():
            continue
        combined[r.epoch] = RunEntry(
            epoch=r.epoch,
            mtime_epoch=r.mtime_epoch or 0,
            file_count=r.file_count,
            total_size_bytes=r.total_size_bytes,
            is_latest=r.is_latest,
        )
    for entry in disk_runs:
        combined[entry.epoch] = entry
    return sorted(combined.values(), key=lambda r: r.mtime_epoch, reverse=True)


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


def resolve_sweep_dir(
    base: Path, namespace: str, name: str, *, epoch: str | None = None
) -> Path | None:
    """Return ``<base>/<ns>/sweeps/<name>/<epoch>/`` or fall through to ``latest.txt``.

    Mirrors :func:`resolve_run_dir` for sweeps. When ``epoch`` is omitted, the
    sweep's ``latest.txt`` pointer is consulted; if that file is absent or
    points at a non-existent epoch dir, ``None`` is returned. The ``epoch``
    string must match :data:`EPOCH_RE` — out-of-shape values yield ``None``
    rather than raising, matching the dual-backed sweep API's tolerant
    "no results yet" semantics.

    Example
    -------
    >>> resolve_sweep_dir(Path("/data"), "bench", "satsweep", epoch="1714069323")
    PosixPath('/data/bench/sweeps/satsweep/1714069323')
    """
    sweep_root = base / namespace / "sweeps" / name
    if not sweep_root.is_dir():
        return None
    if epoch is None:
        epoch = resolve_sweep_latest(base, namespace, name)
        if epoch is None:
            return None
    if not EPOCH_RE.match(epoch):
        return None
    candidate = sweep_root / epoch
    return candidate if candidate.is_dir() else None


def write_sweep_latest(base: Path, namespace: str, name: str, epoch: str) -> None:
    """Persist ``<base>/<ns>/sweeps/<name>/latest.txt`` with the given epoch.

    Creates the sweep root if absent. Mirrors :func:`write_latest` for the
    sweep side; sweep-controllers call this at the end of each aggregate
    write so subsequent reads default to the freshest epoch.

    Example
    -------
    >>> write_sweep_latest(Path("/data"), "bench", "satsweep", "1714069323")
    """
    sweep_root = base / namespace / "sweeps" / name
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / LATEST_POINTER).write_text(epoch)


def resolve_sweep_latest(base: Path, namespace: str, name: str) -> str | None:
    """Read ``<base>/<ns>/sweeps/<name>/latest.txt`` or return ``None``.

    Returns ``None`` when the pointer file is absent or its contents do not
    match :data:`EPOCH_RE` — corrupt pointer files are treated as "no
    latest known" rather than propagated as garbage.

    Example
    -------
    >>> resolve_sweep_latest(Path("/data"), "bench", "satsweep")
    '1714069323'
    """
    pointer = base / namespace / "sweeps" / name / LATEST_POINTER
    if not pointer.is_file():
        return None
    epoch = pointer.read_text().strip()
    return epoch if EPOCH_RE.match(epoch) else None


def list_sweep_epochs(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """List sweep epochs under ``<base>/<ns>/sweeps/<name>/``, ascending by epoch.

    Each entry carries its own ``is_latest`` flag, determined against
    ``latest.txt``. ``file_count`` is the count of immediate children under
    the epoch dir (children.json + aggregate.json + conditions.json + ...);
    ``total_size_bytes`` sums regular-file sizes for symmetry with
    :func:`list_runs`. Directories whose stat fails (permission, race) are
    silently skipped — no partial entry leaks back to the caller.

    Example
    -------
    >>> list_sweep_epochs(Path("/data"), "bench", "satsweep")
    [RunEntry(epoch='1714069323', mtime_epoch=1714069324, file_count=3,
              total_size_bytes=8421, is_latest=True)]
    """
    sweep_root = base / namespace / "sweeps" / name
    if not sweep_root.is_dir():
        return []
    latest = resolve_sweep_latest(base, namespace, name)
    out: list[RunEntry] = []
    for p in sweep_root.iterdir():
        if not p.is_dir() or not EPOCH_RE.match(p.name):
            continue
        try:
            mtime = int(p.stat().st_mtime)
            children = list(p.iterdir())
            file_count = len(children)
            total_size_bytes = sum(c.stat().st_size for c in children if c.is_file())
        except OSError:
            continue
        out.append(
            RunEntry(
                epoch=p.name,
                mtime_epoch=mtime,
                file_count=file_count,
                total_size_bytes=total_size_bytes,
                is_latest=(p.name == latest),
            )
        )
    return sorted(out, key=lambda e: e.epoch)


async def list_sweep_epochs_async(
    base: Path, namespace: str, name: str
) -> list[RunEntry]:
    """Index-first variant of :func:`list_sweep_epochs` for async callers.

    Reads distinct ``sweep_epoch`` values from the SQLite index first;
    on a hit, fills :class:`RunEntry` rows from disk stats (the index
    only tracks per-variation rows, not aggregate file counts). On a
    miss falls back to the legacy disk-walk via :func:`list_sweep_epochs`.

    Example:
        >>> entries = await list_sweep_epochs_async(Path("/data"), "bench", "satsweep")
    """
    from aiperf.operator import runs_index as _runs_index

    try:
        epochs = await _runs_index.list_sweep_epochs_for_sweep(namespace, name)
    except RuntimeError:
        epochs = []

    disk_epochs = list_sweep_epochs(base, namespace, name)
    if not epochs:
        return disk_epochs

    by_epoch = {entry.epoch: entry for entry in disk_epochs}
    sweep_root = base / namespace / "sweeps" / name
    latest = resolve_sweep_latest(base, namespace, name)
    for epoch in epochs:
        if epoch in by_epoch:
            continue
        epoch_dir = sweep_root / epoch
        if not epoch_dir.is_dir():
            continue
        try:
            mtime = int(epoch_dir.stat().st_mtime)
            children = list(epoch_dir.iterdir())
            file_count = len(children)
            total_size_bytes = sum(c.stat().st_size for c in children if c.is_file())
        except OSError:
            continue
        by_epoch[epoch] = RunEntry(
            epoch=epoch,
            mtime_epoch=mtime,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            is_latest=(epoch == latest),
        )
    return sorted(by_epoch.values(), key=lambda e: e.epoch)


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
            _schedule_index_drop(namespace, name, child.name)
        except OSError as exc:
            logger.warning(
                "retention: failed to remove %s/%s/%s: %s",
                namespace,
                name,
                child.name,
                exc,
            )
    return deleted


def _schedule_lazy_backfill_runs(
    base: Path, namespace: str, name: str, runs: list[RunEntry]
) -> None:
    """Best-effort fire-and-forget ``runs_index.lazy_backfill_run`` per epoch.

    Called from sync :func:`list_runs` so async callers (FastAPI handlers
    wrapped via ``asyncio.to_thread``, asyncio test loops) get the SQLite
    index converged toward disk truth without blocking the read. When no
    loop is running (pure-sync CLI / retention path), silently skip — the
    operator's startup ``runs_index.bootstrap`` covers that gap.

    Imported lazily to keep ``results_layout`` import-cycle-free; the
    operator package re-exports ``runs_index`` so a lazy attribute load
    is the cheapest way to avoid a top-level circular import.
    """
    if not runs:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        from aiperf.operator import runs_index as _runs_index
    except ImportError as exc:  # pragma: no cover - defensive
        logger.warning("runs_index unavailable for lazy backfill: %s", exc)
        return
    for entry in runs:
        try:
            loop.create_task(
                _runs_index.lazy_backfill_run(base, namespace, name, entry.epoch)
            )
        except Exception as exc:  # noqa: BLE001 - index path must never break reads
            logger.warning(
                "runs_index.lazy_backfill_run task failed for %s/%s/%s: %s",
                namespace,
                name,
                entry.epoch,
                exc,
            )


def _schedule_index_drop(namespace: str, name: str, epoch: str) -> None:
    """Best-effort fire-and-forget ``runs_index.delete_run`` after retention rmtree.

    ``enforce_retention`` is sync (called from the sync helper in
    ``handlers/completion._run_retention_pass`` inside an async kopf
    handler) so we cannot ``await``. Schedule onto the running loop via
    ``create_task``; if there's no running loop (sync test or CLI dry
    run) we simply skip — the disk is the source of truth and the next
    bootstrap pass will re-converge the index.

    Imported lazily to keep ``results_layout`` import-cycle-free; the
    operator package re-exports ``runs_index`` so a lazy attribute load
    is the cheapest way to avoid a top-level circular import.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        from aiperf.operator import runs_index as _runs_index
    except ImportError as exc:  # pragma: no cover - defensive
        logger.warning("runs_index unavailable for retention drop: %s", exc)
        return
    try:
        loop.create_task(_runs_index.delete_run(namespace, name, epoch))
    except Exception as exc:  # noqa: BLE001 - index path must never break retention
        logger.warning(
            "runs_index.delete_run task failed during retention for %s/%s/%s: %s",
            namespace,
            name,
            epoch,
            exc,
        )


def epoch_key_from_body(body: dict) -> str:
    """Parse ``metadata.creationTimestamp`` into a decimal epoch-seconds string.

    Matches the legacy dynamo ``EPOCH=$(date +%s)`` convention so run
    directories sort chronologically. Whole-second timestamps keep the legacy
    epoch-seconds shape; fractional timestamps append six microsecond digits so
    same-name resubmits created within one second do not collide.

    Example:
        >>> epoch_key_from_body({"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}})
        '1714069323'
    """
    ts = body["metadata"]["creationTimestamp"]
    # ``fromisoformat`` on 3.10 rejects trailing Z; swap for the explicit
    # UTC offset so the parse works across supported Python versions.
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    seconds = int(dt.timestamp())
    if dt.microsecond == 0:
        return str(seconds)
    return f"{seconds}{dt.microsecond:06d}"
