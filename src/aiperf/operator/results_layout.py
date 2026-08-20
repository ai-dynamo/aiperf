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
import shutil
import sqlite3
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aiperf.common.results_markers import EPOCH_RE, READY_MARKER_NAME

logger = logging.getLogger(__name__)

LATEST_POINTER = "latest.txt"
# Six digits keeps the whole-second key the same 16-digit width as the
# fractional-second key (``f"{seconds}{microsecond:06d}"``), so every emitted
# run key stays <= JS Number.MAX_SAFE_INTEGER (9_007_199_254_740_991). A wider
# suffix produced 19-digit keys (~1.7e18) that the operator UI silently rounded
# when it round-tripped ``status.runEpoch`` through a JSON number, building a
# ``/runs/<epoch>`` URL that never matched the on-disk directory.
_UID_SUFFIX_MODULUS = 1_000_000

__all__ = [
    "LATEST_POINTER",
    "RunEntry",
    "enforce_retention",
    "epoch_key_from_body",
    "is_run_ready",
    "job_dir",
    "list_run_epochs",
    "list_runs",
    "list_runs_async",
    "list_sweep_epochs",
    "list_sweep_epochs_async",
    "reconcile_latest",
    "reconcile_sweep_latest",
    "resolve_latest",
    "resolve_run_dir",
    "resolve_sweep_dir",
    "resolve_sweep_latest",
    "run_dir",
    "schedule_index_drops",
    "write_latest",
    "write_sweep_latest",
]


def _validate_epoch(epoch: str) -> None:
    """Reject any epoch that is not 9-20 decimal digits.

    Guards the latest-pointer writers against persisting an unresolvable or
    path-escaping value: ``"latest"`` (the symbolic sentinel ``resolve_run_dir``
    treats specially), ``"../escaped"`` (path traversal into a sibling dir),
    and lengths outside the legacy/uid-suffix epoch range. The repr is included
    in the message so the rejected value is visible in nested validation logs.

    Raises:
        ValueError: if ``epoch`` does not match :data:`EPOCH_RE`.
    """
    if not EPOCH_RE.match(epoch):
        raise ValueError(f"epoch must be 9-20 decimal digits, got {epoch!r}")


def _epoch_wall_seconds(epoch: str) -> int:
    """Extract the leading whole-seconds component shared by every key format.

    ``epoch_key_from_body`` emits keys of differing total widths — a 16-digit
    fractional-second key (``f"{seconds}{microsecond:06d}"``) and a 16-digit
    whole-second key carrying a 6-digit uid-derived collision suffix — but both
    forms prefix the same 10-digit epoch-seconds. The two suffix spaces overlap,
    so comparing whole keys as plain integers sorts by suffix value, not
    wall-clock: a genuinely later fractional run can look "older" than an
    earlier uid-suffixed run that happens to carry a larger suffix. Comparing
    only this leading component restores wall-clock ordering across both
    formats.
    """
    return int(epoch[:10])


def _existing_pointer_is_newer(pointer: Path, epoch: str) -> bool:
    """Return True if ``pointer`` already names a wall-clock-newer epoch.

    A delayed older completion must not roll ``latest.txt`` backward from a
    newer run. The comparison is on the leading whole-seconds component
    (:func:`_epoch_wall_seconds`) rather than the full collision-suffixed key,
    so a later fractional-second run is never mistaken for older than an
    earlier uid-suffixed one. Both the stored and candidate epochs are
    validated decimal strings here. A missing or corrupt pointer is treated as
    "not newer" so the candidate wins.
    """
    if not pointer.is_file():
        return False
    current = pointer.read_text().strip()
    if not EPOCH_RE.match(current):
        return False
    return _epoch_wall_seconds(current) > _epoch_wall_seconds(epoch)


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


def is_run_ready(run_path: Path) -> bool:
    """Return whether final run artifacts have been durably published."""
    return (run_path / READY_MARKER_NAME).is_file()


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
    runs = _walk_runs(base, namespace, name)
    _schedule_lazy_backfill_runs(base, namespace, name, runs)
    return runs


def _walk_runs(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """Pure recursive PVC walk producing newest-first :class:`RunEntry` rows.

    Split out from :func:`list_runs` so :func:`list_runs_async` can run the
    blocking ``iterdir``/``stat`` storm under ``asyncio.to_thread`` without the
    fire-and-forget ``_schedule_lazy_backfill_runs`` call — which needs a
    running loop and therefore must stay on the main event loop, not a worker
    thread. Sync callers go through :func:`list_runs`, which schedules backfill
    on the loop when one is running.
    """
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    latest = resolve_latest(base, namespace, name)
    runs: list[RunEntry] = []
    for p in parent.iterdir():
        if not p.is_dir() or not EPOCH_RE.match(p.name):
            continue
        try:
            mtime = int(p.stat().st_mtime)
            files = [
                f for f in p.iterdir() if f.is_file() and f.name != READY_MARKER_NAME
            ]
            total_size_bytes = sum(f.stat().st_size for f in files)
        except OSError:
            continue
        runs.append(
            RunEntry(
                epoch=p.name,
                mtime_epoch=mtime,
                file_count=len(files),
                total_size_bytes=total_size_bytes,
                is_latest=(p.name == latest),
            )
        )
    runs.sort(key=lambda r: r.mtime_epoch, reverse=True)
    return runs


async def list_runs_async(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """Index-first variant of :func:`list_runs` for async callers.

    Reads the SQLite ``runs_index`` first. Rows are returned directly only
    after bootstrap has proven catalog coverage and no disk publication is
    pending. Otherwise, when the job dir exists on disk, falls back to the
    legacy disk-walk and fires a ``lazy_backfill_run`` task per epoch so the
    index converges. This is the path used by the operator's FastAPI handlers.

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
    except (RuntimeError, sqlite3.DatabaseError):
        rows = []

    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []

    latest = resolve_latest(base, namespace, name)
    indexed_latest = {row.epoch for row in rows if row.is_latest}
    current_latest = {latest} if latest is not None else set()
    if (
        rows
        and indexed_latest == current_latest
        and all((parent / row.epoch).is_dir() for row in rows)
        and _runs_index.catalog_is_complete(base)
    ):
        return [
            RunEntry(
                epoch=row.epoch,
                mtime_epoch=row.mtime_epoch or 0,
                file_count=row.file_count,
                total_size_bytes=row.total_size_bytes,
                is_latest=row.is_latest,
            )
            for row in rows
        ]

    disk_runs = await asyncio.to_thread(_walk_runs, base, namespace, name)
    _schedule_lazy_backfill_runs(base, namespace, name, disk_runs)
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

    Rejects epochs that do not match :data:`EPOCH_RE` (9-20 decimal digits)
    so a symbolic value (``"latest"``), a path-traversal segment
    (``"../escaped"``), or an out-of-range length can never be persisted into
    ``latest.txt`` where ``resolve_latest`` would later hand it back to a path
    join. A delayed older completion is also ignored: if the current pointer
    already names a numerically newer epoch the write is a no-op, so a
    late-arriving stale epoch never rolls the pointer backward.

    Raises:
        ValueError: if ``epoch`` is not 9-20 decimal digits.

    Example:
        >>> write_latest(Path("/data"), "bench", "warmup-7f2a", "1714069323")
    """
    _validate_epoch(epoch)
    target = job_dir(base, namespace, name)
    if _existing_pointer_is_newer(target / LATEST_POINTER, epoch):
        return
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


def reconcile_latest(base: Path, namespace: str, name: str) -> str | None:
    """Point ``latest.txt`` at the newest surviving epoch, or remove it.

    Unlike :func:`write_latest`, reconciliation is allowed to move the pointer
    backward after retention deletes the previously latest epoch. Replacement
    selection uses the epoch key's wall-clock component, not mutable directory
    mtime. Returns the selected epoch, or ``None`` when no run directories
    survive.

    Example:
        >>> reconcile_latest(Path("/data"), "bench", "warmup-7f2a")
        '1714069323'
    """
    target = job_dir(base, namespace, name)
    pointer = target / LATEST_POINTER
    epochs = list_run_epochs(base, namespace, name)
    if not epochs:
        pointer.unlink(missing_ok=True)
        return None

    epoch = max(epochs, key=lambda value: (_epoch_wall_seconds(value), value))
    staged = target / f"{LATEST_POINTER}.tmp"
    staged.write_text(epoch)
    os.replace(staged, pointer)
    return epoch


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
    if not EPOCH_RE.match(epoch):
        return None
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
    sweep side: rejects non-:data:`EPOCH_RE` epochs and refuses to roll the
    pointer back to a numerically older epoch. Sweep-controllers call this at
    the end of each aggregate write so subsequent reads default to the
    freshest epoch.

    Raises:
        ValueError: if ``epoch`` is not 9-20 decimal digits.

    Example
    -------
    >>> write_sweep_latest(Path("/data"), "bench", "satsweep", "1714069323")
    """
    _validate_epoch(epoch)
    sweep_root = base / namespace / "sweeps" / name
    pointer = sweep_root / LATEST_POINTER
    if _existing_pointer_is_newer(pointer, epoch):
        return
    sweep_root.mkdir(parents=True, exist_ok=True)
    # Stage-then-replace like write_latest: a reader must never observe a
    # partially written pointer. The duplicate in handlers/sweep did this and
    # this one did not, so whichever writer ran was a coin flip on atomicity.
    staged = sweep_root / f"{LATEST_POINTER}.tmp"
    staged.write_text(epoch)
    os.replace(staged, pointer)


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


def reconcile_sweep_latest(base: Path, namespace: str, name: str) -> str | None:
    """Point a sweep's latest pointer at its newest surviving epoch.

    Retention is allowed to move the pointer backward after deleting the
    previously latest archive. Returns the selected epoch, or ``None`` after
    removing the pointer when no archive epochs survive.
    """
    sweep_root = base / namespace / "sweeps" / name
    pointer = sweep_root / LATEST_POINTER
    epochs = [entry.epoch for entry in list_sweep_epochs(base, namespace, name)]
    if not epochs:
        pointer.unlink(missing_ok=True)
        return None

    epoch = max(epochs, key=lambda value: (_epoch_wall_seconds(value), value))
    staged = sweep_root / f"{LATEST_POINTER}.tmp"
    staged.write_text(epoch)
    os.replace(staged, pointer)
    return epoch


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
    protect_epoch: str | None,
    retain_days: int = 0,
    dry_run: bool = False,
) -> list[str]:
    """Prune old run dirs by count and optionally age (conservative intersection).

    A run is deleted only when BOTH enabled policies agree to reap it (unless
    ``protect_epoch`` overrides):

    - Count policy keeps the ``keep`` newest by mtime.
    - Age policy keeps runs whose mtime is within ``retain_days`` days.

    ``retain_days=0`` disables the age protection, so behavior falls back to
    count-only. ``protect_epoch`` is always retained when provided, regardless
    of either policy — the active run must never be deleted out from under the
    writer.

    When ``dry_run=True``, the function performs the same policy evaluation
    and returns the list of epochs that WOULD be deleted, but touches no
    files on disk. This powers the ``aiperf kube results list-runs --preview``
    CLI flow so operators can see the reap plan before enabling retention.

    Returns the list of deleted (or would-be-deleted, if ``dry_run``) epoch
    strings. I/O failures on individual deletions are logged and swallowed
    so one corrupt dir never blocks retention on the rest.

    Pure filesystem work only — safe to offload via ``asyncio.to_thread``.
    Callers on an event loop must pass the returned epochs to
    :func:`schedule_index_drops` so the runs index converges with disk;
    scheduling cannot happen here because there is no running loop inside
    a worker thread.

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
        count_reap = child.name not in count_keepers
        age_reap = age_cutoff is None or child.stat().st_mtime < age_cutoff
        if not (count_reap and age_reap):
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
    if not _runs_index.is_open() or _runs_index.is_readonly():
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


def schedule_index_drops(namespace: str, name: str, epochs: list[str]) -> None:
    """Fire-and-forget ``runs_index.delete_run`` for retention-deleted epochs.

    Companion to :func:`enforce_retention`: the prune itself is pure
    filesystem work that completion offloads via ``asyncio.to_thread``, so
    index-drop scheduling must happen back on the event loop — inside a
    worker thread ``asyncio.get_running_loop()`` raises and the drops would
    silently be skipped.

    Example:
        >>> deleted = enforce_retention(Path("/data"), "bench", "warmup-7f2a", keep=10, protect_epoch="1714069323")
        >>> schedule_index_drops("bench", "warmup-7f2a", deleted)
    """
    for epoch in epochs:
        _schedule_index_drop(namespace, name, epoch)


def _schedule_index_drop(namespace: str, name: str, epoch: str) -> None:
    """Best-effort fire-and-forget ``runs_index.delete_run`` after retention rmtree.

    Schedule onto the running loop via ``create_task``; if there's no
    running loop (sync test or CLI dry run) we simply skip — the disk is
    the source of truth, and ``runs_index.bootstrap`` prunes rows whose run
    dir no longer exists (``_prune_stale_run_rows``) at the next operator
    startup, so a skipped or crash-lost drop re-converges then.

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
    if not _runs_index.is_open() or _runs_index.is_readonly():
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
    """Parse ``metadata.creationTimestamp`` into a decimal run key.

    Matches the legacy dynamo ``EPOCH=$(date +%s)`` convention for bodies that
    have no Kubernetes uid, preserving compatibility with already-written epoch
    directories. Fractional timestamps append six microsecond digits. Kubernetes
    whole-second timestamps append a deterministic uid-derived suffix so
    same-name resubmits created inside the same API-server second do not collide.

    Example:
        >>> epoch_key_from_body({"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}})
        '1714069323'
    """
    metadata = body["metadata"]
    ts = metadata["creationTimestamp"]
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    seconds = int(dt.timestamp())
    if dt.microsecond != 0:
        return f"{seconds}{dt.microsecond:06d}"
    uid = metadata.get("uid")
    if not uid:
        return str(seconds)
    suffix = zlib.crc32(str(uid).encode("utf-8")) % _UID_SUFFIX_MODULUS
    return f"{seconds}{suffix:06d}"


def epoch_key_seconds(epoch_key: str) -> int | None:
    """Recover the whole-second instant from a key made by ``epoch_key_from_body``.

    The inverse of the producer above, and it must stay beside it. The key is
    epoch-SECONDS, optionally carrying a six-digit suffix -- real microseconds
    for a fractional timestamp, or a uid-derived disambiguator for a
    whole-second Kubernetes one. Reading the whole string as a POSIX timestamp
    therefore over-reads a suffixed key by six orders of magnitude.

    That is not hypothetical: ``sweep_union`` did exactly that, and the first
    sweep created after suffixes were introduced made
    ``datetime.fromtimestamp(1785882875890543)`` raise "year 56594345 is out of
    range". FastAPI surfaced it as a 422 on ``GET /api/v1/sweeps``, so ONE
    unreadable directory took down the whole sweeps list -- older, unsuffixed
    sweeps included.

    Both forms stay readable, because directories written before the change are
    still on the PVC. Returns ``None`` for anything unparseable so a malformed
    directory name is skipped rather than propagated as an exception.
    """
    if not epoch_key.isdigit():
        return None
    # 10 digits covers epoch-seconds until 2286; anything longer carries the
    # six-digit suffix. Compare on length rather than magnitude so the rule is
    # the same one the producer applies.
    seconds = int(epoch_key[:-6]) if len(epoch_key) > 10 else int(epoch_key)
    # Guard the ends anyway: a hand-made directory could still be out of range.
    try:
        datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return seconds
