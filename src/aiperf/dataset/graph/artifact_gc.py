# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reclaim graph build artifacts orphaned by a run that died abruptly.

A graph build writes two multi-GB dirs under the mmap base path (the system
temp dir by default, which is a RAM-backed tmpfs on many hosts):
``aiperf_graph_segments_<id>/`` and ``aiperf_graph_meta_<id>/``.
``DatasetManager._cleanup`` reclaims them on the stop path, but that path does
not run when the process dies abruptly -- the ``os._exit`` force-kill in
``cli_runner``, a SIGKILL from the service manager, or a hard crash. Those runs
leave the dirs behind, and nothing else ever removes them.

An age-only sweep cannot fix that safely: the dirs are keyed by ``benchmark_id``
rather than pid, so "old" says nothing about whether a long-running concurrent
benchmark is still reading one. The owner lock supplies the missing liveness
signal. The OS drops a file lock when the holding process dies by ANY means,
including SIGKILL and ``os._exit``, so:

- lock file present and ACQUIRABLE -> its run is gone; the dir is an orphan
- lock file present and CONTENDED  -> a live run owns the dir; leave it
- no lock file yet                 -> the run is between ``mkdir`` and its
  first acquire, so fall back to an age grace and only reclaim a dir old
  enough that no starting run could still be in that window

On a filesystem without file-lock support (some NFS and FUSE mounts) liveness
is unprovable, so both entry points fail closed: nothing is reclaimed and the
dir leaks rather than risking deletion out from under a live run.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import time
from pathlib import Path

import orjson
from filelock import FileLock, Timeout

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)

GRAPH_ARTIFACT_DIR_PREFIXES = ("aiperf_graph_segments_", "aiperf_graph_meta_")
"""Directory-name prefixes a graph build creates under the mmap base path."""

OWNER_LOCK_SUFFIX = ".aiperf-owner.lock"
"""Suffix for sibling lock files held for the owning run's lifetime."""

OWNER_IDENTITY_FILENAME = ".aiperf-owner.json"
"""Which host owns the dir, so a peer sharing the volume never reclaims it."""

FOREIGN_HOST_GRACE_SECONDS = 7 * 24 * 3600.0
"""Age at which another host's dir is reclaimed anyway.

The only signal available across hosts is age, so this is set far beyond any
plausible benchmark rather than tuned: it exists so a decommissioned peer's
artifacts cannot accumulate forever, not to reclaim promptly.
"""

ORPHAN_GRACE_SECONDS = 300.0
"""How old a lock-less artifact dir must be before it counts as an orphan.

Covers only the ``mkdir``-to-first-acquire window, which is milliseconds in
practice; the generous default keeps a stalled or swapped-out starting run
from having its dir swept out from under it.
"""


def owner_lock_path(artifact_dir: Path) -> Path:
    """The sibling owner-lock path for one artifact dir.

    Windows denies deletion of a directory containing an open lock file, so
    the lock must live outside the directory it protects.
    """
    artifact = Path(artifact_dir)
    return artifact.parent / f".{artifact.name}{OWNER_LOCK_SUFFIX}"


def owner_identity_path(artifact_dir: Path) -> Path:
    """The owner-identity path for one artifact dir."""
    return Path(artifact_dir) / OWNER_IDENTITY_FILENAME


def current_boot_id() -> str | None:
    """An identifier for the running boot, or ``None`` when unavailable.

    ``psutil.boot_time`` is already a dependency and works on every platform
    AIPerf supports. Rounded to whole seconds because some platforms derive it
    from a clock that NTP can nudge, and a stable string is all this needs.
    """
    try:
        import psutil

        return str(int(psutil.boot_time()))
    except Exception:
        return None


def write_owner_identity(
    artifact_dir: Path, *, host: str | None = None, boot_id: str | None = None
) -> None:
    """Stamp which host and boot own ``artifact_dir``. Never raises."""
    payload = {
        "host": host or socket.gethostname(),
        "boot_id": boot_id if boot_id is not None else current_boot_id(),
        "pid": os.getpid(),
    }
    with contextlib.suppress(OSError):
        owner_identity_path(artifact_dir).write_bytes(orjson.dumps(payload))


def read_owner_identity(artifact_dir: Path) -> dict[str, object] | None:
    """The recorded owner of ``artifact_dir``, or ``None`` when unreadable."""
    try:
        return orjson.loads(owner_identity_path(artifact_dir).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None


def acquire_owner_lock(artifact_dir: Path) -> FileLock | None:
    """Claim ``artifact_dir`` for this process, or return ``None``.

    ``None`` means the caller must not treat itself as the owner: either a
    concurrent run holds the lock, or the filesystem cannot provide one. The
    returned lock must be held for as long as the dir is in use -- releasing it
    early makes the dir look like an orphan to a concurrent sweep.
    """
    lock = FileLock(str(owner_lock_path(artifact_dir)))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return None
    except (NotImplementedError, OSError) as e:
        # No flock on this filesystem (some NFS/FUSE), or the lock file is
        # unwritable. Liveness is unprovable, so fail closed: this dir is
        # never reclaimed rather than risking a live run's store.
        _logger.debug(
            lambda exc=e: f"Owner lock unavailable for {artifact_dir} ({exc!r}); "
            "orphan reclamation is disabled for this dir."
        )
        return None
    write_owner_identity(artifact_dir)
    return lock


def _is_orphan(artifact_dir: Path, grace_seconds: float) -> FileLock | None:
    """Return a held lock proving ``artifact_dir`` is an orphan, else ``None``.

    The caller removes the dir while still holding the returned lock, so a
    concurrent sweep cannot also claim it, then releases.
    """
    try:
        age = time.time() - artifact_dir.stat().st_mtime
    except OSError:
        return None

    identity = read_owner_identity(artifact_dir)

    # A dir stamped by an earlier boot cannot have a live owner -- no process
    # survives a reboot, and any flock or pid recorded against it died with it.
    # That is a proof, so it outranks both the lock and every age grace. The
    # same conclusion PostgreSQL reached for crash-orphaned temp files: identify
    # orphans by boot id rather than inferring them from age.
    stamped_boot = identity.get("boot_id") if identity is not None else None
    this_boot = current_boot_id()
    if stamped_boot is not None and this_boot is not None and stamped_boot != this_boot:
        return _SENTINEL_ORPHAN

    if identity is not None and identity.get("host") != socket.gethostname():
        # A shared MMAP_BASE_PATH (the documented Kubernetes layout) is visible
        # from hosts whose kernels do not share flock state, so an acquire here
        # says nothing about a peer's live run. Age is the only safe signal, and
        # only at a horizon no real run could still be inside.
        return _SENTINEL_ORPHAN if age >= FOREIGN_HOST_GRACE_SECONDS else None

    lock_file = owner_lock_path(artifact_dir)
    if not lock_file.exists():
        # No lock yet: the owner is either mid-startup or predates the lock.
        # Age is the only signal available.
        return _SENTINEL_ORPHAN if age >= grace_seconds else None
    return acquire_owner_lock(artifact_dir)


class _NoLock:
    """Stand-in for the lock-less orphan case so callers have one release path."""

    def release(self) -> None:
        return None


_SENTINEL_ORPHAN = _NoLock()


def sweep_orphaned_graph_artifacts(
    base_path: Path | str, *, grace_seconds: float = ORPHAN_GRACE_SECONDS
) -> list[Path]:
    """Remove graph artifact dirs left by runs that are no longer alive.

    Best-effort housekeeping: every per-dir failure is swallowed so a
    permission error on one stale dir cannot fail the run that is starting.

    Returns:
        The dirs actually removed, sorted by name, for logging and tests.
    """
    base = Path(base_path)
    removed: list[Path] = []
    candidates = sorted(
        d
        for prefix in GRAPH_ARTIFACT_DIR_PREFIXES
        for d in base.glob(f"{prefix}*")
        if d.is_dir()
    )
    for artifact_dir in candidates:
        lock = _is_orphan(artifact_dir, grace_seconds)
        if lock is None:
            continue
        try:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            if not artifact_dir.exists():
                removed.append(artifact_dir)
        finally:
            with contextlib.suppress(OSError, RuntimeError):
                lock.release()
    if removed:
        _logger.info(
            f"Reclaimed {len(removed)} orphaned graph artifact dir(s) under "
            f"{base} left by runs that exited without cleanup."
        )
    return removed
