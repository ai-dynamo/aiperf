# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed readiness markers for an exported results directory.

The controller, the records manager, the operator's results API, and the
Kubernetes results sidecar all agree on this on-disk contract: top-level result
files stay hidden from consumers until ``write_ready_marker`` commits, and a
crash mid-export leaves the processing marker behind so the directory reads as
not-ready rather than partially-ready.

This module is deliberately dependency-free (stdlib plus a lazily imported
``orjson``). ``aiperf.kubernetes.results_sidecar`` re-exports every name here
and adds the FastAPI/uvicorn server on top; importing that module for the
marker contract alone would drag uvicorn, FastAPI, starlette and aiofiles into
processes that never serve a request.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

READY_MARKER_NAME = ".aiperf_results_ready.json"
PROCESSING_MARKER_NAME = ".aiperf_results_processing.json"
CHECKPOINTS_DIR_NAME = "checkpoints"
_RESERVED_MARKER_NAMES = frozenset({READY_MARKER_NAME, PROCESSING_MARKER_NAME})


def ready_marker_path(base_dir: Path) -> Path:
    """Return the sidecar readiness marker path."""
    return base_dir / READY_MARKER_NAME


def checkpoints_dir(base_dir: Path) -> Path:
    """Return the checkpoint directory under the results directory."""
    return base_dir / CHECKPOINTS_DIR_NAME


def processing_marker_path(base_dir: Path) -> Path:
    """Return the sidecar processing marker path."""
    return base_dir / PROCESSING_MARKER_NAME


def write_processing_marker(base_dir: Path) -> Path:
    """Begin a fail-closed result-export transaction.

    Any ready marker from an earlier attempt is removed and persisted before
    the processing marker is installed. A crash at any point before the new
    ready marker commits therefore leaves top-level result files hidden.
    """
    import orjson

    marker = processing_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stale_ready = ready_marker_path(base_dir)
    try:
        stale_ready.unlink()
    except FileNotFoundError:
        pass
    else:
        _fsync_directory(base_dir)
    _atomic_write_marker(marker, orjson.dumps({"processing": True}))
    return marker


def clear_processing_marker(base_dir: Path) -> None:
    """Remove the processing marker once final exports are stable."""
    marker = processing_marker_path(base_dir)
    try:
        marker.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(base_dir)


def write_ready_marker(
    base_dir: Path,
    *,
    was_cancelled: bool = False,
    terminal_phase: str | None = None,
    terminal_error: str | None = None,
) -> Path:
    """Durably publish readiness after every result artifact is stable."""
    import orjson

    marker = ready_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload: dict[str, bool | str] = {
            "ready": True,
            "was_cancelled": was_cancelled,
        }
        if terminal_phase is not None:
            payload["terminal_phase"] = terminal_phase
        if terminal_error is not None:
            payload["terminal_error"] = terminal_error
        _atomic_write_marker(marker, orjson.dumps(payload))
        clear_processing_marker(base_dir)
    except BaseException:
        _rollback_failed_ready_commit(base_dir)
        raise
    return marker


def _atomic_write_marker(path: Path, payload: bytes) -> None:
    """Install a fully written, fsynced marker with one atomic rename."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        file_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(file_descriptor, remaining)
                if written <= 0:
                    raise OSError(
                        f"Failed to write result marker temporary file {temporary}"
                    )
                remaining = remaining[written:]
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        _best_effort_remove_marker(temporary)
        _best_effort_remove_marker(path)
        raise


def _rollback_failed_ready_commit(base_dir: Path) -> None:
    """Best-effort rollback that restores the fail-closed processing state."""
    import orjson

    try:
        _best_effort_remove_marker(ready_marker_path(base_dir))
        processing = processing_marker_path(base_dir)
        if processing.is_file():
            return
        _atomic_write_marker(processing, orjson.dumps({"processing": True}))
    except BaseException:
        # Readiness is already absent, so even a failed status-marker restore
        # leaves the results sidecar closed to top-level artifacts.
        logger.warning(
            "Failed to restore the processing marker after rolling back readiness",
            exc_info=True,
        )


def _best_effort_remove_marker(path: Path) -> None:
    """Remove a failed marker installation without masking its original error."""
    try:
        path.unlink()
    except OSError:
        return
    with contextlib.suppress(OSError):
        _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes when directory fsync is supported."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _safe_resolve(base_dir: Path, filename: str) -> Path | None:
    """Resolve a path under base_dir, rejecting traversal."""
    try:
        resolved = (base_dir / filename).resolve()
        resolved.relative_to(base_dir.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _is_ready(base_dir: Path) -> bool:
    """Whether the controller has finished exporting results."""
    return ready_marker_path(base_dir).is_file()


def _is_processing(base_dir: Path) -> bool:
    """Whether the controller is still exporting results."""
    return processing_marker_path(base_dir).is_file()


def _is_checkpoint_path(base_dir: Path, path: Path) -> bool:
    """Whether a path points at a checkpoint artifact under the results dir."""
    try:
        relative = path.relative_to(base_dir.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == CHECKPOINTS_DIR_NAME


def _safe_size(entry: Path) -> int | None:
    """Return the file size, or ``None`` if the file vanished mid-listing.

    A checkpoint parquet can be rotated/unlinked by the writer between
    ``rglob`` enumeration and ``stat``; treat the race as "skip this entry"
    rather than 500-ing the whole listing.
    """
    try:
        return entry.stat().st_size
    except OSError:
        return None
