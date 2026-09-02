# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed readiness markers for an exported results directory.

The controller, the records manager, the operator's results API, and the
Kubernetes results sidecar all agree on this on-disk contract: top-level result
files stay hidden from consumers until ``write_ready_marker`` commits, and a
crash mid-export leaves the processing marker behind so the directory reads as
not-ready rather than partially-ready.

This module is deliberately dependency-free (stdlib plus a lazily imported
``orjson``). ``aiperf.kubernetes.results_sidecar`` imports the marker helpers
it serves and adds the FastAPI/uvicorn server on top; importing that module
for the marker contract alone would drag uvicorn, FastAPI, starlette and
aiofiles into processes that never serve a request.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

logger = logging.getLogger(__name__)

READY_MARKER_NAME = ".aiperf_results_ready.json"
PROCESSING_MARKER_NAME = ".aiperf_results_processing.json"
CHECKPOINTS_DIR_NAME = "checkpoints"

# Run-directory key format, owned here so the operator and the config resolver
# cannot drift apart. This is exactly the shape ``results_layout.epoch_key_from
# _body`` emits: bare epoch-seconds (10 digits today, 9 for pre-2001 legacy
# directories), optionally followed by ONE six-digit suffix -- real microseconds
# for a fractional timestamp, or a uid-derived disambiguator for a whole-second
# Kubernetes one. Nothing in between is producible, and accepting an
# in-between length is actively harmful: ``epoch_key_seconds`` strips the last
# six digits from any key longer than 10, so an 11-14 digit value would decode
# to a nonsense near-1970 instant instead of being rejected.
# Anchored with ``\A``/``\Z``, not ``^``/``$``: Python's ``$`` also matches
# before a trailing newline, so ``^...$`` accepted "1714069323\n" as a valid
# run key. Callers happen to ``.strip()`` today, which made that latent rather
# than live -- the anchors close it at the source instead.
EPOCH_RE = re.compile(r"\A\d{9,10}(\d{6})?\Z")
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
    partial: bool = False,
    failed_exporters: list[str] | None = None,
) -> Path:
    """Durably publish readiness after every result artifact is stable.

    ``partial``/``failed_exporters`` let the controller publish readiness even
    when one exporter failed while a sibling exporter's artifact is already
    stable on disk: consumers can still serve the successful files instead of
    the whole run reading as unavailable.
    """
    import orjson

    marker = ready_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload: dict[str, bool | str | list[str]] = {
            "ready": True,
            "was_cancelled": was_cancelled,
            "partial": partial,
        }
        if terminal_phase is not None:
            payload["terminal_phase"] = terminal_phase
        if terminal_error is not None:
            payload["terminal_error"] = terminal_error
        if failed_exporters:
            payload["failed_exporters"] = list(failed_exporters)
        _atomic_write_marker(marker, orjson.dumps(payload))
        clear_processing_marker(base_dir)
    except BaseException:
        _rollback_failed_ready_commit(base_dir)
        raise
    return marker


def read_ready_marker_info(base_dir: Path) -> tuple[bool, list[str]]:
    """Return ``(partial, failed_exporters)`` recorded on the ready marker.

    Best-effort: a missing, unreadable, or malformed marker reports a clean
    (non-partial) result rather than raising, since the marker's mere
    existence -- not its optional fields -- is what gates readiness.
    """
    import orjson

    if not _is_ready(base_dir):
        return False, []
    try:
        payload = orjson.loads(ready_marker_path(base_dir).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False, []
    if not isinstance(payload, dict):
        return False, []
    partial = bool(payload.get("partial", False))
    failed_exporters = payload.get("failed_exporters") or []
    if not isinstance(failed_exporters, list):
        return partial, []
    return partial, [str(item) for item in failed_exporters]


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
    if (
        "\x00" in filename
        or PurePosixPath(filename).is_absolute()
        or PureWindowsPath(filename).is_absolute()
    ):
        return None
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


RESULT_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}


class ResultFileUnavailable(Exception):
    """A requested result file cannot be served.

    Transport-neutral so the marker policy stays importable without FastAPI;
    the HTTP adapters translate ``status_code``/``detail`` verbatim.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def content_type_for(file_path: Path) -> str:
    """Return the download content type for a result artifact."""
    return RESULT_CONTENT_TYPES.get(
        file_path.suffix.lower(), "application/octet-stream"
    )


def not_ready_detail(base_dir: Path) -> str:
    """Return the 404 detail served while an export is still in flight."""
    processing_detail = " export still processing;" if _is_processing(base_dir) else ""
    return (
        f"Results not ready for {base_dir.name};{processing_detail} "
        f"marker file {READY_MARKER_NAME} not present — retry after completion"
    )


def resolve_result_file(base_dir: Path, filename: str) -> Path:
    """Resolve a downloadable result file, or raise ``ResultFileUnavailable``.

    Enforces the same fail-closed policy on both results surfaces: traversal
    and marker names are rejected outright, top-level artifacts stay hidden
    until the ready marker commits, and checkpoint artifacts bypass that gate
    because they are written incrementally by design.
    """
    file_path = _safe_resolve(base_dir, filename)
    if file_path is None or file_path.name in _RESERVED_MARKER_NAMES:
        raise ResultFileUnavailable(
            400,
            f"Invalid filename {filename!r}: path traversal or reserved marker name",
        )
    if not _is_ready(base_dir) and not _is_checkpoint_path(
        base_dir.resolve(), file_path
    ):
        raise ResultFileUnavailable(404, not_ready_detail(base_dir))
    return file_path


def collect_ready_artifacts(
    base_dir: Path, *, recursive: bool
) -> list[tuple[str, int]]:
    """Enumerate ``(relative_name, size)`` pairs for downloadable artifacts.

    Top-level artifacts are gated on the ready marker; checkpoint artifacts
    under ``checkpoints/`` are always listed so an in-flight run's checkpoint
    stream stays fetchable. Marker files are never listed, and files that
    vanish mid-walk are skipped rather than failing the whole listing.

    ``recursive`` selects the surface's intended scope: the in-process router
    lists only the run's own top-level artifacts (its artifact directory also
    holds upload/scratch subdirectories that are not results), while the
    sidecar walks its whole ``/results`` volume so nested sweep harvests
    (``<ns>/sweeps/<sweep>/<epoch>/...``) surface too.
    """
    cp_dir = checkpoints_dir(base_dir)
    files: list[tuple[str, int]] = []

    def _append(entry: Path) -> None:
        size = _safe_size(entry)
        if size is not None:
            files.append((entry.relative_to(base_dir).as_posix(), size))

    if _is_ready(base_dir):
        entries = base_dir.rglob("*") if recursive else base_dir.iterdir()
        for entry in entries:
            if (
                not entry.is_file()
                or entry.name in _RESERVED_MARKER_NAMES
                or cp_dir in entry.parents
            ):
                continue
            _append(entry)

    if cp_dir.is_dir():
        for entry in cp_dir.rglob("*"):
            if entry.is_file():
                _append(entry)

    return sorted(files, key=lambda item: item[0])
