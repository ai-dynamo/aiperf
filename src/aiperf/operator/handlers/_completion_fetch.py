# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-level result-fetch machinery for the completion handler.

Kept private to ``aiperf.operator.handlers``; public entry points live in
``completion.py`` (and are re-exported there for test-patch stability).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.results_sidecar import CHECKPOINTS_DIR_NAME
from aiperf.kubernetes.spec_converter import (
    DEFAULT_KEY_EXPORT_NAMES,
    key_export_names_from_body,
)
from aiperf.operator.client_cache import (
    get_cancellation_event,
    get_or_create_progress_client,
    is_cancellation_requested,
    job_key,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.results_layout import (
    epoch_key_from_body,
    run_dir,
)

logger = logging.getLogger(__name__)

# Progress-aware retry: give up after this many CONSECUTIVE attempts with
# no new bytes/files arriving in the results dir. Tuning rationale below.
_NO_PROGRESS_STAGNATION_LIMIT = 5

# Key result files that indicate a complete export. If downloads
# succeed but none of these are present, export is still in progress
# and we should retry to capture the full set.
_KEY_FILES = DEFAULT_KEY_EXPORT_NAMES.names
_T = TypeVar("_T")


class _FetchCancelled(Exception):
    """Raised internally when CR deletion interrupts completion I/O."""


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise _FetchCancelled


async def _await_or_cancel(
    operation: Awaitable[_T], cancellation_event: asyncio.Event | None
) -> _T:
    """Await ``operation`` while allowing a job cancellation event to win."""
    if cancellation_event is None:
        return await operation

    operation_task = asyncio.ensure_future(operation)
    cancellation_task = asyncio.create_task(cancellation_event.wait())
    try:
        done, _ = await asyncio.wait(
            (operation_task, cancellation_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise _FetchCancelled
    except BaseException:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise
    finally:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)


class _IncompleteResultsError(Exception):
    """Raised when results are partially fetched (metrics or files missing)."""

    def __init__(
        self,
        metrics: dict[str, Any] | None,
        downloaded: list[str],
        checkpoints: list[str],
    ) -> None:
        self.metrics = metrics
        self.downloaded = downloaded
        self.checkpoints = checkpoints
        super().__init__("Incomplete results")

    def to_fetch_result(self, job_id: str) -> ControllerFetchResult:
        """Convert to a ControllerFetchResult with appropriate error message."""
        error = ""
        if not self.metrics and not self.downloaded:
            error = "Failed to fetch results"
            logger.warning(f"No metrics or files retrieved for {job_id}")
        elif not self.metrics:
            error = "Failed to fetch metrics (files downloaded)"
            logger.warning(
                f"Metrics fetch failed for {job_id}, "
                f"files downloaded: {len(self.downloaded)}"
            )
        elif not self.downloaded:
            error = "Failed to download result files (metrics fetched)"
            logger.warning(f"File download failed for {job_id}, metrics retrieved")

        return ControllerFetchResult(
            metrics=self.metrics,
            downloaded=self.downloaded,
            checkpoints=self.checkpoints,
            error=error,
        )


def _snapshot_bytes(dest_dir: Path) -> int:
    """Total bytes of all files under dest_dir; 0 if the dir does not exist."""
    if not dest_dir.exists():
        return 0
    total = 0
    for entry in dest_dir.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                # File vanished between rglob and stat — skip silently.
                continue
    return total


async def _try_fetch_once(
    fetch_once: Callable[[], Awaitable[ControllerFetchResult]],
    *,
    description: str,
    attempt: int,
) -> tuple[ControllerFetchResult | None, BaseException | None]:
    """Run one fetch attempt. Returns (result, exc) with exactly one non-None."""
    try:
        return await fetch_once(), None
    except _FetchCancelled:
        raise
    except _IncompleteResultsError as e:
        return None, e
    except (TimeoutError, aiohttp.ClientError, OSError, ApiException) as e:
        # Transient errors (network, IO) count as no-progress attempts
        # too; if they persist past stagnation_limit we bubble them up.
        logger.debug(
            "%s attempt %d raised %s; treating as no-progress",
            description,
            attempt,
            type(e).__name__,
        )
        return None, e
    except Exception as e:  # noqa: BLE001 - progress-aware retry must tolerate any fetch error as no-progress; stagnation limit bubbles persistent errors up
        logger.debug(
            "%s attempt %d raised %s; treating as no-progress",
            description,
            attempt,
            type(e).__name__,
        )
        return None, e


async def _fetch_with_progress_aware_retry(
    fetch_once: Callable[[], Awaitable[ControllerFetchResult]],
    *,
    dest_dir: Path,
    job_id: str,
    initial_delay: float,
    description: str,
    is_cancelled: Callable[[], bool] | None = None,
    cancellation_event: asyncio.Event | None = None,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    stagnation_limit: int = _NO_PROGRESS_STAGNATION_LIMIT,
) -> ControllerFetchResult:
    """Retry ``fetch_once`` until it returns, or until no progress has been
    made for ``stagnation_limit`` consecutive attempts.

    Unlike a plain count-based retry, this tolerates controllers that take
    minutes to finalise their export. Records-manager is single-core bound on
    pure-Python aggregation; at high concurrency its final ``summarize()``
    can run longer than any fixed retry window. As long as the partial
    checkpoint on disk is still growing or new files are landing, we know
    the controller is still working and should keep waiting. Only if nothing
    changes across ``stagnation_limit`` consecutive attempts (wall-clock
    ~60s at default cap) do we give up.

    Progress signal: total bytes of all files directly under ``dest_dir``
    and its subtree. A file that grows counts as progress; a file that
    shrinks or disappears does not.

    ``is_cancelled``, if supplied, is polled between attempts and at each
    sleep wakeup; returning True aborts the loop promptly. This avoids
    blocking a CR deletion behind a full stagnation window.
    """
    del job_id  # accepted for API symmetry with fetch_results_with_retry
    delay = initial_delay
    attempt = 0
    no_progress_streak = 0
    last_bytes = _snapshot_bytes(dest_dir)
    while True:
        if is_cancelled is not None and is_cancelled():
            return await fetch_once()
        attempt += 1
        result, pending_exc = await _try_fetch_once(
            fetch_once, description=description, attempt=attempt
        )
        if result is not None:
            return result

        if is_cancelled is not None and is_cancelled():
            return await fetch_once()

        bytes_now = _snapshot_bytes(dest_dir)
        no_progress_streak = _update_progress_streak(
            bytes_now=bytes_now,
            last_bytes=last_bytes,
            streak=no_progress_streak,
            stagnation_limit=stagnation_limit,
            description=description,
            attempt=attempt,
            delay=delay,
            pending_exc=pending_exc,
        )
        last_bytes = bytes_now
        await _await_or_cancel(
            asyncio.sleep(delay * random.uniform(0.8, 1.2)),
            cancellation_event,
        )
        delay = min(delay * backoff_multiplier, max_delay)


def _update_progress_streak(
    *,
    bytes_now: int,
    last_bytes: int,
    streak: int,
    stagnation_limit: int,
    description: str,
    attempt: int,
    delay: float,
    pending_exc: BaseException | None,
) -> int:
    """Return the new no-progress streak count; raise ``pending_exc`` on stall.

    Progress = bytes_now grew. No progress advances the streak by one; reaching
    ``stagnation_limit`` raises the most recent ``pending_exc`` instead of
    returning, which matches the original count-based behaviour.
    """
    if bytes_now > last_bytes:
        logger.debug(
            "%s attempt %d progressing: %d -> %d bytes on disk; retrying in %.1fs",
            description,
            attempt,
            last_bytes,
            bytes_now,
            delay,
        )
        return 0

    streak += 1
    if streak >= stagnation_limit:
        logger.warning(
            "%s stalled at %d bytes for %d consecutive attempts — giving up",
            description,
            bytes_now,
            streak,
        )
        assert pending_exc is not None
        raise pending_exc
    logger.debug(
        "%s attempt %d no new bytes (%d) — stagnation %d/%d",
        description,
        attempt,
        bytes_now,
        streak,
        stagnation_limit,
    )
    return streak


def _split_downloaded(paths: list[str] | None) -> tuple[list[str], list[str]]:
    final_files: list[str] = []
    checkpoint_files: list[str] = []
    for path in paths or []:
        if path.startswith(f"{CHECKPOINTS_DIR_NAME}/"):
            checkpoint_files.append(path)
        else:
            final_files.append(path)
    return final_files, checkpoint_files


def _merge_downloaded(
    current: list[str] | None, new: list[str] | None
) -> list[str] | None:
    if not new:
        return current
    if not current:
        return list(new)
    return sorted(set(current) | set(new))


async def _download_final_and_sidecar(
    *,
    progress_client: ProgressClient,
    controller_host: str,
    dest_dir: Path,
    state: dict[str, Any],
    key_files: frozenset[str] = _KEY_FILES,
    is_cancelled: Callable[[], bool] | None = None,
    cancellation_event: asyncio.Event | None = None,
) -> None:
    """Download results via the primary client and, if key files are still
    missing, fall back to the sidecar port. Mutates ``state`` in place.
    """
    if not OperatorEnvironment.RESULTS.DIR.exists():
        return

    _raise_if_cancelled(is_cancelled)
    downloaded = await _await_or_cancel(
        progress_client.download_all_results(controller_host, dest_dir),
        cancellation_event,
    )
    if downloaded:
        final_files, checkpoint_files = _split_downloaded(downloaded)
        state["downloaded"] = _merge_downloaded(state["downloaded"], final_files)
        state["checkpoints"] = _merge_downloaded(state["checkpoints"], checkpoint_files)

    _raise_if_cancelled(is_cancelled)
    sidecar_port = K8sEnvironment.PORTS.RESULTS_SIDECAR
    has_key_file = bool(key_files & set(state["downloaded"] or []))
    if has_key_file or sidecar_port == K8sEnvironment.PORTS.API_SERVICE:
        return

    async with ProgressClient(port=sidecar_port) as sidecar_client:
        sidecar_downloaded = await _await_or_cancel(
            sidecar_client.download_all_results(controller_host, dest_dir),
            cancellation_event,
        )
    if sidecar_downloaded:
        final_files, checkpoint_files = _split_downloaded(sidecar_downloaded)
        state["downloaded"] = _merge_downloaded(state["downloaded"], final_files)
        state["checkpoints"] = _merge_downloaded(state["checkpoints"], checkpoint_files)


async def _run_fetch_loop_safely(
    fetch_once: Callable[[], Awaitable[ControllerFetchResult]],
    *,
    dest_dir: Path,
    job_id: str,
    retry_delay: float,
    stagnation_limit: int,
    is_cancelled: Callable[[], bool],
    state: dict[str, Any],
    cancellation_event: asyncio.Event | None = None,
) -> ControllerFetchResult:
    """Drive ``_fetch_with_progress_aware_retry`` and convert all failures into
    a ControllerFetchResult carrying whatever partial state we have, rather
    than re-raising into the kopf reconcile.
    """
    try:
        return await _fetch_with_progress_aware_retry(
            fetch_once,
            dest_dir=dest_dir,
            job_id=job_id,
            initial_delay=retry_delay,
            description=f"results fetch for {job_id}",
            is_cancelled=is_cancelled,
            cancellation_event=cancellation_event,
            stagnation_limit=stagnation_limit,
        )
    except _FetchCancelled:
        return ControllerFetchResult(
            metrics=state["metrics"],
            downloaded=state["downloaded"] or [],
            checkpoints=state["checkpoints"] or [],
            error="Cancelled by CR deletion",
        )
    except _IncompleteResultsError as e:
        return e.to_fetch_result(job_id)
    except (TimeoutError, aiohttp.ClientError, OSError, ApiException) as e:
        logger.warning(f"Results fetch failed for {job_id}: {e}")
        return ControllerFetchResult(
            metrics=state["metrics"],
            downloaded=state["downloaded"] or [],
            checkpoints=state["checkpoints"] or [],
            error=f"Failed to fetch results: {e}",
        )
    except Exception as e:  # noqa: BLE001 - fetch retry is best-effort; any error returns whatever partial state we have rather than re-raising into the kopf reconcile
        logger.warning(f"Results fetch failed for {job_id}: {e}")
        return ControllerFetchResult(
            metrics=state["metrics"],
            downloaded=state["downloaded"] or [],
            checkpoints=state["checkpoints"] or [],
            error=f"Failed to fetch results: {e}",
        )


async def fetch_results_with_retry(
    controller_host: str,
    namespace: str,
    job_id: str,
    *,
    max_retries: int = OperatorEnvironment.RESULTS.MAX_RETRIES,
    retry_delay: float = OperatorEnvironment.RESULTS.RETRY_DELAY,
    dest_dir: Path | None = None,
    body: dict[str, Any] | None = None,
) -> ControllerFetchResult:
    """Fetch results from controller pod with retry logic.

    Uses the cached ProgressClient for the job. Falls back to creating
    a temporary client if no cached one exists (e.g. after restart).

    Args:
        controller_host: Controller pod DNS name.
        namespace: Kubernetes namespace (used for results directory scoping).
        job_id: Job identifier for results directory.
        max_retries: Maximum retry attempts.
        retry_delay: Delay between retries (with exponential backoff).
        dest_dir: Explicit destination directory for results. When None,
            derives the epoch-keyed path from ``body``; passing both as None
            is a programmer error and raises ``ValueError``.
        body: AIPerfJob CR body used to compute the ``<ns>/<name>/<epoch>/``
            run directory. Required in production; tests may omit it when
            the caller explicitly passes ``dest_dir``.

    Returns:
        ControllerFetchResult with metrics dict and list of downloaded files.
    """
    body_uid = (body.get("metadata") or {}).get("uid") if body is not None else None
    key = job_key(
        namespace,
        job_id,
        str(body_uid) if body_uid is not None else None,
    )
    progress_client = await get_or_create_progress_client(key)
    cancellation_event = get_cancellation_event(key)

    if dest_dir is None:
        if body is None:
            raise ValueError(
                f"fetch_results_with_retry({namespace}/{job_id}): need either "
                "dest_dir or body to derive the epoch-keyed run directory"
            )
        epoch = epoch_key_from_body(body)
        dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)

    # Mutable state shared across retry attempts so partial progress
    # (e.g. metrics fetched but files not yet) survives retries.
    # Use None (not yet attempted) vs [] (attempted, no files) to avoid
    # treating a valid empty download list as "not yet fetched".
    state: dict[str, Any] = {"metrics": None, "downloaded": None, "checkpoints": None}
    key_files = key_export_names_from_body(body).names

    async def _fetch_once() -> ControllerFetchResult:
        return await _fetch_once_into_state(
            key=key,
            controller_host=controller_host,
            dest_dir=dest_dir,
            progress_client=progress_client,
            state=state,
            key_files=key_files,
            cancellation_event=cancellation_event,
        )

    return await _run_fetch_loop_safely(
        _fetch_once,
        dest_dir=dest_dir,
        job_id=job_id,
        retry_delay=retry_delay,
        stagnation_limit=max(max_retries, 1),
        is_cancelled=lambda: is_cancellation_requested(key),
        cancellation_event=cancellation_event,
        state=state,
    )


async def _fetch_once_into_state(
    *,
    key: str,
    controller_host: str,
    dest_dir: Path,
    progress_client: ProgressClient,
    state: dict[str, Any],
    key_files: frozenset[str] = _KEY_FILES,
    cancellation_event: asyncio.Event | None = None,
) -> ControllerFetchResult:
    """Single fetch attempt: short-circuits on cancellation, accumulates
    metrics + downloads into ``state``, raises ``_IncompleteResultsError``
    if the key export files aren't present yet.
    """
    if is_cancellation_requested(key):
        return ControllerFetchResult(
            metrics=state["metrics"],
            downloaded=state["downloaded"] or [],
            checkpoints=state["checkpoints"] or [],
            error="Cancelled by CR deletion",
        )

    if state["metrics"] is None:
        state["metrics"] = await _await_or_cancel(
            progress_client.get_metrics(controller_host), cancellation_event
        )

    _raise_if_cancelled(lambda: is_cancellation_requested(key))

    await _download_final_and_sidecar(
        progress_client=progress_client,
        controller_host=controller_host,
        dest_dir=dest_dir,
        state=state,
        key_files=key_files,
        is_cancelled=lambda: is_cancellation_requested(key),
        cancellation_event=cancellation_event,
    )

    if state["downloaded"] is not None:
        if key_files & set(state["downloaded"]):
            return ControllerFetchResult(
                metrics=state["metrics"],
                downloaded=state["downloaded"],
                checkpoints=state["checkpoints"] or [],
            )
        logger.info(
            f"Downloaded {len(state['downloaded'])} files but missing key "
            f"export files, retrying..."
        )
    raise _IncompleteResultsError(
        state["metrics"],
        state["downloaded"] or [],
        state["checkpoints"] or [],
    )
