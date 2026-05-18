# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entrypoint for the MLflow post-run export.

Kept in its own module so ``multiprocessing.get_context("spawn")`` can import
the target function by fully-qualified name without pulling in the rest of
``mlflow_data_exporter``'s heavyweight imports at interpreter startup time in
the child. Also keeps ``mlflow_data_exporter.py`` under the file-size
ergonomics baseline.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from contextlib import suppress
from typing import Any, Protocol

from aiperf.exporters.exporter_config import ExporterConfig


class _WarningLogger(Protocol):
    def __call__(self, message: str) -> None: ...


def run_export_in_subprocess(
    exporter_config: ExporterConfig,
    result_queue: Any,
) -> None:
    """Subprocess entrypoint for ``MLflowDataExporter._export_sync``.

    Reports success as ``None`` and failure as a ``repr(exc)`` string on
    ``result_queue`` so the parent can log without requiring pickled
    exceptions (some MLflow errors don't round-trip through pickle).
    """
    # Import lazily so the module-level cost stays minimal for callers that
    # only read ``run_export_in_subprocess`` as a spawn target.
    from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter

    try:
        exporter = MLflowDataExporter(exporter_config=exporter_config)
        exporter._export_sync()
        result_queue.put(None)
    except Exception as exc:  # noqa: BLE001 - all errors must travel back to parent
        result_queue.put(repr(exc))


async def export_with_timeout(
    exporter_config: ExporterConfig,
    export_timeout: float,
    warn: _WarningLogger,
) -> None:
    """Run the MLflow export in a terminable subprocess.

    An ``asyncio.to_thread`` wrapper only releases the awaiter — the thread
    keeps running inside the MLflow SDK (holding TCP connections and the run
    open) until the socket unblocks, which can take minutes against an
    unreachable tracking server. Running the export in a ``spawn`` subprocess
    and calling ``terminate()`` on timeout gives us a real upper bound.

    Uses the ``spawn`` context so the child gets a fresh interpreter — mlflow
    keeps module-level state (tracking URI, active run) that must not be
    inherited from the parent via fork.
    """
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue[str | None] = ctx.Queue(maxsize=1)
    # Re-lookup the spawn target by attribute so tests that monkeypatch
    # ``run_export_in_subprocess`` on this module reach the subprocess.
    spawn_target = globals()["run_export_in_subprocess"]
    process = ctx.Process(
        target=spawn_target,
        args=(exporter_config, result_queue),
        name="aiperf-mlflow-export",
        daemon=True,
    )
    process.start()
    try:
        await asyncio.to_thread(process.join, export_timeout)
        if process.is_alive():
            warn(
                f"MLflow export timed out after {export_timeout}s. "
                "The tracking server may be unreachable. Terminating subprocess."
            )
            process.terminate()
            await asyncio.to_thread(process.join, 5.0)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 1.0)
            return
        # Subprocess returned; read status from the queue if present.
        #
        # Track whether the queue yielded any status so we can surface silent
        # crashes via exitcode. If the child dies before reaching the ``try``
        # in ``run_export_in_subprocess`` (spawn bootstrap failure, SIGKILL/OOM,
        # native crash) the queue stays empty and an exitcode-only check is
        # the last signal we have that something went wrong.
        queue_reported_status = False
        with suppress(Exception):
            error_msg = result_queue.get_nowait()
            queue_reported_status = True
            if error_msg is not None:
                warn(f"MLflow export subprocess reported: {error_msg}")
        exitcode = process.exitcode
        if not queue_reported_status and exitcode is not None and exitcode != 0:
            warn(
                "MLflow export subprocess exited with non-zero status and "
                f"left no status on the result queue. exitcode={exitcode}. "
                "Likely a spawn bootstrap failure, SIGKILL/OOM, or native crash."
            )
    finally:
        with suppress(Exception):
            result_queue.close()
            result_queue.cancel_join_thread()
