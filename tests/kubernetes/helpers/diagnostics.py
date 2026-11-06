# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-time cluster diagnostics capture for Kubernetes E2E tests.

The CI workflows' "Collect cluster diagnostics on failure" step runs AFTER
the pytest session has torn down its fixtures — workloads deleted, operator
uninstalled, Kind cluster gone — so every captured artifact was empty or
``connection refused``. This helper runs the same capture from INSIDE the
fixture teardown (operator fixture before uninstall, cluster fixture before
delete, where ``kind export logs`` still finds terminated pods' container
logs on the node filesystem), and only when the session saw failures.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger

logger = AIPerfLogger(__name__)

_DIAG_TIMEOUT_SECONDS = 60


async def _run_capture(cmd: list[str], out_path: Path, timeout: int) -> None:
    """Run a command, writing combined stdout+stderr to ``out_path``.

    Best-effort by construction: every failure (timeout, non-zero exit,
    missing binary) is recorded in the output file, never raised — a
    diagnostics capture must not fail the test session's teardown.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            stdout = f"<timed out after {timeout}s: {' '.join(cmd)}>".encode()
    except Exception as exc:  # noqa: BLE001 — capture must never raise
        stdout = f"<capture failed: {exc}>".encode()
    out_path.write_bytes(stdout)


async def _capture_guarded(cmd: list[str], out_path: Path, timeout: int) -> None:
    """``_run_capture`` plus a last-resort guard: diagnostics must never break
    fixture teardown, even if the capture itself fails unexpectedly."""
    try:
        await _run_capture(cmd, out_path, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[diagnostics] capture failed for {out_path.name}: {exc}")


def _diagnostic_commands(context: str | None) -> list[tuple[str, list[str]]]:
    """(filename, kubectl command) pairs captured on every failure dump."""
    k = ["kubectl", *(["--context", context] if context else [])]
    return [
        ("all-resources.txt", [*k, "get", "all", "-A"]),
        ("events.txt", [*k, "get", "events", "-A", "--sort-by=.lastTimestamp"]),
        ("describe-nodes.txt", [*k, "describe", "nodes"]),
        ("describe-pods.txt", [*k, "describe", "pods", "-A"]),
        ("aiperfjobs.yaml", [*k, "get", "aiperfjobs", "-A", "-o", "yaml"]),
        ("aiperfsweeps.yaml", [*k, "get", "aiperfsweeps", "-A", "-o", "yaml"]),
        ("jobsets.yaml", [*k, "get", "jobsets", "-A", "-o", "yaml"]),
        ("pods.yaml", [*k, "get", "pods", "-A", "-o", "yaml"]),
    ]


async def dump_cluster_diagnostics(
    context: str | None,
    out_dir: Path,
    cluster_name: str | None = None,
    timeout: int = _DIAG_TIMEOUT_SECONDS,
) -> Path:
    """Dump cluster state and logs to ``out_dir``; never raises.

    Args:
        context: kubectl context of the cluster (None uses the default).
        out_dir: Destination directory (created if missing).
        cluster_name: When set (local Kind runs), also ``kind export logs``
            into ``out_dir/kind`` — that captures container logs of pods the
            test already deleted, which kubectl logs can no longer reach.
        timeout: Per-command timeout in seconds.

    Returns:
        The output directory.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(f"[diagnostics] cannot create {out_dir}: {exc}")
        return out_dir
    logger.info(f"[diagnostics] Capturing cluster state to {out_dir}")

    for filename, cmd in _diagnostic_commands(context):
        await _capture_guarded(cmd, out_dir / filename, timeout)

    # Per-pod logs for everything still present.
    proc_out = out_dir / "_pod-index.txt"
    await _capture_guarded(
        [
            "kubectl",
            *(["--context", context] if context else []),
            "get",
            "pods",
            "-A",
            "-o",
            "jsonpath={range .items[*]}{.metadata.namespace} {.metadata.name}{'\\n'}{end}",
        ],
        proc_out,
        timeout,
    )
    for line in (proc_out.read_text() if proc_out.exists() else "").splitlines():
        namespace, _, pod = line.partition(" ")
        if not namespace or not pod:
            continue
        await _capture_guarded(
            [
                "kubectl",
                *(["--context", context] if context else []),
                "logs",
                "-n",
                namespace,
                pod,
                "--all-containers",
                "--prefix",
                "--tail=-1",
            ],
            out_dir / f"logs-{namespace}-{pod}.txt",
            timeout,
        )
    proc_out.unlink(missing_ok=True)

    if cluster_name:
        await _capture_guarded(
            ["kind", "export", "logs", str(out_dir / "kind"), "--name", cluster_name],
            out_dir / "kind-export.txt",
            timeout * 2,
        )

    logger.info(f"[diagnostics] Capture complete: {out_dir}")
    return out_dir
