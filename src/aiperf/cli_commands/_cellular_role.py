# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared orchestration for the native cellular frontend roles.

``aiperf controller`` and ``aiperf cell`` are the Kubernetes-cellular counterparts
of ``aiperf profile``: each is the Python frontend (the orchestrator) on its pod. It
resolves Config v2, projects it through ``rust_wire``, and launches ``aiperf-runner``
over stdio -- the runner becomes a controller or a cell from the ``CELL_*`` env the
operator sets, and (once the ai-dynamo/velo cell transport lands) the cells stream
their record shards to the controller, which merges them and runs the native export
plane.

The one thing the frontend owns beyond ``aiperf profile`` is reporting up to
Kubernetes. Only the CONTROLLER reports -- it holds the aggregate view:
- live progress -> the owning AIPerfJob ``.status`` while the run is in flight
  (``completion_signal.report_benchmark_progress``), tailed from the runner's
  ``AIPERF_CELLULAR_HEARTBEAT_LOG`` NDJSON cadence;
- completion -> the ``benchmark-complete`` annotation once the run exits successfully
  (``completion_signal.signal_benchmark_complete``), mirroring the mesh
  ``SystemController``'s completion timing.

A cell reports nothing (it has only its own slice); it just runs and ships its shard.
Both push to Kubernetes with the pod's in-cluster client + ``aiperfjobs/status`` RBAC;
off-cluster (no ``AIPERF_JOB_ID``) the reporters no-op, so the same command runs
locally for debugging.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
    from aiperf.config.flags import CLIConfig
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

# The env var the runner's heartbeat lane (rust/runner/src/heartbeat_lane.rs) reads:
# when it names a writable path, the runner appends one NDJSON
# ``{"event":"metrics_heartbeat","counters":{"issued","completed","errored"}}`` line
# per phase-progress cadence. The controller frontend points it at a scratch file and
# tails it for the live ``requestsCompleted`` it pushes to the CR status.
_HEARTBEAT_LOG_ENV = "AIPERF_CELLULAR_HEARTBEAT_LOG"
_HEARTBEAT_FILE = ".aiperf-controller-heartbeat.jsonl"
# How often the controller pushes a status patch (seconds). Independent of the
# runner's own heartbeat cadence; we only push the latest snapshot we have read.
_PROGRESS_PUSH_INTERVAL_S = 5.0


def run_cellular_role(cli_config: CLIConfig, *, role: str) -> None:
    """Resolve Config v2 and run this pod's cellular role to completion.

    Args:
        cli_config: The cyclopts CLIConfig carrying ``--config`` (+ any overrides).
        role: ``"controller"`` or ``"cell"``.
    """
    from aiperf.cli_utils import exit_on_error
    from aiperf.config.loader.errors import ConfigurationError

    # A cell needs no config: it fetches its sliced protocol-v2 envelope from the
    # controller over velo (using the operator-set AIPERF_CELL_* env), so it just
    # execs ``aiperf-runner --cell``. Only the controller reads Config v2 + reports.
    if role == "cell":
        _run_cell()
        return

    with exit_on_error(title="Error Running AIPerf Cellular Role", show_traceback=False):
        from aiperf.cli_runner import _make_benchmark_run
        from aiperf.config.flags.resolver import resolve_config
        from aiperf.config.loader import build_benchmark_plan

        # ``--config`` names the Config v2 file the operator mounted from the
        # ``{jobset}-config`` ConfigMap (jobset_helpers._config_path ->
        # /etc/aiperf/config.yaml); resolve_config loads it exactly as
        # ``aiperf profile --config`` does locally.
        config = resolve_config(cli_config, cli_config.config_file)
        plan = build_benchmark_plan(config)
        # Kubernetes launches exactly one benchmark per pod (sweeps are expanded into
        # child AIPerfJobs by the sweep-controller), so the plan is single-config.
        run = _make_benchmark_run(plan.configs[0], variables=plan.variables)

    with exit_on_error(
        title="Error Running AIPerf Cellular Role",
        quiet_for=(ConfigurationError,),
    ):
        asyncio.run(_run_controller(run))


def _run_cell() -> None:
    """Run this pod as a cell: exec ``aiperf-runner --cell`` and wait.

    The velo cell reads no config and no stdin -- it fetches its ``(cell_id,
    cell_count)`` budget slice + sliced execute envelope from the controller over
    velo, using the ``AIPERF_CELL_ID`` / ``AIPERF_CELL_COUNT`` /
    ``AIPERF_CELL_CONTROLLER_ADDR`` env the operator set (rust: main.rs::run_cell).
    It streams its records shard back to the controller, which owns the aggregate
    view, so a cell reports nothing to Kubernetes. The runner inherits this process's
    env, so the operator's CELL_* vars flow straight through.
    """
    from aiperf.orchestrator.native_execution import (
        resolve_native_binary,
        run_cell_process,
    )

    binary = resolve_native_binary()
    logger.info("launching cell: %s --cell", binary)
    returncode = run_cell_process(binary)
    if returncode != 0:
        raise RuntimeError(f"cell exited with {returncode}")


async def _run_controller(run: BenchmarkRun) -> None:
    """Run this pod as the controller: launch the runner, push live progress to the
    AIPerfJob CR while it runs, and signal completion when it finishes.

    The runner executes on a worker thread (it is a blocking stdio child); a
    concurrent reporter task tails its heartbeat NDJSON and patches the CR status on a
    fixed cadence. On a successful exit the completion annotation is set, which the
    operator's ``on_benchmark_complete`` watcher turns into the full completion.
    """
    heartbeat_path = run.artifact_dir / _HEARTBEAT_FILE
    _prepare_heartbeat(heartbeat_path)

    reporter = _ControllerProgressReporter(run, heartbeat_path)
    reporter_task = asyncio.create_task(reporter.run())
    try:
        result = await asyncio.to_thread(_execute, run)
    finally:
        reporter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reporter_task

    if not result.success:
        raise RuntimeError(f"controller run failed: {result.error}")

    # Mark the exported results dir ready BEFORE signalling completion. The
    # results-sidecar (serving this same /results volume) refuses to serve result
    # files until the ``.aiperf_results_ready.json`` marker exists, and the
    # operator's on_benchmark_complete -> fetch_results_with_retry downloads
    # through the sidecar the instant it sees the completion annotation. Writing
    # the marker first (the mesh SystemController's contract, which the native
    # cellular controller must uphold) closes the race where the fetch races the
    # export and the run is stamped Failed with results already on disk.
    _mark_results_ready(run)

    # Push the terminal progress + the FULL native-v2 report snapshot, then signal
    # completion -- same ordering as the mesh SystemController (report-then-complete
    # after export). The final snapshot supersedes the live approximations with the
    # committed, exact report metrics.
    await reporter.report_once(final=True)
    await _report_final_snapshot(run)
    from aiperf.kubernetes.completion_signal import signal_benchmark_complete

    await signal_benchmark_complete()


def _mark_results_ready(run: BenchmarkRun) -> None:
    """Write the results-sidecar readiness marker into the run's artifact dir.

    Best-effort: off-cluster (no sidecar) the marker is harmless, and a write
    failure is logged rather than aborting a successful run.
    """
    from aiperf.kubernetes.results_sidecar import write_ready_marker

    try:
        write_ready_marker(run.artifact_dir)
    except OSError as e:
        logger.warning("could not write results-ready marker: %s", e)


async def _report_final_snapshot(run: BenchmarkRun) -> None:
    """Patch the committed ``native-v2.json`` metrics into ``.status.snapshot``.

    After export the controller's ``native-v2.json`` holds the exact, full metric set
    (every catalog metric, not just the live heartbeat's TTFT/ITL/latency); mirror its
    ``metrics`` block (+ summary) into the CR so ``kubectl get aiperfjob`` carries the
    authoritative report snapshot. Best-effort: a missing/oversized report just leaves
    the last live snapshot in place.
    """
    import orjson

    from aiperf.kubernetes.completion_signal import report_benchmark_snapshot

    report_path = run.artifact_dir / "native-v2.json"
    try:
        report = orjson.loads(report_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("final snapshot: could not read %s: %s", report_path, e)
        return
    snapshot = {
        "source": "report",
        "schemaVersion": report.get("schema_version"),
        "metrics": report.get("metrics") or {},
        "summary": report.get("summary") or {},
    }
    with contextlib.suppress(Exception):
        await report_benchmark_snapshot(snapshot)


def _prepare_heartbeat(path: Path) -> None:
    """Point the runner's heartbeat lane at ``path`` and clear any stale file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.unlink()
    os.environ[_HEARTBEAT_LOG_ENV] = str(path)


def _execute(run: BenchmarkRun) -> RunResult:
    """Launch the native ``aiperf --execute`` child for this run and return its result.

    Reuses the canonical native executor, so rust_wire projection, stderr
    forwarding, terminal parsing, native-report loading, and the native export plane
    are identical to ``aiperf profile`` -- the cellular role only adds the CR
    reporting around it.
    """
    from aiperf.orchestrator.native_execution import NativeExecutor

    return NativeExecutor(base_dir=run.artifact_dir).execute_sync(run)


def _profiling_request_budget(run: BenchmarkRun) -> int | None:
    """The profiling phase's request budget, for the progress percent, if any."""
    for phase in run.cfg.phases:
        if getattr(phase, "name", None) == "profiling":
            return getattr(phase, "requests", None)
    return None


class _ControllerProgressReporter:
    """Tails the runner heartbeat NDJSON and pushes progress to the AIPerfJob status.

    The runner appends one ``metrics_heartbeat`` line per cadence with a monotonic
    ``counters.completed``; we keep the latest and patch ``.status.phases.profiling``
    every :data:`_PROGRESS_PUSH_INTERVAL_S`. Best-effort throughout: a missing
    heartbeat file (cadence not yet fired) or a transient status-patch error just
    skips a tick. Off-cluster the underlying push no-ops.
    """

    def __init__(self, run: BenchmarkRun, heartbeat_path: Path) -> None:
        self._run = run
        self._path = heartbeat_path
        self._total = _profiling_request_budget(run)
        self._completed = 0
        self._latest_event: dict[str, Any] | None = None

    async def run(self) -> None:
        """Push progress on a fixed cadence until cancelled."""
        while True:
            await asyncio.sleep(_PROGRESS_PUSH_INTERVAL_S)
            await self.report_once()

    async def report_once(self, *, final: bool = False) -> None:
        """Read the latest heartbeat and patch the CR status once (best-effort).

        Pushes two things: the counter into ``.status.phases.profiling`` (progress
        bar) and the native-v2-level metric snapshot (counters + TTFT/ITL/latency
        distributions) into ``.status.snapshot``, so ``kubectl get aiperfjob`` shows
        live metrics at report fidelity.
        """
        self._read_latest_event()
        if self._completed == 0 and not final:
            return
        from aiperf.kubernetes.completion_signal import (
            report_benchmark_progress,
            report_benchmark_snapshot,
        )

        with contextlib.suppress(Exception):
            await report_benchmark_progress(
                phase="profiling",
                requests_completed=self._completed,
                requests_total=self._total,
                # Must be a value in the AIPerfJob CRD .status.phase enum
                # (Pending/Queued/Initializing/Running/Completed/Failed/Cancelled);
                # the per-phase name lives in .status.phases.profiling, not here.
                overall_phase="Running",
            )
        snapshot = self._live_snapshot()
        if snapshot is not None:
            with contextlib.suppress(Exception):
                await report_benchmark_snapshot(snapshot)

    def _read_latest_event(self) -> None:
        """Keep the last complete heartbeat line (full event) + its completed count.

        Reads the whole (small, one-line-per-cadence) file and keeps the last valid
        line; a partial final line fails to parse and is ignored until the next tick.
        """
        try:
            text = self._path.read_text()
        except OSError:
            return
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                event: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._latest_event = event
            completed = (event.get("counters") or {}).get("completed")
            if isinstance(completed, int):
                self._completed = max(self._completed, completed)
            return

    def _live_snapshot(self) -> dict[str, Any] | None:
        """Shape the latest heartbeat event into a native-v2-level ``.status.snapshot``.

        Reprojects the runner's ``metrics_heartbeat`` (counters + percentile-projected
        TTFT / ITL / request-latency sketches) into the report's metric vocabulary, so
        the live snapshot key-for-key foreshadows the final ``native-v2.json`` metrics.
        """
        event = self._latest_event
        if event is None:
            return None
        metrics: dict[str, Any] = {}
        for source, tag in (
            ("ttft_ms", "time_to_first_token"),
            ("itl_ms", "inter_token_latency"),
            ("latency_ms", "request_latency"),
        ):
            sketch = event.get(source)
            if isinstance(sketch, dict) and sketch.get("count"):
                metrics[tag] = {
                    "unit": "ms",
                    "count": sketch.get("count"),
                    "min": sketch.get("min"),
                    "max": sketch.get("max"),
                    "percentiles": sketch.get("percentiles") or {},
                }
        return {
            "source": "live",
            "counters": event.get("counters") or {},
            "saturation": event.get("saturation") or {},
            "metrics": metrics,
        }
