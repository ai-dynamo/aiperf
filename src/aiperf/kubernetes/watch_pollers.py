# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8s API pollers for the watch command."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    AIPERF_SWEEP_GROUP,
    AIPERF_SWEEP_PLURAL,
    AIPERF_SWEEP_VERSION,
)
from aiperf.kubernetes.watch_models import (
    MetricsSnapshot,
    ProgressSnapshot,
    WorkersSnapshot,
)

# Re-exported so callers (watch_orchestrator) and tests can patch the full
# poller set on a single module. The pod/event pollers live in a sibling module
# to keep file size down, but they are part of the watch_pollers public surface.
from aiperf.kubernetes.watch_pod_event_pollers import EventPoller, PodPoller
from aiperf.operator.status import parse_timestamp

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger(__name__)

__all__ = [
    "CRPoller",
    "EventPoller",
    "PodPoller",
    "SweepCRPoller",
]

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


class CRPoller:
    """Polls AIPerfJob CR status for phase, metrics, conditions, and progress.

    State is held on instance attributes that are refreshed each time
    ``poll()`` runs. Callers **must** ``await poll()`` at least once before
    reading any attribute; until the first successful poll the fields default
    to empty collections, ``None``, or ``"Unknown"``.

    ``poll()`` is idempotent and failure-tolerant: transient K8s API errors
    are logged at DEBUG and the existing field values are left untouched, so
    a caller can safely invoke it on a fixed interval without try/except.

    Example:
        >>> async with k8s_client() as api:
        ...     poller = CRPoller(api, "aiperf-bench-7f2a", "aiperf-bench")
        ...     await poller.poll()
        ...     print(poller.phase, poller.workers.ready, poller.workers.total)
    """

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        self._api = api
        self._job_id = job_id
        self._namespace = namespace
        self.target_kind: str = "AIPerfJob"
        self.phase: str = "Unknown"
        self.current_phase: str | None = None
        self.workers: WorkersSnapshot = WorkersSnapshot()
        self.metrics: MetricsSnapshot | None = None
        self.progress: ProgressSnapshot | None = None
        self.conditions: dict[str, bool] = {}
        self.elapsed_seconds: float = 0.0
        self.model: str | None = None
        self.endpoint: str | None = None
        self.image: str | None = None
        self.results: dict[str, Any] | None = None
        self.error: str | None = None
        self.sweep_runs_completed: int | None = None
        self.sweep_runs_failed: int | None = None
        self.sweep_runs_cancelled: int | None = None
        self.sweep_runs_total: int | None = None
        self.child_job_ids: list[str] = []
        self.start_time: datetime | None = None
        self.raw_metrics: dict[str, Any] = {}
        self.server_metrics: dict[str, Any] = {}

    async def poll(self) -> None:
        """Fetch the latest CR state and populate this poller's fields."""
        raw = await self._get_raw_cr()
        if not raw:
            return

        status = raw.get("status", {})
        spec = raw.get("spec", {})

        self.phase = status.get("phase", "Pending")
        self.current_phase = status.get("currentPhase")
        self.error = status.get("error")

        self._populate_workers(status)
        self._populate_elapsed_time(status)
        self._populate_metrics(status)
        self._populate_progress(status)
        self._populate_conditions(status)
        self._populate_metadata(spec, status)
        self._populate_results(status)
        self._populate_summary_metrics(status)

    def _populate_workers(self, status: dict[str, Any]) -> None:
        workers = status.get("workers", {})
        self.workers = WorkersSnapshot(
            ready=workers.get("ready", 0),
            total=workers.get("total", 0),
        )

    def _populate_elapsed_time(self, status: dict[str, Any]) -> None:
        start_str = status.get("startTime")
        if not start_str:
            return
        try:
            self.start_time = parse_timestamp(start_str)
            self.elapsed_seconds = (datetime.now(UTC) - self.start_time).total_seconds()
        except (ValueError, TypeError):
            pass

    def _populate_metrics(self, status: dict[str, Any]) -> None:
        live = status.get("liveMetrics", {})
        metrics_dict = live.get("metrics", {})
        self.raw_metrics = metrics_dict
        self.server_metrics = status.get("serverMetrics", {})

        if not metrics_dict:
            return

        request_count = int(_metric_avg(metrics_dict, "request_count"))
        error_count = int(_metric_avg(metrics_dict, "error_count"))
        rps = _metric_avg(metrics_dict, "request_throughput")
        # Invariant: 0 <= goodput <= rps. error_count and request_count are
        # averaged independently from staggered windows in liveMetrics, so
        # error_count > request_count is observable and would otherwise yield
        # a negative goodput.
        if request_count > 0:
            goodput = rps * max(0.0, (request_count - error_count) / request_count)
        else:
            goodput = 0.0
        self.metrics = MetricsSnapshot(
            request_throughput_rps=rps,
            request_latency_avg_ms=_metric_avg(metrics_dict, "request_latency"),
            request_latency_p50_ms=_metric_stat(metrics_dict, "request_latency", "p50"),
            request_latency_p99_ms=_metric_stat(metrics_dict, "request_latency", "p99"),
            ttft_avg_ms=_metric_avg(metrics_dict, "time_to_first_token"),
            ttft_p50_ms=_metric_stat(metrics_dict, "time_to_first_token", "p50"),
            ttft_p99_ms=_metric_stat(metrics_dict, "time_to_first_token", "p99"),
            time_to_second_token_avg_ms=_metric_avg(
                metrics_dict, "time_to_second_token"
            ),
            inter_token_latency_avg_ms=_metric_avg(metrics_dict, "inter_token_latency"),
            inter_token_latency_p99_ms=_metric_stat(
                metrics_dict, "inter_token_latency", "p99"
            ),
            output_token_throughput_tps=_metric_avg(
                metrics_dict, "output_token_throughput"
            ),
            total_token_throughput_tps=_metric_avg(
                metrics_dict, "total_token_throughput"
            ),
            prefill_throughput_per_user_tps=_metric_avg(
                metrics_dict, "prefill_throughput_per_user"
            ),
            output_token_throughput_per_user_tps=_metric_avg(
                metrics_dict, "output_token_throughput_per_user"
            ),
            request_count=request_count,
            error_count=error_count,
            goodput_rps=goodput,
            streaming=live.get("streaming", False),
        )

    def _populate_progress(self, status: dict[str, Any]) -> None:
        phases = status.get("phases", {})
        if not phases:
            return
        phase_key = self.current_phase or next(iter(phases), None)
        if not phase_key or phase_key not in phases:
            return
        p = phases[phase_key]
        # Use records progress when requests are done (drain phase)
        req_pct = p.get("requestsProgressPercent", 0.0)
        rec_pct = p.get("recordsProgressPercent", 0.0)
        sending_done = p.get("sendingComplete", False)
        pct = rec_pct if sending_done and rec_pct < 100 else req_pct
        self.progress = ProgressSnapshot(
            percent=pct,
            requests_completed=p.get("requestsCompleted", 0),
            requests_total=p.get("requestsTotal", 0),
            eta_seconds=p.get("requestsEtaSeconds"),
            duration_target_seconds=p.get("durationTargetSeconds"),
        )

    def _populate_conditions(self, status: dict[str, Any]) -> None:
        for cond in status.get("conditions", []):
            cond_type = cond.get("type", "")
            snake = _camel_to_snake(cond_type)
            self.conditions[snake] = cond.get("status") == "True"

    def _populate_metadata(self, spec: dict[str, Any], status: dict[str, Any]) -> None:
        # models/endpoint accept multiple shorthand forms per the CRD;
        # normalize before indexing to avoid KeyError/TypeError when the user
        # supplies a dict or scalar string.
        benchmark = spec.get("benchmark", {})
        models_cfg = benchmark.get("models", [])
        if isinstance(models_cfg, list):
            model_items = models_cfg
        elif isinstance(models_cfg, dict):
            model_items = models_cfg.get("items") or models_cfg.get("modelNames") or []
        else:
            model_items = []
        if model_items:
            first = model_items[0]
            self.model = (
                first.get("name", first) if isinstance(first, dict) else str(first)
            )
        else:
            self.model = status.get("model")

        urls = benchmark.get("endpoint", {}).get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        self.endpoint = urls[0] if urls else status.get("endpoint")
        self.image = spec.get("image")

    def _populate_results(self, status: dict[str, Any]) -> None:
        if self.phase == "Completed":
            self.results = status.get("results")

    def _populate_summary_metrics(self, status: dict[str, Any]) -> None:
        """Populate ``self.metrics`` from ``status.liveSummary`` / ``status.summary``.

        Live ``status.liveMetrics.metrics`` is preferred (handled by
        ``_populate_metrics``); this is the fallback for archived jobs whose
        ``liveMetrics`` was already pruned. The summary is the curated nested
        ``{metric_tag: {avg, p50, p99, ...}}`` shape written by
        ``MetricsSummary.from_metrics`` — same shape as ``liveMetrics.metrics``,
        so ``_metric_stat`` works for both.
        """
        summary = status.get("liveSummary") or status.get("summary")
        if not summary or self.metrics:
            return
        self.metrics = MetricsSnapshot(
            request_throughput_rps=_metric_avg(summary, "request_throughput"),
            request_latency_avg_ms=_metric_avg(summary, "request_latency"),
            request_latency_p99_ms=_metric_stat(summary, "request_latency", "p99"),
            ttft_avg_ms=_metric_avg(summary, "time_to_first_token"),
            ttft_p99_ms=_metric_stat(summary, "time_to_first_token", "p99"),
            output_token_throughput_tps=_metric_avg(summary, "output_token_throughput"),
            total_token_throughput_tps=_metric_avg(summary, "total_token_throughput"),
        )

    async def _get_raw_cr(self) -> dict[str, Any] | None:
        """Get the raw AIPerfJob CR dict from the K8s API."""
        custom = client.CustomObjectsApi(self._api)
        try:
            raw = await custom.get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=self._namespace,
                name=self._job_id,
            )
            return raw
        except ApiException as e:
            if e.status != 404:
                logger.debug(f"Failed to fetch CR {self._job_id}: {e}")
            return None
        except (TimeoutError, aiohttp.ClientError, OSError):
            logger.debug(f"Failed to fetch CR {self._job_id}", exc_info=True)
            return None


class SweepCRPoller(CRPoller):
    """Polls AIPerfSweep CR status for phase and child-run progress."""

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        super().__init__(api, job_id, namespace)
        self.target_kind = "AIPerfSweep"

    async def poll(self) -> None:
        """Fetch the latest AIPerfSweep state and populate this poller's fields."""
        raw = await self._get_raw_cr()
        if not raw:
            return

        status = raw.get("status", {})
        spec = raw.get("spec", {})

        self.phase = status.get("phase", "Pending")
        self.current_phase = None
        self.error = status.get("error")

        self._populate_elapsed_time(status)
        self._populate_conditions(status)
        self._populate_metadata(spec, status)
        self._populate_sweep_runs(status)
        self._populate_child_job_ids(status)
        self._populate_results(status)

    def _populate_sweep_runs(self, status: dict[str, Any]) -> None:
        run_states = status.get("runStates", {})
        self.sweep_runs_completed = _as_int(
            run_states.get("completed", status.get("completedRuns"))
        )
        self.sweep_runs_failed = _as_int(
            run_states.get("failed", status.get("failedRuns"))
        )
        self.sweep_runs_cancelled = _as_int(
            run_states.get("cancelled", status.get("cancelledRuns"))
        )
        total = status.get("totalRuns") or status.get("maxTotalRuns")
        if total is None:
            total = sum(
                count or 0
                for count in (
                    self.sweep_runs_completed,
                    self.sweep_runs_failed,
                    self.sweep_runs_cancelled,
                    _as_int(run_states.get("pending")),
                    _as_int(run_states.get("running")),
                )
            )
        self.sweep_runs_total = _as_int(total)

    def _populate_child_job_ids(self, status: dict[str, Any]) -> None:
        """Populate child AIPerfJob names from live and terminal sweep status."""
        child_job_ids: list[str] = []
        seen: set[str] = set()

        def append_child_name(child_name: object) -> None:
            if not isinstance(child_name, str) or not child_name or child_name in seen:
                return
            child_job_ids.append(child_name)
            seen.add(child_name)

        current_child_ref = status.get("currentChildRef")
        if isinstance(current_child_ref, dict):
            append_child_name(current_child_ref.get("name"))

        for run in status.get("runs", []):
            if not isinstance(run, dict):
                continue
            append_child_name(run.get("childName"))
        self.child_job_ids = child_job_ids

    async def _get_raw_cr(self) -> dict[str, Any] | None:
        """Get the raw AIPerfSweep CR dict from the K8s API."""
        custom = client.CustomObjectsApi(self._api)
        try:
            raw = await custom.get_namespaced_custom_object(
                group=AIPERF_SWEEP_GROUP,
                version=AIPERF_SWEEP_VERSION,
                plural=AIPERF_SWEEP_PLURAL,
                namespace=self._namespace,
                name=self._job_id,
            )
            return raw
        except ApiException as e:
            if e.status != 404:
                logger.debug(f"Failed to fetch sweep CR {self._job_id}: {e}")
            return None
        except (TimeoutError, aiohttp.ClientError, OSError):
            logger.debug(f"Failed to fetch sweep CR {self._job_id}", exc_info=True)
            return None


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _metric_avg(metrics: dict, key: str) -> float:
    m = metrics.get(key, {})
    return m.get("avg", 0.0) if isinstance(m, dict) else 0.0


def _metric_stat(metrics: dict, key: str, stat: str) -> float:
    m = metrics.get(key, {})
    return m.get(stat, 0.0) if isinstance(m, dict) else 0.0


def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()
