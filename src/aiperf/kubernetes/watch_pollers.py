# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8s API pollers for the watch command."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from aiperf.kubernetes.watch_models import (
    EventSnapshot,
    MetricsSnapshot,
    PodSnapshot,
    ProgressSnapshot,
    WorkersSnapshot,
)
from aiperf.operator.status import parse_timestamp

logger = logging.getLogger(__name__)

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


class CRPoller:
    """Polls AIPerfJob CR status for phase, metrics, conditions, progress."""

    def __init__(self, client: Any, job_id: str, namespace: str) -> None:
        self._client = client
        self._job_id = job_id
        self._namespace = namespace
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
        self.start_time: datetime | None = None
        self.raw_metrics: dict[str, Any] = {}
        self.server_metrics: dict[str, Any] = {}

    async def poll(self) -> None:
        """Fetch latest CR status."""
        raw = await self._get_raw_cr()
        if not raw:
            return

        status = raw.get("status", {})
        spec = raw.get("spec", {})

        self.phase = status.get("phase", "Pending")
        self.current_phase = status.get("currentPhase")
        self.error = status.get("error")

        # Workers
        workers = status.get("workers", {})
        self.workers = WorkersSnapshot(
            ready=workers.get("ready", 0),
            total=workers.get("total", 0),
        )

        # Elapsed time
        start_str = status.get("startTime")
        if start_str:
            try:
                self.start_time = parse_timestamp(start_str)
                self.elapsed_seconds = (
                    datetime.now(timezone.utc) - self.start_time
                ).total_seconds()
            except (ValueError, TypeError):
                pass

        # Live metrics from CR status
        live = status.get("liveMetrics", {})
        metrics_dict = live.get("metrics", {})

        # Raw pass-through: every metric the controller reports, unprocessed
        self.raw_metrics = metrics_dict

        # Server metrics (Prometheus scrapes from inference server)
        self.server_metrics = status.get("serverMetrics", {})

        if metrics_dict:
            request_count = int(_metric_avg(metrics_dict, "request_count"))
            error_count = int(_metric_avg(metrics_dict, "error_count"))
            rps = _metric_avg(metrics_dict, "request_throughput")
            goodput = (
                rps * ((request_count - error_count) / request_count)
                if request_count > 0
                else 0.0
            )
            self.metrics = MetricsSnapshot(
                request_throughput_rps=rps,
                request_latency_avg_ms=_metric_avg(metrics_dict, "request_latency"),
                request_latency_p50_ms=_metric_stat(
                    metrics_dict, "request_latency", "p50"
                ),
                request_latency_p99_ms=_metric_stat(
                    metrics_dict, "request_latency", "p99"
                ),
                ttft_avg_ms=_metric_avg(metrics_dict, "time_to_first_token"),
                ttft_p50_ms=_metric_stat(metrics_dict, "time_to_first_token", "p50"),
                ttft_p99_ms=_metric_stat(metrics_dict, "time_to_first_token", "p99"),
                time_to_second_token_avg_ms=_metric_avg(
                    metrics_dict, "time_to_second_token"
                ),
                inter_token_latency_avg_ms=_metric_avg(
                    metrics_dict, "inter_token_latency"
                ),
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

        # Progress from phases
        phases = status.get("phases", {})
        if phases:
            # Use the current phase or last phase
            phase_key = self.current_phase or next(iter(phases), None)
            if phase_key and phase_key in phases:
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

        # Conditions
        for cond in status.get("conditions", []):
            cond_type = cond.get("type", "")
            snake = _camel_to_snake(cond_type)
            self.conditions[snake] = cond.get("status") == "True"

        # Metadata (from spec or status). models/endpoint accept multiple
        # shorthand forms per the CRD; normalize before indexing to avoid
        # KeyError/TypeError when the user supplies a dict or scalar string.
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

        # Results on completion
        if self.phase == "Completed":
            self.results = status.get("results")

        # Summary metrics
        summary = status.get("liveSummary") or status.get("summary")
        if summary and not self.metrics:
            self.metrics = MetricsSnapshot(
                request_throughput_rps=summary.get("throughput_rps", 0),
                request_latency_avg_ms=summary.get("latency_avg_ms", 0),
                request_latency_p99_ms=summary.get("latency_p99_ms", 0),
                ttft_avg_ms=summary.get("ttft_avg_ms", 0),
                ttft_p99_ms=summary.get("ttft_p99_ms", 0),
            )

    async def _get_raw_cr(self) -> dict[str, Any] | None:
        """Get the raw AIPerfJob CR dict from the K8s API."""
        try:
            from aiperf.kubernetes.client import AsyncAIPerfJob

            async for j in self._client._api.async_get(
                AsyncAIPerfJob,
                namespace=self._namespace,
                field_selector=f"metadata.name={self._job_id}",
            ):
                return j.raw
        except Exception:
            logger.debug(f"Failed to fetch CR {self._job_id}", exc_info=True)
        return None


class PodPoller:
    """Polls K8s Pod API for pod status and container states."""

    def __init__(self, client: Any, job_id: str, namespace: str) -> None:
        self._client = client
        self._job_id = job_id
        self._namespace = namespace
        self.pods: list[PodSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest pod status."""
        raw_pods = await self._client.get_pods(
            self._namespace, self._client.job_selector(self._job_id)
        )
        self.pods = [PodSnapshot.from_raw(p.raw) for p in raw_pods]


class EventPoller:
    """Polls K8s Event API filtered to this job's resources."""

    def __init__(self, client: Any, job_id: str, namespace: str) -> None:
        self._client = client
        self._job_id = job_id
        self._namespace = namespace
        self.events: list[EventSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest events."""
        events = await self._client.get_events(self._namespace)

        filtered = []
        for ev in events:
            if self._job_id not in ev.involved_object:
                continue
            filtered.append(
                EventSnapshot(
                    timestamp=ev.last_timestamp.isoformat()
                    if ev.last_timestamp
                    else "",
                    type=ev.type,
                    reason=ev.reason,
                    object=ev.involved_object,
                    message=ev.message,
                    count=1,
                )
            )

        self.events = sorted(filtered, key=lambda e: e.timestamp)[-20:]


def _metric_avg(metrics: dict, key: str) -> float:
    m = metrics.get(key, {})
    return m.get("avg", 0.0) if isinstance(m, dict) else 0.0


def _metric_stat(metrics: dict, key: str, stat: str) -> float:
    m = metrics.get(key, {})
    return m.get(stat, 0.0) if isinstance(m, dict) else 0.0


def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()
