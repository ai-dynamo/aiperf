# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8s API pollers for the watch command."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import job_selector
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)
from aiperf.kubernetes.watch_models import (
    EventSnapshot,
    MetricsSnapshot,
    PodSnapshot,
    ProgressSnapshot,
    WorkersSnapshot,
)
from aiperf.operator.status import parse_timestamp

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger(__name__)

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


class CRPoller:
    """Polls AIPerfJob CR status for phase, metrics, conditions, progress."""

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        self._api = api
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
        except Exception:
            logger.debug(f"Failed to fetch CR {self._job_id}", exc_info=True)
            return None


class PodPoller:
    """Polls K8s Pod API for pod status and container states."""

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        self._api = api
        self._job_id = job_id
        self._namespace = namespace
        self.pods: list[PodSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest pod status."""
        core = client.CoreV1Api(self._api)
        try:
            pod_list = await core.list_namespaced_pod(
                self._namespace,
                label_selector=job_selector(self._job_id),
            )
        except ApiException:
            return
        self.pods = [PodSnapshot.from_raw(_pod_to_raw(p)) for p in pod_list.items]


class EventPoller:
    """Polls K8s Event API filtered to this job's resources."""

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        self._api = api
        self._job_id = job_id
        self._namespace = namespace
        self.events: list[EventSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest events."""
        core = client.CoreV1Api(self._api)
        try:
            ev_list = await core.list_namespaced_event(self._namespace)
        except ApiException:
            return

        filtered = []
        for ev in ev_list.items:
            involved = ev.involved_object
            involved_name = involved.name if involved and involved.name else ""
            if self._job_id not in involved_name:
                continue
            ts = ev.last_timestamp
            filtered.append(
                EventSnapshot(
                    timestamp=ts.isoformat() if ts else "",
                    type=ev.type or "",
                    reason=ev.reason or "",
                    object=involved_name,
                    message=ev.message or "",
                    count=1,
                )
            )

        self.events = sorted(filtered, key=lambda e: e.timestamp)[-20:]


def _pod_to_raw(pod: Any) -> dict[str, Any]:
    """Serialize a V1Pod back to the raw dict shape PodSnapshot.from_raw expects."""
    metadata = pod.metadata
    status = pod.status
    raw_metadata: dict[str, Any] = {}
    if metadata:
        raw_metadata = {
            "name": metadata.name or "",
            "namespace": metadata.namespace or "",
        }
        if metadata.creation_timestamp:
            ts = metadata.creation_timestamp
            raw_metadata["creationTimestamp"] = (
                ts.isoformat() if isinstance(ts, datetime) else str(ts)
            )
        if metadata.labels:
            raw_metadata["labels"] = dict(metadata.labels)

    raw_status: dict[str, Any] = {}
    if status:
        if status.phase:
            raw_status["phase"] = status.phase
        containers_raw: list[dict[str, Any]] = []
        for cs in status.container_statuses or []:
            entry = {
                "name": cs.name or "",
                "ready": bool(cs.ready),
                "restartCount": cs.restart_count or 0,
            }
            state_dict: dict[str, Any] = {}
            if cs.state:
                if cs.state.running is not None:
                    state_dict["running"] = {}
                elif cs.state.waiting is not None:
                    w = cs.state.waiting
                    state_dict["waiting"] = {
                        k: v
                        for k, v in {"reason": w.reason, "message": w.message}.items()
                        if v is not None
                    }
                elif cs.state.terminated is not None:
                    t = cs.state.terminated
                    state_dict["terminated"] = {
                        k: v
                        for k, v in {
                            "reason": t.reason,
                            "message": t.message,
                            "exitCode": t.exit_code,
                        }.items()
                        if v is not None
                    }
            if state_dict:
                entry["state"] = state_dict
            containers_raw.append(entry)
        if containers_raw:
            raw_status["containerStatuses"] = containers_raw

    return {"metadata": raw_metadata, "status": raw_status}


def _metric_avg(metrics: dict, key: str) -> float:
    m = metrics.get(key, {})
    return m.get("avg", 0.0) if isinstance(m, dict) else 0.0


def _metric_stat(metrics: dict, key: str, stat: str) -> float:
    m = metrics.get(key, {})
    return m.get(stat, 0.0) if isinstance(m, dict) else 0.0


def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()
