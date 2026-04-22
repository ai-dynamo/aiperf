# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch orchestrator: coordinates pollers, diagnosis, and rendering."""

from __future__ import annotations

import asyncio
import dataclasses
import signal
from datetime import datetime, timezone
from typing import Any

from aiperf.kubernetes.watch_models import WatchSnapshot
from aiperf.operator.status import Phase


class WatchOrchestrator:
    """Coordinates K8s polling, diagnosis, and rendering into a refresh loop."""

    def __init__(
        self,
        *,
        job_id: str | None = None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        all_jobs: bool = False,
        renderer: Any = None,
        interval: float = 2.0,
        follow_logs: bool = False,
    ) -> None:
        self._job_id = job_id
        self._namespace = namespace
        self._kubeconfig = kubeconfig
        self._kube_context = kube_context
        self._all_jobs = all_jobs
        self._renderer = renderer
        self._interval = interval
        self._follow_logs = follow_logs
        self._running = True

    async def run(self) -> None:
        """Main watch loop."""
        from aiperf.kubernetes import cli_helpers
        from aiperf.kubernetes.client import k8s_client
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE
        from aiperf.kubernetes.watch_diagnosis import diagnose
        from aiperf.kubernetes.watch_pollers import CRPoller, EventPoller, PodPoller

        ns = self._namespace or DEFAULT_BENCHMARK_NAMESPACE

        # Resolve job_id
        if self._job_id:
            job_id = self._job_id
        else:
            resolved = cli_helpers.resolve_job_id_and_namespace(None, ns)
            if not resolved:
                return
            job_id, ns = resolved

        async with k8s_client(
            kubeconfig=self._kubeconfig,
            context=self._kube_context,
        ) as api:
            cr_poller = CRPoller(api, job_id, ns)
            pod_poller = PodPoller(api, job_id, ns)
            event_poller = EventPoller(api, job_id, ns)

            # Handle Ctrl+C
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop)

            if self._renderer:
                self._renderer.start()

            poll_count = 0
            try:
                while self._running:
                    # Poll all sources concurrently
                    tasks = [cr_poller.poll()]
                    # Pod and event polling is slower, do it less frequently
                    if poll_count % 3 == 0:
                        tasks.append(pod_poller.poll())
                        tasks.append(event_poller.poll())

                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Build snapshot
                    snapshot = WatchSnapshot(
                        timestamp=datetime.now(timezone.utc),
                        job_id=job_id,
                        namespace=ns,
                        phase=cr_poller.phase,
                        current_phase=cr_poller.current_phase,
                        elapsed_seconds=cr_poller.elapsed_seconds,
                        progress=cr_poller.progress,
                        metrics=cr_poller.metrics,
                        workers=cr_poller.workers,
                        pods=pod_poller.pods,
                        events=event_poller.events,
                        conditions=cr_poller.conditions,
                        raw_metrics=cr_poller.raw_metrics,
                        server_metrics=cr_poller.server_metrics,
                        model=cr_poller.model,
                        endpoint=cr_poller.endpoint,
                        image=cr_poller.image,
                        results=cr_poller.results,
                        error=cr_poller.error,
                    )

                    # Run diagnosis
                    diagnosis = diagnose(snapshot)
                    snapshot = dataclasses.replace(snapshot, diagnosis=diagnosis)

                    # Render
                    if self._renderer:
                        self._renderer.render(snapshot)

                    # Exit on terminal phase
                    if cr_poller.phase in (
                        Phase.COMPLETED,
                        Phase.FAILED,
                        Phase.CANCELLED,
                    ):
                        break

                    poll_count += 1
                    await asyncio.sleep(self._interval)

            finally:
                if self._renderer:
                    self._renderer.stop()

    def _stop(self) -> None:
        self._running = False
