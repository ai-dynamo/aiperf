# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch orchestrator: coordinates pollers, diagnosis, and rendering."""

from __future__ import annotations

import asyncio
import dataclasses
import signal
from datetime import UTC, datetime
from typing import Protocol

from aiperf.common.constants import IS_WINDOWS
from aiperf.kubernetes import cli_helpers
from aiperf.kubernetes.watch_models import WatchSnapshot
from aiperf.operator.status import Phase


class WatchRenderer(Protocol):
    """Structural type for a ``WatchOrchestrator`` renderer.

    Any object that exposes the three methods below (text, Rich, JSON, test
    double, ...) satisfies this protocol. ``start()`` and ``stop()`` bracket
    the watch session, and ``render()`` is called once per poll cycle with
    the latest snapshot.
    """

    def start(self) -> None: ...
    def render(self, snapshot: WatchSnapshot) -> None: ...
    def stop(self) -> None: ...


class WatchOrchestrator:
    """Single-use driver that polls K8s, diagnoses state, and renders frames.

    ``run()`` owns the full lifecycle: it installs ``SIGINT``/``SIGTERM``
    handlers on the running event loop, polls the CR/pods/events on a fixed
    interval, produces a ``WatchSnapshot`` + ``Diagnosis``, hands each frame
    to the renderer, and exits when the job reaches a terminal phase
    (``Completed``/``Failed``/``Cancelled``) or the process receives a signal.
    Instances are **single-use** -- create a new one per watch session.

    Example:
        >>> orch = WatchOrchestrator(
        ...     job_id="aiperf-bench-7f2a",
        ...     namespace="aiperf-bench",
        ...     renderer=RichRenderer(),
        ...     interval=2.0,
        ... )
        >>> await orch.run()
    """

    def __init__(
        self,
        *,
        job_id: str | None = None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        all_jobs: bool = False,
        renderer: WatchRenderer | None = None,
        interval: float = 2.0,
        follow_logs: bool = False,
        quiet: bool = False,
    ) -> None:
        """Configure the orchestrator.

        Args:
            job_id: Target ``AIPerfJob`` or ``AIPerfSweep`` CR name. If
                ``None``, the target is resolved from the last benchmark via
                ``cli_helpers.resolve_target``.
            namespace: Kubernetes namespace for the CR. Falls back to
                ``DEFAULT_BENCHMARK_NAMESPACE`` when ``None``.
            kubeconfig: Optional path to a kubeconfig file; ``None`` uses the
                default kubeconfig discovery (``KUBECONFIG`` env, in-cluster,
                ``~/.kube/config``).
            kube_context: Optional kubeconfig context name to select.
            all_jobs: Reserved for future multi-job watch support; currently
                unused but preserved for CLI parity.
            renderer: Object matching the ``WatchRenderer`` protocol. When
                ``None``, the orchestrator runs silently (useful for tests).
            interval: Seconds between CR polls. Pod/event polls run every 3rd
                iteration.
            follow_logs: Reserved for future log-tailing support; currently
                unused but preserved for CLI parity.
            quiet: Suppress human status lines while resolving defaults.
        """
        self._job_id = job_id
        self._namespace = namespace
        self._kubeconfig = kubeconfig
        self._kube_context = kube_context
        self._all_jobs = all_jobs
        self._renderer = renderer
        self._interval = interval
        self._follow_logs = follow_logs
        self._quiet = quiet
        self._running = True

    async def run(self) -> None:
        """Main watch loop."""
        from aiperf.kubernetes.client import k8s_client
        from aiperf.kubernetes.watch_pollers import (
            CRPoller,
            EventPoller,
            PodPoller,
            SweepCRPoller,
        )

        resolved = await self._resolve_target()
        if resolved is None:
            return
        job_id, ns, target_kind = resolved

        async with k8s_client(
            kubeconfig=self._kubeconfig,
            context=self._kube_context,
        ) as api:
            cr_poller = (
                SweepCRPoller(api, job_id, ns)
                if target_kind == "AIPerfSweep"
                else CRPoller(api, job_id, ns)
            )
            child_job_ids_provider = (
                (lambda: cr_poller.child_job_ids)
                if target_kind == "AIPerfSweep"
                else None
            )
            pod_poller = PodPoller(
                api,
                job_id,
                ns,
                job_ids_provider=child_job_ids_provider,
            )
            event_poller = EventPoller(api, job_id, ns)

            self._install_signal_handlers()

            if self._renderer:
                self._renderer.start()
            try:
                await self._poll_loop(
                    job_id=job_id,
                    ns=ns,
                    cr_poller=cr_poller,
                    pod_poller=pod_poller,
                    event_poller=event_poller,
                )
            finally:
                if self._renderer:
                    self._renderer.stop()

    async def _resolve_target(self) -> tuple[str, str, str] | None:
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        ns = self._namespace or DEFAULT_BENCHMARK_NAMESPACE
        target = await cli_helpers.resolve_target(
            self._job_id,
            ns,
            kubeconfig=self._kubeconfig,
            kube_context=self._kube_context,
            quiet=self._quiet,
        )
        if target is None:
            return None
        try:
            target_kind = (
                "AIPerfSweep" if hasattr(target, "sweep_info") else "AIPerfJob"
            )
            return target.name, target.namespace, target_kind
        finally:
            await target.aclose()

    def _install_signal_handlers(self) -> None:
        # ``run()`` always invokes this from inside the running loop;
        # ``get_event_loop()`` is deprecated in this context and raises in newer
        # Python.
        loop = asyncio.get_running_loop()
        # Windows ProactorEventLoop does not implement add_signal_handler and
        # raises NotImplementedError, which would abort ``aiperf kube watch``
        # before any polling starts. Fall back to signal.signal() (supported on
        # Windows for SIGINT/SIGBREAK), mirroring
        # aiperf.controller.system_mixins.SignalHandlerMixin.
        if IS_WINDOWS:

            def windows_signal_handler(_sig: int, _frame: object) -> None:
                self._stop()

            signal.signal(signal.SIGINT, windows_signal_handler)
            sigbreak = getattr(signal, "SIGBREAK", None)
            if sigbreak is not None:
                signal.signal(sigbreak, windows_signal_handler)
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop)

    async def _poll_loop(
        self,
        *,
        job_id: str,
        ns: str,
        cr_poller: object,
        pod_poller: object,
        event_poller: object,
    ) -> None:
        from aiperf.kubernetes.watch_diagnosis import diagnose

        poll_count = 0
        while self._running:
            tasks = []
            if cr_poller.target_kind == "AIPerfSweep":
                await asyncio.gather(cr_poller.poll(), return_exceptions=True)
            else:
                tasks.append(cr_poller.poll())
            # Pod and event polling is slower, do it less frequently
            if poll_count % 3 == 0:
                tasks.append(pod_poller.poll())
                tasks.append(event_poller.poll())

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            snapshot = self._build_snapshot(
                job_id=job_id,
                ns=ns,
                cr_poller=cr_poller,
                pod_poller=pod_poller,
                event_poller=event_poller,
            )
            snapshot = dataclasses.replace(snapshot, diagnosis=diagnose(snapshot))

            if self._renderer:
                self._renderer.render(snapshot)

            if self._is_terminal_phase(cr_poller.target_kind, cr_poller.phase):
                break

            poll_count += 1
            await asyncio.sleep(self._interval)

    @staticmethod
    def _is_terminal_phase(target_kind: str, phase: str) -> bool:
        if target_kind == "AIPerfSweep":
            return phase in {"Succeeded", "PartiallyFailed", "Failed", "Cancelled"}
        return phase in {Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED}

    @staticmethod
    def _build_snapshot(
        *,
        job_id: str,
        ns: str,
        cr_poller: object,
        pod_poller: object,
        event_poller: object,
    ) -> WatchSnapshot:
        return WatchSnapshot(
            timestamp=datetime.now(UTC),
            job_id=job_id,
            namespace=ns,
            phase=cr_poller.phase,
            target_kind=cr_poller.target_kind,
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
            sweep_runs_completed=cr_poller.sweep_runs_completed,
            sweep_runs_failed=cr_poller.sweep_runs_failed,
            sweep_runs_cancelled=cr_poller.sweep_runs_cancelled,
            sweep_runs_total=cr_poller.sweep_runs_total,
            error=cr_poller.error,
        )

    def _stop(self) -> None:
        self._running = False
