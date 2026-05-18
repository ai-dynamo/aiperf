# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the Kubernetes watch orchestrator.

Focuses on:
- default job resolution returning no target before any Kubernetes I/O starts;
- renderer lifecycle ordering across start, render, terminal, and exception paths;
- poller exception isolation so one failing poll does not suppress rendering;
- signal-handler registration constraints for Ctrl-C / termination.

Out of scope: CR, pod, event parsing details covered by sibling watch poller tests.
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from types import TracebackType

import pytest
from pytest import param

from aiperf.kubernetes import cli_helpers, watch_orchestrator, watch_pollers
from aiperf.kubernetes.watch_models import WatchSnapshot
from aiperf.kubernetes.watch_orchestrator import WatchOrchestrator
from aiperf.operator.status import Phase


# ============================================================================
# Helpers
# ============================================================================


class _FakeK8sContext:
    """Async context manager recording whether Kubernetes access was opened."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self) -> object:
        self._events.append("k8s:enter")
        return object()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._events.append("k8s:exit")
        return False


class _RecordingRenderer:
    """Renderer double that records lifecycle calls and captured snapshots."""

    def __init__(
        self,
        events: list[str],
        *,
        start_error: RuntimeError | None = None,
        render_error: RuntimeError | None = None,
    ) -> None:
        self.events = events
        self.snapshots: list[WatchSnapshot] = []
        self._start_error = start_error
        self._render_error = render_error

    def start(self) -> None:
        self.events.append("renderer:start")
        if self._start_error is not None:
            raise self._start_error

    def render(self, snapshot: WatchSnapshot) -> None:
        self.events.append(f"renderer:render:{snapshot.phase}")
        self.snapshots.append(snapshot)
        if self._render_error is not None:
            raise self._render_error

    def stop(self) -> None:
        self.events.append("renderer:stop")


class _FakeCRPoller:
    """CR poller double exposing the state consumed by WatchSnapshot."""

    def __init__(
        self,
        events: list[str],
        *,
        phase: str = Phase.RUNNING,
        poll_error: RuntimeError | None = None,
    ) -> None:
        self.events = events
        self.phase = phase
        self.current_phase = "warmup"
        self.elapsed_seconds = 42.0
        self.progress = None
        self.metrics = None
        self.workers = None
        self.conditions: dict[str, bool] = {}
        self.raw_metrics: dict[str, object] = {}
        self.server_metrics: dict[str, object] = {}
        self.model = "meta-llama/Llama-3-8B"
        self.endpoint = "http://localhost:8000/v1/chat/completions"
        self.image = "nvcr.io/nvidia/aiperf:bench-2026-05-18"
        self.results = None
        self.error = None
        self._poll_error = poll_error

    async def poll(self) -> None:
        self.events.append("cr:poll")
        if self._poll_error is not None:
            raise self._poll_error


class _FakePodPoller:
    """Pod poller double with optional poll failure."""

    def __init__(
        self,
        events: list[str],
        *,
        poll_error: RuntimeError | None = None,
    ) -> None:
        self.events = events
        self.pods = []
        self._poll_error = poll_error

    async def poll(self) -> None:
        self.events.append("pod:poll")
        if self._poll_error is not None:
            raise self._poll_error


class _FakeEventPoller:
    """Event poller double with optional poll failure."""

    def __init__(
        self,
        events: list[str],
        *,
        poll_error: RuntimeError | None = None,
    ) -> None:
        self._timeline = events
        self.events: list[object] = []
        self._poll_error = poll_error

    async def poll(self) -> None:
        self._timeline.append("event:poll")
        if self._poll_error is not None:
            raise self._poll_error


def _install_fake_watch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    cr_poller: _FakeCRPoller,
    pod_poller: _FakePodPoller | None = None,
    event_poller: _FakeEventPoller | None = None,
) -> None:
    """Replace Kubernetes I/O and poller construction with deterministic fakes."""
    pod = pod_poller or _FakePodPoller(events)
    event = event_poller or _FakeEventPoller(events)

    def k8s_client(*, kubeconfig: str | None = None, context: str | None = None) -> _FakeK8sContext:
        events.append(f"k8s:client:{kubeconfig or 'default'}:{context or 'default'}")
        return _FakeK8sContext(events)

    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", k8s_client)
    monkeypatch.setattr(watch_pollers, "CRPoller", lambda api, job_id, ns: cr_poller)
    monkeypatch.setattr(watch_pollers, "PodPoller", lambda api, job_id, ns: pod)
    monkeypatch.setattr(watch_pollers, "EventPoller", lambda api, job_id, ns: event)
    monkeypatch.setattr(
        WatchOrchestrator,
        "_install_signal_handlers",
        lambda self: events.append("signals:install"),
    )


# ============================================================================
# Resolution and quiet-mode edges
# ============================================================================


class TestWatchOrchestratorResolutionEdges:
    """Default target resolution must fail closed before cluster access."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "quiet",
        [
            False,
            True,
        ],
    )  # fmt: skip
    async def test_run_resolver_returns_none_skips_renderer_and_kubernetes_io(
        self, monkeypatch: pytest.MonkeyPatch, quiet: bool
    ) -> None:
        calls: list[tuple[object, str, bool]] = []
        renderer = _RecordingRenderer([])

        def resolve(job_id: object, namespace: str, *, quiet: bool) -> tuple[str, str] | None:
            calls.append((job_id, namespace, quiet))
            return None

        def forbidden_k8s_client(
            *, kubeconfig: str | None = None, context: str | None = None
        ) -> _FakeK8sContext:
            raise AssertionError("resolver miss must not open Kubernetes client")

        monkeypatch.setattr(cli_helpers, "resolve_job_id_and_namespace", resolve)
        monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", forbidden_k8s_client)

        orchestrator = WatchOrchestrator(renderer=renderer, quiet=quiet)

        await orchestrator.run()

        assert calls == [(None, "aiperf-benchmarks", quiet)]
        assert renderer.events == []


# ============================================================================
# Renderer lifecycle and terminal-state edges
# ============================================================================


class TestWatchOrchestratorRendererLifecycleEdges:
    """Renderer start/stop bracketing is observable across failure modes."""

    @pytest.mark.asyncio
    async def test_run_renderer_render_raises_still_stops_renderer_before_exiting_k8s(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        renderer = _RecordingRenderer(
            events,
            render_error=RuntimeError("renderer render failed for aiperf-bench-7f2a"),
        )
        _install_fake_watch_runtime(
            monkeypatch,
            events,
            cr_poller=_FakeCRPoller(events, phase=Phase.RUNNING),
        )
        orchestrator = WatchOrchestrator(
            job_id="aiperf-bench-7f2a",
            namespace="aiperf-bench",
            renderer=renderer,
            interval=0.01,
        )

        with pytest.raises(RuntimeError, match=r"renderer render failed.*aiperf-bench-7f2a"):
            await orchestrator.run()

        assert events == [
            "k8s:client:default:default",
            "k8s:enter",
            "signals:install",
            "renderer:start",
            "cr:poll",
            "pod:poll",
            "event:poll",
            "renderer:render:Running",
            "renderer:stop",
            "k8s:exit",
        ]

    @pytest.mark.asyncio
    async def test_run_renderer_start_raises_does_not_call_stop_for_unstarted_renderer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        renderer = _RecordingRenderer(
            events,
            start_error=RuntimeError("renderer start failed for aiperf-bench-7f2a"),
        )
        _install_fake_watch_runtime(
            monkeypatch,
            events,
            cr_poller=_FakeCRPoller(events, phase=Phase.RUNNING),
        )
        orchestrator = WatchOrchestrator(
            job_id="aiperf-bench-7f2a",
            namespace="aiperf-bench",
            renderer=renderer,
        )

        with pytest.raises(RuntimeError, match=r"renderer start failed.*aiperf-bench-7f2a"):
            await orchestrator.run()

        assert events == [
            "k8s:client:default:default",
            "k8s:enter",
            "signals:install",
            "renderer:start",
            "k8s:exit",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terminal_phase",
        [
            param(Phase.COMPLETED, id="completed-stops-without-sleep"),
            param(Phase.FAILED, id="failed-stops-without-sleep"),
            param(Phase.CANCELLED, id="cancelled-stops-without-sleep"),
        ],
    )  # fmt: skip
    async def test_run_terminal_phase_renders_once_and_skips_interval_sleep(
        self, monkeypatch: pytest.MonkeyPatch, terminal_phase: Phase
    ) -> None:
        events: list[str] = []
        renderer = _RecordingRenderer(events)
        _install_fake_watch_runtime(
            monkeypatch,
            events,
            cr_poller=_FakeCRPoller(events, phase=terminal_phase),
        )

        async def forbidden_sleep(delay: float) -> None:
            raise AssertionError(f"terminal watch must not sleep for {delay}s")

        monkeypatch.setattr(watch_orchestrator.asyncio, "sleep", forbidden_sleep)
        orchestrator = WatchOrchestrator(
            job_id="aiperf-bench-7f2a",
            namespace="aiperf-bench",
            renderer=renderer,
            interval=90.0,
        )

        await orchestrator.run()

        assert [snapshot.phase for snapshot in renderer.snapshots] == [terminal_phase]
        assert events == [
            "k8s:client:default:default",
            "k8s:enter",
            "signals:install",
            "renderer:start",
            "cr:poll",
            "pod:poll",
            "event:poll",
            "renderer:render:" + terminal_phase,
            "renderer:stop",
            "k8s:exit",
        ]


# ============================================================================
# Poller and signal adversaries
# ============================================================================


class TestWatchOrchestratorPollerAndSignalEdges:
    """Poll failures are isolated; process signals only flip the watch loop flag."""

    @pytest.mark.asyncio
    async def test_run_poller_exception_still_renders_snapshot_from_last_known_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        renderer = _RecordingRenderer(events)
        _install_fake_watch_runtime(
            monkeypatch,
            events,
            cr_poller=_FakeCRPoller(
                events,
                phase=Phase.COMPLETED,
                poll_error=RuntimeError("apiserver read timeout for aiperf-bench-7f2a"),
            ),
            pod_poller=_FakePodPoller(
                events,
                poll_error=RuntimeError("pod list failed for aiperf-bench-7f2a"),
            ),
        )
        orchestrator = WatchOrchestrator(
            job_id="aiperf-bench-7f2a",
            namespace="aiperf-bench",
            renderer=renderer,
        )

        await orchestrator.run()

        assert [snapshot.phase for snapshot in renderer.snapshots] == [Phase.COMPLETED]
        assert renderer.snapshots[0].job_id == "aiperf-bench-7f2a"
        assert events == [
            "k8s:client:default:default",
            "k8s:enter",
            "signals:install",
            "renderer:start",
            "cr:poll",
            "pod:poll",
            "event:poll",
            "renderer:render:Completed",
            "renderer:stop",
            "k8s:exit",
        ]

    @pytest.mark.asyncio
    async def test_install_signal_handlers_registers_sigint_and_sigterm_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registrations: list[tuple[signal.Signals, Callable[[], None]]] = []
        loop = watch_orchestrator.asyncio.get_running_loop()

        def add_signal_handler(sig: signal.Signals, callback: Callable[[], None]) -> None:
            registrations.append((sig, callback))

        monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
        orchestrator = WatchOrchestrator(job_id="aiperf-bench-7f2a")

        orchestrator._install_signal_handlers()

        assert [sig for sig, _ in registrations] == [signal.SIGINT, signal.SIGTERM]
        for _, callback in registrations:
            callback()
        assert orchestrator._running is False
