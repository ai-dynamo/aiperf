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
        target_kind: str = "AIPerfJob",
        poll_error: RuntimeError | None = None,
    ) -> None:
        self.events = events
        self.target_kind = target_kind
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
        self.sweep_runs_completed = None
        self.sweep_runs_failed = None
        self.sweep_runs_cancelled = None
        self.sweep_runs_total = None
        self.child_job_ids: list[str] = []
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

    def k8s_client(
        *, kubeconfig: str | None = None, context: str | None = None
    ) -> _FakeK8sContext:
        events.append(f"k8s:client:{kubeconfig or 'default'}:{context or 'default'}")
        return _FakeK8sContext(events)

    class _ResolvedTarget:
        def __init__(self, name: str, namespace: str) -> None:
            self.name = name
            self.namespace = namespace
            if cr_poller.target_kind == "AIPerfSweep":
                self.sweep_info = object()
            else:
                self.job_info = object()

        async def aclose(self) -> None:
            events.append("resolver:close")

    async def resolve_target(
        name: str | None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        *,
        quiet: bool = False,
    ) -> _ResolvedTarget:
        target_name = name or "aiperf-bench-7f2a"
        target_namespace = namespace or "aiperf-bench"
        events.append(f"resolver:target:{target_name}:{target_namespace}")
        return _ResolvedTarget(target_name, target_namespace)

    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", k8s_client)
    monkeypatch.setattr(cli_helpers, "resolve_target", resolve_target)
    monkeypatch.setattr(watch_pollers, "CRPoller", lambda api, job_id, ns: cr_poller)
    monkeypatch.setattr(
        watch_pollers, "SweepCRPoller", lambda api, job_id, ns: cr_poller
    )
    monkeypatch.setattr(
        watch_pollers,
        "PodPoller",
        lambda api, job_id, ns, label_selector=None, job_ids_provider=None: pod,
    )
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
        calls: list[tuple[object, str, str | None, str | None, bool]] = []
        renderer = _RecordingRenderer([])

        async def resolve_target(
            name: str | None,
            namespace: str | None = None,
            kubeconfig: str | None = None,
            kube_context: str | None = None,
            *,
            quiet: bool = False,
        ) -> None:
            calls.append((name, namespace or "", kubeconfig, kube_context, quiet))
            return None

        def forbidden_k8s_client(
            *, kubeconfig: str | None = None, context: str | None = None
        ) -> _FakeK8sContext:
            raise AssertionError("resolver miss must not open Kubernetes client")

        monkeypatch.setattr(cli_helpers, "resolve_target", resolve_target)
        monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", forbidden_k8s_client)

        orchestrator = WatchOrchestrator(renderer=renderer, quiet=quiet)

        await orchestrator.run()

        assert calls == [(None, "aiperf-benchmarks", None, None, quiet)]
        assert renderer.events == []


# ============================================================================
# Sweep target edges
# ============================================================================


class _RecordingCustomObjectsApi:
    """CustomObjectsApi double that records the plural used by a CR poller."""

    calls: list[dict[str, str]] = []

    def __init__(self, api: object) -> None:
        self.api = api

    async def get_namespaced_custom_object(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "group": group,
                "version": version,
                "plural": plural,
                "namespace": namespace,
                "name": name,
            }
        )
        return {
            "spec": {},
            "status": {
                "phase": "Succeeded",
                "runStates": {"completed": 3, "failed": 1, "cancelled": 1},
                "totalRuns": 5,
                "runs": [
                    {"childName": "latency-sweep-v00"},
                    {"childName": "latency-sweep-v01"},
                    {"childName": "latency-sweep-v00"},
                    {"childName": ""},
                    {"variation": 2},
                ],
            },
        }


@pytest.mark.asyncio
async def test_sweep_cr_poller_reads_aiperfsweeps_plural_and_run_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweep CR polling must not read the AIPerfJob plural or emit 0/0 runs."""
    _RecordingCustomObjectsApi.calls = []
    monkeypatch.setattr(
        watch_pollers.client, "CustomObjectsApi", _RecordingCustomObjectsApi
    )

    poller = watch_pollers.SweepCRPoller(object(), "latency-sweep", "aiperf-bench")

    await poller.poll()

    assert _RecordingCustomObjectsApi.calls == [
        {
            "group": "aiperf.nvidia.com",
            "version": "v1alpha1",
            "plural": "aiperfsweeps",
            "namespace": "aiperf-bench",
            "name": "latency-sweep",
        }
    ]
    assert poller.target_kind == "AIPerfSweep"
    assert poller.phase == "Succeeded"
    assert poller.sweep_runs_completed == 3
    assert poller.sweep_runs_failed == 1
    assert poller.sweep_runs_cancelled == 1
    assert poller.sweep_runs_total == 5
    assert poller.child_job_ids == ["latency-sweep-v00", "latency-sweep-v01"]


@pytest.mark.asyncio
async def test_run_sweep_terminal_phase_uses_sweep_poller_and_stops_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIPerfSweep terminal phases differ from AIPerfJob phases."""
    events: list[str] = []
    renderer = _RecordingRenderer(events)
    sweep_poller = _FakeCRPoller(
        events,
        phase="Succeeded",
        target_kind="AIPerfSweep",
    )
    sweep_poller.current_phase = None
    sweep_poller.sweep_runs_completed = 3
    sweep_poller.sweep_runs_failed = 1
    sweep_poller.sweep_runs_cancelled = 1
    sweep_poller.sweep_runs_total = 5
    sweep_poller.child_job_ids = ["latency-sweep-v00", "latency-sweep-v01"]
    job_poller = _FakeCRPoller(
        events,
        phase=Phase.RUNNING,
        target_kind="AIPerfJob",
    )
    pod_poller = _FakePodPoller(events)
    event_poller = _FakeEventPoller(events)

    def k8s_client(
        *, kubeconfig: str | None = None, context: str | None = None
    ) -> _FakeK8sContext:
        events.append(f"k8s:client:{kubeconfig or 'default'}:{context or 'default'}")
        return _FakeK8sContext(events)

    class _ResolvedSweepTarget:
        name = "latency-sweep"
        namespace = "aiperf-bench"
        sweep_info = object()

        async def aclose(self) -> None:
            events.append("resolver:close")

    async def resolve_target(
        name: str | None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        *,
        quiet: bool = False,
    ) -> _ResolvedSweepTarget:
        events.append(f"resolver:target:{name}:{namespace}")
        return _ResolvedSweepTarget()

    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", k8s_client)
    monkeypatch.setattr(cli_helpers, "resolve_target", resolve_target)
    monkeypatch.setattr(
        watch_pollers,
        "CRPoller",
        lambda api, job_id, ns: events.append("construct:job-cr") or job_poller,
    )
    monkeypatch.setattr(
        watch_pollers,
        "SweepCRPoller",
        lambda api, job_id, ns: events.append("construct:sweep-cr") or sweep_poller,
    )
    monkeypatch.setattr(
        watch_pollers,
        "PodPoller",
        lambda api,
        job_id,
        ns,
        label_selector=None,
        job_ids_provider=None: events.append(
            f"construct:pod:{label_selector}:{','.join(job_ids_provider() or []) if job_ids_provider else 'no-provider'}"
        )
        or pod_poller,
    )
    monkeypatch.setattr(
        watch_pollers, "EventPoller", lambda api, job_id, ns: event_poller
    )
    monkeypatch.setattr(
        WatchOrchestrator,
        "_install_signal_handlers",
        lambda self: events.append("signals:install"),
    )

    async def forbidden_sleep(delay: float) -> None:
        raise AssertionError(f"terminal sweep watch must not sleep for {delay}s")

    monkeypatch.setattr(watch_orchestrator.asyncio, "sleep", forbidden_sleep)
    orchestrator = WatchOrchestrator(
        job_id="latency-sweep",
        namespace="aiperf-bench",
        renderer=renderer,
        interval=90.0,
    )

    await orchestrator.run()

    assert "construct:sweep-cr" in events
    assert "construct:job-cr" not in events
    assert len(renderer.snapshots) == 1
    snapshot = renderer.snapshots[0]
    assert snapshot.target_kind == "AIPerfSweep"
    assert snapshot.phase == "Succeeded"
    assert snapshot.sweep_runs_completed == 3
    assert snapshot.sweep_runs_failed == 1
    assert snapshot.sweep_runs_cancelled == 1
    assert snapshot.sweep_runs_total == 5
    assert events == [
        "resolver:target:latency-sweep:aiperf-bench",
        "resolver:close",
        "k8s:client:default:default",
        "k8s:enter",
        "construct:sweep-cr",
        "construct:pod:None:latency-sweep-v00,latency-sweep-v01",
        "signals:install",
        "renderer:start",
        "cr:poll",
        "pod:poll",
        "event:poll",
        "renderer:render:Succeeded",
        "renderer:stop",
        "k8s:exit",
    ]


@pytest.mark.asyncio
async def test_run_no_target_stored_sweep_uses_sweep_poller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Last-benchmark resolution must preserve sweep target kind."""
    events: list[str] = []
    renderer = _RecordingRenderer(events)
    sweep_poller = _FakeCRPoller(
        events,
        phase="Succeeded",
        target_kind="AIPerfSweep",
    )
    sweep_poller.child_job_ids = ["latency-sweep-v00", "latency-sweep-v01"]
    job_poller = _FakeCRPoller(events, phase=Phase.RUNNING, target_kind="AIPerfJob")
    pod_poller = _FakePodPoller(events)
    event_poller = _FakeEventPoller(events)

    def k8s_client(
        *, kubeconfig: str | None = None, context: str | None = None
    ) -> _FakeK8sContext:
        events.append(f"k8s:client:{kubeconfig or 'default'}:{context or 'default'}")
        return _FakeK8sContext(events)

    def forbidden_job_only_resolver(
        job_id: object, namespace: str | None, *, quiet: bool = False
    ) -> tuple[str, str] | None:
        raise AssertionError("watch no-target path must use resolve_target")

    class _ResolvedSweepTarget:
        name = "latency-sweep"
        namespace = "aiperf-bench"
        sweep_info = object()

        async def aclose(self) -> None:
            events.append("resolver:close")

    async def resolve_target(
        name: str | None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        *,
        quiet: bool = False,
    ) -> _ResolvedSweepTarget:
        events.append(f"resolver:target:{name}:{namespace}")
        return _ResolvedSweepTarget()

    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", k8s_client)
    monkeypatch.setattr(
        cli_helpers, "resolve_job_id_and_namespace", forbidden_job_only_resolver
    )
    monkeypatch.setattr(cli_helpers, "resolve_target", resolve_target)
    monkeypatch.setattr(
        watch_pollers,
        "CRPoller",
        lambda api, job_id, ns: events.append("construct:job-cr") or job_poller,
    )
    monkeypatch.setattr(
        watch_pollers,
        "SweepCRPoller",
        lambda api, job_id, ns: events.append("construct:sweep-cr") or sweep_poller,
    )
    monkeypatch.setattr(
        watch_pollers,
        "PodPoller",
        lambda api,
        job_id,
        ns,
        label_selector=None,
        job_ids_provider=None: events.append(
            f"construct:pod:{label_selector}:{','.join(job_ids_provider() or []) if job_ids_provider else 'no-provider'}"
        )
        or pod_poller,
    )
    monkeypatch.setattr(
        watch_pollers, "EventPoller", lambda api, job_id, ns: event_poller
    )
    monkeypatch.setattr(
        WatchOrchestrator,
        "_install_signal_handlers",
        lambda self: events.append("signals:install"),
    )

    async def forbidden_sleep(delay: float) -> None:
        raise AssertionError(f"terminal sweep watch must not sleep for {delay}s")

    monkeypatch.setattr(watch_orchestrator.asyncio, "sleep", forbidden_sleep)
    orchestrator = WatchOrchestrator(renderer=renderer, interval=90.0)

    await orchestrator.run()

    assert "construct:sweep-cr" in events
    assert "construct:job-cr" not in events
    assert renderer.snapshots[0].target_kind == "AIPerfSweep"
    assert events == [
        "resolver:target:None:aiperf-benchmarks",
        "resolver:close",
        "k8s:client:default:default",
        "k8s:enter",
        "construct:sweep-cr",
        "construct:pod:None:latency-sweep-v00,latency-sweep-v01",
        "signals:install",
        "renderer:start",
        "cr:poll",
        "pod:poll",
        "event:poll",
        "renderer:render:Succeeded",
        "renderer:stop",
        "k8s:exit",
    ]


class _RecordingCoreV1Api:
    """CoreV1Api double that records pod label selectors."""

    selectors: list[str | None] = []

    def __init__(self, api: object) -> None:
        self.api = api

    async def list_namespaced_pod(
        self, namespace: str, *, label_selector: str | None = None
    ) -> object:
        self.selectors.append(label_selector)
        return type("PodList", (), {"items": []})()


class _CurrentChildSweepCustomObjectsApi:
    """CustomObjectsApi double returning a live sweep child without terminal runs."""

    def __init__(self, api: object) -> None:
        self.api = api

    async def get_namespaced_custom_object(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, object]:
        return {
            "spec": {},
            "status": {
                "phase": "Running",
                "currentChildRef": {"name": "sweep-v00"},
            },
        }


@pytest.mark.asyncio
async def test_sweep_pod_selector_uses_current_child_ref_before_runs_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active sweeps must poll pods for the live child before runs[] is written."""
    _RecordingCoreV1Api.selectors = []
    monkeypatch.setattr(
        watch_pollers.client, "CustomObjectsApi", _CurrentChildSweepCustomObjectsApi
    )
    monkeypatch.setattr(watch_pollers.client, "CoreV1Api", _RecordingCoreV1Api)
    sweep_poller = watch_pollers.SweepCRPoller(object(), "sweep", "aiperf-bench")

    await sweep_poller.poll()
    pod_poller = watch_pollers.PodPoller(
        object(),
        "sweep",
        "aiperf-bench",
        job_ids_provider=lambda: sweep_poller.child_job_ids,
    )
    await pod_poller.poll()

    assert sweep_poller.child_job_ids == ["sweep-v00"]
    assert _RecordingCoreV1Api.selectors == [
        "app=aiperf,aiperf.nvidia.com/job-id=sweep-v00"
    ]
    selector = _RecordingCoreV1Api.selectors[0] or ""
    assert "__pending_sweep_children__" not in selector


@pytest.mark.asyncio
async def test_pod_poller_uses_child_job_id_set_selector_for_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweep pod polling must target generated child JobSet pod job-id labels."""
    _RecordingCoreV1Api.selectors = []
    monkeypatch.setattr(watch_pollers.client, "CoreV1Api", _RecordingCoreV1Api)
    poller = watch_pollers.PodPoller(
        object(),
        "latency-sweep",
        "aiperf-bench",
        job_ids_provider=lambda: ["latency-sweep-v00", "latency-sweep-v01"],
    )

    await poller.poll()

    assert _RecordingCoreV1Api.selectors == [
        "app=aiperf,aiperf.nvidia.com/job-id in (latency-sweep-v00,latency-sweep-v01)"
    ]
    selector = _RecordingCoreV1Api.selectors[0] or ""
    assert "aiperf.nvidia.com/sweep" not in selector
    assert "aiperf.nvidia.com/job-id=latency-sweep" not in selector


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

        with pytest.raises(
            RuntimeError, match=r"renderer render failed.*aiperf-bench-7f2a"
        ):
            await orchestrator.run()

        assert events == [
            "resolver:target:aiperf-bench-7f2a:aiperf-bench",
            "resolver:close",
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

        with pytest.raises(
            RuntimeError, match=r"renderer start failed.*aiperf-bench-7f2a"
        ):
            await orchestrator.run()

        assert events == [
            "resolver:target:aiperf-bench-7f2a:aiperf-bench",
            "resolver:close",
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
            "resolver:target:aiperf-bench-7f2a:aiperf-bench",
            "resolver:close",
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
            "resolver:target:aiperf-bench-7f2a:aiperf-bench",
            "resolver:close",
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

        def add_signal_handler(
            sig: signal.Signals, callback: Callable[[], None]
        ) -> None:
            registrations.append((sig, callback))

        # Pin the POSIX branch so this test passes on windows-latest CI too.
        monkeypatch.setattr(watch_orchestrator, "IS_WINDOWS", False)
        monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
        orchestrator = WatchOrchestrator(job_id="aiperf-bench-7f2a")

        orchestrator._install_signal_handlers()

        assert [sig for sig, _ in registrations] == [signal.SIGINT, signal.SIGTERM]
        for _, callback in registrations:
            callback()
        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_install_signal_handlers_on_windows_falls_back_to_signal_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_sigbreak = 21  # Windows SIGBREAK value; absent on POSIX.
        signal_registrations: list[tuple[int, Callable[[int, object], None]]] = []
        loop_registrations: list[tuple[object, ...]] = []
        loop = watch_orchestrator.asyncio.get_running_loop()

        def fake_signal(sig: int, handler: Callable[[int, object], None]) -> None:
            signal_registrations.append((sig, handler))

        monkeypatch.setattr(watch_orchestrator, "IS_WINDOWS", True)
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda *args: loop_registrations.append(args)
        )
        monkeypatch.setattr(watch_orchestrator.signal, "signal", fake_signal)
        monkeypatch.setattr(
            watch_orchestrator.signal, "SIGBREAK", fake_sigbreak, raising=False
        )
        orchestrator = WatchOrchestrator(job_id="aiperf-bench-7f2a")

        orchestrator._install_signal_handlers()

        assert loop_registrations == []
        assert [sig for sig, _ in signal_registrations] == [
            signal.SIGINT,
            fake_sigbreak,
        ]
        for sig, handler in signal_registrations:
            handler(sig, None)
        assert orchestrator._running is False
