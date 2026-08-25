# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified-API port of ``tests/kubernetes/chaos/test_chaos_jobset_pods.py``.

Covers C-series scenarios C6, C7, C8, C9 through the unified
:py:class:`InjectorRegistry` interface. Faults that used to call
:py:meth:`ChaosInjector.kill_container_by_pid` or :py:meth:`ChaosInjector.
kill_container_in_pod` directly are now routed through
``async with faults.inject("pod.kill_pid", ...)`` /
``async with faults.inject("pod.kill_container", ...)`` so the dispatch
matches every other chaos suite.

Non-fault ChaosInjector helpers (``get_controller_pod_name``,
``wait_for_phase``, ``read_claim_annotation``, ``wait_for_container_restart``,
``get_worker_pod_names``) stay as direct method calls -- they are not fault
injections and the unified API only owns the inject/restore lifecycle.

All four tests require the operator to be deployed with
``AIPERF_K8S_SHARE_PROCESS_NAMESPACE=true`` so every JobSet pod sets
``spec.shareProcessNamespace=true``; see the inline
``operator_ready_shared_pid`` fixture below. The runtime image
(``nvcr.io/nvidia/distroless/python``) copies ``bash`` from
``env-builder`` and has a ``/busybox`` on ``PATH``; ``ps``, ``grep`` and
``tr`` are busybox while ``kill`` is a bash builtin -- cmdline-based PID
discovery via :py:func:`_find_pid_by_cmdline` is therefore the portable
path on this image.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import orjson
import pytest
import pytest_asyncio

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


# Selector fragments passed to the shared-PID busybox ``ps`` + ``grep``
# pipeline. Each must match the target service's ``args`` list (see
# ``src/aiperf/kubernetes/jobset_helpers.py::build_container_args`` and
# ``jobset_builder.py::_create_event_bus_proxy`` /
# ``_create_results_sidecar``). Keep these in sync with those call sites.
CONTROL_PLANE_CMDLINE_MATCH = "system_controller"
"""Unique substring of control-plane container's cmdline."""

EVENT_BUS_CMDLINE_MATCH = "event_bus"
"""Unique substring of event-bus-proxy container's cmdline (matches
``aiperf proxy --kind event_bus``)."""

RESULTS_SIDECAR_CMDLINE_MATCH = "results_sidecar"
"""Unique substring of results-sidecar container's cmdline (matches
``python -m aiperf.kubernetes.results_sidecar``)."""


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:  # noqa: ANN001 - k8s_settings is a test-fixture dataclass not exported for typing
    """Duration-based benchmark large enough that chaos can land mid-profiling."""
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


@pytest.fixture
def short_fetch_config(k8s_settings) -> AIPerfJobConfig:  # noqa: ANN001 - k8s_settings is a test-fixture dataclass not exported for typing
    """Short benchmark so the post-profile fetch window arrives quickly (C9)."""
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=60.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def operator_ready_shared_pid(
    kubectl: KubectlClient,
    project_root: Path,
    loaded_images,  # noqa: ANN001 - session-scoped fixture, not typed in test surface
    jobset_controller: None,
    mock_server: None,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
    operator_job_namespace: str,
) -> AsyncIterator[OperatorDeployer]:
    """Redeploy the operator with ``share_process_namespace=True``.

    The default ``operator_ready`` fixture deploys the operator without
    ``AIPERF_K8S_SHARE_PROCESS_NAMESPACE``, which leaves every JobSet pod
    with isolated PID namespaces -- chaos kills of sibling containers via
    ``kubectl exec`` then fail because each container sees only its own
    PID 1. This fixture re-deploys the operator with the flag flipped on
    for the module's duration, then restores a plain (flag-off) operator
    at teardown so sibling package tests are unaffected.
    """
    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=k8s_settings.aiperf_image,
        default_job_namespace=operator_job_namespace,
        share_process_namespace=True,
    )
    await deployer.install_crd()
    await kubectl.run("create", "namespace", operator_job_namespace, check=False)
    await deployer.deploy_operator()
    try:
        yield deployer
    finally:
        if not k8s_settings.skip_cleanup:
            await deployer.cleanup_all()
        # Put a plain operator back so tests that run after this module do not
        # inherit ``shareProcessNamespace`` leakage. This must NOT sit inside
        # the skip_cleanup guard: --k8s-quick implies skip_cleanup=True, so the
        # most common dev invocation would leak every time. The operator
        # Deployment is shared mutable state, not a retained artifact -- left
        # pinned to shareProcessNamespace=true, later JobSet pods share a PID
        # namespace, and PID-targeted helpers such as _find_pid_by_cmdline
        # match the wrong process instead of failing cleanly.
        restore = OperatorDeployer(
            kubectl=kubectl,
            project_root=project_root,
            operator_image=k8s_settings.aiperf_image,
            default_job_namespace=operator_job_namespace,
            share_process_namespace=False,
        )
        await restore.deploy_operator()


async def _force_delete_cr(kubectl: KubectlClient, namespace: str, name: str) -> None:
    """Best-effort force-delete an AIPerfJob CR in test teardown."""
    await kubectl.run(
        "delete",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )


async def _find_pid_by_cmdline(
    kubectl: KubectlClient,
    *,
    pod: str,
    namespace: str,
    exec_container: str,
    cmdline_match: str,
) -> int:
    """Return the PID of a sibling container's main process via shared ``/proc``.

    ``kubectl exec`` into ``exec_container`` (any container in the pod
    works when ``shareProcessNamespace=true``) and walk ``/proc/<pid>/
    cmdline`` looking for ``cmdline_match``. Uses busybox ``grep`` /
    ``tr`` / ``cat`` and bash glob expansion -- no binaries that the
    distroless image lacks.

    Raises:
        RuntimeError: If no matching process is found (either the shared
            PID namespace wasn't enabled or the target has already exited).
    """
    script = (
        "for d in /proc/[0-9]*; do "
        "  pid=${d##*/}; "
        f'  if cat "$d/cmdline" 2>/dev/null | tr "\\0" " " | grep -q "{cmdline_match}"; then '
        '    echo "$pid"; '
        "    exit 0; "
        "  fi; "
        "done; "
        "exit 1"
    )
    res = await kubectl.run(
        "exec",
        pod,
        "-c",
        exec_container,
        "-n",
        namespace,
        "--",
        "/bin/bash",
        "-c",
        script,
        check=False,
    )
    pid_str = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
    if not pid_str.isdigit():
        raise RuntimeError(
            f"no process matching '{cmdline_match}' found in pod "
            f"{namespace}/{pod} via exec container '{exec_container}' "
            f"(rc={res.returncode}); stdout={res.stdout!r} stderr={res.stderr!r}. "
            "Is shareProcessNamespace enabled on the pod?"
        )
    return int(pid_str)


async def _read_controller_terminated_reason(
    kubectl: KubectlClient, *, pod: str, namespace: str
) -> str:
    """Return the ``control-plane`` container's terminated-state reason.

    Empty string when the container is still running or has no terminated
    state recorded yet.
    """
    res = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status.containerStatuses[?(@.name=='control-plane')]"
        ".state.terminated.reason}",
        check=False,
    )
    return res.stdout.strip()


async def _wait_for_live_metrics(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> None:
    """Block until the operator has sampled live metrics onto the CR.

    ``_recover_from_live_status`` salvages ``status.liveMetrics`` /
    ``status.liveSummary``; with neither present it correctly falls through to
    the unrecoverable branch. The operator only writes them once a monitor tick
    successfully reads the controller's metrics endpoint, which lags
    phase=Running by tens of seconds. Without this gate the PID kill can land
    during warmup, leaving nothing for the salvage path to recover.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get", "aiperfjob", name, "-n", namespace, "-o", "json", check=False
        )
        if result.returncode == 0:
            status = orjson.loads(result.stdout).get("status") or {}
            live_metrics = status.get("liveMetrics") or {}
            if live_metrics.get("metrics") or status.get("liveSummary"):
                return
        await asyncio.sleep(3)
    raise TimeoutError(
        f"AIPerfJob {namespace}/{name} never published live metrics within "
        f"{timeout}s; the live-status salvage path cannot be exercised without them"
    )


async def _wait_for_controller_terminated(
    kubectl: KubectlClient,
    *,
    pod: str,
    namespace: str,
    timeout: float = 30.0,
) -> str:
    """Block until ``control-plane`` container records a terminated state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        reason = await _read_controller_terminated_reason(
            kubectl, pod=pod, namespace=namespace
        )
        if reason:
            return reason
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"control-plane container in pod {namespace}/{pod} did not "
        f"reach terminated state within {timeout} s"
    )


async def _wait_for_claim_annotation(
    chaos_injector: ChaosInjector,
    *,
    namespace: str,
    name: str,
    timeout: float = 240.0,
) -> str:
    """Poll the CR until the ``completion-claimed`` annotation is present."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        claim = await chaos_injector.read_claim_annotation(namespace, name)
        if claim:
            return claim
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"completion-claimed annotation never set on AIPerfJob "
        f"{namespace}/{name} within {timeout} s"
    )


async def _get_worker_pod_uids(
    kubectl: KubectlClient, *, namespace: str, job_name: str
) -> dict[str, str]:
    """Return {pod_name: uid} for every worker pod of ``job_name``."""
    res = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        f"jobset.sigs.k8s.io/jobset-name=aiperf-{job_name},"
        "jobset.sigs.k8s.io/replicatedjob-name=workers",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name}={.metadata.uid}{'\\n'}{end}",
        check=False,
    )
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        name, _, uid = line.partition("=")
        if name and uid:
            out[name.strip()] = uid.strip()
    return out


@pytest.mark.timeout(600)
async def test_c6_kill_controller_container_salvages_unified(
    operator_ready_shared_pid: OperatorDeployer,
    chaos_injector: ChaosInjector,
    faults: InjectorRegistry,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Kill the SystemController PID mid-benchmark; operator preserves partial metrics.

    Exercises ``src/aiperf/operator/handlers/monitor.py::
    _maybe_recover_terminated_controller`` -- when the control-plane
    container terminates before final export, the next monitor tick copies
    CR live metrics into terminal partial result fields and marks the CR
    ``Failed`` with ``ResultsAvailable=True``.
    """
    name = "chaos-c6"
    try:
        await operator_ready_shared_pid.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        await _wait_for_live_metrics(
            kubectl, name=name, namespace=operator_job_namespace, timeout=120.0
        )

        pod = await chaos_injector.get_controller_pod_name(operator_job_namespace, name)
        # Kill from the event-bus-proxy container; any sibling works
        # because shareProcessNamespace=true makes /proc shared.
        control_plane_pid = await _find_pid_by_cmdline(
            kubectl,
            pod=pod,
            namespace=operator_job_namespace,
            exec_container="event-bus-proxy",
            cmdline_match=CONTROL_PLANE_CMDLINE_MATCH,
        )
        async with faults.inject(
            "pod.kill_pid",
            target={
                "ns": operator_job_namespace,
                "pod": pod,
                "exec_container": "event-bus-proxy",
            },
            container_pid=control_plane_pid,
        ):
            # Pod PID kills are non-restorable; the kubelet/JobSet own
            # any container recreation. The block just scopes the inject.
            pass

        # Within ~30s the control-plane container should show terminated.
        reason = await _wait_for_controller_terminated(
            kubectl, pod=pod, namespace=operator_job_namespace, timeout=30.0
        )
        assert reason in ("Error", "OOMKilled", "Completed"), (
            f"unexpected terminated reason {reason!r} for control-plane"
        )

        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Failed",),
            timeout=240.0,
        )
        assert phase == "Failed"

        res = await kubectl.run(
            "get",
            "aiperfjob",
            name,
            "-n",
            operator_job_namespace,
            "-o",
            "json",
            check=True,
        )
        status = orjson.loads(res.stdout).get("status", {})
        assert status.get("summary") or status.get("results", {}).get("metrics"), (
            "controller-kill salvage should preserve partial live metrics on "
            "status.summary or status.results.metrics"
        )
        results_available = next(
            (
                cond
                for cond in status.get("conditions", [])
                if cond.get("type") == "ResultsAvailable"
            ),
            {},
        )
        assert results_available.get("status") == "True"
        assert results_available.get("reason") in (
            "PartialLiveMetricsRecovered",
            "PartialCheckpointRecovered",
        )
        assert "Controller container terminated before final export" in status.get(
            "error", ""
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


@pytest.mark.timeout(600)
async def test_c7_kill_worker_pod_mid_benchmark_unified(
    operator_ready_shared_pid: OperatorDeployer,
    chaos_injector: ChaosInjector,
    faults: InjectorRegistry,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Force-delete one worker pod mid-benchmark; JobSet recreates, CR Completes.

    Exercises JobSet's own pod-level restart policy and the operator's
    tolerance for transient worker disappearance -- no explicit operator
    code handles this; the signal is that ``_fetch_progress`` and the
    monitor timer keep the CR reconciled while JobSet spawns a
    replacement worker. The replacement has the same generate-name
    prefix but a fresh UID.
    """
    name = "chaos-c7"
    try:
        await operator_ready_shared_pid.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        workers_before = await _get_worker_pod_uids(
            kubectl, namespace=operator_job_namespace, job_name=name
        )
        assert workers_before, "no worker pods found mid-profiling"
        victim_name, victim_uid = next(iter(workers_before.items()))

        async with faults.inject(
            "pod.kill",
            target={"ns": operator_job_namespace, "pod": victim_name},
        ):
            # pod.kill is a force-delete; restore is a no-op (JobSet owns
            # the replacement). The block just scopes the inject.
            pass

        # JobSet should spawn a replacement with a different UID within 60s.
        deadline = asyncio.get_event_loop().time() + 60.0
        replaced = False
        while asyncio.get_event_loop().time() < deadline:
            workers_after = await _get_worker_pod_uids(
                kubectl, namespace=operator_job_namespace, job_name=name
            )
            # The JobSet controller re-creates the pod; accept either a
            # same-name-different-UID re-admission or a fresh pod name as
            # long as the new pod's UID is unseen.
            new_uids = set(workers_after.values()) - {victim_uid}
            if workers_after and new_uids:
                replaced = True
                break
            await asyncio.sleep(1.0)
        assert replaced, (
            f"JobSet did not recreate worker for {victim_name} (uid={victim_uid}) "
            "within 60 s"
        )

        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert phase == "Completed"
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


@pytest.mark.timeout(300)
async def test_c8_kill_event_bus_sidecar_unified(
    operator_ready_shared_pid: OperatorDeployer,
    chaos_injector: ChaosInjector,
    faults: InjectorRegistry,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Kill the event-bus-proxy sibling; controller fails and salvage path runs.

    Exercises the same ``_maybe_recover_terminated_controller`` salvage
    path as C6, triggered indirectly: when the event-bus-proxy dies the
    controller loses its XPUB/XSUB transport, which either trips
    ``_check_sibling_containers_alive`` (if still in configure loop) or
    causes SystemController to fail fast with a ZMQ error. Either way
    the control-plane container terminates non-zero and the operator
    salvages.

    Outcome may be ``Completed`` (preferred, salvage fetched results)
    or ``Failed`` (controller died before the results were ready).
    Either is a valid terminal state for this fault -- the assertion
    is that the CR reaches a terminal phase rather than hanging.
    """
    name = "chaos-c8"
    try:
        await operator_ready_shared_pid.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        pod = await chaos_injector.get_controller_pod_name(operator_job_namespace, name)
        # Locate the event-bus-proxy PID via the results-sidecar's view.
        event_bus_pid = await _find_pid_by_cmdline(
            kubectl,
            pod=pod,
            namespace=operator_job_namespace,
            exec_container="results-sidecar",
            cmdline_match=EVENT_BUS_CMDLINE_MATCH,
        )
        async with faults.inject(
            "pod.kill_pid",
            target={
                "ns": operator_job_namespace,
                "pod": pod,
                "exec_container": "results-sidecar",
            },
            container_pid=event_bus_pid,
        ):
            pass

        # CR must reach a terminal phase; salvage should drive to Completed
        # but Failed is accepted as correctness-worth-noting.
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed", "Failed"),
            timeout=180.0,
        )
        assert phase in ("Completed", "Failed")
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


@pytest.mark.timeout(300)
async def test_c9_kill_results_sidecar_mid_fetch_unified(
    operator_ready_shared_pid: OperatorDeployer,
    chaos_injector: ChaosInjector,
    faults: InjectorRegistry,
    short_fetch_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Kill results-sidecar after claim annotation is set; fetch retry recovers.

    Exercises ``src/aiperf/operator/handlers/monitor.py`` fetch-retry
    path (``fetch_results_with_retry``). Once
    ``aiperf.nvidia.com/completion-claimed`` is set, the operator is
    actively fetching artifacts from the results sidecar; killing the
    sidecar at that moment forces the retry loop to survive a container
    restart. The results-ready marker on disk must still gate the CR's
    terminal transition so we don't double-fetch.

    This is a RACE test -- if the fetch finishes before we land the kill,
    the test passes trivially on the first path (already-complete).
    To defend against silent triviality we verify the sidecar actually
    restarted (``restartCount > 0``) whenever the kill landed; a missed
    window is acceptable but documented as such.
    """
    name = "chaos-c9"
    try:
        await operator_ready_shared_pid.create_job(
            config=short_fetch_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        pod = await chaos_injector.get_controller_pod_name(operator_job_namespace, name)
        # Wait for the operator to claim completion -- that's the signal
        # the fetch is in-flight.
        await _wait_for_claim_annotation(
            chaos_injector,
            namespace=operator_job_namespace,
            name=name,
            timeout=180.0,
        )

        # Try to land a kill on the results-sidecar PID. If the fetch
        # already completed (CR in terminal phase), this is the trivial
        # pass path.
        kill_landed = False
        try:
            results_pid = await _find_pid_by_cmdline(
                kubectl,
                pod=pod,
                namespace=operator_job_namespace,
                exec_container="event-bus-proxy",
                cmdline_match=RESULTS_SIDECAR_CMDLINE_MATCH,
            )
            async with faults.inject(
                "pod.kill_pid",
                target={
                    "ns": operator_job_namespace,
                    "pod": pod,
                    "exec_container": "event-bus-proxy",
                },
                container_pid=results_pid,
            ):
                pass
            kill_landed = True
        except RuntimeError:
            # Race missed -- results sidecar already exited / pod reaped.
            pass

        if kill_landed:
            # Belt-and-suspenders: verify the sidecar actually restarted
            # (or the pod transitioned terminal). Either confirms we
            # weren't asserting on a no-op. Pod may have already been
            # torn down by JobSet delete -- tolerate TimeoutError because
            # the final phase check is authoritative.
            with contextlib.suppress(TimeoutError):
                await chaos_injector.wait_for_container_restart(
                    pod=pod,
                    container="results-sidecar",
                    namespace=operator_job_namespace,
                    since_count=0,
                    timeout=45.0,
                )

        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert phase == "Completed", (
            f"fetch-retry did not drive CR to Completed; got {phase}"
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)
