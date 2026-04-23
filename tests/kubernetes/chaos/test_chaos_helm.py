# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: Helm chart lifecycle resilience.

Covers scenarios H1, H2, H3, H4 from the chaos-expansion design doc
(``docs/superpowers/specs/2026-04-23-chaos-expansion-design.md``).

Every test in this module owns its Helm release: it performs its own
install and an unconditional ``finally``-clause uninstall plus namespace
cleanup. These tests deliberately do NOT use the module-scoped
``helm_deployed`` fixture because they exercise install / upgrade /
uninstall edge cases that must run against a freshly-deployed chart.

Exercises these operator/Helm code paths:

* ``deploy/helm/aiperf-operator`` chart templates and values resolution.
* ``src/aiperf/operator/handlers/create.py`` preflight validation when
  the JobSet CRD is missing (H4).
* ``tests/kubernetes/helpers/helm.py::HelmDeployer.install_chart`` /
  ``upgrade_chart`` / ``uninstall_chart``.
* kopf reconcile continuity across an operator-pod recreation (H2).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.helpers.helm import (
    HelmClient,
    HelmDeployer,
    HelmValues,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]

# Budget applied per scenario. 5 min is generous — most scenarios settle
# in under 2 min on a warm kind cluster, but helm install --wait plus
# CRD-establish plus first-pull image load can eat the slack on cold
# runs.
_PER_TEST_TIMEOUT = 300


def _unique_release_name(prefix: str) -> str:
    """Short, DNS-1035-compliant release name for per-test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _chaos_helm_values(k8s_settings) -> HelmValues:  # noqa: ANN001  (pytest fixture type)
    """HelmValues preset for chaos tests — forces local-only image."""
    return HelmValues(
        image_repository="aiperf",
        image_tag="local",
        image_pull_policy="Never",
        default_image=k8s_settings.aiperf_image,
        default_image_pull_policy="Never",
        storage_enabled=False,
    )


async def _make_isolated_deployer(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    values: HelmValues,
    operator_namespace: str,
    job_namespace: str,
    release_name: str,
) -> HelmDeployer:
    """Build a HelmDeployer with a test-unique release name + namespace."""
    deployer = HelmDeployer(
        kubectl=kubectl,
        helm=helm_client,
        project_root=project_root,
        values=values,
        operator_namespace=operator_namespace,
        default_job_namespace=job_namespace,
    )
    deployer.RELEASE_NAME = release_name
    await kubectl.run("create", "namespace", job_namespace, check=False)
    return deployer


async def _force_cleanup_release(
    deployer: HelmDeployer, kubectl: KubectlClient
) -> None:
    """Best-effort uninstall + namespace purge; never raises.

    Clears finalizers on any stuck AIPerfJob CR in the job namespace
    first so kubectl delete does not block on dangling owner-ref
    reconciliation.
    """
    try:
        aiperfs = await kubectl.run(
            "get",
            "aiperfjobs",
            "-n",
            deployer.default_job_namespace,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        if aiperfs.returncode == 0 and aiperfs.stdout.strip():
            for name in aiperfs.stdout.strip().split():
                await kubectl.run(
                    "patch",
                    "aiperfjob",
                    name,
                    "-n",
                    deployer.default_job_namespace,
                    "--type=json",
                    '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                    check=False,
                )
                await kubectl.run(
                    "delete",
                    "aiperfjob",
                    name,
                    "-n",
                    deployer.default_job_namespace,
                    "--ignore-not-found",
                    "--wait=false",
                    check=False,
                )
    except Exception:  # noqa: BLE001,S110  (cleanup must never raise)
        pass

    with contextlib.suppress(asyncio.TimeoutError, Exception):
        await asyncio.wait_for(deployer.uninstall_chart(wait=False), timeout=60)

    # Drop the operator namespace entirely so re-install is clean.
    await kubectl.run(
        "delete",
        "namespace",
        deployer.OPERATOR_NAMESPACE,
        "--ignore-not-found",
        "--wait=false",
        check=False,
    )


@pytest.mark.timeout(_PER_TEST_TIMEOUT)
async def test_h1_install_job_uninstall_reinstall_is_clean(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001  (dependency for endpoint reachability)
) -> None:
    """H1: install chart, run a tiny job, uninstall, re-install, run again.

    Asserts the chart leaves no orphaned AIPerfJob CRs, PVCs, or
    namespace finalizers. Exercises:

    * ``HelmDeployer.install_chart`` / ``uninstall_chart``
    * operator create -> completed lifecycle end to end
    * chart idempotency: a fresh install after uninstall must work on
      the same namespace with no residual cruft.
    """
    release = _unique_release_name("chaos-h1")
    operator_ns = f"aiperf-chaos-h1-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h1-jobs-{worker_namespace_suffix}"
    values = _chaos_helm_values(k8s_settings)

    deployer = await _make_isolated_deployer(
        kubectl, helm_client, project_root, values, operator_ns, job_ns, release
    )
    config = AIPerfJobConfig(
        concurrency=2,
        request_count=30,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )

    try:
        # Phase 1: install + run job to Completed.
        await deployer.install_chart(wait=True, timeout="3m")
        result_a = await deployer.run_job(config, name="h1-run-a", timeout=240)
        assert result_a.success, (
            f"H1 first run failed: phase={result_a.status.phase if result_a.status else '?'}"
        )

        # Phase 2: uninstall cleanly.
        await deployer.uninstall_chart(wait=True)

        # Assert no AIPerfJob CRs left anywhere.
        remaining_crs = await kubectl.run(
            "get",
            "aiperfjobs",
            "-A",
            "-o",
            "jsonpath={.items[*].metadata.namespace}/{.items[*].metadata.name}",
            check=False,
        )
        own_remaining = [
            line
            for line in remaining_crs.stdout.splitlines()
            if release in line or "h1-" in line
        ]
        assert not own_remaining, (
            f"H1 left AIPerfJob CRs after uninstall: {own_remaining!r}"
        )

        # Assert no PVCs left in operator namespace (storage was disabled,
        # but we still check defensively).
        pvcs = await kubectl.run(
            "get",
            "pvc",
            "-n",
            operator_ns,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        assert not pvcs.stdout.strip(), (
            f"H1 left PVCs in {operator_ns}: {pvcs.stdout.strip()!r}"
        )

        # Phase 3: re-install on the same namespace.
        await deployer.install_chart(wait=True, timeout="3m")
        result_b = await deployer.run_job(config, name="h1-run-b", timeout=240)
        assert result_b.success, (
            "H1 second run (after uninstall + reinstall) failed: "
            f"phase={result_b.status.phase if result_b.status else '?'}"
        )
    finally:
        await _force_cleanup_release(deployer, kubectl)


@pytest.mark.timeout(_PER_TEST_TIMEOUT)
async def test_h2_upgrade_with_inflight_job_preserves_cr(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H2: helm upgrade with a Running CR must not drop the CR.

    The operator pod is likely re-created mid-upgrade; kopf's reconcile
    resumes monitoring on restart (the same code path exercised by C4
    in test_chaos_operator_resilience.py). The CR must stay in Running
    through the upgrade and converge to Completed afterwards.
    """
    release = _unique_release_name("chaos-h2")
    operator_ns = f"aiperf-chaos-h2-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h2-jobs-{worker_namespace_suffix}"
    values = _chaos_helm_values(k8s_settings)

    deployer = await _make_isolated_deployer(
        kubectl, helm_client, project_root, values, operator_ns, job_ns, release
    )
    injector = ChaosInjector(kubectl=kubectl)
    longrun = AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    cr_name = "chaos-h2"

    try:
        await deployer.install_chart(wait=True, timeout="3m")

        await deployer.create_job(longrun, name=cr_name, namespace=job_ns)

        # Wait for profiling so the upgrade lands mid-run, not during
        # pre-benchmark init where the CR could legitimately blip phases.
        await injector.wait_for_phase(
            job_ns,
            cr_name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        # Trivial no-op values tweak: bump monitor interval. This forces
        # a rolling update of the operator Deployment.
        bumped = _chaos_helm_values(k8s_settings)
        bumped.monitor_interval = "12.0"
        await deployer.upgrade_chart(values=bumped, wait=True, timeout="3m")

        # Observe the CR for 20 s post-upgrade: it must stay queryable
        # (phase in {Running, Completed}) — NOT disappear.
        observe_deadline = time.monotonic() + 20.0
        while time.monotonic() < observe_deadline:
            check = await kubectl.run(
                "get",
                "aiperfjob",
                cr_name,
                "-n",
                job_ns,
                "-o",
                "jsonpath={.status.phase}",
                check=False,
            )
            # During operator rollover the CR persists in etcd; phase
            # may briefly read empty, but kubectl must succeed.
            assert check.returncode == 0, (
                f"H2 CR lookup failed mid-upgrade: rc={check.returncode} "
                f"stderr={check.stderr!r}"
            )
            phase = check.stdout.strip()
            assert phase in ("", "Running", "Completed"), (
                f"H2 CR regressed mid-upgrade: phase={phase!r}"
            )
            await asyncio.sleep(2.0)

        # Eventual convergence: CR reaches Completed within benchmark
        # duration + generous margin.
        final_phase = await injector.wait_for_phase(
            job_ns,
            cr_name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert final_phase == "Completed"
    finally:
        await kubectl.run(
            "delete",
            "aiperfjob",
            cr_name,
            "-n",
            job_ns,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
        await _force_cleanup_release(deployer, kubectl)


@pytest.mark.timeout(_PER_TEST_TIMEOUT)
async def test_h3_invalid_values_fail_fast_and_recover(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H3: install with bogus image + impossible memory fails fast, then recovers.

    Exercises Helm's ``--wait`` timeout path: the operator pod can never
    become Ready because the image doesn't exist and memory is too tight
    to schedule. After cleanup, a valid install on the same namespace
    must succeed — no residual cruft.
    """
    release = _unique_release_name("chaos-h3")
    operator_ns = f"aiperf-chaos-h3-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h3-jobs-{worker_namespace_suffix}"

    bad_values = HelmValues(
        image_repository="does-not-exist.example.com/nothing",
        image_tag="nope",
        image_pull_policy="Always",
        default_image=k8s_settings.aiperf_image,
        default_image_pull_policy="Never",
        storage_enabled=False,
        resources_limits_memory="1Mi",
    )

    bad_deployer = await _make_isolated_deployer(
        kubectl,
        helm_client,
        project_root,
        bad_values,
        operator_ns,
        job_ns,
        release,
    )
    good_deployer: HelmDeployer | None = None

    try:
        # Phase 1: invalid install — expect either helm non-zero exit
        # (Pending pod never becomes Ready within the wait timeout) or a
        # pod that stays non-Ready. Use a short helm timeout so we do
        # not spend minutes of the test budget here.
        install_failed = False
        try:
            await bad_deployer.install_chart(wait=True, timeout="45s")
        except Exception:  # noqa: BLE001  (helm install can raise CalledProcessError or RuntimeError; both mean "failed fast")
            install_failed = True

        if not install_failed:
            # helm returned 0 but the pod must not be Running with Ready
            # replicas — verify the deploy is genuinely broken.
            ready = await kubectl.run(
                "get",
                "deployment",
                release,
                "-n",
                operator_ns,
                "-o",
                "jsonpath={.status.readyReplicas}",
                check=False,
            )
            ready_str = ready.stdout.strip()
            assert ready_str in ("", "0"), (
                f"H3 invalid values install somehow reported ready replicas: {ready_str!r}"
            )

        # Phase 2: purge and reinstall with valid values on the same
        # namespace. This proves there is no residual cruft blocking
        # recovery.
        await _force_cleanup_release(bad_deployer, kubectl)

        good_values = _chaos_helm_values(k8s_settings)
        good_deployer = await _make_isolated_deployer(
            kubectl,
            helm_client,
            project_root,
            good_values,
            operator_ns,
            job_ns,
            release,
        )
        await good_deployer.install_chart(wait=True, timeout="3m")

        config = AIPerfJobConfig(
            concurrency=2,
            request_count=30,
            warmup_request_count=5,
            image=k8s_settings.aiperf_image,
        )
        result = await good_deployer.run_job(config, name="h3-recovery", timeout=240)
        assert result.success, (
            "H3 recovery install could not run a job: "
            f"phase={result.status.phase if result.status else '?'}"
        )
    finally:
        if good_deployer is not None:
            await _force_cleanup_release(good_deployer, kubectl)
        else:
            await _force_cleanup_release(bad_deployer, kubectl)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "H4 is exploratory: the chart does not pre-validate the JobSet CRD, "
        "so operator behavior when the CRD is absent is observed rather than "
        "contract-asserted. Mark xfail strict=False so either a clean "
        "surfaced error (test passes) or an operator hang (test xfails) is "
        "acceptable — both are documented in findings-2026-04-23-v2.md."
    ),
)
@pytest.mark.timeout(_PER_TEST_TIMEOUT)
async def test_h4_missing_jobset_crd_surfaces_error(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H4: install chart without JobSet CRD, observe operator error surfacing.

    IMPORTANT: This test deletes the cluster-wide ``jobsets.jobset.x-k8s.io``
    CRD. Other k8s_slow tests in the same session depend on it, so this
    test reinstalls the CRD via the same URL the ``jobset_controller``
    fixture uses before returning, regardless of outcome.

    Exercises: ``src/aiperf/operator/handlers/create.py`` — when the
    operator tries to create a JobSet for a newly-applied AIPerfJob CR,
    apiserver should return 404 on the JobSet kind. The operator should
    surface this through ``status.conditions`` or reach phase=Failed
    within 60 s. If it hangs silently, that is itself a bug worth
    capturing (xfail strict=False).
    """
    from dev.versions import JOBSET_CRD_URL_TEMPLATE, JOBSET_VERSION

    release = _unique_release_name("chaos-h4")
    operator_ns = f"aiperf-chaos-h4-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h4-jobs-{worker_namespace_suffix}"
    values = _chaos_helm_values(k8s_settings)

    deployer = await _make_isolated_deployer(
        kubectl, helm_client, project_root, values, operator_ns, job_ns, release
    )
    cr_name = "chaos-h4"
    crd_url = JOBSET_CRD_URL_TEMPLATE.format(version=JOBSET_VERSION)

    try:
        # Ensure JobSet CRD is absent. --ignore-not-found so we are idempotent.
        await kubectl.run(
            "delete",
            "crd",
            "jobsets.jobset.x-k8s.io",
            "--ignore-not-found",
            "--wait=true",
            check=False,
        )

        await deployer.install_chart(wait=True, timeout="3m")

        config = AIPerfJobConfig(
            concurrency=2,
            request_count=30,
            warmup_request_count=5,
            image=k8s_settings.aiperf_image,
        )
        await deployer.create_job(config, name=cr_name, namespace=job_ns)

        # Observe for up to 60 s. Expect phase=Failed OR a status
        # condition whose message mentions JobSet / CRD.
        deadline = time.monotonic() + 60.0
        surfaced = False
        observed_phase = ""
        while time.monotonic() < deadline:
            status = await kubectl.run(
                "get",
                "aiperfjob",
                cr_name,
                "-n",
                job_ns,
                "-o",
                "jsonpath={.status.phase}",
                check=False,
            )
            observed_phase = status.stdout.strip()
            if observed_phase == "Failed":
                surfaced = True
                break
            cond = await kubectl.run(
                "get",
                "aiperfjob",
                cr_name,
                "-n",
                job_ns,
                "-o",
                "jsonpath={.status.conditions[*].message}",
                check=False,
            )
            lowered = cond.stdout.lower()
            if "jobset" in lowered and (
                "not found" in lowered
                or "missing" in lowered
                or "no matches" in lowered
            ):
                surfaced = True
                break
            await asyncio.sleep(2.0)

        assert surfaced, (
            "H4: operator did not surface missing-JobSet-CRD error within 60 s "
            f"(final phase={observed_phase!r}). This is a bug worth fixing: "
            "the create handler should fail fast, not silently retry."
        )
    finally:
        # Restore JobSet CRD so downstream tests are not wrecked.
        with contextlib.suppress(Exception):
            await kubectl.apply_server_side(crd_url)
        await kubectl.run(
            "delete",
            "aiperfjob",
            cr_name,
            "-n",
            job_ns,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
        await _force_cleanup_release(deployer, kubectl)
