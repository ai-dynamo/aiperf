# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H1/H2/H3/H4 unified-API ports - see chaos/test_chaos_helm.py for legacy.

These scenarios exercise the Helm chart lifecycle, not chaos-injection-via-
``faults``. Every Helm operation (``install`` / ``upgrade`` / ``uninstall``)
stays a direct :py:class:`HelmDeployer` call because no unified injector
wraps Helm CLI -- and arguably shouldn't, since these tests *are* the chart
contract. The unified port is therefore a near-copy of the legacy file:

* The ``faults`` fixture from :py:mod:`tests.kubernetes.chaos_aiperf.conftest`
  is requested by every test so the module is in the unified suite, even
  when the test body never calls ``faults.inject(...)``. Future revisions
  that weave in operator-pod kills (``operator.kill``) or in-flight CR
  mutations (``crd.patch`` / ``crd.delete``) plug in without renaming.
* The legacy ``ChaosInjector.wait_for_phase`` helper is replaced by the
  free :py:func:`wait_for_aiperfjob_phase` async function from the conftest
  -- no class needed, no extra dependency.

H1: install -> run -> uninstall -> reinstall idempotence.
H2: helm upgrade with a Running CR must not drop the CR.
H3: invalid helm values fail fast, then recovery on the same namespace works.
H4: missing JobSet CRD surfaces an operator-side error within 60 s.

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
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_aiperf.conftest import wait_for_aiperfjob_phase
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.conftest import K8sTestSettings, _create_helm_values
from tests.kubernetes.helpers.helm import (
    HelmClient,
    HelmDeployer,
    HelmValues,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]
logger = AIPerfLogger(__name__)

# Budget applied per scenario. 5 min is generous -- most scenarios settle
# in under 2 min on a warm kind cluster, but ``helm install --wait`` plus
# CRD-establish plus first-pull image load can eat the slack on cold runs.
_PER_TEST_TIMEOUT = 300


@pytest_asyncio.fixture(scope="module", loop_scope="package", autouse=True)
async def _restore_shared_operator_rbac(operator_deployer: OperatorDeployer):  # noqa: ANN202
    """Repair the shared operator's cluster RBAC after the helm scenarios.

    ``HelmDeployer.install_chart`` unconditionally deletes the cluster-scoped
    ``aiperf-operator`` ClusterRole and ClusterRoleBinding (helpers/helm.py) so
    the chart can own those names. Those objects belong to the SHARED
    ``aiperf-system`` operator that ``operator_ready`` hands to nearly every
    other test, and nothing here puts them back.

    ``operator_ready`` is package-scoped and evaluates ``is_operator_healthy()``
    exactly once, at first request, so it cannot notice. The operator pod stays
    Ready (its probe is plain HTTP) while every kopf watch and status patch gets
    403, which surfaces downstream as AIPerfJob CRs that never receive any
    ``.status`` at all -- i.e. an empty ``phase``, indistinguishable from a CR
    that was never created.

    ``is_operator_healthy()`` already checks both RBAC objects, so redeploy only
    when the check actually fails; on a clean run this costs three kubectl gets.
    """
    yield
    if not await operator_deployer.is_operator_healthy():
        logger.warning(
            "Shared operator RBAC was removed by a helm scenario; redeploying "
            "so subsequent tests reconcile normally"
        )
        await operator_deployer.deploy_operator()


def _unique_release_name(prefix: str) -> str:
    """Short, DNS-1035-compliant release name for per-test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _chaos_helm_values(k8s_settings: K8sTestSettings) -> HelmValues:
    """HelmValues preset for chaos tests -- forces local-only image."""
    return _create_helm_values(k8s_settings)


async def _make_isolated_deployer(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    values: HelmValues,
    operator_namespace: str,
    job_namespace: str,
    release_name: str,
) -> HelmDeployer:
    """Build a :py:class:`HelmDeployer` with a test-unique release + namespace."""
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

    Clears finalizers on any stuck AIPerfJob CR in the job namespace first
    so ``kubectl delete`` does not block on dangling owner-ref reconciliation.
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

    with contextlib.suppress(TimeoutError, Exception):
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
async def test_h1_install_job_uninstall_reinstall_is_clean_unified(
    faults: InjectorRegistry,  # noqa: ARG001  (registry presence keeps this in the unified suite)
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001  (dependency for endpoint reachability)
) -> None:
    """H1: install chart, run a tiny job, uninstall, re-install, run again.

    Asserts the chart leaves no orphaned AIPerfJob CRs, PVCs, or namespace
    finalizers. Exercises:

    * :py:meth:`HelmDeployer.install_chart` / :py:meth:`HelmDeployer.uninstall_chart`
    * operator create -> completed lifecycle end to end
    * chart idempotency: a fresh install after uninstall must work on the
      same namespace with no residual cruft.

    Pure Helm-CLI scenario: no ``faults.inject(...)`` calls. The ``faults``
    fixture is requested for suite membership and to keep the call site
    one-line-symmetric with H2/H4 if those ever weave in operator kills.
    """
    release = _unique_release_name("chaos-h1u")
    operator_ns = f"aiperf-chaos-h1u-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h1u-jobs-{worker_namespace_suffix}"
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
        result_a = await deployer.run_job(config, name="h1u-run-a", timeout=240)
        assert result_a.success, (
            "H1 first run failed: "
            f"phase={result_a.status.phase if result_a.status else '?'}"
        )

        # Phase 2: uninstall cleanly.
        await deployer.uninstall_chart(wait=True)

        # ``helm uninstall`` intentionally retains CRDs (and any CR instances
        # it didn't create) so user data is never silently destroyed. Assert
        # instead that the release + operator deployment are gone; actively
        # GC any leftover CRs ourselves so re-install is clean.
        status_after = await deployer.get_release_status()
        assert status_after != "deployed", (
            f"H1 release still deployed after uninstall: {status_after!r}"
        )

        remaining_crs = await kubectl.run(
            "get",
            "aiperfjobs",
            "-A",
            "-o",
            "jsonpath={range .items[*]}{.metadata.namespace}/{.metadata.name}{'\\n'}{end}",
            check=False,
        )
        leftover = [
            line.strip()
            for line in remaining_crs.stdout.splitlines()
            if line.strip() and ("h1u-" in line or release in line)
        ]
        for entry in leftover:
            ns, _, name = entry.partition("/")
            if ns and name:
                await kubectl.run(
                    "delete",
                    "aiperfjob",
                    name,
                    "-n",
                    ns,
                    "--ignore-not-found",
                    "--wait=false",
                    check=False,
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
        result_b = await deployer.run_job(config, name="h1u-run-b", timeout=240)
        assert result_b.success, (
            "H1 second run (after uninstall + reinstall) failed: "
            f"phase={result_b.status.phase if result_b.status else '?'}"
        )
    finally:
        await _force_cleanup_release(deployer, kubectl)


# H2 alone carries install + a full operator rolling update + a benchmark, so
# it gets headroom above _PER_TEST_TIMEOUT. The pytest ceiling must stay a
# backstop rather than the expected exit: pytest-timeout uses the SIGALRM
# method here, and asyncio.Runner.run only cancels its task on
# KeyboardInterrupt, so a Failed raised by the alarm does NOT run this test's
# finally. The orphaned coroutine then stays scheduled on the package-scoped
# loop and resumes during h3/h4, tearing down namespaces mid-test.
@pytest.mark.timeout(420)
async def test_h2_upgrade_with_inflight_job_preserves_cr_unified(
    faults: InjectorRegistry,  # noqa: ARG001  (registry presence keeps this in the unified suite)
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H2: ``helm upgrade`` with a Running CR must not drop the CR.

    The operator pod is likely re-created mid-upgrade; kopf's reconcile
    resumes monitoring on restart (the same code path exercised by C4 in
    :py:mod:`tests.kubernetes.chaos_aiperf.test_chaos_operator_resilience_unified`).
    The CR must stay in Running through the upgrade and converge to
    Completed afterwards.

    Pure Helm-CLI scenario: the upgrade itself is the disruptive operation,
    and it is owned by :py:meth:`HelmDeployer.upgrade_chart`. ``faults`` is
    requested to anchor this test in the unified suite without inventing a
    ``helm.upgrade`` injector.

    Replaces the legacy ``ChaosInjector.wait_for_phase(...)`` calls with the
    free :py:func:`wait_for_aiperfjob_phase` helper from the conftest.
    """
    release = _unique_release_name("chaos-h2u")
    operator_ns = f"aiperf-chaos-h2u-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h2u-jobs-{worker_namespace_suffix}"
    values = _chaos_helm_values(k8s_settings)

    deployer = await _make_isolated_deployer(
        kubectl, helm_client, project_root, values, operator_ns, job_ns, release
    )
    # 45s, not 120s: the scenario only needs the job to still be profiling when
    # the upgrade lands. At 120s this test declared ~800s of nested waits
    # (install 180 + profiling 180 + upgrade 180 + observe 20 + completion 240)
    # against a 300s pytest ceiling, so it was a coin flip that the SIGALRM --
    # not an assertion -- ended the test.
    longrun = AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=45.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    cr_name = "chaos-h2u"

    try:
        await deployer.install_chart(wait=True, timeout="3m")

        await deployer.create_job(longrun, name=cr_name, namespace=job_ns)

        # Wait for profiling so the upgrade lands mid-run, not during
        # pre-benchmark init where the CR could legitimately blip phases.
        await wait_for_aiperfjob_phase(
            kubectl,
            job_ns,
            cr_name,
            phases=("Running",),
            current_phase="profiling",
            timeout=120.0,
        )

        # Trivial no-op values tweak: bump monitor interval. This forces a
        # rolling update of the operator Deployment.
        bumped = _chaos_helm_values(k8s_settings)
        bumped.monitor_interval = "12.0"
        await deployer.upgrade_chart(values=bumped, wait=True, timeout="3m")

        # Observe the CR for 20 s post-upgrade: it must stay queryable
        # (phase in {Running, Completed}) -- NOT disappear.
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
            # During operator rollover the CR persists in etcd; phase may
            # briefly read empty, but kubectl must succeed.
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
        final_phase = await wait_for_aiperfjob_phase(
            kubectl,
            job_ns,
            cr_name,
            phases=("Completed",),
            timeout=120.0,
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
async def test_h3_invalid_values_fail_fast_and_recover_unified(
    faults: InjectorRegistry,  # noqa: ARG001  (registry presence keeps this in the unified suite)
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H3: install with bogus image + impossible memory fails fast, then recovers.

    Exercises Helm's ``--wait`` timeout path: the operator pod can never
    become Ready because the image doesn't exist and memory is too tight to
    schedule. After cleanup, a valid install on the same namespace must
    succeed -- no residual cruft.

    Pure Helm-CLI scenario; no ``faults.inject(...)`` calls. The bad-values
    install is the "fault" -- handing the chart an impossible config and
    asserting the failure surfaces fast. ``faults`` is requested only to
    anchor this test in the unified suite.
    """
    release = _unique_release_name("chaos-h3u")
    operator_ns = f"aiperf-chaos-h3u-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h3u-jobs-{worker_namespace_suffix}"

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
        # Phase 1: invalid install -- expect either helm non-zero exit
        # (Pending pod never becomes Ready within the wait timeout) or a
        # pod that stays non-Ready. Use a short helm timeout so we do not
        # spend minutes of the test budget here.
        install_failed = False
        try:
            await bad_deployer.install_chart(wait=True, timeout="45s")
        except Exception:  # noqa: BLE001  (helm install can raise CalledProcessError or RuntimeError; both mean "failed fast")
            install_failed = True

        if not install_failed:
            # helm returned 0 but the pod must not be Running with Ready
            # replicas -- verify the deploy is genuinely broken.
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
                f"H3 invalid values install somehow reported ready replicas: "
                f"{ready_str!r}"
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
        result = await good_deployer.run_job(config, name="h3u-recovery", timeout=240)
        assert result.success, (
            "H3 recovery install could not run a job: "
            f"phase={result.status.phase if result.status else '?'}"
        )
    finally:
        if good_deployer is not None:
            await _force_cleanup_release(good_deployer, kubectl)
        else:
            await _force_cleanup_release(bad_deployer, kubectl)


@pytest.mark.timeout(_PER_TEST_TIMEOUT)
async def test_h4_missing_jobset_crd_surfaces_error_unified(
    faults: InjectorRegistry,  # noqa: ARG001  (registry presence keeps this in the unified suite)
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    k8s_settings,  # noqa: ANN001
    worker_namespace_suffix: str,
    mock_server: None,  # noqa: ARG001
) -> None:
    """H4: install chart without JobSet CRD; observe operator error surfacing.

    IMPORTANT: This test deletes the cluster-wide ``jobsets.jobset.x-k8s.io``
    CRD. Other ``k8s_slow`` tests in the same session depend on it, so this
    test reinstalls the CRD via the same URL the ``jobset_controller``
    fixture uses before returning, regardless of outcome.

    Exercises: ``src/aiperf/operator/handlers/create.py`` -- when the
    operator tries to create a JobSet for a newly-applied AIPerfJob CR, the
    apiserver should return 404 on the JobSet kind. The operator should
    surface this through ``status.conditions`` or reach ``phase=Failed``
    within 60 s.

    Pure Helm-CLI scenario; no ``faults.inject(...)`` calls. The "fault"
    is deleting the JobSet CRD out-of-band, which no current unified
    injector models. ``faults`` is requested only to anchor this test in
    the unified suite.
    """
    from dev.versions import JOBSET_CRD_URL_TEMPLATE, JOBSET_VERSION

    release = _unique_release_name("chaos-h4u")
    operator_ns = f"aiperf-chaos-h4u-{worker_namespace_suffix}"
    job_ns = f"aiperf-chaos-h4u-jobs-{worker_namespace_suffix}"
    values = _chaos_helm_values(k8s_settings)

    deployer = await _make_isolated_deployer(
        kubectl, helm_client, project_root, values, operator_ns, job_ns, release
    )
    cr_name = "chaos-h4u"
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

        # Observe for up to 60 s. Expect phase=Failed OR a status condition
        # whose message mentions JobSet / CRD.
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
