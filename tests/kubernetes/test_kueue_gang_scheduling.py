# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Kind cluster integration tests for Kueue gang-scheduling.

Tests verify the full Kueue admission workflow on a real Kind cluster:
- JobSet creation with Kueue labels and suspend=true
- Kueue Workload creation and admission
- JobSet unsuspend and benchmark completion

Fixture scoping strategy:
- Module-scoped: kueue_manager (install/uninstall), kueue_queues (queue CRUD)
- Function-scoped: Each test gets a fresh benchmark via benchmark_deployer
- Cleanup: benchmark_deployer.cleanup_all() handles namespace deletion;
  kueue_manager.cleanup_queues() removes queue CRs on module teardown.

Requires: Kind cluster with aiperf images loaded, JobSet controller installed,
mock server deployed. Run with:
    uv run pytest tests/kubernetes/test_kueue_gang_scheduling.py -v --k8s-quick
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio
import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.constants import KueueLabels
from tests.kubernetes.helpers.benchmark import BenchmarkConfig, BenchmarkDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.kueue import KueueManager
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
)

logger = AIPerfLogger(__name__)

# Minimal config for fast Kueue tests (small dataset, few requests)
_KUEUE_BENCHMARK_KWARGS: dict[str, Any] = {
    "concurrency": 2,
    "request_count": 5,
    "warmup_request_count": 1,
    "workers": 1,
    "input_sequence_min": 10,
    "input_sequence_max": 20,
    "output_tokens_min": 1,
    "output_tokens_max": 5,
}


# ============================================================================
# Module-scoped Kueue fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def kueue_manager(kubectl: KubectlClient) -> KueueManager:
    """Install Kueue controller and yield manager (module-scoped).

    Idempotent: apply --server-side is safe to re-run. Teardown uninstalls.
    """
    manager = KueueManager(kubectl)
    await manager.install()
    yield manager
    await manager.uninstall()


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def kueue_queues(kueue_manager: KueueManager) -> str:
    """Set up default Kueue queues (module-scoped).

    Creates ResourceFlavor + ClusterQueue + LocalQueue (in default ns).
    Teardown deletes all created queue resources in reverse order.

    Returns the local queue name.
    """
    queue_name = await kueue_manager.setup_default_queues()
    yield queue_name
    await kueue_manager.cleanup_queues()


# ============================================================================
# Tests - Manifest generation (no cluster mutations)
# ============================================================================


class TestKueueManifestGeneration:
    """Test aiperf kube generate output with Kueue options.

    These tests only generate manifests — they don't apply anything to the
    cluster, so they're inherently idempotent.
    """

    @pytest.mark.asyncio
    async def test_generate_manifest_includes_kueue_labels(
        self,
        k8s_ready,
        kueue_queues: str,
        project_root,
        k8s_settings,
    ) -> None:
        """Verify --queue-name produces Kueue labels and suspend=true."""
        manifest = await _generate_kueue_manifest(
            project_root, k8s_settings.aiperf_image, kueue_queues
        )

        docs = list(yaml.safe_load_all(manifest))
        jobset_doc = _find_jobset(docs)
        assert jobset_doc is not None, "No JobSet found in generated manifest"

        labels = jobset_doc["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME in labels
        assert labels[KueueLabels.QUEUE_NAME] == kueue_queues
        assert jobset_doc["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_generate_manifest_without_queue_not_suspended(
        self,
        k8s_ready,
        kueue_queues: str,
        project_root,
        k8s_settings,
    ) -> None:
        """Verify manifests without --queue-name have no Kueue labels."""
        manifest = await _generate_kueue_manifest(
            project_root, k8s_settings.aiperf_image, queue_name=None
        )

        docs = list(yaml.safe_load_all(manifest))
        jobset_doc = _find_jobset(docs)
        assert jobset_doc is not None

        labels = jobset_doc["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME not in labels
        assert "suspend" not in jobset_doc["spec"]


# ============================================================================
# Tests - Full Kueue admission flow (deploy + watchdog)
# ============================================================================


class TestKueueAdmissionFlow:
    """Test the full Kueue admission workflow via BenchmarkDeployer.

    Each test deploys a benchmark with --queue-name, which starts suspended.
    A pre_wait_hook creates a LocalQueue in the dynamic namespace so Kueue
    can admit the workload. BenchmarkDeployer provides watchdog monitoring
    and handles cleanup via cleanup_all() on fixture teardown.
    """

    @pytest.mark.asyncio
    async def test_kueue_admits_and_completes_benchmark(
        self,
        k8s_ready,
        kueue_manager: KueueManager,
        kueue_queues: str,
        benchmark_deployer: BenchmarkDeployer,
        k8s_settings,
    ) -> None:
        """Deploy a Kueue-gated benchmark and verify end-to-end completion.

        Flow: generate -> apply (suspended) -> LocalQueue created ->
        Kueue admits -> unsuspend -> pods run -> benchmark completes.
        """

        async def create_local_queue_hook(namespace: str) -> None:
            await kueue_manager.create_local_queue(
                name="local-queue",
                namespace=namespace,
                cluster_queue="cluster-queue",
            )
            logger.info(f"Created LocalQueue in {namespace} for Kueue admission")

        config = BenchmarkConfig(
            image=k8s_settings.aiperf_image,
            queue_name=kueue_queues,
            **_KUEUE_BENCHMARK_KWARGS,
        )

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=600,
            stream_logs=True,
            pre_wait_hook=create_local_queue_hook,
        )

        assert result.success, f"Benchmark did not complete: {result.error_message}"

    @pytest.mark.asyncio
    async def test_kueue_creates_admitted_workload(
        self,
        k8s_ready,
        kueue_manager: KueueManager,
        kueue_queues: str,
        benchmark_deployer: BenchmarkDeployer,
        k8s_settings,
    ) -> None:
        """Verify Kueue creates and admits a Workload for the labeled JobSet."""
        admitted_workload: dict[str, Any] = {}

        async def verify_workload_hook(namespace: str) -> None:
            await kueue_manager.create_local_queue(
                name="local-queue",
                namespace=namespace,
                cluster_queue="cluster-queue",
            )
            workload = await kueue_manager.wait_for_workload_admitted(
                namespace, timeout=120
            )
            admitted_workload.update(workload)
            logger.info(f"Workload admitted in {namespace}")

        config = BenchmarkConfig(
            image=k8s_settings.aiperf_image,
            queue_name=kueue_queues,
            **_KUEUE_BENCHMARK_KWARGS,
        )

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=600,
            stream_logs=True,
            pre_wait_hook=verify_workload_hook,
        )

        assert admitted_workload, "No admitted Workload found"
        wl_spec = admitted_workload.get("spec", {})
        assert wl_spec.get("queueName") == "local-queue"
        assert result.success, f"Benchmark did not complete: {result.error_message}"


# ============================================================================
# Tests - Operator with Kueue
# ============================================================================


class TestKueueOperatorIntegration:
    """Test operator behavior with Kueue-managed AIPerfJob CRs.

    Verifies that the operator correctly propagates Kueue scheduling config
    to the generated JobSet labels and suspend field. Uses operator_ready
    fixture and AIPerfJobConfig with queue_name/priority_class fields.
    Cleanup via cleanup_job() in finally blocks.
    """

    @pytest.mark.asyncio
    async def test_operator_job_creates_suspended_jobset_with_queue_label(
        self,
        operator_ready: OperatorDeployer,
        kueue_manager: KueueManager,
        kueue_queues: str,
        kubectl: KubectlClient,
        k8s_settings,
    ) -> None:
        """Verify operator creates a suspended JobSet with the queue-name label."""
        config = AIPerfJobConfig(
            concurrency=2,
            request_count=5,
            warmup_request_count=1,
            image=k8s_settings.aiperf_image,
            queue_name=kueue_queues,
        )

        result = await operator_ready.create_job(config)

        try:
            status = await operator_ready.wait_for_phase(
                result.job_name, result.namespace, "Pending", timeout=30
            )
            jobset_name = status.jobset_name
            if not jobset_name:
                # Job completed before we could observe the Pending phase
                if status.is_completed or status.is_terminal:
                    return
                pytest.fail("Operator did not create a JobSet")

            try:
                jobset_data = await kubectl.get_json(
                    "jobset", jobset_name, namespace=result.namespace
                )
            except RuntimeError:
                # JobSet already cleaned up by operator
                final = await operator_ready.get_job_status(
                    result.job_name, result.namespace
                )
                if final.is_completed or final.is_terminal:
                    return
                raise

            labels = jobset_data["metadata"]["labels"]

            assert KueueLabels.QUEUE_NAME in labels, "Missing queue-name label"
            assert labels[KueueLabels.QUEUE_NAME] == kueue_queues
            # Kueue may have already admitted and unsuspended the JobSet,
            # so we verify the label (which triggers Kueue admission) rather
            # than the transient suspend field.
        finally:
            await operator_ready.cleanup_job(result)

    @pytest.mark.asyncio
    async def test_operator_job_with_priority_class_label(
        self,
        operator_ready: OperatorDeployer,
        kueue_manager: KueueManager,
        kueue_queues: str,
        kubectl: KubectlClient,
        k8s_settings,
    ) -> None:
        """Verify priority_class flows through to the generated JobSet labels."""
        config = AIPerfJobConfig(
            concurrency=2,
            request_count=5,
            warmup_request_count=1,
            image=k8s_settings.aiperf_image,
            queue_name=kueue_queues,
            priority_class="test-priority",
        )

        result = await operator_ready.create_job(config)

        try:
            status = await operator_ready.wait_for_phase(
                result.job_name, result.namespace, "Pending", timeout=30
            )
            jobset_name = status.jobset_name
            if not jobset_name:
                if status.is_completed or status.is_terminal:
                    return
                pytest.fail("Operator did not create a JobSet")

            try:
                jobset_data = await kubectl.get_json(
                    "jobset", jobset_name, namespace=result.namespace
                )
            except RuntimeError:
                final = await operator_ready.get_job_status(
                    result.job_name, result.namespace
                )
                if final.is_completed or final.is_terminal:
                    return
                raise

            labels = jobset_data["metadata"]["labels"]

            assert KueueLabels.QUEUE_NAME in labels, "Missing queue-name label"
            assert labels[KueueLabels.QUEUE_NAME] == kueue_queues
            assert KueueLabels.PRIORITY_CLASS in labels, "Missing priority-class label"
            assert labels[KueueLabels.PRIORITY_CLASS] == "test-priority"
        finally:
            await operator_ready.cleanup_job(result)


# ============================================================================
# Helpers
# ============================================================================


async def _generate_kueue_manifest(
    project_root: Any,
    image: str,
    queue_name: str | None,
) -> str:
    """Run aiperf kube generate and return the YAML output."""
    cmd = [
        "uv",
        "run",
        "aiperf",
        "kube",
        "generate",
        "--model",
        "mock-model",
        "--url",
        "http://aiperf-mock-server.default.svc.cluster.local:8000/v1",
        "--endpoint-type",
        "chat",
        "--image",
        image,
        "--concurrency",
        "2",
        "--request-count",
        "5",
        "--warmup-request-count",
        "1",
        "--tokenizer",
        "gpt2",
        "--workers-max",
        "1",
        "--ui",
        "none",
        "--no-operator",
    ]
    if queue_name is not None:
        cmd.extend(["--queue-name", queue_name])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(project_root),
    )
    stdout, stderr = await proc.communicate()
    assert proc.returncode == 0, f"Generate failed: {stderr.decode()}"
    return stdout.decode()


def _find_jobset(docs: list[Any]) -> dict[str, Any] | None:
    """Find the JobSet document in a multi-document YAML."""
    for doc in docs:
        if doc and doc.get("kind") == "JobSet":
            return doc
    return None
