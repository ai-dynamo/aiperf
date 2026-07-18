# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.jobset module."""

from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import orjson
import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.cr_refs import (
    JOBSET_API_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import (
    JOBSET_FALLBACK_VERSION,
    JOBSET_GITHUB_REPO,
    AIPerfContainerSpec,
    AIPerfJobSetSpec,
    AIPerfReplicatedJobSpec,
    aggregator_children,
    aggregator_count,
    aggregator_tier_counts,
    aggregator_tier_job_name,
    get_jobset_install_hint,
    get_jobset_manifest_url,
    get_latest_jobset_version,
)
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib


class TestJobSetAPIConstants:
    """Tests for JobSet API coordinates in cr_refs."""

    def test_default_values(self) -> None:
        """JobSet cr_refs constants match expected API coordinates."""
        assert JOBSET_GROUP == "jobset.x-k8s.io"
        assert JOBSET_VERSION == "v1alpha2"
        assert JOBSET_PLURAL == "jobsets"

    def test_api_version(self) -> None:
        """JOBSET_API_VERSION composes group and version correctly."""
        assert JOBSET_API_VERSION == "jobset.x-k8s.io/v1alpha2"


class TestContainerSpec:
    """Tests for AIPerfContainerSpec model."""

    def test_minimal_container(self) -> None:
        """Test creating a minimal container spec."""
        container = AIPerfContainerSpec(name="test", image="nginx:latest")
        assert container.name == "test"
        assert container.image == "nginx:latest"
        assert container.command == []
        assert container.args == []

    def test_container_to_k8s_spec(self) -> None:
        """Test converting container spec to Kubernetes format."""
        container = AIPerfContainerSpec(
            name="worker",
            image="aiperf:latest",
            command=["aiperf"],
            args=["service", "--type", "worker"],
            env=[{"name": "FOO", "value": "bar"}],
            resources={"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}},
            ports=[{"containerPort": 8080, "name": "health"}],
        )
        spec = container.to_k8s_spec()
        assert spec["name"] == "worker"
        assert spec["image"] == "aiperf:latest"
        assert spec["command"] == ["aiperf"]
        assert spec["args"] == ["service", "--type", "worker"]
        assert spec["env"] == [{"name": "FOO", "value": "bar"}]
        assert spec["resources"]["requests"]["cpu"] == "100m"
        assert spec["ports"][0]["containerPort"] == 8080

    def test_container_to_k8s_spec_with_probes(self) -> None:
        """Test container spec with health probes."""
        container = AIPerfContainerSpec(
            name="test",
            image="nginx:latest",
            liveness_probe={"httpGet": {"path": "/healthz", "port": 8080}},
            readiness_probe={"httpGet": {"path": "/readyz", "port": 8080}},
        )
        spec = container.to_k8s_spec()
        assert "livenessProbe" in spec
        assert spec["livenessProbe"]["httpGet"]["path"] == "/healthz"
        assert "readinessProbe" in spec
        assert spec["readinessProbe"]["httpGet"]["path"] == "/readyz"

    def test_container_to_k8s_spec_excludes_empty(self) -> None:
        """Test that empty fields are excluded from Kubernetes spec."""
        container = AIPerfContainerSpec(name="test", image="nginx:latest")
        spec = container.to_k8s_spec()
        assert "command" not in spec
        assert "args" not in spec
        assert "env" not in spec
        assert "livenessProbe" not in spec


class TestReplicatedJobSpec:
    """Tests for AIPerfReplicatedJobSpec model."""

    def test_default_values(self) -> None:
        """Test AIPerfReplicatedJobSpec has expected defaults."""
        job = AIPerfReplicatedJobSpec(name="test")
        assert job.name == "test"
        assert job.replicas == 1
        assert job.restart_policy == "OnFailure"
        assert job.backoff_limit == 0

    def test_to_k8s_spec_basic(self) -> None:
        """Test converting replicated job to Kubernetes format."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(
            name="workers",
            replicas=3,
            containers=[container],
            volumes=[{"name": "data", "emptyDir": {}}],
        )
        spec = job.to_k8s_spec()
        assert spec["name"] == "workers"
        assert spec["replicas"] == 3
        # JobSet handles replication via replicas, each Job runs 1 pod
        assert spec["template"]["spec"]["parallelism"] == 1
        assert spec["template"]["spec"]["completions"] == 1
        assert (
            spec["template"]["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"
        )

    def test_to_k8s_spec_with_customization(self) -> None:
        """Test replicated job with pod customization."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            annotations={"custom/annotation": "value"},
            labels={"custom-label": "value"},
            image_pull_secrets=[{"name": "my-registry"}],
            service_account_name="my-sa",
        )
        job = AIPerfReplicatedJobSpec(
            name="workers",
            replicas=2,
            containers=[container],
            pod_template=custom,
        )
        spec = job.to_k8s_spec()
        pod_spec = spec["template"]["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"] == {"gpu": "true"}
        assert len(pod_spec["tolerations"]) == 1
        assert pod_spec["imagePullSecrets"] == [{"name": "my-registry"}]
        assert pod_spec["serviceAccountName"] == "my-sa"
        # Check metadata
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["annotations"]["custom/annotation"] == "value"
        # Custom labels are merged with base labels
        assert pod_meta["labels"]["custom-label"] == "value"
        assert pod_meta["labels"]["app"] == "aiperf"

    def test_to_k8s_spec_has_base_labels(self) -> None:
        """Test that pods always have base AIPerf labels."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(name="workers", containers=[container])
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["labels"]["app"] == "aiperf"

    def test_to_k8s_spec_with_job_id_label(self) -> None:
        """Test that job_id is added to pod labels when set."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(
            name="workers", containers=[container], job_id="my-benchmark"
        )
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["labels"]["app"] == "aiperf"
        assert pod_meta["labels"]["aiperf.nvidia.com/job-id"] == "my-benchmark"

    def test_to_k8s_spec_custom_labels_preserve_reserved_labels(self) -> None:
        """Test that custom labels cannot override reserved AIPerf labels."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(labels={"app": "custom-app", "team": "platform"})
        job = AIPerfReplicatedJobSpec(
            name="workers",
            containers=[container],
            pod_template=custom,
            job_id="test-job",
        )
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["labels"]["app"] == "aiperf"
        assert pod_meta["labels"]["team"] == "platform"
        assert pod_meta["labels"]["aiperf.nvidia.com/job-id"] == "test-job"


class TestJobSetSpec:
    """Tests for AIPerfJobSetSpec model."""

    @pytest.fixture
    def basic_jobset_spec(self) -> AIPerfJobSetSpec:
        """Create a basic AIPerfJobSetSpec for testing."""
        return AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
            worker_replicas=2,
            workers_per_pod=2,
            record_processors_per_pod=1,
        )

    def test_create_basic_jobset(self, basic_jobset_spec: AIPerfJobSetSpec) -> None:
        """Test creating a basic AIPerfJobSetSpec."""
        assert basic_jobset_spec.name == "aiperf-test"
        assert basic_jobset_spec.namespace == "default"
        assert basic_jobset_spec.job_id == "test-123"
        assert basic_jobset_spec.image == "aiperf:latest"
        assert basic_jobset_spec.worker_replicas == 2

    def test_to_k8s_manifest_structure(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test JobSet manifest has correct structure."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert manifest["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
        assert manifest["kind"] == "JobSet"
        assert manifest["metadata"]["name"] == "aiperf-test"
        assert manifest["metadata"]["namespace"] == "default"
        assert manifest["metadata"]["labels"]["app"] == "aiperf"
        assert manifest["metadata"]["labels"]["aiperf.nvidia.com/job-id"] == "test-123"

    def test_to_k8s_manifest_has_controller_and_cells(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test JobSet manifest contains the controller and cells jobs."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        jobs = manifest["spec"]["replicatedJobs"]
        assert len(jobs) == 2
        job_names = [j["name"] for j in jobs]
        assert "controller" in job_names
        assert "cells" in job_names

    def test_to_k8s_manifest_controller_replicas(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test controller has exactly 1 replica."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        assert controller_job["replicas"] == 1

    def test_to_k8s_manifest_cells_replicas(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test the cells job has the correct replica count (one per cell)."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        assert cells_job["replicas"] == 2

    def test_to_k8s_manifest_success_policy(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test JobSet has correct success policy."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert manifest["spec"]["successPolicy"]["operator"] == "All"
        assert manifest["spec"]["successPolicy"]["targetReplicatedJobs"] == [
            "controller"
        ]

    def test_to_k8s_manifest_no_failure_policy(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test JobSet has no explicit failurePolicy (default fast-fail behavior).

        The operator's _classify_jobset_failure logic relies on replicatedJobsStatus
        being populated by the default JobSet failure path. An explicit failurePolicy
        triggers via rule name rather than incrementing replicatedJobsStatus.failed,
        which breaks the non-fatal worker failure classification.
        """
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert "failurePolicy" not in manifest["spec"]

    def test_to_k8s_manifest_ttl(self, basic_jobset_spec: AIPerfJobSetSpec) -> None:
        """Test JobSet TTL is set from environment default."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert "ttlSecondsAfterFinished" in manifest["spec"]

    def test_cells_job_ttl_is_zero_for_immediate_gc(self) -> None:
        """Cell jobs must GC immediately; the outer JobSet carries the real TTL."""
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
            ttl_seconds=600,
        )
        manifest = spec.to_k8s_manifest()
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        assert cells_job["template"]["spec"]["ttlSecondsAfterFinished"] == 0

    def test_to_k8s_manifest_custom_ttl(self) -> None:
        """Test JobSet with custom TTL."""
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            ttl_seconds=600,
        )
        manifest = spec.to_k8s_manifest()
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 600

    def test_jobset_manifest_keep_failed_pods_preserves_retries_disables_ttls(
        self,
    ) -> None:
        """Debug retention mode disables TTL cleanup but keeps worker retries.

        keepFailedPods and backoffLimit are independent: workers should still
        retry on transient startup failures (e.g., ConfigMap mount races) even
        when pods are being retained for debugging.
        """
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            worker_replicas=2,
            keep_failed_pods=True,
        )

        manifest = spec.to_k8s_manifest()
        controller_job, worker_job = manifest["spec"]["replicatedJobs"]

        assert controller_job["template"]["spec"]["backoffLimit"] == 0
        assert worker_job["template"]["spec"]["backoffLimit"] == 20
        assert "ttlSecondsAfterFinished" not in worker_job["template"]["spec"]
        assert "ttlSecondsAfterFinished" not in manifest["spec"]

    def test_jobset_manifest_default_keeps_existing_retry_and_ttl_behavior(
        self,
    ) -> None:
        """Default mode preserves worker retries and puts TTL on the outer JobSet.

        Worker jobs are forced to ttl=0 for immediate GC; the benchmark-wide TTL
        lives on the JobSet itself.
        """
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            worker_replicas=2,
            ttl_seconds=600,
            keep_failed_pods=False,
        )

        manifest = spec.to_k8s_manifest()
        controller_job, worker_job = manifest["spec"]["replicatedJobs"]

        assert controller_job["template"]["spec"]["backoffLimit"] == 0
        assert worker_job["template"]["spec"]["backoffLimit"] == 20
        assert worker_job["template"]["spec"]["ttlSecondsAfterFinished"] == 0
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 600

    def test_cell_containers_have_no_probe_timeout_override(self) -> None:
        """Cell containers carry none of the retired mesh service-probe env.

        The native cell runner speaks no ZMQ service-health protocol; the
        AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT override is gone. Cells fail fast
        under the cells-job backoff limit (container restarts), not in-process retries.
        """
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
        )

        manifest = spec.to_k8s_manifest()
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        for container in cells_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]:
            env_names = {e["name"] for e in container.get("env", [])}
            assert "AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT" not in env_names, (
                f"container {container['name']} should not override the probe timeout"
            )

    def test_to_k8s_manifest_controller_containers(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test controller pod has expected containers.

        The cellular controller pod is one aiperf runner in controller mode plus a
        results-serving sidecar — no mesh control-plane/event-bus/manager/api containers.
        """
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        container_names = [c["name"] for c in containers]
        assert container_names == ["controller", "results-sidecar"]

    def test_to_k8s_manifest_cell_containers(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test the cells pod has exactly the single cell runner container.

        A cell pod is one aiperf runner slice; there is no worker-group-manager,
        per-worker, or per-record-processor container in the cellular topology.
        """
        manifest = basic_jobset_spec.to_k8s_manifest()
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        containers = cells_job["template"]["spec"]["template"]["spec"]["containers"]
        assert [c["name"] for c in containers] == ["cell"]

    def test_cell_container_runs_cell_frontend_against_config(self) -> None:
        """The cell container runs the `aiperf cell` frontend against the mounted Config v2."""
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            cells=1,
        )

        manifest = spec.to_k8s_manifest()
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        cell = cells_job["template"]["spec"]["template"]["spec"]["containers"][0]

        assert cell["name"] == "cell"
        assert cell["command"] == ["aiperf"]
        assert cell["args"] == [
            "cell",
            "--config",
            f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/config.yaml",
        ]

    def test_to_k8s_manifest_containers_have_image(
        self, basic_jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test all containers have the correct image."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert container["image"] == "aiperf:latest"

    def test_to_k8s_manifest_with_pod_customization(self) -> None:
        """Test JobSet with pod customization."""
        custom = PodTemplateConfig(
            node_selector={"accelerator": "gpu"},
            annotations={"prometheus.io/scrape": "true"},
            env=[{"name": "DEBUG", "value": "true"}],
        )
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            pod_template=custom,
        )
        manifest = spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        pod_spec = controller_job["template"]["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"] == {"accelerator": "gpu"}

    def test_to_k8s_manifest_volumes(self, basic_jobset_spec: AIPerfJobSetSpec) -> None:
        """Test JobSet pods have required volumes."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        volumes = controller_job["template"]["spec"]["template"]["spec"]["volumes"]
        volume_names = [v["name"] for v in volumes]
        # Cellular runner volumes: config (mounted Config v2), results, HF cache, scratch.
        # The retired mesh `ipc` (ZMQ socket dir) and `datasets` (mmap) volumes are gone.
        assert "config" in volume_names
        assert "results" in volume_names
        assert "tokenizer-cache" in volume_names
        assert "tmp" in volume_names
        assert "ipc" not in volume_names
        assert "datasets" not in volume_names


class TestJobSetSpecContainerDetails:
    """Tests for AIPerfJobSetSpec container configuration details."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
            resource_mode="guaranteed",
        )
        return spec.to_k8s_manifest()

    def test_only_results_sidecar_has_health_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Only the always-on results sidecar keeps liveness/readiness probes; the
        cellular runner containers (controller/cell/aggregator) run to terminal
        under a fail-fast restart policy and carry no probes.
        """
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    assert "livenessProbe" in container, (
                        f"{container['name']} missing livenessProbe"
                    )
                    assert "readinessProbe" in container, (
                        f"{container['name']} missing readinessProbe"
                    )
                else:
                    assert "livenessProbe" not in container, (
                        f"{container['name']} unexpectedly has livenessProbe"
                    )
                    assert "readinessProbe" not in container, (
                        f"{container['name']} unexpectedly has readinessProbe"
                    )

    def test_containers_have_resources(self, jobset_manifest: dict[str, Any]) -> None:
        """Test that containers have resource specifications."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "resources" in container, (
                    f"{container['name']} missing resources"
                )
                assert "requests" in container["resources"]
                assert "limits" in container["resources"]

    def test_default_single_worker_requests_fit_high_fanout_smoke_jobs(self) -> None:
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            worker_replicas=1,
            workers_per_pod=1,
            record_processors_per_pod=1,
        )
        manifest = spec.to_k8s_manifest()
        total_cpu = 0.0
        total_memory_mib = 0
        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                requests = container["resources"]["requests"]
                total_cpu += parse_cpu(requests["cpu"])
                total_memory_mib += parse_memory_mib(requests["memory"])

        assert total_cpu <= 0.61
        # Controller-pod containers (~1792 MiB) + WORKER_POD default (4 GiB).
        # Bumped from 1536 → 6144 alongside the memory-estimator recalibration
        # against the 2026-04-30 ISL/OSL memory sweep (per-process Python
        # baseline 150 MiB; controller container limits now sized to fit
        # measured RSS 1080-1161 MiB with PEAK_MARGIN headroom).
        assert total_memory_mib <= 6144

    def test_resource_mode_none_omits_resources(self) -> None:
        """Test that resourceMode=none omits the container resources block."""
        manifest = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            resource_mode="none",
        ).to_k8s_manifest()

        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "resources" not in container, (
                    f"{container['name']} unexpectedly had resources"
                )

    def test_resource_mode_burstable_has_requests_only(self) -> None:
        """Test that resourceMode=burstable emits requests without limits."""
        manifest = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            resource_mode="burstable",
        ).to_k8s_manifest()

        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                assert "resources" in container, (
                    f"{container['name']} missing resources"
                )
                assert "requests" in container["resources"]
                assert "limits" not in container["resources"]

    def test_resource_mode_default_is_burstable(self) -> None:
        """Default resourceMode is burstable so the controller can grow during
        aggregation without being OOM-killed by cgroup limits.

        Regression for: a 1-hour DGX run was OOM-killed in the controller's
        aggregation phase under the previous Guaranteed-by-default mode. The
        burstable default leaves the workload memory budget unbounded by
        Kubernetes, deferring eviction policy to node pressure.
        """
        spec = AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
        )
        assert spec.resource_mode == "burstable"

        manifest = spec.to_k8s_manifest()
        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                assert "resources" in container, (
                    f"{container['name']} missing resources"
                )
                assert "limits" not in container["resources"], (
                    f"{container['name']} unexpectedly has limits set under "
                    f"the burstable default; got {container['resources']}"
                )

    def test_containers_have_env_vars(self, jobset_manifest: dict[str, Any]) -> None:
        """Test that containers have environment variables."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "env" in container, f"{container['name']} missing env"
                if container["name"] == "results-sidecar":
                    continue
                env_names = [e["name"] for e in container["env"]]
                assert "AIPERF_JOB_ID" in env_names
                assert "AIPERF_NAMESPACE" in env_names

    def test_cell_containers_have_controller_addr(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Cell containers dial the controller via AIPERF_CELL_CONTROLLER_ADDR (tcp://…:9500),
        replacing the retired mesh AIPERF_K8S_ZMQ_CONTROLLER_HOST."""
        cells_job = next(
            j for j in jobset_manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        containers = cells_job["template"]["spec"]["template"]["spec"]["containers"]
        for container in containers:
            env_dict = {e["name"]: e.get("value") for e in container["env"]}
            addr = env_dict.get("AIPERF_CELL_CONTROLLER_ADDR")
            assert addr is not None, f"{container['name']} missing controller addr"
            assert addr.startswith("tcp://")
            assert addr.endswith(":9500")
            assert "AIPERF_K8S_ZMQ_CONTROLLER_HOST" not in env_dict

    def test_cell_containers_get_cell_id_from_job_index(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Each cell derives its partition index (CELL_ID) from its JobSet job-index
        label, and knows the total cell count — replacing per-worker service IDs."""
        cells_job = next(
            j for j in jobset_manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        cell = cells_job["template"]["spec"]["template"]["spec"]["containers"][0]
        env_by_name = {e["name"]: e for e in cell["env"]}
        cell_id = env_by_name["AIPERF_CELL_ID"]
        assert (
            cell_id["valueFrom"]["fieldRef"]["fieldPath"]
            == "metadata.labels['jobset.sigs.k8s.io/job-index']"
        )
        assert env_by_name["AIPERF_CELL_COUNT"]["value"] == "2"

    def test_controller_exposes_cell_controller_port(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """The controller container exposes its cell-transport bootstrap port (9500)."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        controller = next(
            c
            for c in controller_job["template"]["spec"]["template"]["spec"][
                "containers"
            ]
            if c["name"] == "controller"
        )
        cell_ctl = next(p for p in controller["ports"] if p["name"] == "cell-ctl")
        assert cell_ctl["containerPort"] == 9500

    def test_cell_controller_skips_all_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """The controller runner container carries no probes — it runs to terminal
        under RestartPolicy.NEVER and merges shards; probes are the sidecar's job."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        controller = next(
            c
            for c in controller_job["template"]["spec"]["template"]["spec"][
                "containers"
            ]
            if c["name"] == "controller"
        )
        assert "readinessProbe" not in controller
        assert "startupProbe" not in controller
        assert "livenessProbe" not in controller

    def test_cell_containers_skip_readiness_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Cell runner containers carry no readiness probe."""
        cells_job = next(
            j for j in jobset_manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        for container in cells_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]:
            assert "readinessProbe" not in container

    def test_cell_containers_skip_startup_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Cell runner containers carry no startup probe."""
        cells_job = next(
            j for j in jobset_manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        for container in cells_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]:
            assert "startupProbe" not in container

    def test_cell_containers_skip_liveness_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Cell runner containers carry no liveness probe."""
        cells_job = next(
            j for j in jobset_manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        for container in cells_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]:
            assert "livenessProbe" not in container

    def test_results_sidecar_exposes_results_port(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test results sidecar exposes its recovery file-serving port."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        sidecar = next(c for c in containers if c["name"] == "results-sidecar")
        assert sidecar["ports"] == [{"containerPort": 9091, "name": "results"}]

    def test_results_sidecar_mounts_results_read_only(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test results sidecar only reads the shared results volume."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        sidecar = next(c for c in containers if c["name"] == "results-sidecar")
        results_mount = next(
            mount for mount in sidecar["volumeMounts"] if mount["name"] == "results"
        )
        assert results_mount["mountPath"] == "/results"
        assert results_mount["readOnly"] is True


class TestJobSetSpecImagePullPolicy:
    """Tests for AIPerfJobSetSpec image pull policy handling."""

    @pytest.mark.parametrize(
        "policy,expected",
        [
            ("Always", "Always"),
            ("Never", "Never"),
            ("IfNotPresent", "IfNotPresent"),
        ],
    )
    def test_image_pull_policy_is_set(self, policy: str, expected: str) -> None:
        """Test image pull policy is correctly set on containers."""
        spec = AIPerfJobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            image_pull_policy=policy,
        )
        manifest = spec.to_k8s_manifest()
        for job in manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert container["imagePullPolicy"] == expected

    def test_invalid_image_pull_policy_raises(self) -> None:
        """Test invalid image pull policy raises ValueError."""
        with pytest.raises(ValueError, match="image_pull_policy"):
            AIPerfJobSetSpec(
                name="test",
                namespace="default",
                job_id="test-123",
                image="aiperf:latest",
                image_pull_policy="invalid",
            )

    def test_none_image_pull_policy_valid(self) -> None:
        """Test None image pull policy is valid (uses Kubernetes default)."""
        spec = AIPerfJobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            image_pull_policy=None,
        )
        assert spec.image_pull_policy is None


class TestContainerSpecImagePullPolicy:
    """Tests for AIPerfContainerSpec image pull policy validation."""

    @pytest.mark.parametrize("policy", ["Always", "Never", "IfNotPresent"])
    def test_valid_image_pull_policy(self, policy: str) -> None:
        """Test valid image pull policies are accepted."""
        container = AIPerfContainerSpec(
            name="test",
            image="aiperf:latest",
            image_pull_policy=policy,
        )
        assert container.image_pull_policy == policy

    def test_invalid_image_pull_policy_raises(self) -> None:
        """Test invalid image pull policy raises ValueError."""
        with pytest.raises(ValueError, match="image_pull_policy"):
            AIPerfContainerSpec(
                name="test",
                image="aiperf:latest",
                image_pull_policy="BadValue",
            )


class TestJobSetSpecDNSConfiguration:
    """Tests for AIPerfJobSetSpec DNS naming and configuration.

    Cells dial the controller pod over the JobSet headless service via
    ``AIPERF_CELL_CONTROLLER_ADDR = tcp://<controller-dns>:9500`` (replacing the
    retired mesh ``AIPERF_K8S_ZMQ_CONTROLLER_HOST``).
    """

    @staticmethod
    def _cell_controller_host(manifest: dict[str, Any]) -> str:
        """Extract the controller DNS host a cell dials from AIPERF_CELL_CONTROLLER_ADDR."""
        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        cell = cells_job["template"]["spec"]["template"]["spec"]["containers"][0]
        addr = next(
            e for e in cell["env"] if e["name"] == "AIPERF_CELL_CONTROLLER_ADDR"
        )["value"]
        # tcp://<host>:<port> -> <host>
        assert addr.startswith("tcp://")
        return addr[len("tcp://") :].rsplit(":", 1)[0]

    def test_controller_dns_format_includes_full_fqdn(self) -> None:
        """Test that controller DNS includes .svc.cluster.local suffix.

        The DNS format must be:
        {jobset-name}-controller-0-0.{jobset-name}.{namespace}.svc.cluster.local

        This ensures proper DNS resolution across the cluster.
        """
        spec = AIPerfJobSetSpec(
            name="my-jobset",
            namespace="test-ns",
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
        )
        controller_host = self._cell_controller_host(spec.to_k8s_manifest())

        assert controller_host.endswith(".svc.cluster.local"), (
            f"DNS should end with .svc.cluster.local, got: {controller_host}"
        )
        assert "my-jobset-controller-0-0" in controller_host
        assert "my-jobset.test-ns" in controller_host

    def test_controller_dns_format_correct_structure(self) -> None:
        """Test controller DNS has correct structure: pod.service.namespace.svc.cluster.local"""
        spec = AIPerfJobSetSpec(
            name="aiperf-abc123",
            namespace="my-namespace",
            job_id="abc123",
            image="aiperf:latest",
            cells=2,
        )
        controller_host = self._cell_controller_host(spec.to_k8s_manifest())

        expected = (
            "aiperf-abc123-controller-0-0.aiperf-abc123.my-namespace.svc.cluster.local"
        )
        assert controller_host == expected

    @pytest.mark.parametrize(
        "namespace",
        ["default", "aiperf", "kube-system", "my-long-namespace-name"],
    )
    def test_controller_dns_with_various_namespaces(self, namespace: str) -> None:
        """Test controller DNS is correct with various namespaces."""
        spec = AIPerfJobSetSpec(
            name="test-jobset",
            namespace=namespace,
            job_id="test-123",
            image="aiperf:latest",
            cells=2,
        )
        controller_host = self._cell_controller_host(spec.to_k8s_manifest())

        assert f".{namespace}." in controller_host
        assert controller_host.endswith(".svc.cluster.local")


class TestJobSetSpecSecurityContext:
    """Tests for AIPerfJobSetSpec security context generation."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = AIPerfJobSetSpec(
            name="security-test",
            namespace="default",
            job_id="test-security",
            image="aiperf:latest",
        )
        return spec.to_k8s_manifest()

    def test_containers_have_security_context(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test all containers have security context."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "securityContext" in container, (
                    f"{container['name']} missing securityContext"
                )

    def test_security_context_runs_as_non_root(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test security context sets runAsNonRoot."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                ctx = container["securityContext"]
                assert ctx["runAsNonRoot"] is True
                assert ctx["runAsUser"] == 1000
                assert ctx["runAsGroup"] == 1000

    def test_security_context_disallows_privilege_escalation(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test security context disallows privilege escalation."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                ctx = container["securityContext"]
                assert ctx["allowPrivilegeEscalation"] is False

    def test_security_context_drops_all_capabilities(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test security context drops all capabilities."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                ctx = container["securityContext"]
                assert ctx["capabilities"]["drop"] == ["ALL"]

    def test_security_context_has_seccomp_profile(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test security context has RuntimeDefault seccomp profile."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                ctx = container["securityContext"]
                assert ctx["seccompProfile"]["type"] == "RuntimeDefault"

    def test_pod_level_security_context(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pod-level security context is set."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            pod_spec = job["template"]["spec"]["template"]["spec"]
            assert "securityContext" in pod_spec
            pod_ctx = pod_spec["securityContext"]
            assert pod_ctx["runAsNonRoot"] is True
            assert pod_ctx["runAsUser"] == 1000
            assert pod_ctx["fsGroup"] == 1000


class TestJobSetSpecStartupProbes:
    """Tests for AIPerfJobSetSpec startup probe generation.

    Cellular runner containers (cell-controller, cell, aggregators) run to
    terminal under a fail-fast RestartPolicy and carry no probes; only the
    always-on results-sidecar keeps a startup probe.
    """

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = AIPerfJobSetSpec(
            name="startup-test",
            namespace="default",
            job_id="test-startup",
            image="aiperf:latest",
            cells=2,
        )
        return spec.to_k8s_manifest()

    @staticmethod
    def _results_sidecar(manifest: dict[str, Any]) -> dict[str, Any]:
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        return next(
            c
            for c in controller_job["template"]["spec"]["template"]["spec"][
                "containers"
            ]
            if c["name"] == "results-sidecar"
        )

    def test_runner_containers_have_no_startup_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Runner containers (controller/cell/aggregator) carry no startup probe."""
        runner_names = {"controller", "cell", "aggregator"}
        seen = 0
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            for container in job["template"]["spec"]["template"]["spec"]["containers"]:
                if container["name"] in runner_names:
                    seen += 1
                    assert "startupProbe" not in container, (
                        f"{container['name']} unexpectedly has a startupProbe"
                    )
        assert seen >= 2  # cell-controller + at least one cell

    def test_results_sidecar_has_startup_probe(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """The results sidecar keeps its startup probe."""
        assert "startupProbe" in self._results_sidecar(jobset_manifest)

    def test_results_sidecar_startup_probe_has_zero_initial_delay(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Startup probe should check immediately."""
        probe = self._results_sidecar(jobset_manifest)["startupProbe"]
        assert probe["initialDelaySeconds"] == 0

    def test_results_sidecar_startup_probe_allows_long_initialization(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Startup probe should allow long initialization."""
        probe = self._results_sidecar(jobset_manifest)["startupProbe"]
        max_startup_time = probe["failureThreshold"] * probe["periodSeconds"]
        assert max_startup_time >= 120, (
            f"too short max startup time: {max_startup_time}s"
        )

    def test_results_sidecar_startup_probe_uses_health_endpoint(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Startup probe should use /healthz on a real port."""
        probe = self._results_sidecar(jobset_manifest)["startupProbe"]
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] > 0


class TestJobSetSpecResourceAggregation:
    """Tests for AIPerfJobSetSpec resource aggregation methods."""

    @pytest.fixture
    def jobset_spec(self) -> AIPerfJobSetSpec:
        """Create a AIPerfJobSetSpec for testing."""
        return AIPerfJobSetSpec(
            name="resource-test",
            namespace="default",
            job_id="test-resources",
            image="aiperf:latest",
        )

    @pytest.mark.parametrize(
        "cpu_value,expected",
        [
            ("100m", 0.1),
            ("500m", 0.5),
            ("1000m", 1.0),
            ("1", 1.0),
            ("2.5", 2.5),
            ("0", 0.0),
            ("0m", 0.0),
            ("", 0.0),
        ],
    )  # fmt: skip
    def test_parse_cpu(self, cpu_value: str, expected: float) -> None:
        """Test CPU value parsing."""
        result = parse_cpu(cpu_value)
        assert result == expected

    @pytest.mark.parametrize(
        "memory_value,expected",
        [
            ("256Mi", 256),
            ("512Mi", 512),
            ("1Gi", 1024),
            ("2Gi", 2048),
            ("0.5Gi", 512),
            ("1024Ki", 1),
            ("0", 0),
            ("", 0),
        ],
    )  # fmt: skip
    def test_parse_memory(self, memory_value: str, expected: int) -> None:
        """Test memory value parsing."""
        result = parse_memory_mib(memory_value)
        assert result == expected

    def test_split_worker_resources_preserve_total_budget(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Split worker-container resources should sum back to WORKER_POD totals."""
        split = jobset_spec._split_worker_pod_resources(
            worker_count=2,
            record_processor_count=1,
        )
        cpu_total = sum(parse_cpu(item["requests"]["cpu"]) for item in split if item)
        memory_total = sum(
            parse_memory_mib(item["requests"]["memory"]) for item in split if item
        )

        expected = K8sEnvironment.WORKER_POD.to_k8s_resources()
        expected_cpu = parse_cpu(expected["requests"]["cpu"])
        expected_memory = parse_memory_mib(expected["requests"]["memory"])

        assert cpu_total == expected_cpu
        assert memory_total == expected_memory


class TestJobSetSpecEnvVars:
    """Tests for AIPerfJobSetSpec environment variable configuration."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = AIPerfJobSetSpec(
            name="env-test",
            namespace="test-namespace",
            job_id="test-env-123",
            image="aiperf:latest",
        )
        return spec.to_k8s_manifest()

    def test_containers_have_job_id_env(self, jobset_manifest: dict[str, Any]) -> None:
        """Test all containers have AIPERF_JOB_ID environment variable."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                env_names = [e["name"] for e in container["env"]]
                assert "AIPERF_JOB_ID" in env_names
                job_id_env = next(
                    e for e in container["env"] if e["name"] == "AIPERF_JOB_ID"
                )
                assert job_id_env["value"] == "test-env-123"

    def test_containers_have_namespace_env(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test all containers have AIPERF_NAMESPACE environment variable."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                env_names = [e["name"] for e in container["env"]]
                assert "AIPERF_NAMESPACE" in env_names
                ns_env = next(
                    e for e in container["env"] if e["name"] == "AIPERF_NAMESPACE"
                )
                assert ns_env["value"] == "test-namespace"

    def test_runner_containers_have_hf_home_env(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Every cellular runner container tokenizes in-process, so it carries the
        shared HF cache location (HF_HOME). This replaces the retired mesh
        AIPERF_DATASET_MMAP_BASE_PATH dataset-manager wiring.
        """
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                env_dict = {e["name"]: e.get("value") for e in container["env"]}
                assert env_dict.get("HF_HOME") == "/aiperf/hf_home"
                # The retired mesh dataset-manager env is gone.
                assert "AIPERF_DATASET_MMAP_BASE_PATH" not in env_dict

    def test_controller_selects_k8s_cell_launcher(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """The controller pod tells the runner not to spawn cell children
        (AIPERF_CELL_LAUNCHER=k8s); the JobSet already created the cell pods. This
        replaces the retired control-plane realtime-metrics env.
        """
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        cell_controller = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert cell_controller["name"] == "controller"
        env_dict = {e["name"]: e.get("value") for e in cell_controller["env"]}
        assert env_dict["AIPERF_CELL_LAUNCHER"] == "k8s"


class TestJobSetSpecVolumes:
    """Tests for AIPerfJobSetSpec volume configuration."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = AIPerfJobSetSpec(
            name="volume-test",
            namespace="default",
            job_id="test-volumes",
            image="aiperf:latest",
        )
        return spec.to_k8s_manifest()

    def test_pods_have_config_volume(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pods have config volume from ConfigMap."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "config" in volume_names

            config_vol = next(v for v in volumes if v["name"] == "config")
            assert "configMap" in config_vol

    def test_pods_have_tokenizer_cache_volume(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Cellular runner pods carry the shared HF tokenizer-cache emptyDir (the runner
        tokenizes in-process). This replaces the retired mesh `ipc` ZMQ socket volume."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "tokenizer-cache" in volume_names
            assert "ipc" not in volume_names

            cache_vol = next(v for v in volumes if v["name"] == "tokenizer-cache")
            assert "emptyDir" in cache_vol

    def test_pods_have_results_volume(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pods have results emptyDir volume."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "results" in volume_names

    def test_pods_have_no_datasets_volume(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """The mesh dataset-manager mmap `datasets` volume is retired: the cellular
        runner resolves its dataset in-process, so no shared datasets volume is emitted."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "datasets" not in volume_names
            assert "tmp" in volume_names

    def test_config_volume_is_readonly(self, jobset_manifest: dict[str, Any]) -> None:
        """Test config volume mount is read-only."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                config_mount = next(
                    (m for m in container["volumeMounts"] if m["name"] == "config"),
                    None,
                )
                if config_mount:
                    assert config_mount.get("readOnly") is True


class TestJobSetSpecNetworkConfig:
    """Tests for AIPerfJobSetSpec network configuration."""

    def test_jobset_enables_dns_hostnames(self) -> None:
        """Test JobSet enables DNS hostnames for pod communication."""
        spec = AIPerfJobSetSpec(
            name="network-test",
            namespace="default",
            job_id="test-network",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        assert "network" in manifest["spec"]
        assert manifest["spec"]["network"]["enableDNSHostnames"] is True

    def test_jobset_success_policy_targets_controller(self) -> None:
        """Test JobSet success policy only targets controller job."""
        spec = AIPerfJobSetSpec(
            name="policy-test",
            namespace="default",
            job_id="test-policy",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        success_policy = manifest["spec"]["successPolicy"]
        assert success_policy["operator"] == "All"
        assert success_policy["targetReplicatedJobs"] == ["controller"]

    def test_controller_has_never_restart_policy(self) -> None:
        """Test controller pod has Never restart policy."""
        spec = AIPerfJobSetSpec(
            name="restart-test",
            namespace="default",
            job_id="test-restart",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        restart_policy = controller_job["template"]["spec"]["template"]["spec"][
            "restartPolicy"
        ]
        assert restart_policy == "Never"

    def test_cells_have_on_failure_restart_policy(self) -> None:
        """Test cell pods have OnFailure restart policy (restartable stateless slices)."""
        spec = AIPerfJobSetSpec(
            name="restart-test",
            namespace="default",
            job_id="test-restart",
            image="aiperf:latest",
            cells=2,
        )
        manifest = spec.to_k8s_manifest()

        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        restart_policy = cells_job["template"]["spec"]["template"]["spec"][
            "restartPolicy"
        ]
        assert restart_policy == "OnFailure"


class TestJobSetSpecWorkerReplicas:
    """Tests for AIPerfJobSetSpec worker replica configuration."""

    @pytest.mark.parametrize("replicas", [1, 2, 5, 10, 50])
    def test_cells_replicas_set_correctly(self, replicas: int) -> None:
        """Test the cells replica count tracks the `cells` field in the manifest."""
        spec = AIPerfJobSetSpec(
            name="replicas-test",
            namespace="default",
            job_id="test-replicas",
            image="aiperf:latest",
            cells=replicas,
        )
        manifest = spec.to_k8s_manifest()

        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        assert cells_job["replicas"] == replicas

    def test_controller_always_has_one_replica(self) -> None:
        """Test controller always has exactly 1 replica regardless of workers."""
        spec = AIPerfJobSetSpec(
            name="replicas-test",
            namespace="default",
            job_id="test-replicas",
            image="aiperf:latest",
            worker_replicas=100,
        )
        manifest = spec.to_k8s_manifest()

        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        assert controller_job["replicas"] == 1


class TestImagePullPolicy:
    """Tests for ImagePullPolicy enum validation via Pydantic models."""

    @pytest.mark.parametrize(
        "value",
        [
            param("Always", id="always"),
            param("Never", id="never"),
            param("IfNotPresent", id="if-not-present"),
            param("always", id="lowercase"),
            param("ALWAYS", id="uppercase"),
        ],
    )  # fmt: skip
    def test_valid_values(self, value: str) -> None:
        """Test valid image pull policies are accepted by AIPerfContainerSpec."""
        spec = AIPerfContainerSpec(
            name="test", image="img:latest", image_pull_policy=value
        )
        assert spec.image_pull_policy is not None

    def test_none_value(self) -> None:
        """Test None image pull policy is accepted."""
        spec = AIPerfContainerSpec(
            name="test", image="img:latest", image_pull_policy=None
        )
        assert spec.image_pull_policy is None

    @pytest.mark.parametrize(
        "value",
        [
            param("Invalid", id="invalid-value"),
            param("", id="empty-string"),
        ],
    )  # fmt: skip
    def test_invalid_values_raise(self, value: str) -> None:
        """Test invalid image pull policies raise ValidationError."""
        with pytest.raises(ValueError):
            AIPerfContainerSpec(
                name="test", image="img:latest", image_pull_policy=value
            )

    def test_enum_values(self) -> None:
        """Test ImagePullPolicy contains expected values."""
        assert {p.value for p in ImagePullPolicy} == {
            "Always",
            "Never",
            "IfNotPresent",
        }


class TestContainerSpecExtended:
    """Extended tests for AIPerfContainerSpec model."""

    def test_container_with_all_fields(self) -> None:
        """Test AIPerfContainerSpec with all fields set."""
        container = AIPerfContainerSpec(
            name="full-container",
            image="aiperf:v1.0.0",
            image_pull_policy="Always",
            command=["python", "-m", "aiperf"],
            args=["--config", "/etc/config.yaml"],
            env=[{"name": "DEBUG", "value": "true"}],
            resources={
                "requests": {"cpu": "500m", "memory": "512Mi"},
                "limits": {"cpu": "1000m", "memory": "1Gi"},
            },
            volume_mounts=[{"name": "data", "mountPath": "/data"}],
            ports=[
                {"containerPort": 8080, "name": "http"},
                {"containerPort": 9090, "name": "metrics"},
            ],
            startup_probe={"httpGet": {"path": "/startup", "port": 8080}},
            liveness_probe={"httpGet": {"path": "/healthz", "port": 8080}},
            readiness_probe={"httpGet": {"path": "/ready", "port": 8080}},
            security_context={"runAsNonRoot": True},
        )
        spec = container.to_k8s_spec()

        assert spec["name"] == "full-container"
        assert spec["image"] == "aiperf:v1.0.0"
        assert spec["imagePullPolicy"] == "Always"
        assert spec["command"] == ["python", "-m", "aiperf"]
        assert spec["args"] == ["--config", "/etc/config.yaml"]
        assert spec["env"] == [{"name": "DEBUG", "value": "true"}]
        assert spec["resources"]["requests"]["cpu"] == "500m"
        assert spec["volumeMounts"] == [{"name": "data", "mountPath": "/data"}]
        assert len(spec["ports"]) == 2
        assert "startupProbe" in spec
        assert "livenessProbe" in spec
        assert "readinessProbe" in spec
        assert spec["securityContext"]["runAsNonRoot"] is True

    def test_container_to_k8s_spec_with_startup_probe(self) -> None:
        """Test container spec includes startup probe when set."""
        container = AIPerfContainerSpec(
            name="test",
            image="nginx:latest",
            startup_probe={
                "httpGet": {"path": "/startup", "port": 8080},
                "initialDelaySeconds": 0,
                "periodSeconds": 5,
            },
        )
        spec = container.to_k8s_spec()
        assert "startupProbe" in spec
        assert spec["startupProbe"]["httpGet"]["path"] == "/startup"
        assert spec["startupProbe"]["initialDelaySeconds"] == 0

    def test_container_none_image_pull_policy_excluded(self) -> None:
        """Test None image pull policy is excluded from spec."""
        container = AIPerfContainerSpec(
            name="test",
            image="nginx:latest",
            image_pull_policy=None,
        )
        spec = container.to_k8s_spec()
        assert "imagePullPolicy" not in spec


class TestReplicatedJobSpecExtended:
    """Extended tests for AIPerfReplicatedJobSpec model."""

    def test_to_k8s_spec_with_backoff_limit(self) -> None:
        """Test replicated job spec includes backoff limit."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(
            name="workers",
            replicas=3,
            containers=[container],
            backoff_limit=5,
        )
        spec = job.to_k8s_spec()
        assert spec["template"]["spec"]["backoffLimit"] == 5

    def test_to_k8s_spec_with_multiple_containers(self) -> None:
        """Test replicated job spec with multiple containers."""
        containers = [
            AIPerfContainerSpec(name="main", image="main:latest"),
            AIPerfContainerSpec(name="sidecar", image="sidecar:latest"),
        ]
        job = AIPerfReplicatedJobSpec(
            name="multi-container",
            containers=containers,
        )
        spec = job.to_k8s_spec()
        pod_containers = spec["template"]["spec"]["template"]["spec"]["containers"]
        assert len(pod_containers) == 2
        names = [c["name"] for c in pod_containers]
        assert "main" in names
        assert "sidecar" in names

    def test_to_k8s_spec_pod_security_context(self) -> None:
        """Test replicated job has pod-level security context."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(name="secure-job", containers=[container])
        spec = job.to_k8s_spec()
        pod_spec = spec["template"]["spec"]["template"]["spec"]
        assert "securityContext" in pod_spec
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["runAsUser"] == 1000
        assert pod_spec["securityContext"]["fsGroup"] == 1000
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"

    def test_to_k8s_spec_without_customization(self) -> None:
        """Test replicated job spec without pod customization."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        job = AIPerfReplicatedJobSpec(
            name="minimal",
            containers=[container],
            pod_template=None,
        )
        spec = job.to_k8s_spec()
        pod_spec = spec["template"]["spec"]["template"]["spec"]
        # Should not have these optional fields
        assert "nodeSelector" not in pod_spec
        assert "tolerations" not in pod_spec
        assert "imagePullSecrets" not in pod_spec
        assert "serviceAccountName" not in pod_spec

    def test_to_k8s_spec_without_annotations(self) -> None:
        """Test replicated job spec without annotations in customization."""
        container = AIPerfContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(node_selector={"zone": "a"})  # No annotations
        job = AIPerfReplicatedJobSpec(
            name="no-annotations",
            containers=[container],
            pod_template=custom,
        )
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        # Should have labels but no annotations
        assert "labels" in pod_meta
        assert "annotations" not in pod_meta


class TestJobSetSpecPrivateMethods:
    """Tests for AIPerfJobSetSpec private methods."""

    @pytest.fixture
    def jobset_spec(self) -> AIPerfJobSetSpec:
        """Create a AIPerfJobSetSpec for testing private methods."""
        return AIPerfJobSetSpec(
            name="test-private",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
        )

    def test_create_health_probe(self, jobset_spec: AIPerfJobSetSpec) -> None:
        """Test _create_health_probe generates correct probe config."""
        probe = jobset_spec._create_health_probe(port=8080)
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == 8080
        assert "initialDelaySeconds" in probe
        assert "periodSeconds" in probe
        assert "timeoutSeconds" in probe
        assert "failureThreshold" in probe

    def test_create_health_probe_custom_path(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_health_probe with custom path."""
        probe = jobset_spec._create_health_probe(port=9090, path="/custom/health")
        assert probe["httpGet"]["path"] == "/custom/health"
        assert probe["httpGet"]["port"] == 9090

    def test_create_startup_probe(self, jobset_spec: AIPerfJobSetSpec) -> None:
        """Test _create_startup_probe generates correct probe config."""
        probe = jobset_spec._create_startup_probe(port=8080)
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == 8080
        assert probe["initialDelaySeconds"] == 0  # Zero for fast first check
        assert probe["periodSeconds"] == 5
        assert probe["failureThreshold"] == 30  # Allow 150s startup time

    def test_create_startup_probe_custom_path(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_startup_probe with custom path."""
        probe = jobset_spec._create_startup_probe(port=8080, path="/startup")
        assert probe["httpGet"]["path"] == "/startup"

    def test_create_security_context(self, jobset_spec: AIPerfJobSetSpec) -> None:
        """Test _create_security_context generates correct context."""
        ctx = jobset_spec._create_security_context()
        assert ctx["runAsNonRoot"] is True
        assert ctx["runAsUser"] == 1000
        assert ctx["runAsGroup"] == 1000
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["readOnlyRootFilesystem"] is True
        assert ctx["capabilities"]["drop"] == ["ALL"]
        assert ctx["seccompProfile"]["type"] == "RuntimeDefault"

    def test_create_env_vars_without_controller_host(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_env_vars without controller_host."""
        env = jobset_spec._create_env_vars()
        env_names = [e["name"] for e in env]
        env_dict = {e["name"]: e.get("value") for e in env}
        assert "AIPERF_DATASET_MMAP_BASE_PATH" in env_names
        assert "AIPERF_JOB_ID" in env_names
        assert "AIPERF_NAMESPACE" in env_names
        assert "AIPERF_K8S_ZMQ_CONTROLLER_HOST" not in env_names
        assert env_dict["AIPERF_SERVICE_REGISTRATION_TIMEOUT"] == str(
            max(
                Environment.SERVICE.REGISTRATION_TIMEOUT,
                K8sEnvironment.JOBSET.WORKER_CONNECTION_PROBE_TIMEOUT * 2,
            )
        )

    def test_create_env_vars_with_controller_host(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_env_vars with controller_host."""
        env = jobset_spec._create_env_vars(controller_host="controller.default.svc")
        env_dict = {e["name"]: e.get("value") for e in env}
        assert env_dict["AIPERF_K8S_ZMQ_CONTROLLER_HOST"] == "controller.default.svc"
        assert env_dict["AIPERF_SERVICE_REGISTRATION_TIMEOUT"] == str(
            max(
                Environment.SERVICE.REGISTRATION_TIMEOUT,
                K8sEnvironment.JOBSET.WORKER_CONNECTION_PROBE_TIMEOUT * 2,
            )
        )

    def test_create_env_vars_with_pod_customization(self) -> None:
        """Test _create_env_vars includes pod customization env vars."""
        custom = PodTemplateConfig(
            env=[{"name": "CUSTOM_VAR", "value": "custom_value"}]
        )
        spec = AIPerfJobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            pod_template=custom,
        )
        env = spec._create_env_vars()
        env_dict = {e["name"]: e.get("value") for e in env}
        assert env_dict["CUSTOM_VAR"] == "custom_value"

    def test_get_volume_mounts(self, jobset_spec: AIPerfJobSetSpec) -> None:
        """Test _get_volume_mounts returns correct mounts."""
        mounts = jobset_spec._get_volume_mounts()
        mount_names = [m["name"] for m in mounts]
        assert "config" in mount_names
        assert "ipc" in mount_names
        assert "results" in mount_names
        assert "datasets" in mount_names

        # Check config mount is readonly
        config_mount = next(m for m in mounts if m["name"] == "config")
        assert config_mount["readOnly"] is True

    def test_get_volume_mounts_with_secrets(self) -> None:
        """Test _get_volume_mounts includes secret mounts from customization."""
        custom = PodTemplateConfig(
            volumes=[
                {"name": "secret-my-secret", "secret": {"secretName": "my-secret"}}
            ],
            volume_mounts=[
                {
                    "name": "secret-my-secret",
                    "mountPath": "/etc/secrets",
                    "readOnly": True,
                }
            ],
        )
        spec = AIPerfJobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            pod_template=custom,
        )
        mounts = spec._get_volume_mounts()
        mount_names = [m["name"] for m in mounts]
        assert "secret-my-secret" in mount_names


class TestJobSetSpecCreateContainer:
    """Tests for AIPerfJobSetSpec._create_container method."""

    @pytest.fixture
    def jobset_spec(self) -> AIPerfJobSetSpec:
        """Create a AIPerfJobSetSpec for testing."""
        return AIPerfJobSetSpec(
            name="container-test",
            namespace="default",
            job_id="test-456",
            image="aiperf:v2.0",
            image_pull_policy="Never",
        )

    def test_create_container_basic(self, jobset_spec: AIPerfJobSetSpec) -> None:
        """Test _create_container creates correct container spec."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="test-container",
            service_type="worker",
            health_port=8080,
            resources=resources,
        )
        assert container.name == "test-container"
        assert container.image == "aiperf:v2.0"
        assert container.image_pull_policy == "Never"
        assert container.command == ["aiperf"]
        assert "service" in container.args
        assert "--type" in container.args
        assert "worker" in container.args
        assert "--health-port" in container.args
        assert "8080" in container.args

    def test_create_container_with_api_port(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container with API port and no health port."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="api-container",
            service_type="api",
            health_port=None,
            resources=resources,
            api_port=9090,
        )
        assert "--api-port" in container.args
        assert "9090" in container.args
        assert "--health-port" not in container.args
        port_names = [p["name"] for p in container.ports]
        assert "health" not in port_names
        assert "api" in port_names

    def test_create_container_with_controller_host(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container with controller_host adds env var."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="worker",
            service_type="worker",
            health_port=8080,
            resources=resources,
            controller_host="controller.svc",
        )
        env_dict = {e["name"]: e.get("value") for e in container.env}
        assert env_dict.get("AIPERF_K8S_ZMQ_CONTROLLER_HOST") == "controller.svc"

    def test_create_container_with_extra_env(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container with extra environment variables."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        extra_env = [{"name": "EXTRA_VAR", "value": "extra_value"}]
        container = jobset_spec._create_container(
            name="test",
            service_type="worker",
            health_port=8080,
            resources=resources,
            extra_env=extra_env,
        )
        env_dict = {e["name"]: e.get("value") for e in container.env}
        assert env_dict.get("EXTRA_VAR") == "extra_value"

    def test_create_container_skip_readiness_probe(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container with skip_readiness_probe=True."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="no-readiness",
            service_type="system_controller",
            health_port=8080,
            resources=resources,
            skip_readiness_probe=True,
        )
        assert container.readiness_probe is None
        assert container.liveness_probe is not None
        assert container.startup_probe is not None

    def test_create_container_skip_startup_probe(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container with skip_startup_probe=True."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="no-startup",
            service_type="worker",
            health_port=8081,
            resources=resources,
            skip_startup_probe=True,
        )
        assert container.startup_probe is None
        assert container.liveness_probe is not None

    def test_create_container_has_security_context(
        self, jobset_spec: AIPerfJobSetSpec
    ) -> None:
        """Test _create_container sets security context."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="secure",
            service_type="worker",
            health_port=8080,
            resources=resources,
        )
        assert container.security_context is not None
        assert container.security_context["runAsNonRoot"] is True


class TestJobSetSpecResourceParsing:
    """Extended tests for AIPerfJobSetSpec resource parsing methods."""

    @pytest.mark.parametrize(
        "cpu_value,expected",
        [
            param("1500m", 1.5, id="millicores-fractional"),
            param("0.25", 0.25, id="decimal-quarter"),
            param("4", 4.0, id="whole-number"),
            param("10m", 0.01, id="small-millicores"),
        ],
    )  # fmt: skip
    def test_parse_cpu_additional(self, cpu_value: str, expected: float) -> None:
        """Test additional CPU parsing cases."""
        result = parse_cpu(cpu_value)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "memory_value,expected",
        [
            param("1.5Gi", 1536, id="fractional-gi"),
            param("2048Ki", 2, id="kibibytes"),
            param("100", 100, id="plain-number"),
            param("0Mi", 0, id="zero-mi"),
        ],
    )  # fmt: skip
    def test_parse_memory_additional(self, memory_value: str, expected: int) -> None:
        """Test additional memory parsing cases."""
        result = parse_memory_mib(memory_value)
        assert result == expected


class TestJobSetSpecTTLEdgeCases:
    """Tests for AIPerfJobSetSpec TTL handling edge cases."""

    def test_ttl_zero(self) -> None:
        """Test JobSet with TTL of 0 (immediate cleanup)."""
        spec = AIPerfJobSetSpec(
            name="ttl-zero",
            namespace="default",
            job_id="test-ttl-zero",
            image="aiperf:latest",
            ttl_seconds=0,
        )
        manifest = spec.to_k8s_manifest()
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 0

    def test_ttl_explicit_none(self) -> None:
        """Test JobSet with explicit None TTL uses environment default."""
        spec = AIPerfJobSetSpec(
            name="ttl-none",
            namespace="default",
            job_id="test-ttl-none",
            image="aiperf:latest",
            ttl_seconds=None,  # Should use environment default
        )
        manifest = spec.to_k8s_manifest()
        # Should have TTL from environment
        assert "ttlSecondsAfterFinished" in manifest["spec"]


class TestJobSetSpecVolumesWithCustomization:
    """Tests for AIPerfJobSetSpec volumes with pod customization."""

    def test_volumes_include_custom_secrets(self) -> None:
        """Test JobSet volumes include custom secret volumes."""
        custom = PodTemplateConfig(
            volumes=[
                {"name": "secret-tls-cert", "secret": {"secretName": "tls-cert"}},
                {"name": "secret-api-keys", "secret": {"secretName": "api-keys"}},
            ],
            volume_mounts=[
                {"name": "secret-tls-cert", "mountPath": "/etc/tls", "readOnly": True},
                {"name": "secret-api-keys", "mountPath": "/etc/keys", "readOnly": True},
            ],
        )
        spec = AIPerfJobSetSpec(
            name="volumes-test",
            namespace="default",
            job_id="test-volumes",
            image="aiperf:latest",
            pod_template=custom,
        )
        manifest = spec.to_k8s_manifest()

        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        volumes = controller_job["template"]["spec"]["template"]["spec"]["volumes"]
        volume_names = [v["name"] for v in volumes]

        # Check custom secret volumes are present
        assert "secret-tls-cert" in volume_names
        assert "secret-api-keys" in volume_names

        # Verify they reference the correct secrets
        tls_vol = next(v for v in volumes if v["name"] == "secret-tls-cert")
        assert tls_vol["secret"]["secretName"] == "tls-cert"


class TestJobSetSpecConfigMapReference:
    """Tests for AIPerfJobSetSpec ConfigMap reference in volumes."""

    def test_config_volume_references_configmap(self) -> None:
        """Test config volume references correct ConfigMap name."""
        spec = AIPerfJobSetSpec(
            name="my-benchmark",
            namespace="default",
            job_id="test-cm",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        volumes = controller_job["template"]["spec"]["template"]["spec"]["volumes"]
        config_vol = next(v for v in volumes if v["name"] == "config")

        # ConfigMap name should match JobSet name + "-config"
        assert config_vol["configMap"]["name"] == "my-benchmark-config"


class TestJobSetSpecBackoffLimits:
    """Tests for AIPerfJobSetSpec backoff limit configuration."""

    def test_controller_backoff_limit(self) -> None:
        """Test controller job uses environment backoff limit."""
        spec = AIPerfJobSetSpec(
            name="backoff-test",
            namespace="default",
            job_id="test-backoff",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        # Backoff limit is in the Job template spec
        assert controller_job["template"]["spec"]["backoffLimit"] >= 0

    def test_cells_backoff_limit(self) -> None:
        """Test the cells job uses the environment worker backoff limit for retries."""
        spec = AIPerfJobSetSpec(
            name="backoff-test",
            namespace="default",
            job_id="test-backoff",
            image="aiperf:latest",
            cells=2,
        )
        manifest = spec.to_k8s_manifest()

        cells_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "cells"
        )
        # Cells are restartable stateless slices: higher backoff limit for retries.
        assert cells_job["template"]["spec"]["backoffLimit"] >= 0


# =============================================================================
# JobSet version helpers
# =============================================================================


class TestGetJobsetManifestUrl:
    """Tests for get_jobset_manifest_url."""

    def test_uses_fallback_when_none(self) -> None:
        url = get_jobset_manifest_url(None)
        assert JOBSET_FALLBACK_VERSION in url
        assert url.endswith("/manifests.yaml")

    def test_uses_provided_version(self) -> None:
        url = get_jobset_manifest_url("v0.7.1")
        assert "v0.7.1" in url
        assert JOBSET_GITHUB_REPO in url

    def test_url_format(self) -> None:
        url = get_jobset_manifest_url("v1.0.0")
        assert url == (
            f"https://github.com/{JOBSET_GITHUB_REPO}"
            "/releases/download/v1.0.0/manifests.yaml"
        )


class TestGetJobsetInstallHint:
    """Tests for get_jobset_install_hint."""

    def test_contains_kubectl_apply(self) -> None:
        hint = get_jobset_install_hint()
        assert "kubectl apply --server-side" in hint

    def test_uses_specified_version(self) -> None:
        hint = get_jobset_install_hint("v0.6.0")
        assert "v0.6.0" in hint


class TestGetLatestJobsetVersion:
    """Tests for get_latest_jobset_version."""

    async def test_returns_tag_on_success(self) -> None:
        mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
        mock_resp.read.return_value = orjson.dumps({"tag_name": "v0.7.1"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.get.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await get_latest_jobset_version() == "v0.7.1"

    async def test_returns_none_on_network_error(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("timeout"))
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await get_latest_jobset_version() is None


class TestJobSetSpecAlwaysConfigFile:
    """Cellular runner containers always run their frontend subcommand against the
    mounted Config v2 (`--config <mount>/config.yaml`), never the retired mesh
    `--benchmark-run run_config.json` invocation."""

    @staticmethod
    def _make_spec() -> "AIPerfJobSetSpec":
        from aiperf.kubernetes.jobset import AIPerfJobSetSpec

        return AIPerfJobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-001",
            image="aiperf:latest",
            cells=2,
        )

    def test_controller_uses_config_flag(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        controller = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        cell_controller = controller["template"]["spec"]["template"]["spec"][
            "containers"
        ][0]
        args = cell_controller["args"]
        assert args[0] == "controller"
        assert "--config" in args
        assert "--benchmark-run" not in args

    def test_points_to_config_yaml(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        controller = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        args = controller["template"]["spec"]["template"]["spec"]["containers"][0][
            "args"
        ]
        idx = args.index("--config")
        assert args[idx + 1] == f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/config.yaml"

    def test_all_runner_containers_use_config_not_benchmark_run(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        # The runner containers are the frontend subcommands (controller/cell/
        # aggregator); the results-sidecar runs a distinct `results-sidecar` command.
        runner_subcommands = {"controller", "cell", "aggregator"}
        seen = 0
        for rj in manifest["spec"]["replicatedJobs"]:
            containers = rj["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                args = container.get("args", [])
                if args and args[0] in runner_subcommands:
                    seen += 1
                    assert "--config" in args
                    assert "--benchmark-run" not in args
        assert seen >= 2  # at least the controller + one cell


def test_build_env_vars_controller_pod_emits_marker() -> None:
    """Controller-pod call sets AIPERF_CONTROLLER_POD=1."""
    from aiperf.kubernetes.jobset_helpers import build_env_vars

    env = build_env_vars(
        job_id="job-7",
        namespace="ns",
        pod_template=PodTemplateConfig(),
        controller_pod=True,
    )
    names = {e["name"]: e.get("value") for e in env}
    assert names.get("AIPERF_CONTROLLER_POD") == "1"


def test_build_env_vars_worker_pod_omits_marker() -> None:
    """Worker-pod call (default controller_pod=False) does not set the marker."""
    from aiperf.kubernetes.jobset_helpers import build_env_vars

    env = build_env_vars(
        job_id="job-7",
        namespace="ns",
        pod_template=PodTemplateConfig(),
    )
    names = [e["name"] for e in env]
    assert "AIPERF_CONTROLLER_POD" not in names


class TestAggregatorTierCounts:
    """`aggregator_tier_counts` mirrors the Rust `tier_counts` sizing exactly."""

    @pytest.mark.parametrize(
        "cells, fanout, expected",
        [
            param(6, 3, [2], id="single-tier-6-3"),
            param(60, 8, [8], id="single-tier-60-8"),
            param(8, 2, [4, 2], id="two-tier-8-2"),
            param(100, 3, [34, 12, 4, 2], id="deep-100-3"),
            param(6, 1, [], id="flat-fanout-1"),
            param(6, 6, [], id="flat-fanout-ge-cells"),
            param(6, None, [], id="flat-fanout-unset"),
        ],
    )
    def test_tier_counts_match_the_reduction_tree(
        self, cells: int, fanout: int | None, expected: list[int]
    ) -> None:
        assert aggregator_tier_counts(cells, fanout) == expected

    @pytest.mark.parametrize("cells", [4, 6, 7, 8, 60, 100, 1000])
    def test_first_tier_equals_aggregator_count(self, cells: int) -> None:
        """The cells ship to tier 1; its node count must equal `aggregator_count`."""
        tiers = aggregator_tier_counts(cells, 3)
        if tiers:
            assert tiers[0] == aggregator_count(cells, 3)
            # Each tier strictly reduces and the top fits within the fanout.
            assert all(b < a for a, b in zip(tiers, tiers[1:]))
            assert tiers[-1] <= 3

    def test_job_name_single_tier_is_byte_identical(self) -> None:
        """A single-tier tree keeps the bare `aggregators` name (unchanged manifest)."""
        assert aggregator_tier_job_name(0, 1) == "aggregators"

    @pytest.mark.parametrize(
        "tier_index, expected",
        [(0, "aggregators-1"), (1, "aggregators-2"), (3, "aggregators-4")],
    )
    def test_job_name_multi_tier_is_one_indexed(
        self, tier_index: int, expected: str
    ) -> None:
        """A multi-tier tree names each tier `aggregators-{k}`, 1-indexed from tier 1."""
        assert aggregator_tier_job_name(tier_index, 3) == expected


def _agg_env(job: dict[str, Any]) -> dict[str, Any]:
    """The aggregator container's env, keyed by name (value or valueFrom)."""
    container = job["template"]["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container["env"]}


def _jobs_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {j["name"]: j for j in manifest["spec"]["replicatedJobs"]}


class TestMultiTierAggregatorPods:
    """The operator emits one `aggregators-{tier}` replicatedJob per tier for depth >= 2."""

    def _manifest(self, cells: int, fanout: int | None) -> dict[str, Any]:
        return AIPerfJobSetSpec(
            name="aiperf-js",
            namespace="ns",
            job_id="j",
            image="img:1",
            cells=cells,
            cell_agg_fanout=fanout,
        ).to_k8s_manifest()

    def test_flat_run_emits_no_aggregator_jobs(self) -> None:
        """No fanout keeps the flat star: just controller + cells."""
        names = [j["name"] for j in self._manifest(4, None)["spec"]["replicatedJobs"]]
        assert names == ["controller", "cells"]

    def test_single_tier_is_one_bare_aggregators_job(self) -> None:
        """6 cells / fanout 3 -> [2]: one `aggregators` job, byte-identical layout."""
        jobs = _jobs_by_name(self._manifest(6, 3))
        assert "aggregators" in jobs
        assert "aggregators-1" not in jobs
        agg = jobs["aggregators"]
        assert agg["replicas"] == aggregator_count(6, 3) == 2
        # Single tier carries neither the tier index nor a ship template (unchanged).
        env = _agg_env(agg)
        assert "AIPERF_AGG_TIER_INDEX" not in env
        assert "AIPERF_AGG_SHIP_ADDR" not in env
        # It still ships to the controller via the shared cell-controller addr.
        assert env["AIPERF_CELL_CONTROLLER_ADDR"] == (
            "tcp://aiperf-js-controller-0-0.aiperf-js.ns.svc.cluster.local:9500"
        )

    def test_depth_two_emits_one_job_per_tier_with_correct_counts(self) -> None:
        """8 cells / fanout 2 -> [4, 2]: aggregators-1 (4 pods), aggregators-2 (2 pods)."""
        manifest = self._manifest(8, 2)
        jobs = _jobs_by_name(manifest)
        assert [j["name"] for j in manifest["spec"]["replicatedJobs"]] == [
            "controller",
            "cells",
            "aggregators-1",
            "aggregators-2",
        ]
        tiers = aggregator_tier_counts(8, 2)
        assert jobs["aggregators-1"]["replicas"] == tiers[0] == 4
        assert jobs["aggregators-2"]["replicas"] == tiers[1] == 2

    def test_lower_tier_ships_to_its_parent_tier_dns(self) -> None:
        """A lower tier gets its tier index + the parent-tier ship-DNS template."""
        jobs = _jobs_by_name(self._manifest(8, 2))
        env = _agg_env(jobs["aggregators-1"])
        assert env["AIPERF_AGG_TIER_INDEX"] == "0"
        # The parent tier is aggregators-2; the `{agg_id}` placeholder is filled pod-side
        # from the round-robin parent (agg_id % parent_count).
        assert env["AIPERF_AGG_SHIP_ADDR"] == (
            "tcp://aiperf-js-aggregators-2-{agg_id}-0."
            "aiperf-js.ns.svc.cluster.local:9700"
        )

    def test_top_tier_ships_to_the_controller(self) -> None:
        """The top tier gets its tier index but no ship template (ships to controller)."""
        jobs = _jobs_by_name(self._manifest(8, 2))
        env = _agg_env(jobs["aggregators-2"])
        assert env["AIPERF_AGG_TIER_INDEX"] == "1"
        assert "AIPERF_AGG_SHIP_ADDR" not in env
        assert env["AIPERF_CELL_CONTROLLER_ADDR"] == (
            "tcp://aiperf-js-controller-0-0.aiperf-js.ns.svc.cluster.local:9500"
        )

    def test_cells_ship_to_tier_one(self) -> None:
        """Cells ship to the tier-1 job (aggregators-1) via the round-robin DNS template."""
        jobs = _jobs_by_name(self._manifest(8, 2))
        cell = jobs["cells"]["template"]["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e.get("value") for e in cell["env"]}
        assert env["AIPERF_CELL_AGG_DNS_TEMPLATE"] == (
            "tcp://aiperf-js-aggregators-1-{agg_id}-0."
            "aiperf-js.ns.svc.cluster.local:9700"
        )
        assert env["AIPERF_CELL_AGG_FANOUT"] == "2"

    def test_controller_carries_the_expect_dont_spawn_gate(self) -> None:
        """The controller keeps the AGG_DNS_TEMPLATE presence gate under multi-tier."""
        jobs = _jobs_by_name(self._manifest(8, 2))
        ctrl = jobs["controller"]["template"]["spec"]["template"]["spec"]["containers"][
            0
        ]
        env = {e["name"]: e.get("value") for e in ctrl["env"]}
        assert env["AIPERF_CELL_LAUNCHER"] == "k8s"
        assert "AIPERF_CELL_AGG_DNS_TEMPLATE" in env

    def test_every_aggregator_pod_binds_all_interfaces(self) -> None:
        """Each tier's pods bind 0.0.0.0:9700 and expose the aggregator port."""
        jobs = _jobs_by_name(self._manifest(8, 2))
        for name in ("aggregators-1", "aggregators-2"):
            job = jobs[name]
            container = job["template"]["spec"]["template"]["spec"]["containers"][0]
            assert container["ports"] == [{"containerPort": 9700, "name": "cell-agg"}]
            env = _agg_env(job)
            assert env["AIPERF_AGG_BIND"] == "tcp://0.0.0.0:9700"
            # AGG_ID comes from the pod's job-index label (the round-robin id).
            agg_id = env["AIPERF_AGG_ID"]["fieldRef"]["fieldPath"]
            assert agg_id == "metadata.labels['jobset.sigs.k8s.io/job-index']"

    def test_tier_barriers_tile_the_children(self) -> None:
        """Per-tier round-robin barriers (aggregator_children) tile every child exactly."""
        # Tier 1 collects the 8 cells across its 4 pods; tier 2 collects the 4 tier-1
        # pods across its 2 pods. The operator sizes the replicas; the runner derives the
        # per-pod barrier from the same aggregator_children round-robin math.
        cells, fanout = 8, 2
        tiers = aggregator_tier_counts(cells, fanout)
        # Tier 1: children are the cells.
        assert sum(aggregator_children(i, tiers[0], cells) for i in range(tiers[0])) == cells
        # Tier 2: children are the tier-1 pods.
        assert (
            sum(aggregator_children(i, tiers[1], tiers[0]) for i in range(tiers[1]))
            == tiers[0]
        )

    def test_deep_tree_emits_all_tiers(self) -> None:
        """100 cells / fanout 3 -> [34, 12, 4, 2]: four aggregator tier jobs."""
        manifest = self._manifest(100, 3)
        jobs = _jobs_by_name(manifest)
        tiers = aggregator_tier_counts(100, 3)
        assert tiers == [34, 12, 4, 2]
        for k, count in enumerate(tiers):
            name = f"aggregators-{k + 1}"
            assert jobs[name]["replicas"] == count
            env = _agg_env(jobs[name])
            assert env["AIPERF_AGG_TIER_INDEX"] == str(k)
            if k + 1 == len(tiers):
                assert "AIPERF_AGG_SHIP_ADDR" not in env
            else:
                assert f"aggregators-{k + 2}" in env["AIPERF_AGG_SHIP_ADDR"]


class TestArtifactPortExposure:
    """The HTTP artifact plane exposes 9600; the velo transport rides the 9500 plane."""

    def _manifest(self, transport: str | None) -> dict[str, Any]:
        pod_template = (
            PodTemplateConfig(
                env=[{"name": "AIPERF_ARTIFACT_TRANSPORT", "value": transport}]
            )
            if transport is not None
            else PodTemplateConfig()
        )
        return AIPerfJobSetSpec(
            name="aiperf-js",
            namespace="ns",
            job_id="j",
            image="img:1",
            cells=2,
            pod_template=pod_template,
        ).to_k8s_manifest()

    def _controller_ports(self, manifest: dict[str, Any]) -> list[str]:
        ctrl = _jobs_by_name(manifest)["controller"]
        container = ctrl["template"]["spec"]["template"]["spec"]["containers"][0]
        return [p["name"] for p in container["ports"]]

    def _cell_env(self, manifest: dict[str, Any]) -> dict[str, Any]:
        cell = _jobs_by_name(manifest)["cells"]
        container = cell["template"]["spec"]["template"]["spec"]["containers"][0]
        return {e["name"]: e.get("value") for e in container["env"]}

    def test_http_default_exposes_artifact_port(self) -> None:
        """The default (HTTP) transport exposes 9600 on the controller."""
        manifest = self._manifest(None)
        assert self._controller_ports(manifest) == ["cell-ctl", "cell-artifact"]

    def test_http_explicit_exposes_artifact_port(self) -> None:
        """An explicit `http` transport still exposes 9600."""
        assert "cell-artifact" in self._controller_ports(self._manifest("http"))

    def test_http_wires_cell_artifact_port_env(self) -> None:
        """A cell under HTTP gets the explicit controller artifact port (9600)."""
        assert self._cell_env(self._manifest(None))["AIPERF_CELL_ARTIFACT_PORT"] == "9600"

    def test_velo_transport_omits_artifact_port(self) -> None:
        """The velo transport rides the 9500 plane; 9600 is not exposed cross-host."""
        manifest = self._manifest("velo")
        assert self._controller_ports(manifest) == ["cell-ctl"]
        assert "AIPERF_CELL_ARTIFACT_PORT" not in self._cell_env(manifest)
