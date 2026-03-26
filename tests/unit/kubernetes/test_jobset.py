# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.jobset module."""

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import orjson
import pytest
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import (
    JOBSET_API,
    JOBSET_FALLBACK_VERSION,
    JOBSET_GITHUB_REPO,
    ContainerSpec,
    JobSetAPIConfig,
    JobSetSpec,
    ReplicatedJobSpec,
    get_jobset_install_hint,
    get_jobset_manifest_url,
    get_latest_jobset_version,
)
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib


class TestJobSetAPIConfig:
    """Tests for JobSetAPIConfig dataclass."""

    def test_default_values(self) -> None:
        """Test JobSetAPIConfig has expected default values."""
        config = JobSetAPIConfig()
        assert config.group == "jobset.x-k8s.io"
        assert config.version == "v1alpha2"
        assert config.plural == "jobsets"

    def test_api_version_property(self) -> None:
        """Test api_version property returns correct format."""
        config = JobSetAPIConfig()
        assert config.api_version == "jobset.x-k8s.io/v1alpha2"

    def test_global_jobset_api(self) -> None:
        """Test global JOBSET_API instance."""
        assert JOBSET_API.api_version == "jobset.x-k8s.io/v1alpha2"


class TestContainerSpec:
    """Tests for ContainerSpec model."""

    def test_minimal_container(self) -> None:
        """Test creating a minimal container spec."""
        container = ContainerSpec(name="test", image="nginx:latest")
        assert container.name == "test"
        assert container.image == "nginx:latest"
        assert container.command == []
        assert container.args == []

    def test_container_to_k8s_spec(self) -> None:
        """Test converting container spec to Kubernetes format."""
        container = ContainerSpec(
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
        container = ContainerSpec(
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
        container = ContainerSpec(name="test", image="nginx:latest")
        spec = container.to_k8s_spec()
        assert "command" not in spec
        assert "args" not in spec
        assert "env" not in spec
        assert "livenessProbe" not in spec


class TestReplicatedJobSpec:
    """Tests for ReplicatedJobSpec model."""

    def test_default_values(self) -> None:
        """Test ReplicatedJobSpec has expected defaults."""
        job = ReplicatedJobSpec(name="test")
        assert job.name == "test"
        assert job.replicas == 1
        assert job.restart_policy == "OnFailure"
        assert job.backoff_limit == 0

    def test_to_k8s_spec_basic(self) -> None:
        """Test converting replicated job to Kubernetes format."""
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(
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
        container = ContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            annotations={"custom/annotation": "value"},
            labels={"custom-label": "value"},
            image_pull_secrets=["my-registry"],
            service_account_name="my-sa",
        )
        job = ReplicatedJobSpec(
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
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(name="workers", containers=[container])
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["labels"]["app"] == "aiperf"

    def test_to_k8s_spec_with_job_id_label(self) -> None:
        """Test that job_id is added to pod labels when set."""
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(
            name="workers", containers=[container], job_id="my-benchmark"
        )
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        assert pod_meta["labels"]["app"] == "aiperf"
        assert pod_meta["labels"]["aiperf.nvidia.com/job-id"] == "my-benchmark"

    def test_to_k8s_spec_custom_labels_override(self) -> None:
        """Test that custom labels can override base labels."""
        container = ContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(labels={"app": "custom-app", "team": "platform"})
        job = ReplicatedJobSpec(
            name="workers",
            containers=[container],
            pod_template=custom,
            job_id="test-job",
        )
        spec = job.to_k8s_spec()
        pod_meta = spec["template"]["spec"]["template"]["metadata"]
        # Custom labels override base labels
        assert pod_meta["labels"]["app"] == "custom-app"
        assert pod_meta["labels"]["team"] == "platform"
        assert pod_meta["labels"]["aiperf.nvidia.com/job-id"] == "test-job"


class TestJobSetSpec:
    """Tests for JobSetSpec model."""

    @pytest.fixture
    def basic_jobset_spec(self) -> JobSetSpec:
        """Create a basic JobSetSpec for testing."""
        return JobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            worker_replicas=2,
            workers_per_pod=2,
            record_processors_per_pod=1,
        )

    def test_create_basic_jobset(self, basic_jobset_spec: JobSetSpec) -> None:
        """Test creating a basic JobSetSpec."""
        assert basic_jobset_spec.name == "aiperf-test"
        assert basic_jobset_spec.namespace == "default"
        assert basic_jobset_spec.job_id == "test-123"
        assert basic_jobset_spec.image == "aiperf:latest"
        assert basic_jobset_spec.worker_replicas == 2

    def test_to_k8s_manifest_structure(self, basic_jobset_spec: JobSetSpec) -> None:
        """Test JobSet manifest has correct structure."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert manifest["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
        assert manifest["kind"] == "JobSet"
        assert manifest["metadata"]["name"] == "aiperf-test"
        assert manifest["metadata"]["namespace"] == "default"
        assert manifest["metadata"]["labels"]["app"] == "aiperf"
        assert manifest["metadata"]["labels"]["aiperf.nvidia.com/job-id"] == "test-123"

    def test_to_k8s_manifest_has_controller_and_workers(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test JobSet manifest contains controller and worker jobs."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        jobs = manifest["spec"]["replicatedJobs"]
        assert len(jobs) == 2
        job_names = [j["name"] for j in jobs]
        assert "controller" in job_names
        assert "workers" in job_names

    def test_to_k8s_manifest_controller_replicas(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test controller has exactly 1 replica."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        assert controller_job["replicas"] == 1

    def test_to_k8s_manifest_worker_replicas(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test workers have correct replica count."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        assert worker_job["replicas"] == 2

    def test_to_k8s_manifest_success_policy(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test JobSet has correct success policy."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert manifest["spec"]["successPolicy"]["operator"] == "All"
        assert manifest["spec"]["successPolicy"]["targetReplicatedJobs"] == [
            "controller"
        ]

    def test_to_k8s_manifest_ttl(self, basic_jobset_spec: JobSetSpec) -> None:
        """Test JobSet TTL is set from environment default."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        assert "ttlSecondsAfterFinished" in manifest["spec"]

    def test_to_k8s_manifest_custom_ttl(self) -> None:
        """Test JobSet with custom TTL."""
        spec = JobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            ttl_seconds=600,
        )
        manifest = spec.to_k8s_manifest()
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 600

    def test_to_k8s_manifest_controller_containers(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test controller pod has expected containers.

        Each control-plane service runs in its own container plus a results sidecar.
        """
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        container_names = [c["name"] for c in containers]
        assert set(container_names) == {
            "control-plane",
            "dataset-manager",
            "timing-manager",
            "worker-manager",
            "records-manager",
            "api",
            "gpu-telemetry-manager",
            "server-metrics-manager",
            "results-sidecar",
        }

    def test_to_k8s_manifest_worker_containers(
        self, basic_jobset_spec: JobSetSpec
    ) -> None:
        """Test worker pod has expected containers.

        Worker pods keep a worker-pod-manager for shared infrastructure, while
        workers and record processors each run in their own container.
        """
        manifest = basic_jobset_spec.to_k8s_manifest()
        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        container_names = [c["name"] for c in containers]
        assert "worker-pod-manager" in container_names
        assert "worker-0" in container_names
        assert "worker-1" in container_names
        assert "record-processor-0" in container_names
        assert len(containers) == 4

    def test_to_k8s_manifest_containers_have_image(
        self, basic_jobset_spec: JobSetSpec
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
        spec = JobSetSpec(
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

    def test_to_k8s_manifest_volumes(self, basic_jobset_spec: JobSetSpec) -> None:
        """Test JobSet pods have required volumes."""
        manifest = basic_jobset_spec.to_k8s_manifest()
        controller_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
        )
        volumes = controller_job["template"]["spec"]["template"]["spec"]["volumes"]
        volume_names = [v["name"] for v in volumes]
        assert "config" in volume_names
        assert "ipc" in volume_names
        assert "results" in volume_names


class TestJobSetSpecContainerDetails:
    """Tests for JobSetSpec container configuration details."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = JobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
        )
        return spec.to_k8s_manifest()

    def test_containers_have_health_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test that containers have liveness and readiness probes.

        Note: control-plane skips readiness probe as it manages its own lifecycle.
        """
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "livenessProbe" in container, (
                    f"{container['name']} missing livenessProbe"
                )
                # control-plane skips readiness probe (manages its own lifecycle)
                if container["name"] != "control-plane":
                    assert "readinessProbe" in container, (
                        f"{container['name']} missing readinessProbe"
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

    def test_resource_mode_none_omits_resources(self) -> None:
        """Test that resourceMode=none omits the container resources block."""
        manifest = JobSetSpec(
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
        manifest = JobSetSpec(
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

    def test_worker_containers_have_controller_host(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test worker containers have AIPERF_K8S_ZMQ_CONTROLLER_HOST env var."""
        worker_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        for container in containers:
            env_names = [e["name"] for e in container["env"]]
            assert "AIPERF_K8S_ZMQ_CONTROLLER_HOST" in env_names, (
                f"{container['name']} missing controller host"
            )

    def test_worker_service_containers_have_unique_service_ids(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test worker and record-processor containers get deterministic service IDs."""
        worker_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]

        worker_0 = next(c for c in containers if c["name"] == "worker-0")
        record_processor_0 = next(
            c for c in containers if c["name"] == "record-processor-0"
        )

        assert "--service-id" in worker_0["args"]
        assert "worker_$(AIPERF_POD_INDEX)_0" in worker_0["args"]
        assert "--service-id" in record_processor_0["args"]
        assert "record_processor_$(AIPERF_POD_INDEX)_0" in record_processor_0["args"]

    def test_worker_container_health_ports_are_unique_within_pod(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test worker pod containers get distinct health ports."""
        worker_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        ports = [
            container["ports"][0]["containerPort"]
            for container in containers
            if container["ports"]
        ]
        assert len(ports) == len(set(ports))

    def test_api_container_has_api_port(self, jobset_manifest: dict[str, Any]) -> None:
        """Test API container exposes both health and API ports."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        api_container = next(c for c in containers if c["name"] == "api")
        port_names = [p["name"] for p in api_container["ports"]]
        assert "api" in port_names
        assert "health" in port_names

    def test_api_container_probes_use_api_port(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test API container probes hit the FastAPI port, not the disabled health port."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        api_container = next(c for c in containers if c["name"] == "api")

        assert api_container["startupProbe"]["httpGet"]["port"] == 9090
        assert api_container["livenessProbe"]["httpGet"]["port"] == 9090
        assert api_container["readinessProbe"]["httpGet"]["port"] == 9090

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
    """Tests for JobSetSpec image pull policy handling."""

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
        spec = JobSetSpec(
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
            JobSetSpec(
                name="test",
                namespace="default",
                job_id="test-123",
                image="aiperf:latest",
                image_pull_policy="invalid",
            )

    def test_none_image_pull_policy_valid(self) -> None:
        """Test None image pull policy is valid (uses Kubernetes default)."""
        spec = JobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            image_pull_policy=None,
        )
        assert spec.image_pull_policy is None


class TestContainerSpecImagePullPolicy:
    """Tests for ContainerSpec image pull policy validation."""

    @pytest.mark.parametrize("policy", ["Always", "Never", "IfNotPresent"])
    def test_valid_image_pull_policy(self, policy: str) -> None:
        """Test valid image pull policies are accepted."""
        container = ContainerSpec(
            name="test",
            image="aiperf:latest",
            image_pull_policy=policy,
        )
        assert container.image_pull_policy == policy

    def test_invalid_image_pull_policy_raises(self) -> None:
        """Test invalid image pull policy raises ValueError."""
        with pytest.raises(ValueError, match="image_pull_policy"):
            ContainerSpec(
                name="test",
                image="aiperf:latest",
                image_pull_policy="BadValue",
            )


class TestJobSetSpecDNSConfiguration:
    """Tests for JobSetSpec DNS naming and configuration."""

    def test_controller_dns_format_includes_full_fqdn(self) -> None:
        """Test that controller DNS includes .svc.cluster.local suffix.

        The DNS format must be:
        {jobset-name}-controller-0-0.{jobset-name}.{namespace}.svc.cluster.local

        This ensures proper DNS resolution across the cluster.
        """
        spec = JobSetSpec(
            name="my-jobset",
            namespace="test-ns",
            job_id="test-123",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        # Find worker-pod-manager and check its AIPERF_K8S_ZMQ_CONTROLLER_HOST env var
        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        wpm_container = next(c for c in containers if c["name"] == "worker-pod-manager")

        controller_host_env = next(
            e
            for e in wpm_container["env"]
            if e["name"] == "AIPERF_K8S_ZMQ_CONTROLLER_HOST"
        )
        controller_host = controller_host_env["value"]

        # Verify full FQDN format
        assert controller_host.endswith(".svc.cluster.local"), (
            f"DNS should end with .svc.cluster.local, got: {controller_host}"
        )
        assert "my-jobset-controller-0-0" in controller_host
        assert "my-jobset.test-ns" in controller_host

    def test_controller_dns_format_correct_structure(self) -> None:
        """Test controller DNS has correct structure: pod.service.namespace.svc.cluster.local"""
        spec = JobSetSpec(
            name="aiperf-abc123",
            namespace="my-namespace",
            job_id="abc123",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        wpm_container = next(c for c in containers if c["name"] == "worker-pod-manager")

        controller_host_env = next(
            e
            for e in wpm_container["env"]
            if e["name"] == "AIPERF_K8S_ZMQ_CONTROLLER_HOST"
        )
        controller_host = controller_host_env["value"]

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
        spec = JobSetSpec(
            name="test-jobset",
            namespace=namespace,
            job_id="test-123",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
        wpm_container = next(c for c in containers if c["name"] == "worker-pod-manager")

        controller_host_env = next(
            e
            for e in wpm_container["env"]
            if e["name"] == "AIPERF_K8S_ZMQ_CONTROLLER_HOST"
        )
        controller_host = controller_host_env["value"]

        # Verify namespace is in the DNS
        assert f".{namespace}." in controller_host
        assert controller_host.endswith(".svc.cluster.local")


class TestJobSetSpecSecurityContext:
    """Tests for JobSetSpec security context generation."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = JobSetSpec(
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
    """Tests for JobSetSpec startup probe generation."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = JobSetSpec(
            name="startup-test",
            namespace="default",
            job_id="test-startup",
            image="aiperf:latest",
        )
        return spec.to_k8s_manifest()

    def test_containers_have_startup_probes(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test all containers have startup probes."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "startupProbe" in container, (
                    f"{container['name']} missing startupProbe"
                )

    def test_startup_probe_has_zero_initial_delay(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test startup probes have zero initial delay for fast first check."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                probe = container["startupProbe"]
                assert probe["initialDelaySeconds"] == 0

    def test_startup_probe_allows_long_initialization(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test startup probes allow containers time to initialize."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                probe = container["startupProbe"]
                # failureThreshold * periodSeconds >= 120s (reasonable startup time)
                max_startup_time = probe["failureThreshold"] * probe["periodSeconds"]
                assert max_startup_time >= 120, (
                    f"{container['name']} has too short max startup time: "
                    f"{max_startup_time}s"
                )

    def test_startup_probe_uses_health_endpoint(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test startup probes use the /healthz endpoint."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                probe = container["startupProbe"]
                assert probe["httpGet"]["path"] == "/healthz"
                assert probe["httpGet"]["port"] > 0


class TestJobSetSpecResourceAggregation:
    """Tests for JobSetSpec resource aggregation methods."""

    @pytest.fixture
    def jobset_spec(self) -> JobSetSpec:
        """Create a JobSetSpec for testing."""
        return JobSetSpec(
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
        self, jobset_spec: JobSetSpec
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
    """Tests for JobSetSpec environment variable configuration."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = JobSetSpec(
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

    def test_containers_have_dataset_path_env(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test all containers have AIPERF_DATASET_MMAP_BASE_PATH."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            containers = job["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                env_names = [e["name"] for e in container["env"]]
                assert "AIPERF_DATASET_MMAP_BASE_PATH" in env_names

    def test_control_plane_has_realtime_metrics_env(
        self, jobset_manifest: dict[str, Any]
    ) -> None:
        """Test control-plane container has realtime metrics enabled."""
        controller_job = next(
            j
            for j in jobset_manifest["spec"]["replicatedJobs"]
            if j["name"] == "controller"
        )
        containers = controller_job["template"]["spec"]["template"]["spec"][
            "containers"
        ]
        control_plane = next(c for c in containers if c["name"] == "control-plane")

        env_dict = {e["name"]: e.get("value") for e in control_plane["env"]}
        assert env_dict["AIPERF_UI_REALTIME_METRICS_ENABLED"] == "true"


class TestJobSetSpecVolumes:
    """Tests for JobSetSpec volume configuration."""

    @pytest.fixture
    def jobset_manifest(self) -> dict[str, Any]:
        """Create a JobSet manifest for testing."""
        spec = JobSetSpec(
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

    def test_pods_have_ipc_volume(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pods have IPC emptyDir volume."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "ipc" in volume_names

            ipc_vol = next(v for v in volumes if v["name"] == "ipc")
            assert "emptyDir" in ipc_vol

    def test_pods_have_results_volume(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pods have results emptyDir volume."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "results" in volume_names

    def test_pods_have_datasets_volume(self, jobset_manifest: dict[str, Any]) -> None:
        """Test pods have datasets emptyDir volume."""
        for job in jobset_manifest["spec"]["replicatedJobs"]:
            volumes = job["template"]["spec"]["template"]["spec"]["volumes"]
            volume_names = [v["name"] for v in volumes]
            assert "datasets" in volume_names

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
    """Tests for JobSetSpec network configuration."""

    def test_jobset_enables_dns_hostnames(self) -> None:
        """Test JobSet enables DNS hostnames for pod communication."""
        spec = JobSetSpec(
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
        spec = JobSetSpec(
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
        spec = JobSetSpec(
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

    def test_workers_have_on_failure_restart_policy(self) -> None:
        """Test worker pods have OnFailure restart policy."""
        spec = JobSetSpec(
            name="restart-test",
            namespace="default",
            job_id="test-restart",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        restart_policy = worker_job["template"]["spec"]["template"]["spec"][
            "restartPolicy"
        ]
        assert restart_policy == "OnFailure"


class TestJobSetSpecWorkerReplicas:
    """Tests for JobSetSpec worker replica configuration."""

    @pytest.mark.parametrize("replicas", [1, 2, 5, 10, 50])
    def test_worker_replicas_set_correctly(self, replicas: int) -> None:
        """Test worker replica count is set correctly in manifest."""
        spec = JobSetSpec(
            name="replicas-test",
            namespace="default",
            job_id="test-replicas",
            image="aiperf:latest",
            worker_replicas=replicas,
        )
        manifest = spec.to_k8s_manifest()

        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        assert worker_job["replicas"] == replicas

    def test_controller_always_has_one_replica(self) -> None:
        """Test controller always has exactly 1 replica regardless of workers."""
        spec = JobSetSpec(
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
        """Test valid image pull policies are accepted by ContainerSpec."""
        spec = ContainerSpec(name="test", image="img:latest", image_pull_policy=value)
        assert spec.image_pull_policy is not None

    def test_none_value(self) -> None:
        """Test None image pull policy is accepted."""
        spec = ContainerSpec(name="test", image="img:latest", image_pull_policy=None)
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
            ContainerSpec(name="test", image="img:latest", image_pull_policy=value)

    def test_enum_values(self) -> None:
        """Test ImagePullPolicy contains expected values."""
        assert {p.value for p in ImagePullPolicy} == {
            "Always",
            "Never",
            "IfNotPresent",
        }


class TestJobSetAPIConfigCustom:
    """Additional tests for JobSetAPIConfig with custom values."""

    def test_custom_values(self) -> None:
        """Test JobSetAPIConfig with custom values."""
        config = JobSetAPIConfig(
            group="custom.example.com",
            version="v1beta1",
            plural="customjobsets",
        )
        assert config.group == "custom.example.com"
        assert config.version == "v1beta1"
        assert config.plural == "customjobsets"
        assert config.api_version == "custom.example.com/v1beta1"

    def test_frozen_dataclass(self) -> None:
        """Test JobSetAPIConfig is immutable."""
        config = JobSetAPIConfig()
        with pytest.raises(FrozenInstanceError):
            config.group = "modified"  # type: ignore[misc]


class TestContainerSpecExtended:
    """Extended tests for ContainerSpec model."""

    def test_container_with_all_fields(self) -> None:
        """Test ContainerSpec with all fields set."""
        container = ContainerSpec(
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
        container = ContainerSpec(
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
        container = ContainerSpec(
            name="test",
            image="nginx:latest",
            image_pull_policy=None,
        )
        spec = container.to_k8s_spec()
        assert "imagePullPolicy" not in spec


class TestReplicatedJobSpecExtended:
    """Extended tests for ReplicatedJobSpec model."""

    def test_to_k8s_spec_with_backoff_limit(self) -> None:
        """Test replicated job spec includes backoff limit."""
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(
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
            ContainerSpec(name="main", image="main:latest"),
            ContainerSpec(name="sidecar", image="sidecar:latest"),
        ]
        job = ReplicatedJobSpec(
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
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(name="secure-job", containers=[container])
        spec = job.to_k8s_spec()
        pod_spec = spec["template"]["spec"]["template"]["spec"]
        assert "securityContext" in pod_spec
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["runAsUser"] == 1000
        assert pod_spec["securityContext"]["fsGroup"] == 1000
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"

    def test_to_k8s_spec_without_customization(self) -> None:
        """Test replicated job spec without pod customization."""
        container = ContainerSpec(name="worker", image="nginx:latest")
        job = ReplicatedJobSpec(
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
        container = ContainerSpec(name="worker", image="nginx:latest")
        custom = PodTemplateConfig(node_selector={"zone": "a"})  # No annotations
        job = ReplicatedJobSpec(
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
    """Tests for JobSetSpec private methods."""

    @pytest.fixture
    def jobset_spec(self) -> JobSetSpec:
        """Create a JobSetSpec for testing private methods."""
        return JobSetSpec(
            name="test-private",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
        )

    def test_create_health_probe(self, jobset_spec: JobSetSpec) -> None:
        """Test _create_health_probe generates correct probe config."""
        probe = jobset_spec._create_health_probe(port=8080)
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == 8080
        assert "initialDelaySeconds" in probe
        assert "periodSeconds" in probe
        assert "timeoutSeconds" in probe
        assert "failureThreshold" in probe

    def test_create_health_probe_custom_path(self, jobset_spec: JobSetSpec) -> None:
        """Test _create_health_probe with custom path."""
        probe = jobset_spec._create_health_probe(port=9090, path="/custom/health")
        assert probe["httpGet"]["path"] == "/custom/health"
        assert probe["httpGet"]["port"] == 9090

    def test_create_startup_probe(self, jobset_spec: JobSetSpec) -> None:
        """Test _create_startup_probe generates correct probe config."""
        probe = jobset_spec._create_startup_probe(port=8080)
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == 8080
        assert probe["initialDelaySeconds"] == 0  # Zero for fast first check
        assert probe["periodSeconds"] == 5
        assert probe["failureThreshold"] == 30  # Allow 150s startup time

    def test_create_startup_probe_custom_path(self, jobset_spec: JobSetSpec) -> None:
        """Test _create_startup_probe with custom path."""
        probe = jobset_spec._create_startup_probe(port=8080, path="/startup")
        assert probe["httpGet"]["path"] == "/startup"

    def test_create_security_context(self, jobset_spec: JobSetSpec) -> None:
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
        self, jobset_spec: JobSetSpec
    ) -> None:
        """Test _create_env_vars without controller_host."""
        env = jobset_spec._create_env_vars()
        env_names = [e["name"] for e in env]
        assert "AIPERF_DATASET_MMAP_BASE_PATH" in env_names
        assert "AIPERF_JOB_ID" in env_names
        assert "AIPERF_NAMESPACE" in env_names
        assert "AIPERF_K8S_ZMQ_CONTROLLER_HOST" not in env_names

    def test_create_env_vars_with_controller_host(
        self, jobset_spec: JobSetSpec
    ) -> None:
        """Test _create_env_vars with controller_host."""
        env = jobset_spec._create_env_vars(controller_host="controller.default.svc")
        env_dict = {e["name"]: e.get("value") for e in env}
        assert env_dict["AIPERF_K8S_ZMQ_CONTROLLER_HOST"] == "controller.default.svc"

    def test_create_env_vars_with_pod_customization(self) -> None:
        """Test _create_env_vars includes pod customization env vars."""
        custom = PodTemplateConfig(
            env=[{"name": "CUSTOM_VAR", "value": "custom_value"}]
        )
        spec = JobSetSpec(
            name="test",
            namespace="default",
            job_id="test-123",
            image="aiperf:latest",
            pod_template=custom,
        )
        env = spec._create_env_vars()
        env_dict = {e["name"]: e.get("value") for e in env}
        assert env_dict["CUSTOM_VAR"] == "custom_value"

    def test_get_volume_mounts(self, jobset_spec: JobSetSpec) -> None:
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
        spec = JobSetSpec(
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
    """Tests for JobSetSpec._create_container method."""

    @pytest.fixture
    def jobset_spec(self) -> JobSetSpec:
        """Create a JobSetSpec for testing."""
        return JobSetSpec(
            name="container-test",
            namespace="default",
            job_id="test-456",
            image="aiperf:v2.0",
            image_pull_policy="Never",
        )

    def test_create_container_basic(self, jobset_spec: JobSetSpec) -> None:
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

    def test_create_container_with_api_port(self, jobset_spec: JobSetSpec) -> None:
        """Test _create_container with API port."""
        resources = {"requests": {"cpu": "100m"}, "limits": {"cpu": "500m"}}
        container = jobset_spec._create_container(
            name="api-container",
            service_type="api",
            health_port=8080,
            resources=resources,
            api_port=9090,
        )
        assert "--api-port" in container.args
        assert "9090" in container.args
        port_names = [p["name"] for p in container.ports]
        assert "health" in port_names
        assert "api" in port_names

    def test_create_container_with_controller_host(
        self, jobset_spec: JobSetSpec
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

    def test_create_container_with_extra_env(self, jobset_spec: JobSetSpec) -> None:
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
        self, jobset_spec: JobSetSpec
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

    def test_create_container_has_security_context(
        self, jobset_spec: JobSetSpec
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
    """Extended tests for JobSetSpec resource parsing methods."""

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
    """Tests for JobSetSpec TTL handling edge cases."""

    def test_ttl_zero(self) -> None:
        """Test JobSet with TTL of 0 (immediate cleanup)."""
        spec = JobSetSpec(
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
        spec = JobSetSpec(
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
    """Tests for JobSetSpec volumes with pod customization."""

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
        spec = JobSetSpec(
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
    """Tests for JobSetSpec ConfigMap reference in volumes."""

    def test_config_volume_references_configmap(self) -> None:
        """Test config volume references correct ConfigMap name."""
        spec = JobSetSpec(
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
    """Tests for JobSetSpec backoff limit configuration."""

    def test_controller_backoff_limit(self) -> None:
        """Test controller job uses environment backoff limit."""
        spec = JobSetSpec(
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

    def test_worker_backoff_limit(self) -> None:
        """Test worker job uses environment backoff limit."""
        spec = JobSetSpec(
            name="backoff-test",
            namespace="default",
            job_id="test-backoff",
            image="aiperf:latest",
        )
        manifest = spec.to_k8s_manifest()

        worker_job = next(
            j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "workers"
        )
        # Workers should have higher backoff limit for retries
        assert worker_job["template"]["spec"]["backoffLimit"] >= 0


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


class TestJobSetSpecAlwaysBenchmarkRun:
    """Verify JobSetSpec always uses --benchmark-run with run_config.json."""

    @staticmethod
    def _make_spec() -> "JobSetSpec":
        from aiperf.kubernetes.jobset import JobSetSpec

        return JobSetSpec(
            name="aiperf-test",
            namespace="default",
            job_id="test-001",
            image="aiperf:latest",
        )

    def test_uses_benchmark_run_flag(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        containers = manifest["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]["containers"]
        args = containers[0]["args"]
        assert "--benchmark-run" in args
        assert "--config-file" not in args

    def test_points_to_run_config_json(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        containers = manifest["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]["containers"]
        args = containers[0]["args"]
        idx = args.index("--benchmark-run")
        assert "run_config.json" in args[idx + 1]

    def test_all_containers_use_benchmark_run(self) -> None:
        spec = self._make_spec()
        manifest = spec.to_k8s_manifest()
        for rj in manifest["spec"]["replicatedJobs"]:
            containers = rj["template"]["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == "results-sidecar":
                    continue
                assert "--benchmark-run" in container["args"]
                assert "--config-file" not in container["args"]
