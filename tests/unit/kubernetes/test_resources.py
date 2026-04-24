# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.resources module."""

import pytest
from pytest import param

from aiperf.config.deployment import DeploymentConfig, PodTemplateConfig
from aiperf.kubernetes.resources import (
    CONFIGMAP_MAX_SIZE_BYTES,
    DNS_LABEL_MAX_LENGTH,
    DNS_LABEL_PATTERN,
    ConfigMapSpec,
    KubernetesDeployment,
    NamespaceSpec,
    RBACSpec,
    validate_dns_label,
)


class TestConfigMapSpec:
    """Tests for ConfigMapSpec model."""

    def test_basic_configmap(self) -> None:
        """Test creating a basic ConfigMap spec."""
        cm = ConfigMapSpec(
            name="test-config",
            namespace="default",
            data={"key": "value"},
        )
        assert cm.name == "test-config"
        assert cm.namespace == "default"
        assert cm.data["key"] == "value"

    def test_to_k8s_manifest(self) -> None:
        """Test converting ConfigMap spec to Kubernetes manifest."""
        cm = ConfigMapSpec(
            name="test-config",
            namespace="aiperf",
            data={"run_config.json": '{"key": "value"}'},
            labels={"app": "aiperf"},
        )
        manifest = cm.to_k8s_manifest()
        assert manifest["apiVersion"] == "v1"
        assert manifest["kind"] == "ConfigMap"
        assert manifest["metadata"]["name"] == "test-config"
        assert manifest["metadata"]["namespace"] == "aiperf"
        assert manifest["metadata"]["labels"]["app"] == "aiperf"
        assert manifest["data"]["run_config.json"] == '{"key": "value"}'

    def test_get_data_size_bytes(self) -> None:
        """Test calculating ConfigMap data size."""
        cm = ConfigMapSpec(
            name="test",
            data={"key1": "value1", "key2": "value2"},
        )
        # key1 + value1 = 4 + 6 = 10, key2 + value2 = 4 + 6 = 10
        assert cm.get_data_size_bytes() == 20

    def test_get_data_size_bytes_empty(self) -> None:
        """Test size calculation for empty ConfigMap."""
        cm = ConfigMapSpec(name="test", data={})
        assert cm.get_data_size_bytes() == 0

    def test_get_data_size_bytes_unicode(self) -> None:
        """Test size calculation with unicode characters."""
        cm = ConfigMapSpec(
            name="test",
            data={"key": "日本語"},  # 3 Japanese characters = 9 bytes in UTF-8
        )
        # key = 3 bytes, 日本語 = 9 bytes
        assert cm.get_data_size_bytes() == 12

    def test_validate_size_within_limit(self) -> None:
        """Test validate_size passes for ConfigMap within limit."""
        cm = ConfigMapSpec(
            name="test",
            data={"key": "value"},
        )
        # Should not raise
        cm.validate_size()

    def test_validate_size_exceeds_limit(self) -> None:
        """Test validate_size raises error when exceeding 1 MiB limit."""
        # Create data that exceeds 1 MiB
        large_value = "x" * (CONFIGMAP_MAX_SIZE_BYTES + 100)
        cm = ConfigMapSpec(
            name="oversized-config",
            data={"large": large_value},
        )
        with pytest.raises(ValueError, match="exceeds Kubernetes limit"):
            cm.validate_size()

    def test_validate_size_error_message_includes_name(self) -> None:
        """Test validate_size error message includes ConfigMap name."""
        large_value = "x" * (CONFIGMAP_MAX_SIZE_BYTES + 100)
        cm = ConfigMapSpec(
            name="my-configmap",
            data={"large": large_value},
        )
        with pytest.raises(ValueError, match="my-configmap"):
            cm.validate_size()

    def test_validate_size_at_boundary(self) -> None:
        """Test validate_size passes at exactly 1 MiB."""
        # Create data exactly at the limit
        # We need key + value to sum to CONFIGMAP_MAX_SIZE_BYTES
        key = "k"
        value_size = CONFIGMAP_MAX_SIZE_BYTES - len(key.encode())
        cm = ConfigMapSpec(
            name="boundary-config",
            data={key: "x" * value_size},
        )
        # Should not raise
        cm.validate_size()
        assert cm.get_data_size_bytes() == CONFIGMAP_MAX_SIZE_BYTES

    def test_validate_size_one_byte_over(self) -> None:
        """Test validate_size fails at exactly 1 byte over limit."""
        key = "k"
        value_size = CONFIGMAP_MAX_SIZE_BYTES - len(key.encode()) + 1
        cm = ConfigMapSpec(
            name="over-limit-config",
            data={key: "x" * value_size},
        )
        with pytest.raises(ValueError, match="exceeds Kubernetes limit"):
            cm.validate_size()

    def test_to_k8s_manifest_includes_all_data_keys(self) -> None:
        """Test to_k8s_manifest includes all data keys."""
        cm = ConfigMapSpec(
            name="multi-key",
            namespace="test-ns",
            data={"key1": "val1", "key2": "val2", "key3": "val3"},
        )
        manifest = cm.to_k8s_manifest()
        assert set(manifest["data"].keys()) == {"key1", "key2", "key3"}

    def test_default_namespace_is_default(self) -> None:
        """Test ConfigMapSpec has 'default' as default namespace."""
        cm = ConfigMapSpec(name="test")
        assert cm.namespace == "default"

    def test_default_data_is_empty_dict(self) -> None:
        """Test ConfigMapSpec has empty dict as default data."""
        cm = ConfigMapSpec(name="test")
        assert cm.data == {}

    def test_default_labels_is_empty_dict(self) -> None:
        """Test ConfigMapSpec has empty dict as default labels."""
        cm = ConfigMapSpec(name="test")
        assert cm.labels == {}


class TestNamespaceSpec:
    """Tests for NamespaceSpec model."""

    def test_basic_namespace(self) -> None:
        """Test creating a basic namespace spec."""
        ns = NamespaceSpec(name="aiperf-test")
        assert ns.name == "aiperf-test"
        assert ns.labels == {}

    def test_namespace_with_labels(self) -> None:
        """Test namespace with labels."""
        ns = NamespaceSpec(
            name="aiperf-test",
            labels={"app": "aiperf", "environment": "test"},
        )
        assert ns.labels["app"] == "aiperf"
        assert ns.labels["environment"] == "test"

    def test_to_k8s_manifest(self) -> None:
        """Test converting namespace spec to Kubernetes manifest."""
        ns = NamespaceSpec(
            name="aiperf-test",
            labels={"app": "aiperf"},
        )
        manifest = ns.to_k8s_manifest()
        assert manifest["apiVersion"] == "v1"
        assert manifest["kind"] == "Namespace"
        assert manifest["metadata"]["name"] == "aiperf-test"
        assert manifest["metadata"]["labels"]["app"] == "aiperf"


class TestRBACSpec:
    """Tests for RBACSpec model."""

    def test_basic_rbac(self) -> None:
        """Test creating a basic RBAC spec."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        assert rbac.name == "aiperf-test"
        assert rbac.namespace == "default"
        assert rbac.service_account == "default"

    def test_rbac_with_service_account(self) -> None:
        """Test RBAC with custom service account."""
        rbac = RBACSpec(
            name="aiperf-test",
            namespace="default",
            job_id="abc12345",
            service_account="aiperf-sa",
        )
        assert rbac.service_account == "aiperf-sa"

    def test_to_role_manifest(self) -> None:
        """Test generating Role manifest."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        assert manifest["apiVersion"] == "rbac.authorization.k8s.io/v1"
        assert manifest["kind"] == "Role"
        assert manifest["metadata"]["name"] == "aiperf-test-role"
        assert manifest["metadata"]["namespace"] == "default"
        assert len(manifest["rules"]) > 0

    def test_role_has_configmap_permissions(self) -> None:
        """Test Role has ConfigMap permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        # Find the configmaps rule
        cm_rule = next(
            (r for r in manifest["rules"] if "configmaps" in r["resources"]), None
        )
        assert cm_rule is not None
        assert "get" in cm_rule["verbs"]
        assert "create" in cm_rule["verbs"]
        assert "update" in cm_rule["verbs"]

    def test_role_has_pod_permissions(self) -> None:
        """Test Role has pod permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        # Find the pods rule
        pods_rule = next(
            (r for r in manifest["rules"] if "pods" in r["resources"]), None
        )
        assert pods_rule is not None
        assert "get" in pods_rule["verbs"]
        assert "list" in pods_rule["verbs"]

    def test_role_has_jobset_permissions(self) -> None:
        """Test Role has JobSet permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        # Find the jobsets rule
        jobset_rule = next(
            (r for r in manifest["rules"] if "jobsets" in r["resources"]), None
        )
        assert jobset_rule is not None
        assert "jobset.x-k8s.io" in jobset_rule["apiGroups"]

    def test_to_role_binding_manifest(self) -> None:
        """Test generating RoleBinding manifest."""
        rbac = RBACSpec(
            name="aiperf-test",
            namespace="default",
            job_id="abc12345",
            service_account="aiperf-sa",
        )
        manifest = rbac.to_role_binding_manifest()
        assert manifest["apiVersion"] == "rbac.authorization.k8s.io/v1"
        assert manifest["kind"] == "RoleBinding"
        assert manifest["metadata"]["name"] == "aiperf-test-binding"
        assert manifest["metadata"]["namespace"] == "default"
        # Check subject
        assert len(manifest["subjects"]) == 1
        assert manifest["subjects"][0]["kind"] == "ServiceAccount"
        assert manifest["subjects"][0]["name"] == "aiperf-sa"
        # Check roleRef
        assert manifest["roleRef"]["kind"] == "Role"
        assert manifest["roleRef"]["name"] == "aiperf-test-role"

    def test_role_has_services_and_endpoints_permissions(self) -> None:
        """Test Role has services and endpoints permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        svc_rule = next(
            (r for r in manifest["rules"] if "services" in r["resources"]), None
        )
        assert svc_rule is not None
        assert "endpoints" in svc_rule["resources"]
        assert "get" in svc_rule["verbs"]
        assert "list" in svc_rule["verbs"]
        assert "watch" in svc_rule["verbs"]
        assert "create" in svc_rule["verbs"]
        assert "delete" in svc_rule["verbs"]

    def test_role_has_events_permissions(self) -> None:
        """Test Role has events permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        events_rule = next(
            (r for r in manifest["rules"] if "events" in r["resources"]), None
        )
        assert events_rule is not None
        assert "" in events_rule["apiGroups"]  # Core API group
        assert "get" in events_rule["verbs"]
        assert "list" in events_rule["verbs"]
        assert "watch" in events_rule["verbs"]
        assert "create" in events_rule["verbs"]
        assert "patch" in events_rule["verbs"]

    def test_role_has_batch_jobs_permissions(self) -> None:
        """Test Role has batch/jobs read permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        jobs_rule = next(
            (r for r in manifest["rules"] if "jobs" in r["resources"]), None
        )
        assert jobs_rule is not None
        assert "batch" in jobs_rule["apiGroups"]
        assert "get" in jobs_rule["verbs"]
        assert "list" in jobs_rule["verbs"]
        assert "watch" in jobs_rule["verbs"]
        # Jobs should be read-only
        assert "create" not in jobs_rule["verbs"]
        assert "delete" not in jobs_rule["verbs"]

    def test_role_has_pods_log_permissions(self) -> None:
        """Test Role has pods/log read permissions."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_manifest()
        pods_rule = next(
            (r for r in manifest["rules"] if "pods" in r["resources"]), None
        )
        assert pods_rule is not None
        assert "pods/log" in pods_rule["resources"]

    def test_role_binding_has_correct_api_group_ref(self) -> None:
        """Test RoleBinding roleRef has correct apiGroup."""
        rbac = RBACSpec(name="aiperf-test", namespace="default", job_id="abc12345")
        manifest = rbac.to_role_binding_manifest()
        assert manifest["roleRef"]["apiGroup"] == "rbac.authorization.k8s.io"

    def test_role_binding_subject_has_namespace(self) -> None:
        """Test RoleBinding subject includes namespace."""
        rbac = RBACSpec(name="aiperf-test", namespace="my-namespace", job_id="abc12345")
        manifest = rbac.to_role_binding_manifest()
        assert manifest["subjects"][0]["namespace"] == "my-namespace"

    def test_helm_benchmark_rbac_is_subset_of_rbac_spec(self) -> None:
        """Helm benchmark-rbac.yaml rules must be a subset of RBACSpec._RULES.

        Prevents drift between operator-deployed and on-the-fly RBAC.
        """
        from pathlib import Path

        import yaml

        helm_file = (
            Path(__file__).parents[3]
            / "deploy/helm/aiperf-operator/templates/benchmark-rbac.yaml"
        )
        if not helm_file.exists():
            pytest.skip("Helm chart not found")

        content = helm_file.read_text()
        # Extract YAML rules between "rules:" and the next "---"
        # by finding raw rule blocks (ignore Helm template directives)
        helm_rules: list[dict] = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- apiGroups:"):
                helm_rules.append({"apiGroups": yaml.safe_load(line.split(": ", 1)[1])})
            elif line.startswith("resources:") and helm_rules:
                helm_rules[-1]["resources"] = yaml.safe_load(line.split(": ", 1)[1])
            elif line.startswith("verbs:") and helm_rules:
                helm_rules[-1]["verbs"] = yaml.safe_load(line.split(": ", 1)[1])

        # Build lookup from RBACSpec._RULES: (apiGroup, resource) -> set(verbs)
        rbac_rules = RBACSpec._RULES
        allowed: dict[tuple[str, str], set[str]] = {}
        for rule in rbac_rules:
            for group in rule["apiGroups"]:
                for resource in rule["resources"]:
                    key = (group, resource)
                    allowed.setdefault(key, set()).update(rule["verbs"])

        # Every Helm rule must be covered by RBACSpec
        for rule in helm_rules:
            for group in rule["apiGroups"]:
                for resource in rule["resources"]:
                    key = (group, resource)
                    assert key in allowed, (
                        f"Helm benchmark-rbac.yaml grants ({group}, {resource}) "
                        f"but RBACSpec._RULES does not"
                    )
                    helm_verbs = set(rule["verbs"])
                    missing = helm_verbs - allowed[key]
                    assert not missing, (
                        f"Helm benchmark-rbac.yaml grants verbs {missing} on "
                        f"({group}, {resource}) not in RBACSpec._RULES"
                    )


class TestKubernetesDeployment:
    """Tests for KubernetesDeployment model."""

    @pytest.fixture
    def basic_deployment(self, sample_config, sample_run) -> KubernetesDeployment:
        """Create a basic KubernetesDeployment for testing."""
        return KubernetesDeployment(
            job_id="test123",
            config=sample_config,
            run=sample_run,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )

    def test_effective_namespace_auto_generated(self, sample_config) -> None:
        """Test effective_namespace uses default shared namespace when None."""
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        deployment = KubernetesDeployment(
            job_id="abc123",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.effective_namespace == DEFAULT_BENCHMARK_NAMESPACE
        assert deployment.auto_namespace is True

    def test_effective_namespace_explicit(self, sample_config) -> None:
        """Test effective_namespace when namespace is specified."""
        deployment = KubernetesDeployment(
            job_id="abc123",
            namespace="my-namespace",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.effective_namespace == "my-namespace"
        assert deployment.auto_namespace is False

    def test_jobset_name(self, basic_deployment: KubernetesDeployment) -> None:
        """Test jobset_name property."""
        assert basic_deployment.jobset_name == "aiperf-test123"

    def test_configmap_name(self, basic_deployment: KubernetesDeployment) -> None:
        """Test configmap_name property."""
        assert basic_deployment.configmap_name == "aiperf-test123-config"

    def test_get_namespace_spec_auto_generated(
        self, basic_deployment: KubernetesDeployment
    ) -> None:
        """Test get_namespace_spec returns spec for default shared namespace."""
        ns_spec = basic_deployment.get_namespace_spec()
        assert ns_spec is not None
        assert ns_spec.name == basic_deployment.effective_namespace
        assert ns_spec.labels["app"] == "aiperf"
        assert ns_spec.labels["aiperf.nvidia.com/auto-generated"] == "true"
        assert "aiperf.nvidia.com/job-id" not in ns_spec.labels

    def test_get_namespace_spec_explicit_returns_none(self, sample_config) -> None:
        """Test get_namespace_spec returns None for explicit namespace."""
        deployment = KubernetesDeployment(
            job_id="abc123",
            namespace="existing-namespace",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.get_namespace_spec() is None

    def test_get_configmap_spec(self, basic_deployment: KubernetesDeployment) -> None:
        """Test get_configmap_spec returns correct spec."""
        cm_spec = basic_deployment.get_configmap_spec()
        assert cm_spec.name == basic_deployment.configmap_name
        assert cm_spec.namespace == basic_deployment.effective_namespace
        assert "run_config.json" in cm_spec.data

    def test_get_rbac_spec(self, basic_deployment: KubernetesDeployment) -> None:
        """Test get_rbac_spec returns correct spec."""
        rbac_spec = basic_deployment.get_rbac_spec()
        assert rbac_spec.name == basic_deployment.jobset_name
        assert rbac_spec.namespace == basic_deployment.effective_namespace

    def test_get_jobset_spec(self, basic_deployment: KubernetesDeployment) -> None:
        """Test get_jobset_spec returns correct spec."""
        jobset_spec = basic_deployment.get_jobset_spec()
        assert jobset_spec.name == basic_deployment.jobset_name
        assert jobset_spec.namespace == basic_deployment.effective_namespace
        assert jobset_spec.job_id == basic_deployment.job_id
        assert jobset_spec.image == basic_deployment.deployment.image

    def test_get_jobset_spec_propagates_resource_mode(self, sample_config) -> None:
        """Test get_jobset_spec carries DeploymentConfig.resource_mode through."""
        deployment = KubernetesDeployment(
            job_id="abc123",
            config=sample_config,
            deployment=DeploymentConfig(
                image="aiperf:latest",
                resource_mode="none",
            ),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.resource_mode == "none"

    def test_get_jobset_spec_propagates_keep_failed_pods(self, sample_config) -> None:
        """Test get_jobset_spec carries keep_failed_pods into JobSet plumbing."""
        deployment = KubernetesDeployment(
            job_id="abc123",
            config=sample_config,
            deployment=DeploymentConfig(
                image="aiperf:latest",
                keep_failed_pods=True,
            ),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.keep_failed_pods is True

    def test_get_all_manifests_auto_namespace(
        self, basic_deployment: KubernetesDeployment
    ) -> None:
        """Test get_all_manifests includes namespace when auto-generated."""
        manifests = basic_deployment.get_all_manifests()
        # Should have: Namespace, Role, RoleBinding, ConfigMap, JobSet
        assert len(manifests) == 5
        kinds = [m["kind"] for m in manifests]
        assert "Namespace" in kinds
        assert "Role" in kinds
        assert "RoleBinding" in kinds
        assert "ConfigMap" in kinds
        assert "JobSet" in kinds

    def test_get_all_manifests_explicit_namespace(self, sample_config) -> None:
        """Test get_all_manifests excludes namespace when explicit."""
        deployment = KubernetesDeployment(
            job_id="abc123",
            namespace="existing",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        manifests = deployment.get_all_manifests()
        # Should have: Role, RoleBinding, ConfigMap, JobSet (no Namespace)
        assert len(manifests) == 4
        kinds = [m["kind"] for m in manifests]
        assert "Namespace" not in kinds

    def test_get_all_manifests_order(
        self, basic_deployment: KubernetesDeployment
    ) -> None:
        """Test manifests are in correct creation order."""
        manifests = basic_deployment.get_all_manifests()
        kinds = [m["kind"] for m in manifests]
        # Namespace first, then RBAC, then ConfigMap, then JobSet
        ns_idx = kinds.index("Namespace")
        role_idx = kinds.index("Role")
        cm_idx = kinds.index("ConfigMap")
        jobset_idx = kinds.index("JobSet")
        assert ns_idx < role_idx < cm_idx < jobset_idx


class TestKubernetesDeploymentWorkers:
    """Tests for KubernetesDeployment worker configuration."""

    def test_worker_replicas_default(self, sample_config) -> None:
        """Test default worker replica count."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.worker_replicas == 1

    @pytest.mark.parametrize("workers", [1, 5, 10, 100])
    def test_worker_replicas_custom(self, workers: int, sample_config) -> None:
        """Test custom worker replica count."""
        deployment = KubernetesDeployment(
            job_id="test",
            worker_replicas=workers,
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.worker_replicas == workers


class TestKubernetesDeploymentTTL:
    """Tests for KubernetesDeployment TTL configuration."""

    def test_ttl_default(self, sample_config) -> None:
        """Test default TTL value."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.deployment.ttl_seconds_after_finished == 300

    @pytest.mark.parametrize("ttl", [0, 60, 300, 3600, None])
    def test_ttl_custom(self, ttl: int | None, sample_config) -> None:
        """Test custom TTL values."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(
                image="aiperf:latest", ttl_seconds_after_finished=ttl
            ),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.ttl_seconds == ttl


class TestKubernetesDeploymentPodCustomization:
    """Tests for KubernetesDeployment pod customization."""

    def test_default_pod_customization(self, sample_config) -> None:
        """Test default pod customization is empty."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.deployment.pod_template.node_selector == {}
        assert deployment.deployment.pod_template.tolerations == []

    def test_pod_customization_propagated_to_jobset(self, sample_config) -> None:
        """Test pod customization is propagated to JobSet spec."""
        custom = PodTemplateConfig(
            node_selector={"gpu": "true"},
            annotations={"monitoring": "enabled"},
        )
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest", pod_template=custom),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.pod_template == custom

    def test_pod_customization_in_manifest(
        self, sample_config, sample_pod_template
    ) -> None:
        """Test pod customization appears in generated manifest."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(
                image="aiperf:latest", pod_template=sample_pod_template
            ),
        )
        manifests = deployment.get_all_manifests()
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        controller_job = jobset["spec"]["replicatedJobs"][0]
        pod_spec = controller_job["template"]["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"] == {"gpu": "true"}


class TestKubernetesDeploymentImagePullPolicy:
    """Tests for KubernetesDeployment image pull policy."""

    @pytest.mark.parametrize(
        "policy",
        ["Always", "Never", "IfNotPresent"],
    )
    def test_image_pull_policy(self, policy: str, sample_config) -> None:
        """Test image pull policy is set correctly."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(
                image="aiperf:latest", image_pull_policy=policy
            ),
        )
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.image_pull_policy == policy


class TestValidateDnsLabel:
    """Tests for validate_dns_label function."""

    @pytest.mark.parametrize(
        "value",
        [
            param("valid", id="simple_lowercase"),
            param("test-123", id="with_hyphen_and_digits"),
            param("a", id="single_char"),
            param("a1", id="letter_digit"),
            param("1a", id="digit_letter"),
            param("abc-def-ghi", id="multiple_hyphens"),
            param("0", id="single_digit"),
            param("a" * 63, id="max_length_63"),
            param("a-b", id="minimal_with_hyphen"),
        ],
    )  # fmt: skip
    def test_valid_dns_labels(self, value: str) -> None:
        """Test valid DNS labels pass validation."""
        result = validate_dns_label(value, "test_field")
        assert result == value

    def test_empty_string_raises(self) -> None:
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dns_label("", "test_field")

    def test_empty_string_error_includes_field_name(self) -> None:
        """Test empty string error message includes field name."""
        with pytest.raises(ValueError, match="my_field cannot be empty"):
            validate_dns_label("", "my_field")

    def test_too_long_raises(self) -> None:
        """Test string exceeding max length raises ValueError."""
        long_value = "a" * 64
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_dns_label(long_value, "test_field")

    def test_too_long_error_includes_value(self) -> None:
        """Test too long error message includes the value."""
        long_value = "a" * 64
        with pytest.raises(ValueError, match=f"'{long_value}'"):
            validate_dns_label(long_value, "test_field")

    def test_custom_max_length(self) -> None:
        """Test custom max_length parameter."""
        with pytest.raises(ValueError, match="exceeds maximum length of 10"):
            validate_dns_label("a" * 11, "test_field", max_length=10)

    def test_custom_max_length_boundary(self) -> None:
        """Test value at exactly custom max_length is valid."""
        result = validate_dns_label("a" * 10, "test_field", max_length=10)
        assert result == "a" * 10

    def test_default_max_length_is_63(self) -> None:
        """Test default max_length matches DNS_LABEL_MAX_LENGTH constant."""
        assert DNS_LABEL_MAX_LENGTH == 63
        # 63 chars should pass
        result = validate_dns_label("a" * 63, "test_field")
        assert len(result) == 63
        # 64 chars should fail
        with pytest.raises(ValueError, match="exceeds maximum length of 63"):
            validate_dns_label("a" * 64, "test_field")

    @pytest.mark.parametrize(
        "value",
        [
            param("Invalid", id="uppercase"),
            param("-invalid", id="starts_with_hyphen"),
            param("invalid-", id="ends_with_hyphen"),
            param("inv_alid", id="underscore"),
            param("inv.alid", id="period"),
            param("inv alid", id="space"),
            param("ALLCAPS", id="all_uppercase"),
            param("CamelCase", id="camel_case"),
            param("test@123", id="at_symbol"),
            param("test#123", id="hash_symbol"),
            param("-", id="single_hyphen"),
            param("--", id="double_hyphen_only"),
        ],
    )  # fmt: skip
    def test_invalid_dns_labels(self, value: str) -> None:
        """Test invalid DNS labels raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            validate_dns_label(value, "test_field")

    def test_invalid_label_error_includes_value_and_field(self) -> None:
        """Test invalid label error message includes value and field name."""
        with pytest.raises(ValueError, match="my_field 'INVALID'"):
            validate_dns_label("INVALID", "my_field")


class TestDnsLabelPattern:
    """Tests for DNS_LABEL_PATTERN regex constant."""

    def test_pattern_matches_valid_labels(self) -> None:
        """Test DNS_LABEL_PATTERN matches valid DNS labels."""
        valid_labels = ["a", "abc", "a1", "1a", "a-b", "abc-123-def"]
        for label in valid_labels:
            assert DNS_LABEL_PATTERN.match(label), f"Pattern should match '{label}'"

    def test_pattern_rejects_invalid_labels(self) -> None:
        """Test DNS_LABEL_PATTERN rejects invalid DNS labels."""
        invalid_labels = ["A", "-a", "a-", "a_b", "a.b", "a b", ""]
        for label in invalid_labels:
            match = DNS_LABEL_PATTERN.match(label)
            if label == "":
                # Empty string doesn't match
                assert match is None
            else:
                # Other invalid patterns should not match
                assert match is None or match.group() != label, (
                    f"Pattern should not match '{label}'"
                )


class TestKubernetesDeploymentJobIdValidation:
    """Tests for KubernetesDeployment job_id validation."""

    def test_valid_job_id(self, sample_config) -> None:
        """Test valid job_id is accepted."""
        deployment = KubernetesDeployment(
            job_id="valid-job-123",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert deployment.job_id == "valid-job-123"

    def test_auto_generated_job_id(self, sample_config) -> None:
        """Test auto-generated job_id is valid DNS label."""
        deployment = KubernetesDeployment(
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        # Auto-generated is 8 hex chars - should be valid
        assert len(deployment.job_id) == 8
        # Should not raise on re-validation
        validate_dns_label(deployment.job_id, "job_id")

    def test_invalid_job_id_uppercase(self, sample_config) -> None:
        """Test uppercase job_id raises validation error."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubernetesDeployment(
                job_id="Invalid",
                config=sample_config,
                deployment=DeploymentConfig(image="aiperf:latest"),
            )

    def test_invalid_job_id_special_chars(self, sample_config) -> None:
        """Test job_id with special characters raises validation error."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubernetesDeployment(
                job_id="job_with_underscores",
                config=sample_config,
                deployment=DeploymentConfig(image="aiperf:latest"),
            )

    def test_job_id_max_length_is_35(self, sample_config) -> None:
        """Test job_id max length is 35 (to fit in resource names)."""
        # 35 chars should pass
        deployment = KubernetesDeployment(
            job_id="a" * 35,
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        assert len(deployment.job_id) == 35

        # 36 chars should fail
        with pytest.raises(ValueError, match="exceeds maximum length of 35"):
            KubernetesDeployment(
                job_id="a" * 36,
                config=sample_config,
                deployment=DeploymentConfig(image="aiperf:latest"),
            )

    def test_job_id_starting_with_hyphen_raises(self, sample_config) -> None:
        """Test job_id starting with hyphen raises validation error."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubernetesDeployment(
                job_id="-invalid",
                config=sample_config,
                deployment=DeploymentConfig(image="aiperf:latest"),
            )

    def test_job_id_ending_with_hyphen_raises(self, sample_config) -> None:
        """Test job_id ending with hyphen raises validation error."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubernetesDeployment(
                job_id="invalid-",
                config=sample_config,
                deployment=DeploymentConfig(image="aiperf:latest"),
            )


class TestKubernetesDeploymentConfigMapValidation:
    """Tests for KubernetesDeployment ConfigMap size validation."""

    def test_get_all_manifests_validates_configmap_size(self, sample_config) -> None:
        """Test get_all_manifests validates ConfigMap size."""
        # This test ensures the validation path is exercised
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        # Should not raise for normal configs
        manifests = deployment.get_all_manifests()
        configmap = next(m for m in manifests if m["kind"] == "ConfigMap")
        assert "data" in configmap

    def test_get_namespace_spec_labels_shared_namespace(self, sample_config) -> None:
        """Test shared namespace labels do not include job_id."""
        deployment = KubernetesDeployment(
            job_id="my-job-123",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        ns_spec = deployment.get_namespace_spec()
        assert ns_spec is not None
        assert ns_spec.labels["app"] == "aiperf"
        assert ns_spec.labels["aiperf.nvidia.com/auto-generated"] == "true"
        assert "aiperf.nvidia.com/job-id" not in ns_spec.labels


class TestKubernetesDeploymentManifestContents:
    """Tests for KubernetesDeployment manifest content details."""

    def test_role_binding_index_in_manifests(self, sample_config) -> None:
        """Test RoleBinding is after Role in manifests."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        manifests = deployment.get_all_manifests()
        kinds = [m["kind"] for m in manifests]
        role_idx = kinds.index("Role")
        binding_idx = kinds.index("RoleBinding")
        assert role_idx < binding_idx

    def test_configmap_contains_run_config(self, sample_config, sample_run) -> None:
        """Test ConfigMap manifest contains run_config.json."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            run=sample_run,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        manifests = deployment.get_all_manifests()
        configmap = next(m for m in manifests if m["kind"] == "ConfigMap")
        assert "run_config.json" in configmap["data"]

    def test_all_manifests_have_namespace(self, sample_config) -> None:
        """Test all namespaced resources have correct namespace."""
        deployment = KubernetesDeployment(
            job_id="test",
            namespace="my-namespace",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest"),
        )
        manifests = deployment.get_all_manifests()
        for manifest in manifests:
            if manifest["kind"] != "Namespace":
                assert manifest["metadata"]["namespace"] == "my-namespace"

    def test_jobset_in_manifests_has_correct_image(self, sample_config) -> None:
        """Test JobSet in manifests has correct image."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="my-registry/aiperf:v1.2.3"),
        )
        manifests = deployment.get_all_manifests()
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        # Check first container of first job
        first_job = jobset["spec"]["replicatedJobs"][0]
        containers = first_job["template"]["spec"]["template"]["spec"]["containers"]
        assert containers[0]["image"] == "my-registry/aiperf:v1.2.3"


class TestKubernetesDeploymentImagePullPolicyPropagation:
    """Tests for KubernetesDeployment image pull policy propagation."""

    def test_image_pull_policy_none_uses_default(self, sample_config) -> None:
        """Test None image_pull_policy lets Kubernetes use default."""
        deployment = KubernetesDeployment(
            job_id="test",
            config=sample_config,
            deployment=DeploymentConfig(image="aiperf:latest", image_pull_policy=None),
        )
        assert deployment.deployment.image_pull_policy is None
        jobset_spec = deployment.get_jobset_spec()
        assert jobset_spec.image_pull_policy is None


class TestConfigMapMaxSizeBytes:
    """Tests for CONFIGMAP_MAX_SIZE_BYTES constant."""

    def test_configmap_max_size_is_1mib(self) -> None:
        """Test CONFIGMAP_MAX_SIZE_BYTES is exactly 1 MiB."""
        assert CONFIGMAP_MAX_SIZE_BYTES == 1_048_576
        assert CONFIGMAP_MAX_SIZE_BYTES == 1024 * 1024


class TestConfigMapSpecFromBenchmarkRun:
    """Tests for ConfigMapSpec.from_benchmark_run()."""

    def _make_benchmark_run(self):
        from pathlib import Path

        from aiperf.config.benchmark import BenchmarkRun
        from aiperf.config.config import BenchmarkConfig

        cfg = BenchmarkConfig.model_validate(
            {
                "models": {"items": [{"name": "test-model"}]},
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": {
                    "main": {
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 128},
                    },
                },
                "phases": {
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 1,
                        "duration": 10,
                    },
                },
            }
        )
        return BenchmarkRun(
            benchmark_id="test-run-001",
            cfg=cfg,
            trial=0,
            artifact_dir=Path("/results"),
        )

    def test_from_benchmark_run_creates_run_config_json(self) -> None:
        run = self._make_benchmark_run()
        cm = ConfigMapSpec.from_benchmark_run(
            name="test-cm", namespace="default", run=run, job_id="test-001"
        )
        assert "run_config.json" in cm.data

    def test_from_benchmark_run_has_correct_labels(self) -> None:
        run = self._make_benchmark_run()
        cm = ConfigMapSpec.from_benchmark_run(
            name="test-cm", namespace="default", run=run, job_id="test-001"
        )
        assert cm.labels["aiperf.nvidia.com/job-id"] == "test-001"
        assert cm.labels["app"] == "aiperf"

    def test_from_benchmark_run_contains_benchmark_id(self) -> None:
        import orjson

        run = self._make_benchmark_run()
        cm = ConfigMapSpec.from_benchmark_run(
            name="test-cm", namespace="default", run=run, job_id="test-001"
        )
        data = orjson.loads(cm.data["run_config.json"])
        assert data["benchmark_id"] == "test-run-001"

    def test_from_benchmark_run_validates_size(self) -> None:
        run = self._make_benchmark_run()
        # Should not raise for a small config
        cm = ConfigMapSpec.from_benchmark_run(
            name="test-cm", namespace="default", run=run, job_id="test-001"
        )
        assert cm.get_data_size_bytes() < CONFIGMAP_MAX_SIZE_BYTES
