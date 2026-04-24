# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.config.kube module."""

import pytest
from pytest import param

from aiperf.config.deployment import (
    DeploymentConfig,
    PodTemplateConfig,
    SchedulingConfig,
)
from aiperf.config.kube import (
    KubeManageOptions,
    KubeOptions,
    SecretMountConfig,
)

# =============================================================================
# SecretMountConfig Tests
# =============================================================================


class TestSecretMountConfig:
    """Tests for SecretMountConfig model."""

    def test_required_fields(self) -> None:
        """Test SecretMountConfig requires name and mount_path."""
        config = SecretMountConfig(name="my-secret", mount_path="/secrets/creds")

        assert config.name == "my-secret"
        assert config.mount_path == "/secrets/creds"
        assert config.sub_path is None

    def test_with_sub_path(self) -> None:
        """Test SecretMountConfig with sub_path."""
        config = SecretMountConfig(
            name="api-keys", mount_path="/secrets/api", sub_path="api_key"
        )

        assert config.name == "api-keys"
        assert config.mount_path == "/secrets/api"
        assert config.sub_path == "api_key"

    def test_missing_name_raises_error(self) -> None:
        """Test SecretMountConfig raises error when name is missing."""
        with pytest.raises(ValueError, match="name"):
            SecretMountConfig(mount_path="/secrets/creds")  # type: ignore

    def test_missing_mount_path_raises_error(self) -> None:
        """Test SecretMountConfig raises error when mount_path is missing."""
        with pytest.raises(ValueError, match="mount_path"):
            SecretMountConfig(name="my-secret")  # type: ignore


# =============================================================================
# KubeManageOptions Tests
# =============================================================================


class TestKubeManageOptions:
    """Tests for KubeManageOptions model."""

    def test_default_values(self) -> None:
        """Test KubeManageOptions has expected default values."""
        options = KubeManageOptions()

        assert options.kubeconfig is None
        assert options.namespace is None

    def test_custom_kubeconfig(self) -> None:
        """Test KubeManageOptions accepts custom kubeconfig."""
        options = KubeManageOptions(kubeconfig="/path/to/kubeconfig")

        assert options.kubeconfig == "/path/to/kubeconfig"

    def test_custom_namespace(self) -> None:
        """Test KubeManageOptions accepts custom namespace."""
        options = KubeManageOptions(namespace="my-namespace")

        assert options.namespace == "my-namespace"

    def test_both_fields_set(self) -> None:
        """Test KubeManageOptions with both fields set."""
        options = KubeManageOptions(
            kubeconfig="~/.kube/prod-config", namespace="production"
        )

        assert options.kubeconfig == "~/.kube/prod-config"
        assert options.namespace == "production"


# =============================================================================
# KubeOptions Inheritance Tests
# =============================================================================


class TestKubeOptionsInheritance:
    """Tests for KubeOptions inheritance from KubeManageOptions."""

    def test_kube_options_inherits_from_manage_options(self) -> None:
        """Test KubeOptions inherits from KubeManageOptions."""
        assert issubclass(KubeOptions, KubeManageOptions)

    def test_kube_options_has_manage_fields(self) -> None:
        """Test KubeOptions includes fields from KubeManageOptions."""
        options = KubeOptions(
            image="aiperf:latest", kubeconfig="/path", namespace="test"
        )

        assert options.kubeconfig == "/path"
        assert options.namespace == "test"
        assert options.image == "aiperf:latest"

    def test_kube_options_isinstance_of_manage_options(self) -> None:
        """Test KubeOptions instance is also instance of KubeManageOptions."""
        options = KubeOptions(image="aiperf:latest")

        assert isinstance(options, KubeManageOptions)

    def test_manage_options_fields_have_defaults_in_kube_options(self) -> None:
        """Test inherited fields have same defaults in KubeOptions."""
        options = KubeOptions(image="aiperf:latest")

        assert options.kubeconfig is None
        assert options.namespace is None


# =============================================================================
# KubeOptions Name Field Tests
# =============================================================================


class TestKubeOptionsNameField:
    """Tests for the KubeOptions name field validation."""

    @pytest.mark.parametrize(
        "name",
        [
            param("my-benchmark", id="simple-name"),
            param("a", id="single-char"),
            param("test-run-2024-01-15", id="with-date"),
            param("a" * 40, id="max-length"),
        ],
    )  # fmt: skip
    def test_name_valid_dns_label(self, name: str) -> None:
        """Test valid DNS label names are accepted."""
        options = KubeOptions(image="aiperf:latest", name=name)
        assert options.name == name

    def test_name_none_is_valid(self) -> None:
        """Test name=None is the default and valid."""
        options = KubeOptions(image="aiperf:latest")
        assert options.name is None

    @pytest.mark.parametrize(
        "name",
        [
            param("UPPERCASE", id="uppercase"),
            param("has spaces", id="spaces"),
            param("-starts-with-dash", id="leading-dash"),
            param("ends-with-dash-", id="trailing-dash"),
            param("has_underscores", id="underscores"),
            param("has.dots", id="dots"),
        ],
    )  # fmt: skip
    def test_name_invalid_raises(self, name: str) -> None:
        """Test invalid DNS label names are rejected."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubeOptions(image="aiperf:latest", name=name)

    def test_name_too_long_raises(self) -> None:
        """Test names exceeding 40 chars are rejected."""
        with pytest.raises(ValueError, match="exceeds maximum length of 40"):
            KubeOptions(image="aiperf:latest", name="a" * 41)


# =============================================================================
# KubeOptions Edge Cases
# =============================================================================


class TestKubeOptionsEdgeCases:
    """Tests for KubeOptions boundary/edge-case values."""

    def test_workers_zero_raises(self) -> None:
        """Test workers=0 is rejected (must be > 0)."""
        with pytest.raises(ValueError, match="workers"):
            KubeOptions(image="aiperf:latest", workers=0)

    def test_workers_negative_raises(self) -> None:
        """Test workers=-1 is rejected (must be > 0)."""
        with pytest.raises(ValueError, match="workers"):
            KubeOptions(image="aiperf:latest", workers=-1)

    def test_workers_one_accepted(self) -> None:
        """Test workers=1 is the minimum valid value."""
        options = KubeOptions(image="aiperf:latest", workers=1)
        assert options.workers == 1

    def test_empty_image_raises(self) -> None:
        """Test image='' is rejected (must be non-empty)."""
        with pytest.raises(ValueError, match="image"):
            KubeOptions(image="")

    def test_ttl_seconds_zero_accepted(self) -> None:
        """Test ttl_seconds=0 is accepted (immediate cleanup)."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=0)
        assert options.ttl_seconds == 0


# =============================================================================
# KubeOptions Tests - Default Values
# =============================================================================


class TestKubeOptionsDefaults:
    """Tests for KubeOptions default values."""

    def test_image_is_required(self) -> None:
        """Test KubeOptions requires image field."""
        with pytest.raises(ValueError, match="image"):
            KubeOptions()  # type: ignore

    def test_default_values(self) -> None:
        """Test KubeOptions has expected default values."""
        options = KubeOptions(image="aiperf:latest")

        assert options.image == "aiperf:latest"
        assert options.namespace is None
        assert options.workers == 10  # Default workers per pod
        assert options.ttl_seconds == 300
        assert options.node_selector == {}
        assert options.tolerations == []
        assert options.annotations == {}
        assert options.labels == {}
        assert options.image_pull_secrets == []
        assert options.env_vars == {}
        assert options.env_from_secrets == {}
        assert options.secret_mounts == []
        assert options.service_account is None


# =============================================================================
# KubeOptions Tests - Custom Values
# =============================================================================


class TestKubeOptionsCustomValues:
    """Tests for KubeOptions with custom values."""

    def test_custom_image(self) -> None:
        """Test KubeOptions accepts custom image."""
        options = KubeOptions(image="myregistry.io/aiperf:v2.0.0")

        assert options.image == "myregistry.io/aiperf:v2.0.0"

    def test_custom_namespace(self) -> None:
        """Test KubeOptions accepts custom namespace."""
        options = KubeOptions(image="aiperf:latest", namespace="benchmark-ns")

        assert options.namespace == "benchmark-ns"

    def test_custom_workers(self) -> None:
        """Test KubeOptions accepts custom worker count."""
        options = KubeOptions(image="aiperf:latest", workers=100)

        assert options.workers == 100

    def test_custom_ttl_seconds(self) -> None:
        """Test KubeOptions accepts custom TTL."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=600)

        assert options.ttl_seconds == 600

    def test_ttl_seconds_none_disables_ttl(self) -> None:
        """Test KubeOptions with None TTL disables auto-cleanup."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=None)

        assert options.ttl_seconds is None

    def test_custom_node_selector(self) -> None:
        """Test KubeOptions accepts custom node selector."""
        options = KubeOptions(
            image="aiperf:latest",
            node_selector={"gpu": "true", "zone": "us-west-1"},
        )

        assert options.node_selector == {"gpu": "true", "zone": "us-west-1"}

    def test_custom_tolerations(self) -> None:
        """Test KubeOptions accepts custom tolerations."""
        tolerations = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            {
                "key": "dedicated",
                "operator": "Equal",
                "value": "ml",
                "effect": "NoSchedule",
            },
        ]
        options = KubeOptions(image="aiperf:latest", tolerations=tolerations)

        assert options.tolerations == tolerations
        assert len(options.tolerations) == 2

    def test_custom_annotations(self) -> None:
        """Test KubeOptions accepts custom annotations."""
        options = KubeOptions(
            image="aiperf:latest",
            annotations={
                "prometheus.io/scrape": "true",
                "prometheus.io/port": "9090",
            },
        )

        assert options.annotations == {
            "prometheus.io/scrape": "true",
            "prometheus.io/port": "9090",
        }

    def test_custom_labels(self) -> None:
        """Test KubeOptions accepts custom labels."""
        options = KubeOptions(
            image="aiperf:latest",
            labels={"team": "ml-platform", "environment": "benchmark"},
        )

        assert options.labels == {"team": "ml-platform", "environment": "benchmark"}

    def test_custom_image_pull_secrets(self) -> None:
        """Test KubeOptions accepts custom image pull secrets."""
        options = KubeOptions(
            image="aiperf:latest",
            image_pull_secrets=["registry-creds", "backup-creds"],
        )

        assert options.image_pull_secrets == ["registry-creds", "backup-creds"]

    def test_custom_env_vars(self) -> None:
        """Test KubeOptions accepts custom environment variables."""
        options = KubeOptions(
            image="aiperf:latest",
            env_vars={"DEBUG": "true", "LOG_LEVEL": "DEBUG"},
        )

        assert options.env_vars == {"DEBUG": "true", "LOG_LEVEL": "DEBUG"}

    def test_custom_env_from_secrets(self) -> None:
        """Test KubeOptions accepts custom env vars from secrets."""
        options = KubeOptions(
            image="aiperf:latest",
            env_from_secrets={
                "API_KEY": "api-secret/key",
                "TOKEN": "auth-secret/token",
            },
        )

        assert options.env_from_secrets == {
            "API_KEY": "api-secret/key",
            "TOKEN": "auth-secret/token",
        }

    def test_custom_secret_mounts(self) -> None:
        """Test KubeOptions accepts custom secret mounts."""
        options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="api-creds", mount_path="/secrets/api"),
                SecretMountConfig(
                    name="tls-certs", mount_path="/certs/tls", sub_path="cert.pem"
                ),
            ],
        )

        assert len(options.secret_mounts) == 2
        assert options.secret_mounts[0].name == "api-creds"
        assert options.secret_mounts[1].sub_path == "cert.pem"

    def test_custom_service_account(self) -> None:
        """Test KubeOptions accepts custom service account."""
        options = KubeOptions(image="aiperf:latest", service_account="aiperf-runner")

        assert options.service_account == "aiperf-runner"


# =============================================================================
# KubeOptions Tests - env_from_secrets Format Validation
# =============================================================================


class TestKubeOptionsEnvFromSecretsValidation:
    """Tests for env_from_secrets 'secret_name/key' format validation."""

    @pytest.mark.parametrize(
        "env_from_secrets",
        [
            param({"API_KEY": "my-secret/api-key"}, id="single-entry"),
            param(
                {"API_KEY": "my-secret/api-key", "TOKEN": "auth-secret/token"},
                id="multiple-entries",
            ),
            param({"VAR": "secret/key/with/extra/slashes"}, id="multiple-slashes-ok"),
            param({}, id="empty-dict"),
        ],
    )  # fmt: skip
    def test_valid_format_accepted(self, env_from_secrets: dict) -> None:
        options = KubeOptions(image="aiperf:latest", env_from_secrets=env_from_secrets)
        assert options.env_from_secrets == env_from_secrets

    @pytest.mark.parametrize(
        "env_from_secrets, bad_key",
        [
            param({"API_KEY": "mysecret"}, "API_KEY", id="no-slash"),
            param({"TOKEN": "authsecret"}, "TOKEN", id="no-slash-second"),
        ],
    )  # fmt: skip
    def test_missing_slash_raises(self, env_from_secrets: dict, bad_key: str) -> None:
        with pytest.raises(ValueError, match="secret_name/key"):
            KubeOptions(image="aiperf:latest", env_from_secrets=env_from_secrets)

    def test_mixed_valid_invalid_raises(self) -> None:
        """One bad entry in a multi-entry dict raises."""
        with pytest.raises(ValueError, match="secret_name/key"):
            KubeOptions(
                image="aiperf:latest",
                env_from_secrets={
                    "GOOD": "good-secret/key",
                    "BAD": "badsecret",
                },
            )


# =============================================================================
# KubeOptions Tests - Full Configuration
# =============================================================================


class TestKubeOptionsFullConfiguration:
    """Tests for KubeOptions with all fields populated."""

    def test_all_fields_populated(self) -> None:
        """Test KubeOptions with all fields set."""
        options = KubeOptions(
            image="myregistry.io/aiperf:v2.0.0",
            namespace="aiperf-benchmark",
            workers=50,
            ttl_seconds=1800,
            node_selector={"gpu": "a100", "zone": "us-central1-a"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
            annotations={"owner": "ml-team"},
            labels={"project": "llm-benchmark"},
            image_pull_secrets=["registry-creds"],
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            env_from_secrets={"HF_TOKEN": "hf-secret/token"},
            secret_mounts=[
                SecretMountConfig(name="api-keys", mount_path="/secrets/api")
            ],
            service_account="benchmark-sa",
        )

        assert options.image == "myregistry.io/aiperf:v2.0.0"
        assert options.namespace == "aiperf-benchmark"
        assert options.workers == 50
        assert options.ttl_seconds == 1800
        assert options.node_selector == {"gpu": "a100", "zone": "us-central1-a"}
        assert len(options.tolerations) == 1
        assert options.annotations == {"owner": "ml-team"}
        assert options.labels == {"project": "llm-benchmark"}
        assert options.image_pull_secrets == ["registry-creds"]
        assert options.env_vars == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        assert options.env_from_secrets == {"HF_TOKEN": "hf-secret/token"}
        assert len(options.secret_mounts) == 1
        assert options.service_account == "benchmark-sa"


# =============================================================================
# KubeOptions Tests - Serialization
# =============================================================================


class TestKubeOptionsSerialization:
    """Tests for KubeOptions serialization."""

    def test_model_dump_minimal(self) -> None:
        """Test model_dump with minimal configuration."""
        options = KubeOptions(image="aiperf:latest")
        data = options.model_dump()

        assert data["image"] == "aiperf:latest"
        assert data["namespace"] is None
        assert data["workers"] == 10  # Default workers per pod
        assert data["ttl_seconds"] == 300
        assert data["node_selector"] == {}
        assert data["tolerations"] == []

    def test_model_dump_with_secret_mounts(self) -> None:
        """Test model_dump includes nested secret mounts."""
        options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="test", mount_path="/test", sub_path="key")
            ],
        )
        data = options.model_dump()

        assert len(data["secret_mounts"]) == 1
        assert data["secret_mounts"][0]["name"] == "test"
        assert data["secret_mounts"][0]["mount_path"] == "/test"
        assert data["secret_mounts"][0]["sub_path"] == "key"

    def test_roundtrip_serialization(self) -> None:
        """Test KubeOptions can be serialized and deserialized."""
        original = KubeOptions(
            image="aiperf:v1",
            namespace="test-ns",
            workers=5,
            node_selector={"gpu": "true"},
        )

        data = original.model_dump()
        restored = KubeOptions(**data)

        assert restored.image == original.image
        assert restored.namespace == original.namespace
        assert restored.workers == original.workers
        assert restored.node_selector == original.node_selector


# =============================================================================
# KubeOptions.to_deployment_config() Tests
# =============================================================================


class TestToDeploymentConfig:
    """Tests for KubeOptions.to_deployment_config()."""

    def test_minimal_options(self) -> None:
        """Test minimal KubeOptions produces correct DeploymentConfig with defaults."""
        options = KubeOptions(image="aiperf:latest")
        result = options.to_deployment_config()

        assert isinstance(result, DeploymentConfig)
        assert result.image == "aiperf:latest"
        assert result.image_pull_policy is None
        assert result.ttl_seconds_after_finished == 300
        assert isinstance(result.pod_template, PodTemplateConfig)
        assert isinstance(result.scheduling, SchedulingConfig)
        assert result.pod_template.env == []
        assert result.scheduling.queue_name is None
        assert result.scheduling.priority_class is None

    def test_env_vars_converted_to_k8s_format(self) -> None:
        """Test dict env_vars are converted to K8s EnvVar list format."""
        options = KubeOptions(
            image="aiperf:latest",
            env_vars={"DEBUG": "true", "LOG_LEVEL": "info"},
        )
        result = options.to_deployment_config()

        assert {"name": "DEBUG", "value": "true"} in result.pod_template.env
        assert {"name": "LOG_LEVEL", "value": "info"} in result.pod_template.env
        assert len(result.pod_template.env) == 2

    def test_env_from_secrets_converted(self) -> None:
        """Test dict env_from_secrets are converted to secretKeyRef EnvVar entries."""
        options = KubeOptions(
            image="aiperf:latest",
            env_from_secrets={"API_KEY": "my-secret/api-key"},
        )
        result = options.to_deployment_config()

        expected = {
            "name": "API_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": "my-secret", "key": "api-key"},
            },
        }
        assert expected in result.pod_template.env

    def test_secret_mounts_converted(self) -> None:
        """Test SecretMountConfig entries are converted to K8s volumes/volumeMounts."""
        options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="creds", mount_path="/secrets/creds"),
                SecretMountConfig(
                    name="tls", mount_path="/certs/tls", sub_path="cert.pem"
                ),
            ],
        )
        result = options.to_deployment_config()

        assert {
            "name": "secret-creds",
            "secret": {"secretName": "creds"},
        } in result.pod_template.volumes
        assert {
            "name": "secret-tls",
            "secret": {"secretName": "tls"},
        } in result.pod_template.volumes

        mounts = result.pod_template.volume_mounts
        assert {
            "name": "secret-creds",
            "mountPath": "/secrets/creds",
            "readOnly": True,
        } in mounts
        assert {
            "name": "secret-tls",
            "mountPath": "/certs/tls",
            "readOnly": True,
            "subPath": "cert.pem",
        } in mounts

    def test_scheduling_fields(self) -> None:
        """Test queue_name and priority_class flow to SchedulingConfig."""
        options = KubeOptions(
            image="aiperf:latest",
            queue_name="team-queue",
            priority_class="high-priority",
        )
        result = options.to_deployment_config()

        assert result.scheduling.queue_name == "team-queue"
        assert result.scheduling.priority_class == "high-priority"

    def test_all_pod_template_fields(self) -> None:
        """Test node_selector, tolerations, annotations, labels, image_pull_secrets, service_account flow correctly."""
        toleration = {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        }
        options = KubeOptions(
            image="aiperf:latest",
            node_selector={"gpu": "true"},
            tolerations=[toleration],
            annotations={"owner": "ml-team"},
            labels={"env": "prod"},
            image_pull_secrets=["registry-creds"],
            service_account="bench-sa",
        )
        result = options.to_deployment_config()

        assert result.pod_template.node_selector == {"gpu": "true"}
        assert result.pod_template.tolerations == [toleration]
        assert result.pod_template.annotations == {"owner": "ml-team"}
        assert result.pod_template.labels == {"env": "prod"}
        assert result.pod_template.image_pull_secrets == ["registry-creds"]
        assert result.pod_template.service_account_name == "bench-sa"
