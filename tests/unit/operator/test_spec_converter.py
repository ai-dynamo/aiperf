# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.spec_converter module."""

from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import CommunicationType
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.operator.spec_converter import (
    CONFIG_FIELDS,
    DEFAULT_CONNECTIONS_PER_WORKER,
    AIPerfJobSpecConverter,
    apply_worker_config,
)
from aiperf.plugin.enums import ServiceRunType, UIType


class TestAIPerfJobSpecConverterInit:
    """Tests for AIPerfJobSpecConverter initialization."""

    def test_init_stores_parameters(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns"
        )
        assert converter.spec == minimal_aiperfjob_spec
        assert converter.name == "test-job"
        assert converter.namespace == "test-ns"

    def test_job_id_defaults_to_name(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns"
        )
        assert converter.job_id == "test-job"

    def test_job_id_can_be_overridden(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns", job_id="custom-id-abc123"
        )
        assert converter.job_id == "custom-id-abc123"
        assert converter.name == "test-job"

    def test_default_connections_per_worker(self) -> None:
        assert DEFAULT_CONNECTIONS_PER_WORKER == 100

    def test_config_fields_contains_expected(self) -> None:
        expected = {"models", "endpoint", "datasets", "phases", "artifacts", "runtime"}
        assert expected.issubset(CONFIG_FIELDS)


class TestToAIPerfConfig:
    """Tests for to_aiperf_config method."""

    def test_returns_aiperf_config(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        assert isinstance(config, AIPerfConfig)

    def test_minimal_config_model_names(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        assert config.get_model_names() == ["test-model"]

    def test_sets_artifact_directory(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        assert config.artifacts.dir == Path("/results")

    def test_full_config(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert config.get_model_names() == ["gpt-4"]
        assert "http://api.example.com/v1/chat/completions" in config.endpoint.urls
        assert "profiling" in config.phases
        assert config.phases["profiling"].concurrency == 500
        assert config.phases["profiling"].requests == 1000

    def test_kubernetes_service_run_type(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert config.runtime.service_run_type == ServiceRunType.KUBERNETES
        assert config.runtime.ui == UIType.SIMPLE

    def test_has_dual_bind_communication(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert config.runtime.communication is not None
        assert config.runtime.communication.type == CommunicationType.DUAL

    def test_api_settings(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        config = converter.to_aiperf_config()

        assert config.runtime.api_host == "0.0.0.0"
        assert config.runtime.api_port == 9090

    def test_dataset_api_url(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        config = converter.to_aiperf_config()

        assert "test-job" in config.runtime.dataset_api_base_url
        assert "my-namespace" in config.runtime.dataset_api_base_url
        assert "/api/dataset" in config.runtime.dataset_api_base_url

    def test_dataset_api_url_uses_job_id(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec,
            "original-name",
            "my-namespace",
            job_id="unique-abc123",
        )
        config = converter.to_aiperf_config()

        assert "unique-abc123" in config.runtime.dataset_api_base_url
        assert "original-name" not in config.runtime.dataset_api_base_url
        assert "aiperf-unique-abc123" in config.runtime.dataset_api_base_url

    def test_deployment_fields_not_in_config(
        self, full_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Deployment fields (image, podTemplate, etc.) should not leak into AIPerfConfig."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()
        config_dict = config.model_dump()
        assert "image" not in config_dict
        assert "podTemplate" not in config_dict
        assert "connectionsPerWorker" not in config_dict


class TestToDeploymentConfig:
    """Tests for to_deployment_config method."""

    def test_empty_pod_template(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        deploy = converter.to_deployment_config()

        assert isinstance(deploy, DeploymentConfig)
        assert deploy.pod_template.node_selector == {}
        assert deploy.pod_template.tolerations == []
        assert deploy.pod_template.annotations == {}
        assert deploy.pod_template.labels == {}
        assert deploy.pod_template.image_pull_secrets == []
        assert deploy.pod_template.env == []
        assert deploy.pod_template.volumes == []
        assert deploy.pod_template.service_account_name is None

    def test_full_pod_customization(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        deploy = converter.to_deployment_config()

        assert deploy.pod_template.node_selector == {"gpu": "true"}
        assert len(deploy.pod_template.tolerations) == 1
        assert deploy.pod_template.tolerations[0]["key"] == "nvidia.com/gpu"
        assert deploy.pod_template.annotations == {"prometheus.io/scrape": "true"}
        assert deploy.pod_template.labels == {"team": "ml-platform"}
        assert deploy.pod_template.image_pull_secrets == ["my-registry-secret"]
        assert deploy.pod_template.service_account_name == "aiperf-sa"

    def test_resource_mode_propagated(self) -> None:
        converter = AIPerfJobSpecConverter(
            {
                "image": "aiperf:latest",
                "resourceMode": "none",
                "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
            },
            "test-job",
            "default",
        )
        deploy = converter.to_deployment_config()
        assert deploy.resource_mode == "none"

    def test_env_vars_direct(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        deploy = converter.to_deployment_config()
        assert {"name": "DEBUG", "value": "true"} in deploy.pod_template.env

    def test_env_vars_from_secrets(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        deploy = converter.to_deployment_config()
        secret_env = {
            "name": "API_KEY",
            "valueFrom": {"secretKeyRef": {"name": "api-secrets", "key": "api-key"}},
        }
        assert secret_env in deploy.pod_template.env

    def test_secret_mounts(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        deploy = converter.to_deployment_config()
        assert any(
            v.get("secret", {}).get("secretName") == "my-creds"
            for v in deploy.pod_template.volumes
        )

    def test_connections_per_worker(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        deploy = converter.to_deployment_config()
        assert deploy.connections_per_worker == 100

    def test_default_image(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        deploy = converter.to_deployment_config()
        assert deploy.image == "nvcr.io/nvidia/aiperf:latest"

    def test_converts_keep_failed_pods_flag(self) -> None:
        converter = AIPerfJobSpecConverter(
            {
                "keepFailedPods": True,
                "benchmark": {
                    "models": ["mock"],
                    "endpoint": {"urls": ["http://example/v1/chat/completions"]},
                    "datasets": {"main": {"type": "synthetic"}},
                    "phases": {
                        "profiling": {
                            "type": "concurrency",
                            "requests": 10,
                            "concurrency": 1,
                        }
                    },
                },
            },
            "test-job",
            "default",
        )

        deploy = converter.to_deployment_config()

        assert deploy.keep_failed_pods is True


class TestCalculateWorkers:
    """Tests for calculate_workers method."""

    def test_default_concurrency(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        workers = converter.calculate_workers()
        assert workers == 1

    def test_high_concurrency(
        self, aiperfjob_spec_high_concurrency: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            aiperfjob_spec_high_concurrency, "test-job", "default"
        )
        workers = converter.calculate_workers()
        # 1000 concurrency / 100 connections per worker = 10 workers
        assert workers == 10

    @pytest.mark.parametrize(
        "concurrency,connections_per_worker,expected",
        [
            (1, 500, 1),
            (500, 500, 1),
            (501, 500, 2),
            (1000, 100, 10),
            (1001, 100, 11),
            (50, 100, 1),
        ],
    )  # fmt: skip
    def test_worker_calculation_formula(
        self, concurrency: int, connections_per_worker: int, expected: int
    ) -> None:
        spec = {
            "connectionsPerWorker": connections_per_worker,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": concurrency,
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()
        assert workers == expected

    def test_minimum_one_worker(self) -> None:
        spec = {
            "connectionsPerWorker": 1000,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()
        assert workers >= 1

    def test_calculate_with_deployment_config(
        self, full_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        dc = converter.to_deployment_config()
        workers = converter.calculate_workers(dc)
        # 500 concurrency / 100 cpw = 5
        assert workers == 5

    def test_runtime_workers_override_takes_precedence(self) -> None:
        spec = {
            "connectionsPerWorker": 250,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "runtime": {"workers": 1250, "workers_per_pod": 25},
                "phases": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 250000,
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()
        assert workers == 1250

    def test_multi_phase_max_concurrency(self) -> None:
        spec = {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "warmup": {
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 10,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 100,
                        "concurrency": 200,
                    },
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()
        # max(10, 200) / 100 = 2
        assert workers == 2


class TestApplyWorkerConfig:
    """Tests for apply_worker_config function."""

    def test_single_pod(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        num_pods = apply_worker_config(config, 5)
        assert num_pods == 1
        assert config.runtime.workers_per_pod == 5
        assert config.runtime.workers == 5

    def test_multiple_pods(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        num_pods = apply_worker_config(config, 50)
        assert num_pods >= 1
        assert config.runtime.workers == num_pods * config.runtime.workers_per_pod

    def test_record_processors_set(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        apply_worker_config(config, 10)
        assert config.runtime.record_processors_per_pod == 10
        assert config.runtime.record_processors == 10


# =============================================================================
# Env var and Jinja2 expansion in operator path
# =============================================================================


def _expansion_spec(benchmark_overrides: dict[str, Any]) -> dict[str, Any]:
    """Minimal spec with benchmark overrides for expansion tests."""
    base: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": {"main": {"type": "synthetic", "entries": 10}},
        "phases": {
            "profiling": {"type": "concurrency", "concurrency": 8, "requests": 100}
        },
    }
    base.update(benchmark_overrides)
    return {"benchmark": base}


class TestEnvVarExpansion:
    """${VAR} in CRD spec is expanded from operator pod's os.environ."""

    def test_env_var_in_url_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDPOINT_HOST", "my-inference-server:8080")
        spec = _expansion_spec(
            {"endpoint": {"urls": ["http://${ENDPOINT_HOST}/v1/chat/completions"]}}
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert config.endpoint.urls == [
            "http://my-inference-server:8080/v1/chat/completions"
        ]

    def test_env_var_with_default_used_when_unset(self) -> None:
        spec = _expansion_spec(
            {
                "endpoint": {
                    "urls": [
                        "http://${MISSING_HOST:fallback-host:9000}/v1/chat/completions"
                    ]
                }
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert config.endpoint.urls == ["http://fallback-host:9000/v1/chat/completions"]

    def test_missing_env_var_no_default_raises(self) -> None:
        from aiperf.config.loader import MissingEnvironmentVariableError

        spec = _expansion_spec(
            {
                "endpoint": {
                    "urls": ["http://${DEFINITELY_UNSET_XYZ_ABC}/v1/chat/completions"]
                }
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        with pytest.raises(
            MissingEnvironmentVariableError, match="DEFINITELY_UNSET_XYZ_ABC"
        ):
            converter.to_aiperf_config()

    def test_env_var_in_api_key_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "sk-test-12345")
        spec = _expansion_spec(
            {
                "endpoint": {
                    "urls": ["http://localhost:8000"],
                    "api_key": "${MY_API_KEY}",
                }
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert config.endpoint.api_key == "sk-test-12345"

    def test_env_var_in_calculate_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_CONCURRENCY", "200")
        spec = {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": "${WORKER_CONCURRENCY}",
                        "requests": 100,
                    }
                },
            }
        }
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        # 200 / 100 = 2 workers
        assert converter.calculate_workers() == 2


class TestJinja2Expansion:
    """{{expr}} in CRD spec is expanded using the dict as context."""

    def test_variables_section_used_in_template(self) -> None:
        spec = _expansion_spec(
            {
                "variables": {"target_concurrency": 16},
                "phases": {
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": "{{ target_concurrency }}",
                        "requests": 100,
                    }
                },
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert config.phases["profiling"].concurrency == 16

    def test_derived_value_from_other_field(self) -> None:
        spec = _expansion_spec(
            {
                "variables": {"base": 8},
                "phases": {
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": "{{ base }}",
                        "requests": "{{ base * 10 }}",
                    }
                },
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert config.phases["profiling"].concurrency == 8
        assert config.phases["profiling"].requests == 80

    def test_variables_key_not_passed_to_pydantic(self) -> None:
        """variables section must be stripped before model_validate (extra='forbid')."""
        spec = _expansion_spec({"variables": {"x": 1}})
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        # Should not raise ValidationError about unknown 'variables' field
        config = converter.to_aiperf_config()
        assert config is not None

    def test_jinja2_in_calculate_workers(self) -> None:
        spec = {
            "benchmark": {
                "variables": {"c": 50},
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": "{{ c }}",
                        "requests": 100,
                    }
                },
            }
        }
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        # 50 / 100 → 1 (ceil)
        assert converter.calculate_workers() == 1
