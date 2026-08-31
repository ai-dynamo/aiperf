# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.spec_converter module."""

import copy
import logging
from pathlib import Path
from typing import Any

import pydantic
import pytest
from pytest import param

from aiperf.common.enums import CommunicationType
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.kubernetes.spec_converter import (
    CONFIG_FIELDS,
    DEFAULT_CONNECTIONS_PER_WORKER,
    AIPerfJobSpecConverter,
    apply_worker_config,
    build_benchmark_run,
    workers_for_concurrency,
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
        assert config.benchmark.get_model_names() == ["test-model"]

    def test_sets_artifact_directory(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        assert config.benchmark.artifacts.dir == Path("/results")

    def test_full_config(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert config.benchmark.get_model_names() == ["gpt-4"]
        assert (
            "http://api.example.com/v1/chat/completions"
            in config.benchmark.endpoint.urls
        )
        assert any(p.name == "profiling" for p in config.benchmark.phases)
        assert (
            next(
                p for p in config.benchmark.phases if p.name == "profiling"
            ).concurrency
            == 500
        )
        assert (
            next(p for p in config.benchmark.phases if p.name == "profiling").requests
            == 1000
        )

    def test_kubernetes_service_run_type(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert config.benchmark.runtime.service_run_type == ServiceRunType.KUBERNETES
        assert config.benchmark.runtime.ui == UIType.SIMPLE

    def test_has_dual_bind_communication(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert config.benchmark.runtime.communication is not None
        assert config.benchmark.runtime.communication.type == CommunicationType.DUAL

    def test_api_settings(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        config = converter.to_aiperf_config()

        assert config.benchmark.runtime.api_host == "0.0.0.0"
        assert config.benchmark.runtime.api_port == 9090

    def test_dataset_api_url(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        config = converter.to_aiperf_config()

        assert "test-job" in config.benchmark.runtime.dataset_api_base_url
        assert "my-namespace" in config.benchmark.runtime.dataset_api_base_url
        assert "/api/dataset" in config.benchmark.runtime.dataset_api_base_url

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

        assert "unique-abc123" in config.benchmark.runtime.dataset_api_base_url
        assert "original-name" not in config.benchmark.runtime.dataset_api_base_url
        assert "aiperf-unique-abc123" in config.benchmark.runtime.dataset_api_base_url

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
        assert deploy.pod_template.image_pull_secrets == [
            {"name": "my-registry-secret"}
        ]
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
                    "datasets": [{"name": "main", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "requests": 10,
                            "concurrency": 1,
                        }
                    ],
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
            # ceil(1001/100) = 11, which fills neither one pod nor a whole
            # number of them, so a derived total rounds up to the next.
            (1001, 100, 20),
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
                "datasets": [{"name": "main", "type": "synthetic"}],
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
                "datasets": [{"name": "main", "type": "synthetic"}],
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
                "datasets": [{"name": "main", "type": "synthetic"}],
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
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "warmup",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 10,
                    },
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 100,
                        "concurrency": 200,
                    },
                ],
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()
        # max(10, 200) / 100 = 2
        assert workers == 2

    def test_calculate_workers_warmup_profiling_shorthand_derives_from_concurrency(
        self,
    ) -> None:
        """[F1] Named phase shorthands must be normalized before concurrency is read.

        ``warmup:``/``profiling:`` are top-level siblings of ``phases:`` that the
        AIPerfConfig before-validator folds into a ``phases`` list. The raw spec
        dict therefore has no ``phases`` key at all. The old implementation read
        that raw dict, saw an empty phase list, fell back to concurrency=1 and
        silently provisioned a single worker for any load. Reading the validated
        config instead makes the shorthand resolve.
        """
        spec = {
            "connectionsPerWorker": 100,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "warmup": {"type": "concurrency", "requests": 10, "concurrency": 10},
                "profiling": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1000,
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")

        # The shorthand keys mean the raw dict genuinely has no "phases" entry,
        # which is exactly what made the old raw-dict read collapse to 1.
        assert converter._get_config_dict().get("phases") is None

        # max(10, 1000) / 100 = 10, not the pre-fix fallback of 1.
        assert converter.calculate_workers() == 10


class TestWorkersForConcurrency:
    """The divisor neutralizer shared by every worker-count call site.

    ``connectionsPerWorker`` is bounded ``ge=1`` by ``DeploymentConfig`` and
    ``minimum: 1`` by both CRDs, so the operator reconcile path only ever
    divides by a validated value. The unvalidated entry points (``aiperf kube
    validate``, ``POST /api/v1/validate``, ``kube profile --dry-run``,
    ``kube generate --operator``) divide raw spec dicts and used to crash.
    """

    @pytest.mark.parametrize(
        ("concurrency", "divisor", "expected"),
        [
            param(16, 1, 16, id="one-connection-per-worker"),
            param(16, 100, 1, id="divisor-above-concurrency"),
            param(1000, 100, 10, id="even-split"),
            param(250, 100, 3, id="ceil-rounds-up"),
        ],
    )  # fmt: skip
    def test_valid_divisor_arithmetic_unchanged(
        self, concurrency: int, divisor: int, expected: int
    ) -> None:
        assert workers_for_concurrency(concurrency, divisor) == expected

    @pytest.mark.parametrize(
        "divisor",
        [
            param(0, id="int-zero"),
            param(0.0, id="float-zero"),
            param(-0.0, id="negative-float-zero"),
            param(False, id="yaml-false"),
            param(1.0e-320, id="denormal-1e-320"),
            param(5e-324, id="smallest-subnormal"),
            param(float("nan"), id="nan"),
            param(-1, id="negative"),
            param(0.5, id="fractional-below-one"),
        ],
    )  # fmt: skip
    def test_divisor_below_one_is_neutralized(self, divisor: object) -> None:
        """Zero-likes and subnormals used to raise ZeroDivisionError/OverflowError."""
        assert workers_for_concurrency(16, divisor) == 16

    @pytest.mark.parametrize(
        "divisor",
        [
            param("abc", id="unparsable-string"),
            param(None, id="null"),
            param([], id="list"),
        ],
    )  # fmt: skip
    def test_non_numeric_divisor_still_raises_type_error(self, divisor: object) -> None:
        """Callers report these as config errors, so they must pass through."""
        with pytest.raises(TypeError):
            workers_for_concurrency(16, divisor)

    def test_infinite_divisor_still_clamps_to_one(self) -> None:
        assert workers_for_concurrency(16, float("inf")) == 1

    @pytest.mark.parametrize(
        "divisor",
        [
            param(0, id="int-zero"),
            param(False, id="yaml-false"),
            param(1.0e-320, id="denormal-1e-320"),
        ],
    )  # fmt: skip
    def test_calculate_workers_survives_hostile_raw_spec(self, divisor: object) -> None:
        spec = {
            "connectionsPerWorker": divisor,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 16,
                    }
                ],
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")

        # The divisor falls back to 1, so every unit of concurrency wants its
        # own worker: 16, rounded up to two whole 10-worker pods.
        assert converter.calculate_workers() == 20

    def test_deployment_config_divisor_wins_over_hostile_raw_spec(self) -> None:
        """The operator reconcile path passes a ``ge=1``-validated DeploymentConfig."""
        spec = {
            "connectionsPerWorker": 0,
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 16,
                    }
                ],
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")

        assert converter.calculate_workers(DeploymentConfig()) == 1

    @pytest.mark.parametrize(
        "divisor",
        [
            param(0, id="int-zero"),
            param(0.0, id="float-zero"),
            param(False, id="yaml-false"),
            param(1.0e-320, id="denormal-1e-320"),
        ],
    )  # fmt: skip
    def test_deployment_config_rejects_hostile_divisor_before_division(
        self, divisor: object
    ) -> None:
        """kopf calls ``to_deployment_config()`` first, so it never reaches the raw path."""
        with pytest.raises(pydantic.ValidationError):
            DeploymentConfig.model_validate({"connectionsPerWorker": divisor})


class TestApplyWorkerConfig:
    """Tests for apply_worker_config function."""

    def test_single_pod(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        num_pods = apply_worker_config(config, 5)
        assert num_pods == 1
        assert config.benchmark.runtime.workers_per_pod == 5
        assert config.benchmark.runtime.workers == 5

    def test_multiple_pods(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        num_pods = apply_worker_config(config, 50)
        assert num_pods == 5
        assert config.benchmark.runtime.workers_per_pod == 10
        assert config.benchmark.runtime.workers == 50

    def test_non_divisible_total_is_rejected(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """A total that cannot fill uniform pods must fail, not be reshaped.

        11 workers at the default 10 per pod has no JobSet representation: a
        replicated job cannot carry a partial final replica.
        """
        config = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        ).to_aiperf_config()

        with pytest.raises(ValueError, match="not a multiple of"):
            apply_worker_config(config, 11)

    def test_non_divisible_total_error_names_the_usable_totals(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """The rejection has to tell the user what to type instead.

        A bare "invalid" leaves them guessing at a number that is only derivable
        from runtime.workersPerPod, which they may never have set explicitly.
        """
        config = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        ).to_aiperf_config()

        with pytest.raises(ValueError) as excinfo:
            apply_worker_config(config, 25)

        message = str(excinfo.value)
        assert "20" in message
        assert "30" in message
        assert "workers_per_pod" in message

    def test_divisible_total_does_not_warn(
        self, minimal_aiperfjob_spec: dict[str, Any], caplog
    ) -> None:
        config = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        ).to_aiperf_config()

        with caplog.at_level(
            logging.WARNING, logger="aiperf.kubernetes.spec_converter"
        ):
            num_pods = apply_worker_config(config, 50)

        assert num_pods == 5
        assert "single pod" not in caplog.text

    def test_record_processors_set(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()
        apply_worker_config(config, 10)
        assert config.benchmark.runtime.record_processors_per_pod == 10
        assert config.benchmark.runtime.record_processors == 10

    def test_authored_record_processor_total_is_preserved(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        spec = copy.deepcopy(minimal_aiperfjob_spec)
        spec["benchmark"]["runtime"] = {
            "workersPerPod": 2,
            "recordProcessors": 4,
        }
        config = AIPerfJobSpecConverter(spec, "test-job", "default").to_aiperf_config()

        num_pods = apply_worker_config(config, 4)

        assert num_pods == 2
        assert config.benchmark.runtime.record_processors_per_pod == 2
        assert config.benchmark.runtime.record_processors == 4

    def test_non_divisible_record_processor_total_is_rejected(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        spec = copy.deepcopy(minimal_aiperfjob_spec)
        spec["benchmark"]["runtime"] = {
            "workersPerPod": 2,
            "recordProcessors": 3,
        }
        config = AIPerfJobSpecConverter(spec, "test-job", "default").to_aiperf_config()

        with pytest.raises(ValueError, match="record_processors.*divisible"):
            apply_worker_config(config, 4)


# =============================================================================
# Env var and Jinja2 expansion in operator path
# =============================================================================


def _expansion_spec(benchmark_overrides: dict[str, Any]) -> dict[str, Any]:
    """Minimal spec with benchmark overrides for expansion tests."""
    base: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": [{"name": "main", "type": "synthetic", "entries": 10}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 8,
                "requests": 100,
            }
        ],
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
        assert config.benchmark.endpoint.urls == [
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
        assert config.benchmark.endpoint.urls == [
            "http://fallback-host:9000/v1/chat/completions"
        ]

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
        assert config.benchmark.endpoint.api_key == "sk-test-12345"

    def test_env_var_in_calculate_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_CONCURRENCY", "200")
        spec = {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": "${WORKER_CONCURRENCY}",
                        "requests": 100,
                    }
                ],
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
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": "{{ target_concurrency }}",
                        "requests": 100,
                    }
                ],
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert (
            next(
                p for p in config.benchmark.phases if p.name == "profiling"
            ).concurrency
            == 16
        )

    def test_derived_value_from_other_field(self) -> None:
        spec = _expansion_spec(
            {
                "variables": {"base": 8},
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": "{{ base }}",
                        "requests": "{{ base * 10 }}",
                    }
                ],
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        config = converter.to_aiperf_config()
        assert (
            next(
                p for p in config.benchmark.phases if p.name == "profiling"
            ).concurrency
            == 8
        )
        assert (
            next(p for p in config.benchmark.phases if p.name == "profiling").requests
            == 80
        )

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
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": "{{ c }}",
                        "requests": 100,
                    }
                ],
            }
        }
        converter = AIPerfJobSpecConverter(spec, "job", "default")
        # 50 / 100 → 1 (ceil)
        assert converter.calculate_workers() == 1


class TestBuildBenchmarkRunOrchestrationGuard:
    """A single-run builder must fail closed on orchestration envelopes."""

    def _base_config(self) -> dict[str, Any]:
        return {
            "models": ["test-model"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": {
                "type": "concurrency",
                "requests": 10,
                "concurrency": 1,
            },
        }

    def test_sweep_raises_without_mutating_input(self) -> None:
        config = self._base_config()
        config["sweep"] = {"parameter": "concurrency", "values": [1, 2, 4]}
        with pytest.raises(ValueError, match="AIPerfSweep"):
            build_benchmark_run(config, "run-1", "default")
        assert "sweep" in config

    def test_multi_run_raises_without_mutating_input(self) -> None:
        config = self._base_config()
        config["multi_run"] = {"trials": 3}
        with pytest.raises(ValueError, match="AIPerfSweep"):
            build_benchmark_run(config, "run-1", "default")
        assert "multi_run" in config

    def test_convergence_raises(self) -> None:
        config = self._base_config()
        config["multi_run"] = {
            "num_runs": 1,
            "convergence": {"metric": "ttft", "min_runs": 1},
        }
        with pytest.raises(ValueError, match="AIPerfSweep"):
            build_benchmark_run(config, "run-1", "default")

    def test_default_materialized_multi_run_is_single_run(self) -> None:
        config = self._base_config()
        config["multi_run"] = {
            "num_runs": 1,
            "cooldown_seconds": 0.0,
            "confidence_level": 0.95,
        }

        run = build_benchmark_run(config, "run-1", "default")

        assert run.trial == 0

    def test_plot_envelope_is_transported_on_benchmark_run(self) -> None:
        from aiperf.config.plot import PlotEnvelopeConfig

        config = {
            "benchmark": {
                **self._base_config(),
                "artifacts": {"auto_plot": True, "plot_required": True},
            },
            "plot": {"visualization": {}},
        }

        run = build_benchmark_run(config, "run-1", "default")

        assert isinstance(run.plot, PlotEnvelopeConfig)
        assert run.cfg.artifacts.auto_plot is True
        assert run.cfg.artifacts.plot_required is True
