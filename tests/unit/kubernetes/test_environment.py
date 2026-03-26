# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.environment module."""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.environment import (
    CONTROLLER_RESOURCE_KEYS,
    K8sEnvironment,
    _HealthProbeSettings,
    _JobSetSettings,
    _PortSettings,
    _resource_settings,
    _ZMQSettings,
)


class TestResourceSettingsToK8sResources:
    """Tests for _ResourceSettings.to_k8s_resources method."""

    @pytest.mark.parametrize(
        "setting_attr,cpu,memory",
        [
            param("SYSTEM_CONTROLLER", "500m", "1Gi", id="system_controller"),
            param("WORKER_MANAGER", "500m", "1Gi", id="worker_manager"),
            param("TIMING_MANAGER", "1000m", "2Gi", id="timing_manager"),
            param("DATASET_MANAGER", "1000m", "2Gi", id="dataset_manager"),
            param("RECORDS_MANAGER", "1000m", "2Gi", id="records_manager"),
            param("API", "1000m", "8Gi", id="api"),
            param("GPU_TELEMETRY_MANAGER", "250m", "512Mi", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER", "250m", "512Mi", id="server_metrics"),
            param("RESULTS_SIDECAR", "250m", "512Mi", id="results_sidecar"),
            param("WORKER_POD", "4000m", "12Gi", id="worker_pod"),
        ],
    )  # fmt: skip
    def test_to_k8s_resources_returns_correct_structure(
        self,
        setting_attr: str,
        cpu: str,
        memory: str,
    ) -> None:
        """Test to_k8s_resources returns correctly structured dict with Guaranteed QoS."""
        setting = getattr(K8sEnvironment, setting_attr)
        resources = setting.to_k8s_resources()

        assert resources == {
            "requests": {"cpu": cpu, "memory": memory},
            "limits": {"cpu": cpu, "memory": memory},
        }

    def test_to_k8s_resources_burstable_omits_limits(self) -> None:
        setting = _resource_settings("TEST_BURST_", "500m", "1Gi")
        resources = setting.to_k8s_resources(burstable=True)

        assert resources == {"requests": {"cpu": "500m", "memory": "1Gi"}}
        assert "limits" not in resources

    def test_to_k8s_resources_guaranteed_includes_limits(self) -> None:
        setting = _resource_settings("TEST_GUAR_", "500m", "1Gi")
        resources = setting.to_k8s_resources(burstable=False)

        assert resources["requests"] == resources["limits"]


class TestK8sEnvironmentControllerContainers:
    """Tests for controller-side per-container resource settings."""

    @pytest.mark.parametrize(
        "setting_attr,cpu,memory",
        [
            param("SYSTEM_CONTROLLER", "500m", "1Gi", id="system_controller"),
            param("WORKER_MANAGER", "500m", "1Gi", id="worker_manager"),
            param("TIMING_MANAGER", "1000m", "2Gi", id="timing_manager"),
            param("DATASET_MANAGER", "1000m", "2Gi", id="dataset_manager"),
            param("RECORDS_MANAGER", "1000m", "2Gi", id="records_manager"),
            param("API", "1000m", "8Gi", id="api"),
            param("GPU_TELEMETRY_MANAGER", "250m", "512Mi", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER", "250m", "512Mi", id="server_metrics"),
        ],
    )  # fmt: skip
    def test_controller_container_default_values(
        self, setting_attr: str, cpu: str, memory: str
    ) -> None:
        setting = getattr(K8sEnvironment, setting_attr)
        assert cpu == setting.CPU
        assert memory == setting.MEMORY

    @pytest.mark.parametrize("setting_attr", CONTROLLER_RESOURCE_KEYS)
    def test_controller_container_guaranteed_qos(self, setting_attr: str) -> None:
        resources = getattr(K8sEnvironment, setting_attr).to_k8s_resources()
        assert resources["requests"] == resources["limits"]


class TestK8sEnvironmentWorkerPod:
    """Tests for K8sEnvironment.WORKER_POD settings."""

    def test_worker_pod_default_values(self) -> None:
        pod = K8sEnvironment.WORKER_POD
        assert pod.CPU == "4000m"
        assert pod.MEMORY == "12Gi"

    def test_worker_pod_guaranteed_qos(self) -> None:
        resources = K8sEnvironment.WORKER_POD.to_k8s_resources()
        assert resources["requests"] == resources["limits"]

    def test_worker_pod_to_k8s_resources(self) -> None:
        resources = K8sEnvironment.WORKER_POD.to_k8s_resources()
        assert resources["requests"]["cpu"] == "4000m"
        assert resources["limits"]["cpu"] == "4000m"
        assert resources["requests"]["memory"] == "12Gi"
        assert resources["limits"]["memory"] == "12Gi"

    def test_k8s_record_processor_scale_factor_default(self) -> None:
        assert K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR == 1


class TestPodResourceEnvOverrides:
    """Tests for pod-level resource env var overrides."""

    def test_system_controller_cpu_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_K8S_SYSTEM_CONTROLLER_CPU", "400m")
        settings = _resource_settings("SYSTEM_CONTROLLER_", "250m", "256Mi")
        assert settings.CPU == "400m"
        assert settings.MEMORY == "256Mi"

    def test_dataset_manager_memory_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_K8S_DATASET_MANAGER_MEMORY", "768Mi")
        settings = _resource_settings("DATASET_MANAGER_", "500m", "512Mi")
        assert settings.MEMORY == "768Mi"
        assert settings.CPU == "500m"

    def test_worker_pod_cpu_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIPERF_K8S_WORKER_POD_CPU", "5000m")
        settings = _resource_settings("WORKER_POD_", "3350m", "6144Mi")
        assert settings.CPU == "5000m"

    def test_worker_pod_memory_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIPERF_K8S_WORKER_POD_MEMORY", "8192Mi")
        settings = _resource_settings("WORKER_POD_", "3350m", "6144Mi")
        assert settings.MEMORY == "8192Mi"

    def test_k8s_record_processor_scale_factor_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_K8S_RECORD_PROCESSOR_SCALE_FACTOR", "2")
        from aiperf.kubernetes.environment import _K8sEnvironment

        settings = _K8sEnvironment()
        assert settings.RECORD_PROCESSOR_SCALE_FACTOR == 2

    def test_override_applies_to_both_requests_and_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overriding CPU/MEMORY sets both request and limit (Guaranteed QoS)."""
        monkeypatch.setenv("AIPERF_K8S_SYSTEM_CONTROLLER_CPU", "400m")
        monkeypatch.setenv("AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY", "384Mi")
        settings = _resource_settings("SYSTEM_CONTROLLER_", "250m", "256Mi")
        resources = settings.to_k8s_resources()
        assert resources["requests"]["cpu"] == "400m"
        assert resources["limits"]["cpu"] == "400m"
        assert resources["requests"]["memory"] == "384Mi"
        assert resources["limits"]["memory"] == "384Mi"

    def test_per_service_settings_exposed(self) -> None:
        """Controller services should expose per-container resource settings."""
        for name in [
            "SYSTEM_CONTROLLER",
            "WORKER_MANAGER",
            "TIMING_MANAGER",
            "DATASET_MANAGER",
            "RECORDS_MANAGER",
            "API",
            "GPU_TELEMETRY_MANAGER",
            "SERVER_METRICS_MANAGER",
        ]:
            assert hasattr(K8sEnvironment, name), (
                f"K8sEnvironment.{name} should exist for per-container controller resources"
            )


class TestK8sEnvironmentHealth:
    """Tests for K8sEnvironment.HEALTH settings."""

    def test_health_default_values(self) -> None:
        """Test health probe has expected default values."""
        health = K8sEnvironment.HEALTH
        assert health.INITIAL_DELAY_SECONDS == 5
        assert health.PERIOD_SECONDS == 10
        assert health.TIMEOUT_SECONDS == 5
        assert health.FAILURE_THRESHOLD == 10
        assert health.SUCCESS_THRESHOLD == 1

    def test_health_values_within_bounds(self) -> None:
        """Test health probe values are within valid bounds."""
        health = K8sEnvironment.HEALTH
        assert 0 <= health.INITIAL_DELAY_SECONDS <= 300
        assert 1 <= health.PERIOD_SECONDS <= 300
        assert 1 <= health.TIMEOUT_SECONDS <= 60
        assert 1 <= health.FAILURE_THRESHOLD <= 20
        assert 1 <= health.SUCCESS_THRESHOLD <= 10


class TestK8sEnvironmentPorts:
    """Tests for K8sEnvironment.PORTS settings."""

    def test_ports_default_values(self) -> None:
        """Test ports have expected default values."""
        ports = K8sEnvironment.PORTS
        assert ports.SYSTEM_CONTROLLER_HEALTH == 8080
        assert ports.WORKER_MANAGER_HEALTH == 8081
        assert ports.TIMING_MANAGER_HEALTH == 8082
        assert ports.DATASET_MANAGER_HEALTH == 8083
        assert ports.RECORDS_MANAGER_HEALTH == 8084
        assert ports.API_SERVICE == 9090
        assert ports.RESULTS_SIDECAR == 9091
        assert ports.API_SERVICE_HEALTH == 8085
        assert ports.WORKER_HEALTH == 8080
        assert ports.RECORD_PROCESSOR_HEALTH == 8081

    def test_ports_telemetry_defaults(self) -> None:
        """Test telemetry service port defaults."""
        ports = K8sEnvironment.PORTS
        assert ports.GPU_TELEMETRY_MANAGER_HEALTH == 8086
        assert ports.SERVER_METRICS_MANAGER_HEALTH == 8087

    def test_ports_unique_on_controller(self) -> None:
        """Test that controller pod ports are unique."""
        ports = K8sEnvironment.PORTS
        controller_ports = [
            ports.SYSTEM_CONTROLLER_HEALTH,
            ports.WORKER_MANAGER_HEALTH,
            ports.TIMING_MANAGER_HEALTH,
            ports.DATASET_MANAGER_HEALTH,
            ports.RECORDS_MANAGER_HEALTH,
            ports.API_SERVICE,
            ports.RESULTS_SIDECAR,
            ports.API_SERVICE_HEALTH,
            ports.GPU_TELEMETRY_MANAGER_HEALTH,
            ports.SERVER_METRICS_MANAGER_HEALTH,
        ]
        assert len(controller_ports) == len(set(controller_ports))

    @pytest.mark.parametrize(
        "port_attr",
        [
            param("SYSTEM_CONTROLLER_HEALTH", id="system_controller"),
            param("WORKER_MANAGER_HEALTH", id="worker_manager"),
            param("TIMING_MANAGER_HEALTH", id="timing_manager"),
            param("DATASET_MANAGER_HEALTH", id="dataset_manager"),
            param("RECORDS_MANAGER_HEALTH", id="records_manager"),
            param("API_SERVICE", id="api_service"),
            param("RESULTS_SIDECAR", id="results_sidecar"),
            param("API_SERVICE_HEALTH", id="api_service_health"),
            param("GPU_TELEMETRY_MANAGER_HEALTH", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER_HEALTH", id="server_metrics"),
            param("WORKER_HEALTH", id="worker_health"),
            param("RECORD_PROCESSOR_HEALTH", id="record_processor"),
        ],
    )  # fmt: skip
    def test_ports_within_valid_range(self, port_attr: str) -> None:
        """Test all port values are within valid range (1-65535)."""
        port_value = getattr(K8sEnvironment.PORTS, port_attr)
        assert 1 <= port_value <= 65535


class TestK8sEnvironmentZMQ:
    """Tests for K8sEnvironment.ZMQ settings."""

    def test_zmq_default_values(self) -> None:
        """Test ZMQ settings have expected default values."""
        zmq = K8sEnvironment.ZMQ
        assert zmq.CONTROLLER_HOST is None
        assert zmq.IPC_PATH == "/aiperf/ipc"


class TestK8sEnvironmentJobSet:
    """Tests for K8sEnvironment.JOBSET settings."""

    def test_jobset_default_values(self) -> None:
        """Test JobSet settings have expected default values."""
        jobset = K8sEnvironment.JOBSET
        assert jobset.TTL_SECONDS_AFTER_FINISHED == 300
        assert jobset.CONTROLLER_BACKOFF_LIMIT == 0
        assert jobset.WORKER_BACKOFF_LIMIT == 3
        assert jobset.CONFIG_MOUNT_PATH == "/etc/aiperf"
        assert jobset.DATASETS_PATH == "/aiperf/datasets"

    def test_jobset_ttl_can_be_zero(self) -> None:
        """Test TTL can be set to zero for immediate cleanup."""
        settings = _JobSetSettings(TTL_SECONDS_AFTER_FINISHED=0)
        assert settings.TTL_SECONDS_AFTER_FINISHED == 0

    def test_jobset_backoff_limits_within_bounds(self) -> None:
        """Test backoff limits are within expected bounds."""
        jobset = K8sEnvironment.JOBSET
        assert 0 <= jobset.CONTROLLER_BACKOFF_LIMIT <= 10
        assert 0 <= jobset.WORKER_BACKOFF_LIMIT <= 20


class TestK8sEnvironmentAllSettings:
    """Tests for K8sEnvironment comprehensive coverage."""

    @pytest.mark.parametrize(
        "setting_name",
        [
            param("SYSTEM_CONTROLLER", id="system_controller"),
            param("WORKER_MANAGER", id="worker_manager"),
            param("TIMING_MANAGER", id="timing_manager"),
            param("DATASET_MANAGER", id="dataset_manager"),
            param("RECORDS_MANAGER", id="records_manager"),
            param("API", id="api"),
            param("GPU_TELEMETRY_MANAGER", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER", id="server_metrics"),
            param("RESULTS_SIDECAR", id="results_sidecar"),
            param("WORKER_POD", id="worker_pod"),
            param("HEALTH", id="health"),
            param("PORTS", id="ports"),
            param("ZMQ", id="zmq"),
            param("JOBSET", id="jobset"),
        ],
    )  # fmt: skip
    def test_all_settings_exist(self, setting_name: str) -> None:
        """Test all expected settings are available."""
        assert hasattr(K8sEnvironment, setting_name)
        setting = getattr(K8sEnvironment, setting_name)
        assert setting is not None

    @pytest.mark.parametrize(
        "resource_setting",
        [
            param(K8sEnvironment.SYSTEM_CONTROLLER, id="system_controller"),
            param(K8sEnvironment.WORKER_MANAGER, id="worker_manager"),
            param(K8sEnvironment.TIMING_MANAGER, id="timing_manager"),
            param(K8sEnvironment.DATASET_MANAGER, id="dataset_manager"),
            param(K8sEnvironment.RECORDS_MANAGER, id="records_manager"),
            param(K8sEnvironment.API, id="api"),
            param(K8sEnvironment.GPU_TELEMETRY_MANAGER, id="gpu_telemetry"),
            param(K8sEnvironment.SERVER_METRICS_MANAGER, id="server_metrics"),
            param(K8sEnvironment.RESULTS_SIDECAR, id="results_sidecar"),
            param(K8sEnvironment.WORKER_POD, id="worker_pod"),
        ],
    )  # fmt: skip
    def test_resource_settings_have_to_k8s_resources(self, resource_setting) -> None:
        """Test all resource settings have to_k8s_resources method."""
        assert hasattr(resource_setting, "to_k8s_resources")
        resources = resource_setting.to_k8s_resources()
        assert "requests" in resources
        assert "limits" in resources
        assert "cpu" in resources["requests"]
        assert "memory" in resources["requests"]
        assert "cpu" in resources["limits"]
        assert "memory" in resources["limits"]


class TestEnvironmentVariableOverrides:
    """Tests for environment variable configuration overrides."""

    def test_system_controller_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test controller-container settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_SYSTEM_CONTROLLER_CPU", "400m")

        settings = _resource_settings("SYSTEM_CONTROLLER_", "250m", "256Mi")
        assert settings.CPU == "400m"
        assert settings.MEMORY == "256Mi"

    def test_worker_pod_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test worker pod settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_WORKER_POD_CPU", "4000m")
        monkeypatch.setenv("AIPERF_K8S_WORKER_POD_MEMORY", "4096Mi")

        settings = _resource_settings("WORKER_POD_", "3350m", "6144Mi")
        assert settings.CPU == "4000m"
        assert settings.MEMORY == "4096Mi"

    def test_health_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test health settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS", "15")
        monkeypatch.setenv("AIPERF_K8S_HEALTH_PERIOD_SECONDS", "30")

        settings = _HealthProbeSettings()
        assert settings.INITIAL_DELAY_SECONDS == 15
        assert settings.PERIOD_SECONDS == 30

    def test_port_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test port settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_PORT_API_SERVICE", "8000")

        settings = _PortSettings()
        assert settings.API_SERVICE == 8000

    def test_zmq_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ZMQ settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_ZMQ_CONTROLLER_HOST", "controller.default.svc")
        monkeypatch.setenv("AIPERF_K8S_ZMQ_IPC_PATH", "/tmp/zmq")

        settings = _ZMQSettings()
        assert settings.CONTROLLER_HOST == "controller.default.svc"
        assert settings.IPC_PATH == "/tmp/zmq"

    def test_jobset_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test JobSet settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED", "600")
        monkeypatch.setenv("AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT", "5")
        monkeypatch.setenv("AIPERF_K8S_JOBSET_CONFIG_MOUNT_PATH", "/custom/config")

        settings = _JobSetSettings()
        assert settings.TTL_SECONDS_AFTER_FINISHED == 600
        assert settings.WORKER_BACKOFF_LIMIT == 5
        assert settings.CONFIG_MOUNT_PATH == "/custom/config"

    @pytest.mark.parametrize(
        "factory_prefix,env_prefix",
        [
            param("TIMING_MANAGER_", "AIPERF_K8S_TIMING_MANAGER_", id="timing"),
            param("DATASET_MANAGER_", "AIPERF_K8S_DATASET_MANAGER_", id="dataset"),
            param("RECORDS_MANAGER_", "AIPERF_K8S_RECORDS_MANAGER_", id="records"),
            param("RECORD_PROCESSOR_", "AIPERF_K8S_RECORD_PROCESSOR_", id="processor"),
            param("GPU_TELEMETRY_MANAGER_", "AIPERF_K8S_GPU_TELEMETRY_MANAGER_", id="gpu"),
            param("SERVER_METRICS_MANAGER_", "AIPERF_K8S_SERVER_METRICS_MANAGER_", id="server"),
        ],
    )  # fmt: skip
    def test_resource_settings_env_prefix(
        self, factory_prefix: str, env_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that resource settings use correct env prefix."""
        monkeypatch.setenv(f"{env_prefix}CPU", "999m")

        settings = _resource_settings(factory_prefix, "100m", "256Mi")
        assert settings.CPU == "999m"


class TestHealthProbeValidation:
    """Tests for health probe settings validation."""

    @pytest.mark.parametrize(
        "field,min_val,max_val,default_val",
        [
            param("INITIAL_DELAY_SECONDS", 0, 300, 5, id="initial_delay"),
            param("PERIOD_SECONDS", 1, 300, 10, id="period"),
            param("TIMEOUT_SECONDS", 1, 60, 5, id="timeout"),
            param("FAILURE_THRESHOLD", 1, 20, 10, id="failure"),
            param("SUCCESS_THRESHOLD", 1, 10, 1, id="success"),
        ],
    )  # fmt: skip
    def test_health_probe_bounds(
        self, field: str, min_val: int, max_val: int, default_val: int
    ) -> None:
        """Test health probe fields have correct bounds."""
        settings = _HealthProbeSettings()
        value = getattr(settings, field)

        assert value == default_val
        assert min_val <= value <= max_val

    def test_health_probe_validation_at_lower_bound(self) -> None:
        """Test health probe accepts values at lower bounds."""
        settings = _HealthProbeSettings(
            INITIAL_DELAY_SECONDS=0,
            PERIOD_SECONDS=1,
            TIMEOUT_SECONDS=1,
            FAILURE_THRESHOLD=1,
            SUCCESS_THRESHOLD=1,
        )
        assert settings.INITIAL_DELAY_SECONDS == 0
        assert settings.PERIOD_SECONDS == 1

    def test_health_probe_validation_at_upper_bound(self) -> None:
        """Test health probe accepts values at upper bounds."""
        settings = _HealthProbeSettings(
            INITIAL_DELAY_SECONDS=300,
            PERIOD_SECONDS=300,
            TIMEOUT_SECONDS=60,
            FAILURE_THRESHOLD=20,
            SUCCESS_THRESHOLD=10,
        )
        assert settings.INITIAL_DELAY_SECONDS == 300
        assert settings.SUCCESS_THRESHOLD == 10

    def test_health_probe_validation_exceeds_upper_bound_raises(self) -> None:
        """Test health probe rejects values exceeding upper bound."""
        with pytest.raises(ValidationError):
            _HealthProbeSettings(INITIAL_DELAY_SECONDS=301)

    def test_health_probe_validation_below_lower_bound_raises(self) -> None:
        """Test health probe rejects values below lower bound."""
        with pytest.raises(ValidationError):
            _HealthProbeSettings(PERIOD_SECONDS=0)


class TestPortValidation:
    """Tests for port settings validation."""

    def test_port_validation_at_bounds(self) -> None:
        """Test port accepts values at bounds."""
        settings = _PortSettings(SYSTEM_CONTROLLER_HEALTH=1, API_SERVICE=65535)
        assert settings.SYSTEM_CONTROLLER_HEALTH == 1
        assert settings.API_SERVICE == 65535

    def test_port_validation_invalid_lower_bound_raises(self) -> None:
        """Test port rejects values below 1."""
        with pytest.raises(ValidationError):
            _PortSettings(SYSTEM_CONTROLLER_HEALTH=0)

    def test_port_validation_invalid_upper_bound_raises(self) -> None:
        """Test port rejects values above 65535."""
        with pytest.raises(ValidationError):
            _PortSettings(API_SERVICE=65536)
