# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pure helpers in aiperf.kubernetes.jobset_helpers.

These tests pin down the small, stateless builders the JobSet manifest
generator composes from. Probe/volume/env semantics are load-bearing for
Kubernetes pod startup, so we verify the exact keys and values produced.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_helpers import (
    build_container_args,
    build_container_ports,
    build_env_vars,
    build_health_probe,
    build_security_context,
    build_service_probes,
    build_shared_volumes,
    build_startup_probe,
    build_volume_mounts,
)


class TestBuildSecurityContext:
    """Hardened security context is applied to every container."""

    def test_defaults_with_empty_pod_template(self) -> None:
        """Empty pod template still yields the baseline hardened context."""
        ctx = build_security_context(PodTemplateConfig())
        assert ctx["runAsNonRoot"] is True
        assert ctx["runAsUser"] == 1000
        assert ctx["runAsGroup"] == 1000
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["readOnlyRootFilesystem"] is True
        assert ctx["capabilities"] == {"drop": ["ALL"]}
        assert ctx["seccompProfile"] == {"type": "RuntimeDefault"}

    def test_override_scalar_keys_replace_defaults(self) -> None:
        """Scalar overrides from the pod template replace the hardened defaults."""
        template = PodTemplateConfig(
            container_security_context={
                "runAsUser": 2000,
                "readOnlyRootFilesystem": False,
            }
        )
        ctx = build_security_context(template)
        assert ctx["runAsUser"] == 2000
        assert ctx["readOnlyRootFilesystem"] is False
        assert ctx["runAsNonRoot"] is True  # untouched

    def test_capabilities_are_merged_not_replaced(self) -> None:
        """Capability overrides merge keys rather than replacing the whole block."""
        template = PodTemplateConfig(
            container_security_context={
                "capabilities": {"add": ["NET_BIND_SERVICE"]},
            }
        )
        ctx = build_security_context(template)
        assert ctx["capabilities"]["drop"] == ["ALL"]
        assert ctx["capabilities"]["add"] == ["NET_BIND_SERVICE"]


class TestBuildHealthProbe:
    """Health probe factory and its path/port plumbing."""

    def test_defaults_to_healthz_path(self) -> None:
        probe = build_health_probe(8080)
        assert probe["httpGet"] == {"path": "/healthz", "port": 8080}

    def test_custom_path_is_honored(self) -> None:
        probe = build_health_probe(8080, path="/readyz")
        assert probe["httpGet"]["path"] == "/readyz"

    def test_timing_fields_come_from_environment(self) -> None:
        """initialDelay/period/timeout are sourced from K8sEnvironment.HEALTH."""
        health = K8sEnvironment.HEALTH
        probe = build_health_probe(8080)
        assert probe["initialDelaySeconds"] == health.INITIAL_DELAY_SECONDS
        assert probe["periodSeconds"] == health.PERIOD_SECONDS
        assert probe["timeoutSeconds"] == health.TIMEOUT_SECONDS
        assert probe["failureThreshold"] == health.FAILURE_THRESHOLD
        assert probe["successThreshold"] == health.SUCCESS_THRESHOLD


class TestBuildStartupProbe:
    """Startup probes use zero initial delay and more lenient failure thresholds."""

    def test_initial_delay_is_zero(self) -> None:
        """Startup probes must poll immediately so slow images can boot without a fixed wait."""
        probe = build_startup_probe(8080)
        assert probe["initialDelaySeconds"] == 0

    def test_uses_startup_specific_threshold(self) -> None:
        """Startup probe uses STARTUP_* tunables, not the liveness tunables."""
        health = K8sEnvironment.HEALTH
        probe = build_startup_probe(8080)
        assert probe["periodSeconds"] == health.STARTUP_PERIOD_SECONDS
        assert probe["failureThreshold"] == health.STARTUP_FAILURE_THRESHOLD
        assert "successThreshold" not in probe

    def test_custom_path_is_passed_through(self) -> None:
        probe = build_startup_probe(9000, path="/alive")
        assert probe["httpGet"] == {"path": "/alive", "port": 9000}


class TestBuildServiceProbes:
    """Skip flags selectively disable individual probes."""

    def test_no_skips_returns_three_probes(self) -> None:
        startup, liveness, readiness = build_service_probes(
            8080,
            skip_startup_probe=False,
            skip_liveness_probe=False,
            skip_readiness_probe=False,
        )
        assert startup is not None
        assert liveness is not None
        assert readiness is not None
        assert readiness["httpGet"]["path"] == "/readyz"
        assert liveness["httpGet"]["path"] == "/healthz"

    @pytest.mark.parametrize(
        "skip_startup,skip_liveness,skip_readiness,expected_nones",
        [
            param(True, False, False, (True, False, False), id="skip-startup-only"),
            param(False, True, False, (False, True, False), id="skip-liveness-only"),
            param(False, False, True, (False, False, True), id="skip-readiness-only"),
            param(True, True, True, (True, True, True), id="skip-all"),
        ],
    )  # fmt: skip
    def test_skip_flags_disable_specific_probes(
        self,
        skip_startup: bool,
        skip_liveness: bool,
        skip_readiness: bool,
        expected_nones: tuple[bool, bool, bool],
    ) -> None:
        """Each skip flag must only clear its corresponding probe."""
        probes = build_service_probes(
            8080,
            skip_startup_probe=skip_startup,
            skip_liveness_probe=skip_liveness,
            skip_readiness_probe=skip_readiness,
        )
        assert tuple(probe is None for probe in probes) == expected_nones


class TestBuildVolumeMounts:
    """Volume mounts include the standard set plus any custom pod-template mounts."""

    def test_default_mount_names_and_paths(self) -> None:
        """All five baseline mounts must appear with expected mountPath values."""
        mounts = build_volume_mounts(PodTemplateConfig())
        names = [m["name"] for m in mounts]
        assert names == ["config", "ipc", "results", "datasets", "tmp"]

        by_name = {m["name"]: m for m in mounts}
        assert by_name["config"]["mountPath"] == K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH
        assert by_name["ipc"]["mountPath"] == K8sEnvironment.ZMQ.IPC_PATH
        assert by_name["results"]["mountPath"] == "/results"
        assert by_name["datasets"]["mountPath"] == K8sEnvironment.JOBSET.DATASETS_PATH
        assert by_name["tmp"]["mountPath"] == "/tmp"

    def test_config_mount_is_readonly(self) -> None:
        """Config is mounted read-only since it's a ConfigMap projection."""
        mounts = build_volume_mounts(PodTemplateConfig())
        config = next(m for m in mounts if m["name"] == "config")
        assert config.get("readOnly") is True

    def test_pod_template_mounts_appended(self) -> None:
        """Custom mounts are appended verbatim after the defaults."""
        template = PodTemplateConfig(
            volume_mounts=[
                {"name": "secrets", "mountPath": "/etc/secrets", "readOnly": True},
            ]
        )
        mounts = build_volume_mounts(template)
        assert mounts[-1] == {
            "name": "secrets",
            "mountPath": "/etc/secrets",
            "readOnly": True,
        }


class TestBuildSharedVolumes:
    """Volume definitions shared by controller and worker pods."""

    def test_config_volume_references_configmap(self) -> None:
        """The config volume must reference the ``<jobset>-config`` ConfigMap."""
        volumes = build_shared_volumes("my-bench", PodTemplateConfig())
        config = next(v for v in volumes if v["name"] == "config")
        assert config["configMap"] == {"name": "my-bench-config"}

    def test_default_volume_kinds(self) -> None:
        """ipc/results/datasets/tmp are all emptyDir; only config is ConfigMap-backed."""
        volumes = build_shared_volumes("bench", PodTemplateConfig())
        by_name = {v["name"]: v for v in volumes}
        assert "emptyDir" in by_name["ipc"]
        assert "emptyDir" in by_name["results"]
        assert "emptyDir" in by_name["datasets"]
        assert "emptyDir" in by_name["tmp"]

    def test_pod_template_volumes_appended(self) -> None:
        """Custom volumes are appended after the defaults."""
        template = PodTemplateConfig(
            volumes=[{"name": "hostcache", "hostPath": {"path": "/cache"}}],
        )
        volumes = build_shared_volumes("bench", template)
        assert volumes[-1] == {"name": "hostcache", "hostPath": {"path": "/cache"}}


class TestBuildContainerArgs:
    """CLI argument construction for ``aiperf service`` containers."""

    def test_minimal_args_with_no_ports_or_id(self) -> None:
        """Only service_type and benchmark-run file are required."""
        args = build_container_args("worker", None, None, None)
        run_file = f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/run_config.json"
        assert args == ["service", "--type", "worker", "--benchmark-run", run_file]

    def test_health_port_appended(self) -> None:
        args = build_container_args("worker", 8080, None, None)
        assert "--health-port" in args
        assert args[args.index("--health-port") + 1] == "8080"

    def test_service_id_appended(self) -> None:
        args = build_container_args("worker", None, None, "worker_0_1")
        assert "--service-id" in args
        assert args[args.index("--service-id") + 1] == "worker_0_1"

    def test_api_port_appended(self) -> None:
        args = build_container_args("api", None, 9090, None)
        assert "--api-port" in args
        assert args[args.index("--api-port") + 1] == "9090"

    def test_all_optional_fields_present_together(self) -> None:
        args = build_container_args("api", 8085, 9090, "api")
        # Order: health-port, service-id, api-port
        assert args.index("--health-port") < args.index("--service-id")
        assert args.index("--service-id") < args.index("--api-port")

    def test_zero_api_port_is_treated_as_unset(self) -> None:
        """api_port=0 is falsy; no --api-port flag is emitted (documents truthiness)."""
        args = build_container_args("worker", 8080, 0, None)
        assert "--api-port" not in args


class TestBuildContainerPorts:
    """Container port list construction."""

    def test_no_ports_when_both_none(self) -> None:
        assert build_container_ports(None, None) == []

    def test_only_health_port(self) -> None:
        ports = build_container_ports(8080, None)
        assert ports == [{"containerPort": 8080, "name": "health"}]

    def test_only_api_port(self) -> None:
        ports = build_container_ports(None, 9090)
        assert ports == [{"containerPort": 9090, "name": "api"}]

    def test_both_ports_appear_in_order(self) -> None:
        """Health precedes api to keep the probing port first."""
        ports = build_container_ports(8080, 9090)
        assert ports[0]["name"] == "health"
        assert ports[1]["name"] == "api"


class TestBuildEnvVars:
    """Environment variable construction for AIPerf containers."""

    def test_baseline_env_keys_present(self) -> None:
        """Job id, namespace, mmap base path, and health bindings are always injected."""
        env = build_env_vars(
            job_id="job-abc",
            namespace="bench",
            pod_template=PodTemplateConfig(),
        )
        keys = {item["name"] for item in env}
        assert {
            "AIPERF_DATASET_MMAP_BASE_PATH",
            "AIPERF_JOB_ID",
            "AIPERF_NAMESPACE",
            "AIPERF_SERVICE_HEALTH_ENABLED",
            "AIPERF_SERVICE_HEALTH_HOST",
            "AIPERF_SERVICE_REGISTRATION_TIMEOUT",
        }.issubset(keys)

    def test_job_id_and_namespace_values(self) -> None:
        env = build_env_vars(
            job_id="job-xyz", namespace="ns-7", pod_template=PodTemplateConfig()
        )
        by_name = {item["name"]: item for item in env}
        assert by_name["AIPERF_JOB_ID"]["value"] == "job-xyz"
        assert by_name["AIPERF_NAMESPACE"]["value"] == "ns-7"

    def test_hf_home_default_added_when_missing(self) -> None:
        """Default HF_HOME points at /tmp/hf_home so read-only root FS still works."""
        env = build_env_vars(
            job_id="j", namespace="n", pod_template=PodTemplateConfig()
        )
        hf = [item for item in env if item["name"] == "HF_HOME"]
        assert len(hf) == 1
        assert hf[0]["value"] == "/tmp/hf_home"

    def test_hf_home_not_added_when_pod_template_sets_it(self) -> None:
        """User-supplied HF_HOME must not be duplicated by the default injection."""
        template = PodTemplateConfig(env=[{"name": "HF_HOME", "value": "/data/hf"}])
        env = build_env_vars(job_id="j", namespace="n", pod_template=template)
        hf = [item for item in env if item["name"] == "HF_HOME"]
        assert len(hf) == 1
        assert hf[0]["value"] == "/data/hf"

    def test_pod_index_included_by_default(self) -> None:
        """AIPERF_POD_INDEX uses a fieldRef to the jobset job-index label."""
        env = build_env_vars(
            job_id="j", namespace="n", pod_template=PodTemplateConfig()
        )
        pod_index = next(
            (item for item in env if item["name"] == "AIPERF_POD_INDEX"), None
        )
        assert pod_index is not None
        field_ref = pod_index["valueFrom"]["fieldRef"]["fieldPath"]
        assert "jobset.sigs.k8s.io/job-index" in field_ref

    def test_pod_index_omitted_when_disabled(self) -> None:
        """Control-plane containers opt out with include_pod_index=False."""
        env = build_env_vars(
            job_id="j",
            namespace="n",
            pod_template=PodTemplateConfig(),
            include_pod_index=False,
        )
        assert all(item["name"] != "AIPERF_POD_INDEX" for item in env)

    def test_controller_host_env_appended_when_provided(self) -> None:
        env = build_env_vars(
            job_id="j",
            namespace="n",
            pod_template=PodTemplateConfig(),
            controller_host="controller.bench.svc",
        )
        host = next(
            item for item in env if item["name"] == "AIPERF_K8S_ZMQ_CONTROLLER_HOST"
        )
        assert host["value"] == "controller.bench.svc"

    def test_controller_host_env_absent_when_none(self) -> None:
        env = build_env_vars(
            job_id="j", namespace="n", pod_template=PodTemplateConfig()
        )
        assert all(item["name"] != "AIPERF_K8S_ZMQ_CONTROLLER_HOST" for item in env)

    def test_pod_template_env_appended_at_end(self) -> None:
        """Pod-template env wins precedence by being injected last."""
        template = PodTemplateConfig(env=[{"name": "CUSTOM", "value": "v"}])
        env = build_env_vars(job_id="j", namespace="n", pod_template=template)
        assert env[-1] == {"name": "CUSTOM", "value": "v"}

    def test_registration_timeout_not_below_probe_floor(self) -> None:
        """Registration timeout must be at least 2 * worker-connection-probe timeout."""
        env = build_env_vars(
            job_id="j", namespace="n", pod_template=PodTemplateConfig()
        )
        timeout = next(
            item
            for item in env
            if item["name"] == "AIPERF_SERVICE_REGISTRATION_TIMEOUT"
        )
        probe_floor = K8sEnvironment.JOBSET.WORKER_CONNECTION_PROBE_TIMEOUT * 2
        expected = max(Environment.SERVICE.REGISTRATION_TIMEOUT, probe_floor)
        assert float(timeout["value"]) == expected
