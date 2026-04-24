# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helper builders for JobSet manifest generation.

Stateless helpers used by :mod:`aiperf.kubernetes.jobset_builder` and
consumed by :class:`aiperf.kubernetes.jobset.AIPerfJobSetSpec`. Kept
separate so the per-spec builder module stays under the file-size limit.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.environment import Environment
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.environment import K8sEnvironment


def build_security_context(pod_template: PodTemplateConfig) -> dict[str, Any]:
    """Create a security context for containers.

    Applies security best practices:
    - Run as non-root user
    - Drop all capabilities
    - Read-only root filesystem (writable emptyDir volumes for data/ipc/results)
    """
    ctx: dict[str, Any] = {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    overrides = pod_template.container_security_context
    if overrides:
        caps = overrides.get("capabilities")
        if isinstance(caps, dict):
            base_caps = dict(ctx.get("capabilities", {}))
            base_caps.update(caps)
            ctx["capabilities"] = base_caps
        for key, value in overrides.items():
            if key == "capabilities":
                continue
            ctx[key] = value
    return ctx


def build_health_probe(port: int, path: str = "/healthz") -> dict[str, Any]:
    """Create a health probe configuration from K8sEnvironment settings."""
    health = K8sEnvironment.HEALTH
    return {
        "httpGet": {"path": path, "port": port},
        "initialDelaySeconds": health.INITIAL_DELAY_SECONDS,
        "periodSeconds": health.PERIOD_SECONDS,
        "timeoutSeconds": health.TIMEOUT_SECONDS,
        "failureThreshold": health.FAILURE_THRESHOLD,
        "successThreshold": health.SUCCESS_THRESHOLD,
    }


def build_startup_probe(port: int, path: str = "/healthz") -> dict[str, Any]:
    """Create a startup probe for slow-starting containers.

    Startup probes allow containers more time to initialize before
    liveness/readiness probes take over. Uses more lenient settings
    to accommodate initialization time.
    """
    health = K8sEnvironment.HEALTH
    return {
        "httpGet": {"path": path, "port": port},
        "initialDelaySeconds": 0,
        "periodSeconds": health.STARTUP_PERIOD_SECONDS,
        "timeoutSeconds": health.TIMEOUT_SECONDS,
        "failureThreshold": health.STARTUP_FAILURE_THRESHOLD,
    }


def build_service_probes(
    probe_port: int | None,
    *,
    skip_startup_probe: bool,
    skip_liveness_probe: bool,
    skip_readiness_probe: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build (startup, liveness, readiness) probes, honoring skip flags."""
    startup = None if skip_startup_probe else build_startup_probe(probe_port)
    liveness = None if skip_liveness_probe else build_health_probe(probe_port)
    readiness = (
        None if skip_readiness_probe else build_health_probe(probe_port, path="/readyz")
    )
    return startup, liveness, readiness


def build_volume_mounts(pod_template: PodTemplateConfig) -> list[dict[str, Any]]:
    """Get all volume mounts including config and IPC."""
    config_path = K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH
    ipc_path = K8sEnvironment.ZMQ.IPC_PATH
    datasets_path = K8sEnvironment.JOBSET.DATASETS_PATH
    mounts: list[dict[str, Any]] = [
        {"name": "config", "mountPath": config_path, "readOnly": True},
        {"name": "ipc", "mountPath": ipc_path},
        {"name": "results", "mountPath": "/results"},
        # Shared dataset volume: dataset-manager writes, API serves to workers
        {"name": "datasets", "mountPath": datasets_path},
        {"name": "tmp", "mountPath": "/tmp"},
    ]
    mounts.extend(pod_template.volume_mounts)
    return mounts


def build_shared_volumes(
    jobset_name: str, pod_template: PodTemplateConfig
) -> list[dict[str, Any]]:
    """Build the shared volume list for both controller and worker pods."""
    volumes: list[dict[str, Any]] = [
        {"name": "config", "configMap": {"name": f"{jobset_name}-config"}},
        {"name": "ipc", "emptyDir": {}},
        {"name": "results", "emptyDir": {}},
        # Shared dataset volume for controller containers (dataset-manager creates, API serves)
        {"name": "datasets", "emptyDir": {}},
        {"name": "tmp", "emptyDir": {}},
    ]
    volumes.extend(pod_template.volumes)
    return volumes


def build_container_args(
    service_type: str,
    health_port: int | None,
    api_port: int | None,
    service_id: str | None,
) -> list[str]:
    """Build the `aiperf service` CLI args for a container."""
    run_file = f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/run_config.json"
    args = [
        "service",
        "--type",
        service_type,
        "--benchmark-run",
        run_file,
    ]
    if health_port is not None:
        args.extend(["--health-port", str(health_port)])
    if service_id:
        args.extend(["--service-id", service_id])
    if api_port:
        args.extend(["--api-port", str(api_port)])
    return args


def build_container_ports(
    health_port: int | None, api_port: int | None
) -> list[dict[str, Any]]:
    """Build the K8s containerPorts list for a service container."""
    ports: list[dict[str, Any]] = []
    if health_port is not None:
        ports.append({"containerPort": health_port, "name": "health"})
    if api_port:
        ports.append({"containerPort": api_port, "name": "api"})
    return ports


def build_env_vars(
    *,
    job_id: str,
    namespace: str,
    pod_template: PodTemplateConfig,
    controller_host: str | None = None,
    include_pod_index: bool = True,
) -> list[dict[str, Any]]:
    """Create environment variables for a container."""
    datasets_path = K8sEnvironment.JOBSET.DATASETS_PATH
    has_hf_home = any(
        (item or {}).get("name") == "HF_HOME" for item in pod_template.env
    )
    # Give the controller enough registration headroom for workers to
    # complete their PUB/SUB connection probes plus one restart cycle if
    # the first-attempt probe fails (Kubernetes restarts on exit).
    registration_timeout = max(
        Environment.SERVICE.REGISTRATION_TIMEOUT,
        K8sEnvironment.JOBSET.WORKER_CONNECTION_PROBE_TIMEOUT * 2,
    )
    env: list[dict[str, Any]] = [
        # Shared dataset path: dataset-manager writes mmap files here,
        # API service serves them to workers via HTTP
        {"name": "AIPERF_DATASET_MMAP_BASE_PATH", "value": datasets_path},
        # Job ID and namespace for the benchmark
        {"name": "AIPERF_JOB_ID", "value": job_id},
        {"name": "AIPERF_NAMESPACE", "value": namespace},
        # Health server must bind to 0.0.0.0 so K8s probes can reach it via the pod IP
        {"name": "AIPERF_SERVICE_HEALTH_ENABLED", "value": "true"},
        {"name": "AIPERF_SERVICE_HEALTH_HOST", "value": "0.0.0.0"},
        # Keep registration timeout configurable instead of hard-coding a
        # separate K8s-only value, so startup tuning and failure detection
        # stay aligned with the active environment configuration.
        {
            "name": "AIPERF_SERVICE_REGISTRATION_TIMEOUT",
            "value": str(registration_timeout),
        },
    ]

    # HF cache must be writable (readOnlyRootFilesystem)
    if not has_hf_home:
        env.append({"name": "HF_HOME", "value": "/tmp/hf_home"})

    if include_pod_index:
        # Expose the JobSet job-index as a unique pod identifier for
        # worker-pod services. JOB_COMPLETION_INDEX is always 0 because
        # each replicated job has completions=1; the JobSet job-index
        # label is the true replica index.
        env.append(
            {
                "name": "AIPERF_POD_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                    }
                },
            }
        )

    if controller_host:
        env.append({"name": "AIPERF_K8S_ZMQ_CONTROLLER_HOST", "value": controller_host})

    # Add custom environment variables from pod template
    env.extend(pod_template.env)
    return env
