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
        # Shared HF tokenizer cache: control-plane warmer writes here, the
        # API container reads via snapshot_download(local_files_only=True),
        # and dataset-manager picks up its tokenizer from the same cache.
        # On worker pods the same mount exists but is unused — workers get
        # their tokenizer bundle from the operator API via the WGM, not HF.
        {"name": "tokenizer-cache", "mountPath": "/aiperf/hf_home"},
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
        # Shared HF cache for the controller pod's warmer + API + dataset-manager.
        {"name": "tokenizer-cache", "emptyDir": {}},
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


# ---------------------------------------------------------------------------
# Native cross-pod cellular topology (aiperf frontend controller + cell pods).
#
# Replaces the retired Python service mesh: instead of one controller pod of
# {system_controller, dataset_manager, timing_manager, records_manager, api}
# services plus worker pods wired over ZMQ, a cellular run is one controller pod
# (which binds a routable transport and merges shards) and `cells` cell pods
# (each a slice that streams its RecordsShardPartition back).
#
# The pods run the PYTHON aiperf frontend subcommands (`aiperf controller` /
# `aiperf cell`), not the bare native binary: the frontend is the orchestrator
# (orchestrator/native_execution.py) -- it reads Config v2, projects it through
# rust_wire, launches `aiperf --execute` over stdio, forwards progress, loads the
# native report, and runs the native export plane. The operator only owns the pod
# topology, the Config v2 mount, the budget partition (CELL_ID/CELL_COUNT), and
# the controller address the cells dial.
#
# The `aiperf controller` / `aiperf cell` subcommands and the cell<->controller
# bootstrap contract are the ai-dynamo/velo integration point; velo replaces the
# raw TCP CELL_CONTROLLER_ADDR with velo discovery. The operator side is stable.
# ---------------------------------------------------------------------------

# The controller serves its velo PeerInfo bootstrap on this port
# (aiperf runner's AIPERF_CONTROLLER_BOOTSTRAP_BIND default 0.0.0.0:9500,
# cellular_controller.rs::controller_bootstrap); cells fetch it from
# tcp://<controller-dns>:9500. The velo DATA plane binds an ephemeral port and is
# reached via the PeerInfo, so only this bootstrap port is a fixed contract.
CELL_CONTROLLER_PORT: int = 9500
# Env contract the velo runner reads (rust: aiperf::cellular::partition +
# cellular_cell + cell_launcher). Names/values must match the runner exactly.
CELL_ID_ENV = "AIPERF_CELL_ID"
CELL_COUNT_ENV = "AIPERF_CELL_COUNT"
CELL_CONTROLLER_ADDR_ENV = "AIPERF_CELL_CONTROLLER_ADDR"
# Selects the runner's cell launcher: "k8s" means the controller does NOT spawn
# cell children (the JobSet already created the cell pods; it waits on the velo
# registration barrier). cell_launcher.rs::CELL_LAUNCHER_ENV, default "local".
CELL_LAUNCHER_ENV = "AIPERF_CELL_LAUNCHER"
# The controller's Stage-E/G artifact HTTP upload server port. When a cellular run
# ships per-record artifacts or serves a file dataset cross-host, the controller
# binds 0.0.0.0:9600 and cells derive its host from their velo tcp:// coordinate +
# this port (cellular_cell.rs DEFAULT_ARTIFACT_PORT / AIPERF_CONTROLLER_ARTIFACT_BIND).
# Synthetic single-turn runs never start the server; exposing the port is harmless
# and lets artifact/file-dataset cellular runs reach the controller cross-pod.
CELL_ARTIFACT_PORT: int = 9600

# Tier-T2 aggregator tree env contract (rust: runner_protocol::cellular_aggregator +
# cellular_cell). Names/values must match the runner exactly.
CELL_AGG_FANOUT_ENV = "AIPERF_CELL_AGG_FANOUT"
CELL_BARRIER_FREE_ENV = "AIPERF_CELL_BARRIER_FREE"
# Set on the CONTROLLER pod (presence = operator wired the aggregator tier, so the
# controller expects M aggregator pods instead of spawning) AND on each CELL pod (the
# concrete ship-DNS template with a `{agg_id}` placeholder the cell fills from its
# round-robin aggregator `cell_id % M`). cellular_aggregator::AGG_DNS_TEMPLATE_ENV.
CELL_AGG_DNS_TEMPLATE_ENV = "AIPERF_CELL_AGG_DNS_TEMPLATE"
# The aggregator pod's velo bind port. Each aggregator pod binds tcp://0.0.0.0:9700
# (all interfaces so its pod DNS resolves); cells ship to {aggregator_dns}:9700.
AGGREGATOR_PORT: int = 9700
AGG_ID_ENV = "AIPERF_AGG_ID"
AGG_BIND_ENV = "AIPERF_AGG_BIND"
# Multi-tier tree env (rust: cellular_aggregator). AGG_TIER_INDEX is this pod's 0-based
# tier in the tier_counts plan; the aggregator derives its collect barrier and parent
# from it + the static cell-count/fanout (a JobSet indexed replicatedJob shares one env
# template, so per-pod placement must be derived). AGG_SHIP_ADDR is the DNS *template*
# (a `{agg_id}` placeholder) a LOWER-tier pod ships its one merged store up to; the pod
# fills it from its round-robin parent `agg_id % parent_count`. The TOP tier sets no
# AGG_SHIP_ADDR and ships to the controller. cellular_aggregator::AGG_TIER_INDEX_ENV /
# AGG_SHIP_ADDR_ENV.
AGG_TIER_INDEX_ENV = "AIPERF_AGG_TIER_INDEX"
AGG_SHIP_ADDR_ENV = "AIPERF_AGG_SHIP_ADDR"

# Per-record artifact plane env the runner reads (rust: cellular_cell). The HTTP
# artifact plane binds CELL_ARTIFACT_PORT on the controller and cells derive its host
# from their routable controller DNS + this port. The velo artifact transport instead
# rides the shared 9500 velo plane (no second port). cellular_cell::CELL_ARTIFACT_PORT_ENV
# / ARTIFACT_TRANSPORT_ENV.
CELL_ARTIFACT_PORT_ENV = "AIPERF_CELL_ARTIFACT_PORT"
ARTIFACT_TRANSPORT_ENV = "AIPERF_ARTIFACT_TRANSPORT"

# HF tokenizer cache dir; the runner tokenizes in-process, so it is the one piece
# of the mesh container env the native pods still need. Matches the tokenizer-cache
# volume mount from build_runner_volume_mounts.
_HF_HOME = "/aiperf/hf_home"


def _config_path() -> str:
    """Path of the mounted Config v2 file the aiperf frontend reads via --config."""
    return f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/config.yaml"


def build_runner_env_vars(pod_template: PodTemplateConfig) -> list[dict[str, Any]]:
    """The clean base env for an aiperf runner (controller or cell) pod.

    Deliberately NOT build_env_vars: the native runner reads none of the mesh
    container env (AIPERF_DATASET_MMAP_BASE_PATH, AIPERF_SERVICE_HEALTH_*,
    AIPERF_SERVICE_REGISTRATION_TIMEOUT, AIPERF_CONTROLLER_POD,
    AIPERF_UI_REALTIME_METRICS_ENABLED, AIPERF_K8S_ZMQ_CONTROLLER_HOST) — those are
    ZMQ registration / service-health / dataset-manager wiring for the retired
    service mesh. A cellular pod needs only the HF tokenizer cache location plus any
    user-supplied podTemplate env (HF_TOKEN, proxies, etc.); the cell partition env
    is layered on top by build_cell_env_vars.
    """
    env: list[dict[str, Any]] = [{"name": "HF_HOME", "value": _HF_HOME}]
    reserved = {item["name"] for item in env}
    env.extend(
        item for item in pod_template.env if (item or {}).get("name") not in reserved
    )
    return env


def build_controller_args() -> list[str]:
    """Build the `aiperf controller` frontend args for the controller pod.

    The controller frontend reads Config v2 (cells count included as
    runtime.cells), projects it through rust_wire, launches aiperf runner in
    controller mode -- binding its cell transport on CELL_CONTROLLER_PORT so the
    sibling cell pods can dial it -- collects one records-shard partition per cell,
    merges them into the single authoritative report, and runs the native export
    plane over the merged result. Subcommand/flags are the velo integration point.
    """
    return ["controller", "--config", _config_path()]


def build_cell_args() -> list[str]:
    """Build the `aiperf cell` frontend args for a cell pod.

    The cell frontend reads the same Config v2, derives its budget slice from
    CELL_ID/CELL_COUNT, projects through rust_wire, launches aiperf runner in cell
    mode, and ships its records-shard partition to CELL_CONTROLLER_ADDR.
    Subcommand/flags are the velo integration point.
    """
    return ["cell", "--config", _config_path()]


def build_aggregator_args() -> list[str]:
    """Build the `aiperf aggregator` frontend args for a tier-T2 aggregator pod.

    The aggregator frontend reads the same Config v2, projects through rust_wire, and
    pipes the run envelope to `aiperf --aggregator` on stdin (the runner needs
    only the merge MetricsConfig from it). The runner then binds AGG_BIND, collects its
    subtree's folded stores, merges them, and ships one merged store to the controller.
    """
    return ["aggregator", "--config", _config_path()]


def build_cell_env_vars(
    *,
    cells: int,
    controller_dns: str,
    agg_fanout: int | None = None,
    agg_ship_template: str | None = None,
    artifact_http_port: int | None = None,
) -> list[dict[str, Any]]:
    """Env for a cell pod: its partition index, the cell count, and the controller.

    CELL_ID is the JobSet job-index of this cell's replicated-job replica (each of
    the `cells` replicas is a distinct indexed job, so the indices tile
    `0..cells-1`), sourced from the same `jobset.sigs.k8s.io/job-index` label the
    mesh used for AIPERF_POD_INDEX. CELL_CONTROLLER_ADDR is the controller pod's
    stable JobSet DNS name plus the transport port.

    Tier-T2: when `agg_fanout` + `agg_ship_template` are given, the cell also gets the
    fanout (to compute M) and the concrete ship-DNS template with a `{agg_id}`
    placeholder. A JobSet indexed replicatedJob shares one env template, so the cell
    fills `{agg_id}` from its own `cell_id % M` pod-side (cellular_cell::ship_target);
    it still fetches its envelope + START from the controller and only *ships* to the
    aggregator, so its partition is byte-identical to the flat topology.
    """
    env: list[dict[str, Any]] = [
        {
            "name": CELL_ID_ENV,
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']"
                }
            },
        },
        {"name": CELL_COUNT_ENV, "value": str(cells)},
        {
            # velo fetches the controller PeerInfo from this bootstrap coordinate;
            # the runner requires the ``tcp://HOST:PORT`` form for k8s
            # (cell_launcher.rs CellLaunchContext.controller_coordinate).
            "name": CELL_CONTROLLER_ADDR_ENV,
            "value": f"tcp://{controller_dns}:{CELL_CONTROLLER_PORT}",
        },
    ]
    if agg_fanout is not None and agg_ship_template is not None:
        env.append({"name": CELL_AGG_FANOUT_ENV, "value": str(agg_fanout)})
        env.append(
            {"name": CELL_AGG_DNS_TEMPLATE_ENV, "value": agg_ship_template}
        )
    if artifact_http_port is not None:
        # HTTP artifact plane: make the controller's artifact-server port explicit so a
        # cell POSTs per-record artifacts to `<controller-dns>:<port>` (it otherwise
        # derives the port from the DEFAULT_ARTIFACT_PORT fallback). Omitted for the velo
        # artifact transport, which rides the shared 9500 velo plane (no second port).
        env.append({"name": CELL_ARTIFACT_PORT_ENV, "value": str(artifact_http_port)})
    return env


def build_controller_env_vars(
    *,
    agg_fanout: int | None = None,
    agg_ship_template: str | None = None,
    barrier_free: bool = False,
) -> list[dict[str, Any]]:
    """Env for the controller pod: select the k8s cell launcher + tier T2/T3 knobs.

    ``AIPERF_CELL_LAUNCHER=k8s`` tells the runner-controller not to spawn cell
    children (the JobSet already created the ``cells`` cell pods); it binds its velo
    bootstrap on the default ``0.0.0.0:9500`` and waits for the cell pods to register.

    Tier-T2: when `agg_fanout` + `agg_ship_template` are given, the controller gets the
    fanout (to compute M = expected_partitions) and the DNS template whose *presence*
    is the k8s "expect, don't spawn" gate (cellular_controller effective_aggregator_count
    / AGG_DNS_TEMPLATE_ENV). Tier-T3: `barrier_free` triggers START immediately.
    """
    env: list[dict[str, Any]] = [{"name": CELL_LAUNCHER_ENV, "value": "k8s"}]
    if agg_fanout is not None and agg_ship_template is not None:
        env.append({"name": CELL_AGG_FANOUT_ENV, "value": str(agg_fanout)})
        env.append(
            {"name": CELL_AGG_DNS_TEMPLATE_ENV, "value": agg_ship_template}
        )
    if barrier_free:
        env.append({"name": CELL_BARRIER_FREE_ENV, "value": "1"})
    return env


def build_aggregator_env_vars(
    *,
    cells: int,
    agg_fanout: int,
    controller_dns: str,
    tier_index: int | None = None,
    ship_template: str | None = None,
) -> list[dict[str, Any]]:
    """Env for an aggregator pod at one tier of the reduction tree.

    The aggregator binds all interfaces (so its pod DNS resolves to it) and collects
    its subtree. Its per-pod collect barrier is derived runner-side from AGG_ID + the
    static cell-count + fanout (+ tier for a multi-tier tree); a JobSet indexed
    replicatedJob shares one env template, so an uneven round-robin split cannot be a
    static value.

    - AGG_ID: the pod's `jobset.sigs.k8s.io/job-index` (like the cell's CELL_ID).
    - AGG_BIND: `tcp://0.0.0.0:{AGGREGATOR_PORT}` (all interfaces, not loopback).
    - CELL_COUNT + CELL_AGG_FANOUT: static, so the runner derives the tier plan and this
      pod's child count via the same gates as the controller.
    - CELL_CONTROLLER_ADDR: where the merged store ships up on the TOP tier.

    Multi-tier (`tier_index`/`ship_template` set): AGG_TIER_INDEX locates this pod's tier
    so the runner picks its tier's node count from the plan; a LOWER tier also gets
    AGG_SHIP_ADDR — the parent-tier DNS *template* whose `{agg_id}` the runner fills from
    its round-robin parent `agg_id % parent_count` (the TOP tier omits it and ships to the
    controller). The single-tier tree passes neither, staying byte-identical.
    """
    env: list[dict[str, Any]] = [
        {
            "name": AGG_ID_ENV,
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']"
                }
            },
        },
        {"name": AGG_BIND_ENV, "value": f"tcp://0.0.0.0:{AGGREGATOR_PORT}"},
        {"name": CELL_COUNT_ENV, "value": str(cells)},
        {"name": CELL_AGG_FANOUT_ENV, "value": str(agg_fanout)},
        {
            "name": CELL_CONTROLLER_ADDR_ENV,
            "value": f"tcp://{controller_dns}:{CELL_CONTROLLER_PORT}",
        },
    ]
    if tier_index is not None:
        env.append({"name": AGG_TIER_INDEX_ENV, "value": str(tier_index)})
    if ship_template is not None:
        env.append({"name": AGG_SHIP_ADDR_ENV, "value": ship_template})
    return env


def build_cr_identity_env(*, job_id: str, namespace: str) -> list[dict[str, Any]]:
    """The AIPerfJob CR identity a run pod needs to patch its own .status.

    Live progress is reported kubectl-natively: the aiperf frontend patches the
    owning AIPerfJob's ``.status`` (phase + per-phase requestsCompleted/rate) from
    the runner's progress, and the operator's orchestration-level results_server
    serves that status back -- so there is no per-run progress API service. The pods
    already carry RBAC to patch ``aiperfjobs/status`` (benchmark-rbac.yaml); this is
    the identity they resolve the CR by. (Distinct from the retired mesh env: these
    two vars name the CR to patch, they are not ZMQ/service wiring.)
    """
    return [
        {"name": "AIPERF_JOB_ID", "value": job_id},
        {"name": "AIPERF_NAMESPACE", "value": namespace},
    ]


def build_runner_volumes(
    jobset_name: str, pod_template: PodTemplateConfig
) -> list[dict[str, Any]]:
    """Pod-level volumes for a cellular runner pod (controller or cell).

    Drops the mesh-only `ipc` (ZMQ IPC socket dir) and `datasets` (dataset-manager
    mmap) volumes from build_shared_volumes: the runner speaks no ZMQ and resolves
    its dataset in-process. Keeps the mounted protocol-v2 run envelope (config), the
    exported results, the HF tokenizer cache, and scratch.
    """
    volumes: list[dict[str, Any]] = [
        {"name": "config", "configMap": {"name": f"{jobset_name}-config"}},
        {"name": "results", "emptyDir": {}},
        {"name": "tokenizer-cache", "emptyDir": {}},
        {"name": "tmp", "emptyDir": {}},
    ]
    volumes.extend(pod_template.volumes)
    return volumes


def build_runner_volume_mounts(pod_template: PodTemplateConfig) -> list[dict[str, Any]]:
    """Volume mounts for a cellular runner container (matches build_runner_volumes)."""
    mounts: list[dict[str, Any]] = [
        {
            "name": "config",
            "mountPath": K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH,
            "readOnly": True,
        },
        {"name": "results", "mountPath": "/results"},
        {"name": "tokenizer-cache", "mountPath": _HF_HOME},
        {"name": "tmp", "mountPath": "/tmp"},
    ]
    mounts.extend(pod_template.volume_mounts)
    return mounts


def build_env_vars(
    *,
    job_id: str,
    namespace: str,
    pod_template: PodTemplateConfig,
    controller_host: str | None = None,
    include_pod_index: bool = True,
    controller_pod: bool = False,
) -> list[dict[str, Any]]:
    """Create environment variables for a container.

    Args:
        controller_pod: When True, this container runs inside the controller
            pod (api / dataset-manager / etc.) -- emit AIPERF_CONTROLLER_POD=1
            so bootstrap.py skips HF offline-mode (the controller needs HF
            egress for prewarming the shared cache). Worker-pod containers
            default to False and inherit offline mode.
    """
    datasets_path = K8sEnvironment.JOBSET.DATASETS_PATH
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

    if controller_pod:
        # Marker consumed by aiperf.common.bootstrap._configure_child_process:
        # presence of this var disables the HF offline-mode default so the
        # controller pod can reach HuggingFace to warm the shared cache.
        env.append({"name": "AIPERF_CONTROLLER_POD", "value": "1"})
        # Records-manager publishes RealtimeMetricsMessage only when this gate
        # is true (records_manager.py::_report_realtime_inference_metrics_task).
        # Set on every controller-pod container so any bus client can publish.
        env.append({"name": "AIPERF_UI_REALTIME_METRICS_ENABLED", "value": "true"})

    # HF cache lives on the shared `tokenizer-cache` emptyDir so the
    # controller pod's warmer, dataset-manager, and api containers all see
    # the same on-disk snapshots. Worker pods carry the mount too but never
    # write to it — they receive bundles from the operator API instead.
    env.append({"name": "HF_HOME", "value": "/aiperf/hf_home"})

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

    reserved_names = {item["name"] for item in env}
    env.extend(
        item
        for item in pod_template.env
        if (item or {}).get("name") not in reserved_names
    )
    return env
