# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Kubernetes deployment configuration models.

These models provide a single source of truth for all Kubernetes deployment
concerns (pod templates, scheduling, images) with camelCase aliases for
CRD round-tripping.
"""

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, ValidationInfo, field_validator
from pydantic.alias_generators import to_camel

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.finite import FiniteFloat
from aiperf.config.base import BaseConfig
from aiperf.kubernetes.enums import ImagePullPolicy

_logger = AIPerfLogger(__name__)

# PodSpec keys that ``extra_pod_spec`` may not set. A raw ``update()`` of these
# would silently defeat the pod hardening applied by
# ``aiperf.kubernetes.jobset_specs``: ``securityContext`` carries runAsNonRoot,
# the ``host*`` namespace shares break pod isolation, and replacing
# ``containers`` wholesale drops the per-container hardened securityContext.
# Use the typed fields (``pod_security_context``, etc.) instead.
DENIED_EXTRA_POD_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "securityContext",
        "containers",
        "hostNetwork",
        "hostPID",
        "hostIPC",
        "hostUsers",
    }
)

# securityContext assertions that no user-supplied override may weaken, as
# (key, forbidden value) pairs. Applied to both the pod-level and
# container-level securityContext escape hatches so the two agree.
_FORBIDDEN_SECURITY_CONTEXT_VALUES: tuple[tuple[str, Any], ...] = (
    ("privileged", True),
    ("allowPrivilegeEscalation", True),
    ("runAsNonRoot", False),
    ("runAsUser", 0),
    ("runAsGroup", 0),
)


def privilege_escalating_keys(value: dict[str, Any]) -> list[str]:
    """Return the securityContext keys in ``value`` set to a forbidden value.

    Args:
        value: A user-supplied securityContext mapping.

    Returns:
        Sorted names of keys whose value would weaken the hardened baseline.
        Empty when the override is safe.
    """
    offenders: list[str] = []
    for key, forbidden in _FORBIDDEN_SECURITY_CONTEXT_VALUES:
        if key not in value:
            continue
        actual = value[key]
        # Guard against bool/int conflation: False == 0 and True == 1 in Python.
        # For bool-forbidden keys, compare truthiness (not identity) so a
        # YAML/JSON int like `privileged: 1` is still caught, while non-bool,
        # non-int types (e.g. strings) are left to the K8s API server's own
        # type validation rather than guessed at here.
        matches = (
            isinstance(actual, (bool, int)) and bool(actual) == forbidden
            if isinstance(forbidden, bool)
            else not isinstance(actual, bool) and actual == forbidden
        )
        if matches:
            offenders.append(key)
    return sorted(offenders)


# Volume source keys that hand a benchmark pod node-level state. These are NOT
# denied: GPU benchmarking legitimately needs host mounts (device nodes, driver
# directories, local NVMe scratch). They are surfaced as warnings so an operator
# reading the logs can see what a submitted job asked for.
RISKY_VOLUME_SOURCE_KEYS: frozenset[str] = frozenset(
    {
        "hostPath",
        "nfs",
        "iscsi",
        "cephfs",
        "glusterfs",
        "rbd",
        "flexVolume",
    }
)

# hostPath prefixes that expose node credentials, the container runtime, or the
# kubelet's own state. A hostPath under one of these is worth calling out by
# name in the warning even among other host mounts.
SENSITIVE_HOST_PATH_PREFIXES: tuple[str, ...] = (
    "/",
    "/etc",
    "/root",
    "/home",
    "/boot",
    "/proc",
    "/var/run",
    "/run",
    "/var/lib/kubelet",
    "/var/lib/docker",
    "/var/lib/containerd",
)


def _is_sensitive_host_path(path: str) -> bool:
    """Return True if a hostPath maps a node directory holding host credentials.

    Args:
        path: The ``hostPath.path`` value from a user-supplied volume.

    Returns:
        True when the path is the node root or sits at/inside a directory that
        carries node credentials, runtime sockets, or kubelet state.
    """
    normalized = path.rstrip("/") or "/"
    if normalized == "/":
        return True
    # "/" is exact-match only above: every path starts with it, so treating it as
    # a prefix would flag benign mounts like /dev/shm.
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}/")
        for prefix in SENSITIVE_HOST_PATH_PREFIXES
        if prefix != "/"
    )


def risky_volume_warnings(
    volumes: list[dict[str, Any]], field_path: str = "podTemplate.volumes"
) -> list[str]:
    """Describe user-supplied volumes that widen the benchmark pod's blast radius.

    Args:
        volumes: Pod volumes in K8s Volume format, as supplied by the user.
        field_path: Dotted config path used to prefix each message.

    Returns:
        One human-readable warning per risky volume. Empty when nothing is risky.
    """
    warnings: list[str] = []
    for index, volume in enumerate(volumes):
        if not isinstance(volume, dict):
            continue
        name = volume.get("name", "<unnamed>")
        for source in sorted(RISKY_VOLUME_SOURCE_KEYS & volume.keys()):
            body = volume.get(source)
            if source == "hostPath" and isinstance(body, dict):
                host_path = str(body.get("path", "<unset>"))
                detail = f"hostPath volume mounting node path {host_path!r}"
                if _is_sensitive_host_path(host_path):
                    detail += (
                        " which exposes node credentials, the container runtime, "
                        "or kubelet state to the benchmark pod"
                    )
            else:
                detail = f"{source} volume backed by storage outside the pod"
            warnings.append(
                f"{field_path}[{index}] ({name!r}): {detail}. This is allowed - "
                "AIPerf benchmarks legitimately need host access - but it is not "
                "covered by AIPerf's pod hardening."
            )
    return warnings


def risky_security_context_details(ctx: dict[str, Any]) -> list[str]:
    """Describe securityContext entries that widen AIPerf's hardened baseline.

    Covers the constructs that are deliberately *not* rejected: a benchmark pod
    legitimately needs ``SYS_ADMIN`` for perf/profiling tooling, an unconfined
    seccomp profile for GPU and tracing workloads, and a writable root
    filesystem for tools that insist on writing outside a mounted volume.

    Args:
        ctx: A user-supplied container or pod securityContext mapping.

    Returns:
        Short phrases naming each widening construct. Empty when there is none.
    """
    details: list[str] = []
    offenders = privilege_escalating_keys(ctx)
    if offenders:
        details.append("sets " + ", ".join(f"{key}={ctx[key]!r}" for key in offenders))
    capabilities = ctx.get("capabilities")
    if isinstance(capabilities, dict):
        added = capabilities.get("add") or []
        if added:
            details.append(f"adds Linux capabilities {sorted(map(str, added))}")
        dropped = capabilities.get("drop")
        if dropped is not None and "ALL" not in {str(item) for item in dropped}:
            details.append(
                f"narrows the dropped capability set to {sorted(map(str, dropped))} "
                "instead of ALL"
            )
    if ctx.get("readOnlyRootFilesystem") is False:
        details.append("disables readOnlyRootFilesystem")
    seccomp = ctx.get("seccompProfile")
    if isinstance(seccomp, dict):
        profile_type = str(seccomp.get("type", ""))
        if profile_type and profile_type != "RuntimeDefault":
            details.append(f"sets seccompProfile.type={profile_type!r}")
    return details


def risky_security_context_warnings(ctx: dict[str, Any], field_path: str) -> list[str]:
    """Build the operator-facing warning for a widening securityContext override.

    Args:
        ctx: A user-supplied container or pod securityContext mapping.
        field_path: Dotted config path used to prefix the message.

    Returns:
        A single-element list when ``ctx`` widens the baseline, else empty.
    """
    details = risky_security_context_details(ctx)
    if not details:
        return []
    return [
        f"{field_path}: {'; '.join(details)}. This is allowed - profiling, GPU, "
        "and tracing workloads legitimately need it - but it widens the hardened "
        "container securityContext AIPerf applies to every benchmark container."
    ]


def risky_init_container_warnings(
    init_containers: list[dict[str, Any]],
    field_path: str = "podTemplate.initContainers",
) -> list[str]:
    """Describe init containers that opt out of the hardened container baseline.

    Args:
        init_containers: InitContainers in K8s Container format.
        field_path: Dotted config path used to prefix each message.

    Returns:
        One human-readable warning per risky init container.
    """
    warnings: list[str] = []
    for index, container in enumerate(init_containers):
        if not isinstance(container, dict):
            continue
        name = container.get("name", "<unnamed>")
        ctx = container.get("securityContext")
        if not isinstance(ctx, dict):
            continue
        details = risky_security_context_details(ctx) or [
            "supplies its own securityContext, so AIPerf's hardened container "
            "baseline is not applied to it"
        ]
        warnings.append(
            f"{field_path}[{index}] ({name!r}): {'; '.join(details)}. This is "
            "allowed - init containers exist for sysctl tweaks and permission "
            "fixups - but the container runs outside AIPerf's hardened baseline."
        )
    return warnings


def _reject_privilege_escalation(value: dict[str, Any], field_name: str) -> None:
    """Raise if a securityContext override would weaken the hardened baseline.

    Args:
        value: The user-supplied securityContext mapping.
        field_name: Config field name, used in the error message.

    Raises:
        ValueError: If the override sets a privilege-escalating value.
    """
    offenders = privilege_escalating_keys(value)
    if offenders:
        raise ValueError(
            f"{field_name} may not set {', '.join(offenders)} to a "
            "privilege-escalating value: it would weaken the hardened "
            "securityContext AIPerf applies to every benchmark pod."
        )


class SchedulingConfig(BaseConfig):
    """Kueue gang-scheduling configuration."""

    model_config = ConfigDict(extra="forbid")

    queue_name: str | None = Field(
        default=None,
        description="Kueue LocalQueue name for gang-scheduling",
    )
    priority_class: str | None = Field(
        default=None,
        description="Kueue WorkloadPriorityClass name (for queue admission ordering). "
        "Distinct from podTemplate.priorityClassName, which is the native K8s "
        "PriorityClass used by the default scheduler for preemption.",
    )


class PodTemplateConfig(BaseConfig):
    """Kubernetes pod template configuration in K8s-native formats."""

    model_config = ConfigDict(extra="forbid")

    env: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Environment variables in K8s EnvVar format",
    )
    volumes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod volumes in K8s Volume format. Host-backed sources "
        "(hostPath, nfs, iscsi, ...) are permitted because GPU benchmarking needs "
        "them, but they are logged as warnings at validation time because they "
        "grant access outside AIPerf's pod hardening.",
    )
    volume_mounts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Volume mounts in K8s VolumeMount format",
    )
    node_selector: dict[str, str] = Field(
        default_factory=dict,
        description="Node selector labels",
    )
    tolerations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod tolerations for scheduling on tainted nodes",
    )
    affinity: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod affinity/anti-affinity rules in K8s Affinity format "
        "(nodeAffinity, podAffinity, podAntiAffinity). Use to co-locate or "
        "separate bench pods from other workloads (e.g. keep benchmark pods "
        "off inference nodes via podAntiAffinity topologyKey=kubernetes.io/hostname).",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod annotations",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod labels",
    )
    image_pull_secrets: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Image pull secrets, K8s LocalObjectReference shape: "
            "`[{name: secretName}, ...]`."
        ),
    )
    service_account_name: str | None = Field(
        default=None,
        description="Service account name for pods. The named ServiceAccount must "
        "exist in the benchmark namespace (preflight fails otherwise), and its "
        "token is projected into every container of every benchmark pod, so any "
        "RBAC bound to it is reachable from benchmark code. Prefer a dedicated, "
        "least-privilege ServiceAccount over a shared or default one. See "
        "docs/kubernetes/rbac-security.md.",
    )
    container_security_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Container securityContext overrides (merged into each container "
        "spec). Privilege-escalating values are rejected: privileged=true, "
        "allowPrivilegeEscalation=true, runAsNonRoot=false, runAsUser=0, runAsGroup=0. "
        "Baseline-widening values are permitted but logged as warnings, because "
        "profiling, GPU, and tracing workloads need them: capabilities.add, a "
        "capabilities.drop that is not ALL, readOnlyRootFilesystem=false, and a "
        "seccompProfile.type other than RuntimeDefault.",
    )
    share_process_namespace: bool = Field(
        default=False,
        description="When true, all containers in the pod share a single PID namespace. "
        "Enables kubectl exec cross-container kills for chaos tests. Keep false in production.",
    )
    priority_class_name: str | None = Field(
        default=None,
        description="Native K8s PriorityClass name for the pod. Distinct from "
        "scheduling.priorityClass, which is the Kueue WorkloadPriorityClass. "
        "Use this to preempt lower-priority pods via the default scheduler.",
    )
    runtime_class_name: str | None = Field(
        default=None,
        description="K8s RuntimeClass name (e.g. 'nvidia' for GPU runtime, "
        "'kata' for sandboxed runtime).",
    )
    scheduler_name: str | None = Field(
        default=None,
        description="Name of the scheduler to dispatch the pod to (e.g. a custom "
        "GPU topology scheduler). Omit to use the default scheduler.",
    )
    topology_spread_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod TopologySpreadConstraints in K8s format. Useful to spread "
        "worker pods evenly across zones/nodes independent of affinity rules.",
    )
    host_aliases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entries appended to the pod's /etc/hosts (K8s HostAlias format: "
        "{ip, hostnames: [...]}). Useful when benchmarking endpoints that aren't in "
        "cluster DNS.",
    )
    dns_policy: Annotated[
        Literal["ClusterFirst", "ClusterFirstWithHostNet", "Default", "None"] | None,
        Field(
            default=None,
            description="Pod DNS policy. Defaults to 'ClusterFirst' (K8s default); "
            "set 'None' to supply dns_config entirely.",
        ),
    ] = None
    dns_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod DNS config (K8s PodDNSConfig format: nameservers, searches, "
        "options). Typically paired with dns_policy='None'.",
    )
    termination_grace_period_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Seconds the kubelet waits for the pod to terminate gracefully "
        "before SIGKILL. Defaults to 30 in K8s; raise for long-running benchmarks "
        "that need extra time to flush artifacts.",
    )
    pod_security_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod-level securityContext (PodSecurityContext format: fsGroup, "
        "runAsUser, runAsNonRoot, supplementalGroups, sysctls, etc.). Distinct from "
        "container_security_context which applies per-container. Privilege-escalating "
        "values are rejected: runAsNonRoot=false, runAsUser=0, runAsGroup=0. "
        "Baseline-widening values (seccompProfile.type other than RuntimeDefault, "
        "etc.) are permitted but logged as warnings.",
    )
    init_containers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="InitContainers that run to completion before the main containers "
        "start. Full K8s Container format. Useful for sysctl tweaks (e.g. bumping "
        "ip_local_port_range), model pre-fetch, or permission fixups. An init "
        "container that does not declare its own securityContext receives AIPerf's "
        "hardened container baseline (minus readOnlyRootFilesystem); one that does "
        "declare a securityContext is passed through verbatim so privileged setup "
        "work stays possible, and is logged as a warning at validation time.",
    )
    extra_pod_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch: raw PodSpec keys merged into the rendered pod spec "
        "last (keys here override typed fields above). Use for K8s PodSpec fields not "
        "yet modeled here (e.g. schedulingGates, resourceClaims, overhead). Typed "
        "fields are preferred when available because preflight checks, env merging, "
        "and securityContext merging only apply to typed fields. Security-critical "
        "keys are NOT overridable and are rejected at validation time: "
        "securityContext, containers, hostNetwork, hostPID, hostIPC, hostUsers. "
        "The renderer drops those keys again before merging, so the hardened pod "
        "securityContext cannot be weakened even by an unvalidated spec.",
    )

    @field_validator("extra_pod_spec")
    @classmethod
    def _reject_denied_extra_pod_spec_keys(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Reject raw PodSpec keys that would bypass AIPerf's pod hardening."""
        denied = sorted(DENIED_EXTRA_POD_SPEC_KEYS & value.keys())
        if denied:
            raise ValueError(
                f"extraPodSpec may not override security-critical PodSpec keys: "
                f"{', '.join(denied)}. Use the typed podTemplate fields instead."
            )
        return value

    @field_validator("volumes")
    @classmethod
    def _warn_on_risky_volumes(
        cls, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Log (never reject) volumes that reach outside the pod sandbox."""
        for warning in risky_volume_warnings(value):
            _logger.warning(warning)
        return value

    @field_validator("init_containers")
    @classmethod
    def _warn_on_risky_init_containers(
        cls, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Log (never reject) init containers that bypass the hardened baseline."""
        for warning in risky_init_container_warnings(value):
            _logger.warning(warning)
        return value

    @field_validator("pod_security_context", "container_security_context")
    @classmethod
    def _reject_weakened_security_context(
        cls, value: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Reject privilege escalation; warn on other baseline-widening keys."""
        field_name = info.field_name or "securityContext"
        _reject_privilege_escalation(value, field_name)
        for warning in risky_security_context_warnings(
            value, f"podTemplate.{to_camel(field_name)}"
        ):
            _logger.warning(warning)
        return value


class DeploymentConfig(BaseConfig):
    """Complete Kubernetes deployment configuration.

    Unifies image settings, pod template, and scheduling into a single model.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Container image for AIPerf",
    )
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent)",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="burstable",
        description="CPU/memory resource mode for controller and worker pods. "
        "'burstable' (default) sets requests only, no limits (Burstable QoS) "
        "so the controller can grow beyond the request during aggregation "
        "without being OOM-killed. "
        "'guaranteed' applies requests==limits (Guaranteed QoS). "
        "'none' omits CPU/memory requests and limits as an escape hatch.",
    )
    connections_per_worker: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent connections each worker handles. "
        "100 keeps the asyncio event loop responsive while amortizing per-process overhead.",
    )
    timeout_seconds: FiniteFloat = Field(
        default=0,
        ge=0,
        description="Job timeout in seconds (0 = no timeout)",
    )
    ttl_seconds_after_finished: int | None = Field(
        default=300,
        ge=0,
        description="TTL after finished (seconds)",
    )
    results_ttl_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Days to retain result files before cleanup",
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )
    cancel: bool = Field(
        default=False,
        description="Set to true to cancel the job",
    )
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig,
        description="Pod template configuration",
    )
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig,
        description="Kueue gang-scheduling configuration. Set "
        "scheduling.queueName to a LocalQueue name to admit this job's "
        "controller + worker pods atomically via Kueue. When unset, the "
        "operator falls back to AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME "
        "(operator-deploy env). Safe to leave unset on clusters without Kueue.",
    )
