# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet specification generation for Kubernetes deployments.

This module generates JobSet YAML for deploying AIPerf as a distributed
benchmark across multiple pods. All resource and port settings are configurable
via environment variables through K8sEnvironment.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig, SchedulingConfig
from aiperf.kubernetes.constants import AIPerfLabels, KueueLabels
from aiperf.kubernetes.cr_refs import JOBSET_API_VERSION
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_helpers import build_runner_volumes
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec, AIPerfReplicatedJobSpec
from aiperf.kubernetes.jobset_urls import (
    JOBSET_FALLBACK_VERSION,
    JOBSET_GITHUB_REPO,
    get_jobset_install_hint,
    get_jobset_manifest_url,
    get_latest_jobset_version,
)

if TYPE_CHECKING:
    from aiperf.kubernetes.jobset_builder import _JobSetManifestBuilder

__all__ = [
    "JOBSET_FALLBACK_VERSION",
    "JOBSET_GITHUB_REPO",
    "AIPerfContainerSpec",
    "AIPerfJobSetSpec",
    "AIPerfReplicatedJobSpec",
    "aggregator_children",
    "aggregator_count",
    "aggregator_dns_name",
    "aggregator_ship_template",
    "aggregator_tier_counts",
    "aggregator_tier_job_name",
    "controller_dns_name",
    "get_jobset_install_hint",
    "get_jobset_manifest_url",
    "get_latest_jobset_version",
]


def controller_dns_name(jobset_name: str, namespace: str) -> str:
    """Build the controller pod DNS hostname for a JobSet.

    JobSet with enableDNSHostnames creates a headless service with the same name
    as the JobSet, and pods get DNS names like:
    {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local

    Since we have exactly 1 controller replica with 1 pod, indices are always 0-0.

    Args:
        jobset_name: The JobSet resource name.
        namespace: Kubernetes namespace.

    Returns:
        Fully qualified DNS hostname for the controller pod.
    """
    return f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"


def aggregator_dns_name(jobset_name: str, namespace: str, agg_id: int) -> str:
    """Build a tier-T2 aggregator pod's DNS hostname for a JobSet.

    The ``aggregators`` replicatedJob is an indexed Job (like ``cells``), so pod
    ``agg_id`` gets the deterministic name
    ``{jobset}-aggregators-{agg_id}-0.{jobset}.{ns}.svc.cluster.local`` under the
    JobSet's ``enableDNSHostnames`` headless service. Cells ship to this coordinate
    (+ the aggregator port); the aggregator ships up to the controller.

    Args:
        jobset_name: The JobSet resource name.
        namespace: Kubernetes namespace.
        agg_id: The aggregator's job-index (``0..M``).

    Returns:
        Fully qualified DNS hostname for aggregator pod ``agg_id``.
    """
    return (
        f"{jobset_name}-aggregators-{agg_id}-0."
        f"{jobset_name}.{namespace}.svc.cluster.local"
    )


def aggregator_tier_job_name(tier_index: int, num_tiers: int) -> str:
    """The ``aggregators`` replicatedJob name for tier ``tier_index`` of a tree of
    ``num_tiers`` tiers.

    A single-tier tree keeps the bare ``aggregators`` name so its manifest stays
    byte-identical to the pre-multi-tier operator. A multi-tier tree names each tier
    ``aggregators-{k}`` 1-indexed from the tier the cells ship to (tier 1) up to the top
    tier that ships to the controller, so each tier is its own indexed replicatedJob with
    a distinct headless-service DNS prefix.
    """
    if num_tiers <= 1:
        return "aggregators"
    return f"aggregators-{tier_index + 1}"


def aggregator_ship_template(
    jobset_name: str, namespace: str, tier_job_name: str = "aggregators"
) -> str:
    """The ship-DNS template a child fills with its round-robin parent id.

    A JobSet indexed replicatedJob shares one env template, so the operator gives every
    child the same concrete ``tcp://…svc.cluster.local:PORT`` coordinate with a single
    ``{agg_id}`` placeholder (jobset/namespace/tier resolved here); each child substitutes
    its own round-robin parent id pod-side (rust ``cellular_cell::ship_target`` for a
    cell's ``cell_id % M``; ``cellular_aggregator::k8s_tier_parent_id`` for a lower-tier
    aggregator's ``agg_id % parent_count``). ``tier_job_name`` selects which tier's pods
    the template points at — the tier-1 ``aggregators``/``aggregators-1`` job for the
    cells, or a parent-tier job for a lower aggregator tier. Set on the cells (to ship),
    the controller (its *presence* is the k8s expect-don't-spawn gate), and each
    lower-tier aggregator (to ship up).
    """
    from aiperf.kubernetes.jobset_helpers import AGGREGATOR_PORT

    return (
        f"tcp://{jobset_name}-{tier_job_name}-{{agg_id}}-0."
        f"{jobset_name}.{namespace}.svc.cluster.local:{AGGREGATOR_PORT}"
    )


def aggregator_count(cells: int, fanout: int | None) -> int | None:
    """Number of tier-T2 aggregators for ``cells`` at ``fanout``, or ``None`` for flat.

    Mirrors the Rust ``cellular_aggregator::aggregator_count`` gate exactly so the
    operator and the controller never disagree on M: unset ``fanout``, ``< 1``, or
    ``>= cells`` keeps the flat star (one aggregator per cell or fewer is pointless);
    otherwise M = ``ceil(cells / fanout)``.
    """
    if fanout is None or fanout < 1 or fanout >= cells:
        return None
    return -(-cells // fanout)  # ceil division


def aggregator_children(agg_id: int, agg_count: int, cells: int) -> int:
    """Cells assigned to aggregator ``agg_id`` under round-robin (``cell_id % M``).

    Mirrors Rust ``cellular_aggregator::children_of`` exactly (the aggregator's
    collect barrier must match what the operator sizes): ``ceil((cells - agg_id) /
    agg_count)``, or 0 when ``agg_id >= cells``.
    """
    if agg_id >= cells:
        return 0
    return -(-(cells - agg_id) // agg_count)  # ceil division


def aggregator_tier_counts(cells: int, fanout: int | None) -> list[int]:
    """Aggregator node counts per tier for ``cells`` reduced by ``fanout``.

    Mirrors Rust ``cellular_aggregator::tier_counts`` exactly. Returns the node count
    of each aggregator tier from tier 1 (the tier the cells ship to) up to the top tier
    that ships to the controller, or ``[]`` for the flat star (``fanout`` unset, ``< 2``,
    or ``>= cells``). Each tier reduces the one below by ``ceil(prev / fanout)``,
    stopping once a tier is ``<= fanout``. The first element equals
    :func:`aggregator_count`, so a length-1 plan is the original single-tier tree.

    :meth:`AIPerfJobSetSpec.to_k8s_manifest` (via
    ``_JobSetManifestBuilder.build_aggregator_tier_jobs``) realizes this plan on
    Kubernetes: one indexed ``aggregators-{tier}`` replicatedJob per element, each tier's
    pods deriving their per-pod round-robin barrier runner-side and, for a lower tier,
    shipping up to their parent tier via an ``AIPERF_AGG_SHIP_ADDR`` DNS template —
    mirroring the same-host ``aggregator_nodes`` wiring. The single-tier plan keeps the
    bare ``aggregators`` job so its manifest is byte-identical to the flat-tree layout.
    """
    if fanout is None or fanout < 2 or fanout >= cells:
        return []
    tiers: list[int] = []
    prev = cells
    while True:
        count = -(-prev // fanout)  # ceil division
        tiers.append(count)
        if count <= fanout:
            break
        prev = count
    return tiers


class AIPerfJobSetSpec(AIPerfBaseModel):
    """Specification for a complete JobSet deployment.

    Resource settings, ports, and health probe configuration are loaded from
    K8sEnvironment and can be customized via AIPERF_K8S_* environment variables.
    """

    name: str = Field(description="JobSet name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    job_id: str = Field(description="Unique benchmark job ID")
    image: str = Field(description="AIPerf container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy for all containers (Always, Never, IfNotPresent). "
        "Set to 'Never' for local development with minikube.",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="burstable",
        description="CPU/memory resource mode for controller and worker pods. "
        "'burstable' (default) emits requests only (no limits) so the controller "
        "can grow beyond the request during aggregation without being OOM-killed. "
        "'guaranteed' emits requests==limits. "
        "'none' omits the resources block.",
    )
    cells: int = Field(
        default=1,
        ge=1,
        description="Number of native cellular cell pods for a cross-pod run. Each "
        "cell is one aiperf runner slice over a (cell_id, cell_count) budget "
        "partition; the controller pod merges their shards. cells=1 is a "
        "single-cell run (still the cellular topology, one cell pod).",
    )
    cell_agg_fanout: int | None = Field(
        default=None,
        ge=1,
        description="Tier-T2 aggregator fan-out: the max cells one aggregator pod "
        "collects. When set and < cells, the operator inserts an `aggregators` "
        "replicatedJob of M=ceil(cells/fanout) pods between the cells and the "
        "controller (each cell ships its folded store to its round-robin aggregator; "
        "each aggregator merges its subtree and ships one store up), lifting the "
        "single-controller fan-in ceiling. Unset or >= cells keeps the flat star. "
        "Fold-only (sketch or exact-fold); the byte-exact retain path stays flat.",
    )
    barrier_free: bool = Field(
        default=False,
        description="Tier-T3 master-less start: the controller triggers START "
        "immediately instead of waiting for all N cell registrations (the O(N) "
        "rendezvous is a scale limit at high cell counts). Aggregate-equivalent to "
        "the synchronized start on data-deterministic metrics.",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod (used for resource calculation). "
        "Defaults to Environment.WORKER.DEFAULT_WORKERS_PER_POD if not set.",
    )
    record_processors_per_pod: int | None = Field(
        default=None,
        description="Actual record processors per worker pod. "
        "Defaults to a Kubernetes scale factor derived from workers_per_pod.",
    )
    ttl_seconds: int | None = Field(
        default=None, description="TTL after finished (uses K8sEnvironment default)"
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )

    # Pod template
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig, description="Pod template configuration"
    )

    # Scheduling
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig, description="Kueue scheduling configuration"
    )
    gpu_telemetry_enabled: bool = Field(
        default=True,
        description="Whether to include the GPU telemetry manager container.",
    )
    server_metrics_enabled: bool = Field(
        default=True,
        description="Whether to include the server metrics manager container.",
    )

    # Optional metadata for discovery
    name_label: str | None = Field(
        default=None, description="Human-readable name label for the JobSet"
    )
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations for the JobSet metadata",
    )

    def _build_manifest_labels(self) -> dict[str, str]:
        """Build top-level JobSet labels (AIPerf, name, Kueue scheduling).

        Kueue queue-name and priority-class fall back to the operator-side
        defaults (`AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` /
        `_PRIORITY_CLASS`) when not set on the CR. This makes Kueue gang-
        scheduling default-on for clusters that have Kueue installed and a
        named LocalQueue, without forcing per-CR opt-in.
        """
        from aiperf.kubernetes.environment import K8sEnvironment

        labels: dict[str, str] = {
            AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
            AIPerfLabels.JOB_ID: self.job_id,
        }
        if self.name_label:
            labels[AIPerfLabels.NAME] = self.name_label
        queue_name = (
            self.scheduling.queue_name or K8sEnvironment.JOBSET.KUEUE_DEFAULT_QUEUE_NAME
        )
        if queue_name:
            labels[KueueLabels.QUEUE_NAME] = queue_name
        priority_class = (
            self.scheduling.priority_class
            or K8sEnvironment.JOBSET.KUEUE_DEFAULT_PRIORITY_CLASS
        )
        if priority_class:
            labels[KueueLabels.PRIORITY_CLASS] = priority_class
        return labels

    def _resolve_manifest_ttl(self) -> int | None:
        """Resolve the top-level JobSet ttlSecondsAfterFinished value, if any."""
        if self.keep_failed_pods:
            return None
        if self.ttl_seconds is not None:
            return self.ttl_seconds
        return K8sEnvironment.JOBSET.TTL_SECONDS_AFTER_FINISHED

    # ------------------------------------------------------------------
    # Thin delegating wrappers for the internal builder. Kept so tests and
    # callers can keep poking at ``_create_*``/``_get_*`` private helpers
    # without reaching into ``jobset_builder`` directly. The implementations
    # live in :mod:`aiperf.kubernetes.jobset_helpers` and
    # :mod:`aiperf.kubernetes.jobset_builder`.
    # ------------------------------------------------------------------

    def _builder(self) -> "_JobSetManifestBuilder":
        from aiperf.kubernetes.jobset_builder import _JobSetManifestBuilder

        return _JobSetManifestBuilder(self)

    def _create_security_context(self) -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_security_context

        return build_security_context(self.pod_template)

    def _create_health_probe(self, port: int, path: str = "/healthz") -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_health_probe

        return build_health_probe(port, path)

    def _create_startup_probe(
        self, port: int, path: str = "/healthz"
    ) -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_startup_probe

        return build_startup_probe(port, path)

    def _create_env_vars(
        self,
        controller_host: str | None = None,
        include_pod_index: bool = True,
        controller_pod: bool = False,
    ) -> list[dict[str, Any]]:
        from aiperf.kubernetes.jobset_helpers import build_env_vars

        return build_env_vars(
            job_id=self.job_id,
            namespace=self.namespace,
            pod_template=self.pod_template,
            controller_host=controller_host,
            include_pod_index=include_pod_index,
            controller_pod=controller_pod,
        )

    def _get_volume_mounts(self) -> list[dict[str, Any]]:
        from aiperf.kubernetes.jobset_helpers import build_volume_mounts

        return build_volume_mounts(self.pod_template)

    def _create_container(self, *args: Any, **kwargs: Any) -> AIPerfContainerSpec:
        return self._builder()._create_container(*args, **kwargs)

    def _split_worker_pod_resources(
        self,
        worker_count: int,
        record_processor_count: int,
    ) -> list[dict[str, dict[str, str]] | None]:
        return self._builder()._split_worker_pod_resources(
            worker_count, record_processor_count
        )

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the complete JobSet Kubernetes manifest."""
        builder = self._builder()
        controller_dns = controller_dns_name(self.name, self.namespace)
        volumes = build_runner_volumes(self.name, self.pod_template)

        # Aggregator reduction tree: one indexed `aggregators-{tier}` replicatedJob per
        # tier of the `tier_counts` plan between the cells and the controller (fold-only;
        # empty keeps the flat star). Cells ship to their round-robin tier-1 aggregator via
        # the DNS template; each tier merges its subtree and ships one store up (to its
        # parent-tier aggregator for a lower tier, the controller for the top tier). A
        # single-tier plan is byte-identical to the pre-multi-tier topology.
        agg_tiers = aggregator_tier_counts(self.cells, self.cell_agg_fanout)
        agg_fanout = self.cell_agg_fanout if agg_tiers else None
        # The cells ship to tier 1 (the first tier the plan reduces the cells into).
        agg_ship_template = (
            aggregator_ship_template(
                self.name,
                self.namespace,
                aggregator_tier_job_name(0, len(agg_tiers)),
            )
            if agg_tiers
            else None
        )

        # Native cross-pod cellular topology: one aiperf runner controller pod that
        # binds the cell transport + merges shards, and `cells` cell pods that each
        # run an aiperf runner budget slice and stream their shard back. Replaces the
        # retired Python service mesh (controller-of-services + worker pods over ZMQ).
        controller_job = builder.build_cellular_controller_replicated_job(
            volumes,
            agg_fanout=agg_fanout,
            agg_ship_template=agg_ship_template,
            barrier_free=self.barrier_free,
        )
        cells_job = builder.build_cell_replicated_job(
            volumes,
            controller_dns,
            agg_fanout=agg_fanout,
            agg_ship_template=agg_ship_template,
        )

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": self._build_manifest_labels(),
        }
        if self.extra_annotations:
            metadata["annotations"] = self.extra_annotations

        manifest: dict[str, Any] = {
            "apiVersion": JOBSET_API_VERSION,
            "kind": "JobSet",
            "metadata": metadata,
            "spec": {
                # Enable DNS hostnames for pod-to-pod communication
                # This creates a headless service with the same name as the JobSet,
                # allowing pods to have DNS names like:
                # {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local
                "network": {
                    "enableDNSHostnames": True,
                },
                "successPolicy": {
                    "operator": "All",
                    "targetReplicatedJobs": ["controller"],
                },
                "replicatedJobs": [
                    controller_job.to_k8s_spec(),
                    cells_job.to_k8s_spec(),
                    *(
                        job.to_k8s_spec()
                        for job in builder.build_aggregator_tier_jobs(
                            volumes,
                            controller_dns,
                            tiers=agg_tiers,
                            fanout=self.cell_agg_fanout,
                        )
                    ),
                ],
            },
        }

        # Kueue requires JobSets to start suspended; it unsuspends after admission
        if self.scheduling.queue_name:
            manifest["spec"]["suspend"] = True

        ttl = self._resolve_manifest_ttl()
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
