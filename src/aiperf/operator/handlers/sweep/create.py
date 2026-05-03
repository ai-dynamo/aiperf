# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.create handler for AIPerfSweep CRs.

Validates spec via AIPerfSweepSpec, computes totalVariations/maxTotalRuns,
provisions RBAC for the sweep-controller pod, and creates the JobSet that
schedules it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import kopf
from pydantic import ValidationError

from aiperf.config.sweep import expand_sweep

logger = logging.getLogger(__name__)


async def handle(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Validate spec, set status, provision RBAC, create sweep-controller JobSet."""
    # Lazy-imported because aiperf.kubernetes.sweep_models eagerly imports
    # aiperf.operator.models — the top-level import would cycle through
    # aiperf.operator.__init__ -> main -> handlers.sweep.create.
    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    try:
        validated = AIPerfSweepSpec.model_validate(spec)
    except ValidationError as e:
        raise kopf.PermanentError(f"AIPerfSweep spec invalid: {e}") from e

    base_benchmark = validated.template.spec.benchmark.model_dump(
        by_alias=True, exclude_none=True, exclude_defaults=True
    )
    if validated.sweep is not None:
        sweep_input = {
            **base_benchmark,
            "sweep": validated.sweep.model_dump(by_alias=True),
        }
    else:
        sweep_input = dict(base_benchmark)

    n_variations, max_total_runs = _compute_cardinality(validated, sweep_input)

    sweep_uid = body["metadata"]["uid"]
    creation_ts = body["metadata"].get("creationTimestamp", "")
    epoch = _epoch_from_creation_ts(creation_ts)

    patch.status["phase"] = "Pending"
    patch.status["totalVariations"] = n_variations
    patch.status["maxTotalRuns"] = max_total_runs
    patch.status["completedRuns"] = 0
    patch.status["failedRuns"] = 0
    patch.status["runEpoch"] = int(epoch) if epoch.isdigit() else 0

    await _provision_rbac(name=name, namespace=namespace, sweep_uid=sweep_uid)
    await _create_sweep_controller_jobset(
        name=name,
        namespace=namespace,
        sweep_uid=sweep_uid,
        epoch=epoch,
        template_spec=validated.template.spec.model_dump(
            by_alias=True, exclude_none=True, exclude_defaults=True
        ),
    )

    jobset_name = f"aiperf-{name}"
    patch.status["runtimeRef"] = {
        "jobSetName": jobset_name,
        "sweepControllerHost": (
            f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"
        ),
    }
    generation = body.get("metadata", {}).get("generation")
    if generation is not None:
        patch.status["observedGeneration"] = int(generation)
    logger.info(
        f"AIPerfSweep {namespace}/{name} created: {n_variations} variations, "
        f"max {max_total_runs} total runs"
    )


def _compute_cardinality(
    validated: Any,
    sweep_input: dict[str, Any],
) -> tuple[int, int]:
    """Compute `(totalVariations, maxTotalRuns)` for the create-handler status.

    Adaptive search (Bayesian Optimization) sweeps don't know the final
    variation count up front -- only an upper bound (`max_iterations`).
    Write that bound so dashboards can render a determinate progress bar;
    early convergence routes through the controller pod's terminal-phase
    write, which supersedes any premature rollup-driven Aggregating phase
    via the existing `_conditional_phase_set` test-op guard in
    `child_rollup.py`.

    For non-adaptive sweeps, expand the grid/scenarios input via
    `expand_sweep` and multiply by `max_trials` (derived from convergence
    max_runs or multi_run.trials, defaulting to 1).
    """
    if validated.convergence is not None:
        max_trials = validated.convergence.max_runs
    elif validated.multi_run is not None and validated.multi_run.trials is not None:
        max_trials = validated.multi_run.trials
    else:
        max_trials = 1

    adaptive = (
        validated.multi_run.adaptive_search if validated.multi_run is not None else None
    )
    if adaptive is not None:
        trials = (
            validated.multi_run.trials if validated.multi_run.trials is not None else 1
        )
        return adaptive.max_iterations, adaptive.max_iterations * trials

    expanded = expand_sweep(sweep_input)
    n_variations = len(expanded)
    return n_variations, n_variations * max_trials


def _epoch_from_creation_ts(ts: str) -> str:
    """Decimal epoch-seconds string from an RFC3339 creationTimestamp.

    K8s `metadata.creationTimestamp` is whole-second by convention, but
    other RFC3339 sources (kopf event payloads with sub-second precision,
    JSON-patched timestamps from non-apiserver writers) may include
    fractional seconds. ``strptime("%Y-%m-%dT%H:%M:%SZ")`` rejects those
    and returns ``"0"`` — collapsing every child name onto epoch 0 and
    defeating across-rerun isolation. ``fromisoformat`` accepts both
    forms by normalizing the trailing ``Z`` to ``+00:00``.
    """
    if not ts:
        return "0"
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z") + "+00:00")
        return str(int(dt.timestamp()))
    except ValueError:
        return "0"


async def _provision_rbac(*, name: str, namespace: str, sweep_uid: str) -> None:
    """Create namespace-scoped ServiceAccount + Role + RoleBinding for sweep-controller."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    sa_name = f"aiperf-sweep-controller-{name}"
    role_name = sa_name
    owner_ref = k8s.V1OwnerReference(
        api_version="aiperf.nvidia.com/v1alpha1",
        kind="AIPerfSweep",
        name=name,
        uid=sweep_uid,
        controller=True,
        block_owner_deletion=True,
    )

    async with k8s_client() as api:
        core = k8s.CoreV1Api(api)
        rbac = k8s.RbacAuthorizationV1Api(api)

        sa = k8s.V1ServiceAccount(
            metadata=k8s.V1ObjectMeta(
                name=sa_name,
                namespace=namespace,
                owner_references=[owner_ref],
            )
        )
        await _create_or_skip_409(core.create_namespaced_service_account, namespace, sa)

        role = k8s.V1Role(
            metadata=k8s.V1ObjectMeta(
                name=role_name,
                namespace=namespace,
                owner_references=[owner_ref],
            ),
            rules=[
                k8s.V1PolicyRule(
                    api_groups=["aiperf.nvidia.com"],
                    resources=["aiperfjobs", "aiperfjobs/status"],
                    verbs=[
                        "create",
                        "get",
                        "list",
                        "watch",
                        "patch",
                        "update",
                        "delete",
                    ],
                ),
                k8s.V1PolicyRule(
                    api_groups=["aiperf.nvidia.com"],
                    resources=["aiperfsweeps", "aiperfsweeps/status"],
                    verbs=["get", "patch", "update"],
                    resource_names=[name],
                ),
                # Emit kubectl-visible events on the parent CR (progress,
                # cancellation acks, aggregation phase).
                k8s.V1PolicyRule(
                    api_groups=[""],
                    resources=["events"],
                    verbs=["create", "patch"],
                ),
            ],
        )
        await _create_or_skip_409(rbac.create_namespaced_role, namespace, role)

        binding = k8s.V1RoleBinding(
            metadata=k8s.V1ObjectMeta(
                name=role_name,
                namespace=namespace,
                owner_references=[owner_ref],
            ),
            subjects=[
                k8s.RbacV1Subject(
                    kind="ServiceAccount",
                    name=sa_name,
                    namespace=namespace,
                )
            ],
            role_ref=k8s.V1RoleRef(
                api_group="rbac.authorization.k8s.io",
                kind="Role",
                name=role_name,
            ),
        )
        await _create_or_skip_409(
            rbac.create_namespaced_role_binding,
            namespace,
            binding,
        )


async def _create_or_skip_409(create_fn: Any, namespace: str, body: Any) -> None:
    """Create resource; tolerate 409 'already exists' for idempotent reconcile.

    Transient apiserver failures (ApiException with non-409 status, connection
    errors, timeouts) raise kopf.TemporaryError so kopf retries with backoff
    rather than hammering the apiserver in an unbounded retry loop.
    """
    import aiohttp
    from kubernetes_asyncio.client import ApiException

    try:
        await create_fn(namespace, body)
    except ApiException as e:
        if e.status == 409:
            return
        raise kopf.TemporaryError(
            f"apiserver rejected create ({e.status}): {e.reason}", delay=30
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during create: {e}", delay=30
        ) from e


async def _create_sweep_controller_jobset(
    *,
    name: str,
    namespace: str,
    sweep_uid: str,
    epoch: str,
    template_spec: dict[str, Any],
) -> None:
    """Create a JobSet whose single replica runs `python -m aiperf.sweep_controller.main`."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    image = template_spec.get("image")
    if not image:
        raise kopf.PermanentError("template.spec.image is required")

    jobset_name = f"aiperf-{name}"
    sa_name = f"aiperf-sweep-controller-{name}"

    container = {
        "name": "sweep-controller",
        "image": image,
        "imagePullPolicy": template_spec.get("imagePullPolicy", "IfNotPresent"),
        "command": ["python", "-m", "aiperf.sweep_controller.main"],
        "env": [
            {"name": "AIPERF_SWEEP_NAME", "value": name},
            {"name": "AIPERF_SWEEP_NAMESPACE", "value": namespace},
            {"name": "AIPERF_SWEEP_UID", "value": sweep_uid},
            {"name": "AIPERF_SWEEP_EPOCH", "value": epoch},
        ],
        "volumeMounts": [{"name": "results", "mountPath": "/results"}],
    }
    # Merge user-supplied container env (template.spec.podTemplate.env) so
    # users can pass HTTP_PROXY, HF_HOME, custom log levels, etc. The
    # controller's reserved AIPERF_SWEEP_* vars take precedence on collision.
    pod_template = template_spec.get("podTemplate") or {}
    user_env = pod_template.get("env") or []
    if user_env:
        reserved = {e["name"] for e in container["env"]}
        container["env"].extend(e for e in user_env if e.get("name") not in reserved)
    # Container-level resources/securityContext from podTemplate. Without
    # these, the sweep-controller pod gets no requests/limits (rejected by
    # ResourceQuota on hardened clusters) and no securityContext (rejected
    # by Pod Security Admission baseline/restricted).
    if pod_template.get("resources") is not None:
        container["resources"] = pod_template["resources"]
    if pod_template.get("containerSecurityContext") is not None:
        container["securityContext"] = pod_template["containerSecurityContext"]

    pod_spec: dict[str, Any] = {
        "restartPolicy": "OnFailure",
        "serviceAccountName": sa_name,
        "containers": [container],
        "volumes": [{"name": "results", "emptyDir": {}}],
    }
    # Lift scheduling primitives + pod-level securityContext from the user's
    # template.spec.podTemplate so the sweep-controller pod can land on the
    # same nodes as its child workers will and meets cluster security
    # baselines.
    for key in (
        "nodeSelector",
        "tolerations",
        "affinity",
        "imagePullSecrets",
        "priorityClassName",
        "runtimeClassName",
        "securityContext",
    ):
        if key in pod_template and pod_template[key] is not None:
            pod_spec[key] = pod_template[key]

    jobset_body = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": jobset_name,
            "namespace": namespace,
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": name,
                    "uid": sweep_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "spec": {
            "replicatedJobs": [
                {
                    "name": "controller",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "completions": 1,
                            "parallelism": 1,
                            "template": {"spec": pod_spec},
                        },
                    },
                },
            ],
        },
    }

    async with k8s_client() as api:
        custom = k8s.CustomObjectsApi(api)
        await _create_or_skip_409_custom(
            custom,
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=namespace,
            plural="jobsets",
            body=jobset_body,
        )


async def _create_or_skip_409_custom(
    custom: Any,
    *,
    group: str,
    version: str,
    namespace: str,
    plural: str,
    body: Any,
) -> None:
    import aiohttp
    from kubernetes_asyncio.client import ApiException

    try:
        await custom.create_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            body=body,
        )
    except ApiException as e:
        if e.status == 409:
            return
        raise kopf.TemporaryError(
            f"apiserver rejected JobSet create ({e.status}): {e.reason}", delay=30
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during JobSet create: {e}", delay=30
        ) from e
