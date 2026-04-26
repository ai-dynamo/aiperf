# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.create handler for AIPerfSweep CRs.

Validates spec via AIPerfSweepSpec, computes totalVariations/maxTotalRuns,
provisions RBAC for the sweep-controller pod, and creates the JobSet that
schedules it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import kopf
from pydantic import ValidationError

from aiperf.config.sweep import expand_sweep
from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

logger = logging.getLogger(__name__)


async def handle(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
) -> None:
    """Validate spec, set status, provision RBAC, create sweep-controller JobSet."""
    try:
        validated = AIPerfSweepSpec.model_validate(spec)
    except ValidationError as e:
        raise kopf.PermanentError(f"AIPerfSweep spec invalid: {e}") from e

    base_benchmark = validated.template.spec.get("benchmark") or {}
    if validated.sweep is not None:
        sweep_input = {
            **base_benchmark,
            "sweep": validated.sweep.model_dump(by_alias=True),
        }
    else:
        sweep_input = dict(base_benchmark)
    expanded = expand_sweep(sweep_input)
    n_variations = len(expanded)

    if validated.convergence is not None:
        max_trials = validated.convergence.max_runs
    elif validated.multi_run is not None and validated.multi_run.trials is not None:
        max_trials = validated.multi_run.trials
    else:
        max_trials = 1

    sweep_uid = body["metadata"]["uid"]
    creation_ts = body["metadata"].get("creationTimestamp", "")
    epoch = _epoch_from_creation_ts(creation_ts)

    patch.status["phase"] = "Pending"
    patch.status["totalVariations"] = n_variations
    patch.status["maxTotalRuns"] = n_variations * max_trials
    patch.status["completedRuns"] = 0
    patch.status["failedRuns"] = 0
    patch.status["runEpoch"] = int(epoch) if epoch.isdigit() else 0

    await _provision_rbac(name=name, namespace=namespace, sweep_uid=sweep_uid)
    await _create_sweep_controller_jobset(
        name=name,
        namespace=namespace,
        sweep_uid=sweep_uid,
        epoch=epoch,
        template_spec=validated.template.spec,
    )

    jobset_name = f"aiperf-{name}"
    patch.status["runtimeRef"] = {
        "jobSetName": jobset_name,
        "sweepControllerHost": (
            f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"
        ),
    }
    logger.info(
        f"AIPerfSweep {namespace}/{name} created: {n_variations} variations, "
        f"max {max_trials} trials/cell"
    )


def _epoch_from_creation_ts(ts: str) -> str:
    """Decimal epoch-seconds string from an RFC3339 creationTimestamp."""
    if not ts:
        return "0"
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return str(int(dt.replace(tzinfo=timezone.utc).timestamp()))
    except ValueError:
        return "0"


async def _provision_rbac(*, name: str, namespace: str, sweep_uid: str) -> None:
    """Create namespace-scoped ServiceAccount + Role + RoleBinding for sweep-controller."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    sa_name = f"aiperf-sweep-controller-{name}"
    role_name = sa_name
    owner_ref = k8s.V1OwnerReference(
        api_version="aiperf.nvidia.com/v1",
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
    """Create resource; tolerate 409 'already exists' for idempotent reconcile."""
    from kubernetes_asyncio.client import ApiException

    try:
        await create_fn(namespace, body)
    except ApiException as e:
        if e.status != 409:
            raise


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

    pod_spec: dict[str, Any] = {
        "restartPolicy": "OnFailure",
        "serviceAccountName": sa_name,
        "containers": [container],
        "volumes": [{"name": "results", "emptyDir": {}}],
    }
    # Lift scheduling primitives from the user's template.spec.podTemplate so the
    # sweep-controller pod can land on the same nodes as its child workers will.
    pod_template = template_spec.get("podTemplate") or {}
    for key in ("nodeSelector", "tolerations", "affinity",
                "imagePullSecrets", "priorityClassName", "runtimeClassName"):
        if key in pod_template and pod_template[key] is not None:
            value = pod_template[key]
            if key == "imagePullSecrets" and value and isinstance(value[0], str):
                # CRD takes a list of bare names; native PodSpec wants {name: ...}.
                value = [{"name": s} for s in value]
            pod_spec[key] = value

    jobset_body = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": jobset_name,
            "namespace": namespace,
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1",
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
        if e.status != 409:
            raise
