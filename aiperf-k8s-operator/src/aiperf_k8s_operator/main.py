# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kopf entry point for isolated native-k8s/v1 reconciliation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kopf
import uvicorn
from kubernetes_asyncio import client
from kubernetes_asyncio import config as kubernetes_config
from kubernetes_asyncio.client.exceptions import ApiException

from .api import create_app
from .contract import (
    ControllerEnvelope,
    validate_envelope,
    validate_sweep_envelope,
)
from .reconciliation import (
    build_config_snapshot,
    build_jobset,
    build_sweep_config_snapshot,
    build_sweep_jobset,
    build_sweep_workload_identity,
    build_workload_identity,
    config_snapshot_name,
    envelope_sha256,
    submitted_status,
    sweep_submitted_status,
    validate_jobset_identity,
    validate_references,
    workload_name,
)
from .results import ResultsIndex
from .settings import OperatorSettings

GROUP = "aiperf.nvidia.com"
VERSION = "v1alpha1"
PLURAL = "aiperfjobs"
SWEEP_PLURAL = "aiperfsweeps"


@dataclass(frozen=True)
class ReferenceMaterial:
    """Validated public bootstrap metadata."""

    metadata_by_name: dict[str, dict[str, Any]]


class KubernetesResultsLifecycle:
    """Best-effort status publication for durably stored results."""

    def __init__(self, custom_objects: Any) -> None:
        self._custom_objects = custom_objects

    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str
    ) -> None:
        """Publish completion only after the durable manifest is readable."""
        resource = await self._custom_objects.get_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job_id,
        )
        envelope = validate_envelope(resource["spec"]["envelope"])
        if (
            namespace != envelope.namespace
            or job_id != envelope.job_id
            or run_id != envelope.run_id
        ):
            raise ValueError("AIPerfJob identity does not match published results")
        object_uid = _validate_current_job(envelope, None, resource)
        status = resource.get("status")
        if (
            not isinstance(status, dict)
            or status.get("phase") != "PublishingResults"
            or status.get("runId") != run_id
            or status.get("jobSet") != job_id
        ):
            raise ValueError(
                "AIPerfJob must be PublishingResults before result publication"
            )
        await self._custom_objects.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job_id,
            body={
                "metadata": {"uid": object_uid},
                "status": {
                    "phase": "Completed",
                    "resultsReady": True,
                    "runId": run_id,
                },
            },
        )


async def _ensure_pending_status(
    envelope: ControllerEnvelope, object_uid: str, custom_objects: Any
) -> bool:
    resource = await custom_objects.get_namespaced_custom_object(
        group=GROUP,
        version=VERSION,
        namespace=envelope.namespace,
        plural=PLURAL,
        name=envelope.job_id,
    )
    _validate_current_job(envelope, object_uid, resource)
    status = resource.get("status")
    if status is None:
        await custom_objects.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=envelope.namespace,
            plural=PLURAL,
            name=envelope.job_id,
            body={
                "metadata": {"uid": object_uid},
                "status": submitted_status(envelope),
            },
        )
        return True
    if not isinstance(status, dict):
        raise ValueError("AIPerfJob status is not an object")
    if (
        status.get("runId") != envelope.run_id
        or status.get("jobSet") != envelope.job_id
    ):
        raise ValueError("AIPerfJob status identity changed during reconciliation")
    phase = status.get("phase")
    if phase in {"Completed", "Failed"}:
        return False
    if phase not in {"Pending", "PublishingResults"}:
        raise ValueError("AIPerfJob status phase is not recognized")
    return True


def _serialized(resource: Any, api_client: Any | None = None) -> dict[str, Any]:
    if isinstance(resource, dict):
        return resource
    if api_client is None:
        raise ValueError("serializing a Kubernetes model requires an active ApiClient")
    serialized = api_client.sanitize_for_serialization(resource)
    if not isinstance(serialized, dict):
        raise ValueError("existing Kubernetes resource is not an object")
    return serialized


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _matches_config_snapshot(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_metadata = actual.get("metadata")
    expected_metadata = expected["metadata"]
    return (
        isinstance(actual_metadata, dict)
        and actual.get("apiVersion") == expected.get("apiVersion")
        and actual.get("kind") == expected.get("kind")
        and actual.get("immutable") is True
        and actual.get("data", {}) == expected.get("data", {})
        and actual.get("binaryData", {}) == expected.get("binaryData", {})
        and actual_metadata.get("name") == expected_metadata.get("name")
        and actual_metadata.get("namespace") == expected_metadata.get("namespace")
        and actual_metadata.get("labels") == expected_metadata.get("labels")
        and actual_metadata.get("annotations") == expected_metadata.get("annotations")
        and actual_metadata.get("ownerReferences")
        == expected_metadata.get("ownerReferences")
    )


async def _ensure_config_snapshot(
    envelope: ControllerEnvelope,
    object_uid: str,
    core: Any,
    api_client: Any | None,
) -> None:
    source = _serialized(
        await core.read_namespaced_config_map(
            name=envelope.config_ref.name,
            namespace=envelope.namespace,
        ),
        api_client,
    )
    desired = build_config_snapshot(envelope, object_uid, source)
    try:
        await core.create_namespaced_config_map(
            namespace=envelope.namespace, body=desired
        )
    except ApiException as error:
        if error.status != 409:
            raise
        existing = _serialized(
            await core.read_namespaced_config_map(
                name=desired["metadata"]["name"],
                namespace=envelope.namespace,
            ),
            api_client,
        )
        if not _matches_config_snapshot(existing, desired):
            raise ValueError(
                "existing configuration snapshot does not match AIPerfJob incarnation"
            ) from error


async def _ensure_namespaced_resource(
    desired: dict[str, Any],
    description: str,
    create: Callable[..., Awaitable[Any]],
    read: Callable[..., Awaitable[Any]],
    api_client: Any | None = None,
) -> None:
    namespace = desired["metadata"]["namespace"]
    name = desired["metadata"]["name"]
    try:
        await create(namespace=namespace, body=desired)
    except ApiException as error:
        if error.status != 409:
            raise
        existing = _serialized(await read(name=name, namespace=namespace), api_client)
        if not _contains(existing, desired):
            raise ValueError(
                f"existing workload {description} does not match submitted envelope"
            ) from error


async def ensure_workload_identity(
    envelope: ControllerEnvelope,
    object_uid: str,
    core: Any,
    rbac: Any,
    api_client: Any | None = None,
) -> None:
    """Create or validate one least-authority identity for this exact run."""
    service_account, role, binding = build_workload_identity(envelope, object_uid)
    await _ensure_namespaced_resource(
        service_account,
        "ServiceAccount",
        core.create_namespaced_service_account,
        core.read_namespaced_service_account,
        api_client,
    )
    await _ensure_namespaced_resource(
        role,
        "Role",
        rbac.create_namespaced_role,
        rbac.read_namespaced_role,
        api_client,
    )
    await _ensure_namespaced_resource(
        binding,
        "RoleBinding",
        rbac.create_namespaced_role_binding,
        rbac.read_namespaced_role_binding,
        api_client,
    )


async def reconcile_job(
    envelope: ControllerEnvelope,
    jobsets: Any,
    metadata_by_name: dict[str, dict[str, Any]],
    results_upload_base_url: str,
    object_uid: str,
) -> dict[str, Any]:
    """Create the exact immutable JobSet projection for one accepted envelope."""
    validate_references(envelope, metadata_by_name, object_uid)
    jobset = build_jobset(envelope, results_upload_base_url, object_uid)
    try:
        await jobsets.create_namespaced_custom_object(
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=envelope.namespace,
            plural="jobsets",
            body=jobset,
        )
    except ApiException as error:
        if error.status != 409:
            raise
        existing = await jobsets.get_namespaced_custom_object(
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=envelope.namespace,
            plural="jobsets",
            name=envelope.job_id,
        )
        validate_jobset_identity(jobset, existing)
    return submitted_status(envelope)


async def _reference_metadata(
    envelope: ControllerEnvelope, secrets: Any
) -> ReferenceMaterial:
    references = [
        *(
            role.bootstrap
            for role in envelope.roles
            if role.name != "cell" and role.bootstrap is not None
        ),
        *envelope.cell_bootstraps,
    ]
    metadata_by_name: dict[str, dict[str, Any]] = {}
    for reference in references:
        secret = await secrets.read_namespaced_secret(
            name=reference.secret_name,
            namespace=envelope.namespace,
        )
        metadata = secret.metadata
        metadata_by_name[reference.secret_name] = {
            "immutable": secret.immutable,
            "metadata": {
                "name": metadata.name,
                "namespace": metadata.namespace,
                "labels": metadata.labels or {},
                "annotations": metadata.annotations or {},
                "ownerReferences": [
                    {
                        "apiVersion": owner.api_version,
                        "kind": owner.kind,
                        "name": owner.name,
                        "uid": owner.uid,
                        "controller": owner.controller,
                        **(
                            {"blockOwnerDeletion": owner.block_owner_deletion}
                            if owner.block_owner_deletion is not None
                            else {}
                        ),
                    }
                    for owner in (metadata.owner_references or [])
                ],
            },
        }
    return ReferenceMaterial(metadata_by_name)


def _validate_current_job(
    envelope: ControllerEnvelope,
    expected_uid: str | None,
    resource: dict[str, Any],
) -> str:
    metadata = resource.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("AIPerfJob metadata is missing during revalidation")
    object_uid = metadata.get("uid")
    if (
        not isinstance(object_uid, str)
        or not object_uid
        or (expected_uid is not None and object_uid != expected_uid)
        or metadata.get("name") != envelope.job_id
        or metadata.get("namespace") != envelope.namespace
        or metadata.get("deletionTimestamp") is not None
    ):
        raise ValueError("AIPerfJob incarnation changed during reconciliation")
    try:
        current = validate_envelope(resource["spec"]["envelope"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("AIPerfJob envelope changed during reconciliation") from error
    if current != envelope:
        raise ValueError("AIPerfJob envelope changed during reconciliation")
    return object_uid


@kopf.on.create(GROUP, VERSION, PLURAL)
async def create_job(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    **_: Any,
) -> None:
    """Validate a submitted envelope and create its immutable JobSet."""
    envelope = validate_envelope(spec["envelope"])
    if envelope.job_id != name or envelope.namespace != namespace:
        raise ValueError("AIPerfJob metadata does not match envelope identity")
    settings = OperatorSettings()
    async with client.ApiClient() as api_client:
        core = client.CoreV1Api(api_client)
        rbac = client.RbacAuthorizationV1Api(api_client)
        custom_objects = client.CustomObjectsApi(api_client)
        references = await _reference_metadata(envelope, core)
        validate_references(envelope, references.metadata_by_name, uid)
        if not await _ensure_pending_status(envelope, uid, custom_objects):
            return
        await _ensure_config_snapshot(envelope, uid, core, api_client)
        await ensure_workload_identity(envelope, uid, core, rbac, api_client)
        await reconcile_job(
            envelope,
            custom_objects,
            references.metadata_by_name,
            settings.results_upload_base_url,
            uid,
        )
        resource = await custom_objects.get_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=name,
        )
        _validate_current_job(envelope, uid, resource)
        current_references = await _reference_metadata(envelope, core)
        validate_references(envelope, current_references.metadata_by_name, uid)


def _failed_jobset_owner(body: Mapping[str, Any]) -> tuple[str, str, str] | None:
    status = body.get("status")
    metadata = body.get("metadata")
    if (
        not isinstance(status, Mapping)
        or status.get("terminalState") != "Failed"
        or not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("name"), str)
        or not isinstance(metadata.get("namespace"), str)
    ):
        return None
    owners = metadata.get("ownerReferences")
    if not isinstance(owners, list):
        return None
    controlling = [
        owner
        for owner in owners
        if isinstance(owner, Mapping) and owner.get("controller") is True
    ]
    if len(controlling) != 1:
        return None
    owner = controlling[0]
    if (
        owner.get("apiVersion") != f"{GROUP}/{VERSION}"
        or owner.get("kind") != "AIPerfJob"
        or owner.get("name") != metadata["name"]
        or not isinstance(owner.get("uid"), str)
    ):
        return None
    return metadata["namespace"], metadata["name"], owner["uid"]


@kopf.on.event(
    "jobset.x-k8s.io",
    "v1alpha2",
    "jobsets",
    labels={"aiperf.nvidia.com/role": "jobset"},
)
async def observe_jobset(event: dict[str, Any], **_: Any) -> None:
    """Make a failed JobSet terminal on its exact owning AIPerfJob."""
    if event.get("type") == "DELETED":
        return
    body = event.get("object")
    if not isinstance(body, Mapping):
        return
    identity = _failed_jobset_owner(body)
    if identity is None:
        return
    namespace, job_id, object_uid = identity
    async with client.ApiClient() as api_client:
        custom_objects = client.CustomObjectsApi(api_client)
        try:
            resource = await custom_objects.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=job_id,
            )
        except ApiException as error:
            if error.status == 404:
                return
            raise
        envelope = validate_envelope(resource["spec"]["envelope"])
        try:
            _validate_current_job(envelope, object_uid, resource)
        except ValueError:
            return
        metadata = body["metadata"]
        labels = metadata.get("labels")
        annotations = metadata.get("annotations")
        if (
            not isinstance(labels, Mapping)
            or labels.get("aiperf.nvidia.com/role") != "jobset"
            or labels.get("aiperf.nvidia.com/run-id") != envelope.run_id
            or not isinstance(annotations, Mapping)
            or annotations.get("aiperf.nvidia.com/sha256") != envelope_sha256(envelope)
        ):
            return
        status = resource.get("status")
        if not isinstance(status, dict) or status.get("phase") not in {
            "Pending",
            "PublishingResults",
        }:
            return
        if (
            status.get("runId") != envelope.run_id
            or status.get("jobSet") != envelope.job_id
        ):
            return
        await custom_objects.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job_id,
            body={
                "metadata": {"uid": object_uid},
                "status": {
                    "phase": "Failed",
                    "runId": envelope.run_id,
                    "jobSet": envelope.job_id,
                },
            },
        )


async def _delete_if_present(
    delete: Callable[..., Awaitable[Any]], **kwargs: Any
) -> None:
    try:
        await delete(**kwargs)
    except ApiException as error:
        if error.status != 404:
            raise


@kopf.on.delete(GROUP, VERSION, PLURAL)
async def delete_job(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    **_: Any,
) -> None:
    """Delete workload resources without releasing PVC-retained results."""
    envelope = validate_envelope(spec["envelope"])
    if envelope.job_id != name or envelope.namespace != namespace or not uid:
        raise ValueError("AIPerfJob deletion identity does not match its envelope")
    identity = workload_name(envelope)
    async with client.ApiClient() as api_client:
        core = client.CoreV1Api(api_client)
        rbac = client.RbacAuthorizationV1Api(api_client)
        custom_objects = client.CustomObjectsApi(api_client)
        await _delete_if_present(
            custom_objects.delete_namespaced_custom_object,
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=namespace,
            plural="jobsets",
            name=name,
            body={"propagationPolicy": "Foreground"},
        )
        await _delete_if_present(
            core.delete_namespaced_service_account,
            name=identity,
            namespace=namespace,
            propagation_policy="Foreground",
        )
        await _delete_if_present(
            rbac.delete_namespaced_role,
            name=identity,
            namespace=namespace,
            propagation_policy="Foreground",
        )
        await _delete_if_present(
            rbac.delete_namespaced_role_binding,
            name=identity,
            namespace=namespace,
            propagation_policy="Foreground",
        )
        await _delete_if_present(
            core.delete_namespaced_config_map,
            name=config_snapshot_name(envelope, uid),
            namespace=namespace,
            propagation_policy="Foreground",
        )
        bootstrap_names = [
            *(
                role.bootstrap.secret_name
                for role in envelope.roles
                if role.bootstrap is not None
            ),
            *(bootstrap.secret_name for bootstrap in envelope.cell_bootstraps),
        ]
        for bootstrap_name in bootstrap_names:
            await _delete_if_present(
                core.delete_namespaced_secret,
                name=bootstrap_name,
                namespace=namespace,
                propagation_policy="Foreground",
            )


@kopf.on.create(GROUP, VERSION, SWEEP_PLURAL)
async def create_sweep(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    **_: Any,
) -> None:
    """Validate a submitted sweep envelope and create its sweep-controller JobSet.

    No sweep logic enters the operator.  Axis expansion, child-run scheduling,
    and result aggregation are entirely the sweep-controller binary's
    responsibility (Task 9).  The operator's Python delta here is one handler,
    one RBAC template, and one status initialisation.
    """
    envelope = validate_sweep_envelope(spec["sweepEnvelope"])
    if envelope.run_id != name or envelope.namespace != namespace:
        raise ValueError("AIPerfSweep metadata does not match envelope identity")
    async with client.ApiClient() as api_client:
        core = client.CoreV1Api(api_client)
        rbac = client.RbacAuthorizationV1Api(api_client)
        custom_objects = client.CustomObjectsApi(api_client)

        # Set initial Pending status if not already present (idempotent on retry).
        resource = await custom_objects.get_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=SWEEP_PLURAL,
            name=name,
        )
        if resource.get("status") is None:
            await custom_objects.patch_namespaced_custom_object_status(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=SWEEP_PLURAL,
                name=name,
                body={
                    "metadata": {"uid": uid},
                    "status": sweep_submitted_status(envelope),
                },
            )

        # Provision the sweep envelope as an immutable ConfigMap so the
        # sweep-controller can mount it at /etc/aiperf/config without reading
        # the base config (the operator never interprets base_config content).
        snapshot = build_sweep_config_snapshot(envelope, uid)
        try:
            await core.create_namespaced_config_map(namespace=namespace, body=snapshot)
        except ApiException as error:
            if error.status != 409:
                raise

        # Provision per-sweep RBAC.
        service_account, role, binding = build_sweep_workload_identity(envelope, uid)
        await _ensure_namespaced_resource(
            service_account,
            "ServiceAccount",
            core.create_namespaced_service_account,
            core.read_namespaced_service_account,
            api_client,
        )
        await _ensure_namespaced_resource(
            role,
            "Role",
            rbac.create_namespaced_role,
            rbac.read_namespaced_role,
            api_client,
        )
        await _ensure_namespaced_resource(
            binding,
            "RoleBinding",
            rbac.create_namespaced_role_binding,
            rbac.read_namespaced_role_binding,
            api_client,
        )

        # Materialise the single sweep-controller JobSet.
        jobset = build_sweep_jobset(envelope, uid)
        try:
            await custom_objects.create_namespaced_custom_object(
                group="jobset.x-k8s.io",
                version="v1alpha2",
                namespace=namespace,
                plural="jobsets",
                body=jobset,
            )
        except ApiException as error:
            if error.status != 409:
                raise


@kopf.on.event(GROUP, VERSION, PLURAL)
async def observe_child_run(event: dict[str, Any], **_: Any) -> None:
    """Roll child AIPerfJob phases into the owning AIPerfSweep status.

    The sweep-controller binary creates child AIPerfJob resources and sets an
    owner reference pointing to the AIPerfSweep.  This handler detects those
    owner references, fetches the parent sweep, and updates status.childRuns,
    status.completedRuns, and status.failedRuns.
    """
    if event.get("type") == "DELETED":
        return
    body = event.get("object")
    if not isinstance(body, Mapping):
        return
    metadata = body.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    owners = metadata.get("ownerReferences")
    if not isinstance(owners, list):
        return
    sweep_owners = [
        owner
        for owner in owners
        if isinstance(owner, Mapping)
        and owner.get("apiVersion") == f"{GROUP}/{VERSION}"
        and owner.get("kind") == "AIPerfSweep"
        and owner.get("controller") is True
    ]
    if not sweep_owners:
        return
    sweep_owner = sweep_owners[0]
    sweep_name = sweep_owner.get("name")
    sweep_uid = sweep_owner.get("uid")
    job_namespace = metadata.get("namespace")
    job_name = metadata.get("name")
    if (
        not isinstance(sweep_name, str)
        or not isinstance(job_namespace, str)
        or not isinstance(job_name, str)
    ):
        return

    job_status = body.get("status")
    phase = "Unknown"
    if isinstance(job_status, Mapping):
        phase = str(job_status.get("phase", "Unknown"))

    async with client.ApiClient() as api_client:
        custom_objects = client.CustomObjectsApi(api_client)
        try:
            sweep = await custom_objects.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=job_namespace,
                plural=SWEEP_PLURAL,
                name=sweep_name,
            )
        except ApiException as error:
            if error.status == 404:
                return
            raise

        sweep_meta = sweep.get("metadata") or {}
        if sweep_meta.get("uid") != sweep_uid:
            return

        sweep_status = sweep.get("status") or {}
        child_runs: list[dict[str, Any]] = list(sweep_status.get("childRuns") or [])

        job_ref = f"{job_namespace}/{job_name}"
        existing_idx = next(
            (i for i, run in enumerate(child_runs) if run.get("name") == job_name),
            None,
        )
        entry: dict[str, Any] = {
            "name": job_name,
            "jobRef": job_ref,
            "phase": phase,
        }
        if existing_idx is not None:
            child_runs[existing_idx] = entry
        else:
            child_runs.append(entry)

        completed_runs = sum(1 for run in child_runs if run.get("phase") == "Completed")
        failed_runs = sum(1 for run in child_runs if run.get("phase") == "Failed")

        # This is a read-modify-write of the whole childRuns list, so the patch
        # carries the resourceVersion it was computed from.  The API server then
        # rejects a write raced by a concurrent invocation with 409 Conflict
        # instead of silently regressing childRuns to a stale value and tripping
        # the CRD's monotonic completedRuns/failedRuns guard.  The 409 propagates
        # so kopf retries the handler against the fresh object.
        patch_metadata: dict[str, Any] = {"uid": sweep_uid}
        resource_version = sweep_meta.get("resourceVersion")
        if isinstance(resource_version, str) and resource_version:
            patch_metadata["resourceVersion"] = resource_version

        await custom_objects.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=job_namespace,
            plural=SWEEP_PLURAL,
            name=sweep_name,
            body={
                "metadata": patch_metadata,
                "status": {
                    "childRuns": child_runs,
                    "completedRuns": completed_runs,
                    "failedRuns": failed_runs,
                },
            },
        )


async def run_services(settings: OperatorSettings) -> None:
    """Run reconciliation and the durable results API as one supervised process."""
    kubernetes_config.load_incluster_config()
    async with client.ApiClient() as api_client:
        custom_objects = client.CustomObjectsApi(api_client)
        index = ResultsIndex(Path(settings.artifact_root))
        index.rebuild()
        application = create_app(
            settings,
            index,
            KubernetesResultsLifecycle(custom_objects),
        )
        server = uvicorn.Server(
            uvicorn.Config(
                application,
                host=settings.api_host,
                port=settings.api_port,
                access_log=False,
            )
        )
        stop_flag = asyncio.Event()
        kopf_settings = kopf.OperatorSettings()
        kopf_settings.posting.enabled = False
        operator_task = asyncio.create_task(
            kopf.operator(
                settings=kopf_settings,
                clusterwide=True,
                standalone=True,
                stop_flag=stop_flag,
                memo=index,
            )
        )
        api_task = asyncio.create_task(server.serve())
        try:
            done, _ = await asyncio.wait(
                {operator_task, api_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            stop_flag.set()
            server.should_exit = True
            await asyncio.gather(operator_task, api_task, return_exceptions=True)


def main() -> None:
    """Launch the supervised operator and durable result service."""
    asyncio.run(run_services(OperatorSettings()))
