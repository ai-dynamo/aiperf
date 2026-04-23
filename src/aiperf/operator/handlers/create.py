# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""on_create handler logic for AIPerfJob CRD.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import kopf
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.cr_refs import (
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator import events
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.health import check_endpoint_health
from aiperf.operator.job_index import index_job_created, save_job_spec_file
from aiperf.operator.k8s_helpers import (
    create_idempotent_config_map,
    create_idempotent_custom_object,
    create_idempotent_role,
    create_idempotent_role_binding,
)
from aiperf.operator.models import AIPerfJobSpec, OwnerReference
from aiperf.operator.spec_converter import (
    AIPerfJobSpecConverter,
    apply_worker_config,
    build_benchmark_run,
)
from aiperf.operator.status import (
    ConditionType,
    Phase,
    StatusBuilder,
    format_timestamp,
)

logger = logging.getLogger(__name__)


def _to_plain(obj: Any) -> Any:
    """Recursively convert kopf Mapping subclasses to plain dicts/lists."""
    from collections.abc import Mapping

    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    return obj


def _validate_spec(
    spec: dict[str, Any],
    body: dict[str, Any],
    status: StatusBuilder,
) -> AIPerfJobSpec:
    """Validate the raw CRD spec and update status conditions."""
    try:
        validated_spec = AIPerfJobSpec.from_crd_spec(spec)
    except ValueError as e:
        status.conditions.set_false(ConditionType.CONFIG_VALID, "SpecInvalid", str(e))
        status.set_phase(Phase.FAILED).set_error(f"Invalid spec: {e}")
        status.finalize()
        events.spec_invalid(body, str(e))
        raise kopf.PermanentError(f"Invalid spec: {e}") from e

    status.conditions.set_true(
        ConditionType.CONFIG_VALID, "SpecValid", "Spec validation passed"
    )
    events.spec_valid(body)
    return validated_spec


async def _check_endpoint_reachable(
    validated_spec: AIPerfJobSpec,
    body: dict[str, Any],
    status: StatusBuilder,
) -> None:
    """Probe the target endpoint and record reachability as a condition."""
    endpoint_url = validated_spec.get_endpoint_url()
    if not endpoint_url:
        return

    if validated_spec.skip_endpoint_check:
        logger.info(
            f"Skipping endpoint reachability probe for {endpoint_url} (skipEndpointCheck=true)"
        )
        return

    health = await check_endpoint_health(endpoint_url)
    if health.reachable:
        status.conditions.set_true(
            ConditionType.ENDPOINT_REACHABLE,
            "EndpointReachable",
            f"Endpoint {endpoint_url} is reachable",
        )
        events.endpoint_reachable(body, endpoint_url)
    else:
        status.conditions.set_false(
            ConditionType.ENDPOINT_REACHABLE,
            "EndpointUnreachable",
            f"Endpoint {endpoint_url} unreachable: {health.error}",
        )
        events.endpoint_unreachable(body, endpoint_url, health.error)
        logger.warning(f"Endpoint {endpoint_url} not reachable: {health.error}")


def _build_deployment(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    job_id: str,
) -> tuple[KubernetesDeployment, int]:
    """Convert raw spec into a KubernetesDeployment. Returns (deployment, total_workers)."""
    converter = AIPerfJobSpecConverter(spec, name, namespace, job_id=job_id)
    config = converter.to_aiperf_config()
    deploy_config = converter.to_deployment_config()
    total_workers = converter.calculate_workers(deploy_config)
    num_pods = apply_worker_config(config, total_workers)

    run = build_benchmark_run(
        run_config=config.model_dump(mode="json", exclude_none=True),
        run_id=job_id,
        namespace=namespace,
    )

    deployment = KubernetesDeployment(
        job_id=job_id,
        namespace=namespace,
        worker_replicas=num_pods,
        config=config,
        run=run,
        deployment=deploy_config,
    )
    return deployment, total_workers


async def _run_preflight_checks(
    api: ApiClient,
    deployment: KubernetesDeployment,
    *,
    deploy_config: Any,
    config: Any,
    total_workers: int,
    num_pods: int,
    body: dict[str, Any],
    namespace: str,
    name: str,
    status: StatusBuilder,
) -> None:
    """Run the operator preflight checker and update conditions.

    Raises ``kopf.PermanentError`` if any check fails.
    """
    from aiperf.kubernetes.preflight import CheckStatus
    from aiperf.operator.preflight import OperatorPreflightChecker

    preflight = OperatorPreflightChecker(
        api=api,
        namespace=namespace,
        deployment=deployment,
        deploy_config=deploy_config,
        config=config,
        total_workers=total_workers,
        num_pods=num_pods,
    )
    preflight_results = await preflight.run_all(
        timeout=OperatorEnvironment.PREFLIGHT_TIMEOUT,
    )

    if not preflight_results.passed:
        failures = [c for c in preflight_results.checks if c.status == CheckStatus.FAIL]
        error_msg = "; ".join(f"{c.name}: {c.message}" for c in failures)
        status.conditions.set_false(
            ConditionType.PREFLIGHT_PASSED,
            "PreflightFailed",
            error_msg,
        )
        status.set_phase(Phase.FAILED).set_error(f"Pre-flight failed: {error_msg}")
        status.finalize()
        events.preflight_failed(body, error_msg)
        raise kopf.PermanentError(
            f"Pre-flight checks failed for {namespace}/{name}: {error_msg}"
        )

    status.conditions.set_true(
        ConditionType.PREFLIGHT_PASSED,
        "PreflightPassed",
        f"All {len(preflight_results.checks)} pre-flight checks passed",
    )
    events.preflight_passed(body, len(preflight_results.checks))

    warnings = [c for c in preflight_results.checks if c.status == CheckStatus.WARN]
    if warnings:
        warning_summary = "; ".join(f"{c.name}: {c.message}" for c in warnings)
        if len(warning_summary) > 512:
            warning_summary = warning_summary[:509] + "..."
        status.conditions.set_true(
            ConditionType.PREFLIGHT_HAS_WARNINGS,
            "PreflightWarnings",
            f"{len(warnings)} check(s) produced warnings: {warning_summary}",
        )
    else:
        status.conditions.set_false(
            ConditionType.PREFLIGHT_HAS_WARNINGS,
            "NoWarnings",
            "No preflight warnings",
        )

    for check in preflight_results.checks:
        if check.status == CheckStatus.WARN:
            events.preflight_warning(body, check.name, check.message)


async def _create_rbac(
    api: ApiClient,
    deployment: KubernetesDeployment,
    namespace: str,
    owner_ref_dict: dict[str, Any],
) -> None:
    """Create Role + RoleBinding for benchmark pods (idempotent on retry)."""
    rbac_spec = deployment.get_rbac_spec()
    role_manifest = rbac_spec.to_role_manifest()
    role_manifest.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
        owner_ref_dict
    )
    await create_idempotent_role(api, role_manifest, namespace)

    binding_manifest = rbac_spec.to_role_binding_manifest()
    binding_manifest.setdefault("metadata", {}).setdefault(
        "ownerReferences", []
    ).append(owner_ref_dict)
    await create_idempotent_role_binding(api, binding_manifest, namespace)
    logger.info(f"Created RBAC for service account '{rbac_spec.service_account}'")


async def _create_configmap(
    api: ApiClient,
    deployment: KubernetesDeployment,
    namespace: str,
    owner_ref_dict: dict[str, Any],
) -> str:
    """Create the benchmark ConfigMap. Returns the ConfigMap name."""
    configmap = deployment.get_configmap_spec().to_k8s_manifest()
    configmap.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
        owner_ref_dict
    )
    await create_idempotent_config_map(api, configmap, namespace)
    configmap_name = configmap["metadata"]["name"]
    logger.info(f"Created ConfigMap {configmap_name}")

    # Brief pause to allow kubelet ConfigMap cache to sync before pods try to
    # mount the volume. Without this, the first attempt after a fresh image
    # deploy fails with "FailedMount: failed to sync configmap cache" because
    # 100 pods race to mount the ConfigMap before kubelets have cached it.
    await asyncio.sleep(OperatorEnvironment.CONFIGMAP_PROPAGATION_DELAY_SECONDS)
    return configmap_name


async def _persist_spec_and_index(
    spec: dict[str, Any],
    namespace: str,
    name: str,
    job_id: str,
) -> None:
    """Persist spec to disk and update the job index.

    Done BEFORE JobSet launch so operator restart can always reconstruct the
    job from disk. If this fails, kopf retries the whole handler rather than
    leaving a JobSet running that the index cannot see. RBAC/ConfigMap
    creates above are idempotent, so retry is safe.
    """
    try:
        plain_spec = _to_plain(spec)
        await save_job_spec_file(namespace, job_id, plain_spec)
        await index_job_created(namespace, job_id, plain_spec)
    except (OSError, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Transient persistence failure for {namespace}/{name}: {e}")
        raise kopf.TemporaryError(
            f"Persisting job spec/index failed: {e}", delay=10
        ) from e


async def _create_jobset(
    api: ApiClient,
    deployment: KubernetesDeployment,
    namespace: str,
    owner_ref_dict: dict[str, Any],
) -> str:
    """Create the JobSet custom resource. Returns the JobSet name."""
    jobset = deployment.get_jobset_spec().to_k8s_manifest()
    jobset.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
        owner_ref_dict
    )
    await create_idempotent_custom_object(
        api=api,
        group=JOBSET_GROUP,
        version=JOBSET_VERSION,
        plural=JOBSET_PLURAL,
        body=jobset,
        namespace=namespace,
    )
    jobset_name = jobset["metadata"]["name"]
    logger.info(f"Created JobSet {jobset_name}")
    return jobset_name


def _finalize_success(
    *,
    patch: kopf.Patch,
    status: StatusBuilder,
    body: dict[str, Any],
    deployment: KubernetesDeployment,
    deploy_config: Any,
    configmap_name: str,
    jobset_name: str,
    job_id: str,
    total_workers: int,
) -> dict[str, Any]:
    """Record success conditions/events and finalize the status patch."""
    status.conditions.set_true(
        ConditionType.RESOURCES_CREATED,
        "ResourcesCreated",
        f"Created ConfigMap/{configmap_name} and JobSet/{jobset_name}",
    )
    events.resources_created(body, configmap_name, jobset_name)
    events.created(body, job_id, total_workers)

    status.set_phase(Phase.PENDING)
    patch.status["startTime"] = format_timestamp()
    patch.status["jobId"] = job_id
    patch.status["jobSetName"] = deployment.jobset_name
    status.set_workers(0, total_workers)

    if deploy_config.results_ttl_days:
        patch.status["resultsTtlDays"] = deploy_config.results_ttl_days

    status.finalize()
    return {"jobSetName": deployment.jobset_name, "workers": total_workers}


async def _create_resources(
    *,
    spec: dict[str, Any],
    body: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    job_id: str,
    status: StatusBuilder,
    patch: kopf.Patch,
) -> dict[str, Any]:
    """Build deployment, run preflight checks, and create all k8s resources."""
    validated_spec = _validate_spec(spec, body, status)
    await _check_endpoint_reachable(validated_spec, body, status)

    deployment, total_workers = _build_deployment(spec, name, namespace, job_id)
    deploy_config = deployment.deployment
    config = deployment.config

    owner_ref_dict = OwnerReference.for_aiperf_job(name, uid).to_k8s_dict()
    async with k8s_client() as api:
        await _run_preflight_checks(
            api,
            deployment,
            deploy_config=deploy_config,
            config=config,
            total_workers=total_workers,
            num_pods=deployment.worker_replicas,
            body=body,
            namespace=namespace,
            name=name,
            status=status,
        )

        await _create_rbac(api, deployment, namespace, owner_ref_dict)
        configmap_name = await _create_configmap(
            api, deployment, namespace, owner_ref_dict
        )
        await _persist_spec_and_index(spec, namespace, name, job_id)
        jobset_name = await _create_jobset(api, deployment, namespace, owner_ref_dict)

        return _finalize_success(
            patch=patch,
            status=status,
            body=body,
            deployment=deployment,
            deploy_config=deploy_config,
            configmap_name=configmap_name,
            jobset_name=jobset_name,
            job_id=job_id,
            total_workers=total_workers,
        )


async def on_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> dict[str, Any]:
    """Create ConfigMap and JobSet for the benchmark job."""
    job_id = name
    logger.info(f"Creating AIPerfJob {namespace}/{name}")

    # Drop any stale cancellation flag left over from a previously-deleted
    # CR with the same name. Without this, deleting and recreating a CR of
    # the same name inherits the sticky cancel flag and the new CR can
    # never exit Pending.
    from aiperf.operator.client_cache import clear_cancellation, job_key

    clear_cancellation(job_key(namespace, job_id))

    status = StatusBuilder(patch)

    try:
        return await _create_resources(
            spec=spec,
            body=body,
            name=name,
            namespace=namespace,
            uid=uid,
            job_id=job_id,
            status=status,
            patch=patch,
        )
    except (kopf.PermanentError, kopf.TemporaryError):
        raise
    except (ApiException, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Transient error creating AIPerfJob {namespace}/{name}: {e}")
        raise kopf.TemporaryError(
            f"Transient error creating AIPerfJob {namespace}/{name}: {e}", delay=30
        ) from e
    except Exception as e:
        logger.exception(f"Failed to create AIPerfJob {namespace}/{name}")
        status.set_phase(Phase.FAILED).set_error(str(e))
        status.finalize()
        events.failed(body, job_id, str(e))
        raise kopf.PermanentError(
            f"Failed to create AIPerfJob {namespace}/{name}: {e}"
        ) from e
