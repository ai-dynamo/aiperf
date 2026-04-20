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
import kr8s
from kr8s.asyncio.objects import ConfigMap, Role, RoleBinding

from aiperf.kubernetes.client import get_api
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator import events
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.health import check_endpoint_health
from aiperf.operator.job_index import index_job_created, save_job_spec_file
from aiperf.operator.k8s_helpers import create_idempotent
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

    status = StatusBuilder(patch)

    try:
        # Step 1: Validate spec
        try:
            validated_spec = AIPerfJobSpec.from_crd_spec(spec)
            status.conditions.set_true(
                ConditionType.CONFIG_VALID, "SpecValid", "Spec validation passed"
            )
            events.spec_valid(body)
        except ValueError as e:
            status.conditions.set_false(
                ConditionType.CONFIG_VALID, "SpecInvalid", str(e)
            )
            status.set_phase(Phase.FAILED).set_error(f"Invalid spec: {e}")
            status.finalize()
            events.spec_invalid(body, str(e))
            raise kopf.PermanentError(f"Invalid spec: {e}") from e

        # Step 2: Check endpoint health
        endpoint_url = validated_spec.get_endpoint_url()
        if endpoint_url:
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

        # Step 3: Convert spec to AIPerfConfig + BenchmarkRun
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

        owner_ref_dict = OwnerReference.for_aiperf_job(name, uid).to_k8s_dict()
        api = await get_api()

        # Step 4: Pre-flight checks
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
            failures = [
                c for c in preflight_results.checks if c.status == CheckStatus.FAIL
            ]
            error_msg = "; ".join(f"{c.name}: {c.message}" for c in failures)
            status.conditions.set_false(
                ConditionType.PREFLIGHT_PASSED,
                "PreflightFailed",
                error_msg,
            )
            status.set_phase(Phase.FAILED).set_error(f"Pre-flight failed: {error_msg}")
            status.finalize()
            events.preflight_failed(body, error_msg)
            raise kopf.PermanentError(f"Pre-flight checks failed: {error_msg}")

        status.conditions.set_true(
            ConditionType.PREFLIGHT_PASSED,
            "PreflightPassed",
            f"All {len(preflight_results.checks)} pre-flight checks passed",
        )
        events.preflight_passed(body, len(preflight_results.checks))

        for check in preflight_results.checks:
            if check.status == CheckStatus.WARN:
                events.preflight_warning(body, check.name, check.message)

        # Step 5: Create RBAC (Role + RoleBinding for benchmark pods)
        # Uses create-or-skip pattern for idempotency if operator retries.
        rbac_spec = deployment.get_rbac_spec()
        role_manifest = rbac_spec.to_role_manifest()
        role_manifest.setdefault("metadata", {}).setdefault(
            "ownerReferences", []
        ).append(owner_ref_dict)
        await create_idempotent(Role, role_manifest, api)

        binding_manifest = rbac_spec.to_role_binding_manifest()
        binding_manifest.setdefault("metadata", {}).setdefault(
            "ownerReferences", []
        ).append(owner_ref_dict)
        await create_idempotent(RoleBinding, binding_manifest, api)
        logger.info(f"Created RBAC for service account '{rbac_spec.service_account}'")

        # Step 6: Create ConfigMap
        configmap = deployment.get_configmap_spec().to_k8s_manifest()
        configmap.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
            owner_ref_dict
        )
        await create_idempotent(ConfigMap, configmap, api)
        configmap_name = configmap["metadata"]["name"]
        logger.info(f"Created ConfigMap {configmap_name}")

        # Brief pause to allow kubelet ConfigMap cache to sync before pods try to
        # mount the volume. Without this, the first attempt after a fresh image
        # deploy fails with "FailedMount: failed to sync configmap cache" because
        # 100 pods race to mount the ConfigMap before kubelets have cached it.
        await asyncio.sleep(OperatorEnvironment.CONFIGMAP_PROPAGATION_DELAY_SECONDS)

        # Step 7: Create JobSet
        jobset = deployment.get_jobset_spec().to_k8s_manifest()
        jobset.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
            owner_ref_dict
        )
        await create_idempotent(AsyncJobSet, jobset, api)
        jobset_name = jobset["metadata"]["name"]
        logger.info(f"Created JobSet {jobset_name}")

        # Set conditions and status
        status.conditions.set_true(
            ConditionType.RESOURCES_CREATED,
            "ResourcesCreated",
            f"Created ConfigMap/{configmap_name} and JobSet/{jobset_name}",
        )
        events.resources_created(body, configmap_name, jobset_name)
        events.created(body, job_id, total_workers)

        # Step 8: Save original CR spec and update job index
        # kopf wraps spec in special Spec/Body types that orjson can't serialize.
        # Recursively convert all Mapping subclasses to plain dicts.
        try:
            plain_spec = _to_plain(spec)
            await save_job_spec_file(namespace, job_id, plain_spec)
            await index_job_created(namespace, job_id, plain_spec)
        except Exception as e:
            logger.warning(f"Failed to save job spec/index for {namespace}/{name}: {e}")

        # Set initial status
        status.set_phase(Phase.PENDING)
        patch.status["startTime"] = format_timestamp()
        patch.status["jobId"] = job_id
        patch.status["jobSetName"] = deployment.jobset_name
        status.set_workers(0, total_workers)

        # Store results TTL if configured
        if deploy_config.results_ttl_days:
            patch.status["resultsTtlDays"] = deploy_config.results_ttl_days

        status.finalize()
        return {"jobSetName": deployment.jobset_name, "workers": total_workers}

    except kopf.PermanentError:
        raise
    except (kr8s.ServerError, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Transient error creating AIPerfJob {namespace}/{name}: {e}")
        status.set_phase(Phase.FAILED).set_error(str(e))
        status.finalize()
        events.failed(body, job_id, str(e))
        raise kopf.TemporaryError(f"Transient error: {e}", delay=30) from e
    except Exception as e:
        logger.exception(f"Failed to create AIPerfJob {namespace}/{name}")
        status.set_phase(Phase.FAILED).set_error(str(e))
        status.finalize()
        events.failed(body, job_id, str(e))
        raise kopf.PermanentError(f"Failed to create: {e}") from e
