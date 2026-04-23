# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct-deploy path for `aiperf kube profile` (no operator)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.cli_commands.kube.profile_deploy import (
    _print_manifests_yaml,
    wait_or_detach,
)
from aiperf.config.kube import KubeOptions

if TYPE_CHECKING:
    from kubernetes_asyncio.client import (
        CoreV1Api,
        CustomObjectsApi,
        RbacAuthorizationV1Api,
    )


async def _apply_manifest(
    manifest: dict[str, Any],
    *,
    core: CoreV1Api,
    rbac: RbacAuthorizationV1Api,
    custom: CustomObjectsApi,
    default_namespace: str,
) -> str | None:
    """Create one K8s resource from a manifest.

    Returns the ``"Kind/name"`` label on success (so the caller can log it),
    or None if the kind is not recognised. 409 AlreadyExists is not handled
    here; the caller catches and distinguishes it.
    """
    kind = manifest["kind"]
    res_name = manifest["metadata"]["name"]
    manifest_ns = manifest["metadata"].get("namespace") or default_namespace
    label = f"{kind}/{res_name}"

    from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION

    if kind == "Namespace":
        await core.create_namespace(body=manifest)
    elif kind == "ConfigMap":
        await core.create_namespaced_config_map(namespace=manifest_ns, body=manifest)
    elif kind == "Role":
        await rbac.create_namespaced_role(namespace=manifest_ns, body=manifest)
    elif kind == "RoleBinding":
        await rbac.create_namespaced_role_binding(namespace=manifest_ns, body=manifest)
    elif kind == "JobSet":
        await custom.create_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=manifest_ns,
            body=manifest,
        )
    else:
        return None
    return label


def _prepare_direct_deploy(
    config: Any,
    kube_options: KubeOptions,
    name: str,
    namespace: str,
) -> tuple[Any, Any, int]:
    """Apply K8s-runtime config overlays and compute the pod count.

    Returns ``(config, deploy_config, num_pods)`` where ``config`` is a
    re-validated :class:`AIPerfConfig` with K8s-specific fields applied.
    """
    import math

    from aiperf.config import AIPerfConfig
    from aiperf.kubernetes.environment import K8sEnvironment
    from aiperf.operator.spec_converter import (
        apply_k8s_runtime_config,
        apply_worker_config,
    )

    config_dict = config.model_dump(mode="json", exclude_none=True)
    apply_k8s_runtime_config(config_dict, name, namespace)
    config = AIPerfConfig.model_validate(config_dict)

    deploy_config = kube_options.to_deployment_config()
    # Only bump TTL when the user didn't explicitly set --ttl-seconds; the
    # direct-mode default keeps pods alive longer so results can be pulled.
    if "ttl_seconds" not in kube_options.model_fields_set:
        deploy_config.ttl_seconds_after_finished = (
            K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
        )

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.phases.values()),
        default=1,
    )
    total_workers = max(
        1, math.ceil(concurrency / deploy_config.connections_per_worker)
    )
    num_pods = apply_worker_config(config, total_workers)
    return config, deploy_config, num_pods


async def _apply_all_manifests(
    manifests: list[dict[str, Any]],
    kube_options: KubeOptions,
    *,
    effective_ns: str,
) -> None:
    """Create every resource in ``manifests``; treat 409 as already-exists."""
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client

    async with k8s_client(
        kubeconfig=kube_options.kubeconfig,
        context=kube_options.kube_context,
    ) as api:
        core = k8s_client_mod.CoreV1Api(api)
        rbac = k8s_client_mod.RbacAuthorizationV1Api(api)
        custom = k8s_client_mod.CustomObjectsApi(api)

        for manifest in manifests:
            kind = manifest["kind"]
            res_name = manifest["metadata"]["name"]
            try:
                label = await _apply_manifest(
                    manifest,
                    core=core,
                    rbac=rbac,
                    custom=custom,
                    default_namespace=effective_ns,
                )
            except ApiException as exc:
                if exc.status == 409:
                    kube_console.print_info(f"{kind}/{res_name} already exists")
                    continue
                raise
            if label is None:
                kube_console.print_warning(f"Unknown resource kind: {kind}, skipping")
                continue
            kube_console.print_success(f"Created {label}")


async def deploy_direct(
    config: Any,
    kube_options: KubeOptions,
    name: str,
    namespace: str,
    *,
    dry_run: bool,
    detach: bool,
    no_wait: bool,
    attach_port: int,
    skip_endpoint_check: bool = False,
) -> None:
    """Deploy directly without the operator (creates all K8s resources)."""
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.resources import KubernetesDeployment

    del (
        skip_endpoint_check
    )  # direct mode does no client-side endpoint check; accepted for CLI parity

    config, deploy_config, num_pods = _prepare_direct_deploy(
        config, kube_options, name, namespace
    )

    deployment = KubernetesDeployment(
        job_id=name,
        namespace=kube_options.namespace,  # None -> auto_namespace creates the namespace
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
        model_names=config.get_model_names(),
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
    )
    effective_ns = deployment.effective_namespace
    manifests = deployment.get_all_manifests()

    if dry_run:
        _print_manifests_yaml(manifests)
        return

    await _apply_all_manifests(manifests, kube_options, effective_ns=effective_ns)

    kube_console.print_cr_submission_summary(
        name=name,
        namespace=effective_ns,
        image=kube_options.image,
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
        model_names=config.get_model_names(),
        connections_per_worker=deploy_config.connections_per_worker,
    )

    kube_console.save_last_benchmark(name, effective_ns, name=kube_options.name)

    await wait_or_detach(
        name,
        effective_ns,
        kube_options,
        detach=detach,
        no_wait=no_wait,
        attach_port=attach_port,
        hint="Retrieve results: aiperf kube results --shutdown",
    )
