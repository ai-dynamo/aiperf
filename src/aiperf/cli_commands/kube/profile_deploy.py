# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-deploy path + shared helpers for `aiperf kube profile`.

Direct-mode deployment lives in :mod:`profile_deploy_direct`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

if TYPE_CHECKING:
    from kubernetes_asyncio.client import CoreV1Api, CustomObjectsApi


AIPERF_KIND = "AIPerfJob"


def _build_cr(
    name: str,
    namespace: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete AIPerfJob CR dict."""
    return {
        "apiVersion": AIPERF_API_VERSION,
        "kind": AIPERF_KIND,
        "metadata": {"name": name, "namespace": namespace},
        "spec": spec,
    }


async def wait_or_detach(
    name: str,
    namespace: str,
    kube_options: KubeOptions,
    *,
    detach: bool,
    no_wait: bool,
    attach_port: int,
    hint: str = "",
) -> None:
    """Either attach to the benchmark or detach with a status message."""
    import sys

    from aiperf.kubernetes import console as kube_console

    is_interactive = sys.stdout.isatty()
    should_detach = detach or not is_interactive

    if not is_interactive and not detach:
        kube_console.print_warning(
            "Non-interactive environment detected, using detach mode"
        )

    if should_detach:
        kube_console.print_detach_info(name, namespace, name=kube_options.name)
        if hint:
            kube_console.print_info(hint)
        return

    try:
        from aiperf.kubernetes import attach as kube_attach

        await kube_attach.auto_attach_workflow(
            name,
            namespace,
            attach_port,
            wait_for_ready=not no_wait,
            kubeconfig=kube_options.kubeconfig,
            kube_context=kube_options.kube_context,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        kube_console.print_interrupt_info(name, namespace)
        if hint:
            kube_console.print_info(hint)


async def operator_available(kube_options: KubeOptions) -> bool:
    """Check if the AIPerfJob CRD is installed on the cluster.

    Returns True if the operator CRD exists (operator mode), False otherwise
    (direct mode). Logs which mode is selected.
    """
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import AIPERF_JOB_GROUP, AIPERF_JOB_PLURAL

    crd_name = f"{AIPERF_JOB_PLURAL}.{AIPERF_JOB_GROUP}"
    try:
        async with k8s_client(
            kubeconfig=kube_options.kubeconfig,
            context=kube_options.kube_context,
        ) as api:
            await k8s_client_mod.ApiextensionsV1Api(
                api
            ).read_custom_resource_definition(crd_name)
        kube_console.print_info("AIPerfJob CRD detected, using operator mode")
        return True
    except ApiException as e:
        if e.status != 404:
            kube_console.print_info(
                f"AIPerfJob CRD not found, deploying directly (no operator) [ApiException: {e}]"
            )
            return False
        kube_console.print_info(
            "AIPerfJob CRD not found, deploying directly (no operator)"
        )
        return False
    except Exception as e:  # noqa: BLE001 - any unrecognized error falls back to direct-deploy mode with a user-facing message
        kube_console.print_info(
            f"AIPerfJob CRD not found, deploying directly (no operator) [{type(e).__name__}: {e}]"
        )
        return False


def _print_manifests_yaml(manifests: list[dict[str, Any]]) -> None:
    """Emit all manifests as a multi-document YAML stream to stdout."""
    import sys

    import ruamel.yaml

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    for i, manifest in enumerate(manifests):
        if i > 0:
            sys.stdout.write("---\n")
        yaml.dump(manifest, sys.stdout)


async def _replace_existing_cr_if_complete(
    custom: CustomObjectsApi,
    *,
    name: str,
    namespace: str,
    kube_context: str | None,
) -> None:
    """Delete a prior AIPerfJob CR with matching name if it finished.

    Raises ``SystemExit`` if the CR is still Running or Pending; returns
    silently when there is no prior CR (404).
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.cr_refs import (
        AIPERF_JOB_GROUP,
        AIPERF_JOB_PLURAL,
        AIPERF_JOB_VERSION,
    )

    try:
        existing = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
    except ApiException as e:
        if e.status == 404:
            return
        raise

    phase = (existing.get("status") or {}).get("phase", "")
    if phase in ("Running", "Pending"):
        ctx_flag = f" --context {kube_context}" if kube_context else ""
        raise SystemExit(
            f"AIPerfJob {name} is already {phase}. "
            f"Delete it first: kubectl{ctx_flag} delete aiperfjob {name} -n {namespace}"
        )
    kube_console.print_info(f"Replacing completed AIPerfJob {name}")
    await custom.delete_namespaced_custom_object(
        group=AIPERF_JOB_GROUP,
        version=AIPERF_JOB_VERSION,
        plural=AIPERF_JOB_PLURAL,
        namespace=namespace,
        name=name,
    )
    await asyncio.sleep(2)


async def _submit_cr(
    custom: CustomObjectsApi,
    core: CoreV1Api,
    cr: dict[str, Any],
    *,
    name: str,
    namespace: str,
    kube_context: str | None,
) -> None:
    """Create namespace (idempotent), replace stale CR, and submit the new CR."""
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.cr_refs import (
        AIPERF_JOB_GROUP,
        AIPERF_JOB_PLURAL,
        AIPERF_JOB_VERSION,
    )

    try:
        await core.create_namespace(
            body=k8s_client_mod.V1Namespace(
                metadata=k8s_client_mod.V1ObjectMeta(name=namespace),
            )
        )
    except ApiException as e:
        if e.status != 409:
            raise

    await _replace_existing_cr_if_complete(
        custom, name=name, namespace=namespace, kube_context=kube_context
    )

    try:
        await custom.create_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            body=cr,
        )
    except ApiException as e:
        import orjson

        detail = ""
        if e.body:
            try:
                body = orjson.loads(e.body)
                detail = body.get("message", "")
            except (orjson.JSONDecodeError, TypeError):
                detail = e.body[:200] if e.body else ""
        raise SystemExit(
            f"Failed to create AIPerfJob {namespace}/{name}: {detail}"
        ) from e


async def deploy_via_operator(
    spec: dict[str, Any],
    kube_options: KubeOptions,
    config: Any,
    name: str,
    namespace: str,
    *,
    dry_run: bool,
    detach: bool,
    no_wait: bool,
    attach_port: int,
) -> None:
    """Deploy by creating an AIPerfJob CR (requires operator)."""
    from kubernetes_asyncio import client as k8s_client_mod

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client

    del no_wait, attach_port  # signature parity with deploy_direct; unused here

    cr = _build_cr(name, namespace, spec)

    if dry_run:
        import orjson

        output = orjson.dumps(cr, option=orjson.OPT_INDENT_2).decode()
        kube_console.console.print(output, highlight=False)
        return

    async with k8s_client(
        kubeconfig=kube_options.kubeconfig,
        context=kube_options.kube_context,
    ) as api:
        core = k8s_client_mod.CoreV1Api(api)
        custom = k8s_client_mod.CustomObjectsApi(api)
        await _submit_cr(
            custom,
            core,
            cr,
            name=name,
            namespace=namespace,
            kube_context=kube_options.kube_context,
        )

    kube_console.print_cr_submission_summary(
        name=name,
        namespace=namespace,
        image=kube_options.image,
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
        model_names=config.get_model_names(),
        connections_per_worker=spec.get("connectionsPerWorker"),
    )

    kube_console.save_last_benchmark(name, namespace, name=kube_options.name)

    if detach:
        ctx_flag = (
            f" --kube-context {kube_options.kube_context}"
            if kube_options.kube_context
            else ""
        )
        kube_console.print_info(
            f"Detached. Watch with: aiperf kube watch {name}{ctx_flag}"
        )
        return

    from aiperf.kubernetes.attach import watch_job

    await watch_job(
        namespace=namespace,
        job_id=name,
        timeout=600,
        kubeconfig=kube_options.kubeconfig,
        kube_context=kube_options.kube_context,
    )
