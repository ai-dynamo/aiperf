# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes client — free functions over kubernetes_asyncio.

The canonical interface is ``k8s_client()`` + free functions
(``list_aiperf_jobs``, ``find_jobset``, …) that take an ``ApiClient``
explicitly and call ``CoreV1Api(api)`` / ``CustomObjectsApi(api)``
inline so the reader sees the native kubernetes_asyncio API surface.

Implementation is split by topic across sibling modules; this file
preserves the single import surface:

- selectors (``job_selector``, ``controller_selector``) — :mod:`client_selectors`
- AIPerfJob CR helpers — :mod:`client_jobs`
- JobSet helpers (and ``delete_namespace``) — :mod:`client_jobsets`
- pod helpers and ``cluster_version`` — :mod:`client_pods`
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient, rest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.client_jobs import (
    cancel_aiperf_job,
    find_aiperf_job,
    find_aiperf_sweep,
    get_raw_aiperfjob,
    get_raw_aiperfjob_status,
    list_aiperf_jobs,
)
from aiperf.kubernetes.client_jobsets import (
    delete_jobset,
    delete_namespace,
    find_jobset,
    list_jobsets,
)
from aiperf.kubernetes.client_pods import (
    cluster_version,
    find_controller_pod,
    find_operator_namespace,
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_pods,
    list_events_for_object,
    list_nodes,
    list_pods_all_namespaces,
    resolve_operator_namespace,
    wait_for_controller_pod_ready,
)
from aiperf.kubernetes.client_selectors import controller_selector, job_selector
from aiperf.kubernetes.credential_retry import (
    credential_retry_delay,
    interactive_credential_wait_enabled,
    is_api_authentication_error,
    is_kubeconfig_authentication_error,
    print_credential_wait,
    print_credentials_restored,
)

__all__ = [
    "asyncio",
    "cancel_aiperf_job",
    "client",
    "cluster_version",
    "controller_selector",
    "delete_jobset",
    "delete_namespace",
    "find_aiperf_job",
    "find_aiperf_sweep",
    "find_aiperfsweep",
    "find_controller_pod",
    "find_jobset",
    "find_operator_namespace",
    "find_operator_pod",
    "find_retrievable_pod",
    "get_pod_summary",
    "get_pods",
    "get_raw_aiperfjob",
    "get_raw_aiperfjob_status",
    "get_raw_aiperfsweep",
    "get_raw_aiperfsweep_status",
    "job_selector",
    "k8s_client",
    "list_aiperf_jobs",
    "list_aiperfsweeps",
    "list_events_for_object",
    "list_jobsets",
    "list_nodes",
    "list_pods_all_namespaces",
    "resolve_operator_namespace",
    "wait_for_controller_pod_ready",
]

# ``client`` (the kubernetes_asyncio module) and ``asyncio`` are re-exported as
# module attributes so tests can patch ``aiperf.kubernetes.client.client.CustomObjectsApi``
# and ``aiperf.kubernetes.client.asyncio.sleep``. Python modules are singletons;
# the patches propagate to the sibling ``client_jobs`` / ``client_jobsets`` /
# ``client_pods`` modules that also import these names.

APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV = "AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE"


def _apply_apiserver_tls_server_name_override() -> None:
    """Apply the chaos-only apiserver TLS hostname override when configured."""
    server_name = os.environ.get(APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV, "").strip()
    if not server_name:
        return
    cfg = client.Configuration.get_default_copy()
    cfg.tls_server_name = server_name
    client.Configuration.set_default(cfg)


class _CredentialWaitingApiClient(ApiClient):
    """ApiClient that reloads kubeconfig after interactive authentication loss."""

    def __init__(self, *, kubeconfig: str | None, context: str | None) -> None:
        super().__init__()
        self._kubeconfig = kubeconfig
        self._context = context
        self._credential_generation = 0
        self._credential_refresh_lock = asyncio.Lock()
        self._credential_wait_announced = False
        self._credential_retry_attempt = 0

    def call_api(self, *args: Any, **kwargs: Any) -> Any:
        """Run an API request, waiting for an interactive login after a 401."""
        if kwargs.get("async_req"):
            return super().call_api(*args, **kwargs)
        return self._call_api_with_credential_wait(args, kwargs)

    async def _call_api_with_credential_wait(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        while True:
            generation = self._credential_generation
            try:
                result = super().call_api(*args, **kwargs)
                response = await result
            except ApiException as error:
                if not is_api_authentication_error(error):
                    raise
            else:
                if self._credential_wait_announced:
                    self._credential_wait_announced = False
                    self._credential_retry_attempt = 0
                    print_credentials_restored(self._context)
                return response

            async with self._credential_refresh_lock:
                if generation != self._credential_generation:
                    continue
                if not self._credential_wait_announced:
                    self._credential_wait_announced = True
                    print_credential_wait(self._context)
                delay = credential_retry_delay(self._credential_retry_attempt)
                self._credential_retry_attempt += 1
                await asyncio.sleep(delay)
                try:
                    await self._reload_kubeconfig()
                except config.ConfigException as error:
                    if not is_kubeconfig_authentication_error(error):
                        raise
                    continue
                self._credential_generation += 1

    async def _reload_kubeconfig(self) -> None:
        """Reload credentials and TLS material into this client.

        ``persist_config=False`` is hardcoded — AIPerf never writes
        refreshed tokens back to the user's kubeconfig file.
        """
        configuration = client.Configuration()
        await config.load_kube_config(
            config_file=self._kubeconfig,
            context=self._context,
            client_configuration=configuration,
            persist_config=False,
        )
        new_rest_client = rest.RESTClientObject(configuration)
        old_rest_client = self.rest_client
        self.configuration = configuration
        self.client_side_validation = configuration.client_side_validation
        self.rest_client = new_rest_client
        await old_rest_client.close()


async def _load_kubeconfig(
    *,
    kubeconfig: str | None,
    context: str | None,
    wait_for_credentials: bool,
) -> None:
    """Load kubeconfig, optionally waiting for an external login to complete.

    Always passes ``persist_config=False`` — AIPerf is a read-only tool and
    must never write refreshed OIDC/GCP tokens back to the user's kubeconfig
    file.  The library default is ``True``, which silently mutates
    ``~/.kube/config`` on every token refresh.
    """
    announced = False
    attempt = 0
    while True:
        try:
            await config.load_kube_config(
                config_file=kubeconfig,
                context=context,
                persist_config=False,
            )
        except config.ConfigException as error:
            if not wait_for_credentials or not is_kubeconfig_authentication_error(
                error
            ):
                raise
            if not announced:
                announced = True
                print_credential_wait(context)
            await asyncio.sleep(credential_retry_delay(attempt))
            attempt += 1
        else:
            if announced:
                print_credentials_restored(context)
            return


@asynccontextmanager
async def k8s_client(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
    wait_for_credentials: bool | None = None,
) -> AsyncIterator[ApiClient]:
    """Load k8s config and yield an ``ApiClient``.

    Tries ``load_incluster_config()`` first (pod-mounted service account), then
    falls back to ``load_kube_config()`` on the given ``kubeconfig``/``context``.
    The ``ApiClient`` is guaranteed to be closed on scope exit.

    Args:
        kubeconfig: Path to a kubeconfig file. ``None`` means use the default
            resolution (``$KUBECONFIG`` or ``~/.kube/config``). Only consulted
            when the in-cluster load fails.
        context: Kubeconfig context name to activate. ``None`` means use the
            current-context from the kubeconfig.
        wait_for_credentials: Wait and retry when kubeconfig authentication is
            rejected. ``None`` enables this only for an interactive terminal.
            In-cluster clients never wait.

    Raises:
        kubernetes_asyncio.config.ConfigException: If both the in-cluster and
            kubeconfig loaders fail (e.g. no service account mounted AND no
            readable kubeconfig / unknown context).

    Example:
        >>> async with k8s_client() as api:
        ...     jobs = await list_aiperf_jobs(api, namespace="aiperf-bench")
        ...     for job in jobs:
        ...         print(job.name, job.phase)
    """
    suppress_noisy_http_loggers()
    using_kubeconfig = False
    should_wait = False
    try:
        config.load_incluster_config()
        _apply_apiserver_tls_server_name_override()
    except config.ConfigException:
        using_kubeconfig = True
        should_wait = (
            interactive_credential_wait_enabled()
            if wait_for_credentials is None
            else wait_for_credentials
        )
        await _load_kubeconfig(
            kubeconfig=kubeconfig,
            context=context,
            wait_for_credentials=should_wait,
        )
    api: ApiClient
    if using_kubeconfig and should_wait:
        api = _CredentialWaitingApiClient(kubeconfig=kubeconfig, context=context)
    else:
        api = ApiClient()
    try:
        yield api
    finally:
        await api.close()


async def list_aiperfsweeps(
    api: ApiClient,
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
) -> list[dict[str, Any]]:
    """List AIPerfSweep CRs.

    Args:
        api: The kubernetes_asyncio ApiClient.
        namespace: When set and ``all_namespaces=False``, list only this namespace.
        all_namespaces: When True, list cluster-wide (cluster-scoped permissions
            required).

    Returns:
        List of raw CR dicts; ``items`` array of the apiserver response.
    """
    co = client.CustomObjectsApi(api)
    if all_namespaces:
        resp = await co.list_cluster_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
        )
    else:
        if namespace is None:
            raise ValueError("namespace must be provided when all_namespaces is False")
        resp = await co.list_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
        )
    return list(resp.get("items", []))


async def find_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Fetch a single AIPerfSweep CR. Returns None on 404; raises on other errors."""
    co = client.CustomObjectsApi(api)
    try:
        return await co.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise


async def get_raw_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Alias of :func:`find_aiperfsweep` matching the AIPerfJob naming convention."""
    return await find_aiperfsweep(api, namespace, name)


async def get_raw_aiperfsweep_status(
    api: ApiClient, name: str, namespace: str
) -> dict[str, Any] | None:
    """Fetch ``status`` subresource of a single AIPerfSweep. Returns None on 404."""
    co = client.CustomObjectsApi(api)
    try:
        body = await co.get_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise
    status = body.get("status")
    return status if isinstance(status, dict) else None
