# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI helper functions for Kubernetes operations.

Job resolution, user confirmation, and formatting utilities shared
across multiple CLI commands.

kubernetes_asyncio-backed operations live as free functions in
``aiperf.kubernetes.client`` (callers pass an ``ApiClient`` explicitly).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiperf.kubernetes.console import (
    get_last_benchmark,
    print_action,
    print_error,
    print_info,
)
from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

    from aiperf.kubernetes.models import AIPerfJobInfo, AIPerfSweepInfo


def format_age(created: str) -> str:
    """Format a Kubernetes timestamp as a human-readable age string.

    Args:
        created: ISO timestamp from Kubernetes (e.g., "2024-01-15T10:30:00Z").

    Returns:
        Age string like "5s", "10m", or "2h".
    """
    if not created:
        return "Unknown"
    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    age_seconds = max((datetime.now(UTC) - created_dt).total_seconds(), 0)
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}m"
    if age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h"
    return f"{int(age_seconds / 86400)}d"


def resolve_job_id_and_namespace(
    job_id: str | None, namespace: str | None, *, quiet: bool = False
) -> tuple[str, str] | None:
    """Resolve job_id and namespace, using last benchmark if not specified.

    Returns (job_id, namespace) tuple if resolved, None if not found.
    """
    if job_id is not None:
        return (job_id, namespace or DEFAULT_BENCHMARK_NAMESPACE)

    last = get_last_benchmark()
    if last is None:
        if not quiet:
            print_error("No job_id specified and no previous benchmark found")
            print_action("Run 'aiperf kube profile' first or specify a job_id")
        return None

    if not quiet:
        print_info(f"Using last benchmark: {last.job_id} in {last.namespace}")
    return (last.job_id, namespace or last.namespace)


class ResolvedJob:
    """Result of resolving a job identifier to an AIPerfJob CR.

    The ``api`` field is an open ``ApiClient``. This helper is designed
    for short-lived CLI commands that exit after use; the client is not
    closed automatically. Long-running contexts MUST NOT use
    ``resolve_job`` -- open the ``ApiClient`` via
    ``async with k8s_client() as api:`` instead.

    Callers with a natural choke point may call ``await resolved.aclose()``
    to release the underlying aiohttp session explicitly.
    """

    __slots__ = ("name", "job_info", "api")

    def __init__(self, name: str, job_info: AIPerfJobInfo, api: ApiClient) -> None:
        self.name = name
        self.job_info = job_info
        self.api = api

    @property
    def jobset_name(self) -> str | None:
        """JobSet name from the CR status."""
        return self.job_info.jobset_name

    @property
    def namespace(self) -> str:
        """Namespace from the CR."""
        return self.job_info.namespace

    @property
    def job_id(self) -> str:
        """Job ID from the CR status."""
        return self.job_info.job_id

    async def aclose(self) -> None:
        """Close the underlying ``ApiClient``. Idempotent.

        Safe to call multiple times; no-ops on an already-closed client.
        """
        await self.api.close()


async def _open_api_client(
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> ApiClient:
    """Load k8s config and return an open ``ApiClient``.

    Separate from ``k8s_client()`` because the returned client survives
    beyond a single ``async with`` block: CLI commands pass it through
    ``ResolvedJob`` to subsequent helpers and it closes when the process
    exits. Tests patch this function to inject a mock api.

    Note: CLAUDE.md says "Always use ``async with k8s_client() as api:``;
    never instantiate ``ApiClient()`` directly." This helper is the
    documented exception -- short-lived CLI commands cannot wrap their
    entire lifetime in one ``async with`` and rely on process exit to
    close the client.
    """
    from kubernetes_asyncio import config
    from kubernetes_asyncio.client import ApiClient as _ApiClient

    from aiperf.common.noisy_loggers import suppress_noisy_http_loggers

    suppress_noisy_http_loggers()
    try:
        config.load_incluster_config()
    except config.ConfigException:
        await config.load_kube_config(config_file=kubeconfig, context=kube_context)
    return _ApiClient()


async def resolve_job(
    job_id: str | None,
    namespace: str | None = None,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
    *,
    quiet: bool = False,
) -> ResolvedJob | None:
    """Resolve a job identifier to an AIPerfJob CR, falling back to JobSet.

    Queries AIPerfJob CRs first. If not found, falls back to JobSet lookup
    and wraps the result.

    Returns a ``ResolvedJob`` holding an open ``ApiClient`` the caller can
    reuse for subsequent kubernetes_asyncio operations. The client is
    deliberately *not* managed by an ``async with`` here -- it survives
    the call so the caller can use it. This is safe for short-lived CLI
    commands that exit after use; long-running contexts should instead
    open the ``ApiClient`` via ``async with k8s_client() as api:``.
    Callers may invoke ``await resolved.aclose()`` at a natural choke
    point to close the client explicitly.

    Args:
        job_id: The job name or ID to search for.
        namespace: Optional namespace to search in.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.
        quiet: Suppress informational user output while resolving defaults.

    Returns:
        ResolvedJob if found, None otherwise.
    """
    from aiperf.kubernetes.client import find_aiperf_job, find_jobset

    resolved = resolve_job_id_and_namespace(job_id, namespace, quiet=quiet)
    if not resolved:
        return None
    job_id, namespace = resolved

    api = await _open_api_client(kubeconfig=kubeconfig, kube_context=kube_context)

    try:
        # Try AIPerfJob CR first
        job_info = await find_aiperf_job(api, job_id, namespace)
        if job_info:
            return ResolvedJob(name=job_id, job_info=job_info, api=api)

        # Fallback to JobSet lookup
        jobset_info = await find_jobset(api, job_id, namespace)
        if not jobset_info:
            if not quiet:
                print_error(f"No AIPerf job found with ID: {job_id}")
                if namespace:
                    print_info(f"Searched namespace: {namespace}")
                else:
                    print_info("Searched all namespaces")
                print_action("Run 'aiperf kube list' to see available jobs")
            await api.close()
            return None

        # Wrap JobSetInfo as a minimal AIPerfJobInfo
        from aiperf.kubernetes.models import AIPerfJobInfo

        job_info = AIPerfJobInfo(
            name=jobset_info.name,
            namespace=jobset_info.namespace,
            phase=jobset_info.status,
            job_id=jobset_info.job_id,
            jobset_name=jobset_info.name,
            created=jobset_info.created,
            model=jobset_info.model,
            endpoint=jobset_info.endpoint,
        )
        return ResolvedJob(name=job_id, job_info=job_info, api=api)
    except BaseException:
        await api.close()
        raise


class ResolvedSweep:
    """Result of resolving a name to an AIPerfSweep CR.

    Mirrors :class:`ResolvedJob` semantics: the ``api`` field is an open,
    short-lived ``ApiClient`` owned by the calling CLI command. Callers
    with a natural choke point may call ``await resolved.aclose()`` to
    release the underlying aiohttp session explicitly.
    """

    __slots__ = ("name", "sweep_info", "api")

    def __init__(self, name: str, sweep_info: AIPerfSweepInfo, api: ApiClient) -> None:
        self.name = name
        self.sweep_info = sweep_info
        self.api = api

    @property
    def namespace(self) -> str:
        """Namespace from the CR."""
        return self.sweep_info.namespace

    @property
    def phase(self) -> str:
        """Current lifecycle phase from the CR status."""
        return self.sweep_info.phase

    async def aclose(self) -> None:
        """Close the underlying ``ApiClient``. Idempotent.

        Safe to call multiple times; no-ops on an already-closed client.
        """
        await self.api.close()


async def resolve_target(
    name: str | None,
    namespace: str | None = None,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
    *,
    quiet: bool = False,
) -> ResolvedJob | ResolvedSweep | None:
    """Resolve a name to either an AIPerfJob or AIPerfSweep CR.

    Resolution order:
      1. AIPerfJob CR (most common; matches today's :func:`resolve_job` behaviour).
      2. AIPerfSweep CR.
      3. JobSet fallback (job-shaped, returns :class:`ResolvedJob`).
      4. ``None`` with helpful error message.

    The returned API client is open and short-lived; treat its lifecycle the
    way :func:`resolve_job` does — long-running contexts MUST instead use
    ``async with k8s_client() as api:``.

    Args:
        name: Either an AIPerfJob/AIPerfSweep resource name or a generated
            ``status.jobId``. ``None`` falls back to the last benchmark
            persisted by ``aiperf kube profile``.
        namespace: Optional namespace to search in.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.
        quiet: Suppress informational user output while resolving defaults.

    Returns:
        :class:`ResolvedJob` if a job (or jobset fallback) is found,
        :class:`ResolvedSweep` if an AIPerfSweep is found, or ``None`` if
        nothing matches.
    """
    from aiperf.kubernetes.client import (
        find_aiperf_job,
        find_aiperf_sweep,
        find_jobset,
    )

    resolved = resolve_job_id_and_namespace(name, namespace, quiet=quiet)
    if not resolved:
        return None
    target_name, namespace = resolved

    api = await _open_api_client(kubeconfig=kubeconfig, kube_context=kube_context)

    try:
        # 1. AIPerfJob CR — same job-first behaviour as resolve_job.
        job_info = await find_aiperf_job(api, target_name, namespace)
        if job_info:
            return ResolvedJob(name=target_name, job_info=job_info, api=api)

        # 2. AIPerfSweep CR.
        sweep_info = await find_aiperf_sweep(api, target_name, namespace)
        if sweep_info:
            return ResolvedSweep(name=target_name, sweep_info=sweep_info, api=api)

        # 3. JobSet fallback — wrap as a minimal AIPerfJobInfo for callers that
        # only know how to consume ResolvedJob.
        jobset_info = await find_jobset(api, target_name, namespace)
        if jobset_info:
            from aiperf.kubernetes.models import AIPerfJobInfo

            job_info = AIPerfJobInfo(
                name=jobset_info.name,
                namespace=jobset_info.namespace,
                phase=jobset_info.status,
                job_id=jobset_info.job_id,
                jobset_name=jobset_info.name,
                created=jobset_info.created,
                model=jobset_info.model,
                endpoint=jobset_info.endpoint,
            )
            return ResolvedJob(name=target_name, job_info=job_info, api=api)

        # 4. Nothing found — name both candidate kinds in the error so the user
        # can pick the right follow-up command.
        print_error(f"No AIPerfJob or AIPerfSweep found with name: {target_name}")
        if namespace:
            print_info(f"Searched namespace: {namespace}")
        else:
            print_info("Searched all namespaces")
        print_action(
            "Run 'aiperf kube list' to see available jobs "
            "(or 'aiperf kube sweeps list' once available) for sweeps"
        )
        await api.close()
        return None
    except BaseException:
        await api.close()
        raise


async def confirm_action(msg: str) -> bool:
    """Prompt user for confirmation. Returns True if confirmed, False if aborted."""
    response = await asyncio.to_thread(input, f"{msg} [y/N] ")
    if response.lower() != "y":
        print_info("Aborted")
        return False
    return True
