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
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from aiperf.kubernetes.console import (
    clear_last_benchmark,
    get_last_benchmark,
    print_action,
    print_error,
    print_info,
)
from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

    from aiperf.kubernetes.models import AIPerfJobInfo


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
    age_seconds = max((datetime.now(timezone.utc) - created_dt).total_seconds(), 0)
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}m"
    if age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h"
    return f"{int(age_seconds / 86400)}d"


def resolve_job_id_and_namespace(
    job_id: str | None, namespace: str | None
) -> tuple[str, str] | None:
    """Resolve job_id and namespace, using last benchmark if not specified.

    Returns (job_id, namespace) tuple if resolved, None if not found.
    """
    if job_id is not None:
        return (job_id, namespace or DEFAULT_BENCHMARK_NAMESPACE)

    last = get_last_benchmark()
    if last is None:
        print_error("No job_id specified and no previous benchmark found")
        print_action("Run 'aiperf kube profile' first or specify a job_id")
        return None

    print_info(f"Using last benchmark: {last.job_id} in {last.namespace}")
    return (last.job_id, namespace or last.namespace)


class ResolvedJob:
    """Result of resolving a job identifier to an AIPerfJob CR."""

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


async def _open_api_client(
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> ApiClient:
    """Load k8s config and return an open ``ApiClient``.

    Separate from ``k8s_client()`` because the returned client survives
    beyond a single ``async with`` block: CLI commands pass it through
    ``ResolvedJob`` to subsequent helpers and it closes when the process
    exits. Tests patch this function to inject a mock api.
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
) -> ResolvedJob | None:
    """Resolve a job identifier to an AIPerfJob CR, falling back to JobSet.

    Queries AIPerfJob CRs first. If not found, falls back to JobSet lookup
    and wraps the result.

    Returns a ``ResolvedJob`` holding an open ``ApiClient`` the caller can
    reuse for subsequent kubernetes_asyncio operations; the client lives
    for the remainder of the (short-lived) CLI command.

    Args:
        job_id: The job name or ID to search for.
        namespace: Optional namespace to search in.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        ResolvedJob if found, None otherwise.
    """
    from aiperf.kubernetes.client import find_aiperf_job, find_jobset

    resolved = resolve_job_id_and_namespace(job_id, namespace)
    if not resolved:
        return None
    job_id, namespace = resolved

    api = await _open_api_client(kubeconfig=kubeconfig, kube_context=kube_context)

    # Try AIPerfJob CR first
    job_info = await find_aiperf_job(api, job_id, namespace)
    if job_info:
        return ResolvedJob(name=job_id, job_info=job_info, api=api)

    # Fallback to JobSet lookup
    jobset_info = await find_jobset(api, job_id, namespace)
    if not jobset_info:
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


async def confirm_action(msg: str) -> bool:
    """Prompt user for confirmation. Returns True if confirmed, False if aborted."""
    response = await asyncio.to_thread(input, f"{msg} [y/N] ")
    if response.lower() != "y":
        print_info("Aborted")
        return False
    return True


def clear_last_benchmark_if_matches(job_id: str) -> None:
    """Clear stored last benchmark info if it matches the given job_id."""
    last = get_last_benchmark()
    if last and last.job_id == job_id:
        clear_last_benchmark()
