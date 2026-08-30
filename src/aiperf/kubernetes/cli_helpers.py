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
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.kubernetes.console import (
    get_last_benchmark,
    print_action,
    print_error,
    print_info,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

    from aiperf.kubernetes.models import AIPerfJobInfo, AIPerfSweepInfo


def _context_namespace(
    kubeconfig: str | None = None, context: str | None = None
) -> str | None:
    """Return the namespace of the selected kubeconfig context, if it sets one.

    ``None`` means "the context exists and pins no namespace" and nothing else.
    An explicitly named ``--kube-context`` that is absent from the kubeconfig,
    or a kubeconfig that cannot be read at all, raises ``ConfigurationError``:
    both used to collapse into the caller's "your context sets no namespace"
    message, which blames the wrong thing for a typo'd context name.

    Split out from :func:`resolve_benchmark_namespace` so tests can drive the
    resolution chain without a kubeconfig on disk.
    """
    from aiperf.config.loader.errors import ConfigurationError

    try:
        from kubernetes_asyncio import config as k8s_config

        contexts, active = k8s_config.list_kube_config_contexts(config_file=kubeconfig)
    except Exception as exc:  # noqa: BLE001 - no kubeconfig is a normal state
        # Only fatal when the user named a context: without one, "no readable
        # kubeconfig" is indistinguishable from "no namespace pinned", and the
        # caller's message already tells them to pass --namespace.
        if context is None:
            return None
        raise ConfigurationError(
            f"Could not read kubeconfig to resolve context {context!r}: {exc}"
        ) from exc

    if context is not None:
        active = next((c for c in contexts or [] if c.get("name") == context), None)
        if active is None:
            known = ", ".join(sorted(c.get("name", "") for c in contexts or []))
            raise ConfigurationError(
                f"kubeconfig context {context!r} not found."
                + (f" Known contexts: {known}" if known else "")
            )
    if not active:
        return None
    return (active.get("context") or {}).get("namespace") or None


def resolve_benchmark_namespace(
    namespace: str | None,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> str:
    """Resolve the namespace a benchmark runs in. Never guesses.

    Explicit ``--namespace`` wins; otherwise the active kubeconfig context's
    namespace; otherwise this raises. There is deliberately no fallback -- not
    ``default``, and not a product-specific namespace. The chart used to create
    an ``aiperf-benchmarks`` namespace and the CLI assumed it, so a user who
    never named a namespace had their pods and results land somewhere they were
    not watching, and hit RBAC failures mid-run rather than at submit time.
    """
    if namespace:
        return namespace

    resolved = _context_namespace(kubeconfig, context)
    if resolved:
        return resolved

    from aiperf.config.loader.errors import ConfigurationError

    raise ConfigurationError(
        "No Kubernetes namespace given and the active kubeconfig context does "
        "not set one. Pass --namespace <ns>, or set a default for your context:\n"
        "  kubectl config set-context --current --namespace=<ns>"
    )


def resolve_job_id_and_namespace(
    job_id: str | None,
    namespace: str | None,
    *,
    quiet: bool = False,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> tuple[str, str] | None:
    """Resolve job_id and namespace, using last benchmark if not specified.

    Returns (job_id, namespace) tuple if resolved, None if not found.
    """
    if job_id is not None:
        return (job_id, resolve_benchmark_namespace(namespace, kubeconfig, context))

    last = get_last_benchmark()
    if last is None:
        if not quiet:
            print_error("No job_id specified and no previous benchmark found")
            print_action("Run 'aiperf kube profile' first or specify a job_id")
        return None

    if not quiet:
        print_info(f"Using last benchmark: {last.job_id} in {last.namespace}")
    return (last.job_id, namespace or last.namespace)


def exit_target_not_found(*, ignore_not_found: bool) -> None:
    """Fail the shell for a target that could not be addressed.

    Target-addressing subcommands narrate rather than gate, so a benchmark that
    merely has no pods left is not an error. A benchmark that does not exist is,
    otherwise the command cannot be used as a CI existence check. ``kubectl``
    spells the opt-out ``--ignore-not-found``; matching it lets scripted callers
    treat a missing benchmark as a no-op while still seeing the printed error.
    """
    if not ignore_not_found:
        raise SystemExit(1)


async def target_exists(api: ApiClient, name: str, namespace: str | None) -> bool:
    """Whether an AIPerfJob, AIPerfSweep, or JobSet named ``name`` exists.

    Cheaper than :func:`resolve_job` for callers that only need presence, and
    used on the empty-pod-list path so the happy path pays no extra round trip.
    """
    from aiperf.kubernetes.client import (
        find_aiperf_job,
        find_aiperf_sweep,
        find_jobset,
    )

    if await find_aiperf_job(api, name, namespace):
        return True
    if await find_aiperf_sweep(api, name, namespace):
        return True
    return bool(await find_jobset(api, name, namespace))


class ResolvedJob:
    """Result of resolving a job identifier to an AIPerfJob CR.

    The ``api`` field is an open ``ApiClient`` leased from the canonical
    ``k8s_client`` context. Callers must invoke ``await resolved.aclose()``
    when their command finishes.
    """

    __slots__ = ("name", "job_info", "api", "_lease")

    def __init__(
        self,
        name: str,
        job_info: AIPerfJobInfo,
        api: ApiClient,
        *,
        lease: _ApiClientLease | None = None,
    ) -> None:
        self.name = name
        self.job_info = job_info
        self.api = api
        self._lease = lease or _ApiClientLease(api)

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
        await self._lease.aclose()


@dataclass(slots=True)
class _ApiClientLease:
    """Own an API client and, when present, its canonical context manager."""

    api: ApiClient
    context: AbstractAsyncContextManager[ApiClient] | None = None
    closed: bool = False

    async def aclose(self) -> None:
        """Release the client exactly once."""
        if self.closed:
            return
        self.closed = True
        if self.context is not None:
            await self.context.__aexit__(None, None, None)
        else:
            await self.api.close()


def _coerce_client_lease(opened: _ApiClientLease | ApiClient) -> _ApiClientLease:
    """Wrap legacy/mocked bare clients while production returns a lease."""
    return opened if isinstance(opened, _ApiClientLease) else _ApiClientLease(opened)


async def _open_api_client(
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> _ApiClientLease:
    """Enter the canonical Kubernetes client context and return its lease."""
    from aiperf.kubernetes.client import k8s_client

    context = k8s_client(kubeconfig=kubeconfig, context=kube_context)
    api = await context.__aenter__()
    return _ApiClientLease(api=api, context=context)


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
    reuse for subsequent kubernetes_asyncio operations. The resolver owns the
    entered ``k8s_client`` context until ``await resolved.aclose()``.

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

    resolved = resolve_job_id_and_namespace(
        job_id, namespace, quiet=quiet, kubeconfig=kubeconfig, context=kube_context
    )
    if not resolved:
        return None
    job_id, namespace = resolved

    lease = _coerce_client_lease(
        await _open_api_client(kubeconfig=kubeconfig, kube_context=kube_context)
    )
    api = lease.api

    try:
        # Try AIPerfJob CR first
        job_info = await find_aiperf_job(api, job_id, namespace)
        if job_info:
            return ResolvedJob(name=job_id, job_info=job_info, api=api, lease=lease)

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
            await lease.aclose()
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
        return ResolvedJob(name=job_id, job_info=job_info, api=api, lease=lease)
    except BaseException:
        await lease.aclose()
        raise


class ResolvedSweep:
    """Result of resolving a name to an AIPerfSweep CR.

    Mirrors :class:`ResolvedJob` semantics: the ``api`` field remains valid
    until the calling command invokes ``await resolved.aclose()``.
    """

    __slots__ = ("name", "sweep_info", "api", "_lease")

    def __init__(
        self,
        name: str,
        sweep_info: AIPerfSweepInfo,
        api: ApiClient,
        *,
        lease: _ApiClientLease | None = None,
    ) -> None:
        self.name = name
        self.sweep_info = sweep_info
        self.api = api
        self._lease = lease or _ApiClientLease(api)

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
        await self._lease.aclose()


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

    The returned API client is leased from ``k8s_client``; release it with the
    result object's ``aclose`` method.

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

    resolved = resolve_job_id_and_namespace(
        name, namespace, quiet=quiet, kubeconfig=kubeconfig, context=kube_context
    )
    if not resolved:
        return None
    target_name, namespace = resolved

    lease = _coerce_client_lease(
        await _open_api_client(kubeconfig=kubeconfig, kube_context=kube_context)
    )
    api = lease.api

    try:
        # 1. AIPerfJob CR — same job-first behaviour as resolve_job.
        job_info = await find_aiperf_job(api, target_name, namespace)
        if job_info:
            return ResolvedJob(
                name=target_name, job_info=job_info, api=api, lease=lease
            )

        # 2. AIPerfSweep CR.
        sweep_info = await find_aiperf_sweep(api, target_name, namespace)
        if sweep_info:
            return ResolvedSweep(
                name=target_name, sweep_info=sweep_info, api=api, lease=lease
            )

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
            return ResolvedJob(
                name=target_name, job_info=job_info, api=api, lease=lease
            )

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
        await lease.aclose()
        return None
    except BaseException:
        await lease.aclose()
        raise


async def confirm_action(msg: str) -> bool:
    """Prompt user for confirmation. Returns True if confirmed, False if aborted."""
    response = await asyncio.to_thread(input, f"{msg} [y/N] ")
    if response.lower() != "y":
        print_info("Aborted")
        return False
    return True
