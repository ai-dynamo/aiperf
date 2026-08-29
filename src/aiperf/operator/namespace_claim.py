# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lease-based namespace ownership for the AIPerf operator.

A cluster-wide operator runs with ``--all-namespaces``, so without a claim it
also reconciles namespaces that host a scoped operator, and two operators fight
over the same AIPerfJob. Ownership here follows the install, not the job:

- A scoped operator (``AIPERF_OPERATOR_ID`` set) writes a
  ``coordination.k8s.io/v1`` Lease named ``aiperf-operator`` into every
  namespace it watches, and renews it on a timer.
- Every operator, the global one included, skips a namespace whose Lease is
  held by a different identity and still fresh.
- The global operator (``id == ""``) never writes a Lease. It owns every
  namespace with no live claim.

The lease duration is deliberately long (minutes, not seconds): a
crash-looping operator should keep its namespace across restarts, and only an
uninstall should let the claim lapse.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from kubernetes_asyncio.client import (
    CoordinationV1Api,
    V1Lease,
    V1LeaseSpec,
    V1ObjectMeta,
)
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.environment import Environment
from aiperf.kubernetes.client import k8s_client

__all__ = [
    "LEASE_NAME",
    "NamespaceClaim",
    "NamespaceClaimConflict",
    "lease_holder_if_live",
    "watched_namespaces_from_argv",
]

LEASE_NAME = "aiperf-operator"

logger = logging.getLogger(__name__)

ApiFactory = Callable[[], contextlib.AbstractAsyncContextManager[Any]]


class NamespaceClaimConflict(RuntimeError):
    """Raised when another operator already holds a live claim on a namespace."""

    def __init__(self, namespace: str, holder: str) -> None:
        super().__init__(
            f"namespace {namespace!r} is claimed by operator {holder!r}; "
            "uninstall that operator or give this one a different set of "
            "watch namespaces"
        )
        self.namespace = namespace
        self.holder = holder


def lease_holder_if_live(lease: V1Lease, *, default_duration: int) -> str | None:
    """Return the lease's holder identity, or ``None`` if it has expired.

    A lease is expired when ``renewTime + leaseDurationSeconds`` is in the past.
    A lease with no renew/acquire timestamp is treated as live: an unreadable
    timestamp must not let a second operator steal a namespace.
    """
    spec: V1LeaseSpec | None = lease.spec
    if spec is None or not spec.holder_identity:
        return None
    stamp = spec.renew_time or spec.acquire_time
    if stamp is None:
        return spec.holder_identity
    duration = spec.lease_duration_seconds or default_duration
    if stamp + timedelta(seconds=duration) < datetime.now(UTC):
        return None
    return spec.holder_identity


def watched_namespaces_from_argv(argv: Sequence[str]) -> list[str]:
    """Extract the namespaces kopf was told to watch from its command line.

    kopf owns namespace scoping via ``--namespace``/``-n`` (repeatable) and
    exposes it to no handler kwarg, so the operator recovers its own scope from
    ``sys.argv``. An empty result means cluster-wide (``--all-namespaces``).
    """
    namespaces: list[str] = []
    expecting = False
    for arg in argv:
        if expecting:
            namespaces.append(arg)
            expecting = False
        elif arg.startswith("--namespace="):
            namespaces.append(arg.split("=", 1)[1])
        elif arg in ("--namespace", "-n"):
            expecting = True
    return [ns for ns in namespaces if ns and not ns.startswith("-")]


@contextlib.asynccontextmanager
async def _default_api_factory() -> AsyncIterator[CoordinationV1Api]:
    async with k8s_client() as api:
        yield CoordinationV1Api(api)


class NamespaceClaim:
    """Tracks which namespaces this operator instance reconciles."""

    def __init__(
        self,
        identity: str | None = None,
        lease_seconds: int | None = None,
        api_factory: ApiFactory | None = None,
    ) -> None:
        self.identity = Environment.OPERATOR.ID if identity is None else identity
        self.lease_seconds = (
            Environment.OPERATOR.CLAIM_LEASE_SECONDS
            if lease_seconds is None
            else lease_seconds
        )
        self._api_factory: ApiFactory = api_factory or _default_api_factory
        self._holders: dict[str, str | None] = {}
        self._ownership: dict[str, bool] = {}
        self._claimed: set[str] = set()
        self._refreshing: set[str] = set()
        self._renew_task: asyncio.Task[None] | None = None

    @property
    def is_global(self) -> bool:
        """Whether this operator is the cluster-wide fallback owner."""
        return not self.identity

    def owns(self, namespace: str) -> bool:
        """Whether this operator reconciles resources in ``namespace``.

        Synchronous by contract: kopf evaluates ``when=`` for every event on
        every resource, so this only reads an in-memory cache. A cache miss
        schedules a background refresh and falls back to the safe default for
        this operator's role -- the global operator assumes ownership (a job in
        an unclaimed namespace must not be dropped), a scoped operator does not
        (it only ever owns namespaces it explicitly claimed at startup).
        """
        cached = self._ownership.get(namespace)
        if cached is not None:
            return cached
        self._schedule_refresh(namespace)
        return self.is_global

    async def acquire(self, namespace: str) -> None:
        """Claim ``namespace`` for this operator.

        The global operator writes nothing and merely resolves the current
        holder. A scoped operator creates the Lease, taking over an expired one
        and refreshing its own, and raises `NamespaceClaimConflict` when a
        different operator holds a live claim.
        """
        if self.is_global:
            await self.holder(namespace)
            return

        async with self._api_factory() as api:
            for attempt in range(2):
                try:
                    await api.create_namespaced_lease(
                        namespace=namespace, body=self._lease_body(namespace)
                    )
                except ApiException as exc:
                    if exc.status != 409:
                        raise
                    try:
                        existing = await api.read_namespaced_lease(
                            name=LEASE_NAME, namespace=namespace
                        )
                    except ApiException as read_exc:
                        # Someone deleted the lease between our conflict and
                        # our read; the namespace is free again, so retry once.
                        if read_exc.status == 404 and attempt == 0:
                            continue
                        raise
                    live_holder = lease_holder_if_live(
                        existing, default_duration=self.lease_seconds
                    )
                    if live_holder is not None and live_holder != self.identity:
                        self._record(namespace, live_holder)
                        raise NamespaceClaimConflict(namespace, live_holder) from None
                    await api.replace_namespaced_lease(
                        name=LEASE_NAME,
                        namespace=namespace,
                        body=self._lease_body(
                            namespace,
                            resource_version=(
                                existing.metadata or V1ObjectMeta()
                            ).resource_version,
                            transitions=(existing.spec.lease_transitions or 0) + 1
                            if existing.spec
                            else 1,
                        ),
                    )
                self._claimed.add(namespace)
                self._record(namespace, self.identity)
                return

    async def holder(self, namespace: str) -> str | None:
        """Return the identity holding a live claim on ``namespace``, if any."""
        async with self._api_factory() as api:
            try:
                lease = await api.read_namespaced_lease(
                    name=LEASE_NAME, namespace=namespace
                )
            except ApiException as exc:
                # 403 means this operator cannot read leases in the namespace,
                # which is indistinguishable from an unclaimed namespace and
                # must not wedge reconciliation.
                if exc.status in (403, 404):
                    self._record(namespace, None)
                    return None
                raise
        live_holder = lease_holder_if_live(lease, default_duration=self.lease_seconds)
        self._record(namespace, live_holder)
        return live_holder

    async def renew(self) -> None:
        """Keep this operator's view of namespace ownership current.

        A scoped operator refreshes the ``renewTime`` on each Lease it holds;
        the global operator re-lists every claim so a namespace whose scoped
        operator went away falls back to it.
        """
        if self.is_global:
            await self.refresh_all()
            return

        missing: list[str] = []
        async with self._api_factory() as api:
            for namespace in sorted(self._claimed):
                try:
                    await api.patch_namespaced_lease(
                        name=LEASE_NAME,
                        namespace=namespace,
                        body={
                            "spec": {
                                "holderIdentity": self.identity,
                                "leaseDurationSeconds": self.lease_seconds,
                                "renewTime": _rfc3339_now(),
                            }
                        },
                    )
                except ApiException as exc:
                    if exc.status == 404:
                        missing.append(namespace)
                    else:
                        logger.warning(
                            "Failed to renew namespace claim on %s: %s", namespace, exc
                        )
        for namespace in missing:
            with contextlib.suppress(NamespaceClaimConflict, ApiException):
                await self.acquire(namespace)

    async def refresh_all(self) -> None:
        """Re-read every ``aiperf-operator`` Lease in the cluster."""
        async with self._api_factory() as api:
            try:
                listing = await api.list_lease_for_all_namespaces(
                    field_selector=f"metadata.name={LEASE_NAME}"
                )
            except ApiException as exc:
                logger.warning("Failed to list operator namespace claims: %s", exc)
                return
        live: dict[str, str] = {}
        for lease in listing.items or []:
            namespace = (lease.metadata or V1ObjectMeta()).namespace
            if not namespace:
                continue
            live_holder = lease_holder_if_live(
                lease, default_duration=self.lease_seconds
            )
            if live_holder is not None:
                live[namespace] = live_holder
        for namespace in set(self._ownership) | set(live):
            self._record(namespace, live.get(namespace))

    async def start(self, namespaces: Iterable[str]) -> None:
        """Claim ``namespaces`` and begin renewing on a timer."""
        for namespace in namespaces:
            await self.acquire(namespace)
        if self.is_global:
            await self.refresh_all()
        if self._renew_task is None:
            self._renew_task = asyncio.create_task(self._renew_loop())

    async def stop(self) -> None:
        """Stop the renewal timer. The Lease is deliberately left behind."""
        if self._renew_task is None:
            return
        self._renew_task.cancel()
        await asyncio.gather(self._renew_task, return_exceptions=True)
        self._renew_task = None

    @property
    def renew_interval(self) -> float:
        """Seconds between renewals: a third of the lease duration."""
        return max(1.0, self.lease_seconds / 3)

    async def _renew_loop(self) -> None:
        while True:
            await asyncio.sleep(self.renew_interval)
            try:
                await self.renew()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Namespace claim renewal failed")

    def _lease_body(
        self,
        namespace: str,
        *,
        resource_version: str | None = None,
        transitions: int = 0,
    ) -> V1Lease:
        now = datetime.now(UTC)
        return V1Lease(
            metadata=V1ObjectMeta(
                name=LEASE_NAME,
                namespace=namespace,
                resource_version=resource_version,
            ),
            spec=V1LeaseSpec(
                holder_identity=self.identity,
                lease_duration_seconds=self.lease_seconds,
                acquire_time=now,
                renew_time=now,
                lease_transitions=transitions,
            ),
        )

    def _record(self, namespace: str, holder: str | None) -> None:
        self._holders[namespace] = holder
        self._ownership[namespace] = (
            holder is None if self.is_global else holder == self.identity
        )

    def _schedule_refresh(self, namespace: str) -> None:
        if namespace in self._refreshing:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refreshing.add(namespace)
        loop.create_task(self._refresh_one(namespace))

    async def _refresh_one(self, namespace: str) -> None:
        try:
            await self.holder(namespace)
        except Exception as exc:
            logger.warning(
                "Failed to resolve owner of namespace %s: %s", namespace, exc
            )
        finally:
            self._refreshing.discard(namespace)


def _rfc3339_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
