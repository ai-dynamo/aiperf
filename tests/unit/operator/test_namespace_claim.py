# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lease-based operator namespace ownership."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from kubernetes_asyncio.client import V1Lease, V1LeaseSpec, V1ObjectMeta
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.operator.namespace_claim import (
    LEASE_NAME,
    NamespaceClaim,
    NamespaceClaimConflict,
    watched_namespaces_from_argv,
)


class FakeCoordinationApi:
    """In-memory stand-in for ``CoordinationV1Api`` lease operations."""

    def __init__(self, leases: dict[str, V1Lease] | None = None) -> None:
        self.leases: dict[str, V1Lease] = leases or {}
        self.creates: list[str] = []
        self.replaces: list[str] = []
        self.patches: list[str] = []

    async def create_namespaced_lease(
        self, *, namespace: str, body: V1Lease
    ) -> V1Lease:
        if namespace in self.leases:
            raise ApiException(status=409, reason="Conflict")
        body.metadata.resource_version = "1"
        self.leases[namespace] = body
        self.creates.append(namespace)
        return body

    async def read_namespaced_lease(self, *, name: str, namespace: str) -> V1Lease:
        if name != LEASE_NAME or namespace not in self.leases:
            raise ApiException(status=404, reason="Not Found")
        return self.leases[namespace]

    async def replace_namespaced_lease(
        self, *, name: str, namespace: str, body: V1Lease
    ) -> V1Lease:
        if namespace not in self.leases:
            raise ApiException(status=404, reason="Not Found")
        self.leases[namespace] = body
        self.replaces.append(namespace)
        return body

    async def patch_namespaced_lease(
        self, *, name: str, namespace: str, body: dict[str, Any]
    ) -> V1Lease:
        if namespace not in self.leases:
            raise ApiException(status=404, reason="Not Found")
        self.patches.append(namespace)
        return self.leases[namespace]

    async def list_lease_for_all_namespaces(self, **_: Any) -> Any:
        class _List:
            def __init__(self, items: list[V1Lease]) -> None:
                self.items = items

        return _List(list(self.leases.values()))


def make_lease(
    namespace: str,
    holder: str,
    *,
    age_seconds: float = 0.0,
    duration_seconds: int = 300,
) -> V1Lease:
    """Build a Lease whose renewTime is ``age_seconds`` in the past."""
    return V1Lease(
        metadata=V1ObjectMeta(
            name=LEASE_NAME, namespace=namespace, resource_version="1"
        ),
        spec=V1LeaseSpec(
            holder_identity=holder,
            lease_duration_seconds=duration_seconds,
            acquire_time=datetime.now(UTC) - timedelta(seconds=age_seconds),
            renew_time=datetime.now(UTC) - timedelta(seconds=age_seconds),
        ),
    )


def claim_for(
    api: FakeCoordinationApi, identity: str, lease_seconds: int = 300
) -> NamespaceClaim:
    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[FakeCoordinationApi]:
        yield api

    return NamespaceClaim(
        identity=identity, lease_seconds=lease_seconds, api_factory=factory
    )


@pytest.mark.asyncio
async def test_acquire_free_namespace_writes_lease_and_owns() -> None:
    api = FakeCoordinationApi()
    claim = claim_for(api, "test-op")

    await claim.acquire("aiperf-test")

    assert api.creates == ["aiperf-test"]
    assert api.leases["aiperf-test"].spec.holder_identity == "test-op"
    assert claim.owns("aiperf-test") is True


@pytest.mark.asyncio
async def test_acquire_namespace_held_by_fresh_other_holder_raises() -> None:
    api = FakeCoordinationApi({"aiperf-test": make_lease("aiperf-test", "other-op")})
    claim = claim_for(api, "test-op")

    with pytest.raises(NamespaceClaimConflict) as excinfo:
        await claim.acquire("aiperf-test")

    assert "other-op" in str(excinfo.value)
    assert claim.owns("aiperf-test") is False
    assert api.replaces == []


@pytest.mark.asyncio
async def test_acquire_takes_over_expired_holder() -> None:
    api = FakeCoordinationApi(
        {
            "aiperf-test": make_lease(
                "aiperf-test", "dead-op", age_seconds=1000, duration_seconds=300
            )
        }
    )
    claim = claim_for(api, "test-op")

    await claim.acquire("aiperf-test")

    assert api.replaces == ["aiperf-test"]
    assert api.leases["aiperf-test"].spec.holder_identity == "test-op"
    assert claim.owns("aiperf-test") is True


@pytest.mark.asyncio
async def test_acquire_is_idempotent_for_same_identity() -> None:
    api = FakeCoordinationApi()
    claim = claim_for(api, "test-op")

    await claim.acquire("aiperf-test")
    await claim.acquire("aiperf-test")

    assert api.creates == ["aiperf-test"]
    assert api.replaces == ["aiperf-test"]
    assert claim.owns("aiperf-test") is True


@pytest.mark.asyncio
async def test_holder_returns_none_for_unclaimed_and_id_for_claimed() -> None:
    api = FakeCoordinationApi({"held": make_lease("held", "scoped-op")})
    claim = claim_for(api, "")

    assert await claim.holder("free") is None
    assert await claim.holder("held") == "scoped-op"


@pytest.mark.asyncio
async def test_holder_ignores_expired_lease() -> None:
    api = FakeCoordinationApi(
        {"stale": make_lease("stale", "dead-op", age_seconds=1000)}
    )
    claim = claim_for(api, "")

    assert await claim.holder("stale") is None


@pytest.mark.asyncio
async def test_global_operator_owns_unclaimed_and_defers_to_scoped() -> None:
    api = FakeCoordinationApi({"held": make_lease("held", "scoped-op")})
    claim = claim_for(api, "")

    await claim.acquire("free")
    await claim.acquire("held")

    assert api.creates == []
    assert claim.owns("free") is True
    assert claim.owns("held") is False


@pytest.mark.asyncio
async def test_renew_refreshes_owned_leases() -> None:
    api = FakeCoordinationApi()
    claim = claim_for(api, "test-op")
    await claim.acquire("aiperf-test")

    await claim.renew()

    assert api.patches == ["aiperf-test"]


@pytest.mark.asyncio
async def test_renew_recreates_a_deleted_lease() -> None:
    api = FakeCoordinationApi()
    claim = claim_for(api, "test-op")
    await claim.acquire("aiperf-test")
    del api.leases["aiperf-test"]

    await claim.renew()

    assert api.creates == ["aiperf-test", "aiperf-test"]
    assert claim.owns("aiperf-test") is True


@pytest.mark.asyncio
async def test_global_refresh_all_clears_expired_claims() -> None:
    lease = make_lease("held", "scoped-op")
    api = FakeCoordinationApi({"held": lease})
    claim = claim_for(api, "")
    await claim.acquire("held")
    assert claim.owns("held") is False

    del api.leases["held"]
    await claim.renew()

    assert claim.owns("held") is True


def test_owns_unknown_namespace_defaults_to_global_yes_scoped_no() -> None:
    api = FakeCoordinationApi()
    assert claim_for(api, "").owns("never-seen") is True
    assert claim_for(api, "scoped").owns("never-seen") is False


def test_watched_namespaces_from_argv_parses_kopf_flags() -> None:
    argv = [
        "kopf",
        "run",
        "-m",
        "aiperf.operator.main",
        "--namespace=aiperf-test",
        "--namespace",
        "other",
        "-n",
        "third",
    ]
    assert watched_namespaces_from_argv(argv) == ["aiperf-test", "other", "third"]


def test_watched_namespaces_from_argv_empty_for_all_namespaces() -> None:
    argv = ["kopf", "run", "-m", "aiperf.operator.main", "--all-namespaces"]
    assert watched_namespaces_from_argv(argv) == []


@pytest.mark.asyncio
async def test_scoped_start_with_no_namespaces_claims_nothing() -> None:
    """A scoped operator given no namespaces can never own one.

    ``start([])`` writes no Lease, so every ``owns()`` falls through to the
    scoped default of False -- the operator would reconcile nothing at all.
    This is why the startup handler refuses the configuration outright.
    """
    api = FakeCoordinationApi()
    claim = claim_for(api, "scoped-op")

    await claim.start([])
    try:
        assert api.creates == []
        assert claim.owns("aiperf-test") is False
    finally:
        await claim.stop()


@pytest.mark.asyncio
async def test_claim_watched_namespaces_rejects_scoped_operator_without_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kopf

    from aiperf.operator import main as operator_main

    monkeypatch.setattr(
        operator_main.sys, "argv", ["kopf", "run", "--all-namespaces"], raising=False
    )
    monkeypatch.setattr(operator_main._CLAIMS, "identity", "scoped-op")

    with pytest.raises(kopf.PermanentError) as excinfo:
        await operator_main.claim_watched_namespaces()

    assert "scoped-op" in str(excinfo.value)
    assert "watchNamespaces" in str(excinfo.value)


def test_every_object_handler_is_gated_on_namespace_ownership() -> None:
    """Every kopf object handler must skip namespaces this operator does not own.

    A handler added without ``when=owns_namespace`` silently makes the global
    operator reconcile jobs inside a scoped operator's namespace again.
    """
    import kopf

    from aiperf.operator import main as operator_main

    registry = kopf.get_default_registry()
    handlers = [
        handler
        for sub in (registry._changing, registry._watching, registry._spawning)
        for handler in sub.get_all_handlers()
    ]
    assert handlers, "kopf registry has no object handlers; import of main failed"

    ungated = [
        str(handler.id)
        for handler in handlers
        if handler.when is not operator_main.owns_namespace
    ]
    assert not ungated, (
        f"kopf object handlers missing when=owns_namespace: {sorted(ungated)}"
    )
