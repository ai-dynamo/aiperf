# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`StoreInjector`.

Uses a mocked :py:class:`KubectlClient` and a mocked
:py:class:`ToxiproxyInjector` -- no cluster required. The internal
:py:class:`NetworkInjector` is reused without patching for the construction
path; tests that need to assert delegation patch ``NetworkInjector.inject``
on the instance so the synthetic :py:class:`FaultSpec` shape is observable.
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.chaos_common.base import (
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.store import StoreInjector


def _make_kubectl_mock() -> AsyncMock:
    """Return an :py:class:`AsyncMock` shaped like :py:class:`KubectlClient`."""
    kubectl = AsyncMock()
    kubectl.run = AsyncMock(
        return_value=subprocess.CompletedProcess(
            args=["kubectl"], returncode=0, stdout="", stderr=""
        )
    )
    return kubectl


def _make_toxiproxy_mock() -> AsyncMock:
    """Return an :py:class:`AsyncMock` shaped like :py:class:`ToxiproxyInjector`.

    Tests that assert delegation patch ``injector._network.inject`` directly
    instead of exercising the real Toxiproxy REST surface.
    """
    return AsyncMock()


@pytest.mark.asyncio
async def test_store_etcd_kill_uses_default_selector_and_namespace() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    spec = FaultSpec(fault_id="store.etcd.kill")
    applied = await injector.inject(spec)

    kubectl.run.assert_awaited_once_with(
        "delete",
        "pod",
        "-l",
        "app.kubernetes.io/name=etcd",
        "-n",
        "dynamo-system",
        "--force",
        "--grace-period=0",
        check=True,
    )
    assert applied.metadata["namespace"] == "dynamo-system"
    assert applied.metadata["selector"] == "app.kubernetes.io/name=etcd"


@pytest.mark.asyncio
async def test_store_etcd_kill_with_target_override_uses_override() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    spec = FaultSpec(
        fault_id="store.etcd.kill",
        target={"ns": "custom-ns", "selector": "app=mine"},
    )
    applied = await injector.inject(spec)

    kubectl.run.assert_awaited_once_with(
        "delete",
        "pod",
        "-l",
        "app=mine",
        "-n",
        "custom-ns",
        "--force",
        "--grace-period=0",
        check=True,
    )
    assert applied.metadata["namespace"] == "custom-ns"
    assert applied.metadata["selector"] == "app=mine"


@pytest.mark.asyncio
async def test_store_nats_kill_uses_default_selector_and_namespace() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    spec = FaultSpec(fault_id="store.nats.kill")
    await injector.inject(spec)

    kubectl.run.assert_awaited_once_with(
        "delete",
        "pod",
        "-l",
        "app=nats",
        "-n",
        "dynamo-system",
        "--force",
        "--grace-period=0",
        check=True,
    )


@pytest.mark.asyncio
async def test_store_etcd_timeout_delegates_to_network_injector() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    sentinel = AsyncMock(name="applied")
    injector._network.inject = AsyncMock(return_value=sentinel)

    spec = FaultSpec(
        fault_id="store.etcd.timeout",
        target={"proxy": "etcd"},
        params={"attributes": {"timeout": 1000}},
    )
    result = await injector.inject(spec)

    assert result is sentinel
    injector._network.inject.assert_awaited_once()
    delegated_spec = injector._network.inject.await_args.args[0]
    assert delegated_spec.fault_id == "network.timeout"
    assert delegated_spec.target == {"proxy": "etcd"}
    assert delegated_spec.params == {"attributes": {"timeout": 1000}}


@pytest.mark.asyncio
async def test_store_etcd_bandwidth_delegates() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    sentinel = AsyncMock(name="applied")
    injector._network.inject = AsyncMock(return_value=sentinel)

    spec = FaultSpec(
        fault_id="store.etcd.bandwidth",
        target={"proxy": "etcd"},
        params={"attributes": {"rate": 1024}},
    )
    result = await injector.inject(spec)

    assert result is sentinel
    delegated_spec = injector._network.inject.await_args.args[0]
    assert delegated_spec.fault_id == "network.bandwidth"
    assert delegated_spec.target == {"proxy": "etcd"}
    assert delegated_spec.params == {"attributes": {"rate": 1024}}


@pytest.mark.asyncio
async def test_store_nats_partition_delegates_with_partition_fault_id() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    sentinel = AsyncMock(name="applied")
    injector._network.inject = AsyncMock(return_value=sentinel)

    spec = FaultSpec(
        fault_id="store.nats.partition",
        target={"proxy": "nats"},
    )
    result = await injector.inject(spec)

    assert result is sentinel
    delegated_spec = injector._network.inject.await_args.args[0]
    assert delegated_spec.fault_id == "network.partition"
    assert delegated_spec.target == {"proxy": "nats"}


@pytest.mark.asyncio
async def test_unknown_store_fault_raises_precondition() -> None:
    kubectl = _make_kubectl_mock()
    toxiproxy = _make_toxiproxy_mock()
    injector = StoreInjector(kubectl, toxiproxy)

    spec = FaultSpec(fault_id="store.etcd.detonate")
    with pytest.raises(FaultPreconditionError, match="does not recognize"):
        await injector.inject(spec)


def test_handles_prefix_match_store() -> None:
    assert StoreInjector.handles("store") is True
    assert StoreInjector.handles("store.etcd.kill") is True
    assert StoreInjector.handles("store.nats.partition") is True
    assert StoreInjector.handles("store.etcd.bandwidth") is True
    assert StoreInjector.handles("network.latency") is False
    assert StoreInjector.handles("pod") is False
    # Prefix safety: "storex.kill" must NOT match the "store" namespace.
    assert StoreInjector.handles("storex.kill") is False
