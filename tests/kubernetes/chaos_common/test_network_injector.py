# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`NetworkInjector`.

Mocks :py:class:`ToxiproxyInjector` so the contract (dispatch from
``fault_id`` to ``add_toxic`` toxic_type, restore via the auto-generated
toxic name, precondition errors on missing target, partition path
disabling/re-enabling the proxy, ``handles()`` prefix matching) is
exercised without touching a real toxiproxy admin API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.kubernetes.chaos_common.base import (
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.network import NetworkInjector


def _mock_toxiproxy() -> AsyncMock:
    """Return an :py:class:`AsyncMock` standing in for :py:class:`ToxiproxyInjector`."""
    toxiproxy = AsyncMock()
    toxiproxy.add_toxic = AsyncMock(return_value={})
    toxiproxy.remove_toxic = AsyncMock(return_value=None)
    toxiproxy.remove_proxy = AsyncMock(return_value=None)
    toxiproxy.base_url = "http://127.0.0.1:1"
    return toxiproxy


@pytest.mark.asyncio
async def test_network_latency_adds_latency_toxic() -> None:
    toxiproxy = _mock_toxiproxy()
    injector = NetworkInjector(toxiproxy=toxiproxy)
    spec = FaultSpec(
        fault_id="network.latency",
        target={"proxy": "mock-server"},
        params={"attributes": {"latency": 500, "jitter": 100}},
    )

    handle = await injector.inject(spec)

    toxiproxy.add_toxic.assert_awaited_once()
    call_kwargs = toxiproxy.add_toxic.await_args.kwargs
    assert call_kwargs["proxy_name"] == "mock-server"
    assert call_kwargs["toxic_type"] == "latency"
    assert call_kwargs["attributes"] == {"latency": 500, "jitter": 100}
    assert call_kwargs["name"].startswith("latency-")
    # The same toxic name must be stashed in metadata for the restore path.
    assert handle.metadata["toxic_name"] == call_kwargs["name"]
    assert handle.metadata["proxy_name"] == "mock-server"


@pytest.mark.asyncio
async def test_network_latency_restore_removes_toxic() -> None:
    toxiproxy = _mock_toxiproxy()
    injector = NetworkInjector(toxiproxy=toxiproxy)
    spec = FaultSpec(
        fault_id="network.latency",
        target={"proxy": "etcd"},
        params={
            "attributes": {"latency": 200},
            "stream": "upstream",
            "toxicity": 0.5,
        },
    )

    async with await injector.inject(spec) as handle:
        toxic_name = handle.metadata["toxic_name"]
        # Stream and toxicity were forwarded.
        call_kwargs = toxiproxy.add_toxic.await_args.kwargs
        assert call_kwargs["stream"] == "upstream"
        assert call_kwargs["toxicity"] == 0.5

    toxiproxy.remove_toxic.assert_awaited_once_with(
        proxy_name="etcd", toxic_name=toxic_name
    )


@pytest.mark.asyncio
async def test_network_timeout_adds_timeout_toxic() -> None:
    toxiproxy = _mock_toxiproxy()
    injector = NetworkInjector(toxiproxy=toxiproxy)
    spec = FaultSpec(
        fault_id="network.timeout",
        target={"proxy": "nats"},
        params={"attributes": {"timeout": 0}},
    )

    await injector.inject(spec)

    call_kwargs = toxiproxy.add_toxic.await_args.kwargs
    assert call_kwargs["toxic_type"] == "timeout"
    assert call_kwargs["attributes"] == {"timeout": 0}
    assert call_kwargs["name"].startswith("timeout-")


@pytest.mark.asyncio
async def test_network_partition_disables_and_restores_proxy() -> None:
    toxiproxy = _mock_toxiproxy()
    injector = NetworkInjector(toxiproxy=toxiproxy)
    spec = FaultSpec(
        fault_id="network.partition",
        target={"proxy": "nixl-0"},
    )

    patch_target = (
        "tests.kubernetes.chaos_common.injectors.network._patch_proxy_enabled"
    )
    with patch(patch_target, new=AsyncMock()) as patcher:
        async with await injector.inject(spec) as handle:
            # First call: disable proxy.
            patcher.assert_awaited_once()
            assert patcher.await_args.args[1] == "nixl-0"
            assert patcher.await_args.kwargs == {"enabled": False}
            assert handle.metadata["proxy_name"] == "nixl-0"

        # Restore re-enables the proxy.
        assert patcher.await_count == 2
        final_call = patcher.await_args
        assert final_call.args[1] == "nixl-0"
        assert final_call.kwargs == {"enabled": True}

    # add_toxic must not be touched on the partition path.
    toxiproxy.add_toxic.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_proxy_target_raises_precondition() -> None:
    toxiproxy = _mock_toxiproxy()
    injector = NetworkInjector(toxiproxy=toxiproxy)
    spec = FaultSpec(
        fault_id="network.latency",
        target={},  # proxy intentionally missing
        params={"attributes": {"latency": 100}},
    )

    with pytest.raises(FaultPreconditionError, match="proxy"):
        await injector.inject(spec)

    toxiproxy.add_toxic.assert_not_awaited()


def test_handles_prefix_match_network() -> None:
    assert NetworkInjector.handles("network")
    assert NetworkInjector.handles("network.latency")
    assert NetworkInjector.handles("network.reset_peer")
    assert NetworkInjector.handles("network.partition")
    assert not NetworkInjector.handles("pod")
    assert not NetworkInjector.handles("store.etcd.timeout")
    # Guard against accidental prefix-extension matches.
    assert not NetworkInjector.handles("networkfoo")
