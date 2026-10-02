# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import (
    find_aiperfsweep,
    get_raw_aiperfsweep,
    get_raw_aiperfsweep_status,
    list_aiperfsweeps,
)


@pytest.mark.asyncio
async def test_list_aiperfsweeps_all_namespaces() -> None:
    api = MagicMock()
    co = MagicMock()
    co.list_cluster_custom_object = AsyncMock(
        return_value={
            "items": [{"metadata": {"name": "s1"}}, {"metadata": {"name": "s2"}}]
        }
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        items = await list_aiperfsweeps(api, all_namespaces=True)
    assert len(items) == 2
    co.list_cluster_custom_object.assert_awaited_once_with(
        group="aiperf.nvidia.com", version="v1alpha1", plural="aiperfsweeps"
    )


@pytest.mark.asyncio
async def test_list_aiperfsweeps_namespaced() -> None:
    api = MagicMock()
    co = MagicMock()
    co.list_namespaced_custom_object = AsyncMock(return_value={"items": []})
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        items = await list_aiperfsweeps(api, namespace="bench")
    assert items == []
    co.list_namespaced_custom_object.assert_awaited_once_with(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        namespace="bench",
        plural="aiperfsweeps",
    )


@pytest.mark.asyncio
async def test_find_aiperfsweep_returns_body() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(
        return_value={"metadata": {"name": "s1"}}
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await find_aiperfsweep(api, "bench", "s1")
    assert body == {"metadata": {"name": "s1"}}


@pytest.mark.asyncio
async def test_find_aiperfsweep_returns_none_on_404() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=404, reason="Not Found")
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await find_aiperfsweep(api, "bench", "nope")
    assert body is None


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_status_returns_status() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object_status = AsyncMock(
        return_value={"status": {"phase": "Running", "completedRuns": 4}}
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        st = await get_raw_aiperfsweep_status(api, "s1", "bench")
    assert st == {"phase": "Running", "completedRuns": 4}


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_status_returns_none_on_404() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=404, reason="Not Found")
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        st = await get_raw_aiperfsweep_status(api, "s1", "bench")
    assert st is None


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_returns_body() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(return_value={"spec": {}, "status": {}})
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await get_raw_aiperfsweep(api, "bench", "s1")
    assert body == {"spec": {}, "status": {}}
