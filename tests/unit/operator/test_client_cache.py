# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator client_cache module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.operator.client_cache import (
    _progress_clients,
    _reset_for_testing,
    _shutdown_sent,
    _warned_pod_restarts,
    close_progress_client,
    get_or_create_progress_client,
    job_key,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset all module-level state between tests."""
    _reset_for_testing()
    yield
    _reset_for_testing()


class TestJobKey:
    """Tests for job_key function."""

    def test_combines_namespace_and_name(self) -> None:
        assert job_key("ns", "job") == "ns/job"

    def test_different_namespaces_different_keys(self) -> None:
        assert job_key("ns1", "job") != job_key("ns2", "job")


class TestGetOrCreateProgressClient:
    """Tests for get_or_create_progress_client."""

    @pytest.mark.asyncio
    async def test_creates_new_client(self) -> None:
        """Verify creates and caches a new ProgressClient."""
        from unittest.mock import patch as mock_patch

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with mock_patch(
            "aiperf.operator.client_cache.ProgressClient", return_value=mock_client
        ):
            result = await get_or_create_progress_client("test/job-1")

        assert result is mock_client
        assert "test/job-1" in _progress_clients
        mock_client.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_cached_client(self) -> None:
        """Verify same client returned for same key."""
        from unittest.mock import patch as mock_patch

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with mock_patch(
            "aiperf.operator.client_cache.ProgressClient", return_value=mock_client
        ) as cls:
            c1 = await get_or_create_progress_client("test/job-1")
            c2 = await get_or_create_progress_client("test/job-1")

        assert c1 is c2
        assert cls.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_access_serialized(self) -> None:
        """Verify Lock prevents duplicate clients from concurrent access."""
        from unittest.mock import patch as mock_patch

        call_count = 0

        def make_client():
            nonlocal call_count
            call_count += 1
            c = AsyncMock()
            c.__aenter__ = AsyncMock(return_value=c)
            return c

        with mock_patch(
            "aiperf.operator.client_cache.ProgressClient", side_effect=make_client
        ):
            results = await asyncio.gather(
                get_or_create_progress_client("test/same-key"),
                get_or_create_progress_client("test/same-key"),
            )

        # Both should get the same client, only one created
        assert results[0] is results[1]
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_at_max_cache_size(self) -> None:
        """Verify oldest client evicted when cache is full."""
        from unittest.mock import patch as mock_patch

        def make_client():
            c = AsyncMock()
            c.__aenter__ = AsyncMock(return_value=c)
            c.__aexit__ = AsyncMock(return_value=None)
            return c

        with (
            mock_patch(
                "aiperf.operator.client_cache.ProgressClient", side_effect=make_client
            ),
            mock_patch("aiperf.operator.client_cache._MAX_CACHE_SIZE", 2),
        ):
            c1 = await get_or_create_progress_client("test/job-1")
            await get_or_create_progress_client("test/job-2")
            await get_or_create_progress_client("test/job-3")

        # job-1 should have been evicted
        assert "test/job-1" not in _progress_clients
        c1.__aexit__.assert_called_once_with(None, None, None)
        assert "test/job-2" in _progress_clients
        assert "test/job-3" in _progress_clients


class TestCloseProgressClient:
    """Tests for close_progress_client."""

    @pytest.mark.asyncio
    async def test_closes_and_removes(self) -> None:
        """Verify close calls __aexit__ and removes from cache."""
        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock(return_value=None)
        _progress_clients["test/job-1"] = mock_client
        _warned_pod_restarts["test/job-1"] = {("pod-1", 5)}
        _shutdown_sent.add("test/job-1")

        await close_progress_client("test/job-1")

        assert "test/job-1" not in _progress_clients
        assert "test/job-1" not in _warned_pod_restarts
        assert "test/job-1" not in _shutdown_sent
        mock_client.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_close_nonexistent_is_noop(self) -> None:
        """Verify closing a non-existent key is safe."""
        await close_progress_client("nonexistent")
        assert "nonexistent" not in _progress_clients


class TestResetForTesting:
    """Tests for _reset_for_testing."""

    def test_clears_all_state(self) -> None:
        _progress_clients["k"] = MagicMock()
        _warned_pod_restarts["k"] = set()
        _shutdown_sent.add("k")

        _reset_for_testing()

        assert len(_progress_clients) == 0
        assert len(_warned_pod_restarts) == 0
        assert len(_shutdown_sent) == 0
