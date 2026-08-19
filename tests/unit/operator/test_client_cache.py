# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator client_cache module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import (
    _build_claim_patch_ops,
    _progress_clients,
    _reset_for_testing,
    _shutdown_sent,
    _submit_claim_patch,
    _warned_pod_restarts,
    close_progress_client,
    get_or_create_progress_client,
    job_key,
    try_claim_completion,
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


class TestCompletionClaim:
    """Tests for durable completion claim JSON patches."""

    def test_existing_annotations_without_claim_tests_parent_not_missing_child(
        self,
    ) -> None:
        """Absent annotation keys are claimable JSON Patch paths.

        Kubernetes rejects ``test /metadata/annotations/<missing-key>`` with
        422 because the path does not exist. The atomic guard must test the
        existing parent annotations object instead, then add the claim key.
        """
        body = {
            "metadata": {
                "annotations": {
                    "aiperf.nvidia.com/other": "value",
                }
            }
        }

        patch_ops = _build_claim_patch_ops(body)

        escaped_key = Annotations.COMPLETION_CLAIMED.replace("/", "~1")
        assert patch_ops[0] == {
            "op": "test",
            "path": "/metadata/annotations",
            "value": {"aiperf.nvidia.com/other": "value"},
        }
        assert patch_ops[1]["op"] == "add"
        assert patch_ops[1]["path"] == f"/metadata/annotations/{escaped_key}"

    def test_missing_annotations_parent_tests_resource_version_before_add(
        self,
    ) -> None:
        """Absent annotations are claimable without allowing stale overwrites.

        JSON Patch ``test /metadata/annotations == null`` fails when the
        annotations member is absent, so the patch must test an existing
        Kubernetes-managed precondition before creating the parent.
        """
        patch_ops = _build_claim_patch_ops({"metadata": {"resourceVersion": "42"}})

        assert patch_ops[0] == {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": "42",
        }
        assert patch_ops[1] == {
            "op": "add",
            "path": "/metadata/annotations",
            "value": {},
        }

    def test_missing_annotations_parent_stale_patch_fails_after_resource_version_changes(
        self,
    ) -> None:
        """Two stale absent-parent claim patches cannot both overwrite."""
        patch_ops = _build_claim_patch_ops({"metadata": {"resourceVersion": "42"}})
        live_body = {"metadata": {"resourceVersion": "42"}}

        self._apply_claim_patch(live_body, patch_ops)
        live_body["metadata"]["resourceVersion"] = "43"

        with pytest.raises(AssertionError):
            self._apply_claim_patch(live_body, patch_ops)

    def test_missing_annotations_parent_without_resource_version_tests_metadata(
        self,
    ) -> None:
        """A CR with no annotations object still gets a safe claim patch."""
        metadata = {"uid": "job-uid"}
        patch_ops = _build_claim_patch_ops({"metadata": metadata})

        # The immutable uid is now tested first, so the claim cannot land on a
        # same-name CR that replaced the one this callback observed. The
        # absent-annotations precondition still follows it.
        assert patch_ops[0] == {
            "op": "test",
            "path": "/metadata/uid",
            "value": "job-uid",
        }
        assert patch_ops[1] == {
            "op": "test",
            "path": "/metadata",
            "value": metadata,
        }
        assert patch_ops[2]["op"] == "add"
        assert patch_ops[2]["path"] == "/metadata/annotations"

    def test_missing_annotations_parent_uid_only_stale_patch_fails_after_annotations_added(
        self,
    ) -> None:
        """Two stale uid-only absent-parent claim patches cannot both overwrite."""
        patch_ops = _build_claim_patch_ops({"metadata": {"uid": "job-uid"}})
        live_body = {"metadata": {"uid": "job-uid"}}

        self._apply_claim_patch(live_body, patch_ops)

        with pytest.raises(AssertionError):
            self._apply_claim_patch(live_body, patch_ops)

    @staticmethod
    def _apply_claim_patch(
        document: dict[str, object], patch_ops: list[dict[str, object]]
    ) -> None:
        """Apply the small JSON Patch subset used by completion claims."""
        for op in patch_ops:
            parent, member = TestCompletionClaim._resolve_parent(
                document, str(op["path"])
            )
            if op["op"] == "test":
                assert parent[member] == op["value"]
            elif op["op"] == "add":
                parent[member] = op["value"]
            else:
                raise AssertionError(f"unsupported patch op: {op['op']}")

    @staticmethod
    def _resolve_parent(
        document: dict[str, object], pointer: str
    ) -> tuple[dict[str, object], str]:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer.removeprefix("/").split("/")
        ]
        parent: dict[str, object] = document
        for part in parts[:-1]:
            parent = parent[part]  # type: ignore[assignment]
        return parent, parts[-1]

    @pytest.mark.asyncio
    async def test_submit_claim_patch_422_is_retryable_error_not_lost_race(
        self,
    ) -> None:
        """422 can mean a malformed/non-race patch and must not poison cache."""
        from contextlib import asynccontextmanager

        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=422, reason="missing path")
        )

        @asynccontextmanager
        async def _fake_client():
            yield MagicMock()

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client", return_value=_fake_client()
            ),
            mock_patch(
                "aiperf.operator.client_cache.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await _submit_claim_patch("ns", "job", [])

        assert result is None

    @pytest.mark.asyncio
    async def test_retryable_claim_patch_error_does_not_mark_shutdown_sent(
        self,
    ) -> None:
        """Unexpected claim failures must allow a later monitor tick to retry."""
        with mock_patch(
            "aiperf.operator.client_cache._submit_claim_patch",
            new_callable=AsyncMock,
            return_value=None,
        ):
            claimed = await try_claim_completion("ns", "job", {"metadata": {}})

        assert claimed is False
        assert "ns/job" not in _shutdown_sent


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


class TestCacheEvictionPolicies:
    """Eviction must not close clients or clear flags that are still in use."""

    @pytest.mark.asyncio
    async def test_access_refreshes_recency(self) -> None:
        """The cache was FIFO despite its LRU docstring.

        Entries were never re-inserted on access, so the longest-lived job --
        the one most likely still running -- was always evicted first, closing
        a client mid-fetch.
        """
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
            # Touching job-1 must make job-2 the eviction candidate.
            assert await get_or_create_progress_client("test/job-1") is c1
            await get_or_create_progress_client("test/job-3")

        assert "test/job-1" in _progress_clients
        assert "test/job-2" not in _progress_clients
        c1.__aexit__.assert_not_called()

    def test_cancellation_evicts_cleared_flags_before_set_ones(self) -> None:
        """A sweep teardown must not drop live cancellation flags.

        Every child of a large sweep is cancelled within seconds, so an
        oldest-first policy evicted flags whose observers were still polling:
        those handlers stopped short-circuiting and kept patching CRs that were
        already being deleted.
        """
        from unittest.mock import patch as mock_patch

        from aiperf.operator.client_cache import (
            _cancellation_events,
            clear_cancellation,
            is_cancellation_requested,
            request_cancellation,
        )

        with mock_patch("aiperf.operator.client_cache._MAX_CACHE_SIZE", 3):
            request_cancellation("ns/live-1")
            request_cancellation("ns/live-2")
            request_cancellation("ns/stale")
            # A cleared flag is the cheap thing to forget.
            clear_cancellation("ns/stale")
            _cancellation_events["ns/stale"] = asyncio.Event()

            request_cancellation("ns/live-3")

        assert is_cancellation_requested("ns/live-1")
        assert is_cancellation_requested("ns/live-2")
        assert is_cancellation_requested("ns/live-3")
        assert "ns/stale" not in _cancellation_events

    def test_cancellation_never_evicts_set_flags_above_cache_limit(self) -> None:
        """Thousands of concurrently deleting sweep children stay cancelled."""
        from unittest.mock import patch as mock_patch

        from aiperf.operator.client_cache import (
            _cancellation_events,
            is_cancellation_requested,
            request_cancellation,
        )

        keys = [f"ns/sweep-v{i:04d}" for i in range(2_001)]
        with mock_patch("aiperf.operator.client_cache._MAX_CACHE_SIZE", 200):
            for key in keys:
                request_cancellation(key)

        assert len(_cancellation_events) == len(keys)
        assert all(is_cancellation_requested(key) for key in keys)
