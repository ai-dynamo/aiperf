# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.completion_signal.

``signal_benchmark_complete`` patches an AIPerfJob CR annotation to notify
the operator that the benchmark is done. Failure modes that must NOT raise:
(a) not running in K8s (env vars unset); (b) transient API errors.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes import completion_signal
from aiperf.kubernetes.constants import Annotations


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure AIPERF_JOB_ID / AIPERF_NAMESPACE don't bleed from the host env."""
    monkeypatch.delenv("AIPERF_JOB_ID", raising=False)
    monkeypatch.delenv("AIPERF_NAMESPACE", raising=False)


def _k8s_client_contextmanager(api_mock):
    """Build an ``@asynccontextmanager`` that yields ``api_mock``.

    Mirrors the real ``k8s_client()`` shape used by production.
    """

    @asynccontextmanager
    async def _cm():
        yield api_mock

    return _cm


def _k8s_client_raising(exc: BaseException):
    """Build an async contextmanager whose ``__aenter__`` raises ``exc``."""

    @asynccontextmanager
    async def _cm():
        raise exc
        yield  # pragma: no cover — unreachable; keeps the function an async gen

    return _cm


class TestEnvironmentGating:
    """No K8s env vars → no-op (used on local dev / bare-metal runs)."""

    @pytest.mark.asyncio
    async def test_returns_false_when_neither_env_set(self) -> None:
        assert await completion_signal.signal_benchmark_complete() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_only_job_id_set(self, monkeypatch) -> None:
        monkeypatch.setenv("AIPERF_JOB_ID", "abc")
        assert await completion_signal.signal_benchmark_complete() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_only_namespace_set(self, monkeypatch) -> None:
        monkeypatch.setenv("AIPERF_NAMESPACE", "default")
        assert await completion_signal.signal_benchmark_complete() is False

    @pytest.mark.asyncio
    async def test_does_not_call_k8s_api_when_env_missing(self) -> None:
        """Must short-circuit before opening a client; otherwise off-cluster
        calls trigger a DNS lookup and a slow failure."""
        called = False

        def _fake_k8s_client():
            nonlocal called
            called = True
            raise AssertionError("k8s_client() must not be invoked when env is missing")

        with patch(
            "aiperf.kubernetes.client.k8s_client",
            new=_fake_k8s_client,
        ):
            await completion_signal.signal_benchmark_complete()
        assert not called


class TestPatchRequest:
    """With env vars set, the function patches the CR via CustomObjectsApi."""

    @pytest.fixture
    def patched_api(self, monkeypatch):
        """Install env vars + stub ``CustomObjectsApi.patch_namespaced_custom_object``."""
        monkeypatch.setenv("AIPERF_JOB_ID", "my-job")
        monkeypatch.setenv("AIPERF_NAMESPACE", "bench")

        # The production code does:
        #   async with k8s_client() as api:
        #       await client.CustomObjectsApi(api).patch_namespaced_custom_object(...)
        # We intercept the CustomObjectsApi constructor so we can assert on the
        # patch call without caring about the ApiClient instance it wraps.
        mock_patch_call = AsyncMock()
        mock_custom_api = MagicMock()
        mock_custom_api.patch_namespaced_custom_object = mock_patch_call

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_k8s_client_contextmanager(MagicMock()),
            ),
            patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom_api,
            ),
        ):
            yield mock_patch_call

    @pytest.mark.asyncio
    async def test_returns_true_on_successful_patch(self, patched_api) -> None:
        assert await completion_signal.signal_benchmark_complete() is True

    @pytest.mark.asyncio
    async def test_patch_targets_correct_crd(self, patched_api) -> None:
        """Patch must address the AIPerfJob CR by group/version/plural + name."""
        await completion_signal.signal_benchmark_complete()

        _args, kwargs = patched_api.call_args
        assert kwargs["group"] == "aiperf.nvidia.com"
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["namespace"] == "bench"
        assert kwargs["name"] == "my-job"

    @pytest.mark.asyncio
    async def test_patch_body_sets_benchmark_complete_annotation(
        self, patched_api
    ) -> None:
        await completion_signal.signal_benchmark_complete()

        _args, kwargs = patched_api.call_args
        annotations = kwargs["body"]["metadata"]["annotations"]
        assert annotations == {Annotations.BENCHMARK_COMPLETE: "true"}

    @pytest.mark.asyncio
    async def test_patch_uses_merge_patch_content_type(self, patched_api) -> None:
        """Strategic-merge-patch would require a schema; merge-patch is correct."""
        await completion_signal.signal_benchmark_complete()

        _args, kwargs = patched_api.call_args
        assert kwargs["_content_type"] == "application/merge-patch+json"


class TestErrorSwallowing:
    """Transient network/API errors must not crash the controller shutdown path."""

    @pytest.fixture(autouse=True)
    def _k8s_env(self, monkeypatch):
        monkeypatch.setenv("AIPERF_JOB_ID", "my-job")
        monkeypatch.setenv("AIPERF_NAMESPACE", "bench")

    @pytest.mark.asyncio
    async def test_api_exception_returns_false_without_raising(self) -> None:
        with patch(
            "aiperf.kubernetes.client.k8s_client",
            new=_k8s_client_raising(ApiException(status=500, reason="boom")),
        ):
            result = await completion_signal.signal_benchmark_complete()

        assert result is False

    @pytest.mark.asyncio
    async def test_aiohttp_error_returns_false_without_raising(self) -> None:
        with patch(
            "aiperf.kubernetes.client.k8s_client",
            new=_k8s_client_raising(aiohttp.ClientError("boom")),
        ):
            result = await completion_signal.signal_benchmark_complete()

        assert result is False

    @pytest.mark.asyncio
    async def test_os_error_returns_false_without_raising(self) -> None:
        with patch(
            "aiperf.kubernetes.client.k8s_client",
            new=_k8s_client_raising(OSError("unreachable")),
        ):
            result = await completion_signal.signal_benchmark_complete()

        assert result is False

    @pytest.mark.asyncio
    async def test_unrelated_exception_still_propagates(self) -> None:
        """Only the three declared exception types are swallowed — anything
        else (e.g. a bug in our own code) must bubble up so it's debuggable."""
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_k8s_client_raising(RuntimeError("unexpected")),
            ),
            pytest.raises(RuntimeError),
        ):
            await completion_signal.signal_benchmark_complete()
