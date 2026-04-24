# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.k8s_helpers``.

Covers :func:`retry_with_backoff` success/retry/exhaustion paths and the
``create_idempotent_*`` wrappers' 409-swallow vs. non-409-reraise behaviour.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator.k8s_helpers import (
    create_idempotent_config_map,
    create_idempotent_custom_object,
    create_idempotent_role,
    create_idempotent_role_binding,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Tests for ``retry_with_backoff``."""

    @pytest.mark.asyncio
    async def test_returns_result_on_first_success(self) -> None:
        """Verify no retry when the first attempt succeeds."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(op, max_retries=3, initial_delay=0.0)

        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        """Verify transient failures are retried and the eventual success returned."""
        calls = 0

        async def op() -> int:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return 42

        with mock_patch("aiperf.operator.k8s_helpers.asyncio.sleep", new=AsyncMock()):
            result = await retry_with_backoff(
                op, max_retries=5, initial_delay=0.0, max_delay=0.0
            )

        assert result == 42
        assert calls == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self) -> None:
        """Verify the last exception is propagated after retries are exhausted."""
        calls = 0

        async def op() -> None:
            nonlocal calls
            calls += 1
            raise ValueError(f"attempt-{calls}")

        with (
            mock_patch("aiperf.operator.k8s_helpers.asyncio.sleep", new=AsyncMock()),
            pytest.raises(ValueError, match="attempt-3"),
        ):
            await retry_with_backoff(
                op, max_retries=2, initial_delay=0.0, max_delay=0.0
            )

        # max_retries=2 => 1 initial + 2 retries = 3 attempts total
        assert calls == 3

    @pytest.mark.asyncio
    async def test_applies_jittered_backoff_between_attempts(self) -> None:
        """Verify a sleep is awaited between every retry."""
        attempts = 0

        async def op() -> None:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("no")

        sleep_mock = AsyncMock()
        with (
            mock_patch("aiperf.operator.k8s_helpers.asyncio.sleep", new=sleep_mock),
            pytest.raises(RuntimeError),
        ):
            await retry_with_backoff(
                op, max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
            )

        assert attempts == 4
        # One sleep between each of 3 retries.
        assert sleep_mock.await_count == 3


class TestCreateIdempotentHelpers:
    """Tests for ``create_idempotent_*`` wrappers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "func_path,api_class,api_method",
        [
            param(
                "aiperf.operator.k8s_helpers.client.CoreV1Api",
                "CoreV1Api",
                "create_namespaced_config_map",
                id="config_map",
            ),
            param(
                "aiperf.operator.k8s_helpers.client.RbacAuthorizationV1Api",
                "RbacAuthorizationV1Api",
                "create_namespaced_role",
                id="role",
            ),
            param(
                "aiperf.operator.k8s_helpers.client.RbacAuthorizationV1Api",
                "RbacAuthorizationV1Api",
                "create_namespaced_role_binding",
                id="role_binding",
            ),
        ],
    )  # fmt: skip
    async def test_simple_helper_swallows_409(
        self, func_path: str, api_class: str, api_method: str
    ) -> None:
        """Verify each helper ignores ApiException 409 (AlreadyExists)."""
        api = MagicMock()
        api_instance = MagicMock()
        method = AsyncMock(side_effect=ApiException(status=409, reason="AlreadyExists"))
        setattr(api_instance, api_method, method)

        func_map = {
            "create_namespaced_config_map": create_idempotent_config_map,
            "create_namespaced_role": create_idempotent_role,
            "create_namespaced_role_binding": create_idempotent_role_binding,
        }
        target_func = func_map[api_method]

        with mock_patch(func_path, return_value=api_instance):
            await target_func(api, body={"kind": "X"}, namespace="ns")

        method.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_config_map_reraises_non_409(self) -> None:
        """Verify 500 from the apiserver is propagated."""
        api = MagicMock()
        api_instance = MagicMock()
        api_instance.create_namespaced_config_map = AsyncMock(
            side_effect=ApiException(status=500, reason="ServerError")
        )

        with (
            mock_patch(
                "aiperf.operator.k8s_helpers.client.CoreV1Api",
                return_value=api_instance,
            ),
            pytest.raises(ApiException) as exc,
        ):
            await create_idempotent_config_map(api, body={}, namespace="ns")

        assert exc.value.status == 500

    @pytest.mark.asyncio
    async def test_custom_object_swallows_409(self) -> None:
        """Verify create_idempotent_custom_object ignores 409."""
        api = MagicMock()
        api_instance = MagicMock()
        api_instance.create_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=409, reason="AlreadyExists")
        )

        with mock_patch(
            "aiperf.operator.k8s_helpers.client.CustomObjectsApi",
            return_value=api_instance,
        ):
            await create_idempotent_custom_object(
                api,
                group="jobset.x-k8s.io",
                version="v1alpha2",
                plural="jobsets",
                body={"kind": "JobSet"},
                namespace="ns",
            )

        api_instance.create_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_object_reraises_non_409(self) -> None:
        """Verify create_idempotent_custom_object propagates non-409 errors."""
        api = MagicMock()
        api_instance = MagicMock()
        api_instance.create_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=422, reason="Invalid")
        )

        with (
            mock_patch(
                "aiperf.operator.k8s_helpers.client.CustomObjectsApi",
                return_value=api_instance,
            ),
            pytest.raises(ApiException) as exc,
        ):
            await create_idempotent_custom_object(
                api,
                group="g",
                version="v",
                plural="p",
                body={},
                namespace="ns",
            )

        assert exc.value.status == 422

    @pytest.mark.asyncio
    async def test_custom_object_success_path(self) -> None:
        """Verify create_idempotent_custom_object returns normally on success."""
        api = MagicMock()
        api_instance = MagicMock()
        api_instance.create_namespaced_custom_object = AsyncMock(return_value=None)

        with mock_patch(
            "aiperf.operator.k8s_helpers.client.CustomObjectsApi",
            return_value=api_instance,
        ):
            await create_idempotent_custom_object(
                api,
                group="g",
                version="v",
                plural="p",
                body={"kind": "Thing"},
                namespace="ns",
            )

        api_instance.create_namespaced_custom_object.assert_awaited_once_with(
            group="g",
            version="v",
            plural="p",
            namespace="ns",
            body={"kind": "Thing"},
        )
