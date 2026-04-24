# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client_jobsets — edge cases not covered elsewhere.

Focuses on gaps left by test_client.py:

- list_jobsets default-namespace fallback, non-404 re-raise
- list_jobsets label-selector construction (part-of label only vs. + job_id)
- find_jobset 404 suppressed on both passes, non-404 re-raised on each
- find_jobset field-selector string on the name-fallback pass
- delete_jobset swallow-and-warn path for non-404/409 aux failures
- delete_namespace non-404 does not raise (covered indirectly; re-asserted)
- _list_jobsets_raw None items path
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.client_jobsets import (
    delete_jobset,
    delete_namespace,
    find_jobset,
    list_jobsets,
)


def _raw_jobset(
    name: str = "js-1",
    namespace: str = "default",
    created: str = "2026-01-15T10:30:00Z",
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal raw JobSet dict."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
            "labels": {"app": "aiperf"},
        },
        "status": status or {},
    }


def _api_exception(status: int) -> ApiException:
    """Construct an ApiException with the given HTTP status code."""
    return ApiException(status=status, reason=f"err-{status}")


class TestListJobsetsSelectorConstruction:
    """Verify the composed label_selector string."""

    @pytest.mark.asyncio
    async def test_default_selector_is_part_of_aiperf(self) -> None:
        """Without job_id, only the AIPerfLabels.SELECTOR is applied."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_jobsets(api, namespace="default")
        selector = mock_custom.list_namespaced_custom_object.call_args.kwargs[
            "label_selector"
        ]
        assert selector == "app=aiperf"

    @pytest.mark.asyncio
    async def test_job_id_ands_into_selector(self) -> None:
        """job_id narrows with a comma-separated 'aiperf.nvidia.com/job-id=<id>'."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_jobsets(api, namespace="ns", job_id="abc")
        selector = mock_custom.list_namespaced_custom_object.call_args.kwargs[
            "label_selector"
        ]
        assert selector == "app=aiperf,aiperf.nvidia.com/job-id=abc"

    @pytest.mark.asyncio
    async def test_default_namespace_when_none(self) -> None:
        """namespace=None + all_namespaces=False resolves to 'default'."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_jobsets(api, namespace=None)
        kwargs = mock_custom.list_namespaced_custom_object.call_args.kwargs
        assert kwargs["namespace"] == "default"

    @pytest.mark.asyncio
    async def test_non_404_raises(self) -> None:
        """Any non-404 ApiException propagates to the caller."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await list_jobsets(api, namespace="ns")

    @pytest.mark.asyncio
    async def test_none_items_coerces_to_empty_list(self) -> None:
        """A response whose 'items' is None (not missing) yields empty list."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": None}
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(api, namespace="ns")
        assert result == []


class TestFindJobsetErrorPaths:
    """Verify error-path behavior of find_jobset's two-pass lookup."""

    @pytest.mark.asyncio
    async def test_first_pass_404_returns_none(self) -> None:
        """404 on the label-selector pass suppresses and returns None (CRD missing)."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "abc", namespace="ns")
        assert result is None

    @pytest.mark.asyncio
    async def test_first_pass_non_404_raises(self) -> None:
        """A 500 on the label-selector pass propagates."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await find_jobset(api, "abc", namespace="ns")

    @pytest.mark.asyncio
    async def test_second_pass_404_returns_none(self) -> None:
        """404 on the name-fallback pass also suppresses to None."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                {"items": []},
                _api_exception(404),
            ]
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "abc", namespace="ns")
        assert result is None

    @pytest.mark.asyncio
    async def test_second_pass_non_404_raises(self) -> None:
        """A 500 on the name-fallback pass propagates."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                {"items": []},
                _api_exception(500),
            ]
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await find_jobset(api, "abc", namespace="ns")

    @pytest.mark.asyncio
    async def test_second_pass_uses_metadata_name_field_selector(self) -> None:
        """The name-fallback pass sets field_selector='metadata.name=<job_id>'."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                {"items": []},
                {"items": [_raw_jobset(name="by-name")]},
            ]
        )
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await find_jobset(api, "abc", namespace="ns")
        second_kwargs = mock_custom.list_namespaced_custom_object.call_args_list[
            1
        ].kwargs
        assert second_kwargs["field_selector"] == "metadata.name=abc"
        # Second pass widens the selector to AIPerfLabels.SELECTOR only
        assert second_kwargs["label_selector"] == "app=aiperf"

    @pytest.mark.asyncio
    async def test_cluster_wide_when_namespace_none(self) -> None:
        """namespace=None routes both passes through list_cluster_custom_object."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset(name="found")]}
        )
        mock_custom.list_namespaced_custom_object = AsyncMock()
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "abc", namespace=None)
        mock_custom.list_cluster_custom_object.assert_awaited_once()
        mock_custom.list_namespaced_custom_object.assert_not_called()
        assert result is not None
        assert result.name == "found"


class TestDeleteJobsetNonSuppressedAuxError:
    """Verify aux-delete failures outside 404/409 log-and-continue."""

    @pytest.mark.asyncio
    async def test_500_on_aux_does_not_raise(self) -> None:
        """A 500 on the ConfigMap delete is logged and execution continues."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.delete_namespaced_custom_object = AsyncMock(return_value={})
        mock_core = MagicMock()
        mock_core.delete_namespaced_config_map = AsyncMock(
            side_effect=_api_exception(500)
        )
        mock_rbac = MagicMock()
        mock_rbac.delete_namespaced_role = AsyncMock(return_value={})
        mock_rbac.delete_namespaced_role_binding = AsyncMock(return_value={})

        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.RbacAuthorizationV1Api",
                return_value=mock_rbac,
            ),
        ):
            await delete_jobset(api, "my-js", "default")
        # Subsequent aux deletes still attempted despite the 500 above
        mock_rbac.delete_namespaced_role.assert_awaited_once()
        mock_rbac.delete_namespaced_role_binding.assert_awaited_once()


class TestDeleteNamespaceNon404:
    """Verify delete_namespace logs (not raises) on non-404 errors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(403, id="forbidden"),
            param(409, id="conflict"),
            param(500, id="server_error"),
        ],
    )  # fmt: skip
    async def test_logs_warning_no_raise(self, status: int) -> None:
        """delete_namespace never re-raises — it's best-effort cleanup."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.delete_namespace = AsyncMock(side_effect=_api_exception(status))
        with patch(
            "aiperf.kubernetes.client_jobsets.client.CoreV1Api",
            return_value=mock_core,
        ):
            await delete_namespace(api, "ns")
        mock_core.delete_namespace.assert_awaited_once_with(name="ns")
