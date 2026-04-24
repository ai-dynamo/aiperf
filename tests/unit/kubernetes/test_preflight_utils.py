# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.preflight_utils module.

Covers:
- check_rbac_access(): SelfSubjectAccessReview with allowed/denied, group inclusion, error propagation
- parse_image_ref(): registry/repo/tag extraction for various image reference formats
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from pytest import param

from aiperf.kubernetes.preflight_utils import check_rbac_access, parse_image_ref

# =============================================================================
# Helpers
# =============================================================================


def _mock_review(allowed: bool | None) -> Any:
    """Build a mock V1SelfSubjectAccessReview-like object with status.allowed."""
    review = MagicMock()
    if allowed is None:
        review.status = None
    else:
        review.status = MagicMock()
        review.status.allowed = allowed
    return review


def _make_authz(
    review: Any | None = None, *, side_effect: Exception | None = None
) -> MagicMock:
    """Build a mock AuthorizationV1Api that returns ``review`` or raises."""
    authz = MagicMock()
    if side_effect is not None:
        authz.create_self_subject_access_review = AsyncMock(side_effect=side_effect)
    else:
        authz.create_self_subject_access_review = AsyncMock(return_value=review)
    return authz


def _patch_authz(mock_authz: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_utils.client.AuthorizationV1Api",
        return_value=mock_authz,
    )


# =============================================================================
# check_rbac_access
# =============================================================================


class TestCheckRbacAccess:
    """Tests for the check_rbac_access utility function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_allowed(self) -> None:
        """Verify True is returned when the SelfSubjectAccessReview says allowed."""
        api = MagicMock(spec=ApiClient)
        with _patch_authz(_make_authz(_mock_review(True))):
            result = await check_rbac_access(
                api, verb="create", resource="pods", group="", namespace="test-ns"
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_allowed(self) -> None:
        """Verify False is returned when the SelfSubjectAccessReview says not allowed."""
        api = MagicMock(spec=ApiClient)
        with _patch_authz(_make_authz(_mock_review(False))):
            result = await check_rbac_access(
                api, verb="delete", resource="pods", group="", namespace="test-ns"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_status_missing(self) -> None:
        """Verify False is returned when the review has no status field."""
        api = MagicMock(spec=ApiClient)
        with _patch_authz(_make_authz(_mock_review(None))):
            result = await check_rbac_access(
                api, verb="get", resource="pods", group="", namespace="test-ns"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_allowed_missing(self) -> None:
        """Verify False is returned when status.allowed is None."""
        api = MagicMock(spec=ApiClient)
        review = MagicMock()
        review.status = MagicMock()
        review.status.allowed = None
        with _patch_authz(_make_authz(review)):
            result = await check_rbac_access(
                api, verb="get", resource="pods", group="", namespace="test-ns"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_propagates_exception_from_api(self) -> None:
        """Verify exceptions from create_self_subject_access_review bubble up."""
        api = MagicMock(spec=ApiClient)
        with (
            _patch_authz(_make_authz(side_effect=RuntimeError("network failure"))),
            pytest.raises(RuntimeError, match="network failure"),
        ):
            await check_rbac_access(
                api, verb="create", resource="pods", group="", namespace="test-ns"
            )

    @pytest.mark.asyncio
    async def test_includes_group_when_nonempty(self) -> None:
        """Verify the group field is included in resourceAttributes only when non-empty."""
        captured: dict[str, Any] = {}

        async def _capture(body: Any, **kwargs: Any) -> Any:
            captured["body"] = body
            return _mock_review(True)

        authz = MagicMock()
        authz.create_self_subject_access_review = _capture

        api = MagicMock(spec=ApiClient)
        with _patch_authz(authz):
            await check_rbac_access(
                api,
                verb="create",
                resource="jobsets",
                group="jobset.x-k8s.io",
                namespace="test-ns",
            )

        body = captured["body"]
        assert body.spec.resource_attributes.group == "jobset.x-k8s.io"

    @pytest.mark.asyncio
    async def test_excludes_group_when_empty(self) -> None:
        """Verify the group field is None in resourceAttributes when group is empty."""
        captured: dict[str, Any] = {}

        async def _capture(body: Any, **kwargs: Any) -> Any:
            captured["body"] = body
            return _mock_review(True)

        authz = MagicMock()
        authz.create_self_subject_access_review = _capture

        api = MagicMock(spec=ApiClient)
        with _patch_authz(authz):
            await check_rbac_access(
                api, verb="get", resource="pods", group="", namespace="test-ns"
            )

        body = captured["body"]
        assert body.spec.resource_attributes.group is None

    @pytest.mark.asyncio
    async def test_namespace_passed_correctly(self) -> None:
        """Verify the namespace is included in the resource attributes."""
        captured: dict[str, Any] = {}

        async def _capture(body: Any, **kwargs: Any) -> Any:
            captured["body"] = body
            return _mock_review(True)

        authz = MagicMock()
        authz.create_self_subject_access_review = _capture

        api = MagicMock(spec=ApiClient)
        with _patch_authz(authz):
            await check_rbac_access(
                api,
                verb="list",
                resource="configmaps",
                group="",
                namespace="my-namespace",
            )

        body = captured["body"]
        assert body.spec.resource_attributes.namespace == "my-namespace"

    @pytest.mark.asyncio
    async def test_verb_and_resource_passed_correctly(self) -> None:
        """Verify verb and resource are included in the resource attributes."""
        captured: dict[str, Any] = {}

        async def _capture(body: Any, **kwargs: Any) -> Any:
            captured["body"] = body
            return _mock_review(True)

        authz = MagicMock()
        authz.create_self_subject_access_review = _capture

        api = MagicMock(spec=ApiClient)
        with _patch_authz(authz):
            await check_rbac_access(
                api, verb="delete", resource="secrets", group="", namespace="ns"
            )

        body = captured["body"]
        assert body.spec.resource_attributes.verb == "delete"
        assert body.spec.resource_attributes.resource == "secrets"


# =============================================================================
# parse_image_ref
# =============================================================================


class TestParseImageRef:
    """Tests for the parse_image_ref utility function."""

    @pytest.mark.parametrize(
        "image,expected",
        [
            param(
                "nvcr.io/nvidia/tritonserver:24.01",
                ("nvcr.io", "nvidia/tritonserver", "24.01"),
                id="full-image-ref",
            ),
            param(
                "python:3.10",
                ("docker.io", "python", "3.10"),
                id="short-docker-hub-name",
            ),
            param(
                "nvcr.io/nvidia/aiperf",
                ("nvcr.io", "nvidia/aiperf", ""),
                id="no-tag",
            ),
            param(
                "repo/img@sha256:abc123",
                ("docker.io", "repo/img", "sha256:abc123"),
                id="digest",
            ),
            param(
                "nginx",
                ("docker.io", "nginx", ""),
                id="single-word",
            ),
            param(
                "nginx:latest",
                ("docker.io", "nginx", "latest"),
                id="single-word-with-tag",
            ),
            param(
                "gcr.io/project/team/img:v1",
                ("gcr.io", "project/team/img", "v1"),
                id="multi-level-repo",
            ),
            param(
                "localhost:5000/myimage:v1",
                ("localhost:5000", "myimage", "v1"),
                id="port-in-registry",
            ),
        ],
    )  # fmt: skip
    def test_parse_image_ref(self, image: str, expected: tuple[str, str, str]) -> None:
        """Verify parse_image_ref correctly splits registry, repo, and tag."""
        assert parse_image_ref(image) == expected
