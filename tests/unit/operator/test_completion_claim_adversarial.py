# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes completion-claim races.

Focuses on:
- JSON-patch atomicity when concurrent monitor/lifecycle handlers claim completion.
- Annotation-map races where unrelated writers mutate metadata between snapshot and patch.
- Conflict re-read behavior for stale bodies, missing live claims, and apiserver errors.
- Retryability: failed claim attempts must not poison the in-process fast-path cache.

Out of scope: result-file parsing and index writes; see
``tests/unit/operator/test_completion_parse_metrics.py`` and
``tests/unit/operator/test_completion_handler.py`` for those contracts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import (
    _reset_for_testing,
    _shutdown_sent,
    job_key,
    try_claim_completion,
)

# =============================================================================
# Helpers
# =============================================================================

_FIXTURE_NAMESPACE = "aiperf-bench"
_FIXTURE_JOB = "aiperf-bench-7f2a"
_FIXTURE_KEY = f"{_FIXTURE_NAMESPACE}/{_FIXTURE_JOB}"
_OTHER_ANNOTATION = "aiperf.nvidia.com/controller-progress"


@pytest.fixture(autouse=True)
def _reset_completion_claim_state() -> None:
    """Reset process-local claim/cancellation state around each adversarial test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


@asynccontextmanager
async def _fake_k8s_client(api: MagicMock) -> AsyncIterator[MagicMock]:
    """Yield the supplied mock ApiClient without opening cluster connections."""
    yield api


def _body_with_annotations(
    *,
    annotations: dict[str, str] | None = None,
    resource_version: str = "91",
) -> dict[str, Any]:
    """Build a realistic AIPerfJob body snapshot for claim-patch tests."""
    metadata: dict[str, Any] = {
        "name": _FIXTURE_JOB,
        "namespace": _FIXTURE_NAMESPACE,
        "uid": "b0bdf9fb-5e30-4e39-a3d1-5262f1bbf9d7",
        "resourceVersion": resource_version,
        "annotations": annotations if annotations is not None else {},
    }
    return {"metadata": metadata}


def _decode_json_pointer(pointer: str) -> list[str]:
    return [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer.removeprefix("/").split("/")
    ]


def _resolve_parent(
    document: dict[str, Any],
    pointer: str,
) -> tuple[dict[str, Any], str]:
    current: Any = document
    parts = _decode_json_pointer(pointer)
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise AssertionError(
                f"JSON pointer {pointer!r} crossed non-object {part!r}"
            )
        current = current[part]
    if not isinstance(current, dict):
        raise AssertionError(f"JSON pointer {pointer!r} parent is not an object")
    return current, parts[-1]


class _AtomicAIPerfJobApi:
    """Small fake apiserver that enforces the JSON Patch ``test`` operation."""

    def __init__(self, live_body: dict[str, Any]) -> None:
        self.live_body = live_body
        self.patch_attempts = 0
        self.get_attempts = 0
        self.patch_bodies: list[list[dict[str, Any]]] = []
        self.content_types: list[str | None] = []

    async def patch_namespaced_custom_object(
        self,
        *,
        namespace: str,
        name: str,
        body: list[dict[str, Any]],
        _content_type: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.patch_attempts += 1
        self.patch_bodies.append(deepcopy(body))
        self.content_types.append(_content_type)
        await asyncio.sleep(0)
        try:
            self._apply_patch(body)
        except (AssertionError, KeyError) as exc:
            raise ApiException(
                status=422,
                reason=f"json patch precondition failed for {namespace}/{name}: {exc}",
            ) from exc
        return deepcopy(self.live_body)

    async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
        self.get_attempts += 1
        return deepcopy(self.live_body)

    def _apply_patch(self, patch_ops: list[dict[str, Any]]) -> None:
        for op in patch_ops:
            parent, member = _resolve_parent(self.live_body, str(op["path"]))
            if op["op"] == "test":
                if parent[member] != op["value"]:
                    raise AssertionError(
                        f"{op['path']} expected {op['value']!r} got {parent[member]!r}"
                    )
            elif op["op"] == "add":
                parent[member] = op["value"]
            else:
                raise AssertionError(f"unsupported patch op {op['op']!r}")


def _assert_claim_absent(body: dict[str, Any]) -> None:
    annotations = body.get("metadata", {}).get("annotations") or {}
    assert Annotations.COMPLETION_CLAIMED not in annotations


# =============================================================================
# JSON-patch atomicity
# =============================================================================


class TestCompletionClaimJsonPatchAtomicity:
    """Race the durable claim path against itself and metadata writers."""

    @pytest.mark.asyncio
    async def test_try_claim_completion_concurrent_handlers_only_one_wins(
        self,
    ) -> None:
        """Two stale handlers must not both pass the CR-level JSON-patch test."""
        live_body = _body_with_annotations(annotations={})
        fake_custom = _AtomicAIPerfJobApi(live_body)
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=fake_custom,
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                new_callable=AsyncMock,
            ),
        ):
            results = await asyncio.gather(
                try_claim_completion(
                    _FIXTURE_NAMESPACE,
                    _FIXTURE_JOB,
                    _body_with_annotations(annotations={}),
                ),
                try_claim_completion(
                    _FIXTURE_NAMESPACE,
                    _FIXTURE_JOB,
                    _body_with_annotations(annotations={}),
                ),
            )

        assert sorted(results) == [False, True]
        assert fake_custom.patch_attempts == 2
        assert fake_custom.content_types == [
            "application/json-patch+json",
            "application/json-patch+json",
        ]
        assert live_body["metadata"]["annotations"][Annotations.COMPLETION_CLAIMED]
        assert _FIXTURE_KEY in _shutdown_sent

    @pytest.mark.asyncio
    async def test_try_claim_completion_unrelated_annotation_race_defers_retry(
        self,
    ) -> None:
        """A stale annotations snapshot must fail closed without poisoning retries."""
        stale_body = _body_with_annotations(
            annotations={_OTHER_ANNOTATION: "requests-complete"},
            resource_version="91",
        )
        live_body = _body_with_annotations(
            annotations={
                _OTHER_ANNOTATION: "requests-complete",
                "aiperf.nvidia.com/system-state": "processing",
            },
            resource_version="92",
        )
        fake_custom = _AtomicAIPerfJobApi(live_body)

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=fake_custom,
            ),
        ):
            first_result = await try_claim_completion(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB, stale_body
            )
            retry_result = await try_claim_completion(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB, deepcopy(live_body)
            )

        assert first_result is False
        assert retry_result is True
        assert fake_custom.patch_attempts == 2
        assert _FIXTURE_KEY in _shutdown_sent
        assert live_body["metadata"]["annotations"][_OTHER_ANNOTATION] == (
            "requests-complete"
        )
        assert live_body["metadata"]["annotations"][
            "aiperf.nvidia.com/system-state"
        ] == ("processing")
        assert live_body["metadata"]["annotations"][Annotations.COMPLETION_CLAIMED]

    @pytest.mark.asyncio
    async def test_try_claim_completion_missing_annotation_parent_stale_resource_version_retries(
        self,
    ) -> None:
        """Absent-parent patches must not add a claim after resourceVersion changes."""
        stale_body = _body_with_annotations(annotations=None, resource_version="91")
        stale_body["metadata"].pop("annotations")
        live_body = _body_with_annotations(
            annotations={"aiperf.nvidia.com/system-state": "processing"},
            resource_version="92",
        )
        fake_custom = _AtomicAIPerfJobApi(live_body)

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=fake_custom,
            ),
        ):
            result = await try_claim_completion(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB, stale_body
            )

        assert result is False
        assert fake_custom.patch_bodies[0][0] == {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": "91",
        }
        _assert_claim_absent(live_body)
        assert _FIXTURE_KEY not in _shutdown_sent


# =============================================================================
# Conflict re-read behavior
# =============================================================================


class TestCompletionClaimConflictReRead:
    """Exercise 409 paths that decide whether a lost claim is cacheable."""

    @pytest.mark.asyncio
    async def test_try_claim_completion_conflict_without_live_claim_allows_later_win(
        self,
    ) -> None:
        """A 409 from an unrelated writer must not latch completion off forever."""
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=[ApiException(status=409, reason="resourceVersion changed"), {}]
        )
        custom.get_namespaced_custom_object = AsyncMock(
            return_value=_body_with_annotations(
                annotations={"aiperf.nvidia.com/system-state": "processing"},
                resource_version="92",
            )
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=custom,
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                new_callable=AsyncMock,
            ),
        ):
            first_result = await try_claim_completion(
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB,
                _body_with_annotations(annotations={}, resource_version="91"),
            )
            assert _FIXTURE_KEY not in _shutdown_sent
            retry_result = await try_claim_completion(
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB,
                _body_with_annotations(annotations={}, resource_version="92"),
            )

        assert first_result is False
        assert retry_result is True
        assert custom.patch_namespaced_custom_object.await_count == 2
        custom.get_namespaced_custom_object.assert_awaited_once()
        assert _FIXTURE_KEY in _shutdown_sent

    @pytest.mark.asyncio
    async def test_try_claim_completion_conflict_reread_api_error_stays_retryable(
        self,
    ) -> None:
        """If live state cannot be read, the operator must not cache a race loss."""
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=409, reason="resourceVersion changed")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=503, reason="apiserver unavailable")
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=custom,
            ),
        ):
            result = await try_claim_completion(
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB,
                _body_with_annotations(annotations={}, resource_version="91"),
            )

        assert result is False
        custom.get_namespaced_custom_object.assert_awaited_once()
        assert _FIXTURE_KEY not in _shutdown_sent

    @pytest.mark.asyncio
    async def test_try_claim_completion_conflict_with_live_claim_caches_race_loss(
        self,
    ) -> None:
        """Only a verified live completion claim should populate the fast path."""
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=409, reason="resourceVersion changed")
        )
        custom.get_namespaced_custom_object = AsyncMock(
            return_value=_body_with_annotations(
                annotations={
                    Annotations.COMPLETION_CLAIMED: "2026-05-18T12:00:00Z",
                },
                resource_version="92",
            )
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=custom,
            ),
        ):
            result = await try_claim_completion(
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB,
                _body_with_annotations(annotations={}, resource_version="91"),
            )

        assert result is False
        custom.get_namespaced_custom_object.assert_awaited_once()
        assert _FIXTURE_KEY in _shutdown_sent
        assert job_key(_FIXTURE_NAMESPACE, _FIXTURE_JOB) in _shutdown_sent
