# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable helpers for Kubernetes resource creation and metadata."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ForeignResourceOwnershipError(RuntimeError):
    """A deterministic child-resource name is occupied by another owner."""


def _metadata(resource: Any) -> Any:
    if isinstance(resource, dict):
        return resource.get("metadata") or {}
    return getattr(resource, "metadata", None)


def _controller_owner_identity(resource: Any) -> tuple[str, str, str] | None:
    """Return the immutable identity of a resource's controller owner."""
    metadata = _metadata(resource)
    if isinstance(metadata, dict):
        refs = metadata.get("ownerReferences") or metadata.get("owner_references") or []
    else:
        refs = getattr(metadata, "owner_references", None) or []
    for ref in refs:
        if isinstance(ref, dict):
            controller = ref.get("controller")
            kind = ref.get("kind")
            name = ref.get("name")
            uid = ref.get("uid")
        else:
            controller = getattr(ref, "controller", None)
            kind = getattr(ref, "kind", None)
            name = getattr(ref, "name", None)
            uid = getattr(ref, "uid", None)
        if controller is True and all(
            isinstance(value, str) for value in (kind, name, uid)
        ):
            return kind, name, uid
    return None


def _resource_name(resource: Any) -> str:
    metadata = _metadata(resource)
    if isinstance(metadata, dict):
        return str(metadata.get("name") or "<unknown>")
    return str(getattr(metadata, "name", None) or "<unknown>")


def _deletion_timestamp(resource: Any) -> Any:
    metadata = _metadata(resource)
    if isinstance(metadata, dict):
        return metadata.get("deletionTimestamp") or metadata.get("deletion_timestamp")
    return getattr(metadata, "deletion_timestamp", None)


def _require_same_controller_owner(existing: Any, desired: Any) -> None:
    """Reject a same-named resource left behind by another CR incarnation."""
    expected = _controller_owner_identity(desired)
    if expected is None:
        return
    actual = _controller_owner_identity(existing)
    if actual == expected and _deletion_timestamp(existing) is None:
        return
    expected_kind, expected_name, expected_uid = expected
    message = (
        f"{_resource_name(desired)} already exists with controller owner "
        f"{actual!r}; expected ({expected_kind!r}, {expected_name!r}, "
        f"{expected_uid!r})"
    )
    if actual == expected:
        raise ApiException(status=409, reason=f"{message}; resource is terminating")
    if actual is not None and actual[:2] == expected[:2]:
        raise ApiException(status=409, reason=message)
    raise ForeignResourceOwnershipError(message)


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    description: str = "operation",
) -> T:
    """Retry an async operation with exponential backoff and jitter.

    Args:
        coro_factory: Zero-arg callable returning an awaitable (called each attempt).
        max_retries: Maximum number of retry attempts after the first failure.
        initial_delay: Seconds to wait before the first retry.
        max_delay: Maximum backoff cap in seconds.
        backoff_multiplier: Multiplier applied to the delay after each retry.
        description: Human-readable label for log messages.

    Returns:
        The result of the first successful call.

    Raises:
        The exception from the final attempt if all retries are exhausted.
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except (
            TimeoutError,
            ApiException,
            aiohttp.ClientError,
            ConnectionError,
            OSError,
        ):
            if attempt >= max_retries:
                raise
            jittered_delay = delay * random.uniform(0.8, 1.2)
            logger.debug(
                "%s attempt %d/%d failed, retrying in %.1fs",
                description,
                attempt + 1,
                max_retries + 1,
                jittered_delay,
            )
            await asyncio.sleep(jittered_delay)
            delay = min(delay * backoff_multiplier, max_delay)

    # Unreachable, but satisfies the type checker
    raise RuntimeError(
        f"{description} failed after {max_retries + 1} attempts"
    )  # pragma: no cover


async def create_idempotent_custom_object(
    api: ApiClient,
    *,
    group: str,
    version: str,
    plural: str,
    body: dict[str, Any],
    namespace: str,
) -> None:
    """Create or adopt a custom resource owned by the same CR incarnation."""
    custom = client.CustomObjectsApi(api)
    try:
        await custom.create_namespaced_custom_object(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            body=body,
        )
    except ApiException as e:
        if e.status != 409:
            raise
        if _controller_owner_identity(body) is None:
            return
        existing = await custom.get_namespaced_custom_object(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            name=_resource_name(body),
        )
        _require_same_controller_owner(existing, body)


async def create_idempotent_config_map(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create or adopt a ConfigMap owned by the same CR incarnation."""
    core = client.CoreV1Api(api)
    try:
        await core.create_namespaced_config_map(namespace=namespace, body=body)
    except ApiException as e:
        if e.status != 409:
            raise
        if _controller_owner_identity(body) is None:
            return
        existing = await core.read_namespaced_config_map(
            name=_resource_name(body), namespace=namespace
        )
        _require_same_controller_owner(existing, body)


async def create_idempotent_role(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create or adopt a Role owned by the same CR incarnation."""
    rbac = client.RbacAuthorizationV1Api(api)
    try:
        await rbac.create_namespaced_role(namespace=namespace, body=body)
    except ApiException as e:
        if e.status != 409:
            raise
        if _controller_owner_identity(body) is None:
            return
        existing = await rbac.read_namespaced_role(
            name=_resource_name(body), namespace=namespace
        )
        _require_same_controller_owner(existing, body)


async def create_idempotent_role_binding(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create or adopt a RoleBinding owned by the same CR incarnation."""
    rbac = client.RbacAuthorizationV1Api(api)
    try:
        await rbac.create_namespaced_role_binding(namespace=namespace, body=body)
    except ApiException as e:
        if e.status != 409:
            raise
        if _controller_owner_identity(body) is None:
            return
        existing = await rbac.read_namespaced_role_binding(
            name=_resource_name(body), namespace=namespace
        )
        _require_same_controller_owner(existing, body)
