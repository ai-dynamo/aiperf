# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable helpers for Kubernetes resource creation and metadata."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
        except Exception:
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
    """Create a custom resource, ignoring AlreadyExists (409)."""
    try:
        await client.CustomObjectsApi(api).create_namespaced_custom_object(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            body=body,
        )
    except ApiException as e:
        if e.status != 409:
            raise


async def create_idempotent_config_map(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create a ConfigMap, ignoring AlreadyExists (409)."""
    try:
        await client.CoreV1Api(api).create_namespaced_config_map(
            namespace=namespace, body=body
        )
    except ApiException as e:
        if e.status != 409:
            raise


async def create_idempotent_role(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create a Role, ignoring AlreadyExists (409)."""
    try:
        await client.RbacAuthorizationV1Api(api).create_namespaced_role(
            namespace=namespace, body=body
        )
    except ApiException as e:
        if e.status != 409:
            raise


async def create_idempotent_role_binding(
    api: ApiClient, body: dict[str, Any], namespace: str
) -> None:
    """Create a RoleBinding, ignoring AlreadyExists (409)."""
    try:
        await client.RbacAuthorizationV1Api(api).create_namespaced_role_binding(
            namespace=namespace, body=body
        )
    except ApiException as e:
        if e.status != 409:
            raise
