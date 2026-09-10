# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants and helper utilities for operator preflight checks."""

from __future__ import annotations

import errno
from typing import Any

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import JOBSET_GROUP
from aiperf.kubernetes.utils import parse_cpu, parse_memory_gib

# Minimum supported Kubernetes version
MIN_K8S_MAJOR = 1
MIN_K8S_MINOR = 24

# Required RBAC permissions for the operator to manage resources.
# (verb, resource, api_group)
OPERATOR_RBAC_PERMISSIONS: list[tuple[str, str, str]] = [
    # Core resources
    ("create", "configmaps", ""),
    ("get", "configmaps", ""),
    ("delete", "configmaps", ""),
    ("create", "roles", "rbac.authorization.k8s.io"),
    ("create", "rolebindings", "rbac.authorization.k8s.io"),
    ("get", "pods", ""),
    ("list", "pods", ""),
    ("get", "pods/log", ""),
    ("create", "events", ""),
    ("patch", "events", ""),
    # JobSet resources
    ("create", "jobsets", JOBSET_GROUP),
    ("get", "jobsets", JOBSET_GROUP),
    ("delete", "jobsets", JOBSET_GROUP),
    ("watch", "jobsets", JOBSET_GROUP),
    ("get", "jobsets/status", JOBSET_GROUP),
]

# Known public registries that don't need pull secrets
PUBLIC_REGISTRIES = frozenset(
    {
        "docker.io",
        "registry-1.docker.io",
        "ghcr.io",
        "quay.io",
        "nvcr.io",
        "registry.k8s.io",
    }
)


def controller_resource_requirements() -> tuple[float, float]:
    """Return total controller-pod CPU cores and memory GiB."""
    from aiperf.kubernetes.environment import (
        CONTROLLER_RESOURCE_KEYS,
        K8sEnvironment,
    )

    cpu = 0.0
    memory = 0.0
    for key in CONTROLLER_RESOURCE_KEYS:
        settings = getattr(K8sEnvironment, key)
        cpu += parse_cpu(settings.CPU)
        memory += parse_memory_gib(settings.MEMORY)
    return cpu, memory


def _is_node_ready_typed(node: Any) -> bool:
    """Check if a typed V1Node indicates Ready status."""
    if not node.status:
        return False
    conditions = node.status.conditions or []
    return any(c.type == "Ready" and c.status == "True" for c in conditions)


_TRANSIENT_OS_ERRNOS: frozenset[int] = frozenset(
    {errno.ECONNREFUSED, errno.ECONNRESET, errno.ETIMEDOUT, errno.EHOSTUNREACH}
)


def _is_transient_error(exc: BaseException) -> bool:
    """Classify whether ``exc`` is a transient cluster-API error worth retrying.

    Transient errors degrade a check to WARN (the operator will retry on the
    next reconcile). Everything else is a permanent FAIL.
    """
    if isinstance(exc, TimeoutError):
        return True
    # aiohttp connector errors (ClientConnectorError, ServerConnectionError, etc.)
    if isinstance(exc, aiohttp.ClientConnectionError):
        return True
    if isinstance(exc, ApiException):
        status = getattr(exc, "status", None)
        # 429 is the apiserver's own back-pressure signal (it ships a
        # Retry-After header), so it is retryable for the same reason 5xx is.
        return bool(status and (status >= 500 or status == 429))
    if isinstance(exc, OSError):
        return exc.errno in _TRANSIENT_OS_ERRNOS
    return False
