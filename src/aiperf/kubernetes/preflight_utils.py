# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pre-flight check utilities.

Functions used by both the CLI ``CLIPreflightChecker`` and the operator
``OperatorPreflightChecker``.
"""

from __future__ import annotations

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1ResourceAttributes,
    V1SelfSubjectAccessReview,
    V1SelfSubjectAccessReviewSpec,
)


async def check_rbac_access(
    api: ApiClient,
    *,
    verb: str,
    resource: str,
    group: str,
    namespace: str,
) -> bool:
    """Check if current user/service-account has a specific RBAC permission.

    Submits a ``SelfSubjectAccessReview`` to the API server.

    Args:
        api: kubernetes_asyncio ApiClient.
        verb: Kubernetes verb (e.g. "create", "get", "delete").
        resource: Resource type (e.g. "pods", "configmaps").
        group: API group (e.g. "rbac.authorization.k8s.io", or "" for core).
        namespace: Namespace to check the permission in.

    Returns:
        True if the access is allowed, False if explicitly denied
        (``status.allowed`` is False or None).

    Raises:
        ApiException: When the API server returns a malformed response with
            ``status=None``. The caller should treat this as a transient
            error (WARN/SKIP), not a definitive denial — a missing ``status``
            block means the request didn't successfully complete review,
            distinct from an explicit ``allowed=False``.
    """
    body = V1SelfSubjectAccessReview(
        spec=V1SelfSubjectAccessReviewSpec(
            resource_attributes=V1ResourceAttributes(
                verb=verb,
                resource=resource,
                namespace=namespace,
                group=group or None,
            ),
        ),
    )
    review = await client.AuthorizationV1Api(api).create_self_subject_access_review(
        body=body,
    )
    if review.status is None:
        raise ApiException(
            status=500,
            reason=(
                f"SelfSubjectAccessReview for {verb} {resource} returned no "
                f"status block; treating as transient apiserver error."
            ),
        )
    return bool(review.status.allowed)


def parse_image_ref(image: str) -> tuple[str, str, str, str]:
    """Parse a container image reference into (registry, repository, tag, digest).

    A reference may carry either a tag (``image:1.0``), a digest
    (``image@sha256:...``), or both (``image:1.0@sha256:...``); the digest is
    always returned in its own slot. Empty strings indicate the field was not
    present, so callers can distinguish ``digest=""`` (use a tag, possibly
    implicit ``latest``) from ``digest="sha256:..."`` (immutable reference, no
    implicit-``latest`` warning needed).

    Args:
        image: Image reference like "nvcr.io/nvidia/tritonserver:24.01" or
            "repo/img@sha256:abc123".

    Returns:
        Tuple of (registry, repository, tag, digest). Registry defaults to
        "docker.io" for short names. ``tag`` and ``digest`` default to "" if
        absent.
    """
    digest = ""
    tag = ""
    remainder = image
    if "@" in remainder:
        remainder, digest = remainder.rsplit("@", 1)
    if ":" in remainder.split("/")[-1]:
        remainder, tag = remainder.rsplit(":", 1)

    parts = remainder.split("/")
    if len(parts) == 1 or ("." not in parts[0] and ":" not in parts[0]):
        return "docker.io", remainder, tag, digest

    return parts[0], "/".join(parts[1:]), tag, digest
