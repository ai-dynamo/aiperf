# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identity fences for delayed AIPerfJob callbacks."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_API_VERSION,
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.operator.status import StatusBuilder

logger = logging.getLogger(__name__)


class StaleAIPerfJobCallback(RuntimeError):
    """Stop delayed work whose resource name now belongs to another UID."""


def body_uid(body: dict[str, Any]) -> str | None:
    """Return the immutable AIPerfJob UID carried by a kopf body snapshot."""
    uid = (body.get("metadata") or {}).get("uid")
    return str(uid) if uid is not None else None


def body_name(body: dict[str, Any], fallback: str) -> str:
    """Return the AIPerfJob resource name carried by a kopf body snapshot."""
    name = (body.get("metadata") or {}).get("name")
    return str(name) if name else fallback


async def current_aiperfjob_body(
    namespace: str,
    name: str,
    expected_uid: str | None,
) -> dict[str, Any] | None:
    """Return the live body only for the expected AIPerfJob UID.

    ``expected_uid=None`` preserves compatibility for direct helpers that do
    not originate from a CR callback. Production kopf handlers always provide
    the immutable UID and therefore fail closed on replacement or unreadable
    identity.
    """
    if expected_uid is None:
        return None
    try:
        async with k8s_client() as api:
            parent = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
            )
    except ApiException as exc:
        if exc.status == 404:
            raise StaleAIPerfJobCallback(
                f"AIPerfJob {namespace}/{name} uid={expected_uid} no longer exists"
            ) from exc
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: identity read failed "
            f"({exc.status}): {exc.reason}; retrying",
            delay=15,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: identity read failed: {exc}; retrying",
            delay=15,
        ) from exc

    metadata = parent.get("metadata") or {}
    live_uid = metadata.get("uid")
    if live_uid != expected_uid:
        raise StaleAIPerfJobCallback(
            f"AIPerfJob {namespace}/{name} uid={expected_uid} was replaced by "
            f"uid={live_uid}"
        )
    return parent


async def current_aiperfjob_resource_version(
    namespace: str,
    name: str,
    expected_uid: str | None,
) -> str | None:
    """Return the live resourceVersion only for the expected AIPerfJob UID."""
    parent = await current_aiperfjob_body(namespace, name, expected_uid)
    if parent is None:
        return None
    metadata = parent.get("metadata") or {}
    resource_version = metadata.get("resourceVersion")
    if resource_version is None:
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: identity read returned no "
            "metadata.resourceVersion; retrying",
            delay=15,
        )
    return str(resource_version)


async def owned_aiperfjob_jobset_uid(
    namespace: str,
    jobset_name: str,
    *,
    parent_name: str,
    parent_uid: str | None,
) -> str | None:
    """Return the exact owned JobSet UID, ``None`` on 404, or reject replacement."""
    if parent_uid is None:
        return None
    try:
        async with k8s_client() as api:
            jobset = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
            )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise kopf.TemporaryError(
            f"JobSet {namespace}/{jobset_name}: identity read failed "
            f"({exc.status}): {exc.reason}; retrying",
            delay=15,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"JobSet {namespace}/{jobset_name}: identity read failed: {exc}; retrying",
            delay=15,
        ) from exc

    return aiperfjob_jobset_uid(
        jobset,
        jobset_name=jobset_name,
        parent_name=parent_name,
        parent_uid=parent_uid,
    )


def aiperfjob_jobset_uid(
    jobset: dict[str, Any],
    *,
    jobset_name: str,
    parent_name: str,
    parent_uid: str,
) -> str:
    """Validate one already-read JobSet snapshot and return its immutable UID."""
    metadata = jobset.get("metadata") or {}
    if metadata.get("name") != jobset_name:
        raise StaleAIPerfJobCallback(
            f"JobSet snapshot name={metadata.get('name')} does not match {jobset_name}"
        )
    owner_references = metadata.get("ownerReferences") or []
    is_exact_owner = any(
        isinstance(ref, dict)
        and ref.get("apiVersion") == AIPERF_JOB_API_VERSION
        and ref.get("kind") == "AIPerfJob"
        and ref.get("name") == parent_name
        and ref.get("uid") == parent_uid
        and ref.get("controller") is True
        for ref in owner_references
    )
    if not is_exact_owner:
        raise StaleAIPerfJobCallback(
            f"JobSet {jobset_name} is not controlled by AIPerfJob "
            f"{parent_name} uid={parent_uid}"
        )
    jobset_uid = metadata.get("uid")
    if jobset_uid is None:
        raise kopf.TemporaryError(
            f"JobSet {jobset_name}: identity read returned no metadata.uid; retrying",
            delay=15,
        )
    return str(jobset_uid)


async def delete_owned_aiperfjob_jobset(
    namespace: str,
    jobset_name: str,
    *,
    parent_name: str,
    parent_uid: str | None,
    context: str,
) -> bool:
    """Delete only the exact JobSet controlled by one AIPerfJob identity.

    Returns ``True`` when the exact object was deleted or was already absent.
    A foreign same-name JobSet or UID-precondition conflict returns ``False``
    so callers abandon stale status side effects. Transient reads and deletes
    raise ``kopf.TemporaryError`` instead of guessing from a resource name.
    """
    if parent_uid is None:
        logger.warning(
            "Skipping unfenced JobSet delete after %s: AIPerfJob %s/%s has no UID",
            context,
            namespace,
            parent_name,
        )
        return False

    try:
        jobset_uid = await owned_aiperfjob_jobset_uid(
            namespace,
            jobset_name,
            parent_name=parent_name,
            parent_uid=parent_uid,
        )
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale JobSet delete after %s: %s", context, exc)
        return False
    if jobset_uid is None:
        return True

    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).delete_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
                body=client.V1DeleteOptions(
                    preconditions=client.V1Preconditions(uid=jobset_uid)
                ),
            )
    except ApiException as exc:
        if exc.status == 404:
            return True
        if exc.status == 409:
            logger.info(
                "Skipping stale JobSet delete after %s: %s/%s changed identity",
                context,
                namespace,
                jobset_name,
            )
            return False
        raise kopf.TemporaryError(
            f"JobSet {namespace}/{jobset_name}: delete failed after {context} "
            f"({exc.status}): {exc.reason}; retrying",
            delay=15,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"JobSet {namespace}/{jobset_name}: delete failed after {context}: "
            f"{exc}; retrying",
            delay=15,
        ) from exc
    return True


def fence_status_patch(sb: StatusBuilder, resource_version: str | None) -> None:
    """Make Kopf's eventual merge patch conditional on the live CR version."""
    if resource_version is None:
        return
    patch = sb._patch
    metadata = getattr(patch, "metadata", None)
    if metadata is not None:
        metadata["resourceVersion"] = resource_version
        return
    if isinstance(patch, dict):
        patch.setdefault("metadata", {})["resourceVersion"] = resource_version
