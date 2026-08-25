# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the destructive kube verbs (delete, cleanup)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_KIND,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    AIPERF_SWEEP_KIND,
    AIPERF_SWEEP_PLURAL,
    AIPerfWorkloadKind,
)

__all__ = [
    "AIPERF_PLURALS",
    "AmbiguousAIPerfTargetError",
    "CliWorkloadKind",
    "NamespaceDeleteIdentity",
    "confirm_action",
    "delete_aiperf_cr",
    "delete_namespace_if_unchanged",
    "find_aiperf_cr",
    "find_deletable_namespace",
    "kind_for_plural",
    "list_aiperf_crs",
    "workload_kind_from_cli",
]

AIPERF_PLURALS: tuple[str, ...] = (AIPERF_JOB_PLURAL, AIPERF_SWEEP_PLURAL)
"""Both AIPerf kinds, job first: a name can only collide across kinds."""

CliWorkloadKind: TypeAlias = Literal["job", "sweep"]
"""Short kind names accepted by destructive CLI commands."""

_KIND_TO_PLURAL: dict[AIPerfWorkloadKind, str] = {
    AIPERF_JOB_KIND: AIPERF_JOB_PLURAL,
    AIPERF_SWEEP_KIND: AIPERF_SWEEP_PLURAL,
}
_PLURAL_TO_KIND: dict[str, AIPerfWorkloadKind] = {
    plural: kind for kind, plural in _KIND_TO_PLURAL.items()
}


class AmbiguousAIPerfTargetError(ValueError):
    """Raised when both workload kinds use the requested name."""


@dataclass(frozen=True, slots=True)
class NamespaceDeleteIdentity:
    """Kubernetes identity captured with an owned namespace marker."""

    uid: str
    """Namespace UID observed during the ownership check."""

    resource_version: str
    """Namespace resource version observed with the ownership marker."""


def workload_kind_from_cli(kind: CliWorkloadKind | None) -> AIPerfWorkloadKind | None:
    """Map a short CLI kind to the canonical Kubernetes resource kind."""
    if kind == "job":
        return AIPERF_JOB_KIND
    if kind == "sweep":
        return AIPERF_SWEEP_KIND
    return None


def kind_for_plural(plural: str) -> AIPerfWorkloadKind:
    """Return the canonical resource kind for an AIPerf plural."""
    return _PLURAL_TO_KIND[plural]


async def find_deletable_namespace(
    core: Any,
    *,
    namespace: str,
    job_id: str,
) -> NamespaceDeleteIdentity | None:
    """Return the stable identity of a CLI-owned, job-specific namespace."""
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.constants import AIPerfLabels

    if namespace != f"aiperf-{job_id}":
        kube_console.print_info(
            f"Namespace {namespace} was not generated for this benchmark; "
            "leaving it in place."
        )
        return None

    try:
        namespace_obj = await core.read_namespace(name=namespace)
    except ApiException as error:
        if error.status == 404:
            kube_console.print_info(f"Namespace {namespace} already gone")
        else:
            kube_console.print_warning(
                f"Could not verify ownership of namespace {namespace}: "
                f"{error.reason}; leaving it in place."
            )
        return None

    metadata = namespace_obj.metadata
    labels = metadata.labels or {}
    if labels.get(AIPerfLabels.AUTO_GENERATED) != "true":
        kube_console.print_info(
            f"Namespace {namespace} does not carry AIPerf's auto-generated marker; "
            "leaving it in place."
        )
        return None
    if labels.get(AIPerfLabels.JOB_ID) != job_id:
        kube_console.print_info(
            f"Namespace {namespace} is not owned by benchmark {job_id}; "
            "leaving it in place."
        )
        return None
    if not metadata.uid or not metadata.resource_version:
        kube_console.print_warning(
            f"Namespace {namespace} has no stable Kubernetes identity; "
            "leaving it in place."
        )
        return None
    return NamespaceDeleteIdentity(
        uid=metadata.uid,
        resource_version=metadata.resource_version,
    )


async def delete_namespace_if_unchanged(
    core: Any,
    *,
    namespace: str,
    identity: NamespaceDeleteIdentity,
) -> None:
    """Delete an owned namespace only while its checked identity is unchanged."""
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console

    try:
        await core.delete_namespace(
            name=namespace,
            body=k8s_client_mod.V1DeleteOptions(
                preconditions=k8s_client_mod.V1Preconditions(
                    uid=identity.uid,
                    resource_version=identity.resource_version,
                )
            ),
        )
        kube_console.print_success(f"Deleted namespace {namespace}")
    except ApiException as error:
        if error.status == 404:
            kube_console.print_info(f"Namespace {namespace} already gone")
        else:
            kube_console.print_warning(
                f"Could not delete namespace {namespace}: {error.reason}"
            )


async def delete_aiperf_cr(
    custom: Any,
    *,
    plural: str,
    kind: AIPerfWorkloadKind,
    namespace: str,
    name: str,
    cr: dict,
) -> bool:
    """Delete the exact CR observed during resolution using API preconditions."""
    from kubernetes_asyncio import client as k8s_client_mod

    from aiperf.kubernetes import console as kube_console

    metadata = cr.get("metadata") or {}
    uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if not uid or not resource_version:
        kube_console.print_error(
            f"Cannot safely delete {kind} {name}: the apiserver response did not "
            "include its UID and resourceVersion."
        )
        return False

    await custom.delete_namespaced_custom_object(
        group=AIPERF_JOB_GROUP,
        version=AIPERF_JOB_VERSION,
        plural=plural,
        namespace=namespace,
        name=name,
        body=k8s_client_mod.V1DeleteOptions(
            preconditions=k8s_client_mod.V1Preconditions(
                uid=uid,
                resource_version=resource_version,
            )
        ),
    )
    kube_console.print_success(f"Deleted {kind} {name}")
    kube_console.clear_last_benchmark_if_matches(name, namespace, kind=kind)
    return True


def confirm_action(message: str) -> bool:
    """Ask before a destructive action; auto-decline when not on a TTY.

    A non-interactive caller (CI, a pipe) cannot answer, and silently
    proceeding to delete would be the wrong default -- hence --force.
    """
    if not sys.stdin.isatty():
        from aiperf.kubernetes import console as kube_console

        kube_console.print_warning(
            "Not attached to a terminal; refusing to delete without --force."
        )
        return False
    answer = input(f"{message} [y/N] ").strip().lower()
    return answer in ("y", "yes")


async def find_aiperf_cr(
    custom: Any,
    *,
    namespace: str,
    name: str,
    kind: AIPerfWorkloadKind | None = None,
) -> tuple[str, dict] | None:
    """Return the single matching AIPerf workload, rejecting ambiguity."""
    from kubernetes_asyncio.client.exceptions import ApiException

    plurals = (_KIND_TO_PLURAL[kind],) if kind is not None else AIPERF_PLURALS
    found: list[tuple[str, dict]] = []
    for plural in plurals:
        try:
            cr = await custom.get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=plural,
                namespace=namespace,
                name=name,
            )
        except ApiException as e:
            if e.status == 404:
                continue
            raise
        found.append((plural, cr))

    if len(found) > 1:
        raise AmbiguousAIPerfTargetError(
            f"Both an AIPerfJob and AIPerfSweep named {name!r} exist in namespace "
            f"{namespace}. Re-run with --kind job or --kind sweep."
        )
    return found[0] if found else None


async def list_aiperf_crs(custom: Any, *, namespace: str) -> list[tuple[str, dict]]:
    """List every AIPerfJob and AIPerfSweep in a namespace as ``(plural, cr)``."""
    from kubernetes_asyncio.client.exceptions import ApiException

    found: list[tuple[str, dict]] = []
    for plural in AIPERF_PLURALS:
        try:
            listing = await custom.list_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=plural,
                namespace=namespace,
            )
        except ApiException as e:
            if e.status == 404:
                continue
            raise
        found.extend((plural, item) for item in listing.get("items", []))
    return found
