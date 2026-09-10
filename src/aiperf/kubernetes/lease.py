# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading of the operator's namespace-ownership Lease.

Kept separate from :mod:`aiperf.operator.namespace_claim`, which owns the
write side, so that ``aiperf kube`` can report namespace ownership without
importing the operator package.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kubernetes_asyncio.client import V1Lease, V1LeaseSpec

__all__ = ["lease_holder_if_live"]


def lease_holder_if_live(lease: V1Lease, *, default_duration: int) -> str | None:
    """Return the lease's holder identity, or ``None`` if it has expired.

    A lease is expired when ``renewTime + leaseDurationSeconds`` is in the past.
    A lease with no renew/acquire timestamp is treated as live: an unreadable
    timestamp must not let a second operator steal a namespace.
    """
    spec: V1LeaseSpec | None = lease.spec
    if spec is None or not spec.holder_identity:
        return None
    stamp = spec.renew_time or spec.acquire_time
    if stamp is None:
        return spec.holder_identity
    if stamp.tzinfo is None:
        # A naive timestamp cannot be compared against an aware "now". The
        # apiserver always serializes RFC3339 UTC, so assuming UTC is correct
        # and keeps a hot kopf ``when=`` filter from raising TypeError.
        stamp = stamp.replace(tzinfo=UTC)
    duration = spec.lease_duration_seconds or default_duration
    if stamp + timedelta(seconds=duration) < datetime.now(UTC):
        return None
    return spec.holder_identity
