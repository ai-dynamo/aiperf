# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Kubernetes ResourceQuota admission evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
)


def _resource_quota_violation(
    hard: Mapping[str, str],
    used: Mapping[str, str],
    *,
    required: float,
    request_keys: tuple[str, str],
    limit_key: str,
    parse_quantity: Callable[[str], float],
    format_quantity: Callable[[float], str],
    resource: str,
) -> str | None:
    """Return a request or limit violation for one resource family."""
    hard_request = hard.get(request_keys[0]) or hard.get(request_keys[1])
    if hard_request:
        try:
            needed = required + parse_quantity(
                used.get(request_keys[0]) or used.get(request_keys[1]) or "0"
            )
            exceeded = needed > parse_quantity(hard_request)
        except (TypeError, ValueError):
            # A malformed quota value (or a non-quantity stub in tests) cannot
            # be evaluated -- treat it as absent rather than surfacing a raw,
            # unparseable value to the user or crashing the whole check.
            exceeded = False
        if exceeded:
            return (
                f"{resource} requests quota: {format_quantity(needed)} needed vs "
                f"{hard_request} limit"
            )

    hard_limit = hard.get(limit_key)
    if hard_limit:
        try:
            needed = required + parse_quantity(used.get(limit_key) or "0")
            exceeded = needed > parse_quantity(hard_limit)
        except (TypeError, ValueError):
            exceeded = False
        if exceeded:
            return (
                f"{resource} limits quota: {format_quantity(needed)} needed vs "
                f"{hard_limit} limit"
            )
    return None


def quota_violation(
    hard: Mapping[str, str],
    used: Mapping[str, str],
    *,
    required_cpu: float,
    required_mem: float,
    required_pods: int,
) -> str | None:
    """Return the first ResourceQuota limit exceeded by the planned workload."""
    if violation := _resource_quota_violation(
        hard,
        used,
        required=required_cpu,
        request_keys=("cpu", "requests.cpu"),
        limit_key="limits.cpu",
        parse_quantity=parse_cpu,
        format_quantity=format_cpu,
        resource="CPU",
    ):
        return violation

    if violation := _resource_quota_violation(
        hard,
        used,
        required=required_mem,
        request_keys=("memory", "requests.memory"),
        limit_key="limits.memory",
        parse_quantity=parse_memory_gib,
        format_quantity=format_memory,
        resource="memory",
    ):
        return violation

    hard_pods = hard.get("pods")
    if hard_pods:
        try:
            used_pods = int(used.get("pods") or 0)
            if used_pods + required_pods > int(hard_pods):
                return (
                    f"pods quota: {used_pods + required_pods} total vs "
                    f"{hard_pods} limit"
                )
        except (TypeError, ValueError):
            pass
    return None
