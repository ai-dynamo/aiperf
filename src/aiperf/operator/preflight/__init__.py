# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator pre-flight checks for AIPerfJob deployments.

Validates cluster readiness before creating any resources. Checks are organized
into tiers:
- Tier 1 (blocking): Kubernetes version, JobSet CRD
- Tier 2 (blocking): RBAC permissions
- Tier 3+ (concurrent): Everything else

On failure, the operator sets the CR to Failed with actionable error messages
and does not create any resources.

This package is split into mixin modules for LLM-ergonomics (file-size limit).
Submodules import ``kubernetes_asyncio.client`` directly; ``client`` is
re-exported here purely as a stable patch target for tests
(``aiperf.operator.preflight.client.*`` resolves to the same module object).
"""

from __future__ import annotations

# Re-exported as the canonical patch target: this is the same module object
# the submodules import, so patching it here reaches every check.
from kubernetes_asyncio import client  # noqa: F401

from aiperf.operator.preflight._checker import (
    OperatorPreflightChecker,
    _is_node_ready_typed,
)

__all__ = ["OperatorPreflightChecker", "_is_node_ready_typed", "client"]
