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

This package is split into mixin modules for LLM-ergonomics (file-size limit);
``client`` is re-exported here so tests can patch
``aiperf.operator.preflight.client.*`` regardless of which submodule invokes it.
"""

from __future__ import annotations

# Re-exported so submodules and tests can resolve ``client.*`` via
# ``aiperf.operator.preflight.client`` — keeps patches centralized.
from kubernetes_asyncio import client  # noqa: F401

from aiperf.operator.preflight._checker import (
    OperatorPreflightChecker,
    _is_node_ready_typed,
)

__all__ = ["OperatorPreflightChecker", "_is_node_ready_typed", "client"]
