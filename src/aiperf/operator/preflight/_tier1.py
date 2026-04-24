# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier 1 / Tier 2 blocking pre-flight checks (cluster compat + RBAC)."""

from __future__ import annotations

import asyncio
import re

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.jobset import get_jobset_install_hint
from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.kubernetes.preflight_utils import check_rbac_access
from aiperf.operator import preflight as _pf
from aiperf.operator.preflight._common import (
    MIN_K8S_MAJOR,
    MIN_K8S_MINOR,
    OPERATOR_RBAC_PERMISSIONS,
)


class _Tier1ChecksMixin:
    """Tier 1 (cluster compatibility) and Tier 2 (RBAC) blocking checks."""

    async def _check_kubernetes_version(self) -> CheckResult:
        """Verify Kubernetes version >= 1.24."""
        vinfo = await _pf.client.VersionApi(self.api).get_code()
        major_str = re.sub(r"[^0-9]", "", vinfo.major or "0")
        minor_str = re.sub(r"[^0-9]", "", vinfo.minor or "0")
        major = int(major_str) if major_str else 0
        minor = int(minor_str) if minor_str else 0
        git_version = vinfo.git_version or "unknown"

        if major > MIN_K8S_MAJOR or (major == MIN_K8S_MAJOR and minor >= MIN_K8S_MINOR):
            return CheckResult(
                name="Kubernetes Version",
                status=CheckStatus.PASS,
                message=f"Kubernetes {git_version} (>= {MIN_K8S_MAJOR}.{MIN_K8S_MINOR} required)",
            )
        return CheckResult(
            name="Kubernetes Version",
            status=CheckStatus.FAIL,
            message=(
                f"Kubernetes {git_version} is below minimum "
                f"{MIN_K8S_MAJOR}.{MIN_K8S_MINOR}. "
                f"Upgrade your cluster to {MIN_K8S_MAJOR}.{MIN_K8S_MINOR}+."
            ),
        )

    async def _check_jobset_crd(self) -> CheckResult:
        """Verify JobSet CRD is installed."""
        try:
            await _pf.client.CustomObjectsApi(self.api).list_cluster_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                limit=1,
            )
            return CheckResult(
                name="JobSet CRD",
                status=CheckStatus.PASS,
                message=f"JobSet CRD ({JOBSET_GROUP}/{JOBSET_VERSION}) installed",
            )
        except ApiException as e:
            if e.status == 404:
                return CheckResult(
                    name="JobSet CRD",
                    status=CheckStatus.FAIL,
                    message=f"JobSet CRD not found. Install with: {get_jobset_install_hint()}",
                )
            return CheckResult(
                name="JobSet CRD",
                status=CheckStatus.FAIL,
                message=f"Error checking JobSet CRD: HTTP {e.status or 'unknown'}",
            )

    async def _check_rbac_permissions(self) -> CheckResult:
        """Verify the operator has all required RBAC permissions."""
        missing = []
        for verb, resource, group in OPERATOR_RBAC_PERMISSIONS:
            try:
                allowed = await check_rbac_access(
                    self.api,
                    verb=verb,
                    resource=resource,
                    group=group,
                    namespace=self.namespace,
                )
                if not allowed:
                    display = f"{group}/{resource}" if group else resource
                    missing.append(f"{verb} {display}")
            except (
                ApiException,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
            ) as e:
                display = f"{group}/{resource}" if group else resource
                missing.append(f"{verb} {display} (check failed: {e})")
            except Exception as e:  # noqa: BLE001 - defensive: any per-permission probe error falls through to 'assume missing' rather than abort the tier
                display = f"{group}/{resource}" if group else resource
                missing.append(f"{verb} {display} (check failed: {e})")

        if missing:
            return CheckResult(
                name="RBAC Permissions",
                status=CheckStatus.FAIL,
                message=(
                    f"Missing {len(missing)} RBAC permission(s): "
                    f"{', '.join(missing)}. "
                    f"Grant permissions in namespace '{self.namespace}'."
                ),
            )
        return CheckResult(
            name="RBAC Permissions",
            status=CheckStatus.PASS,
            message=f"All {len(OPERATOR_RBAC_PERMISSIONS)} required permissions granted",
        )
