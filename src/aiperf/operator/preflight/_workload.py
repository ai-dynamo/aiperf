# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload-level pre-flight checks (secrets, image, configmap, dry-run)."""

from __future__ import annotations

import asyncio

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.kubernetes.preflight_utils import parse_image_ref
from aiperf.kubernetes.resources import CONFIGMAP_MAX_SIZE_BYTES
from aiperf.operator import preflight as _pf
from aiperf.operator.preflight._common import PUBLIC_REGISTRIES


def _collect_referenced_secrets(pod_template) -> set[str]:
    """Collect all secret names referenced by pull secrets, volumes, and env."""
    needed: set[str] = set()
    needed.update(pod_template.image_pull_secrets)
    for vol in pod_template.volumes:
        secret = vol.get("secret", {})
        if secret_name := secret.get("secretName"):
            needed.add(secret_name)
    for env_var in pod_template.env:
        value_from = env_var.get("valueFrom", {})
        secret_ref = value_from.get("secretKeyRef", {})
        if secret_name := secret_ref.get("name"):
            needed.add(secret_name)
    return needed


async def _probe_secrets(
    core, namespace: str, names: list[str]
) -> tuple[list[str], list[str]]:
    """Read each secret; return (missing, permission_denied)."""
    missing: list[str] = []
    permission_denied: list[str] = []
    for secret_name in names:
        try:
            await core.read_namespaced_secret(name=secret_name, namespace=namespace)
        except ApiException as e:
            if e.status == 403:
                permission_denied.append(secret_name)
            else:
                # 404 or other error — treat as missing to fail preflight loudly.
                missing.append(secret_name)
    return missing, permission_denied


class _WorkloadChecksMixin:
    """Checks bound to the specific workload: secrets, image, ConfigMap, dry-run."""

    async def _check_secrets(self) -> CheckResult:
        """Verify all referenced secrets exist."""
        needed = _collect_referenced_secrets(self.deploy_config.pod_template)
        if not needed:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.SKIP,
                message="No secrets referenced",
            )

        core = _pf.client.CoreV1Api(self.api)
        missing, permission_denied = await _probe_secrets(
            core, self.namespace, sorted(needed)
        )

        if missing:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.FAIL,
                message=(
                    f"Secret(s) not found: {', '.join(missing)}. "
                    f"Create with: kubectl create secret -n {self.namespace}"
                ),
            )
        if permission_denied:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.WARN,
                message=f"Cannot verify secret(s): {', '.join(permission_denied)} (permission denied)",
            )
        return CheckResult(
            name="Secrets",
            status=CheckStatus.PASS,
            message=f"All {len(needed)} secret(s) verified",
        )

    async def _check_image_reference(self) -> CheckResult:
        """Validate image format and warn on implicit latest or missing pull secrets."""
        image = self.deploy_config.image
        if not image:
            return CheckResult(
                name="Image Reference",
                status=CheckStatus.FAIL,
                message="No container image specified",
            )

        registry, _repo, tag = parse_image_ref(image)

        warnings = []
        if not tag:
            warnings.append(
                "Image uses implicit 'latest' tag which may cause inconsistent deployments"
            )

        has_pull_secrets = bool(self.deploy_config.pod_template.image_pull_secrets)
        if registry not in PUBLIC_REGISTRIES and not has_pull_secrets:
            warnings.append(
                f"Registry '{registry}' may require authentication "
                f"but no imagePullSecrets configured"
            )

        if warnings:
            return CheckResult(
                name="Image Reference",
                status=CheckStatus.WARN,
                message=f"Image '{image}': {'; '.join(warnings)}",
            )
        return CheckResult(
            name="Image Reference",
            status=CheckStatus.PASS,
            message=f"Image '{image}' reference is valid",
        )

    async def _check_configmap_size(self) -> CheckResult:
        """Verify generated ConfigMap data fits within 1 MiB limit."""
        try:
            cm_spec = self.deployment.get_configmap_spec()
            size_bytes = cm_spec.get_data_size_bytes()
            max_bytes = CONFIGMAP_MAX_SIZE_BYTES
            if size_bytes > max_bytes:
                size_mib = size_bytes / (1024 * 1024)
                return CheckResult(
                    name="ConfigMap Size",
                    status=CheckStatus.FAIL,
                    message=(
                        f"ConfigMap data size ({size_mib:.2f} MiB) exceeds "
                        f"1 MiB limit. Reduce config size."
                    ),
                )
            return CheckResult(
                name="ConfigMap Size",
                status=CheckStatus.PASS,
                message=f"ConfigMap size OK ({size_bytes:,} bytes)",
            )
        except (ValueError, TypeError, OSError) as e:
            return CheckResult(
                name="ConfigMap Size",
                status=CheckStatus.FAIL,
                message=f"Could not compute ConfigMap size: {e}",
            )

    async def _check_dry_run(self) -> CheckResult:
        """POST JobSet manifest with dryRun=All to catch API server rejections."""
        try:
            jobset_manifest = self.deployment.get_jobset_spec().to_k8s_manifest()
            await _pf.client.CustomObjectsApi(self.api).create_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=self.namespace,
                body=jobset_manifest,
                dry_run="All",
            )
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.PASS,
                message="Server dry-run accepted the JobSet manifest",
            )
        except ApiException as e:
            msg = str(e)
            if e.body:
                try:
                    import orjson

                    body = orjson.loads(e.body)
                    msg = body.get("message", msg)
                except (ValueError, TypeError, orjson.JSONDecodeError):
                    # e.body was not well-formed JSON; fall back to str(e).
                    pass
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.FAIL,
                message=(
                    f"Server dry-run rejected JobSet: {msg}. "
                    f"Fix: check OPA/Gatekeeper policies or admission webhooks."
                ),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.WARN,
                message=f"Dry run check failed: {e}",
            )
