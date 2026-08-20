# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload-level pre-flight checks (secrets, image, configmap, dry-run)."""

from __future__ import annotations

import aiohttp
import orjson
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.kubernetes.preflight_utils import parse_image_ref
from aiperf.kubernetes.resources import CONFIGMAP_MAX_SIZE_BYTES
from aiperf.operator import preflight as _pf
from aiperf.operator.preflight._checker import _is_transient_error
from aiperf.operator.preflight._common import PUBLIC_REGISTRIES


def _collect_referenced_secrets(
    pod_template,
) -> dict[str, dict[str | None, bool]]:
    """Collect each secret's required object and key references."""
    needed: dict[str, dict[str | None, bool]] = {}

    def add(name: str, key: str | None = None, *, required: bool = True) -> None:
        references = needed.setdefault(name, {})
        references[key] = references.get(key, False) or required

    for ips in pod_template.image_pull_secrets:
        if name := ips.get("name"):
            add(name)
    for vol in pod_template.volumes:
        secret = vol.get("secret", {})
        if secret_name := secret.get("secretName"):
            add(secret_name)
    for env_var in pod_template.env:
        value_from = env_var.get("valueFrom", {})
        secret_ref = value_from.get("secretKeyRef", {})
        if secret_name := secret_ref.get("name"):
            add(
                secret_name,
                secret_ref.get("key"),
                required=not bool(secret_ref.get("optional", False)),
            )
    return needed


async def _probe_secrets(
    core, namespace: str, needed: dict[str, dict[str | None, bool]]
) -> tuple[list[str], list[str], list[str]]:
    """Read each secret; return (missing, permission_denied, missing_keys)."""
    missing: list[str] = []
    permission_denied: list[str] = []
    missing_keys: list[str] = []
    for secret_name, references in needed.items():
        required_keys = {
            key for key, required in references.items() if key is not None and required
        }
        required_secret = any(references.values())
        try:
            secret = await core.read_namespaced_secret(
                name=secret_name, namespace=namespace
            )
            data = getattr(secret, "data", None) or {}
            missing_keys.extend(
                f"{secret_name}/{key}" for key in sorted(required_keys - data.keys())
            )
        except ApiException as e:
            if e.status == 403:
                permission_denied.append(secret_name)
            else:
                # 404 or other error — treat as missing to fail preflight loudly.
                if required_secret:
                    missing.append(secret_name)
    return missing, permission_denied, missing_keys


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
        missing, permission_denied, missing_keys = await _probe_secrets(
            core, self.namespace, needed
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
        if missing_keys:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.FAIL,
                message=f"Secret key(s) not found: {', '.join(missing_keys)}",
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

        registry, _repo, tag, digest = parse_image_ref(image)

        warnings = []
        if not tag and not digest:
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
            # A 5xx from the apiserver is not a manifest rejection. Catching
            # every ApiException here preempted the dispatcher's own transient
            # classification (_is_transient_error -> WARN + retry), so a 503 or
            # 429 permanently failed the job with a message blaming OPA.
            if _is_transient_error(e):
                raise
            msg = str(e)
            if e.body:
                try:
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
        except (TimeoutError, aiohttp.ClientError, OSError) as e:
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.WARN,
                message=f"Dry run check failed: {e}",
            )
