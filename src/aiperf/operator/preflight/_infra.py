# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infrastructure-level pre-flight checks (controllers, DNS, policies, Kueue, PSA)."""

from __future__ import annotations

import asyncio

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import (
    KUEUE_GROUP,
    KUEUE_LOCALQUEUE_PLURAL,
    KUEUE_VERSION,
)
from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.operator import preflight as _pf


class _InfraChecksMixin:
    """Checks for in-cluster controllers, DNS, network policies, Kueue, and PSA."""

    async def _check_jobset_controller(self) -> CheckResult:
        """Check if JobSet controller is running in jobset-system."""
        try:
            deploy_list = await _pf.client.AppsV1Api(
                self.api
            ).list_namespaced_deployment(
                namespace="jobset-system",
            )
            for deploy in deploy_list.items:
                name = (deploy.metadata.name if deploy.metadata else "") or ""
                if "jobset" in name.lower():
                    ready = (
                        deploy.status.ready_replicas
                        if deploy.status and deploy.status.ready_replicas
                        else 0
                    )
                    if ready and ready > 0:
                        return CheckResult(
                            name="JobSet Controller",
                            status=CheckStatus.PASS,
                            message="JobSet controller is running",
                        )
                    return CheckResult(
                        name="JobSet Controller",
                        status=CheckStatus.WARN,
                        message="JobSet controller found but not ready",
                        hints=["Check: kubectl get pods -n jobset-system"],
                    )
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.WARN,
                message="JobSet controller not found in jobset-system namespace",
            )
        except ApiException as e:
            if e.status == 403:
                return CheckResult(
                    name="JobSet Controller",
                    status=CheckStatus.SKIP,
                    message="Cannot check jobset-system namespace (permission denied)",
                )
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.WARN,
                message=f"Could not verify JobSet controller: {e}",
            )

    async def _check_service_account(self) -> CheckResult:
        """Verify custom service account exists if specified."""
        sa_name = self.deploy_config.pod_template.service_account_name
        if not sa_name:
            return CheckResult(
                name="Service Account",
                status=CheckStatus.SKIP,
                message="No custom service account specified",
            )
        try:
            await _pf.client.CoreV1Api(self.api).read_namespaced_service_account(
                name=sa_name, namespace=self.namespace
            )
            return CheckResult(
                name="Service Account",
                status=CheckStatus.PASS,
                message=f"Service account '{sa_name}' exists",
            )
        except ApiException as e:
            if e.status == 404:
                return CheckResult(
                    name="Service Account",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Service account '{sa_name}' not found in namespace "
                        f"'{self.namespace}'. Pod creation will fail."
                    ),
                    hints=[
                        f"kubectl create serviceaccount {sa_name} -n {self.namespace}"
                    ],
                )
            return CheckResult(
                name="Service Account",
                status=CheckStatus.WARN,
                message=f"Could not verify service account: {e}",
            )

    async def _check_dns(self) -> CheckResult:
        """Verify CoreDNS is running in kube-system."""
        try:
            deploy_list = await _pf.client.AppsV1Api(
                self.api
            ).list_namespaced_deployment(
                namespace="kube-system",
            )
            for deploy in deploy_list.items:
                name = (deploy.metadata.name if deploy.metadata else "") or ""
                if "coredns" in name.lower():
                    ready = (
                        deploy.status.ready_replicas
                        if deploy.status and deploy.status.ready_replicas
                        else 0
                    )
                    if ready and ready > 0:
                        return CheckResult(
                            name="DNS Resolution",
                            status=CheckStatus.PASS,
                            message="CoreDNS is running",
                        )
                    return CheckResult(
                        name="DNS Resolution",
                        status=CheckStatus.WARN,
                        message="CoreDNS found but not ready",
                        hints=[
                            "Check: kubectl get pods -n kube-system -l k8s-app=kube-dns"
                        ],
                    )
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message="CoreDNS not found in kube-system",
            )
        except ApiException as e:
            if e.status == 403:
                return CheckResult(
                    name="DNS Resolution",
                    status=CheckStatus.SKIP,
                    message="Cannot check kube-system namespace (permission denied)",
                )
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message=f"Could not verify DNS: {e}",
            )

    async def _check_network_policies(self) -> CheckResult:
        """Warn if restrictive network policies exist in namespace."""
        try:
            policy_list = await _pf.client.NetworkingV1Api(
                self.api
            ).list_namespaced_network_policy(namespace=self.namespace)
            policies = policy_list.items
            if not policies:
                return CheckResult(
                    name="Network Policies",
                    status=CheckStatus.PASS,
                    message="No network policies found (unrestricted)",
                )
            policy_names = [
                (p.metadata.name if p.metadata else "") or "" for p in policies
            ]
            return CheckResult(
                name="Network Policies",
                status=CheckStatus.WARN,
                message=(
                    f"Found {len(policies)} network policy(ies): {', '.join(policy_names)}. "
                    f"Ensure pod-to-pod communication is allowed."
                ),
            )
        except ApiException as e:
            if e.status == 403:
                return CheckResult(
                    name="Network Policies",
                    status=CheckStatus.SKIP,
                    message="Cannot check network policies (permission denied)",
                )
            return CheckResult(
                name="Network Policies",
                status=CheckStatus.WARN,
                message=f"Could not check network policies: {e}",
            )

    async def _check_kueue_queue(self) -> CheckResult:
        """Verify Kueue queue configuration.

        If scheduling.queueName is set, verify the LocalQueue exists.
        If not set, check whether Kueue is installed and warn that the job
        will bypass gang-scheduling unless a namespace default queue is configured.
        """

        queue_name = self.deploy_config.scheduling.queue_name

        if queue_name:
            return await self._verify_kueue_local_queue(queue_name)

        # No explicit queue — check if Kueue is installed
        kueue_installed = await self._is_kueue_installed()
        if not kueue_installed:
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.SKIP,
                message="Kueue not installed (queue check skipped)",
            )

        # Kueue is installed — check namespace default queue annotation
        has_default = await self._namespace_has_default_queue()
        if has_default:
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.PASS,
                message=(
                    "No explicit queue specified but namespace has "
                    "kueue.x-k8s.io/default-queue-name annotation"
                ),
            )

        return CheckResult(
            name="Kueue Queue",
            status=CheckStatus.WARN,
            message=(
                "Kueue is installed but no queue configured. "
                "Job will bypass gang-scheduling and quota management."
            ),
            hints=[
                "Set scheduling.queueName in the CR spec, or",
                f"Annotate the namespace: kubectl annotate namespace {self.namespace} "
                "kueue.x-k8s.io/default-queue-name=<queue-name>",
            ],
        )

    async def _verify_kueue_local_queue(self, queue_name: str) -> CheckResult:
        """Verify a specific Kueue LocalQueue exists."""
        try:
            await _pf.client.CustomObjectsApi(self.api).get_namespaced_custom_object(
                group=KUEUE_GROUP,
                version=KUEUE_VERSION,
                plural=KUEUE_LOCALQUEUE_PLURAL,
                namespace=self.namespace,
                name=queue_name,
            )
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.PASS,
                message=f"Kueue LocalQueue '{queue_name}' exists",
            )
        except ApiException as e:
            status_code = e.status or 0
            if status_code == 404:
                # Distinguish "CRD not installed" vs "queue not found".
                # If Kueue isn't installed at all, we get 404 on the queue
                # but also 404 on the plural list — treat a missing CRD as SKIP,
                # a missing queue as FAIL.
                kueue_installed = await self._is_kueue_installed()
                if not kueue_installed:
                    return CheckResult(
                        name="Kueue Queue",
                        status=CheckStatus.SKIP,
                        message="Kueue CRD not installed (queue check skipped)",
                    )
                return CheckResult(
                    name="Kueue Queue",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Kueue LocalQueue '{queue_name}' not found. "
                        f"Create it or remove scheduling.queueName from spec."
                    ),
                )
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.WARN,
                message=f"Could not verify Kueue queue: HTTP {status_code or 'unknown'}",
            )

    async def _is_kueue_installed(self) -> bool:
        """Check if the Kueue CRD is available on the cluster."""
        try:
            await _pf.client.CustomObjectsApi(self.api).list_namespaced_custom_object(
                group=KUEUE_GROUP,
                version=KUEUE_VERSION,
                plural=KUEUE_LOCALQUEUE_PLURAL,
                namespace=self.namespace,
                limit=1,
            )
            return True
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _namespace_has_default_queue(self) -> bool:
        """Check if the namespace has a Kueue default queue annotation."""
        try:
            ns = await _pf.client.CoreV1Api(self.api).read_namespace(
                name=self.namespace
            )
            annotations = (ns.metadata.annotations or {}) if ns.metadata else {}
            return bool(annotations.get("kueue.x-k8s.io/default-queue-name"))
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _check_pod_security_admission(self) -> CheckResult:
        """Check namespace PSA labels for compatibility."""
        try:
            ns = await _pf.client.CoreV1Api(self.api).read_namespace(
                name=self.namespace
            )
            labels = (ns.metadata.labels or {}) if ns.metadata else {}

            psa_enforce = labels.get("pod-security.kubernetes.io/enforce")
            if not psa_enforce:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.PASS,
                    message="No PSA enforcement label on namespace",
                )

            # Our pods run as non-root (UID 1000) with seccomp=RuntimeDefault
            # and drop all capabilities — compatible with "restricted"
            compatible_levels = {"privileged", "baseline", "restricted"}
            if psa_enforce in compatible_levels:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.PASS,
                    message=f"PSA enforce level '{psa_enforce}' is compatible",
                )
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Unknown PSA enforce level '{psa_enforce}'",
            )
        except ApiException as e:
            if e.status == 404:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.WARN,
                    message=f"Namespace '{self.namespace}' not found",
                )
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Could not check PSA: {e}",
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Could not check PSA: {e}",
            )
