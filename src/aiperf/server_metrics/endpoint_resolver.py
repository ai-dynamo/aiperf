# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint resolution and auto-discovery for ServerMetricsManager.

Owns the default/explicit scrape target lists and the Kubernetes auto-discovery
plumbing so the manager can focus on collector lifecycle and result publication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from aiperf.common.enums import ServerMetricsDiscoveryMode
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.server_metrics.discovery.kubernetes import (
    discover_kubernetes_endpoints,
    is_running_in_kubernetes,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
    from aiperf.config.artifacts import ServerMetricsDiscoveryConfig


class _Logger(Protocol):
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def debug(self, msg: object) -> None: ...


def _dedup_normalized(urls: list[str]) -> list[str]:
    """Normalize metrics URLs and deduplicate preserving order."""
    result: list[str] = []
    for url in urls:
        normalized = normalize_metrics_endpoint_url(url)
        if normalized not in result:
            result.append(normalized)
    return result


class EndpointResolver:
    """Builds the scrape target list and runs auto-discovery.

    Keeps default (inference-derived) and explicit (user-provided) endpoint
    lists separate so Kubernetes discovery can replace load-balanced defaults
    without discarding explicit endpoints.
    """

    def __init__(self, run: BenchmarkRun) -> None:
        server_metrics_config = run.cfg.server_metrics
        self.discovery: ServerMetricsDiscoveryConfig | None = (
            server_metrics_config.discovery if server_metrics_config else None
        )
        self.default_endpoints: list[str] = _dedup_normalized(
            list(run.cfg.endpoint.urls)
        )
        self.explicit_endpoints: list[str] = (
            _dedup_normalized(list(server_metrics_config.urls))
            if server_metrics_config and server_metrics_config.urls
            else []
        )

    def build_endpoints(self, *, include_default_endpoints: bool) -> list[str]:
        """Build the current scrape target list with stable deduplication."""
        endpoints: list[str] = []
        if include_default_endpoints:
            for url in self.default_endpoints:
                if url not in endpoints:
                    endpoints.append(url)
        for url in self.explicit_endpoints:
            if url not in endpoints:
                endpoints.append(url)
        return endpoints

    def should_include_default_endpoints(self) -> bool:
        """Whether inference-derived metrics URLs should remain scrape targets.

        Kubernetes discovery should replace the default inference endpoint scrape
        target because that endpoint often resolves to a load balancer rather than
        a single pod, which can corrupt cumulative metrics.
        """
        if self.discovery is None:
            return True
        mode = self.discovery.mode
        if mode == ServerMetricsDiscoveryMode.DISABLED:
            return True
        if mode == ServerMetricsDiscoveryMode.KUBERNETES:
            return False
        if mode == ServerMetricsDiscoveryMode.AUTO:
            return not is_running_in_kubernetes()
        return True

    async def run_discovery(self, logger: _Logger) -> list[str]:
        """Run metrics endpoint auto-discovery based on configuration."""
        if self.discovery is None:
            return []
        mode = self.discovery.mode
        if mode == ServerMetricsDiscoveryMode.DISABLED:
            return []
        if mode == ServerMetricsDiscoveryMode.KUBERNETES:
            return await self._run_kubernetes_discovery(logger)
        if is_running_in_kubernetes():
            return await self._run_auto_discovery(logger)
        logger.debug(lambda: "Server Metrics: Not in K8s, skipping auto-discovery")
        return []

    async def _run_kubernetes_discovery(self, logger: _Logger) -> list[str]:
        if not is_running_in_kubernetes():
            logger.warning(
                "Server Metrics: Kubernetes discovery requested but not running in K8s cluster"
            )
            return []
        logger.info("Server Metrics: Running Kubernetes discovery...")
        try:
            return await discover_kubernetes_endpoints(
                namespace=self.discovery.namespace,  # type: ignore[union-attr]
                label_selector=self.discovery.label_selector,  # type: ignore[union-attr]
            )
        except Exception as e:  # noqa: BLE001 - discovery is best-effort; fall through
            logger.warning(f"Server Metrics: Kubernetes discovery failed: {e}")
            return []

    async def _run_auto_discovery(self, logger: _Logger) -> list[str]:
        # Derive namespace from endpoint URLs if not explicitly set.
        # Service DNS: "svc-name.namespace.svc.cluster.local" -> namespace
        ns = self.discovery.namespace  # type: ignore[union-attr]
        if ns is None:
            ns = self.extract_namespace_from_endpoints()
        logger.info(
            f"Server Metrics: Running Kubernetes auto-discovery"
            f"{f' (namespace={ns})' if ns else ''}..."
        )
        try:
            return await discover_kubernetes_endpoints(
                namespace=ns,
                label_selector=self.discovery.label_selector,  # type: ignore[union-attr]
            )
        except Exception as e:  # noqa: BLE001 - discovery is best-effort; fall through
            logger.warning(f"Server Metrics: Kubernetes auto-discovery failed: {e}")
            return []

    def extract_namespace_from_endpoints(self) -> str | None:
        """Extract K8s namespace from endpoint service DNS names.

        Parses ``svc-name.namespace.svc.cluster.local`` to extract namespace.
        Returns the first namespace found, or None.
        """
        for url in self.explicit_endpoints + self.default_endpoints:
            try:
                host = urlparse(url).hostname or ""
                parts = host.split(".")
                if len(parts) >= 3 and parts[2] == "svc":
                    return parts[1]
            except (ValueError, AttributeError):
                continue
        return None
