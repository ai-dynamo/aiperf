# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint health checking for the operator."""

from __future__ import annotations

import asyncio
import socket

import aiohttp

from aiperf.kubernetes.crd_models import EndpointHealthResult
from aiperf.operator.environment import OperatorEnvironment

_HEALTH_PATHS = ["/health", "/v1/health", "/v1/models", "/"]


async def _probe_once(url: str, timeout: float) -> EndpointHealthResult:
    """Run one health-check attempt across all candidate paths.

    Returns a result with ``reachable=True`` on first 2xx/3xx/4xx response,
    a DNS-failure result when a socket.gaierror is encountered (caller should
    retry), or an unreachable result when all paths fail for other reasons.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), connector=connector
    ) as session:
        for path in _HEALTH_PATHS:
            check_url = url.rstrip("/") + path
            try:
                async with session.get(check_url) as response:
                    if response.status < 500:
                        return EndpointHealthResult(reachable=True, error="")
            except aiohttp.ClientConnectorError as e:
                if isinstance(e.os_error, socket.gaierror):
                    return EndpointHealthResult(
                        reachable=False,
                        error=f"DNS resolution failed for {check_url}: {e.os_error}",
                    )
            except aiohttp.ClientError:
                continue
            except (TimeoutError, OSError) as e:
                return EndpointHealthResult(
                    reachable=False, error=f"Unexpected error: {e}"
                )
            except Exception as e:  # noqa: BLE001 - defensive: any unexpected error must surface as a reachable=False result, never as a raise into the kopf on_create handler
                return EndpointHealthResult(
                    reachable=False, error=f"Unexpected error: {e}"
                )

    return EndpointHealthResult(
        reachable=False, error="All health endpoints unreachable"
    )


async def check_endpoint_health(
    url: str, timeout: float = OperatorEnvironment.ENDPOINT_CHECK_TIMEOUT
) -> EndpointHealthResult:
    """Check if LLM endpoint is reachable.

    Tries a single canonical health path first, falling back to alternatives
    only if the first fails.  On transient DNS resolution failures the check
    is retried up to ``ENDPOINT_CHECK_DNS_RETRIES`` times with a short delay
    so that a brief CoreDNS blip does not permanently fail the benchmark.

    Args:
        url: Endpoint URL to check.
        timeout: Per-request timeout in seconds.

    Returns:
        EndpointHealthResult with reachability status and error message.
    """
    max_dns_retries: int = OperatorEnvironment.ENDPOINT_CHECK_DNS_RETRIES
    dns_retry_delay: float = OperatorEnvironment.ENDPOINT_CHECK_DNS_RETRY_DELAY

    result = EndpointHealthResult(
        reachable=False, error="All health endpoints unreachable"
    )
    for attempt in range(max_dns_retries):
        result = await _probe_once(url, timeout)
        if result.reachable or not result.error.startswith("DNS resolution failed"):
            break
        if attempt < max_dns_retries - 1:
            await asyncio.sleep(dns_retry_delay)

    return result
