# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and run endpoint control hooks (reset_kv_cache, server_profiler)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.control_plane_http import ControlPlaneHttpError, control_plane_post
from aiperf.common.redact import redact_url
from aiperf.config.control_hooks import (
    DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS,
    DEFAULT_RESET_KV_CACHE_PATH,
    DEFAULT_SERVER_PROFILER_START_PATH,
    DEFAULT_SERVER_PROFILER_STOP_PATH,
)
from aiperf.config.endpoint import EndpointConfig

_logger = AIPerfLogger(__name__)


def endpoint_origin(url: str) -> str:
    """Return ``scheme://netloc`` for an endpoint URL (strip path/query)."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def join_origin_path(origin: str, path: str) -> str:
    """Join an origin and a leading-slash relative path into an absolute URL."""
    return origin.rstrip("/") + path


@dataclass(slots=True)
class PreparedEndpointControlHooks:
    """Resolved control-hook URLs and timeouts for one endpoint config."""

    timeout_s: float
    """Timeout seconds for reset_kv_cache POSTs."""
    reset_urls: list[str]
    """Absolute reset URLs, one per endpoint origin (empty when disabled)."""
    profiler_start_urls: list[str]
    """Absolute profiler-start URLs (empty when disabled)."""
    profiler_stop_urls: list[str]
    """Absolute profiler-stop URLs (empty when disabled)."""
    profiler_timeout_s: float
    """Timeout seconds for profiler start/stop POSTs."""


def unique_endpoint_origins(urls: list[str]) -> list[str]:
    """Return order-preserving unique ``scheme://netloc`` origins from URLs."""
    seen: set[str] = set()
    origins: list[str] = []
    for url in urls:
        origin = endpoint_origin(url)
        if origin in seen:
            continue
        seen.add(origin)
        origins.append(origin)
    return origins


def prepare_endpoint_control_hooks(
    endpoint: EndpointConfig,
) -> PreparedEndpointControlHooks:
    """Resolve relative control-hook paths against each unique endpoint origin."""
    origins = unique_endpoint_origins(endpoint.urls)
    reset_path = (
        endpoint.reset_kv_cache.path
        if endpoint.reset_kv_cache and endpoint.reset_kv_cache.path
        else DEFAULT_RESET_KV_CACHE_PATH
    )
    start_path = (
        endpoint.server_profiler.start_path
        if endpoint.server_profiler and endpoint.server_profiler.start_path
        else DEFAULT_SERVER_PROFILER_START_PATH
    )
    stop_path = (
        endpoint.server_profiler.stop_path
        if endpoint.server_profiler and endpoint.server_profiler.stop_path
        else DEFAULT_SERVER_PROFILER_STOP_PATH
    )
    reset_timeout = (
        endpoint.reset_kv_cache.timeout_seconds
        if endpoint.reset_kv_cache and endpoint.reset_kv_cache.timeout_seconds
        else DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS
    )
    profiler_timeout = (
        endpoint.server_profiler.timeout_seconds
        if endpoint.server_profiler and endpoint.server_profiler.timeout_seconds
        else DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS
    )
    return PreparedEndpointControlHooks(
        timeout_s=float(reset_timeout),
        reset_urls=(
            [join_origin_path(o, reset_path) for o in origins]
            if endpoint.reset_kv_cache is not None
            else []
        ),
        profiler_start_urls=(
            [join_origin_path(o, start_path) for o in origins]
            if endpoint.server_profiler is not None
            else []
        ),
        profiler_stop_urls=(
            [join_origin_path(o, stop_path) for o in origins]
            if endpoint.server_profiler is not None
            else []
        ),
        profiler_timeout_s=float(profiler_timeout),
    )


async def run_reset_kv_cache(
    hooks: PreparedEndpointControlHooks,
    headers: dict[str, str],
) -> None:
    """POST reset_kv_cache to every prepared reset URL (fatal on first failure)."""
    for url in hooks.reset_urls:
        await control_plane_post(url=url, headers=headers, timeout_s=hooks.timeout_s)


async def start_server_profiler(
    hooks: PreparedEndpointControlHooks,
    headers: dict[str, str],
) -> None:
    """POST profiler start to every origin; reverse-stop on partial failure."""
    started: list[str] = []
    try:
        for url in hooks.profiler_start_urls:
            await control_plane_post(
                url=url, headers=headers, timeout_s=hooks.profiler_timeout_s
            )
            started.append(url)
    except BaseException:
        # BaseException so task cancellation still triggers cleanup; the
        # original exception is re-raised unchanged after the stops run.
        # Best-effort reverse-order stop cleanup on partial start failure.
        cleanup_errors: list[str] = []
        for stop_url in reversed(hooks.profiler_stop_urls[: len(started)]):
            try:
                await control_plane_post(
                    url=stop_url,
                    headers=headers,
                    timeout_s=hooks.profiler_timeout_s,
                )
            except Exception as cleanup_exc:
                msg = (
                    f"profiler cleanup stop {redact_url(stop_url)} failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                cleanup_errors.append(msg)
                _logger.warning(msg)
        if cleanup_errors:
            _logger.warning(
                f"server_profiler partial-start cleanup had "
                f"{len(cleanup_errors)} failure(s)"
            )
        raise


async def stop_server_profiler(
    hooks: PreparedEndpointControlHooks,
    headers: dict[str, str],
) -> None:
    """Best-effort stop on every origin; raise one aggregated error if any fail.

    Cancellation does not short-circuit the loop: remaining origins are still
    attempted, then the ``CancelledError`` is re-raised so the caller's
    cancellation stays observable.
    """
    failures: list[str] = []
    cancelled: asyncio.CancelledError | None = None
    for url in hooks.profiler_stop_urls:
        try:
            await control_plane_post(
                url=url, headers=headers, timeout_s=hooks.profiler_timeout_s
            )
        except ControlPlaneHttpError as error:
            failures.append(str(error))
        except asyncio.CancelledError as error:
            cancelled = error
    if cancelled is not None:
        if failures:
            _logger.warning(
                f"server_profiler stop failed for {len(failures)} origin(s) "
                f"during cancellation: " + "; ".join(failures)
            )
        raise cancelled
    if failures:
        raise ControlPlaneHttpError(
            "server_profiler stop failed for "
            f"{len(failures)} origin(s): " + "; ".join(failures)
        )
