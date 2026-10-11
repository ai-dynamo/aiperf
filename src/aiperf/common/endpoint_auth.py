# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auth-header construction for readiness probes and control-plane hooks."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from aiperf.plugin.enums import EndpointType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aiperf.auth.base_signer import RequestSignerProtocol
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.endpoint import EndpointConfig

# Anthropic Messages authenticates with x-api-key plus a required
# anthropic-version header. Bearer auth or a missing version returns a 4xx,
# which the readiness probe's "status < 500 == ready" rule would misread as
# ready. Mirrors MessagesEndpoint.get_endpoint_headers().
_ANTHROPIC_VERSION = "2023-06-01"


def auth_headers_for_endpoint(cfg: EndpointConfig) -> dict[str, str]:
    """Build auth headers matching the endpoint's scheme (Bearer or Anthropic)."""
    # Custom endpoint.headers pass through intentionally: they can carry
    # proprietary auth, gateway routing, or tracing metadata that control-plane
    # and readiness endpoints need just as much as inference does. Documented in
    # docs/tutorials/benchmark-control-hooks.md.
    headers = dict(cfg.headers or {})
    # A configured auth_type owns the credential: the request path suppresses
    # api_key entirely (see BaseEndpoint/MessagesEndpoint.get_endpoint_headers)
    # so the signer's Authorization header is not overwritten. The control
    # plane and readiness probes must suppress it identically, otherwise
    # preflight authenticates differently than the benchmark it gates.
    key_auth = bool(cfg.api_key) and not cfg.auth_type
    if cfg.type == EndpointType.MESSAGES:
        headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        if key_auth:
            # Hard-assign so --api-key overrides any preconfigured x-api-key,
            # matching MessagesEndpoint.get_endpoint_headers(); otherwise
            # preflight would probe a different key than real requests use.
            headers["x-api-key"] = cfg.api_key
    elif key_auth:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers


async def sign_request(
    signer: RequestSignerProtocol | None,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> tuple[str, dict[str, str], bytes | None]:
    """Apply ``signer`` to one request, resolving the optional overrides.

    Returns the caller's own ``(url, headers, body)`` unchanged when no
    signer is configured. Sign per attempt, never once per URL: a SigV4
    signature embeds ``x-amz-date`` and AWS rejects it outside a five-minute
    skew window, which readiness polling and control-hook retry backoff both
    routinely exceed.
    """
    if signer is None:
        return url, headers, body
    signed = await signer.sign(method, url, headers, body)
    return (
        signed.url if signed.url is not None else url,
        signed.headers,
        signed.body if signed.body is not None else body,
    )


def no_redirect_kwargs(signer: RequestSignerProtocol | None) -> dict[str, bool]:
    """Request kwargs that stop a *signed* request from following redirects.

    aiohttp drops ``Authorization`` when a redirect crosses origins, but has no
    equivalent rule for custom headers, so ``x-amz-security-token`` -- a bearer
    credential valid wherever it lands -- would follow. Refusing costs nothing:
    SigV4 signs the ``Host`` header, so a signature replayed at another origin
    is invalid there anyway, and following the redirect could only leak the
    token and then fail.

    Returns an empty dict when no signer is configured, so unsigned callers keep
    their existing redirect behavior and every signed call site can spread this
    unconditionally.
    """
    return {"allow_redirects": False} if signer is not None else {}


@asynccontextmanager
async def endpoint_signer(
    cfg: BenchmarkConfig,
) -> AsyncIterator[RequestSignerProtocol | None]:
    """Build, start, and tear down the endpoint's request signer, if any.

    Yields ``None`` when ``endpoint.auth_type`` is unset, so callers can pass
    the result straight to :func:`sign_request` without branching. The signer
    is a full lifecycle object (background credential re-resolution), so it is
    started and stopped rather than merely constructed.
    """
    auth_type = cfg.endpoint.auth_type
    if not auth_type:
        yield None
        return

    # Imported lazily: aiperf.common must not import aiperf.auth /
    # aiperf.plugin models at module scope (circular import via
    # ModelEndpointInfo -> aiperf.config).
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    signer_class = plugins.get_class(PluginType.REQUEST_SIGNER, auth_type)
    signer = signer_class(model_endpoint=ModelEndpointInfo.from_config(cfg))
    # Start inside the try: a signer that fails partway through start-up has
    # already resolved credentials and scheduled its refresh background task,
    # and the finally is what guarantees those are torn down here rather than
    # via AIPerfLifecycleMixin internals.
    try:
        try:
            await signer.initialize_and_start()
        except asyncio.CancelledError as exc:
            # AIPerfLifecycleMixin._fail() reports a start-up failure by raising
            # CancelledError -- a BaseException, not an Exception. Callers guard
            # control hooks with `except Exception` (see
            # _single_run.maybe_reset_kv_cache_before_run), so left as-is a
            # missing-credentials failure skips their error path entirely and
            # surfaces as a raw traceback.
            #
            # Only normalize when this task was not actually cancelled: a real
            # cancel must keep propagating or cooperative shutdown breaks.
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            cause = exc.__cause__
            if isinstance(cause, Exception):
                raise cause from exc
            raise
        yield signer
    finally:
        await signer.stop()
