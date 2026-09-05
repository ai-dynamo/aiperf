# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated HTTP client for endpoint control-plane POSTs.

Control traffic must not share the inference connection pool, retries, or
ambient HTTP(S)_PROXY settings used by benchmark requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiohttp

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.endpoint_auth import sign_request
from aiperf.common.redact import redact_url

if TYPE_CHECKING:
    from aiperf.auth.base_signer import RequestSignerProtocol

_logger = AIPerfLogger(__name__)


class ControlPlaneHttpError(RuntimeError):
    """Control-plane HTTP failure.

    ``retryable`` distinguishes transport-level failures (timeout, connection
    refused) - which may resolve if the caller waits and retries - from an
    explicit non-2xx response, which is a real rejection retrying won't fix
    in the general case. ``status_code`` carries the response status for a
    non-2xx failure (``None`` for a transport-level failure) so a caller with
    endpoint-specific knowledge can still treat certain statuses (e.g. 503)
    as retryable without this module hardcoding that policy.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


async def control_plane_post(
    *,
    url: str,
    headers: dict[str, str],
    timeout_s: float,
    signer: RequestSignerProtocol | None = None,
) -> None:
    """POST an empty body to a control URL. Success = any 2xx. No retries.

    Transport failures (timeouts, connection errors, ``ClientError``) are
    raised as ``ControlPlaneHttpError`` so callers share one error type.

    ``asyncio.CancelledError`` is intentionally not wrapped, preserving task
    cancellation semantics; callers needing cleanup on cancellation must
    handle it explicitly.

    Signing happens here rather than in the caller so each retry attempt
    carries a fresh signature: a SigV4 signature expires within a five-minute
    skew window that control-hook backoff can outlive.
    """
    body = b""
    url, headers, signed_body = await sign_request(
        signer, method="POST", url=url, headers=headers, body=body
    )
    body = signed_body if signed_body is not None else body
    safe_url = redact_url(url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    # trust_env=False keeps loopback / cluster traffic off ambient HTTP(S)_PROXY.
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=False) as session,
            session.post(url, headers=headers, data=body) as resp,
        ):
            if 200 <= resp.status < 300:
                _logger.debug(lambda: f"control_plane POST {safe_url} -> {resp.status}")
                return
            # Drain body without including it in the error (may contain secrets).
            await resp.read()
            raise ControlPlaneHttpError(
                f"control_plane POST {safe_url} failed with status {resp.status}",
                status_code=resp.status,
            )
    except ControlPlaneHttpError:
        raise
    except (TimeoutError, aiohttp.ClientError) as exc:
        raise ControlPlaneHttpError(
            f"control_plane POST {safe_url} failed: {type(exc).__name__}: {exc}",
            retryable=True,
        ) from exc
