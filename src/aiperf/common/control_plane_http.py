# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated HTTP client for endpoint control-plane POSTs.

Control traffic must not share the inference connection pool, retries, or
ambient HTTP(S)_PROXY settings used by benchmark requests.
"""

from __future__ import annotations

import aiohttp

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.redact import redact_url

_logger = AIPerfLogger(__name__)


class ControlPlaneHttpError(RuntimeError):
    """Non-retryable control-plane HTTP failure."""


async def control_plane_post(
    *,
    url: str,
    headers: dict[str, str],
    timeout_s: float,
) -> None:
    """POST an empty body to a control URL. Success = any 2xx. No retries.

    Transport failures (timeouts, connection errors, ``ClientError``) are
    raised as ``ControlPlaneHttpError`` so callers share one error type.

    ``asyncio.CancelledError`` is intentionally not wrapped, preserving task
    cancellation semantics; callers needing cleanup on cancellation must
    handle it explicitly.
    """
    safe_url = redact_url(url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    # trust_env=False keeps loopback / cluster traffic off ambient HTTP(S)_PROXY.
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=False) as session,
            session.post(url, headers=headers, data=b"") as resp,
        ):
            if 200 <= resp.status < 300:
                _logger.debug(lambda: f"control_plane POST {safe_url} -> {resp.status}")
                return
            # Drain body without including it in the error (may contain secrets).
            await resp.read()
            raise ControlPlaneHttpError(
                f"control_plane POST {safe_url} failed with status {resp.status}"
            )
    except ControlPlaneHttpError:
        raise
    except (TimeoutError, aiohttp.ClientError) as exc:
        raise ControlPlaneHttpError(
            f"control_plane POST {safe_url} failed: {type(exc).__name__}: {exc}"
        ) from exc
