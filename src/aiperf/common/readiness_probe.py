# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight readiness probe for the target inference server(s).

Polls `{url}/v1/models` on each URL until the requested model id appears in
the response, or raises `TimeoutError` when the deadline elapses. If
`/v1/models` returns 404, falls back to a single GET on the base URL so
servers that don't expose a model list (or only expose a healthcheck) still
pass the probe when they respond at all.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import aiohttp
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.transports.aiohttp_client import AioHttpClient

_logger = AIPerfLogger(__name__)

# Minimum per-request timeout for the probe. `interval_s` can be very small
# (e.g. 0.5s for fast-retrying tests); using it directly as a request timeout
# would cause spurious failures on networks with any real latency.
_MIN_REQUEST_TIMEOUT_S = 5.0


def _model_in_payload(payload_text: str, model_name: str) -> bool:
    """Return True if `model_name` appears as a `data[].id` entry in the JSON body."""
    try:
        payload = orjson.loads(payload_text)
    except (orjson.JSONDecodeError, ValueError):
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("id") == model_name for entry in data
    )


async def _wait_for_single_url(
    client: AioHttpClient,
    url: str,
    model_name: str,
    timeout_s: float,
    interval_s: float,
    headers: dict[str, str],
) -> None:
    """Poll one URL until model is ready or deadline elapses."""
    deadline = time.monotonic() + timeout_s
    models_url = url.rstrip("/") + "/v1/models"
    request_timeout_base = max(interval_s, _MIN_REQUEST_TIMEOUT_S)
    attempt = 0

    while True:
        attempt += 1

        # Cap the per-request timeout by the remaining budget so a slow or
        # hung response can never run past the global deadline.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s waiting for model "
                f"'{model_name}' to become ready at {url} "
                f"(checked {attempt - 1} time(s))"
            )
        request_timeout = aiohttp.ClientTimeout(
            total=min(request_timeout_base, remaining)
        )
        record = await client.get_request(
            models_url, headers=headers, timeout=request_timeout
        )

        if record.status == 200 and record.responses:
            body = (
                record.responses[0].text if hasattr(record.responses[0], "text") else ""
            )
            if _model_in_payload(body, model_name):
                _logger.info(
                    f"Model '{model_name}' ready at {url} after {attempt} attempt(s)"
                )
                return
            _logger.info(
                f"Model '{model_name}' not yet in {models_url} (attempt {attempt}), "
                f"retrying in {interval_s}s"
            )
        elif record.status == 404:
            # Fallback: server doesn't expose /v1/models. Try the base URL; if
            # it answers 2xx we accept it as "server up" and move on. Cap the
            # fallback request by the same per-request budget.
            fallback_remaining = deadline - time.monotonic()
            if fallback_remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for model "
                    f"'{model_name}' to become ready at {url} "
                    f"(checked {attempt} time(s))"
                )
            fallback_timeout = aiohttp.ClientTimeout(
                total=min(request_timeout_base, fallback_remaining)
            )
            fallback = await client.get_request(
                url, headers=headers, timeout=fallback_timeout
            )
            if fallback.status is not None and 200 <= fallback.status < 300:
                _logger.info(
                    f"/v1/models not available at {url}; base URL responded "
                    f"{fallback.status} — accepting as ready"
                )
                return
            _logger.info(
                f"/v1/models returned 404 and base URL returned "
                f"{fallback.status or 'error'} at {url} (attempt {attempt}), "
                f"retrying in {interval_s}s"
            )
        else:
            status_repr = (
                record.status if record.status is not None else "connection error"
            )
            error_repr = record.error.message if record.error else ""
            _logger.info(
                f"Probe to {models_url} returned {status_repr} "
                f"{('(' + error_repr + ') ') if error_repr else ''}"
                f"(attempt {attempt}), retrying in {interval_s}s"
            )

        # Pre-check at the top of the loop raises on deadline; here we only
        # need to sleep before the next attempt, capped so we never sleep
        # past the deadline.
        sleep_for = min(interval_s, max(0.0, deadline - time.monotonic()))
        await asyncio.sleep(sleep_for)


async def wait_for_models_ready(
    urls: list[str],
    model_names: list[str],
    timeout_s: float,
    interval_s: float,
    headers: dict[str, str],
) -> None:
    """Block until every model in `model_names` is ready on every URL, or raise TimeoutError.

    URLs and models are checked sequentially — at typical fleet sizes (1-4 URLs,
    1-2 models) this keeps log output legible, and the caller's timeout is
    per-URL-per-model.
    """
    # Imported lazily to avoid a circular import: aiperf.common is imported
    # before aiperf.transports, and AioHttpClient pulls in a mixin that
    # back-imports from aiperf.transports.aiohttp_client.
    from aiperf.transports.aiohttp_client import AioHttpClient

    _logger.info(
        f"Waiting for model(s) {model_names} to be ready at: {', '.join(urls)} "
        f"(timeout={timeout_s}s, interval={interval_s}s)"
    )
    client = AioHttpClient(timeout=max(interval_s, 5.0))
    try:
        for url in urls:
            for model_name in model_names:
                await _wait_for_single_url(
                    client=client,
                    url=url,
                    model_name=model_name,
                    timeout_s=timeout_s,
                    interval_s=interval_s,
                    headers=headers,
                )
    finally:
        await client.close()
