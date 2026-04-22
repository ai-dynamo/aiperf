# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight endpoint readiness checker.

Probes every (URL, model) pair before benchmarking starts. Two probe modes,
chosen via ``endpoint.ready_check_mode``:

- ``models``   — GET ``{url}/v1/models`` and verify the model id appears in
  ``data[]``. Cheap (no tokens consumed). Falls back to a plain GET on the
  base URL if ``/v1/models`` returns 404 so servers without a model list
  still pass when they're responsive.
- ``inference`` — POST a canned 1-token inference request. Strongest signal:
  proves the model weights are loaded and a forward pass works.
- ``both`` — run ``models`` first on each URL, then ``inference``.

Retries every ``ready_check_interval`` seconds until the overall
``ready_check_timeout`` elapses. Uses the same ``AioHttpClient`` as the load
phase so proxy/TLS/header behavior is identical.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import aiohttp
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.transports.aiohttp_client import AioHttpClient

logger = AIPerfLogger(__name__)

# "Lo" — the first message ever sent over a network. On Oct 29, 1969, UCLA
# tried to transmit "login" over the ARPANET but the system crashed after
# two characters. A one-byte prompt keeps token cost minimal on paid backends.
_READINESS_PROMPT = "Lo"

_CANNED_PAYLOADS: dict[str, dict] = {
    "chat": {
        "messages": [{"role": "user", "content": _READINESS_PROMPT}],
        "max_tokens": 1,
    },
    "completions": {
        "prompt": _READINESS_PROMPT,
        "max_tokens": 1,
    },
    "embeddings": {
        "input": _READINESS_PROMPT,
    },
}

_DEFAULT_PATHS: dict[str, str] = {
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
}

# Floor on per-attempt HTTP timeout. Retry interval can be small for tests
# but network round trips need breathing room.
_MIN_REQUEST_TIMEOUT_S = 5.0


ReadyCheckMode = Literal["models", "inference", "both"]


async def wait_for_endpoint(
    urls: list[str],
    model_names: list[str],
    *,
    mode: ReadyCheckMode = "inference",
    endpoint_type: str = "chat",
    path: str | None = None,
    timeout: float = 0.0,
    interval: float = 5.0,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Block until every (URL, model) pair passes the configured probe.

    Args:
        urls: Every endpoint URL that will receive load.
        model_names: Every model name that will be benchmarked. At least one
            is required when ``mode`` is ``models`` or ``both``.
        mode: Probe strategy (see module docstring).
        endpoint_type: OpenAI-compatible type: chat/completions/embeddings.
            Only used by the inference probe.
        path: Override the default API path for inference probes. Ignored
            when the URL already has a non-root path.
        timeout: Overall budget in seconds (<=0 skips the check entirely).
            Applied per (URL, model) pair — a generous default is fine.
        interval: Seconds between retry attempts.
        api_key: Bearer token for Authorization header.
        headers: Additional headers merged into each probe request.

    Raises:
        TimeoutError: If any pair fails to become ready within ``timeout``.
        ValueError: If ``mode`` requires model_names and none were supplied.
    """
    if timeout <= 0:
        return

    if not urls:
        return

    if mode in ("models", "both") and not model_names:
        raise ValueError(f"ready_check_mode={mode!r} requires at least one model name")

    merged_headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        merged_headers["Authorization"] = f"Bearer {api_key}"
    if headers:
        merged_headers.update(headers)

    logger.info(
        f"Waiting for endpoint readiness (mode={mode}, timeout={timeout}s, "
        f"interval={interval}s) across {len(urls)} URL(s) x {len(model_names)} model(s)"
    )

    client = AioHttpClient(timeout=max(interval, _MIN_REQUEST_TIMEOUT_S))
    try:
        for url in urls:
            if mode in ("models", "both"):
                for model_name in model_names:
                    await _wait_models(
                        client,
                        url,
                        model_name,
                        timeout=timeout,
                        interval=interval,
                        headers=merged_headers,
                    )
            if mode in ("inference", "both"):
                # For the inference probe the model id is echoed back in
                # the response but not required for "server can generate".
                # Pick the first model (or "default") — any successful
                # generation proves the stack is live.
                probe_model = model_names[0] if model_names else "default"
                await _wait_inference(
                    client,
                    url,
                    endpoint_type=endpoint_type,
                    path=path,
                    model=probe_model,
                    timeout=timeout,
                    interval=interval,
                    headers=merged_headers,
                )
    finally:
        await client.close()


async def _wait_models(
    client: AioHttpClient,
    url: str,
    model_name: str,
    *,
    timeout: float,
    interval: float,
    headers: dict[str, str],
) -> None:
    """Poll ``{url}/v1/models`` until ``model_name`` appears in ``data[]``.

    Falls back to a single GET on the base URL if ``/v1/models`` returns 404
    on any attempt — so servers that don't expose a model list still pass
    when they respond at all.
    """
    deadline = time.monotonic() + timeout
    models_url = url.rstrip("/") + "/v1/models"
    attempt = 0

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for model "
                f"{model_name!r} at {url} (checked {attempt - 1}x)"
            )
        per_request = aiohttp.ClientTimeout(
            total=min(max(interval, _MIN_REQUEST_TIMEOUT_S), remaining)
        )
        record = await client.get_request(
            models_url, headers=headers, timeout=per_request
        )

        if record.status == 200 and record.responses:
            body = getattr(record.responses[0], "text", "") or ""
            if _model_in_payload(body, model_name):
                logger.info(
                    f"Model {model_name!r} ready at {url} after {attempt} attempt(s)"
                )
                return
            logger.info(
                f"Model {model_name!r} not in /v1/models yet at {url} "
                f"(attempt {attempt}); retrying in {interval}s"
            )
        elif record.status == 404:
            fallback_remaining = deadline - time.monotonic()
            if fallback_remaining > 0:
                fallback_timeout = aiohttp.ClientTimeout(
                    total=min(max(interval, _MIN_REQUEST_TIMEOUT_S), fallback_remaining)
                )
                fallback = await client.get_request(
                    url, headers=headers, timeout=fallback_timeout
                )
                if fallback.status is not None and 200 <= fallback.status < 300:
                    logger.info(
                        f"/v1/models 404 at {url}; base URL replied "
                        f"{fallback.status} — accepting as ready"
                    )
                    return
                logger.info(
                    f"/v1/models 404 and base URL replied "
                    f"{fallback.status or 'error'} at {url} "
                    f"(attempt {attempt}); retrying in {interval}s"
                )
        else:
            status_repr = record.status if record.status is not None else "conn-error"
            logger.info(
                f"/v1/models probe {status_repr} at {url} "
                f"(attempt {attempt}); retrying in {interval}s"
            )

        await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))


async def _wait_inference(
    client: AioHttpClient,
    url: str,
    *,
    endpoint_type: str,
    path: str | None,
    model: str,
    timeout: float,
    interval: float,
    headers: dict[str, str],
) -> None:
    """Poll the inference endpoint with a canned 1-token request until ready.

    Any ``status < 500`` counts as ready — 4xx proves the server is live
    (wrong auth/path/model will surface as the real benchmark's first
    request rather than hanging this probe).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.path and parsed.path != "/":
        request_url = url.rstrip("/")
    else:
        endpoint_path = path or _DEFAULT_PATHS.get(
            endpoint_type, _DEFAULT_PATHS["chat"]
        )
        request_url = url.rstrip("/") + endpoint_path

    payload = dict(_CANNED_PAYLOADS.get(endpoint_type, _CANNED_PAYLOADS["chat"]))
    payload["model"] = model

    deadline = time.monotonic() + timeout
    body = orjson.dumps(payload)
    attempt = 0

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s probing {request_url} "
                f"with model {model!r} (checked {attempt - 1}x)"
            )
        per_request = aiohttp.ClientTimeout(total=min(max(interval, 30.0), remaining))
        record = await client.post_request(
            request_url, headers=headers, data=body, timeout=per_request
        )

        status = record.status
        if status is not None and status < 500:
            logger.info(
                f"Inference probe ready at {request_url} "
                f"(status={status}, attempt {attempt})"
            )
            return

        status_repr = status if status is not None else "conn-error"
        logger.info(
            f"Inference probe {status_repr} at {request_url} "
            f"(attempt {attempt}); retrying in {interval}s"
        )
        await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def _model_in_payload(payload_text: str, model_name: str) -> bool:
    """Return True if ``model_name`` appears as a ``data[].id`` in the JSON body."""
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
