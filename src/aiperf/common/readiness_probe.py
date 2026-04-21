# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight endpoint readiness probe.

Probes every configured (URL, model) pair before benchmarking starts. Three
probe strategies, selected via ``endpoint.wait_for_model_mode``:

- ``inference`` (default) — POST a canned 1-token inference request to the
  configured endpoint. Strongest signal: proves the full serving stack
  (frontend, scheduler, worker, forward pass) is live. Any HTTP status
  below 500 counts as ready — 4xx surfaces the same way on the first real
  benchmark request and doesn't warrant hanging the probe.
- ``models`` — GET ``{url}/v1/models`` and verify the model id appears in
  ``data[]``. Cheap, no tokens consumed. Falls back to a plain GET on the
  base URL if ``/v1/models`` returns 404 so servers without a model list
  still pass when they're responsive. Note: some frontends (including
  Dynamo) can return 200 from ``/v1/models`` before the backend workers
  are actually able to serve — ``inference`` is the more trustworthy
  signal there.
- ``both`` — run ``models`` first on each URL, then ``inference``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Literal

import aiohttp
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.transports.aiohttp_client import AioHttpClient

_logger = AIPerfLogger(__name__)

# "Lo" — the first message ever sent over a network. On Oct 29, 1969, UCLA
# tried to transmit "login" over the ARPANET but the system crashed after
# two characters. A one-byte prompt keeps token cost and KV-cache impact
# minimal on paid / metered backends.
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

# Floor on per-request HTTP timeout. Retry interval can be small for tests
# but network round trips need breathing room.
_MIN_REQUEST_TIMEOUT_S = 5.0

ReadyCheckMode = Literal["models", "inference", "both"]


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


async def _wait_models(
    client: AioHttpClient,
    url: str,
    model_name: str,
    timeout_s: float,
    interval_s: float,
    headers: dict[str, str],
) -> None:
    """Poll ``{url}/v1/models`` until ``model_name`` appears in ``data[]``.

    Falls back to a single GET on the base URL if ``/v1/models`` returns 404
    on any attempt — so servers that don't expose a model list still pass
    when they respond at all.
    """
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


async def _wait_inference(
    client: AioHttpClient,
    url: str,
    model_name: str,
    endpoint_type: str,
    custom_endpoint: str | None,
    timeout_s: float,
    interval_s: float,
    headers: dict[str, str],
) -> None:
    """POST a canned 1-token request to the inference endpoint until it works.

    Any response with ``status < 500`` counts as ready — 4xx means the
    server is live but our payload was rejected (bad auth / bad model /
    bad path), which surfaces the same way on the first real benchmark
    request. Only 5xx and connection errors trigger retries.
    """
    from urllib.parse import urlparse

    # Respect a caller-supplied path (e.g. --custom-endpoint), otherwise if
    # the URL already carries a non-root path use that, otherwise append
    # the OpenAI default for the endpoint type.
    parsed = urlparse(url)
    if custom_endpoint:
        request_url = url.rstrip("/") + "/" + custom_endpoint.lstrip("/")
    elif parsed.path and parsed.path != "/":
        request_url = url.rstrip("/")
    else:
        endpoint_path = _DEFAULT_PATHS.get(endpoint_type, _DEFAULT_PATHS["chat"])
        request_url = url.rstrip("/") + endpoint_path

    payload = dict(_CANNED_PAYLOADS.get(endpoint_type, _CANNED_PAYLOADS["chat"]))
    payload["model"] = model_name
    body = orjson.dumps(payload)

    # Inference requests need more breathing room than a trivial models GET:
    # model load can push even a 1-token forward pass into the seconds range
    # on first request. Floor higher than the models probe.
    request_timeout_base = max(interval_s, 30.0)
    deadline = time.monotonic() + timeout_s
    attempt = 0

    request_headers = {"Content-Type": "application/json", **headers}

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s probing {request_url} "
                f"with model '{model_name}' (checked {attempt - 1} time(s))"
            )
        request_timeout = aiohttp.ClientTimeout(
            total=min(request_timeout_base, remaining)
        )
        record = await client.post_request(
            request_url,
            payload=body,
            headers=request_headers,
            timeout=request_timeout,
        )

        status = record.status
        if status is not None and status < 500:
            _logger.info(
                f"Inference probe ready at {request_url} "
                f"(status={status}, attempt {attempt})"
            )
            return

        status_repr = status if status is not None else "connection error"
        error_repr = record.error.message if record.error else ""
        _logger.info(
            f"Inference probe to {request_url} returned {status_repr} "
            f"{('(' + error_repr + ') ') if error_repr else ''}"
            f"(attempt {attempt}), retrying in {interval_s}s"
        )

        sleep_for = min(interval_s, max(0.0, deadline - time.monotonic()))
        await asyncio.sleep(sleep_for)


async def wait_for_endpoint(
    urls: list[str],
    model_names: list[str],
    mode: ReadyCheckMode,
    endpoint_type: str,
    custom_endpoint: str | None,
    timeout_s: float,
    interval_s: float,
    headers: dict[str, str],
) -> None:
    """Block until every configured (URL, model) pair passes the probe.

    URLs and models are checked sequentially so log output stays legible at
    typical fleet sizes (1-4 URLs, 1-2 models). The caller's ``timeout_s``
    is applied per probe invocation, so the worst-case total wall-clock is
    roughly ``timeout_s * len(urls) * len(models)`` — pick a generous value.
    """
    # Imported lazily to avoid a circular import: aiperf.common is imported
    # before aiperf.transports, and AioHttpClient pulls in a mixin that
    # back-imports from aiperf.transports.aiohttp_client.
    from aiperf.transports.aiohttp_client import AioHttpClient

    if mode in ("models", "both") and not model_names:
        raise ValueError(
            f"wait-for-model mode={mode!r} requires at least one model name"
        )
    if not urls:
        return

    _logger.info(
        f"Waiting for endpoint readiness (mode={mode}, timeout={timeout_s}s, "
        f"interval={interval_s}s) across {len(urls)} URL(s) x "
        f"{len(model_names)} model(s)"
    )

    client = AioHttpClient(timeout=max(interval_s, _MIN_REQUEST_TIMEOUT_S))
    try:
        for url in urls:
            if mode in ("models", "both"):
                for model_name in model_names:
                    await _wait_models(
                        client=client,
                        url=url,
                        model_name=model_name,
                        timeout_s=timeout_s,
                        interval_s=interval_s,
                        headers=headers,
                    )
            if mode in ("inference", "both"):
                # For the inference probe, any successful generation proves
                # the stack is live — we don't need to loop across every
                # model name. Use the first configured model (or fall back
                # to "default" so the probe still works when model_names is
                # empty, which the validator above permits only in pure
                # inference mode).
                probe_model = model_names[0] if model_names else "default"
                await _wait_inference(
                    client=client,
                    url=url,
                    model_name=probe_model,
                    endpoint_type=endpoint_type,
                    custom_endpoint=custom_endpoint,
                    timeout_s=timeout_s,
                    interval_s=interval_s,
                    headers=headers,
                )
    finally:
        await client.close()
