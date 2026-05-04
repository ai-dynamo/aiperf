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
from typing import TYPE_CHECKING, Any, Literal

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


def _models_timeout(
    *,
    deadline: float,
    request_timeout_base: float,
    timeout_s: float,
    model_name: str,
    url: str,
    checked_attempts: int,
) -> aiohttp.ClientTimeout:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s waiting for model "
            f"'{model_name}' to become ready at {url} "
            f"(checked {checked_attempts} time(s))"
        )
    return aiohttp.ClientTimeout(total=min(request_timeout_base, remaining))


def _inference_timeout(
    *,
    deadline: float,
    request_timeout_base: float,
    timeout_s: float,
    request_url: str,
    model_name: str,
    checked_attempts: int,
) -> aiohttp.ClientTimeout:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s probing {request_url} "
            f"with model '{model_name}' (checked {checked_attempts} time(s))"
        )
    return aiohttp.ClientTimeout(total=min(request_timeout_base, remaining))


async def _sleep_until_next_attempt(*, deadline: float, interval_s: float) -> None:
    sleep_for = min(interval_s, max(0.0, deadline - time.monotonic()))
    await asyncio.sleep(sleep_for)


def _response_status_and_error(record: Any) -> tuple[int | str, str]:
    status_repr = record.status if record.status is not None else "connection error"
    error_repr = record.error.message if record.error else ""
    return status_repr, error_repr


def _models_response_ready(
    *,
    record: Any,
    model_name: str,
    url: str,
    models_url: str,
    attempt: int,
    interval_s: float,
) -> bool:
    body = record.responses[0].text if hasattr(record.responses[0], "text") else ""
    if _model_in_payload(body, model_name):
        _logger.info(f"Model '{model_name}' ready at {url} after {attempt} attempt(s)")
        return True
    _logger.info(
        f"Model '{model_name}' not yet in {models_url} (attempt {attempt}), "
        f"retrying in {interval_s}s"
    )
    return False


async def _base_url_ready_after_models_404(
    *,
    client: AioHttpClient,
    url: str,
    model_name: str,
    deadline: float,
    timeout_s: float,
    request_timeout_base: float,
    attempt: int,
    interval_s: float,
    headers: dict[str, str],
) -> bool:
    fallback_timeout = _models_timeout(
        deadline=deadline,
        request_timeout_base=request_timeout_base,
        timeout_s=timeout_s,
        model_name=model_name,
        url=url,
        checked_attempts=attempt,
    )
    fallback = await client.get_request(url, headers=headers, timeout=fallback_timeout)
    if fallback.status is not None and 200 <= fallback.status < 300:
        _logger.info(
            f"/v1/models not available at {url}; base URL responded "
            f"{fallback.status} — accepting as ready"
        )
        return True
    _logger.info(
        f"/v1/models returned 404 and base URL returned "
        f"{fallback.status or 'error'} at {url} (attempt {attempt}), "
        f"retrying in {interval_s}s"
    )
    return False


def _build_inference_probe_request(
    *,
    url: str,
    model_name: str,
    endpoint_type: str,
    custom_endpoint: str | None,
) -> tuple[str, bytes]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    endpoint_path = _DEFAULT_PATHS.get(endpoint_type)
    payload_template = _CANNED_PAYLOADS.get(endpoint_type)
    if endpoint_path is None or payload_template is None:
        _logger.warning(
            f"Inference readiness probe has no dedicated request template for "
            f"endpoint type '{endpoint_type}'; using the chat-compatible "
            f"fallback. This checks server liveness but may not prove model "
            f"readiness for that endpoint type."
        )

    if custom_endpoint:
        request_url = url.rstrip("/") + "/" + custom_endpoint.lstrip("/")
    elif parsed.path and parsed.path != "/":
        request_url = url.rstrip("/")
    else:
        request_url = url.rstrip("/") + (endpoint_path or _DEFAULT_PATHS["chat"])

    payload = dict(payload_template or _CANNED_PAYLOADS["chat"])
    payload["model"] = model_name
    return request_url, orjson.dumps(payload)


async def _wait_models(
    *,
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
        request_timeout = _models_timeout(
            deadline=deadline,
            request_timeout_base=request_timeout_base,
            timeout_s=timeout_s,
            model_name=model_name,
            url=url,
            checked_attempts=attempt - 1,
        )
        record = await client.get_request(
            models_url, headers=headers, timeout=request_timeout
        )

        if record.status == 200 and record.responses:
            if _models_response_ready(
                record=record,
                model_name=model_name,
                url=url,
                models_url=models_url,
                attempt=attempt,
                interval_s=interval_s,
            ):
                return
        elif record.status == 404:
            if await _base_url_ready_after_models_404(
                client=client,
                url=url,
                model_name=model_name,
                deadline=deadline,
                timeout_s=timeout_s,
                request_timeout_base=request_timeout_base,
                attempt=attempt,
                interval_s=interval_s,
                headers=headers,
            ):
                return
        else:
            status_repr, error_repr = _response_status_and_error(record)
            _logger.info(
                f"Probe to {models_url} returned {status_repr} "
                f"{('(' + error_repr + ') ') if error_repr else ''}"
                f"(attempt {attempt}), retrying in {interval_s}s"
            )

        await _sleep_until_next_attempt(deadline=deadline, interval_s=interval_s)


async def _wait_inference(
    *,
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
    request_url, body = _build_inference_probe_request(
        url=url,
        model_name=model_name,
        endpoint_type=endpoint_type,
        custom_endpoint=custom_endpoint,
    )
    # Inference requests need more breathing room than a trivial models GET:
    # model load can push even a 1-token forward pass into the seconds range
    # on first request. Floor higher than the models probe.
    request_timeout_base = max(interval_s, 30.0)
    deadline = time.monotonic() + timeout_s
    attempt = 0

    request_headers = {"Content-Type": "application/json", **headers}

    while True:
        attempt += 1
        request_timeout = _inference_timeout(
            deadline=deadline,
            request_timeout_base=request_timeout_base,
            timeout_s=timeout_s,
            request_url=request_url,
            model_name=model_name,
            checked_attempts=attempt - 1,
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

        status_repr, error_repr = _response_status_and_error(record)
        _logger.info(
            f"Inference probe to {request_url} returned {status_repr} "
            f"{('(' + error_repr + ') ') if error_repr else ''}"
            f"(attempt {attempt}), retrying in {interval_s}s"
        )

        await _sleep_until_next_attempt(deadline=deadline, interval_s=interval_s)


async def wait_for_endpoint(
    *,
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
                # Probe every configured model. In a multi-model deployment
                # (e.g. --model foo,bar with round_robin selection), `foo`
                # being ready tells us nothing about `bar` — different
                # weights, possibly on different workers. Fall back to
                # ["default"] when model_names is empty, which the
                # validator above permits only in pure inference mode.
                probe_models = model_names if model_names else ["default"]
                for probe_model in probe_models:
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
