# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model autodetection helpers.

Used by ``aiperf profile`` when ``--model/--model-names`` is omitted: poll
``GET {base_url}/v1/models`` until a model appears (or the timeout is
exhausted), then return the first discovered model ID.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.transports.aiohttp_client import AioHttpClient

_logger = AIPerfLogger(__name__)

_MIN_REQUEST_TIMEOUT_S = 5.0


def _models_url_from_base(base_url: str) -> str:
    """Build the /v1/models URL, handling bases that already end in /v1."""
    _base = base_url.rstrip("/")
    return _base + ("/models" if _base.endswith("/v1") else "/v1/models")


def _extract_ids_from_payload(body_text: str) -> list[str]:
    try:
        payload = orjson.loads(body_text)
    except (orjson.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [
        entry["id"]
        for entry in data
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and entry.get("id")
    ]


def _log_and_return_chosen(ids: list[str], models_url: str) -> str:
    chosen = ids[0]
    if len(ids) > 1:
        _logger.warning(
            f"{len(ids)} models returned by {models_url}; "
            "pass --model to select one explicitly"
        )
        _logger.warning(f"No --model provided; using first listed model '{chosen}'")
    else:
        _logger.info(f"Auto-detected model '{chosen}' from {models_url}")
    return chosen


def _parse_ids_from_record(record: Any) -> list[str]:
    if record.status != 200 or not record.responses:
        return []
    body_text = getattr(record.responses[0], "text", None)
    if not isinstance(body_text, str) or not body_text:
        return []
    return _extract_ids_from_payload(body_text)


async def autodetect_names(
    *,
    urls: list[str],
    headers: dict[str, str],
    timeout_s: float = 10.0,
    interval_s: float = 5.0,
) -> list[str]:
    """Poll ``GET {url}/v1/models`` until a model appears and return it.

    Retries on transient failures (connection errors, non-200 responses, empty
    ``data[]``) until ``timeout_s`` is exhausted.  Only the first URL is used
    for discovery; pass ``--model`` explicitly when multiple URLs serve
    different model sets.

    Args:
        urls: Base URLs. Only the first is used for autodetection.
        headers: HTTP headers to attach to every request.
        timeout_s: Total wall-clock budget. Raised as ``TimeoutError`` when
            exhausted without a successful discovery.
        interval_s: Pause between retries.

    Returns:
        A single-element list containing the first discovered model ID.

    Raises:
        ValueError: ``urls`` is empty.
        TimeoutError: No model discovered within ``timeout_s``.
    """
    if not urls:
        raise ValueError("Autodetection requires at least one --url base URL")

    base_url = urls[0]
    models_url = _models_url_from_base(base_url)
    deadline = time.monotonic() + timeout_s
    request_timeout_base = max(interval_s, _MIN_REQUEST_TIMEOUT_S)
    attempt = 0

    client = AioHttpClient(timeout=request_timeout_base)
    try:
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s auto-detecting models from "
                    f"{models_url} (checked {attempt - 1} time(s))"
                )
            request_timeout = aiohttp.ClientTimeout(
                total=min(request_timeout_base, remaining)
            )
            record = await client.get_request(
                models_url, headers=headers, timeout=request_timeout
            )
            ids = _parse_ids_from_record(record)
            if ids:
                return [_log_and_return_chosen(ids, models_url)]

            status_repr = (
                record.status if record.status is not None else "connection error"
            )
            _logger.info(
                f"Auto-detect probe to {models_url} returned {status_repr} "
                f"(attempt {attempt}), retrying in {interval_s}s"
            )
            await asyncio.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))
    finally:
        await client.close()
