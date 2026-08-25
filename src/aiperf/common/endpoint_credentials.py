# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-free transport and rehydration for endpoint credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import orjson

from aiperf.common.redact import (
    REDACTED_VALUE,
    extract_sensitive_headers,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

AIPERF_INJECTED_API_KEY = "AIPERF_INJECTED_API_KEY"
AIPERF_INJECTED_HEADERS = "AIPERF_INJECTED_HEADERS"
AIPERF_INJECTED_ENDPOINT_URLS = "AIPERF_INJECTED_ENDPOINT_URLS"
OPENAI_API_KEY = "OPENAI_API_KEY"


@dataclass(frozen=True, slots=True)
class EndpointCredentialInjection:
    """Endpoint secrets consumed from process environment variables."""

    api_key: str | None
    """API key supplied through the private transport or compatibility alias."""

    api_key_from_alias: bool
    """True when ``api_key`` came from ``OPENAI_API_KEY`` rather than the
    private ``AIPERF_INJECTED_API_KEY`` transport. Alias values are ambient
    shell state, so they may only rehydrate an authored-then-redacted key."""

    headers: dict[str, str] | None
    """Credential-bearing headers decoded from the injected JSON object."""

    urls: list[str] | None
    """Full endpoint URLs decoded from the injected JSON string list."""


def parse_injected_dict(name: str, raw: str | None) -> dict[str, str] | None:
    """Decode a JSON object whose values must all be strings."""
    if not raw:
        return None
    try:
        decoded = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"{name} contains invalid JSON: {e}") from e
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{name} must decode to a JSON object, got {type(decoded).__name__}"
        )
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in decoded.items()
    ):
        raise ValueError(f"{name} must decode to a JSON object with string values")
    return decoded


def parse_injected_str_list(name: str, raw: str | None) -> list[str] | None:
    """Decode a JSON list whose values must all be strings."""
    if not raw:
        return None
    try:
        decoded = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"{name} contains invalid JSON: {e}") from e
    if not isinstance(decoded, list) or not all(
        isinstance(url, str) for url in decoded
    ):
        raise ValueError(f"{name} must decode to a JSON list of strings")
    return decoded


def consume_endpoint_credentials() -> EndpointCredentialInjection:
    """Pop, validate, and return endpoint credential environment variables.

    The private ``AIPERF_INJECTED_API_KEY`` name takes precedence over the
    ``OPENAI_API_KEY`` compatibility alias, which covers hand-replayed
    ``run_config.json`` files where only the conventional shell variable is set.
    The alias is reported through ``api_key_from_alias`` so
    :func:`apply_endpoint_credentials` can restrict it to rehydrating a key the
    user actually authored: an ambient shell variable must never become the
    credential of an endpoint that was configured without one.

    Every recognized variable — including ``OPENAI_API_KEY`` — is popped rather
    than read. The orchestrator resolves ``endpoint.api_key`` in the parent
    process (YAML ``${OPENAI_API_KEY}`` substitution and CLI parsing both happen
    there) and forwards the resolved value through ``AIPERF_INJECTED_API_KEY``,
    so nothing downstream of this call needs the raw variable. Leaving it in
    place would publish a common shell credential to every spawned service and
    to ``/proc/<pid>/environ``. Popping mutates only this process's environment,
    never the user's shell.
    """
    private_api_key_present = AIPERF_INJECTED_API_KEY in os.environ
    private_api_key = os.environ.pop(AIPERF_INJECTED_API_KEY, None)
    openai_api_key = os.environ.pop(OPENAI_API_KEY, None)
    headers_raw = os.environ.pop(AIPERF_INJECTED_HEADERS, None)
    urls_raw = os.environ.pop(AIPERF_INJECTED_ENDPOINT_URLS, None)
    return EndpointCredentialInjection(
        api_key=(private_api_key if private_api_key_present else openai_api_key),
        api_key_from_alias=not private_api_key_present,
        headers=parse_injected_dict(AIPERF_INJECTED_HEADERS, headers_raw),
        urls=parse_injected_str_list(AIPERF_INJECTED_ENDPOINT_URLS, urls_raw),
    )


def apply_endpoint_credentials(
    run: BenchmarkRun,
    credentials: EndpointCredentialInjection,
    *,
    require_resolved: bool = False,
) -> None:
    """Overlay injected endpoint credentials onto ``run`` in place.

    An API key from the private ``AIPERF_INJECTED_API_KEY`` transport fills an
    unset or redacted config value; one from the ``OPENAI_API_KEY``
    compatibility alias fills only a redacted value, because the placeholder is
    the sole proof that the user authored a key at all. Without that gate an
    ambient shell variable would be sent as a Bearer token to an endpoint the
    user never configured with credentials. Injected headers override
    same-named authored headers, and injected URLs replace the authored URL
    list. When ``require_resolved`` is true, raise ``ValueError`` if any
    redacted API key, sensitive header, or URL remains after the overlay.
    """
    endpoint = run.cfg.endpoint
    fillable = (
        {REDACTED_VALUE} if credentials.api_key_from_alias else {None, REDACTED_VALUE}
    )
    if credentials.api_key is not None and endpoint.api_key in fillable:
        endpoint.api_key = credentials.api_key
    if credentials.headers:
        endpoint.headers.update(credentials.headers)
    if credentials.urls:
        endpoint.urls = credentials.urls

    if not require_resolved:
        return

    missing: list[str] = []
    if endpoint.api_key == REDACTED_VALUE:
        missing.append(f"{AIPERF_INJECTED_API_KEY} (or {OPENAI_API_KEY})")
    if any(
        value == REDACTED_VALUE
        for value in extract_sensitive_headers(endpoint.headers).values()
    ):
        missing.append(AIPERF_INJECTED_HEADERS)
    if any(REDACTED_VALUE in url for url in endpoint.urls):
        missing.append(AIPERF_INJECTED_ENDPOINT_URLS)
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "benchmark-run contains redacted endpoint credentials; supply them "
            f"through the environment variable(s): {names}"
        )
