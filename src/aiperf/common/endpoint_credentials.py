# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-free transport and rehydration for endpoint credentials."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.redact import (
    REDACTED_VALUE,
    extract_sensitive_headers,
    redact_string,
    redact_url,
)

if TYPE_CHECKING:
    from aiperf.config.endpoint import EndpointConfig
    from aiperf.config.resolution.plan import BenchmarkRun

AIPERF_INJECTED_API_KEY = "AIPERF_INJECTED_API_KEY"
AIPERF_INJECTED_HEADERS = "AIPERF_INJECTED_HEADERS"
AIPERF_INJECTED_ENDPOINT_URLS = "AIPERF_INJECTED_ENDPOINT_URLS"
OPENAI_API_KEY = "OPENAI_API_KEY"

_SWEEP_CREDENTIAL_PATH_SEGMENTS = frozenset(
    {
        "apikey",
        "xapikey",
        "authorization",
        "proxyauthorization",
        "token",
        "apitoken",
        "authtoken",
        "accesstoken",
        "bearertoken",
        "idtoken",
        "refreshtoken",
        "secret",
        "clientsecret",
        "password",
        "passwd",
        "credential",
        "awsaccesskeyid",
        "signature",
        "ocpapimsubscriptionkey",
        "xgoogapikey",
        "xfunctionskey",
        "aegsaskey",
        "xamzsecuritytoken",
    }
)
_SWEEP_CREDENTIAL_PATH_SUFFIXES = (
    "apikey",
    "apitoken",
    "authorization",
    "authtoken",
    "accesstoken",
    "bearertoken",
    "idtoken",
    "refreshtoken",
    "clientsecret",
    "secret",
    "password",
    "passwd",
    "credential",
    "signature",
)
_SWEEP_CREDENTIAL_PATH_PREFIXES = (
    "apikey",
    "authorization",
    "clientsecret",
    "secret",
    "password",
    "passwd",
    "credential",
)
_SWEEP_JSON_DISPLAY_FIELDS = frozenset({"values", "variationvalues"})
_SWEEP_LABEL_DISPLAY_FIELDS = frozenset({"label", "variationlabel"})


def _normalize_sweep_path_segment(segment: str) -> str:
    return "".join(character for character in segment.lower() if character.isalnum())


def _sweep_path_segments(path: str) -> list[str]:
    return [
        _normalize_sweep_path_segment(segment)
        for segment in path.replace("[", ".").replace("]", "").split(".")
        if segment
    ]


def _is_sweep_credential_path(path: str) -> bool:
    segments = _sweep_path_segments(path)
    if any(
        left == "endpoint" and right == "headers"
        for left, right in zip(segments, segments[1:], strict=False)
    ):
        return True
    return any(
        segment in _SWEEP_CREDENTIAL_PATH_SEGMENTS
        or any(segment.endswith(suffix) for suffix in _SWEEP_CREDENTIAL_PATH_SUFFIXES)
        or any(segment.startswith(prefix) for prefix in _SWEEP_CREDENTIAL_PATH_PREFIXES)
        for segment in segments
    )


def _redact_sweep_string(value: str, path: str) -> str:
    segments = _sweep_path_segments(path)
    if segments and segments[-1] in _SWEEP_JSON_DISPLAY_FIELDS:
        try:
            decoded = orjson.loads(value)
        except orjson.JSONDecodeError:
            pass
        else:
            if isinstance(decoded, dict | list):
                redacted = redact_sweep_public_data(decoded, path=path)
                if redacted == decoded:
                    return value
                return orjson.dumps(redacted).decode()
    return redact_url(redact_string(value))


def redact_sweep_public_data(value: Any, *, path: str = "") -> Any:
    """Return a redacted display projection of sweep-derived data.

    The raw planner/executor values remain untouched. This projection is for
    labels, annotations, status, manifests, snapshots, and API responses.
    Dotted parameter keys are treated as path segments, so both
    ``{"endpoint.apiKey": "..."}`` and nested endpoint mappings are covered.
    """
    if path and _is_sweep_credential_path(path):
        return REDACTED_VALUE
    if isinstance(value, dict):
        return {
            key: redact_sweep_public_data(
                child,
                path=f"{path}.{key}" if path else str(key),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            redact_sweep_public_data(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            redact_sweep_public_data(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, str):
        segments = _sweep_path_segments(path)
        if segments and segments[-1] in _SWEEP_LABEL_DISPLAY_FIELDS:
            return redact_sweep_display_label(value)
        return _redact_sweep_string(value, path)
    return value


def redact_sweep_display_label(label: str) -> str:
    """Return a credential-safe sweep or variation display label."""
    components: list[str] = []
    for component in label.split(","):
        path, separator, _value = component.partition("=")
        if separator and _is_sweep_credential_path(path.strip()):
            component = f"{path}={REDACTED_VALUE}"
        components.append(component)
    return _redact_sweep_string(",".join(components), "variation_label")


def _first_sweep_credential_path(value: Any, *, path: str) -> str | None:
    if _is_sweep_credential_path(path):
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found = _first_sweep_credential_path(child, path=child_path)
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            found = _first_sweep_credential_path(child, path=f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(value, str) and _redact_sweep_string(value, path) != value:
        return path
    return None


def _sweep_parameter_candidates(data: dict[str, Any]) -> list[tuple[str, Any]]:
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        return []
    return [(str(path), values) for path, values in parameters.items()]


def _sweep_dimension_candidates(data: dict[str, Any]) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    for field in ("searchSpace", "search_space", "dimensions"):
        dimensions = data.get(field)
        if not isinstance(dimensions, list):
            continue
        for dimension in dimensions:
            if not isinstance(dimension, dict):
                continue
            dimension_path = dimension.get("path")
            if isinstance(dimension_path, str):
                candidates.append((dimension_path, dimension.get("choices", [])))
    return candidates


def _sweep_scenario_candidates(data: dict[str, Any]) -> list[tuple[str, Any]]:
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    return [(f"sweep.runs[{index}]", run) for index, run in enumerate(runs)]


def validate_kubernetes_sweep_credential_axes(sweep: Any) -> None:
    """Reject Kubernetes sweep axes that could persist credential values.

    Kubernetes sweep values are copied into child annotations, parent status,
    aggregate artifacts, and API responses. Per-variation secret transport is
    not supported, so credential-bearing axes fail before any child is created.
    """
    if sweep is None:
        return
    data = (
        sweep.model_dump(mode="python", by_alias=True, exclude_none=True)
        if hasattr(sweep, "model_dump")
        else sweep
    )
    if not isinstance(data, dict):
        return

    candidates = [
        *_sweep_parameter_candidates(data),
        *_sweep_dimension_candidates(data),
        *_sweep_scenario_candidates(data),
    ]

    for path, values in candidates:
        credential_path = _first_sweep_credential_path(values, path=path)
        if credential_path is None:
            continue
        raise ValueError(
            "Kubernetes sweeps cannot vary credential-bearing values; "
            f"sweep path {credential_path!r} is not supported. Keep endpoint "
            "credentials fixed and inject them through Secret-backed pod "
            "environment variables."
        )


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


_MIN_REDACTABLE_SECRET_LENGTH = 4


def credential_values(
    *,
    credentials: EndpointCredentialInjection | None = None,
    endpoint: Any = None,
) -> list[str]:
    """Return every plaintext endpoint credential value known to this process.

    Collected from the injected environment payload and/or a resolved
    ``EndpointConfig``: the API key, the values of credential-bearing headers,
    and any endpoint URL carrying userinfo or a credential query parameter.
    Attributes are read defensively (``isinstance`` gated) so a partially
    constructed or mocked endpoint can never turn error reporting into a second
    failure. The redaction placeholder is never returned as a secret.
    """
    values: list[str] = []

    if credentials is not None:
        if isinstance(credentials.api_key, str):
            values.append(credentials.api_key)
        if credentials.headers:
            values.extend(
                value
                for value in credentials.headers.values()
                if isinstance(value, str)
            )
        if credentials.urls:
            values.extend(
                url
                for url in credentials.urls
                if isinstance(url, str) and redact_url(url) != url
            )

    if endpoint is not None:
        api_key = getattr(endpoint, "api_key", None)
        if isinstance(api_key, str):
            values.append(api_key)
        headers = getattr(endpoint, "headers", None)
        if isinstance(headers, dict):
            values.extend(
                value
                for value in extract_sensitive_headers(headers).values()
                if isinstance(value, str)
            )
        urls = getattr(endpoint, "urls", None)
        if isinstance(urls, list | tuple):
            values.extend(
                url for url in urls if isinstance(url, str) and redact_url(url) != url
            )

    return [value for value in values if value and value != REDACTED_VALUE]


def redact_credential_text(text: str, secrets: Iterable[str]) -> str:
    """Scrub known credential values out of arbitrary text.

    Exact-value replacement runs first because an exception message or
    traceback frame can embed a raw key with none of the ``api_key=`` /
    ``Authorization:`` context that pattern-based redaction keys off. The
    generic string and URL redactors then run as a fallback for credentials
    this process never held. Values shorter than
    ``_MIN_REDACTABLE_SECRET_LENGTH`` are skipped: a one- or two-character
    "credential" would shred every unrelated word in a traceback while
    protecting nothing real.
    """
    unique = {
        secret
        for secret in secrets
        if len(secret) >= _MIN_REDACTABLE_SECRET_LENGTH and secret != REDACTED_VALUE
    }
    # Longest first so a secret that contains another (a userinfo URL and the
    # password inside it) is replaced whole rather than left partially intact.
    for secret in sorted(unique, key=len, reverse=True):
        text = text.replace(secret, REDACTED_VALUE)
    return redact_url(redact_string(text))


def validate_kubernetes_credential_transport(
    endpoint: EndpointConfig,
    pod_env: list[dict[str, object]],
) -> None:
    """Reject credentialed runs without matching Secret-backed pod env vars."""
    secret_env_names = {
        item.get("name")
        for item in pod_env
        if isinstance(item, dict)
        and isinstance(item.get("valueFrom"), dict)
        and isinstance(item["valueFrom"].get("secretKeyRef"), dict)
    }
    missing: list[str] = []
    if endpoint.api_key is not None and not (
        {AIPERF_INJECTED_API_KEY, OPENAI_API_KEY} & secret_env_names
    ):
        missing.append(f"{AIPERF_INJECTED_API_KEY} (or {OPENAI_API_KEY})")
    if (
        extract_sensitive_headers(endpoint.headers)
        and AIPERF_INJECTED_HEADERS not in secret_env_names
    ):
        missing.append(AIPERF_INJECTED_HEADERS)
    from aiperf.common.redact import redact_url

    if (
        any(REDACTED_VALUE in url or redact_url(url) != url for url in endpoint.urls)
        and AIPERF_INJECTED_ENDPOINT_URLS not in secret_env_names
    ):
        missing.append(AIPERF_INJECTED_ENDPOINT_URLS)
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "Kubernetes endpoint credentials must come from Secret-backed pod "
            f"environment variables. Configure --env-from-secrets for: {names}"
        )
