# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized redaction utilities for sensitive data (API keys, auth tokens, etc.)."""

import re
from collections.abc import Sequence

REDACTED_VALUE = "<redacted>"

# Header names (case-insensitive) whose values carry credentials.
# Covers standard HTTP auth plus all major LLM/cloud API providers:
#   - authorization / proxy-authorization: Bearer tokens (OpenAI, Groq, Cohere, Mistral,
#     Together, HuggingFace, Google, NVIDIA NIM, etc.), Basic auth
#   - x-api-key: Anthropic
#   - api-key: Azure OpenAI
#   - ocp-apim-subscription-key: Azure API Management
#   - x-goog-api-key: Google Cloud / Vertex AI
#   - x-functions-key: Azure Functions
#   - aeg-sas-key: Azure Event Grid
#   - x-amz-security-token: AWS STS temporary credentials
_SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "ocp-apim-subscription-key",
        "x-goog-api-key",
        "x-functions-key",
        "aeg-sas-key",
        "x-amz-security-token",
    }
)

# Pre-compiled regex patterns for redacting credentials in arbitrary strings.
# Patterns must handle plain text ("Authorization: Bearer <key>"),
# JSON-serialized ('"Authorization":"Bearer <key>"'), and Python repr
# ("'Authorization': 'Bearer <key>'") forms.
# Build alternation from non-auth headers for string-level redaction.
# Authorization/proxy-authorization are handled separately (they have Bearer/Basic schemes).
_NON_AUTH_SENSITIVE_HEADERS = _SENSITIVE_HEADER_NAMES - {
    "authorization",
    "proxy-authorization",
}
_NON_AUTH_HEADER_ALT = "|".join(
    re.escape(h) for h in sorted(_NON_AUTH_SENSITIVE_HEADERS)
)

_STRING_REDACTION_PATTERNS = [
    # Authorization / Proxy-Authorization: redact the entire value including multi-token
    # schemes like SigV4 (AWS4-HMAC-SHA256 Credential=..., Signature=...).
    # Uses [^'"\}\n]+ to consume everything up to the enclosing quote/brace/newline,
    # which correctly handles JSON, Python repr, and plain text contexts.
    (
        re.compile(
            r"""(?i)((?:proxy-)?authorization['":\s]*(?:bearer|basic)?\s*)[^'"\}\n]+"""
        ),
        rf"\1{REDACTED_VALUE}",
    ),
    # api_key=<value>, token=<value>, secret=<value> (query string style)
    (
        re.compile(r"(?i)\b(api[-_ ]?key|token|secret)\s*=\s*[^&\s]+"),
        rf"\1={REDACTED_VALUE}",
    ),
    # Other credential-carrying headers (plain text, JSON, Python repr)
    (
        re.compile(rf"""(?i)({_NON_AUTH_HEADER_ALT})['":\s]*[^\s,;'"\}}]+"""),
        rf"\1: {REDACTED_VALUE}",
    ),
]


def redact_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Return a copy of headers with sensitive values replaced by REDACTED_VALUE.

    Matches against _SENSITIVE_HEADER_NAMES (case-insensitive).
    Returns None if input is None.
    """
    if headers is None:
        return None
    return {
        k: (REDACTED_VALUE if k.lower() in _SENSITIVE_HEADER_NAMES else v)
        for k, v in headers.items()
    }


def redact_header_tuples(
    headers: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return a copy of header tuples with sensitive values replaced by REDACTED_VALUE.

    Same logic as redact_headers but for (name, value) tuple format
    used by InputConfig.headers.
    """
    return [
        (name, REDACTED_VALUE if name.lower() in _SENSITIVE_HEADER_NAMES else value)
        for name, value in headers
    ]


def redact_string(value: str) -> str:
    """Redact credentials embedded in an arbitrary string (e.g., exception repr)."""
    for pattern, repl in _STRING_REDACTION_PATTERNS:
        value = pattern.sub(repl, value)
    return value


# CLI argument patterns that contain secrets.
# Each pattern captures a prefix group (\1) and replaces the secret with REDACTED_VALUE.
# Build header name alternation from _SENSITIVE_HEADER_NAMES so they stay in sync.
_SENSITIVE_HEADER_ALT = "|".join(re.escape(h) for h in _SENSITIVE_HEADER_NAMES)
_CLI_SECRET_PATTERNS: Sequence[re.Pattern[str]] = (
    # --api-key <value> or --api-key=<value>
    re.compile(r"(--api-key[\s=])'?[^'\s]+'?"),
    # Single-quoted: --header 'Authorization:Bearer token' / -H 'X-API-Key:val'
    re.compile(rf"((?:--header|-H)\s+)'(?i:{_SENSITIVE_HEADER_ALT})[:\s][^']+'"),
    # Double-quoted: --header "Authorization:Bearer token"
    re.compile(rf'((?:--header|-H)\s+)"(?i:{_SENSITIVE_HEADER_ALT})[:\s][^"]+"'),
    # Unquoted with space-separated value: --header Authorization:Bearer token
    re.compile(rf"((?:--header|-H)\s+)(?i:{_SENSITIVE_HEADER_ALT})\S*\s+\S+"),
    # Unquoted single-token: --header X-API-Key:value
    re.compile(rf"((?:--header|-H)\s+)(?i:{_SENSITIVE_HEADER_ALT})\S+"),
)

# URL-typed CLI flags whose values may carry `user:password@` userinfo. Redaction
# rewrites the *value only* — preserves the flag, surrounding quotes, and any
# non-credential URL parts. Kept separate from _CLI_SECRET_PATTERNS because the
# replacement is structural (strip userinfo) rather than wholesale (<redacted>).
_URL_FLAG_ALT = "|".join(
    re.escape(f) for f in ("--url", "-u", "--otel-url", "--mlflow-tracking-uri")
)
# Match: (flag)(sep)(open quote?)(value)(close quote?)
# Group 1: flag + separator, group 2: optional open quote, group 3: raw value,
# group 4: optional close quote. The value is captured as non-greedy so the
# close quote and value stay aligned.
_URL_FLAG_PATTERN = re.compile(
    rf"((?:{_URL_FLAG_ALT})[\s=])(['\"])?([^'\"\s]+)(['\"])?"
)

# Safety net for stray scheme-prefixed URLs carrying userinfo — required because
# URL-typed flags like ``--url`` use ``consume_multiple=True`` (see
# ``endpoint_config.py``), so 2nd+ values are not captured by the per-flag
# matcher. The userinfo segment is bounded by /, ?, #, whitespace, quote, @ so
# a path ``@`` like ``http://host/users@example.com`` never matches, and
# non-URL args with ``@`` (e.g. ``--model foo@bar``) are untouched.
_STRAY_URL_USERINFO_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)[^\s'\"@/?#]+@")

# Safety net for stray scheme-less URLs carrying userinfo (``user:pass@host``).
# ``EndpointConfig.urls`` auto-prefixes scheme-less values, so users can write
# ``--url user1:pass1@host1 user2:pass2@host2`` and ``consume_multiple=True``
# means only the first value is caught by ``_URL_FLAG_PATTERN``. Applied only
# to tokens inside a ``--url``/``-u`` consumption window (see
# ``_redact_stray_bare_userinfo_after_url_flags``); a global sweep would eat
# legitimate ``key:value@...`` tokens on non-URL flags like ``--header
# X-User-Email:alice@example.com`` or ``--mlflow-tag owner:alice@acme.com``.
_STRAY_BARE_USERINFO_PATTERN = re.compile(r"(^|[\s'\"])([^\s:@'\"/?#]+:[^\s@'\"/?#]+)@")

# Flag tokens that open a ``consume_multiple=True`` URL-value window. Only
# these flags have the multi-value leak — ``--otel-url`` / ``--mlflow-tracking-uri``
# take a single value and are already covered by ``_URL_FLAG_PATTERN``.
_MULTI_VALUE_URL_FLAGS: frozenset[str] = frozenset({"--url", "-u"})


def _redact_url_flag_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    open_quote = match.group(2) or ""
    value = match.group(3)
    close_quote = match.group(4) or ""
    return f"{prefix}{open_quote}{redact_url(value)}{close_quote}"


def _redact_stray_bare_userinfo_after_url_flags(cmd: str) -> str:
    """Redact scheme-less ``user:pass@host`` tokens inside ``--url``/``-u`` windows.

    A consume-multiple URL flag captures every subsequent positional token up
    to the next ``--flag``/``-X`` or end of string. Only within that window do
    we treat ``key:value@rest`` as a credential — outside it, such tokens are
    legitimate (``--header X-User-Email:alice@example.com``,
    ``--mlflow-tag owner:alice@acme.com``).

    The first value of ``--url``/``-u`` is already redacted by
    ``_URL_FLAG_PATTERN``; this pass handles 2nd+ values only.
    """
    # Split on whitespace runs but keep them so we can rejoin losslessly.
    tokens = re.split(r"(\s+)", cmd)
    out: list[str] = []
    in_url_window = False
    for token in tokens:
        if not token:
            out.append(token)
            continue
        if token.isspace():
            out.append(token)
            continue
        if token in _MULTI_VALUE_URL_FLAGS:
            in_url_window = True
            out.append(token)
            continue
        # Any other flag closes the window (including ``--url=value`` which is
        # a single-value form). Conservative: treat any ``-`` or ``--`` prefix
        # as a new flag.
        if (
            token.startswith("-")
            and len(token) >= 2
            and (token.startswith("--") or token[1].isalpha())
        ):
            in_url_window = False
            out.append(token)
            continue
        if in_url_window:
            token = _STRAY_BARE_USERINFO_PATTERN.sub(rf"\1{REDACTED_VALUE}@", token)
        out.append(token)
    return "".join(out)


def redact_cli_command(cmd: str) -> str:
    """Redact secrets from a CLI command string.

    Redacts:
    - ``--api-key`` values.
    - Credentialed header values (``--header Authorization: Bearer …`` etc.).
    - Userinfo embedded in URL-typed flag values (``--url``, ``-u``,
      ``--otel-url``, ``--mlflow-tracking-uri``).
    - Stray scheme-prefixed URLs with userinfo that slipped past the URL-flag
      matcher — e.g. 2nd+ URL under ``--url u1 u2 u3`` (``consume_multiple=True``).
      Restricted to ``scheme://`` URLs, so non-URL args like ``--model foo@bar``
      pass through unchanged.
    - Stray scheme-less bare ``user:pass@host`` tokens inside a ``--url``/``-u``
      consumption window. Scoped to that window so benign ``key:value@...``
      tokens on non-URL flags (``--header X-User-Email:alice@example.com``,
      ``--mlflow-tag owner:alice@acme.com``) pass through untouched.
    """
    for pattern in _CLI_SECRET_PATTERNS:
        cmd = pattern.sub(rf"\1'{REDACTED_VALUE}'", cmd)
    cmd = _URL_FLAG_PATTERN.sub(_redact_url_flag_match, cmd)
    cmd = _STRAY_URL_USERINFO_PATTERN.sub(rf"\1{REDACTED_VALUE}@", cmd)
    cmd = _redact_stray_bare_userinfo_after_url_flags(cmd)
    return cmd


def redact_url(url: str) -> str:
    """Strip userinfo (user:password@) from a URL to prevent credential leakage.

    Handles URIs of any scheme (http, https, postgresql, mysql, sqlite, file,
    etc.) plus bare ``user:pass@host`` URLs. Returns the URL unchanged if no
    credentials are embedded.

    Matters for MLflow: ``--mlflow-tracking-uri`` commonly carries backends
    like ``postgresql://user:secret@db:5432/mlflow`` — limiting redaction to
    ``http(s)://`` would leak those credentials.
    """
    # Any scheme followed by ``://`` then userinfo up to ``@``. The userinfo
    # segment is bounded by /, ?, # so ``@`` inside a path or query string
    # (e.g. ``/users@example.com``, ``?email=a@b.com``) is never matched.
    result = re.sub(
        r"([A-Za-z][A-Za-z0-9+.\-]*://)([^@/?#]+)@",
        r"\1" + REDACTED_VALUE + "@",
        url,
    )
    # Once a scheme is present, do not fall through to the bare-userinfo regex:
    # that regex matches on ``^[^@:]+:[^@]+@`` which would eat ``https://host/?a@b``
    # starting from the ``https:`` prefix.
    if result != url or "://" in url:
        return result
    # Bare userinfo: user:pass@host (no scheme prefix, must contain : before @)
    return re.sub(r"^([^@:]+:[^@]+)@", REDACTED_VALUE + "@", url)
