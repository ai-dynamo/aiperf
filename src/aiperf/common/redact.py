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


def redact_cli_command(cmd: str) -> str:
    """Redact secrets from a CLI command string (--api-key, sensitive --header values)."""
    for pattern in _CLI_SECRET_PATTERNS:
        cmd = pattern.sub(rf"\1'{REDACTED_VALUE}'", cmd)
    return cmd
