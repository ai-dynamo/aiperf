# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""URL origin comparison used to decide whether credentials may follow a request."""

from __future__ import annotations

from urllib.parse import urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}

# Only these survive a hop to an origin the benchmarked server chose. An
# allowlist rather than a denylist because nothing marks "X-Acme-Token" as a
# secret -- a name-based denylist can only catch the names it thought of. The
# natural content_url is a presigned S3 URL, which authenticates through its own
# query signature and needs no inherited headers at all.
_FOREIGN_ORIGIN_HEADER_ALLOWLIST = frozenset({"user-agent"})


def strip_credentials_for_foreign_origin(
    headers: dict[str, str], *, extra_allowed: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Keep only allowlisted headers, matching names case-insensitively.

    HTTP header names are case-insensitive but these live in a plain dict, so a
    lowercase ``authorization`` would otherwise slip past a literal comparison.
    ``extra_allowed`` adds lowercase, credential-free header names for callers
    whose requests need them to survive the hop (e.g. ``range``).
    """
    allowed = _FOREIGN_ORIGIN_HEADER_ALLOWLIST | extra_allowed
    return {name: value for name, value in headers.items() if name.lower() in allowed}


def origin_of(url: str) -> tuple[str, str | None, int | None] | None:
    """Scheme, host and normalized port, or None when the URL will not parse.

    ``urlsplit`` accepts a malformed port and only raises when ``.port`` is
    read, so the read happens here inside the guard; leaving it to the caller
    lets ``ValueError`` escape the fail-closed contract.

    An omitted port is normalized to the scheme default, so ``https://host``
    and ``https://host:443`` compare equal. The scheme stays part of the tuple,
    so ``http://host:443`` is still a different origin from ``https://host``.
    """
    try:
        parts = urlsplit(url)
        port = parts.port or DEFAULT_PORTS.get(parts.scheme)
        return parts.scheme, parts.hostname, port
    except ValueError:
        return None


def same_origin(url: str, reference: str) -> bool:
    """Whether ``url`` targets the same scheme, host, and port as ``reference``.

    Used to decide whether a URL the benchmarked server handed us may be signed
    with the endpoint's AWS credentials, so it fails closed: a URL that does not
    parse counts as foreign rather than raising into the caller.
    """
    origin = origin_of(url)
    # Explicit None check: two unparseable URLs must not compare equal and be
    # read as "same origin".
    return origin is not None and origin == origin_of(reference)
