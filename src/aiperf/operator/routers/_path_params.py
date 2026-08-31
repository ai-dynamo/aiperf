# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict validation for ``{namespace}`` / ``{name}`` path parameters.

Results-serving routes join user-supplied path segments into filesystem
paths under the operator's results PVC (``<base>/<namespace>/<name>/...``).
Starlette percent-decodes path parameters AFTER route matching, so an
encoded traversal like ``..%2F..`` matches a single-segment route and then
reaches the handler as ``../..`` — joined unvalidated, it escapes the
results base directory. Every router that touches the PVC must run these
validators BEFORE any path join.

The allowlists mirror what Kubernetes itself accepts, so no legitimate
object is rejected:

- Namespaces are RFC 1123 DNS labels (lowercase alphanumeric or ``-``,
  max 63 chars) — dots are not legal in namespace names.
- AIPerfJob / AIPerfSweep names are RFC 1123 DNS subdomains (dot-separated
  DNS labels, max 253 chars). ``.`` and ``..`` can never match because
  every dot-separated label must start and end with an alphanumeric.
"""

from __future__ import annotations

from fastapi import HTTPException

from aiperf.kubernetes.constants import (
    DNS_LABEL_MAX as _DNS_LABEL_MAX,
)
from aiperf.kubernetes.constants import (
    DNS_LABEL_RE,
    DNS_SUBDOMAIN_RE,
)
from aiperf.kubernetes.constants import (
    DNS_SUBDOMAIN_MAX as _DNS_SUBDOMAIN_MAX,
)

__all__ = [
    "DNS_LABEL_RE",
    "DNS_SUBDOMAIN_RE",
    "validate_name_param",
    "validate_namespace_param",
    "validate_results_path_params",
]


def validate_namespace_param(namespace: str) -> str:
    """Reject a ``{namespace}`` path parameter that is not a DNS-1123 label.

    Example:
        >>> validate_namespace_param("bench-prod")
        'bench-prod'

    Raises:
        HTTPException: 400 when ``namespace`` cannot be a Kubernetes
            namespace name (e.g. a decoded ``../..`` traversal segment).
    """
    if len(namespace) <= _DNS_LABEL_MAX and DNS_LABEL_RE.match(namespace):
        return namespace
    raise HTTPException(
        400,
        f"Invalid namespace {namespace!r}: Kubernetes namespaces are "
        "DNS-1123 labels (lowercase alphanumeric or '-', max 63 chars)",
    )


def validate_name_param(name: str, *, param: str = "name") -> str:
    """Reject a job/sweep ``{name}`` parameter that is not a DNS-1123 subdomain.

    Example:
        >>> validate_name_param("llama-3.1-8b-load", param="job_id")
        'llama-3.1-8b-load'

    Raises:
        HTTPException: 400 when ``name`` cannot be a Kubernetes object name
            (e.g. a decoded ``../..`` traversal segment).
    """
    if len(name) <= _DNS_SUBDOMAIN_MAX and DNS_SUBDOMAIN_RE.match(name):
        return name
    raise HTTPException(
        400,
        f"Invalid {param} {name!r}: Kubernetes object names are DNS-1123 "
        "subdomains (lowercase alphanumeric, '-' or '.', max 253 chars)",
    )


def validate_results_path_params(namespace: str, name: str) -> None:
    """Validate a ``{namespace}``/``{name}`` pair before any PVC path join.

    Example:
        >>> validate_results_path_params("bench-prod", "llama-3-8b-load")

    Raises:
        HTTPException: 400 when either segment is not a valid Kubernetes name.
    """
    validate_namespace_param(namespace)
    validate_name_param(name)
