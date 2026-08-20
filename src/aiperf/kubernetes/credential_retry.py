# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared detection and user feedback for expired Kubernetes credentials."""

from __future__ import annotations

import sys

from kubernetes_asyncio.client.exceptions import ApiException
from rich.markup import escape

_AUTH_ERROR_FRAGMENTS = (
    "cannot refresh without refresh-token",
    "no valid id-token",
    "not logged in",
    "logged out",
    "server has asked for the client to provide credentials",
    "you must be logged in to the server",
)
_EXEC_AUTH_ERROR_FRAGMENTS = (
    "exec: process returned",
    "getting credentials: exec:",
)
_CONFIG_ERROR_EXCLUSIONS = (
    "executable not found",
    "invalid apiversion",
    "malformed",
    "no such file",
)


def interactive_credential_wait_enabled() -> bool:
    """Return whether a person can recover credentials from this terminal."""
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def is_api_authentication_error(error: BaseException) -> bool:
    """Return whether an API failure means credentials were not accepted."""
    return isinstance(error, ApiException) and error.status == 401


def is_kubeconfig_authentication_error(error: BaseException) -> bool:
    """Classify refreshable OIDC and exec-provider authentication failures."""
    detail = str(error).lower()
    if any(fragment in detail for fragment in _CONFIG_ERROR_EXCLUSIONS):
        return False
    return any(fragment in detail for fragment in _AUTH_ERROR_FRAGMENTS) or any(
        fragment in detail for fragment in _EXEC_AUTH_ERROR_FRAGMENTS
    )


def is_kubectl_authentication_error(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> bool:
    """Return whether a failed kubectl invocation lost authentication."""
    if returncode == 0:
        return False
    detail = f"{stdout}\n{stderr}".lower()
    if any(fragment in detail for fragment in _CONFIG_ERROR_EXCLUSIONS):
        return False
    return any(fragment in detail for fragment in _AUTH_ERROR_FRAGMENTS) or (
        any(fragment in detail for fragment in _EXEC_AUTH_ERROR_FRAGMENTS)
        and "exit" in detail
    )


def credential_retry_delay(attempt: int) -> float:
    """Return capped exponential delay for a zero-based retry attempt."""
    return min(2.0 * (2**attempt), 15.0)


def print_credential_wait(context: str | None) -> None:
    """Tell an interactive user how to unblock credential recovery."""
    from aiperf.kubernetes import console as kube_console

    target = escape(context) if context else "the current Kubernetes context"
    kube_console.print_warning(
        f"Kubernetes credentials for {target} are not valid. Complete your normal "
        "login in another terminal; AIPerf will keep retrying. Press Ctrl-C to stop."
    )


def print_credentials_restored(context: str | None) -> None:
    """Tell an interactive user that Kubernetes requests resumed."""
    from aiperf.kubernetes import console as kube_console

    target = escape(context) if context else "the current Kubernetes context"
    kube_console.print_success(f"Kubernetes credentials restored for {target}")
