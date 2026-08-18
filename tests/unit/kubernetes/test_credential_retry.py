# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kubernetes credential-loss classification and retry timing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.credential_retry import (
    credential_retry_delay,
    interactive_credential_wait_enabled,
    is_api_authentication_error,
    is_kubeconfig_authentication_error,
    is_kubectl_authentication_error,
)


@pytest.mark.parametrize(
    "status,expected",
    [
        param(401, True, id="unauthorized"),
        param(403, False, id="forbidden"),
        param(404, False, id="not-found"),
    ],
)  # fmt: skip
def test_api_authentication_error_only_accepts_401(
    status: int,
    expected: bool,
) -> None:
    assert is_api_authentication_error(ApiException(status=status)) is expected


@pytest.mark.parametrize(
    "detail,expected",
    [
        param("oidc: No valid id-token, and cannot refresh without refresh-token", True, id="oidc"),
        param("exec: process returned 1. user is logged out", True, id="exec-logged-out"),
        param("exec: executable not found", False, id="missing-exec"),
        param("invalid apiVersion in exec credential plugin", False, id="invalid-plugin"),
        param("certificate signed by unknown authority", False, id="tls"),
    ],
)  # fmt: skip
def test_kubeconfig_authentication_error_is_narrow(
    detail: str,
    expected: bool,
) -> None:
    assert is_kubeconfig_authentication_error(RuntimeError(detail)) is expected


@pytest.mark.parametrize(
    "stderr,expected",
    [
        param("error: You must be logged in to the server", True, id="logged-out"),
        param("the server has asked for the client to provide credentials", True, id="credentials"),
        param("Error from server (Forbidden): pods is forbidden", False, id="rbac"),
        param("Unable to connect to the server: connection refused", False, id="network"),
        param("getting credentials: exec: executable kubelogin not found", False, id="missing-provider"),
    ],
)  # fmt: skip
def test_kubectl_authentication_error_is_narrow(
    stderr: str,
    expected: bool,
) -> None:
    assert (
        is_kubectl_authentication_error(
            returncode=1,
            stdout="",
            stderr=stderr,
        )
        is expected
    )


def test_interactive_wait_requires_input_and_output_ttys() -> None:
    stdin = MagicMock()
    stdout = MagicMock()
    stdin.isatty.return_value = True
    stdout.isatty.return_value = False
    with (
        patch("aiperf.kubernetes.credential_retry.sys.stdin", stdin),
        patch("aiperf.kubernetes.credential_retry.sys.stdout", stdout),
    ):
        assert interactive_credential_wait_enabled() is False

    stdout.isatty.return_value = True
    with (
        patch("aiperf.kubernetes.credential_retry.sys.stdin", stdin),
        patch("aiperf.kubernetes.credential_retry.sys.stdout", stdout),
    ):
        assert interactive_credential_wait_enabled() is True


def test_credential_retry_delay_caps_at_fifteen_seconds() -> None:
    assert [credential_retry_delay(attempt) for attempt in range(6)] == [
        2.0,
        4.0,
        8.0,
        15.0,
        15.0,
        15.0,
    ]
