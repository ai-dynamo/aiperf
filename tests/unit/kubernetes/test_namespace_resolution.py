# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark namespace is explicit or inherited -- never guessed."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiperf.config.loader.errors import ConfigurationError
from aiperf.kubernetes.cli_helpers import resolve_benchmark_namespace


def test_explicit_namespace_wins() -> None:
    with patch(
        "aiperf.kubernetes.cli_helpers._context_namespace", return_value="from-ctx"
    ):
        assert resolve_benchmark_namespace("explicit") == "explicit"


def test_falls_back_to_the_kubeconfig_context_namespace() -> None:
    with patch(
        "aiperf.kubernetes.cli_helpers._context_namespace", return_value="team-ns"
    ):
        assert resolve_benchmark_namespace(None) == "team-ns"


def test_raises_when_no_namespace_can_be_determined() -> None:
    """Never guess. A benchmark that lands somewhere the user did not name
    scatters pods and results into a namespace nobody is watching, and fails
    RBAC halfway through instead of immediately."""
    with (
        patch("aiperf.kubernetes.cli_helpers._context_namespace", return_value=None),
        pytest.raises(ConfigurationError) as exc,
    ):
        resolve_benchmark_namespace(None)

    message = str(exc.value)
    assert "--namespace" in message, message
    assert "set-context" in message, message


def test_never_silently_uses_default() -> None:
    """`default` is a guess like any other, and a particularly bad one."""
    with (
        patch("aiperf.kubernetes.cli_helpers._context_namespace", return_value=None),
        pytest.raises(ConfigurationError),
    ):
        resolve_benchmark_namespace(None)


def test_the_retired_constant_is_gone() -> None:
    import aiperf.kubernetes.constants as constants

    assert not hasattr(constants, "DEFAULT_BENCHMARK_NAMESPACE")


def test_context_namespace_survives_a_missing_kubeconfig() -> None:
    """No kubeconfig on disk is a normal state, not a crash.

    The resolver still has to raise its own message in that case, so the
    lookup must degrade to ``None`` rather than propagating whatever the
    kubernetes client raises for an unreadable config file.
    """
    from aiperf.kubernetes.cli_helpers import _context_namespace

    assert _context_namespace(kubeconfig="/nonexistent/kubeconfig") is None


def test_namespace_keeps_its_short_alias() -> None:
    """`-n` is the short flag for --namespace across every `aiperf kube` command.

    It is declared once on KubeManageOptions, so dropping or reordering that
    name list silently removes -n from all 18 kube subcommands at once.
    """
    from aiperf.config.kube import KubeManageOptions

    field = KubeManageOptions.model_fields["namespace"]
    names: list[str] = []
    for item in field.metadata:
        name = getattr(item, "name", None)
        if isinstance(name, list | tuple):
            names.extend(name)

    assert "-n" in names, names
    assert "--namespace" in names, names
