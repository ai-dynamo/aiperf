# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube._app`.

Focuses on:
- The cyclopts ``App`` exposes the expected name and help string.
- All declared subcommands are registered (and only those).
- Each subcommand's lazy-load target points at an importable module exposing ``app``.
"""

from __future__ import annotations

import importlib

import pytest
from cyclopts import App
from pytest import param

from aiperf.cli_commands.kube._app import app as kube_app

# Subcommands the kube App is expected to expose. Order is the order the file
# registers them — we don't assert ordering, just membership.
EXPECTED_SUBCOMMANDS: tuple[str, ...] = (
    "init",
    "validate",
    "profile",
    "sweep",
    "generate",
    "cancel",
    "delete",
    "cleanup",
    "shutdown",
    "attach",
    "list",
    "logs",
    "results",
    "show",
    "debug",
    "preflight",
    "dashboard",
)


# ============================================================
# App identity
# ============================================================


class TestKubeAppIdentity:
    """Verify the top-level App's name/help fields."""

    def test_app_is_cyclopts_app_instance(self) -> None:
        assert isinstance(kube_app, App)

    def test_app_name_is_kube(self) -> None:
        # `App.name` is a tuple of names in cyclopts; "kube" must be the
        # primary command label users reach via `aiperf kube ...`.
        assert "kube" in tuple(kube_app.name)

    def test_app_help_is_descriptive(self) -> None:
        assert kube_app.help
        assert "Kubernetes" in kube_app.help


# ============================================================
# Command registration
# ============================================================


class TestKubeAppRegistration:
    """Verify every expected subcommand is registered exactly once."""

    @pytest.mark.parametrize(
        "subcommand",
        [param(name, id=name) for name in EXPECTED_SUBCOMMANDS],
    )  # fmt: skip
    def test_each_expected_subcommand_is_registered(self, subcommand: str) -> None:
        assert subcommand in kube_app, (
            f"expected `aiperf kube {subcommand}` to be registered"
        )

    def test_all_expected_subcommands_present(self) -> None:
        registered = set(kube_app)
        # Strip cyclopts auto-injected flags.
        registered_commands = {name for name in registered if not name.startswith("-")}
        assert set(EXPECTED_SUBCOMMANDS) <= registered_commands

    def test_no_unexpected_subcommands(self) -> None:
        """Surface drift: catch silently-added subcommands so this test must be updated."""
        registered = {name for name in kube_app if not name.startswith("-")}
        unexpected = registered - set(EXPECTED_SUBCOMMANDS)
        assert not unexpected, (
            f"new subcommands registered without test update: {sorted(unexpected)}"
        )

    def test_each_subcommand_is_app_instance(self) -> None:
        for name in EXPECTED_SUBCOMMANDS:
            cmd = kube_app[name]
            assert cmd is not None, f"`{name}` resolved to None"


# ============================================================
# Lazy-load targets
# ============================================================


class TestLazyLoadTargets:
    """Verify each registered subcommand resolves to an importable module with `app`."""

    @pytest.mark.parametrize(
        "module_name",
        [
            param("aiperf.cli_commands.kube.init", id="init"),
            param("aiperf.cli_commands.kube.validate", id="validate"),
            param("aiperf.cli_commands.kube.profile", id="profile"),
            param("aiperf.cli_commands.kube.sweep", id="sweep"),
            param("aiperf.cli_commands.kube.generate", id="generate"),
            param("aiperf.cli_commands.kube.attach", id="attach"),
            param("aiperf.cli_commands.kube.list_", id="list_"),
            param("aiperf.cli_commands.kube.logs", id="logs"),
            param("aiperf.cli_commands.kube.results", id="results"),
            param("aiperf.cli_commands.kube.show", id="show"),
            param("aiperf.cli_commands.kube.debug", id="debug"),
            param("aiperf.cli_commands.kube.preflight", id="preflight"),
            param("aiperf.cli_commands.kube.dashboard", id="dashboard"),
        ],
    )  # fmt: skip
    def test_module_imports_and_exposes_app(self, module_name: str) -> None:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "app"), f"{module_name} must define an `app` attribute"
        assert isinstance(mod.app, App), f"{module_name}.app must be a cyclopts App"
