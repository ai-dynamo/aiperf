# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signature pinning for the kube CLI v1 cutover.

These tests don't exercise behavior; they pin the parameter names that the
cyclopts dispatcher relies on after the ``CLIModel -> UserConfig +
ServiceConfig`` migration. If a future refactor renames or drops one of these
parameters without updating the dispatcher, the tests fail loudly here rather
than at first-use of ``aiperf kube ...``.
"""

from __future__ import annotations

import inspect

from aiperf.cli_commands.kube._kube_common import resolve_config
from aiperf.cli_commands.kube.generate import generate
from aiperf.cli_commands.kube.profile import profile
from aiperf.cli_commands.kube.sweep import sweep


def test_kube_profile_takes_user_config() -> None:
    sig = inspect.signature(profile)
    names = {p.name for p in sig.parameters.values()}
    assert "user_config" in names
    assert "service_config" in names
    assert "cli_model" not in names


def test_kube_generate_takes_user_config() -> None:
    sig = inspect.signature(generate)
    names = {p.name for p in sig.parameters.values()}
    assert "user_config" in names
    assert "service_config" in names
    assert "cli_model" not in names


def test_kube_sweep_takes_user_config() -> None:
    sig = inspect.signature(sweep)
    names = {p.name for p in sig.parameters.values()}
    assert "user_config" in names
    assert "service_config" in names
    assert "cli_model" not in names


def test_resolve_config_takes_user_and_service() -> None:
    sig = inspect.signature(resolve_config)
    names = {p.name for p in sig.parameters.values()}
    assert "user_config" in names
    assert "service_config" in names
    assert "config_file" in names
    assert "cli_model" not in names
