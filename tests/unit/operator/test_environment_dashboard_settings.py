# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DASHBOARD nested settings on _OperatorEnvironment."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("env", "expected_port", "expected_enabled"),
    [
        ({}, 0, False),
        ({"AIPERF_DASHBOARD_PORT": "8082"}, 8082, False),
        (
            {"AIPERF_DASHBOARD_PORT": "8082", "AIPERF_DASHBOARD_PROXY_ENABLED": "1"},
            8082,
            True,
        ),
        (
            {"AIPERF_DASHBOARD_PROXY_ENABLED": "true"},
            0,
            True,
        ),
    ],
)
def test_dashboard_settings_load_from_env(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected_port: int,
    expected_enabled: bool,
) -> None:
    for k in ("AIPERF_DASHBOARD_PORT", "AIPERF_DASHBOARD_PROXY_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from aiperf.operator import environment as mod

    importlib.reload(mod)
    assert expected_port == mod.OperatorEnvironment.DASHBOARD.PORT
    assert mod.OperatorEnvironment.DASHBOARD.PROXY_ENABLED is expected_enabled
