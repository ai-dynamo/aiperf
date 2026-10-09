# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped environment defaults reach real child processes."""

import asyncio
import os
import sys

import pytest

from aiperf.common.environment import _Environment
from aiperf.common.scenario import get_scenario


@pytest.mark.integration
@pytest.mark.asyncio
async def test_defaults_inherit_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("AIPERF_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AIPERF_HTTP_TCP_USER_TIMEOUT", "450000")
    settings = _Environment()
    original_dataset = settings.DATASET
    scenario = get_scenario("inferencex-agentx-mvp").model_copy(
        update={
            "environment_defaults": {
                "DATASET": {"CONFIGURATION_TIMEOUT": 900},
                "SERVICE": {"PROFILE_CONFIGURE_TIMEOUT": 900},
                "HTTP": {"TCP_USER_TIMEOUT": 600000},
            },
        }
    )
    with (
        pytest.raises(RuntimeError, match="finish run"),
        settings.defaults(scenario.environment_defaults),
    ):
        assert settings.DATASET.CONFIGURATION_TIMEOUT == 900
        assert settings.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 900
        assert settings.HTTP.TCP_USER_TIMEOUT == 450000
        with settings.defaults({"HTTP": {"TCP_KEEPIDLE": 90}}):
            assert settings.HTTP.TCP_KEEPIDLE == 90
            assert settings.HTTP.TCP_USER_TIMEOUT == 450000
        assert settings.HTTP.TCP_KEEPIDLE == 60
        assert "AIPERF_HTTP_TCP_KEEPIDLE" not in os.environ
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from aiperf.common.environment import Environment as e; "
            "print(e.DATASET.CONFIGURATION_TIMEOUT, "
            "e.SERVICE.PROFILE_CONFIGURE_TIMEOUT, e.HTTP.TCP_USER_TIMEOUT)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
        assert child.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == "900.0 900.0 450000"
        raise RuntimeError("finish run")
    assert settings.DATASET is original_dataset
    assert settings.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 600
    assert "AIPERF_DATASET_CONFIGURATION_TIMEOUT" not in os.environ
    assert "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT" not in os.environ
    assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "450000"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_defaults_reject_noncanonical_keys_without_changing_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIPERF_HTTP_TCP_KEEPIDLE", "33")
    monkeypatch.delenv("AIPERF_HTTP_tcp_keepidle", raising=False)
    monkeypatch.delenv("AIPERF_HTTP_TCP_KEEPINTVL", raising=False)
    settings = _Environment()
    original_http = settings.HTTP

    with (
        pytest.raises(
            ValueError, match=r"Unknown environment setting: HTTP\.tcp_keepidle"
        ),
        settings.defaults({"HTTP": {"TCP_KEEPINTVL": 17, "tcp_keepidle": 90}}),
    ):
        pytest.fail("Noncanonical setting must be rejected before entering the context")

    assert settings.HTTP is original_http
    assert settings.HTTP.TCP_KEEPIDLE == 33
    assert "AIPERF_HTTP_tcp_keepidle" not in os.environ
    assert "AIPERF_HTTP_TCP_KEEPINTVL" not in os.environ
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "from aiperf.common.environment import Environment; "
        "print(Environment.HTTP.TCP_KEEPIDLE)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
    assert child.returncode == 0, stderr.decode()
    assert stdout.decode().strip() == "33"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("untargeted_group", [False, True])
async def test_defaults_normalize_parent_and_child_and_restore(
    monkeypatch: pytest.MonkeyPatch, untargeted_group: bool
) -> None:
    for key in tuple(os.environ):
        if key.startswith("AIPERF_"):
            monkeypatch.delenv(key)
    settings = _Environment()
    if untargeted_group:
        settings.DEV.SHOW_INTERNAL_METRICS = True
    original_dev = settings.DEV
    original_fields_set = settings.DEV.model_fields_set.copy()
    defaults = (
        {"HTTP": {"TCP_KEEPIDLE": 90}}
        if untargeted_group
        else {"DEV": {"SHOW_INTERNAL_METRICS": True}}
    )

    with (
        pytest.raises(RuntimeError, match="finish run"),
        settings.defaults(defaults),
    ):
        assert settings.DEV.SHOW_INTERNAL_METRICS is False
        assert settings.DEV.model_fields_set == original_fields_set | (
            set() if untargeted_group else {"SHOW_INTERNAL_METRICS"}
        )
        normalized_dev = settings.DEV
        with settings.defaults({"DEV": {"MODE": True}}):
            assert settings.DEV.MODE is True
            assert settings.DEV.SHOW_INTERNAL_METRICS is False
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "from aiperf.common.environment import Environment as e; "
                "print(e.DEV.MODE, e.DEV.SHOW_INTERNAL_METRICS)",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
            assert child.returncode == 0, stderr.decode()
            assert stdout.decode().strip() == "True False"
        assert settings.DEV is normalized_dev
        assert settings.DEV.MODE is False
        assert "AIPERF_DEV_MODE" not in os.environ
        raise RuntimeError("finish run")

    assert settings.DEV is original_dev
    assert settings.DEV.SHOW_INTERNAL_METRICS is untargeted_group
    assert settings.DEV.model_fields_set == original_fields_set
    assert "AIPERF_DEV_SHOW_INTERNAL_METRICS" not in os.environ
    assert "AIPERF_HTTP_TCP_KEEPIDLE" not in os.environ
