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
