# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX runtime settings reach spawned processes and stay scoped to the run."""

import asyncio
import os
import sys

import pytest

from aiperf.common import environment
from aiperf.common.scenario import get_scenario


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agentx_runtime_defaults_inherit_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("AIPERF_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AIPERF_HTTP_TCP_USER_TIMEOUT", "450000")
    settings = environment._Environment()
    monkeypatch.setattr(environment, "Environment", settings)
    original_dataset = settings.DATASET
    original_service = settings.SERVICE

    with (
        pytest.raises(RuntimeError, match="finish run"),
        settings.defaults(get_scenario("agentx").environment_defaults),
    ):
        assert settings.DATASET.CONFIGURATION_TIMEOUT == 1800
        assert settings.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 1800
        assert settings.UI.REALTIME_METRICS_ENABLED is True
        assert settings.HTTP.TCP_USER_TIMEOUT == 450000
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from aiperf.common.environment import Environment as e; "
            "print(e.DATASET.CONFIGURATION_TIMEOUT, "
            "e.SERVICE.PROFILE_CONFIGURE_TIMEOUT, "
            "e.UI.REALTIME_METRICS_ENABLED, e.HTTP.TCP_USER_TIMEOUT)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
        assert child.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == "1800.0 1800.0 True 450000"
        raise RuntimeError("finish run")
    assert settings.DATASET is original_dataset
    assert settings.SERVICE is original_service
    assert "AIPERF_DATASET_CONFIGURATION_TIMEOUT" not in os.environ
    assert "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT" not in os.environ
    assert "AIPERF_UI_REALTIME_METRICS_ENABLED" not in os.environ
    assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "450000"
    with settings.defaults(get_scenario("inferencex-agentx-mvp").environment_defaults):
        assert settings.DATASET.CONFIGURATION_TIMEOUT == 300
        assert settings.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 600
