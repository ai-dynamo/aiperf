# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.environment import Environment


def test_baseline_defaults() -> None:
    assert Environment.BASELINE.GATE_TIMEOUT_S == 5.0
    assert Environment.BASELINE.GATE_ENABLED is True


def test_baseline_env_override(monkeypatch) -> None:
    from aiperf.common.environment import _Environment

    monkeypatch.setenv("AIPERF_BASELINE_GATE_TIMEOUT_S", "1.25")
    monkeypatch.setenv("AIPERF_BASELINE_GATE_ENABLED", "0")
    env = _Environment()
    assert env.BASELINE.GATE_TIMEOUT_S == 1.25
    assert env.BASELINE.GATE_ENABLED is False
