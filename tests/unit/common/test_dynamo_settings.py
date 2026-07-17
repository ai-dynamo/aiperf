# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from aiperf.common.environment import _DynamoSettings


def test_dynamo_settings_defaults() -> None:
    s = _DynamoSettings()
    assert s.MAX_SUBAGENT_DEPTH == 16


def test_dynamo_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH", "4")
    s = _DynamoSettings()
    assert s.MAX_SUBAGENT_DEPTH == 4
