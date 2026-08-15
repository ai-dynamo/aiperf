# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_DynamoSettings defaults and AIPERF_DYNAMO_ env-var overrides."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.environment import _DynamoSettings


@pytest.mark.parametrize(
    ("env_value", "expected_depth"),
    [
        param(None, 16, id="default"),
        param("4", 4, id="env_override"),
    ],
)  # fmt: skip
def test_max_subagent_depth(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected_depth: int
) -> None:
    """MAX_SUBAGENT_DEPTH defaults to 16 and is overridable via the env prefix."""
    if env_value is None:
        monkeypatch.delenv("AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH", raising=False)
    else:
        monkeypatch.setenv("AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH", env_value)
    assert expected_depth == _DynamoSettings().MAX_SUBAGENT_DEPTH
