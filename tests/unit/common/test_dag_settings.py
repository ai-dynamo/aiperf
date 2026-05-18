# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


class TestDagSettings:
    """Test suite for the _DagSettings environment configuration."""

    def test_default_fail_fast_false(self, monkeypatch):
        monkeypatch.delenv("AIPERF_DAG_FAIL_FAST", raising=False)
        from aiperf.common.environment import _DagSettings

        assert _DagSettings().FAIL_FAST is False

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("0", False),
            ("false", False),
        ],
    )
    def test_env_override(self, monkeypatch, raw: str, expected: bool):
        monkeypatch.setenv("AIPERF_DAG_FAIL_FAST", raw)
        from aiperf.common.environment import _DagSettings

        assert _DagSettings().FAIL_FAST is expected

    def test_aggregator_exposes_dag(self, monkeypatch):
        monkeypatch.delenv("AIPERF_DAG_FAIL_FAST", raising=False)
        from aiperf.common.environment import _DagSettings, _Environment

        env = _Environment()
        assert isinstance(env.DAG, _DagSettings)
        assert env.DAG.FAIL_FAST is False
