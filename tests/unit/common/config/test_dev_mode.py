# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.environment import _Environment


class TestDevMode:
    """Verify ``AIPERF_DEV_MODE`` env var maps to ``Environment.DEV.MODE``.

    These tests construct a FRESH ``_Environment()`` instance per case
    rather than reloading the module — module reload would replace the
    global ``Environment`` singleton, but already-imported callers (e.g.
    ``BranchOrchestrator`` reading ``Environment.DAG.FAIL_FAST`` at
    construction) still hold a reference to the OLD singleton, breaking
    every later test that ``monkeypatch.setattr``-s on ``Environment.*``
    in the same process. Direct instantiation avoids that pollution.
    """

    def test_dev_mode_on(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "1")
        env = _Environment()
        assert env.DEV.MODE is True

    def test_dev_mode_off(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "0")
        env = _Environment()
        assert env.DEV.MODE is False
