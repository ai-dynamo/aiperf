# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.environment import _Environment


class TestDevMode:
    """AIPERF_DEV_MODE drives Environment.DEV.MODE.

    Construct a fresh ``_Environment()`` (the same pattern as
    ``test_environment.py``) rather than ``importlib.reload``-ing the shared
    ``aiperf.common.environment`` module. Reloading re-executes the module body,
    rebinding the module-level ``Environment = _Environment()`` singleton to a
    NEW instance -- but every module that already did
    ``from aiperf.common.environment import Environment`` (weka_trace,
    agentic_replay, mmap_cache, ...) keeps its reference to the OLD instance.
    conftest/test ``monkeypatch.setattr(Environment.DATASET, ...)`` calls then
    mutate the NEW instance while those consumers read the OLD one, so settings
    silently fall back to defaults. Under xdist's dynamic ``--dist load`` that
    left any test landing on this worker AFTER this one reading a stale
    Environment -- a flaky, cross-subsystem failure (e.g. tool-shaping/aux
    settings not applying). A fresh local ``_Environment()`` reads the
    monkeypatched env var without touching the shared singleton.
    """

    def test_dev_mode_on(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "1")
        assert _Environment().DEV.MODE is True

    def test_dev_mode_off(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "0")
        assert _Environment().DEV.MODE is False
