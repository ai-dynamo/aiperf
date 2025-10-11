# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib


class TestDevMode:
    def test_dev_mode_on(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "1")
        env_module = importlib.reload(
            importlib.import_module("aiperf.common.environment")
        )
        Environment = env_module.Environment

        assert Environment.DEV_MODE is True

    def test_dev_mode_off(self, monkeypatch):
        monkeypatch.setenv("AIPERF_DEV_MODE", "0")
        env_module = importlib.reload(
            importlib.import_module("aiperf.common.environment")
        )
        Environment = env_module.Environment

        assert Environment.DEV_MODE is False
