# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib


class TestDevMode:
    def test_dev_mode_on(self, monkeypatch, capsys):
        monkeypatch.setenv("AIPERF_DEV_MODE", "1")
        constants = importlib.reload(importlib.import_module("aiperf.common.constants"))

        assert constants.AIPERF_DEV_MODE is True

        monkeypatch.setattr("aiperf.common.config.dev_config.AIPERF_DEV_MODE", True)
        _ = importlib.reload(
            importlib.import_module("aiperf.common.config.service_config")
        )
        _ = importlib.reload(importlib.import_module("aiperf.common.config.dev_config"))
        _ = importlib.reload(importlib.import_module("aiperf.common.config"))
        cli = importlib.reload(importlib.import_module("aiperf.cli"))

        cli.app(["profile", "-h"])
        captured = capsys.readouterr()
        assert "Developer Mode is active" in captured.out

    def test_dev_mode_off(self, monkeypatch, capsys):
        monkeypatch.setenv("AIPERF_DEV_MODE", "0")
        constants = importlib.reload(importlib.import_module("aiperf.common.constants"))

        assert constants.AIPERF_DEV_MODE is False

        monkeypatch.setattr("aiperf.common.config.dev_config.AIPERF_DEV_MODE", False)
        _ = importlib.reload(
            importlib.import_module("aiperf.common.config.service_config")
        )
        _ = importlib.reload(importlib.import_module("aiperf.common.config.dev_config"))
        _ = importlib.reload(importlib.import_module("aiperf.common.config"))
        cli = importlib.reload(importlib.import_module("aiperf.cli"))

        cli.app(["-h"])
        captured = capsys.readouterr()
        assert "Developer Mode is active" not in captured.out
