# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError


class TestChatSettings:
    """Test suite for the _ChatSettings environment configuration
    (the AIPERF_CHAT_* timeouts used by `aiperf chat`)."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AIPERF_CHAT_CONNECT_TIMEOUT", raising=False)
        monkeypatch.delenv("AIPERF_CHAT_READ_TIMEOUT", raising=False)
        from aiperf.common.environment import _ChatSettings

        settings = _ChatSettings()
        assert settings.CONNECT_TIMEOUT == 10.0
        assert settings.READ_TIMEOUT == 300.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("AIPERF_CHAT_CONNECT_TIMEOUT", "3")
        monkeypatch.setenv("AIPERF_CHAT_READ_TIMEOUT", "42.5")
        from aiperf.common.environment import _ChatSettings

        settings = _ChatSettings()
        assert settings.CONNECT_TIMEOUT == 3.0
        assert settings.READ_TIMEOUT == 42.5

    @pytest.mark.parametrize("var", ["CONNECT_TIMEOUT", "READ_TIMEOUT"])
    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_non_positive_rejected(self, monkeypatch, var: str, bad: str):
        monkeypatch.setenv(f"AIPERF_CHAT_{var}", bad)
        from aiperf.common.environment import _ChatSettings

        with pytest.raises(ValidationError):
            _ChatSettings()

    def test_aggregator_exposes_chat(self, monkeypatch):
        monkeypatch.delenv("AIPERF_CHAT_CONNECT_TIMEOUT", raising=False)
        monkeypatch.delenv("AIPERF_CHAT_READ_TIMEOUT", raising=False)
        from aiperf.common.environment import _ChatSettings, _Environment

        env = _Environment()
        assert isinstance(env.CHAT, _ChatSettings)
        assert env.CHAT.CONNECT_TIMEOUT == 10.0
