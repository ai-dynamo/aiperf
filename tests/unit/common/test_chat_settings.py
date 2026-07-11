# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param


class TestChatSettings:
    """Test suite for the _ChatSettings environment configuration
    (the AIPERF_CHAT_* timeouts used by `aiperf chat`)."""

    def test_chat_settings_env_unset_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIPERF_CHAT_CONNECT_TIMEOUT", raising=False)
        monkeypatch.delenv("AIPERF_CHAT_READ_TIMEOUT", raising=False)
        from aiperf.common.environment import _ChatSettings

        settings = _ChatSettings()
        assert settings.CONNECT_TIMEOUT == 10.0
        assert settings.READ_TIMEOUT == 300.0

    def test_chat_settings_env_set_overrides_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_CHAT_CONNECT_TIMEOUT", "3")
        monkeypatch.setenv("AIPERF_CHAT_READ_TIMEOUT", "42.5")
        from aiperf.common.environment import _ChatSettings

        settings = _ChatSettings()
        assert settings.CONNECT_TIMEOUT == 3.0
        assert settings.READ_TIMEOUT == 42.5

    @pytest.mark.parametrize(
        "var,bad",
        [
            param("CONNECT_TIMEOUT", "0", id="connect_zero"),
            param("CONNECT_TIMEOUT", "-1", id="connect_negative"),
            param("READ_TIMEOUT", "0", id="read_zero"),
            param("READ_TIMEOUT", "-1", id="read_negative"),
        ],
    )  # fmt: skip
    def test_chat_settings_non_positive_timeout_raises(
        self, monkeypatch: pytest.MonkeyPatch, var: str, bad: str
    ) -> None:
        monkeypatch.setenv(f"AIPERF_CHAT_{var}", bad)
        from aiperf.common.environment import _ChatSettings

        with pytest.raises(ValidationError):
            _ChatSettings()

    def test_environment_aggregator_exposes_chat_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIPERF_CHAT_CONNECT_TIMEOUT", raising=False)
        monkeypatch.delenv("AIPERF_CHAT_READ_TIMEOUT", raising=False)
        from aiperf.common.environment import _ChatSettings, _Environment

        env = _Environment()
        assert isinstance(env.CHAT, _ChatSettings)
        assert env.CHAT.CONNECT_TIMEOUT == 10.0
