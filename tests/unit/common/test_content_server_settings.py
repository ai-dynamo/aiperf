# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public environment compatibility for the native content server."""

import pytest
from pydantic import ValidationError

from aiperf.common.environment import _ContentServerSettings, _Environment


def test_content_server_defaults_match_content_server_branch() -> None:
    settings = _ContentServerSettings()

    assert settings.ENABLED is False
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8090
    assert settings.CONTENT_DIR == ""
    assert settings.MAX_TRACKED_RECORDS == 10_000


def test_content_server_environment_values_reach_fresh_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIPERF_CONTENT_SERVER_ENABLED", "true")
    monkeypatch.setenv("AIPERF_CONTENT_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("AIPERF_CONTENT_SERVER_PORT", "9090")
    monkeypatch.setenv("AIPERF_CONTENT_SERVER_CONTENT_DIR", "/tmp/content")
    monkeypatch.setenv("AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS", "1234")

    settings = _Environment().CONTENT_SERVER

    assert settings.ENABLED is True
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 9090
    assert settings.CONTENT_DIR == "/tmp/content"
    assert settings.MAX_TRACKED_RECORDS == 1234


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AIPERF_CONTENT_SERVER_HOST", ""),
        ("AIPERF_CONTENT_SERVER_PORT", "0"),
        ("AIPERF_CONTENT_SERVER_PORT", "65536"),
        ("AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS", "99"),
        ("AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS", "1000001"),
    ],
)
def test_content_server_environment_bounds_fail_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        _ContentServerSettings()
