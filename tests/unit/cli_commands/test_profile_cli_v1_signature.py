# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signature test verifying profile CLI takes v1 UserConfig+ServiceConfig."""

import inspect
from io import StringIO

import pytest
from rich.console import Console

from aiperf.cli_commands.profile import profile
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig
from aiperf.plugin.enums import EndpointType


def test_profile_cli_takes_user_and_service_config() -> None:
    sig = inspect.signature(profile)
    annots = {p.name: p.annotation for p in sig.parameters.values()}
    assert "user_config" in annots, (
        f"profile() must take user_config, got: {list(annots)}"
    )
    assert "service_config" in annots, (
        f"profile() must take service_config, got: {list(annots)}"
    )


def test_profile_cli_validation_error_exits_without_traceback(monkeypatch) -> None:
    from aiperf import cli_utils

    output = StringIO()
    monkeypatch.setattr(
        cli_utils,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=120),
    )
    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["mock-model"],
            urls=["http://localhost:8000"],
            type=EndpointType.CHAT,
        ),
        loadgen=LoadGeneratorConfig(request_count=1, concurrency=0),
    )

    with pytest.raises(SystemExit) as exc_info:
        profile(user_config=user_config, service_config=ServiceConfig())

    text = output.getvalue()
    assert exc_info.value.code == 1
    assert "Error Running AIPerf System" in text
    assert "concurrency" in text
    assert "Traceback" not in text
