# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 converter runtime helpers (artifacts + logging/runtime).

Covers `build_artifacts` and `build_logging_runtime`, which port the
old `_cli_sections.py` helpers onto the v1 UserConfig + ServiceConfig pair
and additionally fold in the four ServiceConfig validators that v1 strips.
"""

import pytest

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._converter_runtime import (
    build_artifacts,
    build_logging_runtime,
)


def test_artifacts_carries_dir_and_slice_duration():
    user = UserConfig.model_validate(
        {"output": {"artifact_directory": "/tmp/a", "slice_duration": 10.0}}
    )
    out = build_artifacts(user)
    assert out["dir"] == "/tmp/a" or str(out["dir"]) == "/tmp/a"
    assert out["slice_duration"] == 10.0


def test_artifacts_includes_cli_command():
    user = UserConfig()
    out = build_artifacts(user)
    assert "cli_command" in out  # synthesized from sys.argv


def test_logging_runtime_verbose_promotes_to_debug():
    user = UserConfig()
    service = ServiceConfig.model_validate({"verbose": True})
    log, _runtime = build_logging_runtime(user, service)
    assert log["level"] in ("DEBUG", "debug")


def test_logging_runtime_extra_verbose_promotes_to_trace():
    user = UserConfig()
    service = ServiceConfig.model_validate({"extra_verbose": True})
    log, _runtime = build_logging_runtime(user, service)
    assert log["level"] in ("TRACE", "trace")


def test_logging_runtime_zmq_tcp_picks_tcp_communication_type():
    user = UserConfig()
    service = ServiceConfig.model_validate({"zmq_tcp": {"host": "127.0.0.1"}})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["communication"]["type"] == "tcp"


def test_logging_runtime_zmq_ipc_picks_ipc_communication_type():
    user = UserConfig()
    service = ServiceConfig.model_validate({"zmq_ipc": {}})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["communication"]["type"] == "ipc"


def test_logging_runtime_api_host_without_port_raises():
    user = UserConfig()
    service = ServiceConfig.model_validate({"api_host": "0.0.0.0"})
    with pytest.raises(ValueError, match="api_host requires"):
        build_logging_runtime(user, service)
