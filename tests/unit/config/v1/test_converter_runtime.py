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


# ---------------------------------------------------------------------------
# Anti-pattern audit: verify converter only emits user-set fields so the
# v2 schema's Pydantic defaults are not silently overridden.
# ---------------------------------------------------------------------------


def test_artifacts_omits_unset_output_fields():
    """When user provides no OutputConfig, only cli_command flows through."""
    user = UserConfig()
    out = build_artifacts(user)
    assert "dir" not in out
    assert "trace" not in out
    assert "per_chunk_data" not in out
    assert "show_trace_timing" not in out
    assert "raw" not in out
    assert "records" not in out
    assert "prefix" not in out
    assert "cli_command" in out


def test_artifacts_omits_unset_fields_when_output_partially_set():
    """User sets only artifact_directory -> trace/raw/per-chunk omitted."""
    user = UserConfig.model_validate({"output": {"artifact_directory": "/tmp/a"}})
    out = build_artifacts(user)
    assert "dir" in out
    assert "trace" not in out
    assert "per_chunk_data" not in out
    assert "show_trace_timing" not in out
    assert "raw" not in out


def test_artifacts_emits_user_set_trace():
    user = UserConfig.model_validate({"output": {"export_http_trace": True}})
    out = build_artifacts(user)
    assert out["trace"] is True


def test_artifacts_emits_user_set_per_chunk_data():
    user = UserConfig.model_validate({"output": {"export_per_chunk_data": True}})
    out = build_artifacts(user)
    assert out["per_chunk_data"] is True


def test_artifacts_emits_user_set_show_trace_timing():
    user = UserConfig.model_validate({"output": {"show_trace_timing": True}})
    out = build_artifacts(user)
    assert out["show_trace_timing"] is True


def test_artifacts_emits_user_set_export_level_raw():
    user = UserConfig.model_validate({"output": {"export_level": "raw"}})
    out = build_artifacts(user)
    assert out["raw"] is True
    assert "records" in out


def test_logging_runtime_omits_log_level_when_unset():
    """When user did not pass --log-level, logging dict is empty."""
    user = UserConfig()
    service = ServiceConfig()
    log, _runtime = build_logging_runtime(user, service)
    assert "level" not in log


def test_logging_runtime_emits_log_level_when_user_set():
    user = UserConfig()
    service = ServiceConfig.model_validate({"log_level": "DEBUG"})
    log, _runtime = build_logging_runtime(user, service)
    assert log["level"] in ("DEBUG", "debug")


def test_logging_runtime_omits_ui_when_unset_in_tty(monkeypatch):
    """User-unset ui_type -> not propagated as a default override.

    (TTY-detection branch may still set ui=NONE when not a TTY; force is_tty
    True so the natural-default branch holds.)
    """
    monkeypatch.setattr("aiperf.common.utils.is_tty", lambda: True)
    user = UserConfig()
    service = ServiceConfig()
    _log, runtime = build_logging_runtime(user, service)
    assert "ui" not in runtime


def test_logging_runtime_emits_ui_when_user_set():
    user = UserConfig()
    service = ServiceConfig.model_validate({"ui_type": "simple"})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["ui"] in ("simple", "SIMPLE")


def test_logging_runtime_omits_record_processors_when_unset():
    user = UserConfig()
    service = ServiceConfig()
    _log, runtime = build_logging_runtime(user, service)
    assert "record_processors" not in runtime


def test_logging_runtime_emits_record_processors_when_user_set():
    user = UserConfig()
    service = ServiceConfig.model_validate({"record_processor_service_count": 4})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["record_processors"] == 4


def test_api_port_emitted_to_runtime_dict_when_user_set():
    user = UserConfig()
    service = ServiceConfig.model_validate({"api_port": 19090})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["api_port"] == 19090


def test_api_host_emitted_when_user_set_with_port():
    user = UserConfig()
    service = ServiceConfig.model_validate({"api_port": 19090, "api_host": "0.0.0.0"})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["api_host"] == "0.0.0.0"
    assert runtime["api_port"] == 19090


def test_api_port_omitted_when_not_user_set():
    user = UserConfig()
    service = ServiceConfig()
    _log, runtime = build_logging_runtime(user, service)
    assert "api_port" not in runtime
    assert "api_host" not in runtime


def test_api_host_without_port_still_raises():
    """The pre-existing validator check must continue to fire."""
    user = UserConfig()
    service = ServiceConfig.model_validate({"api_host": "0.0.0.0"})
    with pytest.raises(ValueError, match="api_host requires"):
        build_logging_runtime(user, service)
