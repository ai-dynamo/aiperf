# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.control_hooks import ResetKvCacheConfig, ServerProfilerConfig
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig


@pytest.mark.parametrize(
    "raw,expect_none",
    [
        param(False, True, id="false"),
        param(None, True, id="omitted-as-none-via-default"),
        param(True, False, id="true-defaults"),
        param({"timeout_seconds": 10, "path": "/reset_prefix_cache"}, False, id="object"),
    ],
)  # fmt: skip
def test_reset_kv_cache_accepts_bool_or_object(raw, expect_none: bool) -> None:
    data: dict = {
        "urls": ["http://127.0.0.1:8000"],
        "type": "chat",
    }
    if raw is not None:
        data["reset_kv_cache"] = raw
    cfg = EndpointConfig.model_validate(data)
    if expect_none:
        assert cfg.reset_kv_cache is None
    else:
        assert cfg.reset_kv_cache is not None


@pytest.mark.parametrize(
    "value",
    [
        param(float("inf"), id="inf"),
        param(float("-inf"), id="neg-inf"),
        param(float("nan"), id="nan"),
    ],
)  # fmt: skip
def test_reset_kv_cache_timeout_seconds_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        ResetKvCacheConfig(timeout_seconds=value)


@pytest.mark.parametrize(
    "value",
    [
        param(float("inf"), id="inf"),
        param(float("-inf"), id="neg-inf"),
        param(float("nan"), id="nan"),
    ],
)  # fmt: skip
def test_reset_kv_cache_max_retry_seconds_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        ResetKvCacheConfig(max_retry_seconds=value)


@pytest.mark.parametrize(
    "value",
    [
        param(float("inf"), id="inf"),
        param(float("-inf"), id="neg-inf"),
        param(float("nan"), id="nan"),
    ],
)  # fmt: skip
def test_server_profiler_timeout_seconds_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        ServerProfilerConfig(timeout_seconds=value)


def test_server_profiler_object_parses_overrides() -> None:
    cfg = EndpointConfig.model_validate(
        {
            "urls": ["http://127.0.0.1:8000"],
            "server_profiler": {
                "timeout_seconds": 10,
                "start_path": "/start_profile",
                "stop_path": "/stop_profile",
            },
        }
    )
    assert cfg.server_profiler is not None
    assert cfg.server_profiler.start_path == "/start_profile"
    assert cfg.server_profiler.stop_path == "/stop_profile"


def test_control_hook_paths_must_be_relative() -> None:
    with pytest.raises(ValidationError, match="relative"):
        EndpointConfig.model_validate(
            {
                "urls": ["http://127.0.0.1:8000"],
                "reset_kv_cache": {
                    "path": "http://bad.example/reset_prefix_cache",
                },
            }
        )


def test_cli_reset_kv_cache_flag_lowers_into_endpoint_dict() -> None:
    cli = CLIConfig(
        model_names=["test"],
        urls=["http://127.0.0.1:8000"],
        reset_kv_cache=True,
        reset_kv_cache_timeout_seconds=10.0,
        reset_kv_cache_path="/reset_prefix_cache",
    )
    endpoint = build_endpoint(cli)
    assert endpoint["reset_kv_cache"] == {
        "timeout_seconds": 10.0,
        "path": "/reset_prefix_cache",
    }


def test_cli_server_profiler_flag_lowers_into_endpoint_dict() -> None:
    cli = CLIConfig(
        model_names=["test"],
        urls=["http://127.0.0.1:8000"],
        server_profiler=True,
        server_profiler_timeout_seconds=12.0,
        server_profiler_start_path="/start_profile",
        server_profiler_stop_path="/stop_profile",
    )
    endpoint = build_endpoint(cli)
    assert endpoint["server_profiler"] == {
        "timeout_seconds": 12.0,
        "start_path": "/start_profile",
        "stop_path": "/stop_profile",
    }


def test_control_hook_path_without_leading_slash_rejected() -> None:
    with pytest.raises(ValidationError, match="relative path starting with '/'"):
        EndpointConfig.model_validate(
            {
                "urls": ["http://127.0.0.1:8000"],
                "reset_kv_cache": {"path": "reset_prefix_cache"},
            }
        )


def test_control_hooks_require_http_transport_gate() -> None:
    cfg = EndpointConfig.model_construct(
        urls=["http://127.0.0.1:8000"],
        reset_kv_cache=ResetKvCacheConfig(),
        server_profiler=None,
        transport="grpc",
    )
    with pytest.raises(ValueError, match="HTTP transport"):
        cfg._validate_control_hooks_require_http()
