# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 build_endpoint and build_models converters."""

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_endpoint import build_endpoint, build_models


def test_build_endpoint_basic():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["llama"],
                "urls": ["http://localhost:8000"],
            },
        }
    )
    out = build_endpoint(user)
    assert out["urls"] == ["http://localhost:8000"]
    # http:// prefix added if missing — verify by inputting bare host:port.
    user2 = UserConfig.model_validate(
        {"endpoint": {"model_names": ["m"], "urls": ["localhost:8000"]}}
    )
    out2 = build_endpoint(user2)
    assert out2["urls"] == ["http://localhost:8000"]


def test_build_endpoint_preserves_explicit_unsupported_scheme():
    user = UserConfig.model_validate(
        {"endpoint": {"model_names": ["m"], "urls": ["ftp://localhost:8000"]}}
    )
    out = build_endpoint(user)
    assert out["urls"] == ["ftp://localhost:8000"]


def test_build_endpoint_passes_extras_when_set():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["m"],
                "urls": ["http://x"],
                "type": "chat",
                "streaming": True,
            },
        }
    )
    out = build_endpoint(user)
    assert out["type"] == "chat"
    assert out["streaming"] is True


def test_build_endpoint_skips_unset_fields():
    user = UserConfig.model_validate(
        {"endpoint": {"model_names": ["m"], "urls": ["http://x"]}}
    )
    out = build_endpoint(user)
    # type defaults but wasn't user-set, should not appear in dict.
    assert "type" not in out


def test_build_endpoint_includes_headers_and_extras_from_input():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "input": {
                "headers": ["X-Trace:abc"],
                "extra": ["temperature:0.7"],
            },
        }
    )
    out = build_endpoint(user)
    assert out["headers"] == {"X-Trace": "abc"}
    assert out["extra"] == {"temperature": 0.7}


def test_build_models_passes_names_and_strategy():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["a", "b"],
                "urls": ["http://x"],
                "model_selection_strategy": "random",
            },
        }
    )
    out = build_models(user)
    assert out["items"] == [{"name": "a"}, {"name": "b"}]
    assert out["strategy"] == "random"


def test_build_models_no_strategy_when_unset():
    user = UserConfig.model_validate(
        {"endpoint": {"model_names": ["a"], "urls": ["http://x"]}}
    )
    out = build_models(user)
    assert out["items"] == [{"name": "a"}]
    assert "strategy" not in out


def test_build_endpoint_maps_ready_check_interval_and_mode():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["m"],
                "urls": ["http://x"],
                "ready_check_timeout": 30.0,
                "ready_check_interval": 2.5,
                "ready_check_mode": "both",
            },
        }
    )

    out = build_endpoint(user)

    assert out["ready_check_timeout"] == 30.0
    assert out["ready_check_interval"] == 2.5
    assert out["ready_check_mode"] == "both"


def test_build_endpoint_maps_video_request_options():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["m"],
                "urls": ["http://x"],
                "download_video_content": True,
                "request_content_type": "multipart/form-data",
                "type": "video_generation",
            },
        }
    )

    out = build_endpoint(user)

    assert out["download_video_content"] is True
    assert str(out["request_content_type"]) == "multipart/form-data"
