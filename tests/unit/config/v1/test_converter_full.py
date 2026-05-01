# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the v1 ``convert_user_to_aiperf`` entrypoint.

Exercises the full composition: each test builds a ``UserConfig`` and a
``ServiceConfig`` from CLI-shaped dicts, runs them through the converter,
and asserts on the resulting validated ``AIPerfConfig``.
"""

from aiperf.common.enums import GPUTelemetryMode
from aiperf.config.config import AIPerfConfig
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
from aiperf.plugin.enums import GPUTelemetryCollectorType


def test_minimal_concurrency_run_produces_valid_aiperf_config():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["llama"], "urls": ["http://localhost:8000"]},
            "loadgen": {"concurrency": 100, "request_count": 1000},
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert isinstance(cfg, AIPerfConfig)
    assert cfg.endpoint.urls == ["http://localhost:8000"]
    assert len(cfg.phases) == 1
    assert cfg.phases[0].name == "profiling"
    assert str(cfg.phases[0].type).lower().endswith("concurrency")
    assert cfg.phases[0].concurrency == 100
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].name == "main"


def test_warmup_phase_added_when_warmup_set():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {
                "concurrency": 10,
                "request_count": 100,
                "warmup_concurrency": 2,
                "warmup_request_count": 10,
            },
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    names = [p.name for p in cfg.phases]
    assert names == ["warmup", "profiling"]
    assert cfg.phases[0].exclude_from_results is True


def test_request_rate_run_picks_poisson_phase_type():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {"request_rate": 50.0, "request_count": 100},
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert str(cfg.phases[0].type).lower().endswith("poisson")
    assert cfg.phases[0].rate == 50.0


def test_public_dataset_uses_dataset_field():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {"concurrency": 1, "request_count": 1},
            "input": {"public_dataset": "sharegpt"},
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    ds = cfg.datasets[0]
    assert ds.name == "main"
    assert ds.dataset == "sharegpt"


def test_full_conversion_request_rate_only_uses_synthetic_default():
    """Top-level converter must produce a valid AIPerfConfig for the minimal
    request-rate invocation. Mirrors the Phase 6 case that previously needed
    the request_count: 1000 workaround."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {"request_rate": 50.0, "benchmark_duration": 30.0},
        }
    )
    cfg = convert_user_to_aiperf(user, ServiceConfig())
    # SyntheticDataset Pydantic default takes effect (no user-set count source)
    assert cfg.datasets[0].entries == 100


def test_convert_user_to_aiperf_preserves_gpu_telemetry_tokens():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "gpu_telemetry": ["pynvml", "dashboard"],
        }
    )

    config = convert_user_to_aiperf(user, ServiceConfig())

    assert config.gpu_telemetry.collector == GPUTelemetryCollectorType.PYNVML
    assert config.gpu_telemetry.mode == GPUTelemetryMode.REALTIME_DASHBOARD


def test_convert_user_to_aiperf_preserves_endpoint_parity_fields():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["video-model"],
                "urls": ["http://server:8000"],
                "type": "video_generation",
                "ready_check_timeout": 30.0,
                "ready_check_interval": 2.5,
                "ready_check_mode": "both",
                "download_video_content": True,
                "request_content_type": "multipart/form-data",
            },
        }
    )
    service = ServiceConfig()

    config = convert_user_to_aiperf(user, service)

    assert config.endpoint.ready_check_timeout == 30.0
    assert config.endpoint.ready_check_interval == 2.5
    assert config.endpoint.ready_check_mode == "both"
    assert config.endpoint.download_video_content is True
    assert str(config.endpoint.request_content_type) == "multipart/form-data"
