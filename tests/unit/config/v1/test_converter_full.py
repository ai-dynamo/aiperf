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
    assert cfg.benchmark.endpoint.urls == ["http://localhost:8000"]
    assert len(cfg.benchmark.phases) == 1
    assert cfg.benchmark.phases[0].name == "profiling"
    assert str(cfg.benchmark.phases[0].type).lower().endswith("concurrency")
    assert cfg.benchmark.phases[0].concurrency == 100
    assert len(cfg.benchmark.datasets) == 1
    assert cfg.benchmark.datasets[0].name == "main"


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
    names = [p.name for p in cfg.benchmark.phases]
    assert names == ["warmup", "profiling"]
    assert cfg.benchmark.phases[0].exclude_from_results is True


def test_request_rate_run_picks_poisson_phase_type():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {"request_rate": 50.0, "request_count": 100},
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert str(cfg.benchmark.phases[0].type).lower().endswith("poisson")
    assert cfg.benchmark.phases[0].rate == 50.0


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
    ds = cfg.benchmark.datasets[0]
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
    assert cfg.benchmark.datasets[0].entries == 100


def test_convert_user_to_aiperf_preserves_gpu_telemetry_tokens():
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "gpu_telemetry": ["pynvml", "dashboard"],
        }
    )

    config = convert_user_to_aiperf(user, ServiceConfig())

    assert config.benchmark.gpu_telemetry.collector == GPUTelemetryCollectorType.PYNVML
    assert config.benchmark.gpu_telemetry.mode == GPUTelemetryMode.REALTIME_DASHBOARD


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

    assert config.benchmark.endpoint.ready_check_timeout == 30.0
    assert config.benchmark.endpoint.ready_check_interval == 2.5
    assert config.benchmark.endpoint.ready_check_mode == "both"
    assert config.benchmark.endpoint.download_video_content is True
    assert str(config.benchmark.endpoint.request_content_type) == "multipart/form-data"


# ============================================================
# --set-consistent-seed default-seed-42 contract
# ============================================================
#
# `--set-consistent-seed` (default True on the v1 LoadGeneratorConfig CLI)
# promises in its docstring: when no explicit `--random-seed` is given, the
# converter defaults the envelope `random_seed` to 42 so multi-trial runs
# produce identical workloads. Plan A Task 8 dropped this fallback when
# seed plumbing moved off `BenchmarkConfig.random_seed` onto
# `BenchmarkPlan.variation_seeds`; the regression manifests as all-None
# `variation_seeds` and silently non-comparable trials. The fix lives at
# the v1->v2 converter layer (NOT on AIPerfConfig) so programmatic users
# constructing AIPerfConfig directly with `set_consistent_seed=True` keep
# entropy-based semantics; only the v1 CLI input shape applies the default.


def test_set_consistent_seed_default_restores_seed_42():
    """When --set-consistent-seed is True (default) and no --random-seed is
    given, the envelope's random_seed defaults to 42 for reproducible trials.
    """
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {
                "concurrency": 1,
                "request_count": 10,
                "num_profile_runs": 3,
            },
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.random_seed == 42


def test_explicit_random_seed_wins_over_default():
    """When --random-seed is passed explicitly, it takes precedence over the
    --set-consistent-seed default."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "input": {"random_seed": 999},
            "loadgen": {
                "concurrency": 1,
                "request_count": 10,
                "num_profile_runs": 3,
            },
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.random_seed == 999


def test_set_consistent_seed_false_keeps_seed_none():
    """When --no-set-consistent-seed is passed, no default is applied; the
    envelope seed stays None (entropy-based per-run draws)."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {
                "concurrency": 1,
                "request_count": 10,
                "num_profile_runs": 3,
                "set_consistent_seed": False,
            },
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.random_seed is None


def test_set_consistent_seed_default_applies_even_for_single_run():
    """The default fires regardless of num_profile_runs; the docstring's
    'Only applies when --num-profile-runs > 1' is a guidance note, not a
    converter gate. Single-run jobs still get a deterministic seed when the
    user didn't opt out via --no-set-consistent-seed."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": {"concurrency": 1, "request_count": 10},
        }
    )
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.random_seed == 42
