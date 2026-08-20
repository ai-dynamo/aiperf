# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for explicit CLI overlays on Config-v2 YAML."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError
from aiperf.plugin.enums import PhaseType

_BASE_YAML = textwrap.dedent("""\
randomSeed: 91
benchmark:
  models:
    strategy: round_robin
    items:
      - name: yaml-model
  endpoint:
    urls: [http://localhost:8000]
    headers: {X-Base: yaml}
    extra: {temperature: 0.2}
    resetKvCache: {path: /yaml-reset}
  datasets:
    - name: workload
      type: synthetic
      entries: 77
      prompts:
        isl: {mean: 128, stddev: 7}
        osl: {mean: 32}
        batchSize: 4
  phases:
    - name: measured
      kind: profiling
      type: poisson
      requests: 20
      rate: 2
      rateRamp: {duration: 9}
      cancellation: {rate: 3, delay: 4}
  slos: {request_latency: 500}
  networkLatency: {enabled: false, pingInterval: 8}
  otel:
    metricsUrl: http://collector:4318/v1/metrics
    streamMetricsEnabled: true
    streamTimingEnabled: true
  mlflow:
    trackingUri: http://mlflow:5000
    experiment: yaml-experiment
""")


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(_BASE_YAML, encoding="utf-8")
    return path


def test_omitted_cli_defaults_preserve_yaml(config_file: Path) -> None:
    config = resolve_config(CLIConfig(), config_file)

    phase = config.benchmark.phases[0]
    dataset = config.benchmark.datasets[0]
    assert config.random_seed == 91
    assert config.benchmark.models.strategy == "round_robin"
    assert config.benchmark.endpoint.headers == {"X-Base": "yaml"}
    assert dataset.entries == 77
    assert dataset.prompts.isl.mean == 128
    assert dataset.prompts.batch_size == 4
    assert phase.rate == 2
    assert phase.rate_ramp.duration == 9


@pytest.mark.parametrize(
    "cli,assertion",
    [
        param(CLIConfig(random_seed=0), lambda c: c.random_seed == 0, id="random-seed-zero"),
        param(
            CLIConfig(model_selection_strategy="random"),
            lambda c: c.benchmark.models.strategy == "random",
            id="model-strategy-without-model-names",
        ),
        param(
            CLIConfig(headers=[]),
            lambda c: c.benchmark.endpoint.headers == {},
            id="empty-headers-clear-yaml",
        ),
        param(
            CLIConfig(extra_inputs=[]),
            lambda c: c.benchmark.endpoint.extra == {},
            id="empty-extra-clears-yaml",
        ),
        param(
            CLIConfig(reset_kv_cache=False),
            lambda c: c.benchmark.endpoint.reset_kv_cache is None,
            id="explicit-false-disables-reset",
        ),
        param(
            CLIConfig(goodput={"request_latency": 125}),
            lambda c: c.benchmark.slos == {"request_latency": 125.0},
            id="goodput-slos",
        ),
        param(
            CLIConfig(network_latency_mean=12.5),
            lambda c: c.benchmark.network_latency.enabled
            and c.benchmark.network_latency.mean_ms == 12.5,
            id="network-latency",
        ),
        param(
            CLIConfig(stream=["metrics"]),
            lambda c: c.benchmark.otel.metrics_url
            == "http://collector:4318/v1/metrics"
            and c.benchmark.otel.stream_metrics_enabled
            and not c.benchmark.otel.stream_timing_enabled,
            id="otel-secondary-uses-yaml-primary",
        ),
        param(
            CLIConfig(mlflow_run_name="cli-run"),
            lambda c: c.benchmark.mlflow.tracking_uri == "http://mlflow:5000"
            and c.benchmark.mlflow.run_name == "cli-run",
            id="mlflow-secondary-uses-yaml-primary",
        ),
    ],
)  # fmt: skip
def test_explicit_cli_value_overrides_yaml(
    config_file: Path, cli: CLIConfig, assertion: Callable[[Any], bool]
) -> None:
    config = resolve_config(cli, config_file)

    assert assertion(config)


def test_dataset_modifiers_merge_without_cli_defaults(config_file: Path) -> None:
    config = resolve_config(
        CLIConfig(prompt_input_tokens_mean=256, prompt_output_tokens_mean=64),
        config_file,
    )

    dataset = config.benchmark.datasets[0]
    assert dataset.name == "workload"
    assert dataset.entries == 77
    assert dataset.prompts.isl.mean == 256
    assert dataset.prompts.isl.stddev == 7
    assert dataset.prompts.osl.mean == 64
    assert dataset.prompts.batch_size == 4


def test_dataset_source_flag_replaces_sole_yaml_dataset(
    config_file: Path, tmp_path: Path
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text":"hello"}\n', encoding="utf-8")

    config = resolve_config(CLIConfig(input_file=str(input_path)), config_file)

    dataset = config.benchmark.datasets[0]
    assert dataset.name == "workload"
    assert dataset.type == "file"
    assert dataset.path == input_path


def test_phase_shape_overrides_target_unique_profiling_phase(
    config_file: Path,
) -> None:
    config = resolve_config(
        CLIConfig(
            request_rate=7,
            arrival_pattern="gamma",
            arrival_smoothness=0.8,
            request_rate_ramp_duration=3,
            request_cancellation_delay=1.5,
        ),
        config_file,
    )

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.GAMMA
    assert phase.rate == 7
    assert phase.smoothness == 0.8
    assert phase.rate_ramp.duration == 3
    assert phase.cancellation.rate == 3
    assert phase.cancellation.delay == 1.5


def test_phase_type_switch_removes_incompatible_camel_case_yaml(
    config_file: Path,
) -> None:
    config = resolve_config(CLIConfig(fixed_schedule=True), config_file)

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.FIXED_SCHEDULE
    assert not hasattr(phase, "rate")
    assert not hasattr(phase, "rate_ramp")


def test_phase_override_rejects_multiple_profiling_targets(
    config_file: Path,
) -> None:
    content = _BASE_YAML.replace(
        "  slos: {request_latency: 500}",
        "    - name: measured_two\n"
        "      kind: profiling\n"
        "      type: concurrency\n"
        "      requests: 2\n"
        "      concurrency: 1\n"
        "  slos: {request_latency: 500}",
    )
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="2 profiling phases"):
        resolve_config(CLIConfig(request_count=5), config_file)


def test_dataset_and_phase_overrides_update_raw_jinja_envelope(
    config_file: Path,
) -> None:
    config = resolve_config(
        CLIConfig(prompt_input_tokens_mean=256, request_count=5), config_file
    )

    assert config._raw_envelope is not None
    raw_benchmark = config._raw_envelope["benchmark"]
    assert raw_benchmark["datasets"][0]["prompts"]["isl"]["mean"] == 256
    assert raw_benchmark["phases"][0]["requests"] == 5


# =============================================================================
# Control hooks: a bare boolean flag enables without wiping YAML sub-fields
# =============================================================================


def test_bare_reset_kv_cache_flag_preserves_yaml_sub_fields(
    config_file: Path,
) -> None:
    """A bare ``--reset-kv-cache`` must enable the hook, not reset its path."""
    content = _BASE_YAML.replace(
        "    resetKvCache: {path: /yaml-reset}",
        "    resetKvCache: {path: /my_custom_reset, timeoutSeconds: 90}",
    )
    config_file.write_text(content, encoding="utf-8")

    config = resolve_config(CLIConfig(reset_kv_cache=True), config_file)

    reset = config.benchmark.endpoint.reset_kv_cache
    assert reset is not None
    assert reset.path == "/my_custom_reset"
    assert reset.timeout_seconds == 90


def test_bare_reset_kv_cache_flag_enables_without_yaml_section(
    config_file: Path,
) -> None:
    content = _BASE_YAML.replace("    resetKvCache: {path: /yaml-reset}\n", "")
    config_file.write_text(content, encoding="utf-8")

    config = resolve_config(CLIConfig(reset_kv_cache=True), config_file)

    assert config.benchmark.endpoint.reset_kv_cache is not None


def test_reset_kv_cache_sub_field_flag_still_overrides_yaml(
    config_file: Path,
) -> None:
    content = _BASE_YAML.replace(
        "    resetKvCache: {path: /yaml-reset}",
        "    resetKvCache: {path: /my_custom_reset, timeoutSeconds: 90}",
    )
    config_file.write_text(content, encoding="utf-8")

    config = resolve_config(
        CLIConfig(reset_kv_cache=True, reset_kv_cache_path="/cli-reset"), config_file
    )

    reset = config.benchmark.endpoint.reset_kv_cache
    assert reset is not None
    assert reset.path == "/cli-reset"
    assert reset.timeout_seconds == 90


def test_bare_server_profiler_flag_preserves_yaml_sub_fields(
    config_file: Path,
) -> None:
    content = _BASE_YAML.replace(
        "    resetKvCache: {path: /yaml-reset}",
        "    serverProfiler: {startPath: /prof_on, stopPath: /prof_off, "
        "timeoutSeconds: 45}",
    )
    config_file.write_text(content, encoding="utf-8")

    config = resolve_config(CLIConfig(server_profiler=True), config_file)

    profiler = config.benchmark.endpoint.server_profiler
    assert profiler is not None
    assert profiler.start_path == "/prof_on"
    assert profiler.stop_path == "/prof_off"
    assert profiler.timeout_seconds == 45


def test_explicitly_empty_headers_still_clear_yaml(config_file: Path) -> None:
    """The empty-dict-replaces rule stays intact for collection flags."""
    config = resolve_config(CLIConfig(headers=[]), config_file)

    assert config.benchmark.endpoint.headers == {}


# =============================================================================
# user_centric phase preservation
# =============================================================================

_USER_CENTRIC_YAML = _BASE_YAML.replace(
    "      type: poisson\n      requests: 20\n      rate: 2\n",
    "      type: user_centric\n      duration: 120\n      rate: 1.0\n      users: 30\n",
)


@pytest.fixture()
def user_centric_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "user_centric.yaml"
    path.write_text(_USER_CENTRIC_YAML, encoding="utf-8")
    return path


def test_request_rate_preserves_yaml_user_centric_phase(
    user_centric_config_file: Path,
) -> None:
    config = resolve_config(CLIConfig(request_rate=50.0), user_centric_config_file)

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.USER_CENTRIC
    assert phase.rate == 50.0
    assert phase.users == 30
    assert phase.duration == 120


def test_arrival_pattern_preserves_yaml_user_centric_phase(
    user_centric_config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        config = resolve_config(
            CLIConfig(arrival_pattern="gamma"), user_centric_config_file
        )

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.USER_CENTRIC
    assert phase.users == 30
    assert phase.rate == 1.0
    assert "--arrival-pattern is ignored" in caplog.text


def test_arrival_smoothness_on_user_centric_phase_is_rejected(
    user_centric_config_file: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="user_centric"):
        resolve_config(CLIConfig(arrival_smoothness=0.5), user_centric_config_file)


def test_request_rate_on_non_user_centric_yaml_phase_is_unchanged(
    config_file: Path,
) -> None:
    """Only an existing user_centric phase is protected."""
    config = resolve_config(CLIConfig(request_rate=50.0), config_file)

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.POISSON
    assert phase.rate == 50.0


def test_request_rate_on_concurrency_yaml_phase_becomes_poisson(
    config_file: Path,
) -> None:
    content = _BASE_YAML.replace(
        "      type: poisson\n      requests: 20\n      rate: 2\n"
        "      rateRamp: {duration: 9}\n",
        "      type: concurrency\n      requests: 20\n      concurrency: 4\n",
    )
    config_file.write_text(content, encoding="utf-8")

    config = resolve_config(CLIConfig(request_rate=50.0), config_file)

    phase = config.benchmark.phases[0]
    assert phase.type == PhaseType.POISSON
    assert phase.rate == 50.0


# =============================================================================
# M4 pin: --num-conversations writes the `sessions` stop condition
# =============================================================================


def test_num_conversations_writes_sessions_stop_condition(config_file: Path) -> None:
    """Intended CLI parity: --num-conversations also bounds the phase."""
    config = resolve_config(CLIConfig(conversation_num=9), config_file)

    assert config.benchmark.phases[0].sessions == 9
    assert config.benchmark.datasets[0].entries == 9


def test_yaml_with_deprecated_exclude_from_results_loads_and_warns(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pre-branch templates carrying the retired key still load."""
    content = _BASE_YAML.replace(
        "      requests: 20\n",
        "      requests: 20\n      excludeFromResults: false\n",
    )
    config_file.write_text(content, encoding="utf-8")

    with caplog.at_level("WARNING"):
        config = resolve_config(CLIConfig(), config_file)

    assert config.benchmark.phases[0].kind == "profiling"
    assert "'excludeFromResults' is deprecated and ignored" in caplog.text


def test_pure_cli_request_rate_still_builds_open_loop_phase() -> None:
    """No YAML phase to protect: --request-rate keeps its open-loop meaning."""
    from aiperf.config.flags.converter import convert_cli_to_aiperf

    config = convert_cli_to_aiperf(
        CLIConfig(
            url="http://localhost:8000",
            model_names=["m"],
            request_rate=50.0,
            request_count=10,
        )
    )

    phase = config.benchmark.phases[-1]
    assert phase.type == PhaseType.POISSON
    assert phase.rate == 50.0
    assert not hasattr(phase, "users")


def test_pure_cli_arrival_pattern_still_builds_open_loop_phase() -> None:
    from aiperf.config.flags.converter import convert_cli_to_aiperf

    config = convert_cli_to_aiperf(
        CLIConfig(
            url="http://localhost:8000",
            model_names=["m"],
            request_rate=50.0,
            arrival_pattern="gamma",
            request_count=10,
        )
    )

    phase = config.benchmark.phases[-1]
    assert phase.type == PhaseType.GAMMA
    assert phase.rate == 50.0


def test_pure_cli_user_centric_rate_still_builds_user_centric_phase() -> None:
    from aiperf.config.flags.converter import convert_cli_to_aiperf

    config = convert_cli_to_aiperf(
        CLIConfig(
            url="http://localhost:8000",
            model_names=["m"],
            user_centric_rate=5.0,
            num_users=3,
            conversation_turn_mean=2,
            request_count=10,
        )
    )

    phase = config.benchmark.phases[-1]
    assert phase.type == PhaseType.USER_CENTRIC
    assert phase.users == 3
    assert phase.rate == 5.0
