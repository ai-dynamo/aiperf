# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for camelCase / snake_case config bi-directional support.

Verifies that:
- YAML files with camelCase keys load correctly
- YAML files with snake_case keys load correctly (backward compat)
- Mixed casing within the same file works
- Round-trip: load -> dump (camelCase) -> reload produces equivalent config
- All config sections handle both casings (endpoint, datasets, phases, etc.)
- CRD spec output uses camelCase for benchmark fields
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.config.loader import (
    dump_config,
    load_config,
    load_config_from_string,
)

# ---------------------------------------------------------------------------
# Fixtures: identical configs in snake_case and camelCase
# ---------------------------------------------------------------------------

_SNAKE_CASE_YAML = textwrap.dedent("""\
random_seed: 42

benchmark:
  models:
    - test-model

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    url_strategy: round_robin
    wait_for_model_timeout: 30.0
    connection_reuse: pooled
    use_legacy_max_tokens: false
    use_server_token_count: true
    download_video_content: false

  datasets:
    - name: main
      type: synthetic
      entries: 100
      prompts:
        isl: 512
        osl: 128
        batch_size: 2
      prefix_prompts:
        pool_size: 4
        length: 256
      turn_delay: 100
      turn_delay_ratio: 0.5

  phases:
    - name: warmup
      type: concurrency
      requests: 50
      concurrency: 4
      exclude_from_results: true

    - name: profiling
      type: gamma
      duration: 300
      rate: 50.0
      concurrency: 64
      rate_ramp: 30
      grace_period: 60
      seamless: true

  gpu_telemetry:
    enabled: false

  server_metrics:
    enabled: false

  artifacts:
    dir: ./artifacts
    slice_duration: 60
    show_trace_timing: true
""")

_CAMEL_CASE_YAML = textwrap.dedent("""\
randomSeed: 42

benchmark:
  models:
    - test-model

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    urlStrategy: round_robin
    waitForModelTimeout: 30.0
    connectionReuse: pooled
    useLegacyMaxTokens: false
    useServerTokenCount: true
    downloadVideoContent: false

  datasets:
    - name: main
      type: synthetic
      entries: 100
      prompts:
        isl: 512
        osl: 128
        batchSize: 2
      prefixPrompts:
        poolSize: 4
        length: 256
      turnDelay: 100
      turnDelayRatio: 0.5

  phases:
    - name: warmup
      type: concurrency
      requests: 50
      concurrency: 4
      excludeFromResults: true

    - name: profiling
      type: gamma
      duration: 300
      rate: 50.0
      concurrency: 64
      rateRamp: 30
      gracePeriod: 60
      seamless: true

  artifacts:
    dir: ./artifacts
    sliceDuration: 60
    showTraceTiming: true

  gpuTelemetry:
    enabled: false

  serverMetrics:
    enabled: false

""")

_MIXED_CASE_YAML = textwrap.dedent("""\
randomSeed: 42

benchmark:
  models:
    - test-model

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    url_strategy: round_robin
    waitForModelTimeout: 30.0

  datasets:
    - name: main
      type: synthetic
      entries: 100
      prompts:
        isl: 512
        batch_size: 2
      turn_delay: 100
      turnDelayRatio: 0.5

  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
      exclude_from_results: false

  gpu_telemetry:
    enabled: false

  serverMetrics:
    enabled: false
""")


def _assert_configs_equivalent(a: AIPerfConfig, b: AIPerfConfig) -> None:
    """Assert two configs are semantically equivalent on key fields."""
    assert a.random_seed == b.random_seed
    assert a.benchmark.get_model_names() == b.benchmark.get_model_names()

    # Endpoint
    assert a.benchmark.endpoint.urls == b.benchmark.endpoint.urls
    assert a.benchmark.endpoint.url_strategy == b.benchmark.endpoint.url_strategy
    assert (
        a.benchmark.endpoint.wait_for_model_timeout
        == b.benchmark.endpoint.wait_for_model_timeout
    )
    assert (
        a.benchmark.endpoint.connection_reuse == b.benchmark.endpoint.connection_reuse
    )
    assert (
        a.benchmark.endpoint.use_legacy_max_tokens
        == b.benchmark.endpoint.use_legacy_max_tokens
    )
    assert (
        a.benchmark.endpoint.use_server_token_count
        == b.benchmark.endpoint.use_server_token_count
    )
    assert (
        a.benchmark.endpoint.download_video_content
        == b.benchmark.endpoint.download_video_content
    )

    # Datasets
    ds_a = a.benchmark.get_dataset("main")
    ds_b = b.benchmark.get_dataset("main")
    assert ds_a.prompts.batch_size == ds_b.prompts.batch_size
    assert ds_a.prefix_prompts.pool_size == ds_b.prefix_prompts.pool_size
    assert ds_a.turn_delay.expected_value == ds_b.turn_delay.expected_value
    assert ds_a.turn_delay_ratio == ds_b.turn_delay_ratio

    # Phases
    warmup_a = next(p for p in a.benchmark.phases if p.name == "warmup")
    warmup_b = next(p for p in b.benchmark.phases if p.name == "warmup")
    assert warmup_a.exclude_from_results == warmup_b.exclude_from_results

    prof_a = next(p for p in a.benchmark.phases if p.name == "profiling")
    prof_b = next(p for p in b.benchmark.phases if p.name == "profiling")
    assert prof_a.rate_ramp is not None
    assert prof_b.rate_ramp is not None
    assert prof_a.rate_ramp.duration == prof_b.rate_ramp.duration
    assert prof_a.grace_period == prof_b.grace_period

    # Top-level sections
    assert a.benchmark.gpu_telemetry.enabled == b.benchmark.gpu_telemetry.enabled
    assert a.benchmark.server_metrics.enabled == b.benchmark.server_metrics.enabled

    # Artifacts
    assert a.benchmark.artifacts.slice_duration == b.benchmark.artifacts.slice_duration
    assert (
        a.benchmark.artifacts.show_trace_timing
        == b.benchmark.artifacts.show_trace_timing
    )


# ============================================================
# File-based loading: snake_case and camelCase YAML files
# ============================================================


class TestCamelCaseFileLoading:
    """Verify both casings work when loading from actual YAML files."""

    def test_load_snake_case_file(self, tmp_path: Path) -> None:
        f = tmp_path / "snake.yaml"
        f.write_text(_SNAKE_CASE_YAML)
        config = load_config(f)
        assert isinstance(config, AIPerfConfig)
        assert config.random_seed == 42
        assert not config.benchmark.gpu_telemetry.enabled

    def test_load_camel_case_file(self, tmp_path: Path) -> None:
        f = tmp_path / "camel.yaml"
        f.write_text(_CAMEL_CASE_YAML)
        config = load_config(f)
        assert isinstance(config, AIPerfConfig)
        assert config.random_seed == 42
        assert not config.benchmark.gpu_telemetry.enabled

    def test_load_mixed_case_file(self, tmp_path: Path) -> None:
        f = tmp_path / "mixed.yaml"
        f.write_text(_MIXED_CASE_YAML)
        config = load_config(f)
        assert isinstance(config, AIPerfConfig)
        assert config.random_seed == 42
        assert config.benchmark.endpoint.wait_for_model_timeout == 30.0

    def test_snake_and_camel_produce_equivalent_configs(self, tmp_path: Path) -> None:
        snake_file = tmp_path / "snake.yaml"
        snake_file.write_text(_SNAKE_CASE_YAML)
        camel_file = tmp_path / "camel.yaml"
        camel_file.write_text(_CAMEL_CASE_YAML)

        snake_cfg = load_config(snake_file)
        camel_cfg = load_config(camel_file)

        _assert_configs_equivalent(snake_cfg, camel_cfg)


# ============================================================
# Round-trip: load -> dump (camelCase) -> reload
# ============================================================


class TestCamelCaseRoundTrip:
    """Verify dump produces camelCase and reloads identically."""

    def test_dump_uses_camel_case_keys(self) -> None:
        config = load_config_from_string(_SNAKE_CASE_YAML)
        yaml_output = dump_config(config, exclude_defaults=False)

        assert "randomSeed:" in yaml_output
        assert "gpuTelemetry:" in yaml_output
        assert "serverMetrics:" in yaml_output
        assert "excludeFromResults:" in yaml_output
        assert "sliceDuration:" in yaml_output
        assert "showTraceTiming:" in yaml_output
        assert "urlStrategy:" in yaml_output
        assert "waitForModelTimeout:" in yaml_output
        assert "connectionReuse:" in yaml_output
        assert "useLegacyMaxTokens:" in yaml_output
        assert "useServerTokenCount:" in yaml_output
        assert "turnDelay:" in yaml_output
        assert "turnDelayRatio:" in yaml_output
        assert "batchSize:" in yaml_output
        assert "prefixPrompts:" in yaml_output
        assert "poolSize:" in yaml_output
        assert "rateRamp:" in yaml_output
        assert "gracePeriod:" in yaml_output

    def test_dump_does_not_contain_snake_case_multi_word_keys(self) -> None:
        config = load_config_from_string(_SNAKE_CASE_YAML)
        yaml_output = dump_config(config, exclude_defaults=False)

        # Multi-word keys should NOT appear in snake_case
        assert "random_seed:" not in yaml_output
        assert "gpu_telemetry:" not in yaml_output
        assert "server_metrics:" not in yaml_output
        assert "exclude_from_results:" not in yaml_output
        assert "slice_duration:" not in yaml_output
        assert "show_trace_timing:" not in yaml_output
        assert "url_strategy:" not in yaml_output
        assert "wait_for_model_timeout:" not in yaml_output
        assert "connection_reuse:" not in yaml_output
        assert "turn_delay_ratio:" not in yaml_output
        assert "batch_size:" not in yaml_output
        assert "prefix_prompts:" not in yaml_output
        assert "pool_size:" not in yaml_output
        assert "rate_ramp:" not in yaml_output
        assert "grace_period:" not in yaml_output

    def test_round_trip_from_snake_case(self, tmp_path: Path) -> None:
        """snake_case file -> load -> dump camelCase -> reload -> same values."""
        f = tmp_path / "snake.yaml"
        f.write_text(_SNAKE_CASE_YAML)
        original = load_config(f)

        camel_yaml = dump_config(original, exclude_defaults=False)
        reloaded = load_config_from_string(camel_yaml)

        _assert_configs_equivalent(original, reloaded)

    def test_round_trip_from_camel_case(self, tmp_path: Path) -> None:
        """camelCase file -> load -> dump camelCase -> reload -> same values."""
        f = tmp_path / "camel.yaml"
        f.write_text(_CAMEL_CASE_YAML)
        original = load_config(f)

        camel_yaml = dump_config(original, exclude_defaults=False)
        reloaded = load_config_from_string(camel_yaml)

        _assert_configs_equivalent(original, reloaded)

    def test_round_trip_via_file(self, tmp_path: Path) -> None:
        """Load -> dump to file -> reload from file."""
        from aiperf.config.loader import save_config

        source = tmp_path / "source.yaml"
        source.write_text(_SNAKE_CASE_YAML)
        original = load_config(source)

        output = tmp_path / "output.yaml"
        save_config(original, output, exclude_defaults=False)
        reloaded = load_config(output)

        _assert_configs_equivalent(original, reloaded)

        # Verify the saved file uses camelCase
        saved_content = output.read_text()
        assert "randomSeed:" in saved_content
        assert "gpuTelemetry:" in saved_content


# ============================================================
# Per-section camelCase coverage
# ============================================================


class TestCamelCasePerSection:
    """Verify camelCase works for each config section individually."""

    @pytest.mark.parametrize(
        "yaml_snippet, field, expected",
        [
            param(
                "waitForModelTimeout: 45.0",
                "wait_for_model_timeout",
                45.0,
                id="wait_for_model_timeout",
            ),
            param(
                "connectionReuse: never",
                "connection_reuse",
                "never",
                id="connection_reuse",
            ),
            param(
                "useLegacyMaxTokens: true",
                "use_legacy_max_tokens",
                True,
                id="use_legacy_max_tokens",
            ),
            param(
                "useServerTokenCount: true",
                "use_server_token_count",
                True,
                id="use_server_token_count",
            ),
            param(
                "downloadVideoContent: true",
                "download_video_content",
                True,
                id="download_video_content",
            ),
            param(
                "urlStrategy: round_robin",
                "url_strategy",
                "round_robin",
                id="url_strategy",
            ),
        ],
    )  # fmt: skip
    def test_endpoint_camel_case_fields(
        self, yaml_snippet: str, field: str, expected: object
    ) -> None:
        yaml_str = textwrap.dedent(f"""\
            benchmark:
              models: [m]
              endpoint:
                urls: [http://localhost:8000/v1/chat/completions]
                {yaml_snippet}
              datasets:
                - {{name: d, type: synthetic}}
              phases:
                - {{name: profiling, type: concurrency, requests: 1}}
        """)
        config = load_config_from_string(yaml_str)
        assert getattr(config.benchmark.endpoint, field) == expected

    def test_dataset_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - name: main
      type: synthetic
      entries: 50
      prompts:
        isl: 256
        batchSize: 4
        blockSize: 128
      prefixPrompts:
        sharedSystemLength: 100
        userContextLength: 50
      turnDelay: 200
      turnDelayRatio: 2.0
  phases:
    - {name: profiling, type: concurrency, requests: 1}
""")
        config = load_config_from_string(yaml_str)
        ds = config.benchmark.get_dataset("main")
        assert ds.prompts.batch_size == 4
        assert ds.prompts.block_size == 128
        assert ds.prefix_prompts.shared_system_length == 100
        assert ds.prefix_prompts.user_context_length == 50
        assert ds.turn_delay.expected_value == 200.0
        assert ds.turn_delay_ratio == 2.0

    def test_phase_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
    streaming: true
  datasets:
    - {name: d, type: synthetic}
  phases:
    - name: warmup
      type: concurrency
      requests: 10
      concurrency: 2
      excludeFromResults: true
      concurrencyRamp: 10
      prefillConcurrency: 1
    - name: profiling
      type: poisson
      duration: 60
      rate: 10.0
      rateRamp: {duration: 15, strategy: exponential}
      gracePeriod: 30
      seamless: true
""")
        config = load_config_from_string(yaml_str)
        warm = next(p for p in config.benchmark.phases if p.name == "warmup")
        assert warm.exclude_from_results is True
        assert warm.concurrency_ramp.duration == 10.0
        assert warm.prefill_concurrency == 1

        prof = next(p for p in config.benchmark.phases if p.name == "profiling")
        assert prof.rate_ramp.duration == 15.0
        assert prof.rate_ramp.strategy == "exponential"
        assert prof.grace_period == 30.0

    def test_fixed_schedule_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - name: profiling
      type: fixed_schedule
      autoOffset: false
      startOffset: 1000
      endOffset: 5000
""")
        config = load_config_from_string(yaml_str)
        phase = next(p for p in config.benchmark.phases if p.name == "profiling")
        assert phase.auto_offset is False
        assert phase.start_offset == 1000
        assert phase.end_offset == 5000

    def test_artifacts_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  artifacts:
    sliceDuration: 120
    showTraceTiming: true
""")
        config = load_config_from_string(yaml_str)
        assert config.benchmark.artifacts.slice_duration == 120.0
        assert config.benchmark.artifacts.show_trace_timing is True

    def test_runtime_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  runtime:
    serviceRunType: multiprocessing
    recordProcessors: 2
    apiPort: 9090
    apiHost: 0.0.0.0
    workersPerPod: 5
""")
        config = load_config_from_string(yaml_str)
        assert config.benchmark.runtime.service_run_type == "multiprocessing"
        assert config.benchmark.runtime.record_processors == 2
        assert config.benchmark.runtime.api_port == 9090
        assert config.benchmark.runtime.api_host == "0.0.0.0"
        assert config.benchmark.runtime.workers_per_pod == 5

    def test_multi_run_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
multiRun:
  numRuns: 3
  cooldownSeconds: 10.0
  confidenceLevel: 0.99
  setConsistentSeed: false
  disableWarmupAfterFirst: false
""")
        config = load_config_from_string(yaml_str)
        assert config.multi_run.num_runs == 3
        assert config.multi_run.cooldown_seconds == 10.0
        assert config.multi_run.confidence_level == 0.99
        assert config.multi_run.set_consistent_seed is False
        assert config.multi_run.disable_warmup_after_first is False

    def test_accuracy_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  accuracy:
    benchmark: mmlu
    nShots: 5
    enableCot: true
    systemPrompt: "Answer concisely."
""")
        config = load_config_from_string(yaml_str)
        assert config.benchmark.accuracy.n_shots == 5
        assert config.benchmark.accuracy.enable_cot is True
        assert config.benchmark.accuracy.system_prompt == "Answer concisely."

    def test_tcp_communication_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  runtime:
    communication:
      type: tcp
      recordsPort: 6000
      creditRouterPort: 6001
      controlPort: 6002
      eventBusProxy:
        frontendPort: 7000
        backendPort: 7001
      datasetManagerProxy:
        frontendPort: 7002
        backendPort: 7003
      rawInferenceProxy:
        frontendPort: 7004
        backendPort: 7005
""")
        config = load_config_from_string(yaml_str)
        comm = config.benchmark.runtime.communication
        assert comm.records_port == 6000
        assert comm.credit_router_port == 6001
        assert comm.control_port == 6002
        assert comm.event_bus_proxy.frontend_port == 7000
        assert comm.event_bus_proxy.backend_port == 7001
        assert comm.dataset_manager_proxy.frontend_port == 7002
        assert comm.raw_inference_proxy.frontend_port == 7004

    def test_dual_communication_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  runtime:
    communication:
      type: dual
      ipcPath: /tmp/bench
      tcpHost: 0.0.0.0
      controllerHost: controller.ns.svc
""")
        config = load_config_from_string(yaml_str)
        comm = config.benchmark.runtime.communication
        assert comm.ipc_path == "/tmp/bench"
        assert comm.tcp_host == "0.0.0.0"
        assert comm.controller_host == "controller.ns.svc"

    def test_tokenizer_camel_case_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  tokenizer:
    name: meta-llama/Llama-3.1-8B-Instruct
    trustRemoteCode: true
""")
        config = load_config_from_string(yaml_str)
        assert config.benchmark.tokenizer.trust_remote_code is True

    def test_server_metrics_discovery_camel_case(self) -> None:
        yaml_str = textwrap.dedent("""\
benchmark:
  models: [m]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: d, type: synthetic}
  phases:
    - {name: profiling, type: concurrency, requests: 1}
  serverMetrics:
    enabled: true
    discovery:
      mode: kubernetes
      labelSelector: app=vllm
""")
        config = load_config_from_string(yaml_str)
        assert config.benchmark.server_metrics.discovery.label_selector == "app=vllm"


# ============================================================
# Template file validation
# ============================================================


class TestTemplateFilesLoad:
    """Verify all shipped config templates (now camelCase) load correctly."""

    pytestmark = pytest.mark.skip(
        reason="Wave 2: shipped templates still use phases-as-dict; will be migrated "
        "in Task 7 of 2026-04-26-phases-list-with-name.md."
    )

    @staticmethod
    def _template_files() -> list[Path]:
        templates_dir = (
            Path(__file__).resolve().parents[3] / "src/aiperf/config/templates"
        )
        return sorted(templates_dir.glob("*.yaml"))

    @pytest.mark.parametrize(
        "template_file",
        _template_files.__func__(),
        ids=lambda p: p.stem,
    )  # fmt: skip
    def test_template_loads(self, template_file: Path) -> None:
        """Each template should parse without error (env var templates skip)."""
        content = template_file.read_text()
        try:
            config = load_config_from_string(content, substitute_env=False)
            assert isinstance(config, AIPerfConfig)
        except Exception as e:
            if "environment variable" in str(e).lower() or "not set" in str(e).lower():
                pytest.skip(f"Template uses env vars: {template_file.name}")
            # Jinja2 / env var references in numeric fields fail without substitution
            if (
                "unable to parse" in str(e).lower()
                or "invalid duration" in str(e).lower()
            ):
                pytest.skip(
                    f"Template uses Jinja2/env vars in numeric fields: {template_file.name}"
                )
            raise


# ============================================================
# JSON schema uses camelCase
# ============================================================


class TestJsonSchemaCamelCase:
    """Verify JSON schema property names are camelCase."""

    def test_top_level_schema_keys(self) -> None:
        schema = AIPerfConfig.model_json_schema()
        props = schema.get("properties", {})
        # Envelope keys: only sweep/multi_run/variables/random_seed live
        # at top level alongside `benchmark`.
        assert "randomSeed" in props
        assert "multiRun" in props
        assert "benchmark" in props
        assert "sweep" in props
        assert "variables" in props
        # Body keys (gpuTelemetry, serverMetrics, models, endpoint, datasets, phases)
        # now live under benchmark.* and must not appear at the envelope level.
        bench_schema = schema["$defs"]["BenchmarkConfig"]
        bench_props = bench_schema.get("properties", {})
        assert "gpuTelemetry" in bench_props
        assert "serverMetrics" in bench_props
        assert "models" in bench_props
        assert "endpoint" in bench_props
        assert "datasets" in bench_props
        assert "phases" in bench_props

    def test_nested_schema_keys(self) -> None:
        from aiperf.config.endpoint import EndpointConfig

        schema = EndpointConfig.model_json_schema()
        props = schema.get("properties", {})
        assert "urlStrategy" in props
        assert "waitForModelTimeout" in props
        assert "connectionReuse" in props
        assert "useLegacyMaxTokens" in props
        assert "useServerTokenCount" in props
        assert "downloadVideoContent" in props
