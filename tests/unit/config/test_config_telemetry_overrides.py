# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry and scenario flags must survive ``--config``.

``build_mlflow``, ``build_otel``, and ``build_network_latency`` have always
existed and are called by the CLI-only converter, but ``build_cli_overrides``
never called them -- so every ``--mlflow-*`` flag, ``--otel-url``, and the
network-latency flags were silently discarded whenever a config file was
supplied. Same story for ``--scenario`` / ``--unsafe-override``, which the
converter writes onto the benchmark body via ``_apply_scenario_fields``.

These flags configure where results are *published*. Dropping them silently
means a run reports to the wrong MLflow experiment, or to none at all, with
no diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def cli(**kwargs: object) -> CLIConfig:
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


@pytest.fixture
def base_yaml(tmp_path: Path) -> Path:
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


def test_mlflow_tracking_uri_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(mlflow_tracking_uri="http://mlflow:5000"), base_yaml)
    assert resolved.benchmark.mlflow.tracking_uri == "http://mlflow:5000"


def test_mlflow_experiment_routes(base_yaml: Path) -> None:
    resolved = resolve_config(
        cli(
            mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment="my-experiment"
        ),
        base_yaml,
    )
    assert resolved.benchmark.mlflow.experiment == "my-experiment"


def test_mlflow_run_name_routes(base_yaml: Path) -> None:
    resolved = resolve_config(
        cli(mlflow_tracking_uri="http://mlflow:5000", mlflow_run_name="run-7"),
        base_yaml,
    )
    assert resolved.benchmark.mlflow.run_name == "run-7"


def test_mlflow_secondary_flag_without_tracking_uri_still_errors(
    base_yaml: Path,
) -> None:
    """The guard stands when neither the CLI nor the YAML supplies a URI."""
    with pytest.raises(ValueError, match="require --mlflow-tracking-uri"):
        resolve_config(cli(mlflow_experiment="orphan"), base_yaml)


def test_mlflow_flag_overrides_yaml_value(base_yaml: Path, tmp_path: Path) -> None:
    """A YAML-supplied mlflow block must lose to the explicit flag."""
    cfg = tmp_path / "with_mlflow.yaml"
    cfg.write_text(
        base_yaml.read_text().replace(
            "  phases:",
            "  mlflow:\n    experiment: from-yaml\n    tracking_uri: http://yaml:5000\n  phases:",
        )
    )
    resolved = resolve_config(cli(mlflow_experiment="from-cli"), cfg)
    assert resolved.benchmark.mlflow.experiment == "from-cli"
    # The flag the user did not pass must survive untouched.
    assert resolved.benchmark.mlflow.tracking_uri == "http://yaml:5000"


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------


def test_otel_url_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(otel_url="http://otel:4318"), base_yaml)
    assert resolved.benchmark.otel.metrics_url is not None


def test_gen_ai_provider_routes(base_yaml: Path) -> None:
    resolved = resolve_config(
        cli(otel_url="http://otel:4318", gen_ai_provider="nvidia"), base_yaml
    )
    assert resolved.benchmark.otel.gen_ai_provider == "nvidia"


# ---------------------------------------------------------------------------
# Network latency
# ---------------------------------------------------------------------------


def test_network_latency_mean_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(network_latency_mean=12.5), base_yaml)
    assert resolved.benchmark.network_latency.mean_ms == 12.5


def test_network_latency_automatic_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(network_latency_automatic=True), base_yaml)
    assert resolved.benchmark.network_latency.enabled is True


# ---------------------------------------------------------------------------
# Scenario lock
# ---------------------------------------------------------------------------


def test_unsafe_override_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(unsafe_override=True), base_yaml)
    assert resolved.benchmark.unsafe_override is True


# ---------------------------------------------------------------------------
# Runtime: --api-host with a YAML-supplied api_port
# ---------------------------------------------------------------------------


@pytest.fixture
def api_port_yaml(base_yaml: Path, tmp_path: Path) -> Path:
    cfg = tmp_path / "with_port.yaml"
    cfg.write_text(
        base_yaml.read_text().replace(
            "  phases:", "  runtime:\n    api_port: 19090\n  phases:"
        )
    )
    return cfg


def test_api_host_accepts_port_from_yaml(api_port_yaml: Path) -> None:
    """--api-host must not demand a flag the config file already supplies.

    build_logging_runtime raises when api_host is set and api_port is None,
    which under --config is exactly the supported case: the port came from
    the file.
    """
    resolved = resolve_config(cli(api_host="0.0.0.0"), api_port_yaml)
    assert resolved.benchmark.runtime.api_host == "0.0.0.0"
    assert resolved.benchmark.runtime.api_port == 19090


def test_api_host_without_any_port_still_errors(base_yaml: Path) -> None:
    """The guard stands when neither the CLI nor the YAML supplies a port."""
    with pytest.raises(ValueError, match="api_host requires api_port"):
        resolve_config(cli(api_host="0.0.0.0"), base_yaml)


@pytest.fixture
def otel_yaml(base_yaml: Path, tmp_path: Path) -> Path:
    cfg = tmp_path / "with_otel.yaml"
    cfg.write_text(
        base_yaml.read_text().replace(
            "  phases:",
            "  otel:\n    metrics_url: http://yaml-otel:4318/v1/metrics\n  phases:",
        )
    )
    return cfg


def test_gen_ai_provider_accepts_otel_url_from_yaml(otel_yaml: Path) -> None:
    """Secondary OTel flags must not demand a URL the config file supplies."""
    resolved = resolve_config(cli(gen_ai_provider="nvidia"), otel_yaml)
    assert resolved.benchmark.otel.gen_ai_provider == "nvidia"
    assert str(resolved.benchmark.otel.metrics_url).startswith("http://yaml-otel")


def test_otel_secondary_flag_without_any_url_still_errors(base_yaml: Path) -> None:
    """The guard stands when neither source supplies a metrics URL."""
    with pytest.raises(ValueError, match="requires --otel-url"):
        resolve_config(cli(gen_ai_provider="nvidia"), base_yaml)


# ---------------------------------------------------------------------------
# Endpoint probes: reset-kv-cache and server-profiler
# ---------------------------------------------------------------------------


def test_reset_kv_cache_routes(base_yaml: Path) -> None:
    resolved = resolve_config(cli(reset_kv_cache=True), base_yaml)
    assert resolved.benchmark.endpoint.reset_kv_cache is not None


def test_reset_kv_cache_path_routes(base_yaml: Path) -> None:
    resolved = resolve_config(
        cli(reset_kv_cache=True, reset_kv_cache_path="/reset"), base_yaml
    )
    assert resolved.benchmark.endpoint.reset_kv_cache.path == "/reset"


def test_server_profiler_routes(base_yaml: Path) -> None:
    resolved = resolve_config(
        cli(server_profiler=True, server_profiler_start_path="/start"), base_yaml
    )
    assert resolved.benchmark.endpoint.server_profiler.start_path == "/start"


def test_probe_flag_overrides_yaml_block(base_yaml: Path, tmp_path: Path) -> None:
    """A YAML-supplied probe block must lose to the explicit flag, field-wise."""
    cfg = tmp_path / "with_probe.yaml"
    cfg.write_text(
        base_yaml.read_text()
        .replace(
            "  dataset:",
            "  endpoint2_placeholder: 0\n  dataset:",
        )
        .replace("  endpoint2_placeholder: 0\n", "")
    )
    cfg.write_text(
        base_yaml.read_text().replace(
            "    url: http://localhost:8000",
            "    url: http://localhost:8000\n"
            "    reset_kv_cache:\n"
            "      path: /from-yaml\n"
            "      timeout_seconds: 30",
        )
    )
    resolved = resolve_config(cli(reset_kv_cache_path="/from-cli"), cfg)
    assert resolved.benchmark.endpoint.reset_kv_cache.path == "/from-cli"
    # The field the user did not pass survives.
    assert resolved.benchmark.endpoint.reset_kv_cache.timeout_seconds == 30
