# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running the Profile subcommand."""

from pathlib import Path
from typing import Annotated

from cyclopts import App
from pydantic import BaseModel

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.cli_parameter import CLIParameter

app = App(name="profile")


def _merge_cli_overrides_into(base: UserConfig, cli: UserConfig) -> UserConfig:
    """Overlay CLI-explicit fields onto a YAML-loaded base config.

    Only fields the user explicitly typed on the CLI (tracked via Pydantic's
    `model_fields_set` at every nesting level) override the base. Fields the
    CLI never touched stay as the YAML loaded them — most importantly the
    `media_mix` array, since no CLI flag targets it.
    """
    _overlay(base, cli)
    return base


def _overlay(base: BaseModel, cli: BaseModel) -> None:
    """Recursively overlay `cli`'s explicitly-set fields onto `base` in place."""
    for field_name in cli.model_fields_set:
        cli_value = getattr(cli, field_name)
        base_value = getattr(base, field_name, None)
        if isinstance(cli_value, BaseModel) and isinstance(base_value, BaseModel):
            _overlay(base_value, cli_value)
        else:
            setattr(base, field_name, cli_value)


def _resolve_user_config(
    cli_user_config: UserConfig | None, user_config_file: Path | None
) -> UserConfig:
    """Resolve the final UserConfig from CLI flags, a config file, or both."""
    from aiperf.common.config.loader import load_user_config
    from aiperf.common.environment import Environment

    file_path = user_config_file or Environment.CONFIG.USER_FILE
    if file_path is not None:
        base = load_user_config(file_path)
        return (
            _merge_cli_overrides_into(base, cli_user_config)
            if cli_user_config is not None
            else base
        )
    if cli_user_config is None:
        raise ValueError(
            "No user configuration provided. Pass CLI flags (e.g., --model, --url) "
            "or set --user-config-file <path> / AIPERF_CONFIG_USER_FILE=<path>."
        )
    return cli_user_config


def _maybe_wait_for_endpoint(user_config: UserConfig) -> None:
    """Block until the configured endpoint is ready, when a probe timeout is set."""
    if user_config.endpoint.wait_for_model_timeout <= 0:
        return

    import asyncio
    import logging

    from aiperf.common.readiness_probe import wait_for_endpoint

    # The probe runs before `run_system_controller` (which installs rich
    # logging), so there are no handlers attached yet. Install a basic stderr
    # handler so probe log messages are visible.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    raw_headers = user_config.input.headers or []
    headers = {str(k): str(v) for k, v in raw_headers}
    if user_config.endpoint.api_key:
        headers["Authorization"] = f"Bearer {user_config.endpoint.api_key}"

    asyncio.run(
        wait_for_endpoint(
            urls=user_config.endpoint.urls,
            model_names=user_config.endpoint.model_names,
            mode=user_config.endpoint.wait_for_model_mode,
            endpoint_type=str(user_config.endpoint.type),
            custom_endpoint=user_config.endpoint.custom_endpoint,
            timeout_s=user_config.endpoint.wait_for_model_timeout,
            interval_s=user_config.endpoint.wait_for_model_interval,
            headers=headers,
        )
    )


@app.default
def profile(
    user_config: UserConfig | None = None,
    service_config: ServiceConfig | None = None,
    *,
    user_config_file: Annotated[
        Path | None,
        CLIParameter(
            help="Path to a user configuration file (JSON or YAML). When set, the "
            "file's values become the baseline UserConfig and individual CLI flags "
            "override the file for global fields (same scope, last-write wins). "
            "Per-archetype overrides inside media_mix are never affected by CLI "
            "flags — they live at a finer scope than any CLI flag can express. "
            "Falls back to the AIPERF_CONFIG_USER_FILE env var."
        ),
    ] = None,
) -> None:
    """Run the Profile subcommand.

    Benchmark generative AI models and measure performance metrics including throughput,
    latency, token statistics, and resource utilization.

    Examples:
        # Basic profiling with streaming
        aiperf profile --model Qwen/Qwen3-0.6B --url localhost:8000 --endpoint-type chat --streaming

        # Concurrency-based benchmarking
        aiperf profile --model your_model --url localhost:8000 --concurrency 10 --request-count 100

        # Request rate benchmarking (Poisson distribution)
        aiperf profile --model your_model --url localhost:8000 --request-rate 5.0 --benchmark-duration 60

        # Time-based benchmarking with grace period
        aiperf profile --model your_model --url localhost:8000 --benchmark-duration 300 --benchmark-grace-period 30

        # Custom dataset with fixed schedule replay
        aiperf profile --model your_model --url localhost:8000 --input-file trace.jsonl --fixed-schedule

        # Multi-turn conversations with ShareGPT dataset
        aiperf profile --model your_model --url localhost:8000 --public-dataset sharegpt --num-sessions 50

        # Goodput measurement with SLOs
        aiperf profile --model your_model --url localhost:8000 --goodput "request_latency:250 inter_token_latency:10"

        # YAML configuration (required for media mix and other complex setups)
        aiperf profile --user-config-file media-mix.yaml

        # YAML baseline with CLI overrides for global fields
        aiperf profile --user-config-file media-mix.yaml --model your_model --url localhost:8000

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
        user_config_file: Path to a YAML or JSON user configuration file
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config.loader import load_service_config

        service_config = service_config or load_service_config()
        resolved_config = _resolve_user_config(user_config, user_config_file)
        _maybe_wait_for_endpoint(resolved_config)
        run_system_controller(resolved_config, service_config)
