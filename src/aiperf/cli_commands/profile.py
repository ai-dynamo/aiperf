# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running the Profile subcommand."""

from cyclopts import App

from aiperf.common.config import ServiceConfig, UserConfig

app = App(name="profile")


@app.default
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
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

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config.loader import load_service_config

        service_config = service_config or load_service_config()
        run_system_controller(user_config, service_config)
