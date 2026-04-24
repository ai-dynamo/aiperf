# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running the Profile subcommand."""

from cyclopts import App

from aiperf.config.cli_model import CLIModel

app = App(name="profile")


@app.default
def profile(
    *,
    cli_model: CLIModel,
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
        cli_model: Dynamically generated CLI model from CLI flags.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_benchmark
        from aiperf.config.cli_converter import build_aiperf_config
        from aiperf.config.loader import build_benchmark_plan

        config_file = getattr(cli_model, "config_file", None)
        if config_file is not None:
            from aiperf.config.loader import load_benchmark_plan

            plan = load_benchmark_plan(config_file)
        else:
            config = build_aiperf_config(cli_model)
            plan = build_benchmark_plan(config)
        run_benchmark(plan)
