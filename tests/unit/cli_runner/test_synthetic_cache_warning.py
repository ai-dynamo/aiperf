# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic pool reuse must be visible before services start."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.cli_runner import run_benchmark
from aiperf.cli_runner._preflight import _preflight_synthetic_prompt_reuse
from aiperf.config import BenchmarkPlan
from aiperf.config.dataset import FileDataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.plan import build_benchmark_plan


def _plan(**overrides: object) -> BenchmarkPlan:
    cli = CLIConfig(
        **{
            "model_names": ["test-model"],
            "prompt_input_tokens_mean": 1024,
            "request_count": 128,
            "conversation_num_dataset_entries": 100,
            **overrides,
        }
    )
    return build_benchmark_plan(resolve_config(cli))


@pytest.mark.parametrize("sampling", ["sequential", "shuffle", "random"])
def test_pool_smaller_than_request_count_warns(
    sampling: str, caplog: pytest.LogCaptureFixture
) -> None:
    plan = _plan(dataset_sampling_strategy=sampling)

    _preflight_synthetic_prompt_reuse(plan)

    assert "128" in caplog.text
    assert "100 entries" in caplog.text
    assert "prefix cache" in caplog.text
    assert "TTFT" in caplog.text
    assert "--num-dataset-entries" in caplog.text


def test_warmup_requests_count_toward_pool_reuse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan = _plan(request_count=100, warmup_request_count=32)

    _preflight_synthetic_prompt_reuse(plan)

    assert "132" in caplog.text
    assert "warmup" in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [
        param({"request_count": 100}, id="equal-pool"),
        param({"conversation_num_dataset_entries": 200}, id="larger-pool"),
        param({"conversation_num": 10}, id="session-limit"),
        param({"conversation_turn_mean": 3}, id="multi-turn"),
        param({"conversation_turn_stddev": 1}, id="variable-turns"),
        param({"cache_bust": "first_turn_prefix"}, id="first-turn-prefix"),
        param({"cache_bust": "system_prefix"}, id="system-prefix"),
        param({"reset_kv_cache": True, "request_count": 100}, id="reset-within-pool"),
        param({"request_count": None, "benchmark_duration": 1}, id="duration-only"),
        param({"prompt_input_tokens_mean": 0, "image_width_mean": 64,
               "image_height_mean": 64}, id="image-only"),
    ],
)  # fmt: skip
def test_no_warning_when_request_budget_does_not_imply_reuse(
    overrides: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    _preflight_synthetic_prompt_reuse(_plan(**overrides))

    assert "prefix cache" not in caplog.text


@pytest.mark.parametrize(
    "target",
    ["first_turn_suffix", "system_suffix", "warmup_isolation_first_turn"],
)
def test_cache_bust_that_preserves_leading_prefix_still_warns(
    target: str, caplog: pytest.LogCaptureFixture
) -> None:
    _preflight_synthetic_prompt_reuse(_plan(cache_bust=target))

    assert "prefix cache" in caplog.text


def test_non_text_datasets_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    plan = _plan()
    plan.configs[0].get_default_dataset().prompts = None

    _preflight_synthetic_prompt_reuse(plan)

    assert "prefix cache" not in caplog.text


def test_default_cli_pool_already_matches_request_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cli = CLIConfig(model_names=["test-model"], request_count=128)
    plan = build_benchmark_plan(resolve_config(cli))

    _preflight_synthetic_prompt_reuse(plan)

    assert plan.configs[0].get_default_dataset().entries == 128
    assert "prefix cache" not in caplog.text


def test_yaml_pool_reuse_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        "benchmark:\n"
        "  model: test-model\n"
        "  endpoint: {type: chat, url: 'http://localhost:8000'}\n"
        "  dataset: {type: synthetic, entries: 100, isl: 1024}\n"
        "  profiling: {type: concurrency, concurrency: 1, requests: 128}\n"
    )
    plan = build_benchmark_plan(resolve_config(CLIConfig(config_file=config_file)))

    _preflight_synthetic_prompt_reuse(plan)

    assert "100 entries" in caplog.text
    assert "128" in caplog.text


def test_reset_before_run_does_not_prevent_within_run_reuse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _preflight_synthetic_prompt_reuse(_plan(reset_kv_cache=True))

    assert "prefix cache" in caplog.text


def test_file_datasets_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    plan = _plan()
    plan.configs[0].datasets = [
        FileDataset(name="main", type="file", path="prompts.jsonl")
    ]

    _preflight_synthetic_prompt_reuse(plan)

    assert "prefix cache" not in caplog.text


def test_repeated_sweep_configurations_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan = _plan(concurrency=[1, 2, 4])

    _preflight_synthetic_prompt_reuse(plan)

    assert sum("prefix cache" in record.message for record in caplog.records) == 1


def test_cli_preflight_warns_before_service_bootstrap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def check_warning(*args: object, **kwargs: object) -> None:
        assert "prefix cache" in caplog.text

    with (
        patch("aiperf.cli_runner._preflight_endpoint_ready"),
        patch("aiperf.cli_runner._preflight_dataset_materialize"),
        patch("aiperf.cli_runner._run_single_benchmark") as execute,
    ):
        execute.side_effect = check_warning
        run_benchmark(_plan())

    execute.assert_called_once()
