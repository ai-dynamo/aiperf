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
    """Build a single-turn plan whose default request budget exceeds its pool."""
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
    """Every sampling strategy can replay entries after exhausting the pool."""
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
    """Warmup can prime the server cache before profiling consumes the pool."""
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
        param({"request_count": None, "benchmark_duration": 1}, id="duration-only"),
        param({"prompt_input_tokens_mean": 0, "image_width_mean": 64,
               "image_height_mean": 64}, id="image-only"),
    ],
)  # fmt: skip
def test_no_warning_when_request_budget_does_not_imply_reuse(
    overrides: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """Count heuristics require text, a shared prefix, and single-turn sessions."""
    _preflight_synthetic_prompt_reuse(_plan(**overrides))

    assert "prefix cache" not in caplog.text


@pytest.mark.parametrize(
    "target",
    ["first_turn_suffix", "system_suffix", "warmup_isolation_first_turn"],
)
def test_cache_bust_that_preserves_leading_prefix_still_warns(
    target: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Changing a suffix or isolating warmup leaves reusable profiling prefixes."""
    _preflight_synthetic_prompt_reuse(_plan(cache_bust=target))

    assert "prefix cache" in caplog.text


def test_non_text_datasets_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Missing text prompts leave no synthetic text prefix to diagnose."""
    plan = _plan()
    plan.configs[0].get_default_dataset().prompts = None

    _preflight_synthetic_prompt_reuse(plan)

    assert "prefix cache" not in caplog.text


def test_default_cli_pool_already_matches_request_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI auto-sizing avoids exhaustion when no smaller pool is requested."""
    cli = CLIConfig(model_names=["test-model"], request_count=128)
    plan = build_benchmark_plan(resolve_config(cli))

    _preflight_synthetic_prompt_reuse(plan)

    assert plan.configs[0].get_default_dataset().entries == 128
    assert "prefix cache" not in caplog.text


@pytest.mark.parametrize("seed", [0, 42])
def test_fixed_seed_warns_even_when_pool_covers_requests(
    seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Cross-run replay remains possible with a full pool, including seed zero."""
    _preflight_synthetic_prompt_reuse(_plan(request_count=100, random_seed=seed))

    assert "fixed random seed" in caplog.text
    assert "across runs" in caplog.text
    assert "--reset-kv-cache" in caplog.text


def test_random_sampling_warns_before_pool_exhaustion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sampling with replacement risks duplicates without exhausting entries."""
    _preflight_synthetic_prompt_reuse(
        _plan(request_count=50, dataset_sampling_strategy="random")
    )

    assert "samples with replacement" in caplog.text
    assert "before the pool is exhausted" in caplog.text
    assert "sequential or shuffle" in caplog.text


@pytest.mark.parametrize(
    "overrides, expects_warning",
    [
        param({}, False, id="serial-partial-session"),
        param({"request_count": 3}, False, id="serial-complete-session"),
        param({"request_count": 4}, True, id="serial-second-session"),
        param({"concurrency": [2]}, True, id="concurrent-sessions"),
        param({"request_count": 1, "concurrency": [2]}, False, id="one-request"),
        param({"request_count": None, "conversation_num": 1}, False, id="one-session"),
        param({"request_count": None, "conversation_num": 2}, True, id="two-sessions"),
        param({"request_count": None, "benchmark_duration": 1}, True, id="duration-only"),
        param({"conversation_turn_stddev": 1}, True, id="variable-turns"),
        param({"warmup_request_count": 1}, True, id="warmup-and-profiling"),
        param({"request_rate": 10}, True, id="unbounded-rate-concurrency"),
    ],
)  # fmt: skip
def test_multi_turn_random_sampling_warning_respects_session_limits(
    overrides: dict[str, object],
    expects_warning: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Random draws happen at session starts, while turns retain the same slot."""
    plan = _plan(
        **{
            "request_count": 2,
            "conversation_turn_mean": 3,
            "dataset_sampling_strategy": "random",
            **overrides,
        }
    )

    _preflight_synthetic_prompt_reuse(plan)

    assert ("samples with replacement" in caplog.text) is expects_warning


@pytest.mark.parametrize("target", ["first_turn_prefix", "system_prefix"])
def test_prefix_cache_bust_suppresses_seed_and_random_warnings(
    target: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A fresh leading prefix prevents cache reuse even with repeated content."""
    _preflight_synthetic_prompt_reuse(
        _plan(random_seed=42, dataset_sampling_strategy="random", cache_bust=target)
    )

    assert "prefix cache" not in caplog.text


def test_cache_reset_suppresses_cross_run_warning_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Resetting before a run still allows random repeats and pool exhaustion."""
    _preflight_synthetic_prompt_reuse(
        _plan(random_seed=42, reset_kv_cache=True, dataset_sampling_strategy="random")
    )

    assert "fixed random seed" not in caplog.text
    assert "samples with replacement" in caplog.text
    assert "128" in caplog.text


def test_one_random_request_without_seed_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single unseeded request cannot replay another entry within its run."""
    _preflight_synthetic_prompt_reuse(
        _plan(request_count=1, dataset_sampling_strategy="random")
    )

    assert "prefix cache" not in caplog.text


def test_random_duration_run_warns_without_a_request_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A duration limit does not rule out repeated draws with replacement."""
    _preflight_synthetic_prompt_reuse(
        _plan(
            request_count=None, benchmark_duration=1, dataset_sampling_strategy="random"
        )
    )

    assert "samples with replacement" in caplog.text


def test_cache_reset_with_adequate_sequential_pool_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reset and distinct sequential entries cover both cache reuse risks."""
    _preflight_synthetic_prompt_reuse(
        _plan(
            request_count=100,
            random_seed=42,
            reset_kv_cache=True,
            dataset_sampling_strategy="sequential",
        )
    )

    assert "prefix cache" not in caplog.text


def test_effective_variation_seed_takes_precedence_over_plan_seed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unseeded variation must not inherit a fixed-seed warning from its plan."""
    plan = _plan(request_count=100, random_seed=42)
    plan.variation_seeds = [None]

    _preflight_synthetic_prompt_reuse(plan)

    assert "fixed random seed" not in caplog.text


@pytest.mark.parametrize(
    "run_seed, expects_warning",
    [
        param(None, False, id="dataset-seed-only"),
        param(42, True, id="fixed-run-seed"),
    ],
)  # fmt: skip
def test_prompt_replay_warning_uses_run_seed_with_dataset_seed(
    run_seed: int | None,
    expects_warning: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dataset seeds control session IDs; prompt content follows the run seed."""
    plan = _plan(request_count=100, random_seed=run_seed)
    plan.configs[0].get_default_dataset().random_seed = 42

    _preflight_synthetic_prompt_reuse(plan)

    assert ("fixed random seed" in caplog.text) is expects_warning


@pytest.mark.parametrize(
    "overrides",
    [
        param({"conversation_turn_mean": 3}, id="multi-turn"),
        param({"request_count": None, "benchmark_duration": 1}, id="duration-only"),
    ],
)  # fmt: skip
def test_fixed_seed_warning_is_independent_of_single_turn_budget(
    overrides: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """Content can replay across runs even when pool exhaustion is uncountable."""
    _preflight_synthetic_prompt_reuse(_plan(random_seed=42, **overrides))

    assert "fixed random seed" in caplog.text


def test_duplicate_seed_and_random_warnings_are_deduplicated_across_sweep(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A concurrency sweep should report each shared cache risk just once."""
    _preflight_synthetic_prompt_reuse(
        _plan(
            request_count=100,
            concurrency=[1, 2, 4],
            random_seed=42,
            dataset_sampling_strategy="random",
        )
    )

    assert sum("fixed random seed" in r.message for r in caplog.records) == 1
    assert sum("samples with replacement" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize(
    "turns",
    [
        param("1", id="integer-turns"),
        param("{mean: 1, stddev: 0.5}", id="fractional-stddev"),
        param("{mean: 1.5, stddev: 0}", id="fractional-mean"),
    ],
)  # fmt: skip
def test_yaml_pool_reuse_warns(
    turns: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Warnings follow generated integer turns, including fractional YAML values."""
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        "benchmark:\n"
        "  model: test-model\n"
        "  endpoint: {type: chat, url: 'http://localhost:8000'}\n"
        f"  dataset: {{type: synthetic, entries: 100, isl: 1024, turns: {turns}}}\n"
        "  profiling: {type: concurrency, concurrency: 1, requests: 128}\n"
    )
    plan = build_benchmark_plan(resolve_config(CLIConfig(config_file=config_file)))

    _preflight_synthetic_prompt_reuse(plan)

    assert "100 entries" in caplog.text
    assert "128" in caplog.text


def test_file_datasets_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """File-backed prompts fall outside the synthetic generation warning."""
    plan = _plan()
    plan.configs[0].datasets = [
        FileDataset(name="main", type="file", path="prompts.jsonl")
    ]

    _preflight_synthetic_prompt_reuse(plan)

    assert "prefix cache" not in caplog.text


def test_repeated_sweep_pool_warnings_are_deduplicated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated sweep configurations must not flood logs with the same budget."""
    plan = _plan(concurrency=[1, 2, 4])

    _preflight_synthetic_prompt_reuse(plan)

    assert sum("request budgets" in record.message for record in caplog.records) == 1


def test_cli_preflight_warns_before_service_bootstrap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Users must see the cache risk before benchmark services execute."""

    def check_warning(*args: object, **kwargs: object) -> None:
        """Observe the warning at the benchmark execution boundary."""
        assert "prefix cache" in caplog.text

    with (
        patch("aiperf.cli_runner._preflight_endpoint_ready"),
        patch("aiperf.cli_runner._preflight_dataset_materialize"),
        patch("aiperf.cli_runner._run_single_benchmark") as execute,
    ):
        execute.side_effect = check_warning
        run_benchmark(_plan())

    execute.assert_called_once()
