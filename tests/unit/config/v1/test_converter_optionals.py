# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 converter optional-section builders."""

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_optionals import (
    build_accuracy,
    build_multi_run,
    build_tokenizer,
)
from aiperf.config.v1._loadgen import LoadGeneratorConfig


def _user_with_loadgen(**fields) -> UserConfig:
    return UserConfig(loadgen=LoadGeneratorConfig(**fields))


def test_build_tokenizer_returns_none_when_unset():
    user = UserConfig()
    assert build_tokenizer(user) is None


def test_build_tokenizer_returns_none_when_section_present_but_empty():
    user = UserConfig.model_validate({"tokenizer": {}})
    assert build_tokenizer(user) is None


def test_build_tokenizer_passes_set_fields():
    user = UserConfig.model_validate(
        {"tokenizer": {"name": "gpt2", "revision": "main"}}
    )
    out = build_tokenizer(user)
    assert out is not None
    assert out["name"] == "gpt2"
    assert out["revision"] == "main"
    assert "trust_remote_code" not in out


def test_build_tokenizer_includes_trust_remote_code_when_set():
    user = UserConfig.model_validate({"tokenizer": {"trust_remote_code": True}})
    out = build_tokenizer(user)
    assert out == {"trust_remote_code": True}


def test_build_accuracy_returns_none_when_unset():
    user = UserConfig()
    assert build_accuracy(user) is None


def test_build_accuracy_returns_none_when_section_present_but_empty():
    user = UserConfig.model_validate({"accuracy": {}})
    assert build_accuracy(user) is None


def test_build_accuracy_passes_set_fields():
    user = UserConfig.model_validate(
        {"accuracy": {"benchmark": "mmlu", "n_shots": 5, "enable_cot": True}}
    )
    out = build_accuracy(user)
    assert out is not None
    assert out["benchmark"] == "mmlu"
    assert out["n_shots"] == 5
    assert out["enable_cot"] is True


def test_build_multi_run_returns_none_when_unset():
    user = UserConfig()
    assert build_multi_run(user) is None


def test_build_multi_run_returns_none_when_loadgen_has_only_non_multirun_fields():
    user = UserConfig.model_validate({"loadgen": {"concurrency": 4}})
    assert build_multi_run(user) is None


def test_build_multi_run_passes_when_num_profile_runs_set():
    user = UserConfig.model_validate(
        {"loadgen": {"num_profile_runs": 3, "confidence_level": 0.95}}
    )
    out = build_multi_run(user)
    assert out is not None
    assert out["num_runs"] == 3
    assert out["confidence_level"] == 0.95


def test_build_multi_run_includes_convergence_fields():
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "convergence_metric": "request_latency",
                "convergence_threshold": 0.05,
            }
        }
    )
    out = build_multi_run(user)
    assert out is not None
    assert out["convergence_metric"] == "request_latency"
    assert out["convergence_threshold"] == 0.05


def test_build_multi_run_emits_typed_adaptive_search_when_set():
    user = _user_with_loadgen(
        search_space=["phases.profiling.concurrency:1,1000:int"],
        search_metric="output_token_throughput",
        search_direction="maximize",
        search_max_iterations=20,
    )
    out = build_multi_run(user)
    assert out is not None
    assert "adaptive_search" in out
    ol = out["adaptive_search"]
    # model_dump'd AdaptiveSearchConfig - typed shape with parsed search_space.
    assert ol["algorithm"] == "bayes"
    assert ol["max_iterations"] == 20
    assert ol["objective_metric"] == "output_token_throughput"
    assert ol["objective_stat"] == "avg"  # default when --search-stat omitted
    assert ol["objective_direction"] == "maximize"
    assert ol["search_space"] == [
        {
            "path": "phases.profiling.concurrency",
            "lo": 1.0,
            "hi": 1000.0,
            "kind": "int",
        },
    ]


def test_build_multi_run_propagates_explicit_stat():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="ttft",
        search_stat="p99",
        search_direction="minimize",
        search_max_iterations=10,
    )
    out = build_multi_run(user)
    assert out["adaptive_search"]["objective_stat"] == "p99"
    assert out["adaptive_search"]["objective_direction"] == "minimize"


def test_build_multi_run_no_adaptive_search_when_unset():
    user = _user_with_loadgen(num_profile_runs=3)
    out = build_multi_run(user)
    assert out == {"num_runs": 3}
    assert "adaptive_search" not in out


def test_build_multi_run_rejects_search_space_without_metric():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_direction="maximize",
        search_max_iterations=20,
    )
    with pytest.raises(TypeError, match="--search-space requires --search-metric"):
        build_multi_run(user)


def test_build_multi_run_rejects_search_space_without_direction():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_max_iterations=20,
    )
    with pytest.raises(TypeError, match="--search-space requires --search-direction"):
        build_multi_run(user)


def test_build_multi_run_rejects_search_space_without_max_iterations():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_direction="maximize",
    )
    with pytest.raises(
        TypeError, match="--search-space requires --search-max-iterations"
    ):
        build_multi_run(user)


def test_build_multi_run_propagates_initial_points_and_seed():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_direction="maximize",
        search_max_iterations=20,
        search_initial_points=3,
        search_random_seed=42,
    )
    out = build_multi_run(user)
    assert out["adaptive_search"]["n_initial_points"] == 3
    assert out["adaptive_search"]["random_seed"] == 42
