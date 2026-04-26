# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 converter optional-section builders."""

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_optionals import (
    build_accuracy,
    build_multi_run,
    build_tokenizer,
)


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
