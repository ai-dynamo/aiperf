# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for v1 -> v2 converter rejecting synthetic-only flags"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


@pytest.fixture
def mc_jsonl(tmp_path: Path) -> Path:
    """A real (empty) JSONL path on disk. ``CLIConfig.input_file``'s"""
    p = tmp_path / "mc.jsonl"
    p.touch()
    return p


def _file_user(mc_jsonl: Path, *, prompt_kwargs: dict | None = None) -> CLIConfig:
    """Build a v1 CLIConfig with ``--input-file`` + mooncake_trace + a"""
    prompt_kwargs = prompt_kwargs or {}
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        **CLIConfig(request_count=5, concurrency=1).model_dump(exclude_unset=True),
        input_file=str(mc_jsonl),
        custom_dataset_type="mooncake_trace",
        **prompt_kwargs,
    )


@pytest.mark.parametrize(
    "prompt_kwargs, expected_flag_fragment",
    [
        param(
            {"prompt_input_tokens_mean": 128},
            "--isl",
            id="isl-mean",
        ),
        param(
            {"prompt_input_tokens_stddev": 10},
            "--isl-stddev",
            id="isl-stddev",
        ),
        param(
            {"prompt_batch_size": 4},
            "--prompt-batch-size",
            id="prompt-batch-size",
        ),
        param(
            {"prompt_sequence_distribution": "256,256:100.0"},
            "--seq-dist",
            id="seq-dist",
        ),
        param(
            {"prompt_prefix_length": 20},
            "--prompt-prefix-length",
            id="prefix-prompt-length",
        ),
        param(
            {"conversation_turn_mean": 3},
            "--conversation-turn-mean",
            id="conversation-turn-mean",
        ),
        param(
            {"conversation_turn_delay_mean": 1.0},
            "--conversation-turn-delay-mean",
            id="conversation-turn-delay-mean",
        ),
    ],
)  # fmt: skip
def test_synthetic_only_flag_rejected_on_file_dataset(
    mc_jsonl: Path, prompt_kwargs: dict, expected_flag_fragment: str
) -> None:
    """Each synthetic-only flag must raise ValueError naming the flag when"""
    user = _file_user(mc_jsonl, prompt_kwargs=prompt_kwargs)
    with pytest.raises(ValueError, match=expected_flag_fragment):
        build_dataset(user)


def test_mooncake_trace_without_synthetic_flags_validates_cleanly(
    mc_jsonl: Path,
) -> None:
    """The fix must not regress the happy path: mooncake_trace with only"""
    user = CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        **CLIConfig(request_count=5, concurrency=1).model_dump(exclude_unset=True),
        input_file=str(mc_jsonl),
        custom_dataset_type="mooncake_trace",
        prompt_output_tokens_mean=64,
    )

    out = build_dataset(user)
    assert out["type"] == "file"
    assert str(out["path"]) == str(mc_jsonl)
    assert out["format"] == "mooncake_trace"
    for forbidden_key in (
        "prompts",
        "prefix_prompts",
        "rankings",
        "audio",
        "images",
        "video",
    ):
        assert forbidden_key not in out, f"FileDataset must not carry {forbidden_key!r}"
    assert out.get("osl") == {"mean": 64}

    aiperf_cfg = convert_cli_to_aiperf(user)
    datasets = aiperf_cfg.benchmark.datasets
    assert len(datasets) == 1
    assert datasets[0].type == "file"
    assert str(datasets[0].path) == str(mc_jsonl)


@pytest.mark.parametrize(
    "extra, expected_flag_fragment",
    [
        param({"conversation_turn_mean": 3}, "--conversation-turn-mean", id="conv-turn-scalar"),
        param({"conversation_turn_mean": [1, 3]}, "--conversation-turn-mean", id="conv-turn-list"),
        param({"prompt_input_tokens_mean": 128}, "--isl", id="isl-scalar"),
        param({"prompt_input_tokens_mean": [128, 256]}, "--isl", id="isl-list"),
        param({"prompt_prefix_length": 20}, "--prompt-prefix-length", id="prefix"),
    ],
)  # fmt: skip
def test_synthetic_only_flag_rejected_on_public_dataset(
    extra: dict, expected_flag_fragment: str
) -> None:
    """Synthetic-only flags must raise a clear ValueError on a PUBLIC dataset"""
    from aiperf.plugin.enums import PublicDatasetType

    user = CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        **CLIConfig(request_count=5, concurrency=1).model_dump(exclude_unset=True),
        public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
        **extra,
    )
    with pytest.raises(ValueError, match=expected_flag_fragment):
        build_dataset(user)
