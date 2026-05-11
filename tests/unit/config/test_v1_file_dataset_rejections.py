# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for v1 -> v2 converter rejecting synthetic-only flags
on file (mooncake_trace, single_turn, ...) datasets.

These flags previously leaked through ``_apply_dataset_type``'s strip into
``FileDataset`` validation and crashed with ``extra_forbidden`` (e.g.
``--isl-block-size`` carried via ``prompts.block_size``, ``--seq-dist`` via
``prompts.sequence_distribution``). The strip in ``_apply_dataset_type``
covers ``prompts``/``prefix_prompts``/``rankings``/``audio``/``images``/
``video`` keys at FILE-type discrimination time, but ``_apply_sequence_distribution``
runs *after* and can re-add ``prompts``. Reject at convert-time instead so
the user sees a clear flag-level error rather than a Pydantic stack trace
or silently-dropped flags (the prior behavior of ``--isl-block-size`` on
mooncake_trace, which masked the use of the hardcoded block-size fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


@pytest.fixture
def mc_jsonl(tmp_path: Path) -> Path:
    """A real (empty) JSONL path on disk. ``CLIConfig.input_file``'s
    ``parse_file`` validator requires existence; the converter only reads
    the *path* (not the contents), so an empty file is sufficient."""
    p = tmp_path / "mc.jsonl"
    p.touch()
    return p


def _file_user(mc_jsonl: Path, *, prompt_kwargs: dict | None = None) -> CLIConfig:
    """Build a v1 CLIConfig with ``--input-file`` + mooncake_trace + a
    synthetic-only prompt field set. ``prompt_kwargs`` keys must be the
    flat ``prompt_*`` attribute names on CLIConfig."""
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
            {"prompt_input_tokens_block_size": 20},
            "--isl-block-size",
            id="isl-block-size",
        ),
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
    ],
)  # fmt: skip
def test_synthetic_only_flag_rejected_on_file_dataset(
    mc_jsonl: Path, prompt_kwargs: dict, expected_flag_fragment: str
) -> None:
    """Each synthetic-only flag must raise ValueError naming the flag when
    paired with --input-file, instead of silently dropping or crashing
    AIPerfConfig validation with extra_forbidden."""
    user = _file_user(mc_jsonl, prompt_kwargs=prompt_kwargs)
    with pytest.raises(ValueError, match=expected_flag_fragment):
        build_dataset(user)


def test_mooncake_trace_without_synthetic_flags_validates_cleanly(
    mc_jsonl: Path,
) -> None:
    """The fix must not regress the happy path: mooncake_trace with only
    file-compatible flags (--input-file, --custom-dataset-type, --osl)
    must build a valid AIPerfConfig with no extra_forbidden fields."""
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
    # Synthetic-only subtables must be absent on the file-typed dict.
    for forbidden_key in (
        "prompts",
        "prefix_prompts",
        "rankings",
        "audio",
        "images",
        "video",
    ):
        assert forbidden_key not in out, f"FileDataset must not carry {forbidden_key!r}"
    # --osl is routed onto the flat FileDataset.osl field (not prompts.osl).
    assert out.get("osl") == {"mean": 64}

    # Full envelope must validate against AIPerfConfig without extra_forbidden.
    aiperf_cfg = convert_cli_to_aiperf(user)
    datasets = aiperf_cfg.benchmark.datasets
    assert len(datasets) == 1
    assert datasets[0].type == "file"
    assert str(datasets[0].path) == str(mc_jsonl)
