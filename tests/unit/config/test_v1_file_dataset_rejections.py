# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for v1 -> v2 converter rejecting synthetic-only flags
on file (mooncake_trace, single_turn, ...) datasets.

These flags previously leaked through ``_apply_dataset_type``'s strip into
``FileDataset`` validation and crashed with ``extra_forbidden`` (e.g.
``--seq-dist`` via ``prompts.sequence_distribution``). The strip in
``_apply_dataset_type`` covers ``prompts``/``prefix_prompts``/``rankings``/
``audio``/``images``/``video`` keys at FILE-type discrimination time, but
``_apply_sequence_distribution`` runs *after* and can re-add ``prompts``.
Reject at convert-time instead so the user sees a clear flag-level error
rather than a Pydantic stack trace or silently-dropped flags.

``--isl-block-size`` is the exception: it is NOT rejected on file datasets
(AIP-1016). Block size partitions the recorded ISL into cache blocks for
trace replay, so it is meaningful, and it routes onto ``FileDataset.block_size``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset.config import FileDataset
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import DatasetFormat


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
    """Synthetic-only flags must raise a clear ValueError on a PUBLIC dataset
    (weka_hf) too -- not silently drop (scalar) or crash with extra_forbidden
    (magic-list) as they did when the rejection was FILE-only."""
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


class TestRandomPoolBatchSizeCarveOut:
    """random_pool must be exempt from the batch-size flag rejection."""

    @pytest.fixture
    def pool_jsonl(self, tmp_path: Path) -> Path:
        p = tmp_path / "pool.jsonl"
        p.touch()
        return p

    @pytest.mark.parametrize(
        "batch_kwarg, expected_field",
        [
            param({"prompt_batch_size": 5}, "prompt_batch_size", id="text"),
            param(
                {"prompt_batch_size": 0}, "prompt_batch_size", id="text-zero-disables"
            ),
            param({"image_batch_size": 3}, "image_batch_size", id="image"),
            param({"audio_batch_size": 2}, "audio_batch_size", id="audio"),
            param({"video_batch_size": 4}, "video_batch_size", id="video"),
        ],
    )  # fmt: skip
    def test_batch_size_allowed_with_random_pool(
        self, pool_jsonl: Path, batch_kwarg: dict, expected_field: str
    ) -> None:
        """Batch-size flags must not raise when custom_dataset_type is random_pool.

        prompt_batch_size=0 must be accepted (not just >=1): it's the documented way
        to disable text for image/audio/video-only random_pool workloads, matching
        image/audio/video_batch_size's existing 0-disables convention.
        """
        cli = CLIConfig(
            model_names=["test-model"],
            input_file=str(pool_jsonl),
            custom_dataset_type="random_pool",
            **batch_kwarg,
        )
        cfg = convert_cli_to_aiperf(cli)
        dataset = cfg.benchmark.datasets[0]
        assert getattr(dataset, expected_field) == next(iter(batch_kwarg.values()))

    @pytest.mark.parametrize(
        "batch_kwarg, expected_flag_fragment",
        [
            param({"prompt_batch_size": 5}, "--prompt-batch-size", id="text"),
            param({"image_batch_size": 3}, "--image-batch-size", id="image"),
        ],
    )  # fmt: skip
    def test_batch_size_still_rejected_for_trace_formats(
        self, mc_jsonl: Path, batch_kwarg: dict, expected_flag_fragment: str
    ) -> None:
        """Batch-size flags must still be rejected for non-random-pool file datasets."""
        cli = CLIConfig(
            model_names=["test-model"],
            input_file=str(mc_jsonl),
            custom_dataset_type="mooncake_trace",
            **batch_kwarg,
        )
        with pytest.raises(ValueError, match=re.escape(expected_flag_fragment)):
            convert_cli_to_aiperf(cli)


def test_batch_size_fields_rejected_on_non_random_pool_format(tmp_path: Path) -> None:
    """FileDataset model validator rejects batch-size fields when format != random_pool."""
    p = tmp_path / "f.jsonl"
    p.touch()
    with pytest.raises(
        ValueError, match="are rejected on formats other than random_pool"
    ):
        FileDataset(
            name="d",
            type="file",
            path=p,
            format=DatasetFormat.MOONCAKE_TRACE,
            prompt_batch_size=8,
        )
