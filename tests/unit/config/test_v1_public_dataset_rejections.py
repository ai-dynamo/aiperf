# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for v1 -> v2 converter rejecting synthetic-only flags
on public (--public-dataset) datasets.

Public datasets source their prompts from the downloaded dataset, so the
synthetic-only shaping flags (ISL, prefix prompts, batch_size, seq-dist,
multimodal batch_size) do not apply. ``_apply_dataset_type`` strips the
``prompts``/``prefix_prompts``/``rankings``/``audio``/``images``/``video``
subtables for PUBLIC datasets, so these flags were previously dropped
*silently* — the benchmark ran as if the user never passed them, with no
error or warning. This was inconsistent with the ``--input-file`` path,
which already raised a clear ValueError naming the offending flag. Reject at
convert-time for public datasets too so the user sees the same flag-level
error instead of a silently-ignored flag.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _public_user(*, prompt_kwargs: dict | None = None) -> CLIConfig:
    """Build a v1 CLIConfig with ``--public-dataset sharegpt`` + a
    synthetic-only prompt field set. ``prompt_kwargs`` keys must be the
    flat ``prompt_*`` attribute names on CLIConfig."""
    prompt_kwargs = prompt_kwargs or {}
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        **CLIConfig(request_count=5, concurrency=1).model_dump(exclude_unset=True),
        public_dataset="sharegpt",
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
        param(
            {"prompt_prefix_pool_size": 5},
            "--prompt-prefix-pool-size",
            id="prefix-prompt-pool-size",
        ),
        param(
            {"prompt_prefix_shared_system_length": 10},
            "--shared-system-prompt-length",
            id="shared-system-prompt-length",
        ),
        param(
            {"prompt_prefix_user_context_length": 10},
            "--user-context-prompt-length",
            id="user-context-prompt-length",
        ),
        param(
            {"image_batch_size": 2},
            "--image-batch-size",
            id="image-batch-size",
        ),
        param(
            {"image_source": "https://example.com/x.png"},
            "--image-source",
            id="image-source",
        ),
        param(
            {"audio_batch_size": 2},
            "--audio-batch-size",
            id="audio-batch-size",
        ),
        param(
            {"video_batch_size": 2},
            "--video-batch-size",
            id="video-batch-size",
        ),
    ],
)  # fmt: skip
def test_synthetic_only_flag_rejected_on_public_dataset(
    prompt_kwargs: dict, expected_flag_fragment: str
) -> None:
    """Each synthetic-only flag must raise ValueError naming the flag when
    paired with --public-dataset, instead of silently dropping it."""
    user = _public_user(prompt_kwargs=prompt_kwargs)
    with pytest.raises(ValueError, match=expected_flag_fragment):
        build_dataset(user)


def test_public_dataset_without_synthetic_flags_validates_cleanly() -> None:
    """The fix must not regress the happy path: a public dataset with only
    public-compatible flags must build a valid AIPerfConfig with no
    synthetic-only subtables leaking onto the public-typed dict."""
    user = CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        **CLIConfig(request_count=5, concurrency=1).model_dump(exclude_unset=True),
        public_dataset="sharegpt",
    )

    out = build_dataset(user)
    assert out["type"] == "public"
    assert out["dataset"] == "sharegpt"
    # Synthetic-only subtables must be absent on the public-typed dict.
    for forbidden_key in (
        "prompts",
        "prefix_prompts",
        "rankings",
        "audio",
        "images",
        "video",
    ):
        assert forbidden_key not in out, (
            f"PublicDataset must not carry {forbidden_key!r}"
        )

    # Full envelope must validate against AIPerfConfig without extra_forbidden.
    aiperf_cfg = convert_cli_to_aiperf(user)
    datasets = aiperf_cfg.benchmark.datasets
    assert len(datasets) == 1
    assert datasets[0].type == "public"
