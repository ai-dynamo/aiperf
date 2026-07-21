# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""--isl-block-size routing for trace datasets.

``block_size`` is fundamentally a TRACE field: the hash-id trace loaders
(mooncake_trace, bailian_trace, burst_gpt_trace, sagemaker_data_capture) decode
each ``hash_id`` into a cached block of this many tokens (default 512 / 16 from
plugin metadata). v1 routed ``--isl-block-size`` onto
``input.prompt.input_tokens.block_size`` which those loaders read; the v2 cutover
strips the ``prompts`` subtable for FILE datasets, so the value must be re-routed
onto the flat ``FileDataset.block_size`` field (``_apply_block_size``).

A prior over-broad rejection blocked ``--isl-block-size`` on ALL non-synthetic
datasets -- breaking the #1 consumer (trace replay). It is now:
  - ACCEPTED for hash-id trace formats -> FileDataset.block_size,
  - REJECTED for weka (weka carries its own inline per-block sizes),
  - REJECTED for other non-hash-id datasets (no block decoding),
  - unchanged for synthetic (-> prompts.block_size).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import PublicDatasetType

_WEKA_PUBLIC = PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    p.touch()
    return p


@pytest.mark.parametrize(
    "fmt",
    ["mooncake_trace", "bailian_trace", "burst_gpt_trace", "sagemaker_data_capture"],
)
def test_block_size_routed_onto_filedataset_for_hash_id_traces(
    trace_file: Path, fmt: str
) -> None:
    out = build_dataset(
        CLIConfig(
            model_names=["m"],
            input_file=str(trace_file),
            custom_dataset_type=fmt,
            prompt_input_tokens_block_size=256,
            prompt_output_tokens_mean=64,
        )
    )
    assert out["type"] == "file"
    assert out["block_size"] == 256
    # Not leaked back onto a prompts subtable (stripped for file datasets).
    assert "prompts" not in out


def test_weka_trace_file_rejects_block_size(trace_file: Path) -> None:
    with pytest.raises(ValueError, match="inline"):
        build_dataset(
            CLIConfig(
                model_names=["m"],
                input_file=str(trace_file),
                custom_dataset_type="weka_trace",
                prompt_input_tokens_block_size=256,
            )
        )


def test_weka_public_rejects_block_size() -> None:
    with pytest.raises(ValueError, match="inline"):
        build_dataset(
            CLIConfig(
                model_names=["m"],
                public_dataset=_WEKA_PUBLIC,
                prompt_input_tokens_block_size=256,
            )
        )


@pytest.mark.parametrize(
    "fmt",
    [
        param("raw_payload", id="raw_payload"),
        param("single_turn", id="single_turn"),
    ],
)
def test_non_hash_id_file_dataset_rejects_block_size(
    trace_file: Path, fmt: str
) -> None:
    """Datasets that do not decode hash-id token blocks reject the flag with a
    clear message (rather than silently no-op)."""
    with pytest.raises(ValueError, match="hash-id token blocks"):
        build_dataset(
            CLIConfig(
                model_names=["m"],
                input_file=str(trace_file),
                custom_dataset_type=fmt,
                prompt_input_tokens_block_size=256,
            )
        )


def test_synthetic_block_size_still_routes_to_prompts() -> None:
    """--isl-block-size on a synthetic dataset is unchanged (prompts.block_size)."""
    out = build_dataset(
        CLIConfig(
            model_names=["m"],
            prompt_input_tokens_mean=128,
            prompt_input_tokens_block_size=64,
        )
    )
    assert out["type"] == "synthetic"
    assert out["prompts"]["block_size"] == 64


def test_loader_consumes_routed_block_size(trace_file: Path) -> None:
    """End-to-end: the routed FileDataset.block_size becomes the loader's
    hash-id decode block size (user override beats the plugin default)."""
    from unittest.mock import Mock

    from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
    from tests.unit.conftest import make_run_from_cli

    pg = Mock()
    pg.generate.return_value = "x"
    pg._decoded_cache = {}
    pg._build_token_sequence.return_value = [1, 2, 3]
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            input_file=str(trace_file),
            custom_dataset_type="mooncake_trace",
            prompt_input_tokens_block_size=256,
            prompt_output_tokens_mean=64,
        )
    )
    loader = MooncakeTraceDatasetLoader(
        filename=str(trace_file),
        run=run,
        prompt_generator=pg,
        default_block_size=512,
    )
    assert loader._block_size == 256  # user override beats plugin default 512
