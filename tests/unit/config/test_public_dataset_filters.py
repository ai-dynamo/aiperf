# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.loader.exgentic import ExgenticDatasetLoader
from aiperf.dataset.loader.hf_instruction_response import (
    HFInstructionResponseDatasetLoader,
)
from aiperf.plugin.enums import PublicDatasetType
from tests.unit.dataset.composer.conftest import make_run


def _config(dataset: PublicDatasetType, filters: list[str]) -> CLIConfig:
    return CLIConfig(
        model_names=["target-model"],
        public_dataset=dataset,
        dataset_filters=filters,
    )


def test_cli_dataset_filters_flow_to_exgentic_loader() -> None:
    cli = _config(
        PublicDatasetType.EXGENTIC,
        [
            "harness=tool_calling_with_shortlisting",
            "source_model=Kimi-K2.5",
        ],
    )
    built = build_dataset(cli)
    composer = PublicDatasetComposer(run=make_run(cli), tokenizer=None)

    kwargs = composer._build_loader_kwargs(
        PublicDatasetType.EXGENTIC, ExgenticDatasetLoader
    )

    assert built["filters"] == {
        "harness": "tool_calling_with_shortlisting",
        "source_model": "Kimi-K2.5",
    }
    assert kwargs["filters"] == built["filters"]


@pytest.mark.parametrize(
    "filters, match",
    [
        param(["harness"], "expected non-empty key=value"),
        param(["harness="], "expected non-empty key=value"),
        param(["harness=a", "harness=b"], "Duplicate"),
    ],
)  # fmt: skip
def test_cli_dataset_filter_rejects_invalid_syntax(
    filters: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_dataset(_config(PublicDatasetType.EXGENTIC, filters))


def test_dataset_filter_rejected_by_loader_without_filter_support() -> None:
    cli = _config(PublicDatasetType.AIMO, ["harness=tool_calling"])
    composer = PublicDatasetComposer(run=make_run(cli), tokenizer=None)

    with pytest.raises(ValueError, match="does not support --dataset-filter"):
        composer._build_loader_kwargs(
            PublicDatasetType.AIMO, HFInstructionResponseDatasetLoader
        )


def test_dataset_filter_requires_public_dataset() -> None:
    with pytest.raises(ValueError, match="requires --public-dataset"):
        build_dataset(
            CLIConfig(model_names=["target-model"], dataset_filters=["harness=x"])
        )


def test_cli_dataset_filter_overrides_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "base.yaml"
    config_file.write_text(
        """
schemaVersion: "2.0"
benchmark:
  model: target-model
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: public
    dataset: exgentic
    filters:
      harness: claude_code
      source_model: DeepSeek-V3.2
  profiling:
    type: concurrency
    requests: 1
    concurrency: 1
"""
    )
    cli = CLIConfig(
        config_file=config_file,
        dataset_filters=["source_model=claude-opus-4-5"],
    )

    dataset = resolve_config(cli).benchmark.get_default_dataset()

    assert dataset.filters == {
        "harness": "claude_code",
        "source_model": "claude-opus-4-5",
    }


def test_cli_dataset_filter_rejects_non_public_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "base.yaml"
    config_file.write_text(
        """
schemaVersion: "2.0"
benchmark:
  model: target-model
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: synthetic
  profiling:
    type: concurrency
    requests: 1
    concurrency: 1
"""
    )

    with pytest.raises(ValueError, match="requires a public dataset"):
        resolve_config(
            CLIConfig(config_file=config_file, dataset_filters=["harness=x"])
        )
