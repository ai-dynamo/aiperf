# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pytest import param

from aiperf.common.tokenizer import Tokenizer
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.provenance import public_dataset_provenance
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.conftest import make_run_from_cli


@pytest.mark.parametrize(
    "options, expected_repo, expected_split",
    [
        param({"hf_mooncake_dataset": "fixture/mooncake"}, "fixture/mooncake", "test", id="mooncake-default-split"),
        param({"hf_mooncake_dataset": "fixture/mooncake", "hf_mooncake_split": "train"}, "fixture/mooncake", "train", id="mooncake-split-override"),
        param({"hf_weka_dataset": "fixture/weka"}, "fixture/weka", "train", id="weka-repo-override"),
        param({"public_dataset": "semianalysis_cc_traces_weka_062126"}, "semianalysisai/cc-traces-weka-062126", "train", id="weka-registry-source"),
    ],
)  # fmt: skip
def test_public_dataset_provenance_matches_effective_hf_source(
    options: dict[str, Any],
    expected_repo: str,
    expected_split: str,
    mock_tokenizer_cls: type[Tokenizer],
) -> None:
    """Recorded source identity must match the source passed to the loader."""
    run = make_run_from_cli(
        CLIConfig(model_names=["test-model"], request_count=1, **options)
    )
    dataset = run.cfg.get_default_dataset()
    composer = PublicDatasetComposer(
        run=run, tokenizer=mock_tokenizer_cls.from_pretrained("test-model")
    )
    loader_class = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset.dataset)
    loader_kwargs = composer._build_loader_kwargs(dataset.dataset, loader_class)

    assert loader_kwargs["hf_dataset_name"] == expected_repo
    assert loader_kwargs["hf_split"] == expected_split
    assert public_dataset_provenance(run.cfg) == {
        "source_type": "public_dataset",
        "loader": str(dataset.dataset),
        "hf_dataset_name": expected_repo,
        "hf_split": expected_split,
    }
