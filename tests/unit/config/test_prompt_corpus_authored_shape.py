# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.enums import PromptCorpus
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.content import PromptConfig, PromptSelectionConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def test_prompt_config_accepts_corpus():
    cfg = PromptConfig(corpus=PromptCorpus.CODING)
    assert cfg.corpus == PromptCorpus.CODING


def test_file_dataset_accepts_prompts_corpus():
    ds = FileDataset.model_validate(
        {
            "name": "default",
            "type": "file",
            "path": "x.jsonl",
            "format": "weka_trace",
            "prompts": {"corpus": "coding"},
        }
    )
    assert ds.prompts is not None
    assert ds.prompts.corpus == PromptCorpus.CODING


def test_get_prompt_corpus_reads_prompts_corpus(tmp_path):
    p = tmp_path / "t.jsonl"
    p.touch()
    cli = CLIConfig.model_construct(
        model_names=["m"],
        input_file=str(p),
        custom_dataset_type=CustomDatasetType.WEKA_TRACE,
    )
    run = make_run_from_cli(cli)
    ds = run.cfg.get_default_dataset()
    ds.prompts = PromptSelectionConfig(corpus=PromptCorpus.SONNET)
    assert run.cfg.get_prompt_corpus() == PromptCorpus.SONNET
