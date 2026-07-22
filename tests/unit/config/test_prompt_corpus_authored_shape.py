# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiperf.common.enums import PromptCorpus
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.content import PromptConfig, PromptSelectionConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def test_prompt_config_uses_corpus_not_prompt_corpus():
    cfg = PromptConfig(corpus=PromptCorpus.CODING)
    assert cfg.corpus == PromptCorpus.CODING
    assert (
        not hasattr(cfg, "prompt_corpus")
        or "prompt_corpus" not in PromptConfig.model_fields
    )


def test_file_dataset_rejects_flat_prompt_corpus():
    with pytest.raises(Exception) as exc:
        FileDataset.model_validate(
            {
                "name": "default",
                "type": "file",
                "path": "x.jsonl",
                "format": "weka_trace",
                "prompt_corpus": "coding",
            }
        )
    assert (
        "prompt_corpus" in str(exc.value).lower() or "extra" in str(exc.value).lower()
    )


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


def test_get_prompt_corpus_reads_prompts_corpus_only(tmp_path):
    p = tmp_path / "t.jsonl"
    p.touch()
    cli = CLIConfig.model_construct(
        model_names=["m"],
        input_file=str(p),
        custom_dataset_type=CustomDatasetType.WEKA_TRACE,
    )
    run = make_run_from_cli(cli)
    ds = run.cfg.get_default_dataset()
    # After Task 3 converter may set this; for this task set authored shape directly:
    ds.prompts = PromptSelectionConfig(corpus=PromptCorpus.SONNET)
    assert run.cfg.get_prompt_corpus() == PromptCorpus.SONNET
