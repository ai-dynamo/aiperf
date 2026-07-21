# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-critical: trace loaders whose plugin metadata declares
``default_prompt_corpus: coding`` (e.g. weka_trace) must decode hash_ids
against the coding corpus, not the default Shakespeare/sonnet corpus.

Using the wrong corpus produces different request bytes, ISL token counts, and
prefix-cache-hit metrics for the weka MVP path, so this is a regression guard.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import PromptCorpus
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

_WEKA_FIXTURE_DIR = "tests/fixtures/weka_traces_small"


@pytest.fixture
def weka_run():
    cli = CLIConfig.model_construct(
        model_names=["test-model"],
        input_file=_WEKA_FIXTURE_DIR,
        custom_dataset_type=CustomDatasetType.WEKA_TRACE,
    )
    return make_run_from_cli(cli)


class TestCodingCorpusInjection:
    def test_weka_uses_coding_generator_not_sonnet(self, weka_run, mock_tokenizer):
        """weka_trace -> CodingContentGenerator (coding corpus), not PromptGenerator."""
        composer = CustomDatasetComposer(run=weka_run, tokenizer=mock_tokenizer)

        # The base composer always builds the default sonnet PromptGenerator.
        assert isinstance(composer.prompt_generator, PromptGenerator)
        assert not isinstance(composer.prompt_generator, CodingContentGenerator)

        composer.create_dataset()

        # The loader gets a CodingContentGenerator injected because weka_trace
        # registers default_prompt_corpus: coding in plugins.yaml.
        assert composer.detected_dataset_type == CustomDatasetType.WEKA_TRACE
        loader_gen = composer.loader.prompt_generator
        assert isinstance(loader_gen, CodingContentGenerator), (
            f"weka must replay against the coding corpus; got {type(loader_gen).__name__}"
        )
        # The coding generator exposes its tool_pool as _tokenized_corpus.
        assert len(loader_gen._tokenized_corpus) > 0

    def test_explicit_prompt_corpus_overrides_loader_default(
        self, weka_run, mock_tokenizer
    ):
        """An explicit FileDataset.prompt_corpus=sonnet overrides the loader's
        coding default. The composer reads via ``run.cfg.get_prompt_corpus()``."""
        # Set the file dataset's prompt_corpus directly: the CLI converter only
        # routes --prompt-corpus into the synthetic ``prompts`` sub-config today
        # (PORT-TODO: route it to FileDataset.prompt_corpus for trace datasets),
        # but the composer reader honors the resolved field for either shape.
        weka_run.cfg.get_default_dataset().prompt_corpus = PromptCorpus.SONNET
        assert weka_run.cfg.get_prompt_corpus() == PromptCorpus.SONNET

        composer = CustomDatasetComposer(run=weka_run, tokenizer=mock_tokenizer)
        composer.create_dataset()

        loader_gen = composer.loader.prompt_generator
        # Explicit SONNET wins: the default sonnet PromptGenerator is used.
        assert isinstance(loader_gen, PromptGenerator)
        assert not isinstance(loader_gen, CodingContentGenerator)
