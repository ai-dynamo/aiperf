# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from tests.unit.conftest import make_run_from_cli


def test_synthetic_default_uses_sonnet_generator(mock_tokenizer):
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    assert isinstance(composer.prompt_generator, PromptGenerator)
    assert not isinstance(composer.prompt_generator, CodingContentGenerator)


def test_synthetic_prompt_corpus_coding_uses_coding_generator(mock_tokenizer):
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
        prompt_corpus="coding",
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    assert isinstance(composer.prompt_generator, CodingContentGenerator)
