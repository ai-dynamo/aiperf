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


def test_synthetic_coding_with_prefix_prompts_builds_dataset(mock_tokenizer):
    """Coding corpus must support prefix prompts (same generator surface as sonnet)."""
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
        prompt_corpus="coding",
        prompt_prefix_pool_size=2,
        prompt_prefix_length=8,
        conversation_num_dataset_entries=2,
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    assert isinstance(composer.prompt_generator, CodingContentGenerator)
    conversations = composer.create_dataset()
    assert len(conversations) == 2
    first_turn = conversations[0].turns[0]
    texts = [t for t in first_turn.texts] if first_turn.texts else []
    assert texts
    assert any(c for t in texts for c in t.contents)


def test_synthetic_coding_with_shared_system_prompt(mock_tokenizer):
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
        prompt_corpus="coding",
        prompt_prefix_shared_system_length=10,
        conversation_num_dataset_entries=1,
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    conversations = composer.create_dataset()
    assert conversations[0].system_message is not None
