# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Injection of the verbatim ``--system-prompt`` into composed conversations.

Covers the three composers that build conversations (synthetic, custom/file,
public) and the prepend-vs-assign branch that keeps a dataset's own authored
system message intact.
"""

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.unit.dataset.composer.conftest import make_run

SYSTEM_TEXT = "You are a production assistant."


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "system.txt"
    path.write_text(SYSTEM_TEXT)
    return path


@pytest.fixture
def multi_turn_file(tmp_path):
    """A dataset whose first turn is a system message the loader hoists."""
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"type": "multi_turn", "session_id": "s1", "turns": ['
        '{"role": "system", "text": "Dataset system."}, '
        '{"role": "user", "text": "hi"}]}\n'
    )
    return path


def _synthetic_cli(**kwargs) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=3,
        prompt_input_tokens_mean=10,
        prompt_output_tokens_mean=5,
        **kwargs,
    )


class TestSyntheticComposerSystemPrompt:
    def test_sets_system_message_on_every_conversation(
        self, prompt_file, mock_tokenizer
    ):
        run = make_run(_synthetic_cli(system_prompt_file=str(prompt_file)))
        conversations = SyntheticDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert len(conversations) == 3
        assert all(c.system_message == SYSTEM_TEXT for c in conversations)

    def test_absent_without_the_flag(self, mock_tokenizer):
        run = make_run(_synthetic_cli())
        conversations = SyntheticDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert all(c.system_message is None for c in conversations)

    def test_inline_text_matches_file_text(self, prompt_file, mock_tokenizer):
        from_file = SyntheticDatasetComposer(
            run=make_run(_synthetic_cli(system_prompt_file=str(prompt_file))),
            tokenizer=mock_tokenizer,
        ).create_dataset()
        from_text = SyntheticDatasetComposer(
            run=make_run(_synthetic_cli(system_prompt=SYSTEM_TEXT)),
            tokenizer=mock_tokenizer,
        ).create_dataset()

        assert from_file[0].system_message == from_text[0].system_message


class TestCustomComposerSystemPrompt:
    def test_prepends_to_authored_system_message(
        self, prompt_file, multi_turn_file, mock_tokenizer
    ):
        """The dataset's own system message is kept, not replaced."""
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(multi_turn_file),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == f"{SYSTEM_TEXT}\n\nDataset system."

    def test_authored_system_message_untouched_without_the_flag(
        self, multi_turn_file, mock_tokenizer
    ):
        run = make_run(
            CLIConfig(model_names=["test-model"], input_file=str(multi_turn_file))
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == "Dataset system."

    def test_assigns_when_dataset_has_no_system_message(
        self, prompt_file, tmp_path, mock_tokenizer
    ):
        data = tmp_path / "plain.jsonl"
        data.write_text('{"text": "hello"}\n')
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(data),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == SYSTEM_TEXT

    def test_injected_without_a_tokenizer(self, prompt_file, tmp_path):
        """Verbatim text needs no tokenizer, unlike the synthetic prefix path."""
        data = tmp_path / "plain.jsonl"
        data.write_text('{"text": "hello"}\n')
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(data),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(run=run, tokenizer=None).create_dataset()

        assert conversations[0].system_message == SYSTEM_TEXT
