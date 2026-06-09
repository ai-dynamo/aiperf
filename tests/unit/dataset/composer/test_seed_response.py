# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for composer synthesis of per-turn seed responses (mid-conversation seeding)."""

from unittest.mock import MagicMock

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.base import BaseDatasetComposer
from tests.unit.dataset.composer.conftest import make_run


class _Composer(BaseDatasetComposer):
    def create_dataset(self):
        return []


def _composer() -> _Composer:
    cfg = CLIConfig(
        model_names=["test-model"],
        model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        conversation_num=1,
        conversation_turn_mean=1,
        prompt_input_tokens_mean=100,
        prompt_output_tokens_mean=50,
    )
    return _Composer(run=make_run(cfg), tokenizer=MagicMock())


class TestSeedResponseSynthesis:
    def test_disabled_by_default(self):
        composer = _composer()
        assert composer._seed_enabled is False
        turn = Turn(max_tokens=50)
        composer._finalize_turn(turn)
        assert turn.seed_response is None

    def test_set_seed_response_builds_assistant_placeholder(self):
        composer = _composer()
        composer.prompt_generator = MagicMock()
        composer.prompt_generator.generate_prompt.return_value = "RESP"

        turn = Turn(max_tokens=50)
        composer._set_seed_response(turn)

        # Sized to the turn's output length, shaped like a captured response.
        composer.prompt_generator.generate_prompt.assert_called_once_with(50)
        assert turn.seed_response is not None
        assert turn.seed_response.role == "assistant"
        assert turn.seed_response.raw_messages == [
            {"role": "assistant", "content": "RESP"}
        ]

    def test_noop_without_max_tokens(self):
        composer = _composer()
        composer.prompt_generator = MagicMock()
        turn = Turn()  # no max_tokens resolved
        composer._set_seed_response(turn)
        assert turn.seed_response is None
        composer.prompt_generator.generate_prompt.assert_not_called()

    def test_finalize_sets_seed_response_when_enabled(self):
        composer = _composer()
        composer._seed_enabled = True
        composer.prompt_generator = MagicMock()
        composer.prompt_generator.generate_prompt.return_value = "R"

        turn = Turn(max_tokens=30, model="test-model")
        composer._finalize_turn(turn)

        assert turn.seed_response is not None
        assert turn.seed_response.raw_messages[0]["content"] == "R"
