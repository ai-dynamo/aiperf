# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums import ComposerType
from aiperf.common.factories import ComposerFactory
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer


@ComposerFactory.register(ComposerType.SYNTHETIC_RANKINGS)
class SyntheticRankingsDatasetComposer(BaseDatasetComposer):
    """Composer that generates synthetic data for the Rankings endpoint.

    Each dataset entry contains one query and multiple passages.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)
        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)
        self._ranking_rng = rng.derive("composer.rankings")

        if not self.include_prompt:
            raise ValueError(
                "Synthetic rankings data generation requires text prompts to be enabled. "
                "Please set --prompt-input-tokens-mean > 0."
            )

    def create_dataset(self) -> list[Conversation]:
        """Generate synthetic dataset for the rankings endpoint.

        Each conversation contains one turn with one query and multiple passages.
        """
        conversations = []
        num_entries = self.config.input.conversation.num_dataset_entries
        num_passages = self.config.input.prompt.batch_size

        for _ in range(num_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())
            turn = self._create_turn(num_passages=num_passages)
            conversation.turns.append(turn)
            conversations.append(conversation)

        return conversations

    def _create_turn(self, num_passages: int) -> Turn:
        """Create a single ranking turn with one synthetic query and multiple synthetic passages."""
        turn = Turn()

        query_text = self.prompt_generator.generate(
            mean=self.config.input.prompt.input_tokens.mean,
            stddev=self.config.input.prompt.input_tokens.stddev,
        )
        query = Text(name="query", contents=[query_text])

        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_text = self.prompt_generator.generate(
                mean=self.config.input.prompt.input_tokens.mean,
                stddev=self.config.input.prompt.input_tokens.stddev,
            )
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])
        self._finalize_turn(turn)
        return turn

    @property
    def include_prompt(self) -> bool:
        """Rankings require prompts to be enabled."""
        return self.config.input.prompt.input_tokens.mean > 0
