# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common import random_generator as rng
from aiperf.common.config import InputDefaults, UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer


class SyntheticRankingsDatasetComposer(BaseDatasetComposer):
    """Composer that generates synthetic data for the Rankings endpoint.

    Each dataset entry contains one query and multiple passages.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)
        self._passages_rng = rng.derive("dataset.rankings.passages")
        self._passages_token_rng = rng.derive("dataset.rankings.passages.tokens")
        self._query_token_rng = rng.derive("dataset.rankings.query.tokens")

        # Set default sampling strategy for synthetic rankings dataset if not explicitly set
        if self.config.input.dataset_sampling_strategy is None:
            self.config.input.dataset_sampling_strategy = (
                InputDefaults.DATASET_SAMPLING_STRATEGY
            )
            self.info(
                f"Using default sampling strategy for synthetic rankings dataset: {InputDefaults.DATASET_SAMPLING_STRATEGY}"
            )

    def create_dataset(self) -> list[Conversation]:
        """Generate synthetic dataset for the rankings endpoint.

        Each conversation contains one turn with one query and multiple passages.
        """
        conversations: list[Conversation] = []
        num_entries = self.config.input.conversation.num_dataset_entries
        num_passages_mean = self.config.input.rankings.passages.mean
        num_passages_std = self.config.input.rankings.passages.stddev

        for _ in range(num_entries):
            num_passages = self._passages_rng.sample_positive_normal_integer(
                num_passages_mean, num_passages_std
            )
            conversation = Conversation(session_id=self.session_id_generator.next())
            turn = self._create_turn(num_passages=num_passages)
            conversation.turns.append(turn)
            conversations.append(conversation)

        return conversations

    def _create_turn(self, num_passages: int) -> Turn:
        """Create a single ranking turn with one synthetic query and multiple synthetic passages."""
        turn = Turn()

        query_text = self.prompt_generator.generate_prompt(
            self.prompt_generator.calculate_num_tokens(
                self.config.input.rankings.query.prompt_token_mean,
                self.config.input.rankings.query.prompt_token_stddev,
            )
        )
        query = Text(name="query", contents=[query_text])

        # Generate passages with rankings-specific token counts (per passage)
        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_text = self.prompt_generator.generate_prompt(
                self.prompt_generator.calculate_num_tokens(
                    self.config.input.rankings.passages.prompt_token_mean,
                    self.config.input.rankings.passages.prompt_token_stddev,
                )
            )
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])
        self._finalize_turn(turn)

        self.debug(
            lambda: f"[rankings] query_len={len(query_text)} chars, passages={num_passages}"
        )
        return turn
