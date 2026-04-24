# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class SyntheticRankingsDatasetComposer(BaseDatasetComposer):
    """Composer that generates synthetic data for the Rankings endpoint.

    Each dataset entry contains one query and multiple passages.
    """

    def __init__(self, run: BenchmarkRun, tokenizer: Tokenizer | None):
        super().__init__(run, tokenizer)

        # Use dataset-specific random seed or fall back to global
        seed = self.dataset_config.random_seed or run.cfg.random_seed
        self.session_id_generator = SessionIDGenerator(seed=seed)
        self._passages_rng = rng.derive("dataset.rankings.passages")
        self._passages_token_rng = rng.derive("dataset.rankings.passages.tokens")
        self._query_token_rng = rng.derive("dataset.rankings.query.tokens")

    def create_dataset(self) -> list[Conversation]:
        """Generate synthetic dataset for the rankings endpoint.

        Each conversation contains one turn with one query and multiple passages.
        """
        conversations: list[Conversation] = []
        num_entries = self.dataset_config.entries

        # Get rankings config
        rankings_config = getattr(self.dataset_config, "rankings", None)
        if rankings_config is None:
            raise ValueError(
                "Rankings config is required for synthetic rankings dataset"
            )

        for _ in range(num_entries):
            num_passages = rankings_config.passages.sample_int(self._passages_rng)
            conversation = Conversation(session_id=self.session_id_generator.next())
            turn = self._create_turn(num_passages=num_passages)
            conversation.turns.append(turn)
            conversations.append(conversation)

        return conversations

    def _create_turn(self, num_passages: int) -> Turn:
        """Create a single ranking turn with one synthetic query and multiple synthetic passages.

        Raises:
            ValueError: If prompt_generator is not available (tokenizer was not configured).
        """
        if self.prompt_generator is None:
            raise ValueError(
                "Rankings dataset generation requires a tokenizer. Either provide a "
                "--tokenizer or use an endpoint that supports tokenization."
            )

        turn = Turn()

        # Get rankings config
        rankings_config = getattr(self.dataset_config, "rankings", None)
        if rankings_config is None:
            raise ValueError(
                "Rankings config is required for synthetic rankings dataset"
            )

        query_num_tokens = rankings_config.query_tokens.sample_int(
            self._query_token_rng
        )
        query_text = self.prompt_generator.generate_prompt(query_num_tokens)
        query = Text(name="query", contents=[query_text])

        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_num_tokens = rankings_config.passage_tokens.sample_int(
                self._passages_token_rng
            )
            passage_text = self.prompt_generator.generate_prompt(passage_num_tokens)
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])
        self._finalize_turn(turn)

        self.debug(
            lambda: f"[rankings] query_len={len(query_text)} chars, passages={num_passages}"
        )
        return turn
