# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import SyntheticDataset
from aiperf.config.dataset.content import RankingsConfig
from aiperf.dataset.composer.base import BaseDatasetComposer

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class SyntheticRankingsDatasetComposer(BaseDatasetComposer):
    """Composer that generates synthetic data for the Rankings endpoint.

    Each dataset entry contains one query and multiple passages.
    """

    def __init__(self, *, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs):
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        dataset = run.cfg.get_default_dataset()
        if not isinstance(dataset, SyntheticDataset):
            raise ValueError(
                "SyntheticRankingsDatasetComposer requires a synthetic dataset."
            )
        # If rankings isn't explicitly configured, fall back to
        # ``RankingsConfig``'s field defaults so the composer always sees a
        # fully populated config.
        self._rankings = dataset.rankings or RankingsConfig()
        self._num_entries = dataset.entries

        self.session_id_generator = SessionIDGenerator(
            seed=dataset.random_seed
            if dataset.random_seed is not None
            else run.random_seed
        )
        self._passages_rng = rng.derive("dataset.rankings.passages")
        self._passages_token_rng = rng.derive("dataset.rankings.passages.tokens")
        self._query_token_rng = rng.derive("dataset.rankings.query.tokens")

    def create_dataset(self) -> list[Conversation]:
        """Generate synthetic dataset for the rankings endpoint.

        Each conversation contains one turn with one query and multiple passages.
        """
        conversations: list[Conversation] = []

        for _ in range(self._num_entries):
            num_passages = self._rankings.passages.sample_int(self._passages_rng)
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

        query_text = self.prompt_generator.generate_prompt(
            self._rankings.query_tokens.sample_int(self._query_token_rng)
        )
        query = Text(name="query", contents=[query_text])

        # Generate passages with rankings-specific token counts (per passage)
        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_text = self.prompt_generator.generate_prompt(
                self._rankings.passage_tokens.sample_int(self._passages_token_rng)
            )
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])
        self._finalize_turn(turn)

        self.debug(
            lambda: f"[rankings] query_len={len(query_text)} chars, passages={num_passages}"
        )
        return turn
