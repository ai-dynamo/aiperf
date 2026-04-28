# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.loader.spec_bench import SpecBenchLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
async def loader(user_config: UserConfig) -> SpecBenchLoader:
    return SpecBenchLoader(user_config=user_config)


@pytest.mark.asyncio
class TestSpecBenchLoader:
    async def test_preferred_sampling_strategy_is_sequential(self, loader):
        assert (
            loader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_load_dataset_parses_jsonl(self, loader):
        raw_jsonl = (
            '{"question_id": 1, "category": "writing", "turns": ["Write a poem."]}\n'
            '{"question_id": 2, "category": "math", "turns": ["Solve x+1=2."]}\n'
        )
        with patch.object(
            loader, "_load_dataset", new=AsyncMock(return_value=raw_jsonl)
        ):
            result = await loader.load_dataset()

        assert len(result["dataset"]) == 2
        assert result["dataset"][0]["question_id"] == 1
        assert result["dataset"][1]["turns"] == ["Solve x+1=2."]

    async def test_load_dataset_skips_blank_lines(self, loader):
        raw_jsonl = (
            '{"question_id": 1, "turns": ["Hello?"]}\n'
            "\n"
            '{"question_id": 2, "turns": ["World?"]}\n'
        )
        with patch.object(
            loader, "_load_dataset", new=AsyncMock(return_value=raw_jsonl)
        ):
            result = await loader.load_dataset()

        assert len(result["dataset"]) == 2

    async def test_converts_entries_to_conversations(self, loader):
        data = {
            "dataset": [
                {"question_id": 1, "turns": ["Write a travel blog post about Hawaii."]},
                {"question_id": 2, "turns": ["Explain quantum entanglement."]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations) == 2
        assert all(isinstance(c, Conversation) for c in conversations)
        assert (
            conversations[0].turns[0].texts[0].contents[0]
            == "Write a travel blog post about Hawaii."
        )

    async def test_each_entry_becomes_single_turn(self, loader):
        data = {"dataset": [{"turns": ["What is 2+2?"]}]}
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations[0].turns) == 1

    async def test_uses_first_turn_only(self, loader):
        data = {
            "dataset": [{"turns": ["First turn prompt.", "Second follow-up turn."]}]
        }
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "First turn prompt."

    async def test_skips_empty_turns(self, loader):
        data = {
            "dataset": [
                {"turns": [""]},
                {"turns": ["   "]},
                {"turns": ["Valid prompt"]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_skips_missing_turns_key(self, loader):
        data = {
            "dataset": [
                {"question_id": 1},
                {"turns": ["Valid prompt"]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_empty_dataset_returns_empty_list(self, loader):
        data = {"dataset": []}
        conversations = await loader.convert_to_conversations(data)
        assert conversations == []

    async def test_session_ids_are_unique(self, loader):
        data = {"dataset": [{"turns": [f"Question {i}"]} for i in range(5)]}
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5
