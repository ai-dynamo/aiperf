# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.mt_bench import MTBenchDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def cli_config() -> CLIConfig:
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
async def loader(cli_config: CLIConfig) -> MTBenchDatasetLoader:
    return MTBenchDatasetLoader(
        run=make_run_from_cli(cli_config),
        hf_dataset_name="HuggingFaceH4/mt_bench_prompts",
    )


@pytest.mark.asyncio
class TestMTBenchDatasetLoader:
    async def test_preferred_sampling_strategy_is_sequential(self, loader):
        assert (
            loader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_two_turn_row_becomes_two_user_turns_in_order(self, loader):
        data = {
            "dataset": [
                {
                    "category": "writing",
                    "prompt": [
                        "Write a travel blog about Hawaii.",
                        "Now start every sentence with A.",
                    ],
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)
        turns = conversations[0].turns
        assert len(turns) == 2
        assert turns[0].texts[0].contents[0] == "Write a travel blog about Hawaii."
        assert turns[1].texts[0].contents[0] == "Now start every sentence with A."

    async def test_turns_carry_no_role(self, loader):
        # The dispatch path treats dataset turns as user turns and injects the
        # live assistant reply between them; no role is set (matches SpecBench).
        data = {"dataset": [{"prompt": ["Turn one.", "Turn two."]}]}
        conversations = await loader.convert_to_conversations(data)
        assert all(turn.role is None for turn in conversations[0].turns)

    async def test_blank_turn_is_skipped(self, loader):
        data = {"dataset": [{"prompt": ["Real turn.", "   ", "Another turn."]}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 2
        assert conversations[0].turns[0].texts[0].contents[0] == "Real turn."
        assert conversations[0].turns[1].texts[0].contents[0] == "Another turn."

    async def test_null_values_in_prompt_are_skipped(self, loader):
        data = {"dataset": [{"prompt": [None, "Valid turn", None]}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid turn"

    async def test_row_with_all_empty_turns_is_skipped(self, loader):
        data = {"dataset": [{"prompt": ["", "   "]}, {"prompt": ["Valid"]}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"

    async def test_non_list_prompt_is_skipped(self, loader):
        data = {"dataset": [{"prompt": "not a list"}, {"prompt": ["Valid"]}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"

    async def test_empty_prompt_list_is_skipped(self, loader):
        data = {"dataset": [{"prompt": []}, {"prompt": ["Valid"]}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_missing_prompt_column_raises_on_first_row(self, loader):
        data = {"dataset": [{"category": "writing", "turns": ["wrong key"]}]}
        with pytest.raises(DatasetLoaderError, match="prompt"):
            await loader.convert_to_conversations(data)

    async def test_session_ids_are_unique(self, loader):
        data = {"dataset": [{"prompt": [f"Q{i}", f"Follow {i}"]} for i in range(5)]}
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_empty_dataset_returns_empty_list(self, loader):
        conversations = await loader.convert_to_conversations({"dataset": []})
        assert conversations == []
