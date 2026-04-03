# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.loader.speed_bench import SpeedBenchLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
async def loader(user_config: UserConfig) -> SpeedBenchLoader:
    return SpeedBenchLoader(
        user_config=user_config,
        hf_dataset_name="nvidia/SPEED-Bench",
        hf_split="test",
        hf_subset="qualitative",
    )


@pytest.fixture
async def coding_loader(user_config: UserConfig) -> SpeedBenchLoader:
    return SpeedBenchLoader(
        user_config=user_config,
        hf_dataset_name="nvidia/SPEED-Bench",
        hf_split="test",
        hf_subset="qualitative",
        category="coding",
    )


def _make_speed_bench_row(
    turns: list[str | None],
    category: str = "coding",
    question_id: str = "q1",
) -> dict:
    return {
        "question_id": question_id,
        "category": category,
        "turns": turns,
    }


@pytest.mark.asyncio
class TestSpeedBenchLoader:
    async def test_preferred_sampling_strategy_is_sequential(self, loader):
        assert (
            loader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_converts_single_turn_row(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["What is Python?"]),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "What is Python?"

    async def test_uses_first_turn_only(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["First turn", "Second turn", "Third turn"]),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "First turn"

    async def test_each_row_becomes_one_conversation(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row([f"Question {i}"], question_id=f"q{i}")
                for i in range(5)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 5
        assert all(isinstance(c, Conversation) for c in conversations)

    async def test_session_ids_are_unique(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row([f"Q{i}"], question_id=f"q{i}") for i in range(5)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_strips_whitespace_from_prompt(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["  What is Python?  "]),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == "What is Python?"

    async def test_empty_dataset_returns_empty_list(self, loader):
        conversations = await loader.convert_to_conversations({"dataset": []})
        assert conversations == []

    @pytest.mark.parametrize(
        "invalid_row",
        [
            pytest.param(
                {"question_id": "q1", "category": "coding"}, id="missing_turns"
            ),
            pytest.param(
                {"question_id": "q1", "category": "coding", "turns": None},
                id="none_turns",
            ),
            pytest.param(
                {"question_id": "q1", "category": "coding", "turns": "not a list"},
                id="non_list_turns",
            ),
            pytest.param(_make_speed_bench_row([]), id="empty_turns_list"),
            pytest.param(_make_speed_bench_row([""]), id="empty_first_turn"),
            pytest.param(_make_speed_bench_row(["   "]), id="whitespace_first_turn"),
            pytest.param(_make_speed_bench_row([None]), id="none_first_turn"),
        ],
    )
    async def test_skips_invalid_row(self, loader, invalid_row):
        data = {
            "dataset": [
                invalid_row,
                _make_speed_bench_row(["Valid"]),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"


@pytest.mark.asyncio
class TestSpeedBenchLoaderCategoryFiltering:
    async def test_no_category_returns_all_rows(self, loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["Code question"], category="coding"),
                _make_speed_bench_row(["Math question"], category="math"),
                _make_speed_bench_row(["Writing prompt"], category="writing"),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 3

    async def test_category_filter_returns_matching_rows(self, coding_loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["Code question"], category="coding"),
                _make_speed_bench_row(["Math question"], category="math"),
                _make_speed_bench_row(["Another code Q"], category="coding"),
                _make_speed_bench_row(["Writing prompt"], category="writing"),
            ]
        }
        conversations = await coding_loader.convert_to_conversations(data)
        assert len(conversations) == 2
        assert conversations[0].turns[0].texts[0].contents[0] == "Code question"
        assert conversations[1].turns[0].texts[0].contents[0] == "Another code Q"

    async def test_category_filter_no_matches_returns_empty(self, coding_loader):
        data = {
            "dataset": [
                _make_speed_bench_row(["Math Q"], category="math"),
                _make_speed_bench_row(["Writing Q"], category="writing"),
            ]
        }
        conversations = await coding_loader.convert_to_conversations(data)
        assert conversations == []

    async def test_category_filter_with_empty_turns_skipped(self, coding_loader):
        data = {
            "dataset": [
                _make_speed_bench_row([], category="coding"),
                _make_speed_bench_row([""], category="coding"),
                _make_speed_bench_row(["Valid code Q"], category="coding"),
            ]
        }
        conversations = await coding_loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid code Q"

    async def test_category_stored_on_loader(self, coding_loader, loader):
        assert coding_loader.category == "coding"
        assert loader.category is None

    async def test_throughput_entropy_tier_filtering(self, user_config):
        low_entropy_loader = SpeedBenchLoader(
            user_config=user_config,
            hf_dataset_name="nvidia/SPEED-Bench",
            hf_split="test",
            hf_subset="throughput_1k",
            category="low_entropy",
        )
        data = {
            "dataset": [
                _make_speed_bench_row(["Code sort"], category="low_entropy"),
                _make_speed_bench_row(["Creative writing"], category="high_entropy"),
                _make_speed_bench_row(["More code"], category="low_entropy"),
                _make_speed_bench_row(["Exam question"], category="mixed"),
            ]
        }
        conversations = await low_entropy_loader.convert_to_conversations(data)
        assert len(conversations) == 2


@pytest.mark.asyncio
class TestSpeedBenchLoaderStreaming:
    async def test_non_streaming_returns_all_rows(self, user_config):
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=3),
        )
        loader = SpeedBenchLoader(
            user_config=config,
            hf_dataset_name="nvidia/SPEED-Bench",
            hf_split="test",
            hf_subset="qualitative",
            streaming=False,
        )
        data = {
            "dataset": [
                _make_speed_bench_row([f"Q{i}"], question_id=f"q{i}") for i in range(10)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 10

    async def test_streaming_capped_by_request_count(self, user_config):
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=3),
        )
        loader = SpeedBenchLoader(
            user_config=config,
            hf_dataset_name="nvidia/SPEED-Bench",
            hf_split="test",
            hf_subset="qualitative",
            streaming=True,
        )
        data = {
            "dataset": [
                _make_speed_bench_row([f"Q{i}"], question_id=f"q{i}") for i in range(10)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 3

    async def test_streaming_with_category_filter(self, user_config):
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=2),
        )
        loader = SpeedBenchLoader(
            user_config=config,
            hf_dataset_name="nvidia/SPEED-Bench",
            hf_split="test",
            hf_subset="qualitative",
            category="coding",
            streaming=True,
        )
        data = {
            "dataset": [
                _make_speed_bench_row(
                    [f"Code {i}"], category="coding", question_id=f"c{i}"
                )
                for i in range(5)
            ]
            + [
                _make_speed_bench_row(
                    [f"Math {i}"], category="math", question_id=f"m{i}"
                )
                for i in range(5)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 2
        assert conversations[0].turns[0].texts[0].contents[0] == "Code 0"
        assert conversations[1].turns[0].texts[0].contents[0] == "Code 1"
