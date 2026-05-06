# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from aiperf.common.models import Conversation
from aiperf.dataset.loader.models import MultiTurn
from aiperf.dataset.loader.speed_bench import SpeedBenchLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


def _make_speed_bench_row(
    question_id: str = "speed-coding-1",
    category: str = "coding",
    messages: list[dict[str, str]] | None = None,
) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": "Implement binary search."}]

    return {
        "question_id": question_id,
        "category": category,
        "sub_category": None,
        "source": "https://example.test/speed-bench",
        "src_id": question_id,
        "difficulty": None,
        "multiturn": len(messages) > 1,
        "messages": messages,
    }


def _write_speed_bench_file(create_jsonl_file, rows: list[dict]) -> str:
    return create_jsonl_file([json.dumps(row) for row in rows])


def _load_speed_bench_file(
    create_jsonl_file,
    default_user_config,
    rows: list[dict],
    category: str | None = None,
):
    filename = _write_speed_bench_file(create_jsonl_file, rows)
    loader = SpeedBenchLoader(
        filename=filename,
        user_config=default_user_config,
        category=category,
    )
    return loader, loader.load_dataset()


class TestSpeedBenchLoader:
    def test_preferred_sampling_strategy_is_sequential(self):
        assert (
            SpeedBenchLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_loads_single_speed_bench_jsonl_row(
        self, create_jsonl_file, default_user_config
    ):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(
                    question_id="speed-coding-1",
                    category="coding",
                    messages=[
                        {
                            "role": "user",
                            "content": "Write a Python function that flips letter case.",
                        }
                    ],
                )
            ],
        )

        assert set(dataset) == {"speed-coding-1"}

        multi_turn = dataset["speed-coding-1"][0]
        assert isinstance(multi_turn, MultiTurn)
        assert multi_turn.session_id == "speed-coding-1"
        assert len(multi_turn.turns) == 1
        assert multi_turn.turns[0].role == "user"
        assert multi_turn.turns[0].text == (
            "Write a Python function that flips letter case."
        )

    def test_loads_each_jsonl_row_as_separate_session(
        self, create_jsonl_file, default_user_config
    ):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(
                    question_id="speed-coding-1",
                    category="coding",
                    messages=[{"role": "user", "content": "Implement merge sort."}],
                ),
                _make_speed_bench_row(
                    question_id="speed-math-1",
                    category="math",
                    messages=[{"role": "user", "content": "Find the factors of 84."}],
                ),
            ],
        )

        assert set(dataset) == {"speed-coding-1", "speed-math-1"}
        assert dataset["speed-coding-1"][0].session_id == "speed-coding-1"
        assert dataset["speed-math-1"][0].session_id == "speed-math-1"
        assert dataset["speed-coding-1"][0].turns[0].text == "Implement merge sort."
        assert dataset["speed-math-1"][0].turns[0].text == "Find the factors of 84."

    def test_loads_all_messages_in_order(self, create_jsonl_file, default_user_config):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(
                    question_id="speed-chat-1",
                    category="qa",
                    messages=[
                        {"role": "system", "content": "Answer tersely."},
                        {"role": "user", "content": "What is Python?"},
                    ],
                )
            ],
        )

        turns = dataset["speed-chat-1"][0].turns
        assert [turn.role for turn in turns] == ["system", "user"]
        assert [turn.text for turn in turns] == ["Answer tersely.", "What is Python?"]

    def test_blank_lines_are_skipped(self, create_jsonl_file, default_user_config):
        row = _make_speed_bench_row(question_id="speed-coding-1")
        filename = create_jsonl_file(["", json.dumps(row), "   "])
        loader = SpeedBenchLoader(filename=filename, user_config=default_user_config)

        dataset = loader.load_dataset()

        assert set(dataset) == {"speed-coding-1"}

    def test_empty_file_returns_empty_dataset(
        self, create_jsonl_file, default_user_config
    ):
        filename = create_jsonl_file([])
        loader = SpeedBenchLoader(filename=filename, user_config=default_user_config)

        assert dict(loader.load_dataset()) == {}

    def test_converts_loaded_dataset_to_conversations(
        self, create_jsonl_file, default_user_config
    ):
        loader, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(
                    question_id="speed-chat-1",
                    category="qa",
                    messages=[
                        {"role": "system", "content": "Answer tersely."},
                        {"role": "user", "content": "What is Python?"},
                    ],
                ),
                _make_speed_bench_row(
                    question_id="speed-coding-1",
                    category="coding",
                    messages=[{"role": "user", "content": "Implement quicksort."}],
                ),
            ],
        )

        conversations = loader.convert_to_conversations(dataset)

        assert len(conversations) == 2
        assert all(
            isinstance(conversation, Conversation) for conversation in conversations
        )

        conversations_by_id = {
            conversation.session_id: conversation for conversation in conversations
        }
        chat_conversation = conversations_by_id["speed-chat-1"]
        assert len(chat_conversation.turns) == 2
        assert chat_conversation.turns[0].role == "system"
        assert chat_conversation.turns[0].texts[0].contents == ["Answer tersely."]
        assert chat_conversation.turns[1].role == "user"
        assert chat_conversation.turns[1].texts[0].contents == ["What is Python?"]

        coding_conversation = conversations_by_id["speed-coding-1"]
        assert len(coding_conversation.turns) == 1
        assert coding_conversation.turns[0].role == "user"
        assert coding_conversation.turns[0].texts[0].contents == [
            "Implement quicksort."
        ]


class TestSpeedBenchLoaderCategoryFiltering:
    def test_no_category_returns_all_rows(self, create_jsonl_file, default_user_config):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(question_id="speed-coding-1", category="coding"),
                _make_speed_bench_row(question_id="speed-math-1", category="math"),
            ],
        )

        assert set(dataset) == {"speed-coding-1", "speed-math-1"}

    def test_category_filter_returns_matching_rows(
        self, create_jsonl_file, default_user_config
    ):
        loader, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(question_id="speed-coding-1", category="coding"),
                _make_speed_bench_row(question_id="speed-math-1", category="math"),
            ],
            category="coding",
        )

        assert loader.category == "coding"
        assert set(dataset) == {"speed-coding-1"}
        assert dataset["speed-coding-1"][0].turns[0].text == "Implement binary search."

    def test_category_filter_no_matches_returns_empty(
        self, create_jsonl_file, default_user_config
    ):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(question_id="speed-math-1", category="math"),
                _make_speed_bench_row(question_id="speed-stem-1", category="stem"),
            ],
            category="coding",
        )

        assert dict(dataset) == {}

    def test_category_stored_on_loader(self, create_jsonl_file, default_user_config):
        unfiltered_loader, _ = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [_make_speed_bench_row(question_id="speed-coding-1", category="coding")],
        )
        filtered_loader, _ = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [_make_speed_bench_row(question_id="speed-coding-1", category="coding")],
            category="coding",
        )

        assert unfiltered_loader.category is None
        assert filtered_loader.category == "coding"

    def test_throughput_entropy_tier_filtering(
        self, create_jsonl_file, default_user_config
    ):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            default_user_config,
            [
                _make_speed_bench_row(
                    question_id="speed-low-entropy-1",
                    category="low_entropy",
                    messages=[{"role": "user", "content": "Complete the code sample."}],
                ),
                _make_speed_bench_row(
                    question_id="speed-high-entropy-1",
                    category="high_entropy",
                    messages=[
                        {"role": "user", "content": "Continue this novel excerpt."}
                    ],
                ),
            ],
            category="low_entropy",
        )

        assert set(dataset) == {"speed-low-entropy-1"}
        assert dataset["speed-low-entropy-1"][0].turns[0].text == (
            "Complete the code sample."
        )
