# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.models import MultiTurn
from aiperf.dataset.loader.speed_bench import SpeedBenchLoader, SpeedBenchRow
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.conftest import make_run_from_cli


def _make_run():
    return make_run_from_cli(CLIConfig(model_names=["test-model"]))


def _qid(label: str) -> str:
    """Pad a short label to a 32-char question_id (SpeedBenchRow constraint)."""
    return label.ljust(32, "0")


def _make_speed_bench_row(
    question_id: str | None = None,
    category: str = "coding",
    messages: list[dict[str, str]] | None = None,
) -> dict:
    if question_id is None:
        question_id = _qid("speed-coding-1")
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
    rows: list[dict],
    category: str | None = None,
    multi_turn: bool = True,
):
    filename = _write_speed_bench_file(create_jsonl_file, rows)
    loader = SpeedBenchLoader(
        run=_make_run(),
        filename=filename,
        category=category,
        multi_turn=multi_turn,
    )
    return loader, loader.load_dataset()


class TestSpeedBenchLoader:
    def test_preferred_sampling_strategy_is_sequential(self):
        assert (
            SpeedBenchLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_loads_single_speed_bench_jsonl_row(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"),
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

        assert set(dataset) == {_qid("speed-coding-1")}

        multi_turn = dataset[_qid("speed-coding-1")][0]
        assert isinstance(multi_turn, MultiTurn)
        assert multi_turn.session_id == _qid("speed-coding-1")
        assert len(multi_turn.turns) == 1
        assert multi_turn.turns[0].role == "user"
        assert multi_turn.turns[0].text == (
            "Write a Python function that flips letter case."
        )

    def test_loads_each_jsonl_row_as_separate_session(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"),
                    category="coding",
                    messages=[{"role": "user", "content": "Implement merge sort."}],
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-math-1"),
                    category="math",
                    messages=[{"role": "user", "content": "Find the factors of 84."}],
                ),
            ],
        )

        assert set(dataset) == {_qid("speed-coding-1"), _qid("speed-math-1")}
        assert dataset[_qid("speed-coding-1")][0].session_id == _qid("speed-coding-1")
        assert dataset[_qid("speed-math-1")][0].session_id == _qid("speed-math-1")
        assert (
            dataset[_qid("speed-coding-1")][0].turns[0].text == "Implement merge sort."
        )
        assert (
            dataset[_qid("speed-math-1")][0].turns[0].text == "Find the factors of 84."
        )

    def test_loads_all_messages_in_order(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-chat-1"),
                    category="qa",
                    messages=[
                        {"role": "system", "content": "Answer tersely."},
                        {"role": "user", "content": "What is Python?"},
                    ],
                )
            ],
        )

        turns = dataset[_qid("speed-chat-1")][0].turns
        assert [turn.role for turn in turns] == ["system", "user"]
        assert [turn.text for turn in turns] == ["Answer tersely.", "What is Python?"]

    def test_blank_lines_are_skipped(self, create_jsonl_file):
        row = _make_speed_bench_row(question_id=_qid("speed-coding-1"))
        filename = create_jsonl_file(["", json.dumps(row), "   "])
        loader = SpeedBenchLoader(filename=filename, run=_make_run())

        dataset = loader.load_dataset()

        assert set(dataset) == {_qid("speed-coding-1")}

    def test_empty_file_returns_empty_dataset(self, create_jsonl_file):
        filename = create_jsonl_file([])
        loader = SpeedBenchLoader(filename=filename, run=_make_run())

        assert dict(loader.load_dataset()) == {}

    def test_converts_loaded_dataset_to_conversations(self, create_jsonl_file):
        loader, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-chat-1"),
                    category="qa",
                    messages=[
                        {"role": "system", "content": "Answer tersely."},
                        {"role": "user", "content": "What is Python?"},
                    ],
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"),
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
        chat_conversation = conversations_by_id[_qid("speed-chat-1")]
        assert len(chat_conversation.turns) == 2
        assert chat_conversation.turns[0].role == "system"
        assert chat_conversation.turns[0].texts[0].contents == ["Answer tersely."]
        assert chat_conversation.turns[1].role == "user"
        assert chat_conversation.turns[1].texts[0].contents == ["What is Python?"]

        coding_conversation = conversations_by_id[_qid("speed-coding-1")]
        assert len(coding_conversation.turns) == 1
        assert coding_conversation.turns[0].role == "user"
        assert coding_conversation.turns[0].texts[0].contents == [
            "Implement quicksort."
        ]


class TestSpeedBenchLoaderCategoryFiltering:
    def test_no_category_returns_all_rows(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"), category="coding"
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-math-1"), category="math"
                ),
            ],
        )

        assert set(dataset) == {_qid("speed-coding-1"), _qid("speed-math-1")}

    def test_category_filter_returns_matching_rows(self, create_jsonl_file):
        loader, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"), category="coding"
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-math-1"), category="math"
                ),
            ],
            category="coding",
        )

        assert loader.category == "coding"
        assert set(dataset) == {_qid("speed-coding-1")}
        assert (
            dataset[_qid("speed-coding-1")][0].turns[0].text
            == "Implement binary search."
        )

    def test_category_filter_no_matches_returns_empty(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-math-1"), category="math"
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-stem-1"), category="stem"
                ),
            ],
            category="coding",
        )

        assert dict(dataset) == {}

    def test_category_stored_on_loader(self, create_jsonl_file):
        unfiltered_loader, _ = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"), category="coding"
                )
            ],
        )
        filtered_loader, _ = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"), category="coding"
                )
            ],
            category="coding",
        )

        assert unfiltered_loader.category is None
        assert filtered_loader.category == "coding"

    def test_throughput_entropy_tier_filtering(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-low-entropy-1"),
                    category="low_entropy",
                    messages=[{"role": "user", "content": "Complete the code sample."}],
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-high-entropy-1"),
                    category="high_entropy",
                    messages=[
                        {"role": "user", "content": "Continue this novel excerpt."}
                    ],
                ),
            ],
            category="low_entropy",
        )

        assert set(dataset) == {_qid("speed-low-entropy-1")}
        assert dataset[_qid("speed-low-entropy-1")][0].turns[0].text == (
            "Complete the code sample."
        )


class TestSpeedBenchLoaderRowValidation:
    def test_load_dataset_missing_question_id_raises_validation_error_naming_field(
        self, create_jsonl_file
    ):
        malformed_row = {
            "category": "coding",
            "messages": [{"role": "user", "content": "Implement binary search."}],
        }
        filename = _write_speed_bench_file(create_jsonl_file, [malformed_row])
        loader = SpeedBenchLoader(filename=filename, run=_make_run())

        with pytest.raises(ValidationError, match="question_id"):
            loader.load_dataset()

    def test_load_dataset_rejects_placeholder_content(self, create_jsonl_file):
        placeholder_row = _make_speed_bench_row(
            question_id="0123456789abcdef0123456789abcdef",
            category="coding",
            messages=[{"role": "user", "content": SpeedBenchRow.TURNS_PLACEHOLDER}],
        )
        filename = _write_speed_bench_file(create_jsonl_file, [placeholder_row])
        loader = SpeedBenchLoader(filename=filename, run=_make_run())

        with pytest.raises(ValidationError, match="placeholder"):
            loader.load_dataset()


class TestSpeedBenchLoaderMultiTurn:
    def test_multi_turn_produces_all_messages(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-chat-1"),
                    messages=[
                        {"role": "user", "content": "First turn"},
                        {"role": "assistant", "content": "Second turn"},
                        {"role": "user", "content": "Third turn"},
                    ],
                )
            ],
            multi_turn=True,
        )

        turns = dataset[_qid("speed-chat-1")][0].turns
        assert len(turns) == 3
        assert [turn.role for turn in turns] == ["user", "assistant", "user"]
        assert [turn.text for turn in turns] == [
            "First turn",
            "Second turn",
            "Third turn",
        ]

    def test_multi_turn_false_loads_first_message_only(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-chat-1"),
                    messages=[
                        {"role": "user", "content": "First turn"},
                        {"role": "user", "content": "Second turn"},
                    ],
                )
            ],
            multi_turn=False,
        )

        turns = dataset[_qid("speed-chat-1")][0].turns
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].text == "First turn"

    def test_multi_turn_with_category_filter(self, create_jsonl_file):
        _, dataset = _load_speed_bench_file(
            create_jsonl_file,
            [
                _make_speed_bench_row(
                    question_id=_qid("speed-coding-1"),
                    category="coding",
                    messages=[
                        {"role": "user", "content": "Code Q1"},
                        {"role": "user", "content": "Code Q2"},
                    ],
                ),
                _make_speed_bench_row(
                    question_id=_qid("speed-math-1"),
                    category="math",
                    messages=[{"role": "user", "content": "Math Q1"}],
                ),
            ],
            category="coding",
            multi_turn=True,
        )

        turns = dataset[_qid("speed-coding-1")][0].turns
        assert set(dataset) == {_qid("speed-coding-1")}
        assert len(turns) == 2
        assert [turn.text for turn in turns] == ["Code Q1", "Code Q2"]
