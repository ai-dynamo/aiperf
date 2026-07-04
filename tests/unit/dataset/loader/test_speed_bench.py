# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior coverage for the file-based SPEED-Bench JSONL loader.

SPEED-Bench prompts are prepared locally (see ``docs/tutorials/speed-bench.md``)
into JSONL rows of ``{question_id, category, messages}`` where ``messages`` is an
OpenAI-style ``role``/``content`` array. The loader is selected by
``--custom-dataset-type speed_bench_*`` and reads the file pointed at by
``--input-file``, so these tests build real JSONL files on disk and exercise
``load_dataset()`` / ``convert_to_conversations()`` directly.
"""

import json
import logging
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.models import MultiTurn
from aiperf.dataset.loader.speed_bench import (
    SpeedBenchLoader,
    SpeedBenchQualitativeLoader,
    SpeedBenchRow,
    SpeedBenchThroughput1KLoader,
    is_speed_bench_row,
)
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
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a SPEED-Bench row carrying the full auxiliary field set."""
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


def _write_speed_bench_file(create_jsonl_file, rows: list[dict[str, Any]]) -> str:
    return create_jsonl_file([json.dumps(row) for row in rows])


def _load_speed_bench_file(
    create_jsonl_file,
    rows: list[dict[str, Any]],
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


class TestSpeedBenchRow:
    """Strict per-row schema validation (question_id/category/messages)."""

    def test_valid_row_parses(self):
        row = SpeedBenchRow.model_validate(_make_speed_bench_row())
        assert row.question_id == _qid("speed-coding-1")
        assert row.category == "coding"
        assert row.messages[0]["role"] == "user"

    def test_placeholder_constant_present(self):
        assert (
            SpeedBenchRow.TURNS_PLACEHOLDER
            == "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
        )

    @pytest.mark.parametrize(
        "row",
        [
            param(
                {
                    "category": "coding",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                id="missing_question_id",
            ),
            param(
                _make_speed_bench_row(question_id="tooshort"),
                id="question_id_too_short",
            ),
            param(
                _make_speed_bench_row(question_id="x" * 33),
                id="question_id_too_long",
            ),
            param(
                _make_speed_bench_row(category=""),
                id="empty_category",
            ),
            param(
                _make_speed_bench_row(messages=[]),
                id="empty_messages",
            ),
            param(
                _make_speed_bench_row(messages=["not-a-dict"]),
                id="message_not_a_dict",
            ),
            param(
                _make_speed_bench_row(messages=[{"content": "Hi"}]),
                id="message_missing_role",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "", "content": "Hi"}]),
                id="message_empty_role",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "   ", "content": "Hi"}]),
                id="message_whitespace_role",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": 7, "content": "Hi"}]),
                id="message_role_not_str",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "user"}]),
                id="message_missing_content",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "user", "content": ""}]),
                id="message_empty_content",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "user", "content": "   "}]),
                id="message_whitespace_content",
            ),
            param(
                _make_speed_bench_row(messages=[{"role": "user", "content": 42}]),
                id="message_content_not_str",
            ),
        ],
    )  # fmt: skip
    def test_invalid_row_raises(self, row: dict[str, Any]):
        with pytest.raises(ValidationError):
            SpeedBenchRow.model_validate(row)

    def test_placeholder_content_raises_naming_placeholder(self):
        row = _make_speed_bench_row(
            messages=[{"role": "user", "content": SpeedBenchRow.TURNS_PLACEHOLDER}]
        )
        with pytest.raises(ValidationError, match="placeholder"):
            SpeedBenchRow.model_validate(row)

    def test_missing_question_id_names_field(self):
        row = {"category": "coding", "messages": [{"role": "user", "content": "Hi"}]}
        with pytest.raises(ValidationError, match="question_id"):
            SpeedBenchRow.model_validate(row)


class TestIsSpeedBenchRow:
    """Auto-detection helper mirrors SpeedBenchRow validity."""

    def test_valid_row_detected(self):
        assert is_speed_bench_row(_make_speed_bench_row()) is True

    @pytest.mark.parametrize(
        "value",
        [
            param(None, id="none"),
            param("string", id="string"),
            param(123, id="int"),
            param([{"role": "user", "content": "Hi"}], id="list"),
            param({"category": "coding"}, id="missing_required_fields"),
            param(
                _make_speed_bench_row(
                    messages=[{"role": "user", "content": SpeedBenchRow.TURNS_PLACEHOLDER}]
                ),
                id="placeholder_content",
            ),
        ],
    )  # fmt: skip
    def test_non_matching_returns_false(self, value: object):
        assert is_speed_bench_row(value) is False


class TestSpeedBenchCanLoad:
    """can_load auto-detection: base is content-only, splits gate on filename."""

    def test_base_loader_matches_by_content_any_filename(self):
        assert SpeedBenchLoader.can_load(_make_speed_bench_row(), "anything.jsonl")
        assert SpeedBenchLoader.can_load(_make_speed_bench_row(), None)

    def test_base_loader_rejects_non_speed_bench(self):
        assert not SpeedBenchLoader.can_load({"session_id": "x", "turns": []})

    def test_split_loader_requires_matching_filename(self):
        row = _make_speed_bench_row()
        assert SpeedBenchQualitativeLoader.can_load(row, "qualitative.jsonl")
        assert SpeedBenchQualitativeLoader.can_load(row, "/data/qualitative.jsonl")
        assert not SpeedBenchQualitativeLoader.can_load(row, "throughput_1k.jsonl")
        assert not SpeedBenchQualitativeLoader.can_load(row, None)

    def test_split_loader_rejects_non_speed_bench_even_with_matching_name(self):
        assert not SpeedBenchThroughput1KLoader.can_load(
            {"session_id": "x", "turns": []}, "throughput_1k.jsonl"
        )


class TestSpeedBenchLoader:
    def test_preferred_sampling_strategy_is_sequential(self):
        assert (
            SpeedBenchLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_constructor_stores_category_and_multi_turn(self, create_jsonl_file):
        filename = _write_speed_bench_file(create_jsonl_file, [_make_speed_bench_row()])
        loader = SpeedBenchLoader(
            run=_make_run(), filename=filename, category="coding", multi_turn=False
        )
        assert loader.category == "coding"
        assert loader.multi_turn is False

    def test_multi_turn_defaults_true(self, create_jsonl_file):
        filename = _write_speed_bench_file(create_jsonl_file, [_make_speed_bench_row()])
        loader = SpeedBenchLoader(run=_make_run(), filename=filename)
        assert loader.multi_turn is True

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

    def test_category_filter_no_matches_warns(self, create_jsonl_file, caplog):
        filename = _write_speed_bench_file(
            create_jsonl_file,
            [_make_speed_bench_row(question_id=_qid("speed-math-1"), category="math")],
        )
        loader = SpeedBenchLoader(run=_make_run(), filename=filename, category="coding")

        with caplog.at_level(logging.WARNING):
            dataset = loader.load_dataset()

        assert dict(dataset) == {}
        assert any("matched no rows" in record.message for record in caplog.records), (
            f"expected empty-match warning, got: {[r.message for r in caplog.records]}"
        )

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

        assert set(dataset) == {_qid("speed-coding-1")}
        turns = dataset[_qid("speed-coding-1")][0].turns
        assert len(turns) == 2
        assert [turn.text for turn in turns] == ["Code Q1", "Code Q2"]
