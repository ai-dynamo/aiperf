# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior coverage for the SPEED-Bench HuggingFace loader.

SPEED-Bench ships JSONL rows of ``{question_id, category, messages}`` where
``messages`` is an OpenAI-style ``role``/``content`` array. The loader is
selected by name through the public-dataset composer (HF-hub transport), so
these tests exercise ``convert_to_conversations({"dataset": [...]})`` directly
with in-memory rows rather than hitting HuggingFace.
"""

from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.speed_bench import (
    SpeedBenchLoader,
    SpeedBenchRow,
    is_speed_bench_row,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.conftest import make_run_from_cli


def _qid(label: str) -> str:
    """Pad a short label to a 32-char question_id (SpeedBenchRow constraint)."""
    return label.ljust(32, "0")


def _make_row(
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


def _make_loader(
    category: str | None = None, multi_turn: bool = True
) -> SpeedBenchLoader:
    """Construct a loader; must be called inside a running event loop."""
    return SpeedBenchLoader(
        run=make_run_from_cli(CLIConfig(model_names=["test-model"])),
        hf_dataset_name="nvidia/SPEED-Bench",
        hf_split="test",
        category=category,
        multi_turn=multi_turn,
    )


async def _convert(
    loader: SpeedBenchLoader, rows: list[dict[str, Any]]
) -> list[Conversation]:
    return await loader.convert_to_conversations({"dataset": rows})


class TestSpeedBenchRow:
    """Strict per-row schema validation (question_id/category/messages)."""

    def test_valid_row_parses(self):
        row = SpeedBenchRow.model_validate(_make_row())
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
                _make_row(question_id="tooshort"),
                id="question_id_too_short",
            ),
            param(
                _make_row(question_id="x" * 33),
                id="question_id_too_long",
            ),
            param(
                _make_row(category=""),
                id="empty_category",
            ),
            param(
                _make_row(messages=[]),
                id="empty_messages",
            ),
            param(
                _make_row(messages=["not-a-dict"]),
                id="message_not_a_dict",
            ),
            param(
                _make_row(messages=[{"content": "Hi"}]),
                id="message_missing_role",
            ),
            param(
                _make_row(messages=[{"role": "", "content": "Hi"}]),
                id="message_empty_role",
            ),
            param(
                _make_row(messages=[{"role": "   ", "content": "Hi"}]),
                id="message_whitespace_role",
            ),
            param(
                _make_row(messages=[{"role": 7, "content": "Hi"}]),
                id="message_role_not_str",
            ),
            param(
                _make_row(messages=[{"role": "user"}]),
                id="message_missing_content",
            ),
            param(
                _make_row(messages=[{"role": "user", "content": ""}]),
                id="message_empty_content",
            ),
            param(
                _make_row(messages=[{"role": "user", "content": "   "}]),
                id="message_whitespace_content",
            ),
            param(
                _make_row(messages=[{"role": "user", "content": 42}]),
                id="message_content_not_str",
            ),
        ],
    )  # fmt: skip
    def test_invalid_row_raises(self, row: dict[str, Any]):
        with pytest.raises(ValidationError):
            SpeedBenchRow.model_validate(row)

    def test_placeholder_content_raises_naming_placeholder(self):
        row = _make_row(
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
        assert is_speed_bench_row(_make_row()) is True

    @pytest.mark.parametrize(
        "value",
        [
            param(None, id="none"),
            param("string", id="string"),
            param(123, id="int"),
            param([{"role": "user", "content": "Hi"}], id="list"),
            param({"category": "coding"}, id="missing_required_fields"),
            param(
                _make_row(
                    messages=[{"role": "user", "content": SpeedBenchRow.TURNS_PLACEHOLDER}]
                ),
                id="placeholder_content",
            ),
        ],
    )  # fmt: skip
    def test_non_matching_returns_false(self, value: object):
        assert is_speed_bench_row(value) is False


@pytest.mark.asyncio
class TestSpeedBenchLoader:
    async def test_preferred_sampling_strategy_is_sequential(self):
        assert (
            SpeedBenchLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_constructor_stores_category_and_multi_turn(self):
        loader = _make_loader(category="coding", multi_turn=False)
        assert loader.category == "coding"
        assert loader.multi_turn is False

    async def test_multi_turn_defaults_true(self):
        assert _make_loader().multi_turn is True

    async def test_loads_single_row_preserving_id_and_role(self):
        loader = _make_loader()
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-coding-1"),
                    messages=[
                        {
                            "role": "user",
                            "content": "Write a Python function that flips case.",
                        }
                    ],
                )
            ],
        )

        assert len(conversations) == 1
        conversation = conversations[0]
        assert conversation.session_id == _qid("speed-coding-1")
        assert len(conversation.turns) == 1
        assert conversation.turns[0].role == "user"
        assert conversation.turns[0].texts[0].contents == [
            "Write a Python function that flips case."
        ]

    async def test_each_row_becomes_separate_conversation(self):
        loader = _make_loader()
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-coding-1"),
                    category="coding",
                    messages=[{"role": "user", "content": "Implement merge sort."}],
                ),
                _make_row(
                    question_id=_qid("speed-math-1"),
                    category="math",
                    messages=[{"role": "user", "content": "Find the factors of 84."}],
                ),
            ],
        )

        by_id = {c.session_id: c for c in conversations}
        assert set(by_id) == {_qid("speed-coding-1"), _qid("speed-math-1")}
        assert by_id[_qid("speed-coding-1")].turns[0].texts[0].contents == [
            "Implement merge sort."
        ]
        assert by_id[_qid("speed-math-1")].turns[0].texts[0].contents == [
            "Find the factors of 84."
        ]

    async def test_preserves_all_messages_and_roles_in_order(self):
        loader = _make_loader()
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-chat-1"),
                    category="qa",
                    messages=[
                        {"role": "system", "content": "Answer tersely."},
                        {"role": "user", "content": "What is Python?"},
                    ],
                )
            ],
        )

        turns = conversations[0].turns
        assert [turn.role for turn in turns] == ["system", "user"]
        assert [turn.texts[0].contents[0] for turn in turns] == [
            "Answer tersely.",
            "What is Python?",
        ]

    async def test_empty_dataset_returns_empty_list(self):
        loader = _make_loader()
        assert await _convert(loader, []) == []


@pytest.mark.asyncio
class TestSpeedBenchLoaderCategoryFiltering:
    async def test_no_category_returns_all_rows(self):
        loader = _make_loader()
        conversations = await _convert(
            loader,
            [
                _make_row(question_id=_qid("speed-coding-1"), category="coding"),
                _make_row(question_id=_qid("speed-math-1"), category="math"),
            ],
        )
        assert {c.session_id for c in conversations} == {
            _qid("speed-coding-1"),
            _qid("speed-math-1"),
        }

    async def test_category_filter_returns_matching_rows(self):
        loader = _make_loader(category="coding")
        conversations = await _convert(
            loader,
            [
                _make_row(question_id=_qid("speed-coding-1"), category="coding"),
                _make_row(question_id=_qid("speed-math-1"), category="math"),
            ],
        )
        assert {c.session_id for c in conversations} == {_qid("speed-coding-1")}

    async def test_category_filter_no_matches_returns_empty(self):
        loader = _make_loader(category="coding")
        conversations = await _convert(
            loader,
            [
                _make_row(question_id=_qid("speed-math-1"), category="math"),
                _make_row(question_id=_qid("speed-stem-1"), category="stem"),
            ],
        )
        assert conversations == []

    async def test_throughput_entropy_tier_filtering(self):
        loader = _make_loader(category="low_entropy")
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-low-1"),
                    category="low_entropy",
                    messages=[{"role": "user", "content": "Complete the code sample."}],
                ),
                _make_row(
                    question_id=_qid("speed-high-1"),
                    category="high_entropy",
                    messages=[
                        {"role": "user", "content": "Continue this novel excerpt."}
                    ],
                ),
            ],
        )
        assert {c.session_id for c in conversations} == {_qid("speed-low-1")}
        assert conversations[0].turns[0].texts[0].contents == [
            "Complete the code sample."
        ]


@pytest.mark.asyncio
class TestSpeedBenchLoaderRowValidation:
    async def test_convert_rejects_placeholder_content(self):
        loader = _make_loader()
        with pytest.raises(ValidationError, match="placeholder"):
            await _convert(
                loader,
                [
                    _make_row(
                        messages=[
                            {"role": "user", "content": SpeedBenchRow.TURNS_PLACEHOLDER}
                        ]
                    )
                ],
            )

    async def test_convert_missing_question_id_raises_naming_field(self):
        loader = _make_loader()
        with pytest.raises(ValidationError, match="question_id"):
            await _convert(
                loader,
                [
                    {
                        "category": "coding",
                        "messages": [{"role": "user", "content": "Hi"}],
                    }
                ],
            )


@pytest.mark.asyncio
class TestSpeedBenchLoaderMultiTurn:
    async def test_multi_turn_produces_all_messages(self):
        loader = _make_loader(multi_turn=True)
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-chat-1"),
                    messages=[
                        {"role": "user", "content": "First turn"},
                        {"role": "assistant", "content": "Second turn"},
                        {"role": "user", "content": "Third turn"},
                    ],
                )
            ],
        )
        turns = conversations[0].turns
        assert [turn.role for turn in turns] == ["user", "assistant", "user"]
        assert [turn.texts[0].contents[0] for turn in turns] == [
            "First turn",
            "Second turn",
            "Third turn",
        ]

    async def test_multi_turn_false_loads_first_message_only(self):
        loader = _make_loader(multi_turn=False)
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-chat-1"),
                    messages=[
                        {"role": "user", "content": "First turn"},
                        {"role": "user", "content": "Second turn"},
                    ],
                )
            ],
        )
        turns = conversations[0].turns
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].texts[0].contents == ["First turn"]

    async def test_multi_turn_with_category_filter(self):
        loader = _make_loader(category="coding", multi_turn=True)
        conversations = await _convert(
            loader,
            [
                _make_row(
                    question_id=_qid("speed-coding-1"),
                    category="coding",
                    messages=[
                        {"role": "user", "content": "Code Q1"},
                        {"role": "user", "content": "Code Q2"},
                    ],
                ),
                _make_row(
                    question_id=_qid("speed-math-1"),
                    category="math",
                    messages=[{"role": "user", "content": "Math Q1"}],
                ),
            ],
        )
        assert {c.session_id for c in conversations} == {_qid("speed-coding-1")}
        turns = conversations[0].turns
        assert [turn.texts[0].contents[0] for turn in turns] == ["Code Q1", "Code Q2"]
