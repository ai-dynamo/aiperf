# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warnings for potentially incremental Mooncake rows using implicit recorded mode."""

import logging
from typing import Any
from unittest.mock import Mock

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from tests.unit.conftest import make_benchmark_run

USER_1 = {"role": "user", "content": "First question"}
USER_2 = {"role": "user", "content": "Follow-up question"}
ASSISTANT_1 = {"role": "assistant", "content": "Recorded answer"}
SYSTEM = {"role": "system", "content": "Be helpful"}


def _omission_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.message
        for record in caplog.records
        if "omit assistant_responses" in record.message
    ]


@pytest.mark.parametrize(
    "message_rows",
    [
        param([[SYSTEM, USER_1], [USER_2]], id="user_only_deltas"),
        param([[USER_1], [USER_1, USER_2]], id="no_recorded_assistant"),
        param([[USER_1, ASSISTANT_1, USER_2], [USER_2]], id="initial_history_then_delta"),
        param([[USER_1, ASSISTANT_1], [SYSTEM, USER_1, ASSISTANT_1, USER_2]], id="non_prefix_history"),
    ],
)  # fmt: skip
def test_implicit_incremental_rows_warn_without_changing_replay(
    message_rows: list[list[dict[str, Any]]],
    mock_prompt_generator: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An advisory surfaces ambiguity while preserving existing recorded behavior."""
    caplog.set_level(logging.WARNING)
    loader = MooncakeTraceDatasetLoader(
        inline_records=[
            {"session_id": "s1", "messages": messages} for messages in message_rows
        ],
        prompt_generator=mock_prompt_generator,
    )

    conversations = loader.convert_to_conversations(loader.load_dataset())

    warnings = _omission_warnings(caplog)
    assert len(warnings) == 1
    assert "including 's1'" in warnings[0]
    assert "each row independently" in warnings[0]
    assert "assistant_responses='live' on every row" in warnings[0]
    assert "explicitly set 'recorded'" in warnings[0]
    assert (
        conversations[0].context_mode
        == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
    assert [turn.raw_messages for turn in conversations[0].turns] == message_rows


@pytest.mark.parametrize(
    "rows",
    [
        param([{"messages": [USER_1]}, {"messages": [USER_1, ASSISTANT_1, USER_2]}], id="complete_recorded_history"),
        param([{"messages": [USER_1, ASSISTANT_1]}, {"messages": [USER_1, ASSISTANT_1]}], id="unchanged_complete_history"),
        param([{"messages": [USER_1]}], id="single_row"),
        param([{"session_id": "s1", "messages": [USER_1]}, {"session_id": "s2", "messages": [USER_2]}], id="independent_single_rows"),
        param([{"messages": [USER_1], "assistant_responses": "recorded"}, {"messages": [USER_2], "assistant_responses": "recorded"}], id="explicit_recorded"),
        param([{"messages": [USER_1], "assistant_responses": "Recorded"}, {"messages": [USER_2]}], id="explicit_first_row"),
        param([{"messages": [USER_1]}, {"messages": [USER_2], "assistant_responses": "recorded"}], id="explicit_later_row"),
        param([{"messages": [USER_1], "assistant_responses": "live"}, {"messages": [USER_2], "assistant_responses": "live"}], id="explicit_live"),
        param([{"payload": {"messages": [USER_1]}}, {"payload": {"messages": [USER_2]}}], id="payload"),
        param([{"text_input": "First"}, {"text_input": "Second"}], id="text"),
        param([{"input_length": 10}, {"input_length": 20}], id="synthetic"),
    ],
)  # fmt: skip
def test_recorded_warning_skips_explicit_choices_and_other_inputs(
    rows: list[dict[str, Any]],
    mock_prompt_generator: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Intentional replay choices and unambiguous inputs need no advisory."""
    caplog.set_level(logging.WARNING)
    loader = MooncakeTraceDatasetLoader(
        inline_records=[{"session_id": "s1", **row} for row in rows],
        prompt_generator=mock_prompt_generator,
    )

    data = loader.load_dataset()

    assert sum(len(traces) for traces in data.values()) == len(rows)
    assert not _omission_warnings(caplog)


def test_recorded_warning_aggregates_once_per_load(
    mock_prompt_generator: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Large files emit one warning, and reusing a loader does not suppress it."""
    caplog.set_level(logging.WARNING)
    loader = MooncakeTraceDatasetLoader(
        inline_records=[
            {"session_id": session_id, "messages": [message]}
            for session_id in ("first", "second")
            for message in (USER_1, USER_2)
        ],
        prompt_generator=mock_prompt_generator,
    )

    for _ in range(2):
        caplog.clear()
        data = loader.load_dataset()
        loader.convert_to_conversations(data)
        warnings = _omission_warnings(caplog)
        assert len(warnings) == 1
        assert "2 multi-row messages session(s)" in warnings[0]
        assert "including 'first'" in warnings[0]


@pytest.mark.parametrize("explicit_recorded", [False, True])
def test_recorded_warning_uses_authored_fields_before_synthesis(
    explicit_recorded: bool,
    mock_prompt_generator: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthesis may materialize defaults but must not hide an omitted choice."""
    caplog.set_level(logging.WARNING)
    mode = {"assistant_responses": "recorded"} if explicit_recorded else {}
    rows = [
        {"session_id": "s1", "timestamp": index * 1000, "messages": [message], **mode}
        for index, message in enumerate((USER_1, USER_2))
    ]
    run = make_benchmark_run(
        extra={
            "datasets": [
                {
                    "name": "default",
                    "type": "file",
                    "format": "mooncake_trace",
                    "records": rows,
                    "synthesis": {"speedup_ratio": 2.0},
                }
            ]
        }
    )
    loader = MooncakeTraceDatasetLoader(
        inline_records=rows, prompt_generator=mock_prompt_generator, run=run
    )

    data = loader.load_dataset()
    conversations = loader.convert_to_conversations(data)

    assert len(_omission_warnings(caplog)) == (0 if explicit_recorded else 1)
    assert [trace.timestamp for trace in data["s1"]] == [0, 500]
    assert (
        conversations[0].context_mode
        == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
    assert [turn.raw_messages for turn in conversations[0].turns] == [
        [USER_1],
        [USER_2],
    ]
