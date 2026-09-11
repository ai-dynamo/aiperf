# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in MooncakeTrace message_mode='delta' replay mode."""

from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

USER_MSG = [{"role": "user", "content": "Hello"}]
TOOL_DEFS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")
    return path


def _load_conversations(
    tmp_path: Path, records: list[dict], default_cfg, mock_prompt_generator, run=None
):
    file = _write_jsonl(tmp_path / "trace.jsonl", records)
    loader = MooncakeTraceDatasetLoader(
        filename=file,
        cfg=default_cfg,
        prompt_generator=mock_prompt_generator,
        run=run,
    )
    return loader.convert_to_conversations(loader.load_dataset())


class TestMessageModeSchema:
    """MooncakeTrace model validation for the message_mode field."""

    def test_message_mode_defaults_to_history(self):
        trace = MooncakeTrace(messages=USER_MSG)
        assert trace.message_mode == "history"

    def test_message_mode_history_explicit(self):
        trace = MooncakeTrace(messages=USER_MSG, message_mode="history")
        assert trace.message_mode == "history"

    def test_message_mode_delta_with_messages(self):
        trace = MooncakeTrace(messages=USER_MSG, message_mode="delta")
        assert trace.message_mode == "delta"

    def test_message_mode_default_preserved_for_non_message_modes(self):
        assert MooncakeTrace(input_length=10).message_mode == "history"
        assert MooncakeTrace(text_input="hi").message_mode == "history"
        assert MooncakeTrace(payload={"prompt": "hi"}).message_mode == "history"

    @pytest.mark.parametrize(
        "fields",
        [
            param({"payload": {"prompt": "hi"}}, id="payload"),
            param({"input_length": 100}, id="synthesized_input_length"),
            param({"input_length": 100, "hash_ids": [1, 2]}, id="synthesized_hash_ids"),
            param({"text_input": "hi"}, id="text_input"),
        ],
    )  # fmt: skip
    def test_delta_without_messages_rejected(self, fields: dict):
        with pytest.raises(ValidationError, match="only supported with 'messages'"):
            MooncakeTrace(message_mode="delta", **fields)

    @pytest.mark.parametrize(
        "bad_mode",
        [
            param("deltas", id="plural"),
            param("Delta", id="wrong_case"),
            param("message_array", id="context_mode_name"),
            param("", id="empty"),
            param(1, id="non_string"),
        ],
    )  # fmt: skip
    def test_unsupported_message_mode_literal_rejected(self, bad_mode):
        with pytest.raises(ValidationError, match="message_mode"):
            MooncakeTrace(messages=USER_MSG, message_mode=bad_mode)

    def test_delta_with_extra_messages_key_rejected(self):
        with pytest.raises(ValidationError, match="'extra' must not contain"):
            MooncakeTrace(
                messages=USER_MSG,
                message_mode="delta",
                extra={"messages": [{"role": "user", "content": "clobber"}]},
            )

    def test_delta_with_safe_extra_allowed(self):
        trace = MooncakeTrace(
            messages=USER_MSG, message_mode="delta", extra={"temperature": 0.5}
        )
        assert trace.extra == {"temperature": 0.5}

    def test_history_with_extra_messages_key_still_allowed(self):
        """Backward compatibility: history mode does not reject extra['messages']."""
        trace = MooncakeTrace(
            messages=USER_MSG, extra={"messages": [{"role": "user", "content": "x"}]}
        )
        assert trace.extra is not None

    def test_delta_with_tools_allowed(self):
        trace = MooncakeTrace(messages=USER_MSG, message_mode="delta", tools=TOOL_DEFS)
        assert trace.tools == TOOL_DEFS

    def test_delta_serialization_round_trip(self):
        trace = MooncakeTrace(
            messages=USER_MSG,
            message_mode="delta",
            tools=TOOL_DEFS,
            output_length=50,
            timestamp=1000,
            session_id="s1",
        )
        round_tripped = MooncakeTrace.model_validate(
            orjson.loads(orjson.dumps(trace.model_dump(exclude_none=True)))
        )
        assert round_tripped == trace
        assert round_tripped.message_mode == "delta"

    def test_history_default_serialization_round_trip(self):
        trace = MooncakeTrace(messages=USER_MSG, output_length=5)
        round_tripped = MooncakeTrace.model_validate(
            orjson.loads(orjson.dumps(trace.model_dump(exclude_none=True)))
        )
        assert round_tripped.message_mode == "history"

    def test_can_load_accepts_delta_record(self):
        assert MooncakeTraceDatasetLoader.can_load(
            {"messages": USER_MSG, "message_mode": "delta"}
        )

    def test_can_load_rejects_delta_payload_record(self):
        assert not MooncakeTraceDatasetLoader.can_load(
            {"payload": {"prompt": "hi"}, "message_mode": "delta"}
        )


class TestDeltaContextModeInference:
    """Loader context-mode inference for delta and history sessions."""

    def test_all_delta_session_uses_deltas_without_responses(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}], "timestamp": 0},
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t2"}], "delay": 250},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        assert len(conversations) == 1
        assert (
            conversations[0].context_mode
            == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )

    def test_history_session_default_unchanged(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "messages": [{"role": "user", "content": "t1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "t2"}]},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        assert (
            conversations[0].context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )

    def test_synthetic_session_context_mode_unchanged(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "input_length": 10, "output_length": 5},
            {"session_id": "s1", "input_length": 20, "output_length": 5, "delay": 10},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        assert conversations[0].context_mode is None

    def test_mixed_delta_and_explicit_history_rejected(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "message_mode": "history", "messages": [{"role": "user", "content": "t2"}]},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="exactly one message_mode"):
            _load_conversations(tmp_path, records, default_cfg, mock_prompt_generator)

    def test_mixed_delta_and_omitted_history_default_rejected(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "messages": [{"role": "user", "content": "t2"}]},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="omitted message_mode defaults"):
            _load_conversations(tmp_path, records, default_cfg, mock_prompt_generator)

    def test_mixed_delta_messages_and_payload_rejected(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "payload": {"prompt": "t2"}},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="exactly one mode"):
            _load_conversations(tmp_path, records, default_cfg, mock_prompt_generator)

    def test_mixed_delta_messages_and_synthetic_rejected(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "input_length": 10},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="synthesized prompts are unsupported"):
            _load_conversations(tmp_path, records, default_cfg, mock_prompt_generator)

    def test_mixed_history_messages_and_payload_still_rejected(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "messages": [{"role": "user", "content": "t1"}]},
            {"session_id": "s1", "payload": {"prompt": "t2"}},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="exactly one mode"):
            _load_conversations(tmp_path, records, default_cfg, mock_prompt_generator)

    def test_independent_sessions_resolve_modes_independently(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "delta-sess", "message_mode": "delta", "messages": [{"role": "user", "content": "d1"}]},
            {"session_id": "history-sess", "messages": [{"role": "user", "content": "h1"}]},
            {"session_id": "synthetic-sess", "input_length": 10},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        modes = {c.session_id: c.context_mode for c in conversations}
        assert modes["delta-sess"] == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        assert (
            modes["history-sess"]
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        assert modes["synthetic-sess"] is None


class TestDeltaTurnConstruction:
    """Turn-level field preservation for delta traces."""

    def test_timestamp_delay_and_caps_preserved(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}], "timestamp": 1000.5, "output_length": 64},
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t2"}], "delay": 250.25, "output_length": 32},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        turns = conversations[0].turns
        assert turns[0].timestamp == 1000.5
        assert turns[0].max_tokens == 64
        assert turns[0].delay is None
        assert turns[1].delay == 250.25
        assert turns[1].max_tokens == 32
        assert turns[1].timestamp is None

    def test_delta_turn_carries_raw_messages_tools_and_extra(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "t1"},
        ]
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": messages, "tools": TOOL_DEFS, "extra": {"temperature": 0.1}},
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t2"}]},
        ]  # fmt: skip
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator
        )
        turns = conversations[0].turns
        assert turns[0].raw_messages == messages
        assert turns[0].raw_tools == TOOL_DEFS
        assert turns[0].extra_body == {"temperature": 0.1}
        assert turns[1].raw_messages == [{"role": "user", "content": "t2"}]
        assert turns[1].raw_tools is None
        assert turns[1].extra_body is None


class TestDeltaEndpointExtraProtection:
    """Endpoint-global extra inputs must not clobber assembled delta history."""

    def _make_run(self, extra_inputs):
        from tests.unit.conftest import make_run_from_cli

        return make_run_from_cli(
            CLIConfig(model_names=["test-model"], extra_inputs=extra_inputs)
        )

    def test_endpoint_extra_messages_rejected_for_delta_sessions(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
        ]  # fmt: skip
        run = self._make_run({"messages": [{"role": "user", "content": "clobber"}]})
        with pytest.raises(ValueError, match="endpoint-level extra input"):
            _load_conversations(
                tmp_path, records, default_cfg, mock_prompt_generator, run=run
            )

    def test_endpoint_extra_without_messages_allowed_for_delta_sessions(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        records = [
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}]},
        ]  # fmt: skip
        run = self._make_run({"temperature": 0.7})
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator, run=run
        )
        assert (
            conversations[0].context_mode
            == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )

    def test_endpoint_extra_messages_still_allowed_for_history_sessions(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        """Narrow fail-closed scope: history-mode replay is unchanged."""
        records = [
            {"session_id": "s1", "messages": [{"role": "user", "content": "t1"}]},
        ]  # fmt: skip
        run = self._make_run({"messages": [{"role": "user", "content": "override"}]})
        conversations = _load_conversations(
            tmp_path, records, default_cfg, mock_prompt_generator, run=run
        )
        assert (
            conversations[0].context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )


class TestDeltaOffsetRejection:
    """Timestamp offset cropping must fail closed for delta sessions.

    start_offset/end_offset drop individual rows before grouping, which would
    silently remove the initial history or preceding turns a later delta row
    depends on. Any configured offset combined with any delta row aborts the
    load; history-mode and synthesized traces keep the existing filtering.
    """

    DELTA_RECORDS = [
        {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t1"}], "timestamp": 1000},
        {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t2"}], "timestamp": 2000},
        {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "t3"}], "timestamp": 3000},
    ]  # fmt: skip

    def _make_offset_run(
        self,
        file: Path,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ):
        from tests.unit.conftest import make_run_from_cli

        input_kwargs: dict = {"input_file": str(file), "fixed_schedule": True}
        if start_offset is not None:
            input_kwargs["fixed_schedule_start_offset"] = start_offset
        if end_offset is not None:
            input_kwargs["fixed_schedule_end_offset"] = end_offset
        return make_run_from_cli(CLIConfig(model_names=["test-model"], **input_kwargs))

    def _load_with_offsets(
        self,
        tmp_path: Path,
        records: list[dict],
        mock_prompt_generator,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ):
        file = _write_jsonl(tmp_path / "trace.jsonl", records)
        run = self._make_offset_run(file, start_offset, end_offset)
        loader = MooncakeTraceDatasetLoader(
            filename=file, prompt_generator=mock_prompt_generator, run=run
        )
        return loader.convert_to_conversations(loader.load_dataset())

    def test_start_offset_prefix_removal_rejected(
        self, tmp_path: Path, mock_prompt_generator
    ):
        """A start offset that would drop the initial history row must abort."""
        with pytest.raises(ValueError, match="message_mode='delta'"):
            self._load_with_offsets(
                tmp_path, self.DELTA_RECORDS, mock_prompt_generator, start_offset=1500
            )

    def test_end_offset_cutoff_rejected(self, tmp_path: Path, mock_prompt_generator):
        """An end offset that would drop trailing delta rows must abort."""
        with pytest.raises(ValueError, match="message_mode='delta'"):
            self._load_with_offsets(
                tmp_path, self.DELTA_RECORDS, mock_prompt_generator, end_offset=2500
            )

    def test_both_offsets_rejected(self, tmp_path: Path, mock_prompt_generator):
        with pytest.raises(ValueError, match="cannot be combined"):
            self._load_with_offsets(
                tmp_path,
                self.DELTA_RECORDS,
                mock_prompt_generator,
                start_offset=1500,
                end_offset=2500,
            )

    def test_offsets_rejected_even_when_no_delta_row_filtered(
        self, tmp_path: Path, mock_prompt_generator
    ):
        """Fail-closed scope: the offset configuration itself is rejected.

        start_offset=0 would keep every row, but permitting it would make
        validity depend on the data's timestamps, so it is rejected outright.
        """
        with pytest.raises(ValueError, match="message_mode='delta'"):
            self._load_with_offsets(
                tmp_path, self.DELTA_RECORDS, mock_prompt_generator, start_offset=0
            )

    def test_delta_row_after_filtered_history_rows_rejected(
        self, tmp_path: Path, mock_prompt_generator
    ):
        """History rows filtered earlier in the file cannot mask the abort.

        The two history-session rows fall before the start offset and are
        discarded first, but the load still fails when the retained delta row
        is parsed -- a partially filtered dataset is never returned.
        """
        records = [
            {"session_id": "hist", "messages": [{"role": "user", "content": "h1"}], "timestamp": 100},
            {"session_id": "hist", "messages": [{"role": "user", "content": "h1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "h2"}], "timestamp": 200},
            {"session_id": "s1", "message_mode": "delta", "messages": [{"role": "user", "content": "d1"}], "timestamp": 2000},
        ]  # fmt: skip
        with pytest.raises(ValueError, match="message_mode='delta'"):
            self._load_with_offsets(
                tmp_path, records, mock_prompt_generator, start_offset=1500
            )

    def test_no_offset_delta_fixed_schedule_allowed(
        self, tmp_path: Path, mock_prompt_generator
    ):
        """Ordinary fixed-schedule delta replay without offsets is unaffected."""
        conversations = self._load_with_offsets(
            tmp_path, self.DELTA_RECORDS, mock_prompt_generator
        )
        assert len(conversations) == 1
        assert (
            conversations[0].context_mode
            == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )
        assert len(conversations[0].turns) == 3

    def test_history_mode_offset_filtering_preserved(
        self, tmp_path: Path, mock_prompt_generator
    ):
        """Backward compatibility: history-mode rows keep offset filtering."""
        records = [
            {"session_id": "a", "messages": [{"role": "user", "content": "m1"}], "timestamp": 1000},
            {"session_id": "b", "messages": [{"role": "user", "content": "m2"}], "timestamp": 2000},
            {"session_id": "c", "messages": [{"role": "user", "content": "m3"}], "timestamp": 2500},
            {"session_id": "d", "messages": [{"role": "user", "content": "m4"}], "timestamp": 3000},
        ]  # fmt: skip
        conversations = self._load_with_offsets(
            tmp_path, records, mock_prompt_generator, start_offset=1500, end_offset=2500
        )
        assert {c.session_id for c in conversations} == {"b", "c"}
        assert all(
            c.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
            for c in conversations
        )
