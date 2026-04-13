# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trace anonymization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.synthesis.anonymize import RawConversationRecord, anonymize_trace


class TestTokenizerApplyChatTemplate:
    """Tests for Tokenizer.apply_chat_template."""

    def test_apply_chat_template_delegates_to_underlying(self) -> None:
        """Test that apply_chat_template calls the underlying tokenizer."""
        tokenizer = Tokenizer()
        mock_hf = MagicMock()
        mock_hf.apply_chat_template.return_value = "<|user|>Hello<|end|>"
        tokenizer._tokenizer = mock_hf

        messages = [{"role": "user", "content": "Hello"}]
        result = tokenizer.apply_chat_template(messages)

        assert result == "<|user|>Hello<|end|>"
        mock_hf.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )

    def test_apply_chat_template_not_initialized_raises(self) -> None:
        """Test that calling apply_chat_template before init raises."""
        tokenizer = Tokenizer()

        with pytest.raises(Exception, match="not initialized"):
            tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])


class TestRawConversationRecord:
    """Tests for input record validation."""

    def test_valid_single_turn(self) -> None:
        data = {
            "timestamp": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "output": "Hi there",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.timestamp == 100
        assert len(record.messages) == 1
        assert record.output == "Hi there"
        assert record.session_id is None

    def test_valid_multi_turn(self) -> None:
        data = {
            "timestamp": 200,
            "session_id": "sess_1",
            "messages": [{"role": "user", "content": "Explain ML"}],
            "output": "Machine learning is...",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.session_id == "sess_1"

    def test_missing_messages_raises(self) -> None:
        data = {"timestamp": 0, "output": "response"}
        with pytest.raises(ValidationError):
            RawConversationRecord.model_validate(data)

    def test_missing_output_raises(self) -> None:
        data = {"timestamp": 0, "messages": [{"role": "user", "content": "Hi"}]}
        with pytest.raises(ValidationError):
            RawConversationRecord.model_validate(data)

    def test_empty_messages_raises(self) -> None:
        data = {"timestamp": 0, "messages": [], "output": "response"}
        with pytest.raises(ValidationError):
            RawConversationRecord.model_validate(data)

    def test_message_missing_role_raises(self) -> None:
        data = {
            "messages": [{"content": "Hello"}],
            "output": "Hi",
        }
        with pytest.raises(ValidationError, match="missing required 'role' key"):
            RawConversationRecord.model_validate(data)

    def test_message_missing_content_raises(self) -> None:
        data = {
            "messages": [{"role": "user"}],
            "output": "Hi",
        }
        with pytest.raises(ValidationError, match="missing required 'content' key"):
            RawConversationRecord.model_validate(data)

    def test_no_timestamp_is_valid(self) -> None:
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "output": "Hi",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.timestamp is None


class TestAnonymizeTrace:
    """Tests for the core anonymize_trace function."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer that simulates realistic behavior."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|user|>Hello<|end|>"
        tokenizer.encode.return_value = list(range(10))
        return tokenizer

    def test_single_turn_produces_valid_output(self, mock_tokenizer: MagicMock) -> None:
        """Test that single-turn input produces valid Mooncake trace output."""
        input_data = [
            {
                "timestamp": 0,
                "messages": [{"role": "user", "content": "Hello"}],
                "output": "Hi there",
            },
            {
                "timestamp": 100,
                "messages": [{"role": "user", "content": "Bye"}],
                "output": "Goodbye",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert output_path.exists()
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            record_0 = orjson.loads(lines[0])
            assert "timestamp" in record_0
            assert "input_length" in record_0
            assert "output_length" in record_0
            assert "hash_ids" in record_0
            assert isinstance(record_0["hash_ids"], list)
            assert record_0["timestamp"] == 0

            record_str = lines[0]
            assert "Hello" not in record_str
            assert "Hi there" not in record_str

            assert result.total_processed == 2
            assert result.total_skipped == 0
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_multi_turn_prefix_sharing(self, mock_tokenizer: MagicMock) -> None:
        """Test that multi-turn sessions produce shared prefix hash_ids."""
        call_count = 0

        def mock_encode(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return list(range(8))
            else:
                return list(range(16))

        mock_tokenizer.encode.side_effect = mock_encode

        def mock_template(messages, tokenize=False, add_generation_prompt=True):
            return f"template_{len(messages)}"

        mock_tokenizer.apply_chat_template.side_effect = mock_template

        input_data = [
            {
                "timestamp": 0,
                "session_id": "s1",
                "messages": [{"role": "user", "content": "Hello"}],
                "output": "Hi",
            },
            {
                "timestamp": 100,
                "session_id": "s1",
                "messages": [{"role": "user", "content": "More"}],
                "output": "Sure",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_mt.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            record_0 = orjson.loads(lines[0])
            record_1 = orjson.loads(lines[1])

            assert record_0["session_id"] == "s1"
            assert record_1["session_id"] == "s1"

            assert len(record_1["hash_ids"]) > len(record_0["hash_ids"])

            for i in range(len(record_0["hash_ids"])):
                assert record_0["hash_ids"][i] == record_1["hash_ids"][i]
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_independent_requests_no_accumulation(
        self, mock_tokenizer: MagicMock
    ) -> None:
        """Test that independent requests (no session_id) don't accumulate messages."""
        # Each independent request should produce the same input_length
        # if their tokenization is the same — NOT growing lengths from accumulation
        mock_tokenizer.encode.return_value = list(range(8))
        mock_tokenizer.apply_chat_template.return_value = "template"

        input_data = [
            {
                "timestamp": 0,
                "messages": [{"role": "user", "content": "Q1"}],
                "output": "A1",
            },
            {
                "timestamp": 100,
                "messages": [{"role": "user", "content": "Q2"}],
                "output": "A2",
            },
            {
                "timestamp": 200,
                "messages": [{"role": "user", "content": "Q3"}],
                "output": "A3",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_no_accum.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 3

            # All independent requests should have the same input_length
            lengths = [orjson.loads(line)["input_length"] for line in lines]
            assert lengths == [8, 8, 8]

            # apply_chat_template should be called with only each request's own messages
            calls = mock_tokenizer.apply_chat_template.call_args_list
            for call in calls:
                messages_arg = call[0][0]
                assert len(messages_arg) == 1  # Not accumulated
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_prefix_overlap_across_requests(self, mock_tokenizer: MagicMock) -> None:
        """Test that independent requests with shared prefixes produce shared hash_ids."""
        mock_tokenizer.encode.return_value = list(range(8))
        mock_tokenizer.apply_chat_template.return_value = "same_template"

        input_data = [
            {
                "timestamp": 0,
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Q1"},
                ],
                "output": "A1",
            },
            {
                "timestamp": 100,
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Q2"},
                ],
                "output": "A2",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_prefix.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            record_0 = orjson.loads(lines[0])
            record_1 = orjson.loads(lines[1])

            assert record_0["hash_ids"] == record_1["hash_ids"]
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_missing_timestamp_warning(self, mock_tokenizer: MagicMock) -> None:
        """Test that missing timestamps produce a warning in the result."""
        input_data = [
            {"messages": [{"role": "user", "content": "Hello"}], "output": "Hi"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_no_ts.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert result.no_timestamps_warning

            lines = output_path.read_text().strip().split("\n")
            record = orjson.loads(lines[0])
            assert "timestamp" not in record
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_malformed_line_skipped(self, mock_tokenizer: MagicMock) -> None:
        """Test that malformed lines are skipped with count."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"messages": [{"role": "user", "content": "Good"}], "output": "Hi"}\n'
            )
            f.write("not valid json\n")
            f.write('{"messages": [], "output": "empty"}\n')
            f.write(
                '{"messages": [{"role": "user", "content": "Also good"}], "output": "Bye"}\n'
            )
            input_path = Path(f.name)

        output_path = input_path.with_name("output_skip.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert result.total_processed == 2
            assert result.total_skipped == 2

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_block_size_zero_raises(self, mock_tokenizer: MagicMock) -> None:
        """Test that block_size <= 0 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"messages": [{"role": "user", "content": "Hi"}], "output": "Hey"}\n'
            )
            input_path = Path(f.name)

        output_path = input_path.with_name("output_bs0.jsonl")

        try:
            with pytest.raises(ValueError, match="block_size must be greater than 0"):
                anonymize_trace(
                    input_file=input_path,
                    output_file=output_path,
                    tokenizer=mock_tokenizer,
                    block_size=0,
                )
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_interleaved_sessions_preserve_global_order(
        self, mock_tokenizer: MagicMock
    ) -> None:
        """Test that interleaved sessions emit records in original input order."""
        mock_tokenizer.encode.return_value = list(range(8))
        mock_tokenizer.apply_chat_template.return_value = "template"

        input_data = [
            {
                "timestamp": 0,
                "session_id": "A",
                "messages": [{"role": "user", "content": "A1"}],
                "output": "respA1",
            },
            {
                "timestamp": 100,
                "session_id": "B",
                "messages": [{"role": "user", "content": "B1"}],
                "output": "respB1",
            },
            {
                "timestamp": 200,
                "session_id": "A",
                "messages": [{"role": "user", "content": "A2"}],
                "output": "respA2",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_interleave.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 3

            records = [orjson.loads(line) for line in lines]
            # Output order must match input order: A, B, A
            assert records[0]["session_id"] == "A"
            assert records[0]["timestamp"] == 0
            assert records[1]["session_id"] == "B"
            assert records[1]["timestamp"] == 100
            assert records[2]["session_id"] == "A"
            assert records[2]["timestamp"] == 200
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
