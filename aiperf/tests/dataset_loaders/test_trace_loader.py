#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Comprehensive unit tests for TraceDatasetLoader class.

This test module provides complete coverage of all methods in the TraceDatasetLoader class,
including initialization, data loading, data conversion, integration scenarios, edge cases,
error conditions, and performance testing.
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from aiperf.common.dataset_models import Conversation, Text, Turn
from aiperf.services.dataset.generator.prompt import PromptGenerator
from aiperf.services.dataset.loader.models import TraceCustomData
from aiperf.services.dataset.loader.trace import TraceDatasetLoader


class TestTraceDatasetLoaderInitialization:
    """Test suite for TraceDatasetLoader initialization and constructor."""

    def test_init_with_valid_parameters(self, mock_prompt_generator):
        """Test basic initialization with valid filename and prompt_generator parameters.

        Verifies that TraceDatasetLoader can be instantiated with valid parameters
        and that the constructor completes without errors.
        """
        filename = "test_trace.jsonl"
        loader = TraceDatasetLoader(filename, mock_prompt_generator)

        assert loader is not None
        assert isinstance(loader, TraceDatasetLoader)

    def test_init_stores_parameters_correctly(self, mock_prompt_generator):
        """Test that filename and prompt_generator are stored as instance variables.

        Verifies that the constructor properly stores the provided filename and
        prompt_generator as instance variables that can be accessed later.
        """
        filename = "test_trace.jsonl"
        loader = TraceDatasetLoader(filename, mock_prompt_generator)

        assert loader.filename == filename
        assert loader.prompt_generator == mock_prompt_generator


class TestTraceDatasetLoaderDataLoading:
    """Test suite for load_dataset method functionality."""

    # ============================================================================
    # Valid Data Loading Tests
    # ============================================================================

    def test_load_dataset_single_session_with_session_id(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test loading data where each line has explicit session_id.

        Verifies that trace data with explicit session_id values are correctly
        loaded and grouped by their session_id in the returned dictionary.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["single_session_explicit"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            assert "session-1" in data
            assert len(data["session-1"]) == 1

            trace_data = data["session-1"][0]
            assert trace_data.session_id == "session-1"
            assert trace_data.input_length == 300
            assert trace_data.output_length == 40
            assert trace_data.hash_ids == [123, 456]
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_single_session_without_session_id(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test loading data where session_id is auto-generated using UUID.

        Verifies that trace data without session_id gets auto-generated UUIDs
        and that each line without session_id gets its own unique session.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["single_session_implicit"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            session_id = list(data.keys())[0]
            # Should be a valid UUID string
            uuid.UUID(session_id)  # This will raise ValueError if not valid UUID

            trace_data = data[session_id][0]
            assert trace_data.session_id is None  # Original data had no session_id
            assert trace_data.input_length == 250
            assert trace_data.output_length == 35
            assert trace_data.hash_ids == [789, 101]
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_multiple_sessions(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test loading data with multiple different session_ids.

        Verifies that trace data with different session_ids are correctly
        grouped together in the returned dictionary by session_id.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["multiple_sessions"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 2
            assert "session-1" in data
            assert "session-2" in data

            # session-1 should have 2 entries
            assert len(data["session-1"]) == 2
            assert data["session-1"][0].input_length == 300
            assert data["session-1"][1].input_length == 200

            # session-2 should have 1 entry
            assert len(data["session-2"]) == 1
            assert data["session-2"][0].input_length == 250
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_mixed_sessions(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test loading data with some lines having session_id, others not.

        Verifies that mixed data (some with explicit session_id, some without)
        is correctly processed with UUIDs generated only for missing session_ids.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["mixed_sessions"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 2
            assert "session-1" in data

            # One explicit session, one auto-generated
            session_keys = list(data.keys())
            auto_generated_key = [k for k in session_keys if k != "session-1"][0]
            uuid.UUID(auto_generated_key)  # Should be valid UUID

            assert len(data["session-1"]) == 1
            assert len(data[auto_generated_key]) == 1
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_timestamp_format(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test loading fixed schedule format with timestamps.

        Verifies that trace data in fixed schedule format (with timestamps)
        is correctly loaded and parsed into TraceCustomData objects.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["timestamp_format"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            session_id = list(data.keys())[0]
            trace_data = data[session_id][0]

            assert trace_data.timestamp == 1000
            assert trace_data.session_id is None
            assert trace_data.delay is None
            assert trace_data.input_length == 300
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_session_based_format(
        self, mock_prompt_generator, sample_trace_data
    ):
        """Test loading session-based format with delays.

        Verifies that trace data in session-based format (with delays and session_ids)
        is correctly loaded and parsed into TraceCustomData objects.
        """
        session_data = sample_trace_data["session_format"]
        content = json.dumps(session_data) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            assert "session-123" in data

            trace_data = data["session-123"][0]
            assert trace_data.session_id == "session-123"
            assert trace_data.delay == 1000
            assert trace_data.timestamp is None
            assert trace_data.input_length == 250
        finally:
            Path(temp_file).unlink(missing_ok=True)

    # ============================================================================
    # File Handling Tests
    # ============================================================================

    def test_load_dataset_empty_file(self, mock_prompt_generator, sample_jsonl_data):
        """Test behavior with empty file.

        Verifies that loading an empty trace file returns an empty dictionary
        and handles the empty file gracefully without errors.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["empty"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert isinstance(data, dict)
            assert len(data) == 0
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_single_line(self, mock_prompt_generator, sample_jsonl_data):
        """Test with file containing single valid line.

        Verifies that a file with a single valid trace line is properly
        loaded and creates the expected single-entry data structure.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["single_session_explicit"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            assert len(data["session-1"]) == 1
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_multiple_lines(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test with file containing multiple valid lines.

        Verifies that a file with multiple valid trace lines is properly
        loaded and creates the expected multi-entry data structure.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["multiple_sessions"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            # Should have 2 sessions with total of 3 lines
            total_lines = sum(len(traces) for traces in data.values())
            assert total_lines == 3
        finally:
            Path(temp_file).unlink(missing_ok=True)

    # ============================================================================
    # Error Handling Tests
    # ============================================================================

    def test_load_dataset_file_not_found(self, mock_prompt_generator):
        """Test FileNotFoundError handling.

        Verifies that attempting to load a non-existent file raises
        FileNotFoundError and is properly handled.
        """
        non_existent_file = "non_existent_file.jsonl"
        loader = TraceDatasetLoader(non_existent_file, mock_prompt_generator)

        with pytest.raises(FileNotFoundError):
            loader.load_dataset()

    def test_load_dataset_invalid_json(self, mock_prompt_generator, sample_jsonl_data):
        """Test malformed JSON line handling.

        Verifies that malformed JSON lines in the trace file are properly
        handled and appropriate errors are raised during parsing.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["invalid_json"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            with pytest.raises(ValidationError):
                loader.load_dataset()
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_invalid_trace_data(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test TraceCustomData validation errors.

        Verifies that invalid trace data (failing TraceCustomData validation)
        is properly handled and validation errors are raised appropriately.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["invalid_data"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            with pytest.raises(ValidationError):
                loader.load_dataset()
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_dataset_permission_denied(self, mock_prompt_generator):
        """Test file permission errors.

        Verifies that permission errors when accessing trace files are
        properly handled and appropriate exceptions are raised.
        """
        # Create a file and remove read permissions
        temp_file = create_temp_file_with_content(
            '{"input_length": 100, "output_length": 20, "hash_ids": [1, 2]}'
        )

        try:
            # Remove read permissions
            Path(temp_file).chmod(0o000)
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            with pytest.raises(PermissionError):
                loader.load_dataset()
        finally:
            # Restore permissions for cleanup
            Path(temp_file).chmod(0o644)
            Path(temp_file).unlink(missing_ok=True)


class TestTraceDatasetLoaderDataConversion:
    """Test suite for convert_to_conversations method functionality."""

    # ============================================================================
    # Basic Conversion Tests
    # ============================================================================

    def test_convert_single_session_single_trace(self, mock_prompt_generator):
        """Test converting one session with one trace entry.

        Verifies that a single session containing one trace entry is correctly
        converted to a Conversation object with one Turn.
        """
        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert isinstance(conv, Conversation)
        assert conv.session_id == "session-1"
        assert len(conv.turns) == 1

        turn = conv.turns[0]
        assert isinstance(turn, Turn)
        assert len(turn.text) == 1
        assert turn.text[0].name == "text"
        assert turn.text[0].content == ["Generated test prompt"]

    def test_convert_single_session_multiple_traces(self, mock_prompt_generator):
        """Test converting one session with multiple trace entries (multi-turn).

        Verifies that a single session containing multiple trace entries is
        converted to a Conversation object with multiple Turns in sequence.
        """
        trace_data1 = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        trace_data2 = TraceCustomData(
            input_length=250, output_length=35, hash_ids=[789, 101]
        )
        data = {"session-1": [trace_data1, trace_data2]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert conv.session_id == "session-1"
        assert len(conv.turns) == 2

        # Check both turns
        for turn in conv.turns:
            assert isinstance(turn, Turn)
            assert len(turn.text) == 1
            assert turn.text[0].content == ["Generated test prompt"]

    def test_convert_multiple_sessions(self, mock_prompt_generator):
        """Test converting multiple sessions with multiple traces each.

        Verifies that multiple sessions, each containing multiple trace entries,
        are converted to multiple Conversation objects with correct structure.
        """
        trace_data1 = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        trace_data2 = TraceCustomData(
            input_length=250, output_length=35, hash_ids=[789, 101]
        )
        data = {"session-1": [trace_data1], "session-2": [trace_data2]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2

        # Check that sessions are correctly mapped
        session_ids = [conv.session_id for conv in conversations]
        assert "session-1" in session_ids
        assert "session-2" in session_ids

        # Each conversation should have one turn
        for conv in conversations:
            assert len(conv.turns) == 1

    # ============================================================================
    # Data Field Handling Tests
    # ============================================================================

    def test_convert_with_timestamps(self, mock_prompt_generator):
        """Test conversion preserving timestamp values.

        Verifies that timestamp values from trace data are correctly preserved
        in the resulting Turn objects during conversion.
        """
        trace_data = TraceCustomData(
            timestamp=1500, input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.timestamp == 1500
        assert turn.delay is None

    def test_convert_with_delays(self, mock_prompt_generator):
        """Test conversion preserving delay values.

        Verifies that delay values from trace data are correctly preserved
        in the resulting Turn objects during conversion.
        """
        trace_data = TraceCustomData(
            session_id="session-1",
            delay=2000,
            input_length=300,
            output_length=40,
            hash_ids=[123, 456],
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.delay == 2000
        assert turn.timestamp is None

    def test_convert_with_hash_ids(self, mock_prompt_generator):
        """Test conversion with hash_ids passed to prompt generator.

        Verifies that hash_ids from trace data are correctly passed to the
        prompt generator during the conversion process.
        """
        hash_ids = [123, 456, 789]
        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=hash_ids
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        # Verify prompt generator was called with correct hash_ids
        mock_prompt_generator.generate.assert_called_once_with(
            mean=300, stddev=0, hash_ids=hash_ids
        )

    def test_convert_input_output_lengths(self, mock_prompt_generator):
        """Test that input_length is used as mean for prompt generation.

        Verifies that the input_length field from trace data is correctly
        used as the mean parameter when calling the prompt generator.
        """
        trace_data = TraceCustomData(
            input_length=500, output_length=100, hash_ids=[1, 2, 3]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        # Verify prompt generator was called with input_length as mean
        mock_prompt_generator.generate.assert_called_once_with(
            mean=500, stddev=0, hash_ids=[1, 2, 3]
        )

    # ============================================================================
    # Prompt Generation Integration Tests
    # ============================================================================

    def test_convert_calls_prompt_generator_correctly(self, mock_prompt_generator):
        """Test that prompt generator is called with correct parameters.

        Verifies that the prompt generator is called with the correct
        mean, stddev, and hash_ids parameters for each trace entry.
        """
        trace_data1 = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        trace_data2 = TraceCustomData(
            input_length=250, output_length=35, hash_ids=[789, 101]
        )
        data = {"session-1": [trace_data1, trace_data2]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        # Should be called twice, once for each trace
        assert mock_prompt_generator.generate.call_count == 2
        calls = mock_prompt_generator.generate.call_args_list

        # Check first call
        assert calls[0][1]["mean"] == 300
        assert calls[0][1]["stddev"] == 0
        assert calls[0][1]["hash_ids"] == [123, 456]

        # Check second call
        assert calls[1][1]["mean"] == 250
        assert calls[1][1]["stddev"] == 0
        assert calls[1][1]["hash_ids"] == [789, 101]

    def test_convert_uses_generated_prompts(self, mock_prompt_generator):
        """Test that generated prompts are included in conversation turns.

        Verifies that the prompts returned by the prompt generator are
        correctly included in the Text content of the resulting Turns.
        """
        mock_prompt_generator.generate.return_value = "Custom generated prompt"

        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.text[0].content == ["Custom generated prompt"]

    def test_convert_with_zero_stddev(self, mock_prompt_generator):
        """Test that stddev=0 is always passed to prompt generator.

        Verifies that the prompt generator is always called with stddev=0
        regardless of the trace data content.
        """
        trace_data = TraceCustomData(
            input_length=500, output_length=100, hash_ids=[1, 2, 3]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        # Always uses stddev=0
        mock_prompt_generator.generate.assert_called_once_with(
            mean=500, stddev=0, hash_ids=[1, 2, 3]
        )

    # ============================================================================
    # Conversation Structure Validation Tests
    # ============================================================================

    def test_convert_creates_correct_conversation_structure(
        self, mock_prompt_generator
    ):
        """Test that Conversation objects have correct session_ids.

        Verifies that the resulting Conversation objects have the correct
        session_id values matching the input trace data.
        """
        trace_data1 = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        trace_data2 = TraceCustomData(
            input_length=250, output_length=35, hash_ids=[789, 101]
        )
        data = {"session-alpha": [trace_data1], "session-beta": [trace_data2]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        session_ids = {conv.session_id for conv in conversations}
        assert session_ids == {"session-alpha", "session-beta"}

        # Each conversation should be properly structured
        for conv in conversations:
            assert isinstance(conv, Conversation)
            assert isinstance(conv.session_id, str)
            assert len(conv.turns) == 1

    def test_convert_creates_correct_turn_structure(self, mock_prompt_generator):
        """Test that Turn objects have correct timestamp/delay/text fields.

        Verifies that the resulting Turn objects have the correct structure
        with properly set timestamp, delay, and text fields.
        """
        trace_data = TraceCustomData(
            timestamp=1500, input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert isinstance(turn, Turn)
        assert turn.timestamp == 1500
        assert turn.delay is None
        assert len(turn.text) == 1
        assert len(turn.image) == 0  # No image data
        assert len(turn.audio) == 0  # No audio data

    def test_convert_creates_correct_text_structure(self, mock_prompt_generator):
        """Test that Text objects have correct name and content.

        Verifies that the resulting Text objects have the correct name ("text")
        and content containing the generated prompts.
        """
        mock_prompt_generator.generate.return_value = "Test generated content"

        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        text = conversations[0].turns[0].text[0]
        assert isinstance(text, Text)
        assert text.name == "text"
        assert text.content == ["Test generated content"]
        assert len(text.content) == 1


class TestTraceDatasetLoaderIntegration:
    """Test suite for end-to-end integration scenarios."""

    # ============================================================================
    # End-to-End Workflow Tests
    # ============================================================================

    def test_full_workflow_fixed_schedule(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test complete workflow with timestamp-based trace data.

        Verifies the complete end-to-end workflow from loading a trace file
        with fixed schedule format to producing final conversation objects.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["timestamp_format"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            # Load dataset
            data = loader.load_dataset()
            assert len(data) == 1

            # Convert to conversations
            conversations = loader.convert_to_conversations(data)
            assert len(conversations) == 1

            conv = conversations[0]
            assert len(conv.turns) == 1
            assert conv.turns[0].timestamp == 1000
            assert conv.turns[0].text[0].content == ["Generated test prompt"]
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_full_workflow_session_based(
        self, mock_prompt_generator, sample_trace_data
    ):
        """Test complete workflow with session-based trace data.

        Verifies the complete end-to-end workflow from loading a trace file
        with session-based format to producing final conversation objects.
        """
        session_data = sample_trace_data["session_format"]
        content = json.dumps(session_data) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            # Load dataset
            data = loader.load_dataset()
            assert len(data) == 1
            assert "session-123" in data

            # Convert to conversations
            conversations = loader.convert_to_conversations(data)
            assert len(conversations) == 1

            conv = conversations[0]
            assert conv.session_id == "session-123"
            assert len(conv.turns) == 1
            assert conv.turns[0].delay == 1000
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_full_workflow_mixed_data(self):
        """Test complete workflow with mixed trace formats.

        Verifies the complete end-to-end workflow from loading a trace file
        with mixed data formats to producing final conversation objects.
        """
        pass

    # ============================================================================
    # Real Data Scenario Tests
    # ============================================================================

    def test_with_realistic_trace_file(self):
        """Test with realistic trace file containing various session patterns.

        Verifies that the loader works correctly with realistic trace files
        that contain complex session patterns and data variations.
        """
        pass

    def test_with_large_input_lengths(self):
        """Test with large input_length values.

        Verifies that the loader handles trace data with large input_length
        values correctly without performance or memory issues.
        """
        pass

    def test_with_complex_hash_ids(self):
        """Test with various hash_ids patterns.

        Verifies that the loader handles trace data with complex hash_ids
        patterns correctly and passes them to the prompt generator.
        """
        pass


class TestTraceDatasetLoaderEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    # ============================================================================
    # Data Validation Edge Cases
    # ============================================================================

    def test_empty_data_conversion(self, mock_prompt_generator):
        """Test converting empty data dictionary.

        Verifies that converting an empty data dictionary returns an empty
        list of conversations and handles the edge case gracefully.
        """
        data = {}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert isinstance(conversations, list)
        assert len(conversations) == 0

    def test_session_with_empty_traces(self, mock_prompt_generator):
        """Test session with empty trace list.

        Verifies that a session with an empty list of traces is handled
        correctly and produces an appropriate conversation structure.
        """
        data = {"session-1": []}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert conv.session_id == "session-1"
        assert len(conv.turns) == 0

    def test_trace_with_minimal_required_fields(self, mock_prompt_generator):
        """Test traces with only required fields.

        Verifies that trace data containing only the minimum required fields
        is processed correctly without optional field dependencies.
        """
        trace_data = TraceCustomData(
            input_length=100, output_length=20, hash_ids=[1, 2, 3]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 1

        turn = conv.turns[0]
        assert turn.timestamp is None
        assert turn.delay is None
        assert len(turn.text) == 1

    # ============================================================================
    # UUID Generation Edge Cases
    # ============================================================================

    def test_uuid_generation_for_missing_session_id(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test that UUID is generated when session_id is None.

        Verifies that when session_id is None in trace data, a UUID is
        automatically generated and used as the session identifier.
        """
        temp_file = create_temp_file_with_content(
            sample_jsonl_data["single_session_implicit"]
        )

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            # Should have one session with auto-generated UUID
            assert len(data) == 1
            session_id = list(data.keys())[0]

            # Verify it's a valid UUID
            uuid_obj = uuid.UUID(session_id)
            assert str(uuid_obj) == session_id
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_uuid_uniqueness(self, mock_prompt_generator):
        """Test that different traces without session_id get unique UUIDs.

        Verifies that multiple trace entries without session_id each get
        unique UUID values and don't share the same session identifier.
        """
        # Create multiple lines without session_id
        content = '{"input_length": 100, "output_length": 20, "hash_ids": [1, 2]}\n'
        content += '{"input_length": 200, "output_length": 30, "hash_ids": [3, 4]}\n'
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            # Should have two different sessions with unique UUIDs
            assert len(data) == 2
            session_ids = list(data.keys())

            # Verify both are valid UUIDs
            for session_id in session_ids:
                uuid.UUID(session_id)

            # Verify they're different
            assert session_ids[0] != session_ids[1]
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_mixed_explicit_and_generated_session_ids(
        self, mock_prompt_generator, sample_jsonl_data
    ):
        """Test mixing explicit and auto-generated session IDs.

        Verifies that trace data with mixed explicit session_ids and auto-generated
        UUIDs are handled correctly and don't interfere with each other.
        """
        temp_file = create_temp_file_with_content(sample_jsonl_data["mixed_sessions"])

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 2
            assert "session-1" in data

            # One explicit session, one auto-generated
            session_keys = list(data.keys())
            auto_generated_key = [k for k in session_keys if k != "session-1"][0]
            uuid.UUID(auto_generated_key)  # Should be valid UUID

            assert len(data["session-1"]) == 1
            assert len(data[auto_generated_key]) == 1
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestTraceDatasetLoaderMocking:
    """Test suite for dependency mocking and interaction testing."""

    # ============================================================================
    # PromptGenerator Mocking Tests
    # ============================================================================

    def test_prompt_generator_called_with_correct_parameters(
        self, mock_prompt_generator
    ):
        """Test prompt generator receives correct mean/stddev/hash_ids.

        Verifies that the prompt generator's generate method is called with
        the correct parameters extracted from the trace data.
        """
        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        mock_prompt_generator.generate.assert_called_once_with(
            mean=300, stddev=0, hash_ids=[123, 456]
        )

    def test_prompt_generator_return_value_used(self, mock_prompt_generator):
        """Test that prompt generator's return value is used in Text content.

        Verifies that the string returned by the prompt generator is correctly
        used as the content in the resulting Text objects.
        """
        mock_prompt_generator.generate.return_value = "Custom test prompt"

        trace_data = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        data = {"session-1": [trace_data]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations(data)

        text_content = conversations[0].turns[0].text[0].content
        assert text_content == ["Custom test prompt"]

    def test_prompt_generator_called_for_each_trace(self, mock_prompt_generator):
        """Test prompt generator called once per trace.

        Verifies that the prompt generator's generate method is called exactly
        once for each trace entry in the data.
        """
        trace_data1 = TraceCustomData(
            input_length=300, output_length=40, hash_ids=[123, 456]
        )
        trace_data2 = TraceCustomData(
            input_length=250, output_length=35, hash_ids=[789, 101]
        )
        trace_data3 = TraceCustomData(
            input_length=200, output_length=30, hash_ids=[111, 222]
        )
        data = {"session-1": [trace_data1, trace_data2], "session-2": [trace_data3]}

        loader = TraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        loader.convert_to_conversations(data)

        # Should be called 3 times total (once per trace)
        assert mock_prompt_generator.generate.call_count == 3

    # ============================================================================
    # File System Mocking Tests
    # ============================================================================

    def test_file_reading_with_different_encodings(self, mock_prompt_generator):
        """Test file reading with various text encodings.

        Verifies that trace files with different text encodings (UTF-8, ASCII, etc.)
        are read correctly and processed without encoding errors.
        """
        # Create test data with unicode characters
        test_data = {"input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
        content = json.dumps(test_data, ensure_ascii=False) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            assert len(data) == 1
            session_id = list(data.keys())[0]
            assert len(data[session_id]) == 1
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_file_reading_with_large_files(self, mock_prompt_generator):
        """Test performance with large trace files.

        Verifies that the loader can handle large trace files efficiently
        without memory issues or performance degradation.
        """
        # Create a file with multiple entries
        lines = []
        for i in range(100):  # 100 entries should be manageable for testing
            line = json.dumps(
                {
                    "session_id": f"session-{i % 10}",  # 10 different sessions
                    "input_length": 300 + i,
                    "output_length": 40 + i,
                    "hash_ids": [i, i + 1, i + 2],
                }
            )
            lines.append(line)

        content = "\n".join(lines) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)
            data = loader.load_dataset()

            # Should have 10 sessions with 10 entries each
            assert len(data) == 10
            total_entries = sum(len(traces) for traces in data.values())
            assert total_entries == 100
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestTraceCustomDataValidation:
    """Test suite for TraceCustomData model validation logic."""

    # ============================================================================
    # Valid Data Creation Tests
    # ============================================================================

    def test_trace_custom_data_with_timestamp(self):
        """Test creating TraceCustomData with timestamp.

        Verifies that TraceCustomData objects can be created with timestamp
        values and that the validation rules are correctly applied.
        """
        trace_data = TraceCustomData(
            timestamp=1500, input_length=300, output_length=40, hash_ids=[123, 456]
        )

        assert trace_data.timestamp == 1500
        assert trace_data.session_id is None
        assert trace_data.delay is None
        assert trace_data.input_length == 300
        assert trace_data.output_length == 40
        assert trace_data.hash_ids == [123, 456]

    def test_trace_custom_data_with_session_and_delay(self):
        """Test creating TraceCustomData with session_id and delay.

        Verifies that TraceCustomData objects can be created with session_id
        and delay values and that the validation rules are correctly applied.
        """
        trace_data = TraceCustomData(
            session_id="session-123",
            delay=2000,
            input_length=250,
            output_length=35,
            hash_ids=[789, 101],
        )

        assert trace_data.session_id == "session-123"
        assert trace_data.delay == 2000
        assert trace_data.timestamp is None
        assert trace_data.input_length == 250
        assert trace_data.output_length == 35
        assert trace_data.hash_ids == [789, 101]

    def test_trace_custom_data_all_fields(self):
        """Test creating TraceCustomData with all valid field combinations.

        Verifies that TraceCustomData objects can be created with all possible
        valid field combinations without validation errors.
        """
        # Test minimal required fields
        trace_minimal = TraceCustomData(
            input_length=100, output_length=20, hash_ids=[1, 2, 3]
        )
        assert trace_minimal.input_length == 100

        # Test with session_id and delay
        trace_session = TraceCustomData(
            session_id="test-session",
            delay=1000,
            input_length=200,
            output_length=30,
            hash_ids=[4, 5, 6],
        )
        assert trace_session.session_id == "test-session"
        assert trace_session.delay == 1000

        # Test with timestamp only
        trace_timestamp = TraceCustomData(
            timestamp=5000, input_length=300, output_length=40, hash_ids=[7, 8, 9]
        )
        assert trace_timestamp.timestamp == 5000

    # ============================================================================
    # Validation Rules Tests
    # ============================================================================

    def test_mutually_exclusive_timestamp_session_id(self):
        """Test that timestamp and session_id cannot both be set.

        Verifies that TraceCustomData validation correctly prevents both
        timestamp and session_id from being set simultaneously.
        """
        with pytest.raises(ValidationError) as exc_info:
            TraceCustomData(
                timestamp=1000,
                session_id="session-123",
                input_length=300,
                output_length=40,
                hash_ids=[123, 456],
            )

        assert "timestamp and session_id cannot both be set" in str(exc_info.value)

    def test_mutually_exclusive_timestamp_delay(self):
        """Test that timestamp and delay cannot both be set.

        Verifies that TraceCustomData validation correctly prevents both
        timestamp and delay from being set simultaneously.
        """
        with pytest.raises(ValidationError) as exc_info:
            TraceCustomData(
                timestamp=1000,
                delay=2000,
                input_length=300,
                output_length=40,
                hash_ids=[123, 456],
            )

        assert "timestamp and delay cannot both be set" in str(exc_info.value)

    def test_required_fields_validation(self):
        """Test that required fields are enforced.

        Verifies that TraceCustomData validation correctly enforces the
        presence of required fields (input_length, output_length, hash_ids).
        """
        # Test missing input_length
        with pytest.raises(ValidationError):
            TraceCustomData(output_length=40, hash_ids=[123, 456])

        # Test missing output_length
        with pytest.raises(ValidationError):
            TraceCustomData(input_length=300, hash_ids=[123, 456])

        # Test missing hash_ids
        with pytest.raises(ValidationError):
            TraceCustomData(input_length=300, output_length=40)


class TestTraceDatasetLoaderPerformance:
    """Test suite for performance characteristics and resource usage."""

    def test_memory_usage_with_large_datasets(self, mock_prompt_generator):
        """Test memory efficiency with large trace files.

        Verifies that the loader uses memory efficiently when processing
        large trace files and doesn't cause memory leaks or excessive usage.
        """
        # Create a moderately large dataset for memory testing
        lines = []
        for i in range(50):
            line = json.dumps(
                {
                    "session_id": f"session-{i}",
                    "input_length": 300,
                    "output_length": 40,
                    "hash_ids": [i, i + 1, i + 2],
                }
            )
            lines.append(line)

        content = "\n".join(lines) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            # Load and convert data
            data = loader.load_dataset()
            conversations = loader.convert_to_conversations(data)

            # Verify results are reasonable
            assert len(data) == 50
            assert len(conversations) == 50

            # Clean up references to test memory behavior
            del data
            del conversations
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_loading_performance(self, mock_prompt_generator):
        """Test loading performance benchmarks.

        Verifies that the loader performs within acceptable time limits
        for various file sizes and provides consistent performance.
        """
        import time

        # Create test data
        lines = []
        for i in range(20):  # Keep small for quick testing
            line = json.dumps(
                {
                    "session_id": f"session-{i % 5}",
                    "input_length": 300,
                    "output_length": 40,
                    "hash_ids": [i, i + 1, i + 2],
                }
            )
            lines.append(line)

        content = "\n".join(lines) + "\n"
        temp_file = create_temp_file_with_content(content)

        try:
            loader = TraceDatasetLoader(temp_file, mock_prompt_generator)

            # Measure loading time
            start_time = time.time()
            data = loader.load_dataset()
            load_time = time.time() - start_time

            # Measure conversion time
            start_time = time.time()
            conversations = loader.convert_to_conversations(data)
            convert_time = time.time() - start_time

            # Verify performance is reasonable (should be very fast for small dataset)
            assert load_time < 1.0  # Should load in less than 1 second
            assert convert_time < 1.0  # Should convert in less than 1 second

            # Verify correctness
            assert len(conversations) == 5  # 5 unique sessions
        finally:
            Path(temp_file).unlink(missing_ok=True)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


@pytest.fixture
def mock_prompt_generator():
    """Create a mock PromptGenerator with configurable return values.

    Returns:
        Mock: A mock PromptGenerator object that can be configured for testing.
    """
    mock_gen = Mock(spec=PromptGenerator)
    mock_gen.generate.return_value = "Generated test prompt"
    return mock_gen


@pytest.fixture
def sample_trace_data():
    """Provide various sample trace data configurations.

    Returns:
        dict: Dictionary containing different trace data patterns for testing.
    """
    return {
        "timestamp_format": {
            "timestamp": 1000,
            "input_length": 300,
            "output_length": 40,
            "hash_ids": [123, 456],
        },
        "session_format": {
            "session_id": "session-123",
            "delay": 1000,
            "input_length": 250,
            "output_length": 35,
            "hash_ids": [789, 101112],
        },
        "minimal_format": {
            "input_length": 100,
            "output_length": 20,
            "hash_ids": [1, 2, 3],
        },
    }


@pytest.fixture
def sample_jsonl_data():
    """Create sample JSONL data strings for different scenarios.

    Returns:
        dict: Dictionary of scenario names to JSONL content strings.
    """
    return {
        "single_session_explicit": '{"session_id": "session-1", "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}\n',
        "single_session_implicit": '{"input_length": 250, "output_length": 35, "hash_ids": [789, 101]}\n',
        "multiple_sessions": (
            '{"session_id": "session-1", "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}\n'
            '{"session_id": "session-2", "input_length": 250, "output_length": 35, "hash_ids": [789, 101]}\n'
            '{"session_id": "session-1", "input_length": 200, "output_length": 30, "hash_ids": [222, 333]}\n'
        ),
        "timestamp_format": (
            '{"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}\n'
        ),
        "mixed_sessions": (
            '{"session_id": "session-1", "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}\n'
            '{"input_length": 250, "output_length": 35, "hash_ids": [789, 101]}\n'
        ),
        "empty": "",
        "invalid_json": '{"session_id": "session-1", "input_length": 300, "output_length": 40, "hash_ids": [123, 456\n',
        "invalid_data": '{"session_id": "session-1", "timestamp": 1000, "delay": 500, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}\n',
    }


@pytest.fixture
def temp_trace_file():
    """Create a temporary trace file for testing.

    Returns:
        str: Path to temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def create_temp_file_with_content(content):
    """Helper function to create temporary file with specific content.

    Args:
        content: String content to write to file.

    Returns:
        str: Path to temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        return f.name
