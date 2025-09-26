# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for DatasetManager._generate_inputs_json_file method.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.factories import RequestConverterFactory
from aiperf.common.models import InputsFile, SessionPayloads


class TestDatasetManagerInputsJsonGeneration:
    """Test suite for inputs.json file generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_inputs_json_success(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test successful generation of inputs.json file."""
        await populated_dataset_manager._generate_inputs_json_file()

        # Verify file content structure
        written_json = json.loads(capture_file_writes.written_content)
        assert "data" in written_json
        assert len(written_json["data"]) == 2

        # Verify session structure
        sessions = {session["session_id"]: session for session in written_json["data"]}
        assert "session_1" in sessions
        assert "session_2" in sessions

        # Verify session_1 has 2 payloads (2 turns)
        assert len(sessions["session_1"]["payloads"]) == 2
        # Verify session_2 has 1 payload (1 turn)
        assert len(sessions["session_2"]["payloads"]) == 1

        # Verify OpenAI chat format structure
        payload = sessions["session_1"]["payloads"][0]
        assert "messages" in payload
        assert "model" in payload
        assert "stream" in payload
        # Verify message structure
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) > 0
        assert "role" in payload["messages"][0]
        assert "content" in payload["messages"][0]

    @pytest.mark.asyncio
    async def test_generate_inputs_json_empty_dataset(
        self,
        empty_dataset_manager,
        capture_file_writes,
    ):
        """Test generation with empty dataset creates empty inputs file."""
        await empty_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        assert written_json == {"data": []}

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_path_construction(
        self,
        populated_dataset_manager,
        tmp_path,
    ):
        """Test that inputs.json file is created in correct artifact directory."""
        # Update the artifact directory
        populated_dataset_manager.user_config.output.artifact_directory = tmp_path

        await populated_dataset_manager._generate_inputs_json_file()

        # Verify file was created in correct location
        expected_path = tmp_path / OutputDefaults.INPUTS_JSON_FILE
        assert expected_path.exists()

        # Verify file content is valid JSON
        with open(expected_path) as f:
            content = json.load(f)
        assert isinstance(content, dict)
        assert "data" in content

        # Verify chat format structure
        for session in content["data"]:
            for payload in session["payloads"]:
                assert "messages" in payload
                assert isinstance(payload["messages"], list)

    @pytest.mark.asyncio
    async def test_generate_inputs_json_uses_correct_endpoint_type(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that the correct endpoint type produces proper payload format."""
        await populated_dataset_manager._generate_inputs_json_file()

        # Verify CHAT endpoint type creates proper chat payloads
        written_json = json.loads(capture_file_writes.written_content)
        for session in written_json["data"]:
            for payload in session["payloads"]:
                # Verify chat format structure
                assert "messages" in payload
                assert isinstance(payload["messages"], list)
                assert all(
                    "role" in msg and "content" in msg for msg in payload["messages"]
                )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_processes_all_turns(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that all turns in all conversations are processed correctly."""
        await populated_dataset_manager._generate_inputs_json_file()

        # session_1 has 2 turns, session_2 has 1 turn = 3 total payload entries
        written_json = json.loads(capture_file_writes.written_content)
        total_payloads = sum(
            len(session["payloads"]) for session in written_json["data"]
        )
        assert total_payloads == 3

        # Verify payloads have correct structure
        for session in written_json["data"]:
            for payload in session["payloads"]:
                assert "messages" in payload
                assert "model" in payload
                assert "stream" in payload
                assert isinstance(payload["messages"], list)
                assert len(payload["messages"]) > 0

    @pytest.mark.asyncio
    async def test_generate_inputs_json_respects_user_config(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that user configuration is properly reflected in the output."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)

        # Verify user config settings are applied
        for session in written_json["data"]:
            for payload in session["payloads"]:
                assert payload["model"] == "test-model"
                assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_error_handling(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test error handling when inputs.json generation fails."""
        with patch.object(
            RequestConverterFactory,
            "create_instance",
            side_effect=Exception("Factory error"),
        ):
            await populated_dataset_manager._generate_inputs_json_file()

            # Verify warning was logged
            assert any(
                "Error generating inputs.json file" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_file_io_error(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test error handling when file I/O fails."""

        def mock_open_error(*args, **kwargs):
            raise OSError("Permission denied")

        with patch("aiperf.dataset.dataset_manager.aiofiles.open", mock_open_error):
            await populated_dataset_manager._generate_inputs_json_file()

            # Verify warning was logged
            assert any(
                "Error generating inputs.json file" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_payload_conversion_error(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test error handling when payload formatting fails."""
        mock_converter = AsyncMock()
        mock_converter.format_payload = AsyncMock(
            side_effect=Exception("Conversion error")
        )

        with patch.object(
            RequestConverterFactory,
            "create_instance",
            return_value=mock_converter,
        ):
            await populated_dataset_manager._generate_inputs_json_file()

            # Verify warning was logged
            assert any(
                "Error generating inputs.json file" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_preserve_session_order(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that sessions are preserved in dataset iteration order."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        session_ids = [session["session_id"] for session in written_json["data"]]

        # Verify order matches dataset iteration order
        expected_order = list(populated_dataset_manager.dataset.keys())
        assert session_ids == expected_order

    @pytest.mark.asyncio
    async def test_generate_inputs_json_payload_structure_completeness(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that all required payload fields are included in generated payloads."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)

        # Find session_2 which has max_completion_tokens set
        session_2 = next(
            session
            for session in written_json["data"]
            if session["session_id"] == "session_2"
        )

        payload = session_2["payloads"][0]
        assert "max_completion_tokens" in payload
        assert payload["max_completion_tokens"] == 100

        # Verify chat structure
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) > 0
        assert "role" in payload["messages"][0]
        assert "content" in payload["messages"][0]

    @pytest.mark.asyncio
    async def test_generate_inputs_json_inputs_file_model_serialization(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that InputsFile model serialization works correctly."""
        await populated_dataset_manager._generate_inputs_json_file()

        # Verify content can be deserialized back to InputsFile model
        written_json = json.loads(capture_file_writes.written_content)
        inputs_file = InputsFile.model_validate(written_json)

        assert isinstance(inputs_file, InputsFile)
        assert len(inputs_file.data) == 2
        assert all(isinstance(session, SessionPayloads) for session in inputs_file.data)

        # Verify payload structure
        for session in inputs_file.data:
            assert session.session_id is not None
            assert len(session.payloads) > 0
            for payload in session.payloads:
                assert isinstance(payload, dict)
                assert "messages" in payload
                assert isinstance(payload["messages"], list)
                assert all(
                    "role" in msg and "content" in msg for msg in payload["messages"]
                )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_logging_behavior(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test that appropriate log messages are generated."""
        await populated_dataset_manager._generate_inputs_json_file()

        # Verify start and completion logs
        log_messages = [record.message for record in caplog.records]
        assert any("Generating inputs.json file" in msg for msg in log_messages)
        assert any("inputs.json file generated" in msg for msg in log_messages)
