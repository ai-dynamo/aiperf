# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for DatasetManager._generate_inputs_json_file method.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.models import InputsFile, SessionPayloads
from aiperf.plugin import plugins


def _validate_chat_payload_structure(payload: dict) -> None:
    """Helper function to validate chat payload structure."""
    assert "messages" in payload
    assert "model" in payload
    assert "stream" in payload
    assert isinstance(payload["messages"], list)
    assert len(payload["messages"]) > 0
    for message in payload["messages"]:
        assert "role" in message
        assert "content" in message


def _validate_inputs_file_structure(content: dict) -> None:
    """Helper function to validate InputsFile structure."""
    assert "data" in content
    assert isinstance(content["data"], list)
    for session in content["data"]:
        assert "session_id" in session
        assert "payloads" in session
        assert isinstance(session["payloads"], list)
        for payload in session["payloads"]:
            _validate_chat_payload_structure(payload)


class TestDatasetManagerInputsJsonGeneration:
    """Test suite for inputs.json file generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_inputs_json_success_with_populated_dataset(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test comprehensive successful generation with populated dataset."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        _validate_inputs_file_structure(written_json)

        # Verify specific dataset content
        assert len(written_json["data"]) == 2
        sessions = {session["session_id"]: session for session in written_json["data"]}
        assert "session_1" in sessions
        assert "session_2" in sessions

        # Verify turn counts match conversation structure
        assert len(sessions["session_1"]["payloads"]) == 2  # 2 turns
        assert len(sessions["session_2"]["payloads"]) == 1  # 1 turn

        # Verify user config is applied
        for session in written_json["data"]:
            for payload in session["payloads"]:
                assert payload["model"] == "test-model"
                assert payload["stream"] is False

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
    async def test_generate_inputs_json_file_path_and_io(
        self,
        populated_dataset_manager,
        tmp_path: Path,
    ):
        """Test file creation in correct location and valid JSON output."""
        populated_dataset_manager.user_config.output.artifact_directory = tmp_path

        await populated_dataset_manager._generate_inputs_json_file()

        expected_path = tmp_path / OutputDefaults.INPUTS_JSON_FILE
        assert expected_path.exists()

        with open(expected_path) as f:
            content = json.load(f)
        _validate_inputs_file_structure(content)

    @pytest.mark.asyncio
    async def test_generate_inputs_json_session_order_preservation(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that sessions are preserved in dataset iteration order."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        session_ids = [session["session_id"] for session in written_json["data"]]
        expected_order = list(populated_dataset_manager.dataset.keys())
        assert session_ids == expected_order

    @pytest.mark.asyncio
    async def test_generate_inputs_json_custom_field_preservation(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that custom fields like max_completion_tokens are preserved."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        session_2 = next(
            session
            for session in written_json["data"]
            if session["session_id"] == "session_2"
        )

        payload = session_2["payloads"][0]
        assert "max_completion_tokens" in payload
        assert payload["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_inputs_json_pydantic_model_compatibility(
        self,
        populated_dataset_manager,
        capture_file_writes,
    ):
        """Test that generated content is compatible with InputsFile Pydantic model."""
        await populated_dataset_manager._generate_inputs_json_file()

        written_json = json.loads(capture_file_writes.written_content)
        inputs_file = InputsFile.model_validate(written_json)

        assert isinstance(inputs_file, InputsFile)
        assert len(inputs_file.data) == 2
        assert all(isinstance(session, SessionPayloads) for session in inputs_file.data)

    @pytest.mark.asyncio
    async def test_generate_inputs_json_plugin_creation_error(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test error handling when plugin class creation fails."""
        with patch.object(
            plugins,
            "get_class",
            side_effect=Exception("Plugin error"),
        ):
            with pytest.raises(Exception, match="Plugin error"):
                await populated_dataset_manager._generate_inputs_json_file()
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
        """Test error handling when file I/O operation fails."""
        with patch(
            "pathlib.Path.write_bytes",
            side_effect=OSError("Permission denied"),
        ):
            await populated_dataset_manager._generate_inputs_json_file()
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
        """Test error handling when payload conversion fails."""
        mock_converter = Mock()
        mock_converter.format_payload = Mock(
            side_effect=Exception("Payload conversion error")
        )
        mock_converter.get_endpoint_headers = Mock(return_value={})
        mock_converter.get_endpoint_params = Mock(return_value={})

        with patch.object(
            plugins,
            "get_class",
            return_value=lambda **kwargs: mock_converter,
        ):
            with pytest.raises(Exception, match="Payload conversion error"):
                await populated_dataset_manager._generate_inputs_json_file()
            assert any(
                "Error generating inputs.json file" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_generate_inputs_json_logging(
        self,
        populated_dataset_manager,
        caplog,
    ):
        """Test that appropriate log messages are generated."""
        caplog.set_level(logging.INFO)

        await populated_dataset_manager._generate_inputs_json_file()

        log_messages = [record.message for record in caplog.records]
        assert any("Generating inputs.json file" in msg for msg in log_messages)
        assert any("inputs.json file generated" in msg for msg in log_messages)
