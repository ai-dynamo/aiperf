# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from rich.logging import RichHandler

from aiperf.common.aiperf_logger import _DEBUG, _TRACE
from aiperf.common.enums import ServiceType
from aiperf.common.logging import (
    StructuredSubprocessLogHandler,
    create_file_handler,
    handle_subprocess_log_line,
    parse_subprocess_log_line,
    setup_logging,
)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_basic(self, cleanup_logging):
        """Test basic setup_logging functionality."""
        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", False):
            setup_logging(ServiceType.DATASET_MANAGER, "test_service")

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], RichHandler)

    def test_setup_logging_structured(self, cleanup_logging):
        """Test setup_logging with structured logging enabled."""
        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", True):
            setup_logging(ServiceType.TIMING_MANAGER, "test_service")

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], StructuredSubprocessLogHandler)

    @pytest.mark.parametrize(
        "service_type,services_set,expected_level",
        [
            (ServiceType.DATASET_MANAGER, {ServiceType.DATASET_MANAGER}, _DEBUG),
            (ServiceType.TIMING_MANAGER, {ServiceType.TIMING_MANAGER}, _TRACE),
        ],
    )
    def test_setup_logging_with_log_levels(
        self, cleanup_logging, service_type, services_set, expected_level
    ):
        """Test setup_logging with debug and trace services."""
        kwarg = "debug_services" if expected_level == _DEBUG else "trace_services"

        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", False):
            setup_logging(service_type, "test_service", **{kwarg: services_set})

        root_logger = logging.getLogger()
        assert root_logger.level == expected_level

    def test_setup_logging_with_file_handler(self, cleanup_logging, temp_log_folder):
        """Test setup_logging with file logging."""
        setup_logging(
            ServiceType.DATASET_MANAGER, "test_service", log_folder=temp_log_folder
        )

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2
        handler_types = [type(h) for h in root_logger.handlers]
        assert RichHandler in handler_types
        assert logging.FileHandler in handler_types

    def test_setup_logging_removes_existing_handlers(self, cleanup_logging):
        """Test that setup_logging removes existing handlers."""
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)

        setup_logging(ServiceType.DATASET_MANAGER, "test_service")

        assert dummy_handler not in root_logger.handlers


class TestCreateFileHandler:
    """Test create_file_handler function."""

    def test_create_file_handler_creates_directory(self, temp_log_folder):
        """Test that create_file_handler creates log directory."""
        assert not temp_log_folder.exists()

        handler = create_file_handler(temp_log_folder, "INFO")

        assert temp_log_folder.exists()
        assert isinstance(handler, logging.FileHandler)
        assert handler.level == logging.INFO

    @pytest.mark.parametrize(
        "level_str,expected_level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
        ],
    )
    def test_create_file_handler_levels(
        self, temp_log_folder, level_str, expected_level
    ):
        """Test create_file_handler with different log levels."""
        temp_log_folder.mkdir(exist_ok=True)

        handler = create_file_handler(temp_log_folder, level_str)

        assert isinstance(handler, logging.FileHandler)
        assert handler.level == expected_level

    def test_create_file_handler_formatter(self, temp_log_folder):
        """Test that create_file_handler sets proper formatter."""
        handler = create_file_handler(temp_log_folder, "WARNING")

        formatter = handler.formatter
        assert formatter is not None
        for field in ["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"]:
            assert field in formatter._fmt


class TestStructuredSubprocessLogHandler:
    """Test StructuredSubprocessLogHandler class."""

    @pytest.mark.parametrize(
        "service_id,expected_id",
        [
            ("test_service_123", "test_service_123"),
            (None, ""),
        ],
    )
    def test_structured_handler_initialization(self, service_id, expected_id):
        """Test StructuredSubprocessLogHandler initialization."""
        handler = StructuredSubprocessLogHandler(service_id)

        assert handler.service_id == expected_id
        assert handler.process_id is not None
        assert handler.process_name is not None

    @patch("builtins.print")
    def test_structured_handler_emit_success(self, mock_print, create_log_record):
        """Test StructuredSubprocessLogHandler emit method."""
        handler = StructuredSubprocessLogHandler("test_service")
        record = create_log_record()

        handler.emit(record)

        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]

        log_data = json.loads(output)
        assert log_data["levelno"] == logging.INFO
        assert log_data["levelname"] == "INFO"
        assert log_data["name"] == "test_logger"
        assert log_data["service_id"] == "test_service"
        assert log_data["pathname"] == "/test/path.py"
        assert log_data["lineno"] == 42
        assert log_data["msg"] == "Test message"

    @patch("builtins.print")
    def test_structured_handler_emit_exception_handling(self, mock_print):
        """Test StructuredSubprocessLogHandler handles exceptions gracefully."""
        handler = StructuredSubprocessLogHandler("test_service")

        record = MagicMock()
        record.getMessage.side_effect = Exception("Test error")

        handler.emit(record)

        mock_print.assert_not_called()


class TestParseSubprocessLogLine:
    """Test parse_subprocess_log_line function."""

    def test_parse_subprocess_log_line_valid_json(self, create_log_data):
        """Test parsing valid JSON log line."""
        log_data = create_log_data(
            name="test_logger",
            levelno=logging.WARNING,
            levelname="WARNING",
            msg="Test warning message",
        )

        json_line = json.dumps(log_data)
        record = parse_subprocess_log_line(json_line)

        assert record is not None
        assert record.name == "test_logger"
        assert record.levelno == logging.WARNING
        assert record.levelname == "WARNING"
        assert record.pathname == "/test/path.py"
        assert record.lineno == 100
        assert record.getMessage() == "Test warning message"
        assert record.service_id == "test_service"
        assert record.processName == "TestProcess"
        assert record.process == 12345

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "This is not valid JSON",
            "",
        ],
    )
    def test_parse_subprocess_log_line_invalid_input(self, invalid_input):
        """Test parsing invalid JSON or empty string returns None."""
        record = parse_subprocess_log_line(invalid_input)
        assert record is None

    def test_parse_subprocess_log_line_partial_data(self):
        """Test parsing JSON with missing optional fields."""
        log_data = {
            "name": "minimal_logger",
            "levelno": logging.ERROR,
            "msg": "Minimal log entry",
        }

        json_line = json.dumps(log_data)
        record = parse_subprocess_log_line(json_line)

        assert record is not None
        assert record.name == "minimal_logger"
        assert record.levelno == logging.ERROR
        assert record.getMessage() == "Minimal log entry"
        assert record.pathname == ""
        assert record.lineno == 0
        assert record.service_id == ""


class TestHandleSubprocessLogLine:
    """Test handle_subprocess_log_line function."""

    def test_handle_subprocess_log_line_structured_success(
        self, cleanup_logging, create_log_data
    ):
        """Test handling structured log line successfully."""
        test_logger = logging.getLogger("test_structured_logger")
        test_handler = logging.StreamHandler()
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.INFO)

        log_data = create_log_data(
            name="test_structured_logger",
            levelno=logging.INFO,
            levelname="INFO",
            msg="Structured log message",
            service_id="structured_service",
        )

        json_line = json.dumps(log_data)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(json_line, "fallback_service")

        mock_handle.assert_called_once()
        record = mock_handle.call_args[0][0]
        assert record.name == "test_structured_logger"
        assert record.getMessage() == "Structured log message"

    def test_handle_subprocess_log_line_fallback_to_unstructured(self, cleanup_logging):
        """Test fallback to unstructured logging for invalid JSON."""
        fallback_service_id = "fallback_service"
        test_logger = logging.getLogger(fallback_service_id)
        test_logger.setLevel(logging.WARNING)

        unstructured_line = "This is a plain text log line"

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(unstructured_line, fallback_service_id)

        mock_handle.assert_called_once()
        record = mock_handle.call_args[0][0]
        assert record.name == fallback_service_id
        assert record.levelno == logging.WARNING
        assert record.getMessage() == unstructured_line

    @pytest.mark.parametrize(
        "logger_level,log_level,should_handle",
        [
            (logging.ERROR, logging.INFO, False),
            (logging.CRITICAL, logging.WARNING, False),
            (logging.INFO, logging.INFO, True),
            (logging.WARNING, logging.ERROR, True),
        ],
    )
    def test_handle_subprocess_log_line_level_filtering(
        self, cleanup_logging, create_log_data, logger_level, log_level, should_handle
    ):
        """Test that log level filtering works correctly."""
        logger_name = f"test_logger_{log_level}"
        test_logger = logging.getLogger(logger_name)
        test_logger.setLevel(logger_level)

        log_data = create_log_data(
            name=logger_name,
            levelno=log_level,
            msg=f"Message at level {log_level}",
        )

        json_line = json.dumps(log_data)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(json_line, "fallback_service")

        if should_handle:
            mock_handle.assert_called_once()
        else:
            mock_handle.assert_not_called()

    def test_handle_subprocess_log_line_empty_line_ignored(self, cleanup_logging):
        """Test that empty lines are ignored gracefully."""
        fallback_service_id = "fallback_service"
        test_logger = logging.getLogger(fallback_service_id)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line("", fallback_service_id)

        mock_handle.assert_not_called()
