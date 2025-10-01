# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def teardown_method(self):
        """Clean up logging configuration after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_logging_basic(self):
        """Test basic setup_logging functionality."""
        service_type = ServiceType.DATASET_MANAGER
        service_id = "test_service"

        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", False):
            setup_logging(service_type, service_id)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], RichHandler)

    def test_setup_logging_structured(self):
        """Test setup_logging with structured logging enabled."""
        service_type = ServiceType.TIMING_MANAGER
        service_id = "test_service"

        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", True):
            setup_logging(service_type, service_id)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], StructuredSubprocessLogHandler)

    def test_setup_logging_with_debug_services(self):
        """Test setup_logging with debug services."""
        service_type = ServiceType.DATASET_MANAGER
        debug_services = {ServiceType.DATASET_MANAGER}

        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", False):
            setup_logging(service_type, "test_service", debug_services=debug_services)

        root_logger = logging.getLogger()
        assert root_logger.level == _DEBUG

    def test_setup_logging_with_trace_services(self):
        """Test setup_logging with trace services."""
        service_type = ServiceType.TIMING_MANAGER
        trace_services = {ServiceType.TIMING_MANAGER}

        with patch("aiperf.common.logging.AIPERF_STRUCTURED_LOGGING", False):
            setup_logging(service_type, "test_service", trace_services=trace_services)

        root_logger = logging.getLogger()
        assert root_logger.level == _TRACE

    def test_setup_logging_with_file_handler(self):
        """Test setup_logging with file logging."""
        service_type = ServiceType.DATASET_MANAGER

        with tempfile.TemporaryDirectory() as temp_dir:
            log_folder = Path(temp_dir)

            setup_logging(service_type, "test_service", log_folder=log_folder)

            root_logger = logging.getLogger()
            # Should have both RichHandler and FileHandler
            assert len(root_logger.handlers) == 2
            handler_types = [type(h) for h in root_logger.handlers]
            assert RichHandler in handler_types
            assert logging.FileHandler in handler_types

    def test_setup_logging_removes_existing_handlers(self):
        """Test that setup_logging removes existing handlers."""
        # Add a dummy handler first
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)

        setup_logging(ServiceType.DATASET_MANAGER, "test_service")

        # Dummy handler should be removed
        assert dummy_handler not in root_logger.handlers


class TestCreateFileHandler:
    """Test create_file_handler function."""

    def test_create_file_handler_creates_directory(self):
        """Test that create_file_handler creates log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_folder = Path(temp_dir) / "logs"
            assert not log_folder.exists()

            handler = create_file_handler(log_folder, "INFO")

            assert log_folder.exists()
            assert isinstance(handler, logging.FileHandler)
            assert handler.level == logging.INFO

    def test_create_file_handler_with_existing_directory(self):
        """Test create_file_handler with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_folder = Path(temp_dir)
            log_folder.mkdir(exist_ok=True)

            handler = create_file_handler(log_folder, "DEBUG")

            assert isinstance(handler, logging.FileHandler)
            assert handler.level == logging.DEBUG

    def test_create_file_handler_formatter(self):
        """Test that create_file_handler sets proper formatter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_folder = Path(temp_dir)

            handler = create_file_handler(log_folder, "WARNING")

            formatter = handler.formatter
            assert formatter is not None
            assert "%(asctime)s" in formatter._fmt
            assert "%(name)s" in formatter._fmt
            assert "%(levelname)s" in formatter._fmt
            assert "%(message)s" in formatter._fmt


class TestStructuredSubprocessLogHandler:
    """Test StructuredSubprocessLogHandler class."""

    def test_structured_handler_initialization(self):
        """Test StructuredSubprocessLogHandler initialization."""
        service_id = "test_service_123"
        handler = StructuredSubprocessLogHandler(service_id)

        assert handler.service_id == service_id
        assert handler.process_id is not None
        assert handler.process_name is not None

    def test_structured_handler_initialization_no_service_id(self):
        """Test StructuredSubprocessLogHandler initialization without service_id."""
        handler = StructuredSubprocessLogHandler()

        assert handler.service_id == ""

    @patch("builtins.print")
    def test_structured_handler_emit_success(self, mock_print):
        """Test StructuredSubprocessLogHandler emit method."""
        handler = StructuredSubprocessLogHandler("test_service")

        # Create a mock log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_func",
            sinfo=None,
        )
        record.created = time.time()

        handler.emit(record)

        # Verify print was called with JSON output
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]

        # Parse the JSON output
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

        # Create a malformed record that might cause issues
        record = MagicMock()
        record.getMessage.side_effect = Exception("Test error")

        # Should not raise an exception
        handler.emit(record)

        # Print should not be called due to exception
        mock_print.assert_not_called()


class TestParseSubprocessLogLine:
    """Test parse_subprocess_log_line function."""

    def test_parse_subprocess_log_line_valid_json(self):
        """Test parsing valid JSON log line."""
        log_data = {
            "created": time.time(),
            "levelno": logging.WARNING,
            "levelname": "WARNING",
            "name": "test_logger",
            "process_name": "TestProcess",
            "process_id": 12345,
            "service_id": "test_service",
            "pathname": "/test/path.py",
            "lineno": 100,
            "msg": "Test warning message",
        }

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

    def test_parse_subprocess_log_line_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        invalid_json = "This is not valid JSON"

        record = parse_subprocess_log_line(invalid_json)

        assert record is None

    def test_parse_subprocess_log_line_empty_string(self):
        """Test parsing empty string returns None."""
        record = parse_subprocess_log_line("")

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
        # Should have defaults for missing fields
        assert record.pathname == ""
        assert record.lineno == 0
        assert record.service_id == ""


class TestHandleSubprocessLogLine:
    """Test handle_subprocess_log_line function."""

    def teardown_method(self):
        """Clean up logging configuration after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_handle_subprocess_log_line_structured_success(self):
        """Test handling structured log line successfully."""
        # Set up a logger to capture the output
        test_logger = logging.getLogger("test_structured_logger")
        test_handler = logging.StreamHandler()
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.INFO)

        log_data = {
            "created": time.time(),
            "levelno": logging.INFO,
            "levelname": "INFO",
            "name": "test_structured_logger",
            "msg": "Structured log message",
            "service_id": "structured_service",
        }

        json_line = json.dumps(log_data)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(json_line, "fallback_service")

        # Should call handle on the structured logger
        mock_handle.assert_called_once()
        record = mock_handle.call_args[0][0]
        assert record.name == "test_structured_logger"
        assert record.getMessage() == "Structured log message"

    def test_handle_subprocess_log_line_fallback_to_unstructured(self):
        """Test fallback to unstructured logging for invalid JSON."""
        fallback_service_id = "fallback_service"
        test_logger = logging.getLogger(fallback_service_id)
        test_logger.setLevel(logging.WARNING)

        unstructured_line = "This is a plain text log line"

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(unstructured_line, fallback_service_id)

        # Should call handle on the fallback logger
        mock_handle.assert_called_once()
        record = mock_handle.call_args[0][0]
        assert record.name == fallback_service_id
        assert record.levelno == logging.WARNING
        assert record.getMessage() == unstructured_line

    def test_handle_subprocess_log_line_logger_level_filtering(self):
        """Test that log level filtering works correctly."""
        test_logger = logging.getLogger("test_level_logger")
        test_logger.setLevel(logging.ERROR)  # Only ERROR and above

        log_data = {
            "levelno": logging.INFO,  # Below ERROR threshold
            "name": "test_level_logger",
            "msg": "Info message that should be filtered",
        }

        json_line = json.dumps(log_data)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(json_line, "fallback_service")

        # Should not call handle due to level filtering
        mock_handle.assert_not_called()

    def test_handle_subprocess_log_line_empty_line_ignored(self):
        """Test that empty lines are ignored gracefully."""
        fallback_service_id = "fallback_service"
        test_logger = logging.getLogger(fallback_service_id)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line("", fallback_service_id)

        # Should not log empty lines
        mock_handle.assert_not_called()

    def test_handle_subprocess_log_line_structured_with_disabled_logger(self):
        """Test structured logging with logger level filtering."""
        test_logger = logging.getLogger("disabled_logger")
        test_logger.setLevel(logging.CRITICAL)  # Very high threshold

        log_data = {
            "levelno": logging.WARNING,  # Below CRITICAL
            "name": "disabled_logger",
            "msg": "This should be filtered out",
        }

        json_line = json.dumps(log_data)

        with patch.object(test_logger, "handle") as mock_handle:
            handle_subprocess_log_line(json_line, "fallback_service")

        # Should not call handle due to level filtering
        mock_handle.assert_not_called()
