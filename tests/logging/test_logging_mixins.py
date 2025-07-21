# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.common.aiperf_logger import (
    _CRITICAL,
    _DEBUG,
    _ERROR,
    _INFO,
    _NOTICE,
    _SUCCESS,
    _TRACE,
    _WARNING,
    AIPerfLogger,
)
from aiperf.common.mixins import AIPerfLoggerMixin


class MockClass(AIPerfLoggerMixin):
    """Mock class for testing the mixin."""

    def __init__(self):
        super().__init__()


@pytest.fixture
def logger():
    """Create a logger instance for testing."""
    return AIPerfLogger("test_logger")


@pytest.fixture
def mock_class():
    """Create a mock class instance for testing the mixin."""
    return MockClass()


class TestAIPerfLoggerMixin:
    """Test cases for AIPerfLoggerMixin class."""

    def test_mixin_initialization(self, mock_class):
        """Test that mixin initializes correctly."""
        assert isinstance(mock_class.logger, AIPerfLogger)
        assert mock_class.logger._logger.name == "MockClass"

    @pytest.mark.parametrize(
        "method_name,level",
        [
            ("trace", _TRACE),
            ("debug", _DEBUG),
            ("info", _INFO),
            ("warning", _WARNING),
            ("notice", _NOTICE),
            ("success", _SUCCESS),
            ("error", _ERROR),
            ("critical", _CRITICAL),
        ],
    )
    def test_mixin_convenience_methods(self, mock_class, method_name, level):
        """Test that mixin convenience methods delegate to logger."""
        mock_class.logger.set_level(_TRACE)
        with patch.object(mock_class, "_log") as mock_log:
            method = getattr(mock_class, method_name)
            method(f"Test message {level}")

            mock_log.assert_called_once_with(level, f"Test message {level}")

    @pytest.mark.parametrize(
        "method_name,level,should_be_logged",
        [
            ("trace", _TRACE, False),
            ("debug", _DEBUG, False),
            ("info", _INFO, True),
            ("notice", _NOTICE, True),
            ("success", _SUCCESS, True),
            ("error", _ERROR, True),
            ("critical", _CRITICAL, True),
            ("exception", _ERROR, True),
        ],
    )
    def test_end_to_end_logging(self, caplog, method_name, level, should_be_logged):
        """Test end-to-end logging functionality."""
        caplog.set_level(_INFO)

        mock_class = MockClass()
        method = getattr(mock_class, method_name)
        method(f"Test message {level}")

        if should_be_logged:
            assert f"Test message {level}" in caplog.text
        else:
            assert f"Test message {level}" not in caplog.text

    def test_lazy_evaluation(self, caplog):
        """Test that lazy evaluation prevents expensive operations."""
        caplog.set_level(_INFO)

        mock_class = MockClass()
        expensive_operation_called = False

        def expensive_operation():
            nonlocal expensive_operation_called
            expensive_operation_called = True
            return "Expensive result"

        mock_class.debug(expensive_operation)

        assert not expensive_operation_called
        assert "Expensive result" not in caplog.text
