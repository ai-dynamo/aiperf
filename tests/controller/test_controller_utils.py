# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.controller.controller_utils import (
    _group_errors_by_details,
    print_exit_errors,
)


def _create_test_console_output(width: int = 120) -> tuple[Console, StringIO]:
    """Helper to create console and output for testing."""
    output = StringIO()
    console = Console(file=output, width=width)
    return console, output


def _create_basic_error(
    error_type: str = "TestError",
    message: str = "Test message",
    operation: str = "Test Operation",
    service_id: str = "test_service",
) -> ExitErrorInfo:
    """Helper to create a basic test error."""
    return ExitErrorInfo(
        error_details=ErrorDetails(type=error_type, message=message),
        operation=operation,
        service_id=service_id,
    )


class TestPrintExitErrors:
    """Test the print_exit_errors function."""

    @pytest.mark.parametrize("errors", [None, []])
    def test_empty_input_handling(self, errors):
        """Test that print_exit_errors handles None and empty list gracefully."""
        # Should not raise any exception
        print_exit_errors(errors)

    @pytest.mark.parametrize(
        "service_id,expected_display",
        [
            ("test_service", "test_service"),
            (None, "N/A"),
            ("", "N/A"),
        ],
    )
    def test_service_id_handling(self, service_id, expected_display):
        """Test service_id display for various values."""
        error = _create_basic_error(service_id=service_id)
        console, output = _create_test_console_output(80)

        print_exit_errors([error], console)

        result = output.getvalue()
        assert f"• Service: {expected_display}" in result
        assert "Operation: Test Operation" in result
        assert "Error: TestError" in result
        assert "Reason: Test message" in result

    def test_multiple_errors(self):
        """Test multiple errors are displayed with proper spacing."""
        errors = [
            _create_basic_error("Error1", "First error", "Op1", "service1"),
            _create_basic_error("Error2", "Second error", "Op2", "service2"),
        ]

        console, output = _create_test_console_output(80)
        print_exit_errors(errors, console)

        result = output.getvalue()
        assert result.count("• Service:") == 2
        assert "service1" in result and "service2" in result
        assert "Error1" in result and "Error2" in result

    def test_text_wrapping(self):
        """Test that long error messages are wrapped."""
        long_message = "This is a very long error message " * 10
        error = _create_basic_error(
            "LongError", long_message, "Long Operation", "service"
        )

        console, output = _create_test_console_output(
            50
        )  # Narrow width to force wrapping
        print_exit_errors([error], console)

        result = output.getvalue()
        assert "Reason:" in result
        assert "This is a very long error message" in result
        assert len(result.split("\n")) > 8

    def test_default_console_creation(self):
        """Test that default console is created when none provided."""
        error = _create_basic_error(
            message="Test", operation="Test Op", service_id="test"
        )

        with patch("aiperf.controller.controller_utils.Console") as mock_console_class:
            mock_console = mock_console_class.return_value
            mock_console.size.width = 80

            print_exit_errors([error])

            mock_console_class.assert_called_once()
            mock_console.print.assert_called_once()
            mock_console.file.flush.assert_called()


class TestGroupErrorsByDetails:
    """Test the _group_errors_by_details function."""

    def test_single_error(self):
        """Test grouping with a single error."""
        error = _create_basic_error(service_id="service1")

        result = _group_errors_by_details([error])

        assert len(result) == 1
        assert error.error_details in result
        assert result[error.error_details] == [error]

    def test_multiple_unique_errors(self):
        """Test grouping with multiple unique errors."""
        error1 = _create_basic_error("Error1", "Message1", "Op1", "service1")
        error2 = _create_basic_error("Error2", "Message2", "Op2", "service2")

        result = _group_errors_by_details([error1, error2])

        assert len(result) == 2
        assert error1.error_details in result
        assert error2.error_details in result
        assert result[error1.error_details] == [error1]
        assert result[error2.error_details] == [error2]

    def test_duplicate_error_details(self):
        """Test grouping with duplicate error details."""
        error_details = ErrorDetails(type="DuplicateError", message="Duplicate message")
        error1 = ExitErrorInfo(
            error_details=error_details, operation="Op1", service_id="service1"
        )
        error2 = ExitErrorInfo(
            error_details=error_details, operation="Op2", service_id="service2"
        )

        result = _group_errors_by_details([error1, error2])

        assert len(result) == 1
        assert error_details in result
        assert len(result[error_details]) == 2
        assert error1 in result[error_details]
        assert error2 in result[error_details]

    def test_mixed_errors(self):
        """Test grouping with a mix of unique and duplicate errors."""
        shared_error_details = ErrorDetails(
            type="SharedError", message="Shared message"
        )
        unique_error_details = ErrorDetails(
            type="UniqueError", message="Unique message"
        )

        error1 = ExitErrorInfo(
            error_details=shared_error_details,
            operation="Op1",
            service_id="service1",
        )
        error2 = ExitErrorInfo(
            error_details=shared_error_details,
            operation="Op1",
            service_id="service2",
        )
        error3 = ExitErrorInfo(
            error_details=unique_error_details,
            operation="Op3",
            service_id="service3",
        )

        result = _group_errors_by_details([error1, error2, error3])

        assert len(result) == 2
        assert shared_error_details in result
        assert unique_error_details in result
        assert len(result[shared_error_details]) == 2
        assert len(result[unique_error_details]) == 1


class TestPrintExitErrorsDeduplication:
    """Test error deduplication and service display formatting."""

    def test_single_error_displays_normally(self):
        """Test that single errors display without grouping artifacts."""
        error = ExitErrorInfo(
            error_details=ErrorDetails(type="SingleError", message="Single message"),
            operation="Single Operation",
            service_id="single_service",
        )

        output = StringIO()
        console = Console(file=output, width=120)
        print_exit_errors([error], console)
        result = output.getvalue()

        assert result.count("• Service:") == 1
        assert "• Service: single_service" in result
        assert "Operation: Single Operation" in result
        assert "SingleError" in result
        assert "Single message" in result

    def test_identical_errors_are_deduplicated(self):
        """Test that identical errors are grouped together."""
        error_details = ErrorDetails(type="DuplicateError", message="Duplicate message")
        errors = [
            ExitErrorInfo(
                error_details=error_details,
                operation="Configure Profiling",
                service_id="service1",
            ),
            ExitErrorInfo(
                error_details=error_details,
                operation="Configure Profiling",
                service_id="service2",
            ),
            ExitErrorInfo(
                error_details=error_details,
                operation="Configure Profiling",
                service_id="service3",
            ),
        ]

        output = StringIO()
        console = Console(file=output, width=120)
        print_exit_errors(errors, console)
        result = output.getvalue()

        # Core deduplication: should show only one error block
        assert result.count("• Services:") == 1
        assert result.count("DuplicateError") == 1
        assert result.count("Duplicate message") == 1

        # Service grouping: should show all affected services
        assert "3 services: service1, service2, service3" in result
        assert "Configure Profiling" in result

    def test_mixed_duplicate_and_unique_errors(self):
        """Test correct handling of both duplicate and unique errors."""
        duplicate_error = ErrorDetails(
            type="DuplicateError", message="Duplicate message"
        )
        unique_error = ErrorDetails(type="UniqueError", message="Unique message")

        errors = [
            ExitErrorInfo(
                error_details=duplicate_error, operation="Op1", service_id="service1"
            ),
            ExitErrorInfo(
                error_details=duplicate_error, operation="Op1", service_id="service2"
            ),
            ExitErrorInfo(
                error_details=unique_error, operation="Op2", service_id="service3"
            ),
        ]

        output = StringIO()
        console = Console(file=output, width=120)
        print_exit_errors(errors, console)
        result = output.getvalue()

        # Should show two distinct error blocks
        assert result.count("• Services:") == 1
        assert result.count("• Service:") == 1

        # Duplicate error should be grouped
        assert "2 services: service1, service2" in result
        assert "DuplicateError" in result

        # Unique error should be individual
        assert "service3" in result
        assert "UniqueError" in result

    @pytest.mark.parametrize(
        "operation1,operation2,expected_operation_display",
        [
            ("Same Operation", "Same Operation", "Same Operation"),
            ("Operation1", "Operation2", "Multiple Operations"),
        ],
    )
    def test_operation_display_logic(
        self, operation1, operation2, expected_operation_display
    ):
        """Test operation display when errors have same/different operations."""
        error_details = ErrorDetails(type="TestError", message="Test message")
        errors = [
            ExitErrorInfo(
                error_details=error_details, operation=operation1, service_id="service1"
            ),
            ExitErrorInfo(
                error_details=error_details, operation=operation2, service_id="service2"
            ),
        ]

        console, output = _create_test_console_output()
        print_exit_errors(errors, console)
        result = output.getvalue()

        assert f"Operation: {expected_operation_display}" in result
        assert "2 services: service1, service2" in result

    @pytest.mark.parametrize(
        "num_services,expected_display",
        [
            (1, "service1"),
            (2, "2 services: service1, service2"),
            (3, "3 services: service1, service2, service3"),
            (5, "5 services: service1, service2, etc."),
        ],
    )
    def test_service_display_formatting(self, num_services, expected_display):
        """Test service display formats based on count."""
        error_details = ErrorDetails(type="TestError", message="Test message")
        errors = [
            ExitErrorInfo(
                error_details=error_details,
                operation="Test Op",
                service_id=f"service{i}",
            )
            for i in range(1, num_services + 1)
        ]

        console, output = _create_test_console_output()
        print_exit_errors(errors, console)
        result = output.getvalue()

        service_label = "Services" if num_services > 1 else "Service"
        assert f"• {service_label}: {expected_display}" in result
