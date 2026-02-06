# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.controller.controller_utils import (
    _group_errors_by_details,
    _handle_tokenizer_errors,
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
            assert mock_console.print.call_count == 2
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


class TestHandleTokenizerErrors:
    """Test the _handle_tokenizer_errors function."""

    def test_returns_non_tokenizer_errors(self):
        """Test that non-tokenizer errors are returned unmodified."""
        non_tokenizer_error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="GenericError", message="Connection refused"
            ),
            operation="Start Service",
            service_id="Worker-1",
        )

        console, _ = _create_test_console_output()
        remaining = _handle_tokenizer_errors([non_tokenizer_error], console)

        assert len(remaining) == 1
        assert remaining[0] == non_tokenizer_error

    def test_filters_tokenizer_errors(self):
        """Test that tokenizer errors are filtered out and displayed."""
        tokenizer_error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="Failed to load tokenizer 'meta-llama/Llama-3.1-8B'",
                cause_chain=["TokenizerError", "RepositoryNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )

        console, output = _create_test_console_output()
        remaining = _handle_tokenizer_errors([tokenizer_error], console)

        assert len(remaining) == 0
        result = output.getvalue()
        # Should display the specialized tokenizer error panel
        assert "Repository Not Found" in result
        assert "meta-llama/Llama-3.1-8B" in result

    def test_mixed_errors_partial_filtering(self):
        """Test that only tokenizer errors are filtered from mixed errors."""
        tokenizer_error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="Failed to load tokenizer 'my-model'",
                cause_chain=["TokenizerError", "RepositoryNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )
        generic_error = ExitErrorInfo(
            error_details=ErrorDetails(type="RuntimeError", message="Worker crashed"),
            operation="Execute Request",
            service_id="Worker-1",
        )

        console, output = _create_test_console_output()
        remaining = _handle_tokenizer_errors([tokenizer_error, generic_error], console)

        assert len(remaining) == 1
        assert remaining[0] == generic_error
        result = output.getvalue()
        assert "Repository Not Found" in result

    @pytest.mark.parametrize(
        ("error_message", "cause_chain", "expected_in_output"),
        [
            (
                "No module named 'sentencepiece'",
                ["TokenizerError", "ModuleNotFoundError"],
                ["sentencepiece", "pip install"],
            ),
            (
                "You are trying to access a gated repo",
                ["TokenizerError", "GatedRepoError"],
                ["Gated", "huggingface-cli login"],
            ),
            (
                "Repository not found",
                ["TokenizerError", "RepositoryNotFoundError"],
                ["Repository Not Found", "huggingface.co/models"],
            ),
        ],
        ids=["missing_module", "gated_repo", "repo_not_found"],
    )
    def test_displays_appropriate_guidance(
        self, error_message, cause_chain, expected_in_output
    ):
        """Test that tokenizer errors display context-specific guidance."""
        error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message=error_message,
                cause_chain=cause_chain,
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )

        console, output = _create_test_console_output()
        _handle_tokenizer_errors([error], console)

        result = output.getvalue()
        for expected in expected_in_output:
            assert expected in result, f"Expected '{expected}' in output"

    def test_extracts_tokenizer_name_when_available(self):
        """Test that tokenizer name is extracted and shown in the error panel."""
        error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="Can't load tokenizer for 'org/my-custom-model'",
                cause_chain=["TokenizerError", "RepositoryNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )

        console, output = _create_test_console_output()
        _handle_tokenizer_errors([error], console)

        result = output.getvalue()
        # The tokenizer name should be extracted and shown
        assert "org/my-custom-model" in result

    def test_uses_placeholder_when_name_not_extractable(self):
        """Test that <unknown> is used when tokenizer name can't be extracted."""
        error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="No module named 'tiktoken'",
                cause_chain=["TokenizerError", "ModuleNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )

        console, output = _create_test_console_output()
        _handle_tokenizer_errors([error], console)

        result = output.getvalue()
        # Should show error panel with tiktoken info
        assert "tiktoken" in result


class TestPrintExitErrorsWithTokenizerIntegration:
    """Test print_exit_errors integration with tokenizer error handling."""

    def test_tokenizer_errors_displayed_first(self):
        """Test that tokenizer errors are handled before generic error display."""
        tokenizer_error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="Can't load tokenizer for 'broken-model'",
                cause_chain=["TokenizerError", "RepositoryNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )
        generic_error = ExitErrorInfo(
            error_details=ErrorDetails(type="RuntimeError", message="Generic error"),
            operation="Some Operation",
            service_id="SomeService-1",
        )

        console, output = _create_test_console_output()
        print_exit_errors([tokenizer_error, generic_error], console)

        result = output.getvalue()
        # Both errors should be displayed
        assert "Repository Not Found" in result
        assert "AIPerf System Exit Errors" in result
        assert "Generic error" in result

    def test_only_tokenizer_errors_skips_generic_panel(self):
        """Test that if only tokenizer errors exist, generic panel is not shown."""
        tokenizer_error = ExitErrorInfo(
            error_details=ErrorDetails(
                type="TokenizerError",
                message="Failed to load tokenizer 'model'",
                cause_chain=["TokenizerError", "RepositoryNotFoundError"],
            ),
            operation="Configure Profiling",
            service_id="DatasetManager-1",
        )

        console, output = _create_test_console_output()
        print_exit_errors([tokenizer_error], console)

        result = output.getvalue()
        # Tokenizer error panel should be shown
        assert "Repository Not Found" in result
        # Generic panel should NOT be shown
        assert "AIPerf System Exit Errors" not in result
