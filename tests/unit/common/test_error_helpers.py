# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.error_helpers import (
    _is_authentication_error,
    _is_model_not_found_error,
    create_tokenizer_error_message,
)


class TestErrorPatternDetection:
    """Test error pattern detection functions."""

    def test_is_model_not_found_error_with_invalid_identifier(self):
        """Test detection of invalid model identifier."""
        error_str = "'xyz123' is not a valid model identifier listed on huggingface"
        assert _is_model_not_found_error(error_str) is True

    def test_is_model_not_found_error_with_not_local_folder(self):
        """Test detection of 'not a local folder' error."""
        error_str = "ost3 is not a local folder and is not a valid model"
        assert _is_model_not_found_error(error_str) is True

    def test_is_model_not_found_error_with_404(self):
        """Test detection of 404 error."""
        error_str = "404 client error: not found for url"
        assert _is_model_not_found_error(error_str) is True

    def test_is_model_not_found_error_negative(self):
        """Test that other errors are not detected as model not found."""
        error_str = "connection timeout occurred"
        assert _is_model_not_found_error(error_str) is False

    def test_is_authentication_error_with_401(self):
        """Test detection of 401 authentication error."""
        error_str = "401 client error: unauthorized for url"
        assert _is_authentication_error(error_str) is True

    def test_is_authentication_error_with_token_message(self):
        """Test detection of token-related auth error."""
        error_str = "make sure to pass a token having permission to this repo"
        assert _is_authentication_error(error_str) is True

    def test_is_authentication_error_with_gated(self):
        """Test detection of gated model error."""
        error_str = "this is a gated model and requires accepting the license"
        assert _is_authentication_error(error_str) is True

    def test_is_authentication_error_negative(self):
        """Test that other errors are not detected as authentication."""
        error_str = "model not found at the specified path"
        assert _is_authentication_error(error_str) is False


class TestErrorMessageCreation:
    """Test error message creation for different scenarios."""

    def test_create_tokenizer_error_message_model_not_found(self):
        """Test helpful message for model not found."""
        original_error = OSError("'custom-model-123' is not a valid model identifier")

        result = create_tokenizer_error_message(
            original_error=original_error,
            tokenizer_name="custom-model-123",
        )

        # Check key guidance elements
        assert "not found" in result
        assert "--tokenizer" in result
        assert "custom-model-123" in result

    def test_create_tokenizer_error_message_authentication(self):
        """Test helpful message for authentication errors."""
        original_error = OSError(
            "401 Client Error: Unauthorized. Make sure to pass a token"
        )

        result = create_tokenizer_error_message(
            original_error=original_error,
            tokenizer_name="meta-llama/Llama-3-70b",
        )

        # Check key guidance elements
        assert "authentication" in result
        assert "HF_TOKEN" in result
        assert "meta-llama/Llama-3-70b" in result

    def test_create_tokenizer_error_message_generic(self):
        """Test generic helpful message for unrecognized errors."""
        original_error = RuntimeError("Some unexpected error occurred")

        result = create_tokenizer_error_message(
            original_error=original_error,
            tokenizer_name="unknown-model",
        )

        # Check key guidance elements
        assert "Failed to initialize tokenizer" in result
        assert "unknown-model" in result
        # Generic message doesn't provide specific guidance since error cause is unknown

    def test_create_tokenizer_error_message_clean_output(self):
        """Test that error message is clean without original error duplication."""
        original_error = OSError("'test' is not a valid model identifier")

        result = create_tokenizer_error_message(
            original_error=original_error,
            tokenizer_name="test",
        )

        # Should be short and concise without duplication
        # (AIPerf's error reporting already shows details in "Cause" field)
        assert "tokenizer" in result.lower()
        assert "test" in result
        # Message should be short (under 150 chars)
        assert len(result) < 150


class TestCaseSensitivity:
    """Test that pattern matching is case-insensitive."""

    def test_pattern_matching_case_insensitive(self):
        """Test that error detection works regardless of case."""
        # Uppercase error message
        upper_error = OSError("'MODEL' IS NOT A VALID MODEL IDENTIFIER")
        result_upper = create_tokenizer_error_message(upper_error, "MODEL")

        # Lowercase error message
        lower_error = OSError("'model' is not a valid model identifier")
        result_lower = create_tokenizer_error_message(lower_error, "model")

        # Both should detect as model not found and create similar messages
        assert "not found" in result_upper
        assert "not found" in result_lower
