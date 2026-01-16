# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def create_tokenizer_error_message(
    original_error: Exception,
    tokenizer_name: str,
) -> str:
    """Create helpful error message for tokenizer initialization failures.

    This function analyzes the original HuggingFace transformers error and provides
    actionable guidance based on common failure patterns.

    Args:
        original_error: The original exception from HuggingFace transformers library.
        tokenizer_name: The tokenizer name/path that failed to load.

    Returns:
        Enhanced error message with user guidance on how to fix the issue.
    """
    error_str = str(original_error).lower()

    # Pattern 1: Model/tokenizer not found on HuggingFace Hub
    if _is_model_not_found_error(error_str):
        return _create_model_not_found_message(tokenizer_name)

    # Pattern 2: Authentication/authorization required (private/gated model)
    if _is_authentication_error(error_str):
        return _create_authentication_required_message(tokenizer_name)

    # Default: Generic helpful message with common solutions
    return _create_generic_error_message(tokenizer_name)


def _is_model_not_found_error(error_str: str) -> bool:
    """Check if error indicates model/tokenizer not found."""
    patterns = [
        "is not a local folder",
        "is not a valid model identifier",
        "does not appear to have a file named",
        "repository not found",
        "404",
        "404 client error",
    ]
    return any(pattern in error_str for pattern in patterns)


def _is_authentication_error(error_str: str) -> bool:
    """Check if error indicates authentication/authorization required."""
    patterns = [
        "401",
        "401 client error",
        "403",
        "authentication",
        "authenticated",
        "make sure to pass a token",
        "private",
        "gated",
        "access to this resource",
        "repository is private",
        "you are not authenticated",
    ]
    return any(pattern in error_str for pattern in patterns)


def _create_model_not_found_message(tokenizer_name: str) -> str:
    """Create message for model/tokenizer not found errors."""
    return f"Tokenizer '{tokenizer_name}' not found. Re-run with: --tokenizer <model-path-or-huggingface-id>"


def _create_authentication_required_message(tokenizer_name: str) -> str:
    """Create message for authentication/authorization errors."""
    return f"Tokenizer '{tokenizer_name}' requires authentication. Set HF_TOKEN environment variable with your HuggingFace token."


def _create_generic_error_message(tokenizer_name: str) -> str:
    """Create generic helpful message when specific pattern not matched."""
    return f"Failed to initialize tokenizer '{tokenizer_name}'."
