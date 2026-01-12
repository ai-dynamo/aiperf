# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        return _create_model_not_found_message(tokenizer_name, original_error)

    # Pattern 2: Authentication/authorization required (private/gated model)
    if _is_authentication_error(error_str):
        return _create_authentication_required_message(tokenizer_name, original_error)

    # Default: Generic helpful message with common solutions
    return _create_generic_error_message(tokenizer_name, original_error)


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


def _create_model_not_found_message(
    tokenizer_name: str, original_error: Exception
) -> str:
    """Create message for model/tokenizer not found errors."""
    return (
        f"Failed to auto-detect tokenizer for '{tokenizer_name}'.\n\n"
        f"The model name is not available on HuggingFace Hub or is not a valid local path.\n\n"
        f"To fix this, re-run your profiling command with an explicit tokenizer:\n"
        f"  --tokenizer <huggingface-model-path-or-local-path>\n\n"
        f"You can search for models at: https://huggingface.co/models\n\n"
        f"Original error: {original_error}"
    )


def _create_authentication_required_message(
    tokenizer_name: str, original_error: Exception
) -> str:
    """Create message for authentication/authorization errors."""
    return (
        f"Tokenizer '{tokenizer_name}' requires HuggingFace authentication.\n\n"
        f"To fix this:\n"
        f"  1. Get your HuggingFace token: https://huggingface.co/settings/tokens\n"
        f"  2. If the model is gated, accept the license: https://huggingface.co/{tokenizer_name}\n"
        f"  3. Set your token as an environment variable:\n"
        f"       export HF_TOKEN=<your-token>\n\n"
        f"Note: The --tokenizer-hf-token CLI flag will be available in a future release.\n\n"
        f"Original error: {original_error}"
    )


def _create_generic_error_message(
    tokenizer_name: str, original_error: Exception
) -> str:
    """Create generic helpful message when specific pattern not matched."""
    return (
        f"Failed to initialize tokenizer '{tokenizer_name}'.\n\n"
        f"Common solutions:\n"
        f"  • Specify an explicit tokenizer: --tokenizer <huggingface-model-path>\n"
        f"  • Use a local tokenizer: --tokenizer /path/to/local/tokenizer\n"
        f"  • Verify the model exists at: https://huggingface.co/{tokenizer_name}\n"
        f"  • Check your internet connection\n"
        f"  • If the model is private/gated, set: export HF_TOKEN=<your-token>\n\n"
        f"Original error: {original_error}"
    )
