# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tokenizer validation console display."""

import pytest

from aiperf.common.tokenizer_display import (
    TokenizerDisplayEntry,
    display_tokenizer_ambiguous_name,
    display_tokenizer_validation_error,
    extract_tokenizer_name_from_error,
    is_tokenizer_error,
    log_tokenizer_validation_results,
)
from tests.unit.common.conftest import make_display_entry


def assert_output_contains(output: str, *expected_strings: str) -> None:
    """Assert that output contains all expected strings."""
    for expected in expected_strings:
        assert expected in output, f"Expected '{expected}' in output"


def assert_output_contains_lowercase(output: str, *expected_strings: str) -> None:
    """Assert that output contains all expected strings (case-insensitive)."""
    output_lower = output.lower()
    for expected in expected_strings:
        assert expected.lower() in output_lower, f"Expected '{expected}' in output"


# Test data
RESOLVED_GPT2 = ("gpt2", "openai-community/gpt2", True)
CANONICAL_LLAMA = ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", False)
RESOLVED_BERT = ("bert", "google-bert/bert-base-uncased", True)


class TestTokenizerDisplayEntry:
    """Tests for TokenizerDisplayEntry dataclass."""

    @pytest.mark.parametrize(
        ("original", "resolved", "was_resolved"),
        [RESOLVED_GPT2, CANONICAL_LLAMA, RESOLVED_BERT],
        ids=["resolved_gpt2", "canonical_llama", "resolved_bert"],
    )
    def test_entry_creation(self, original, resolved, was_resolved):
        """Test creating display entries with various configurations."""
        entry = TokenizerDisplayEntry(
            original_name=original, resolved_name=resolved, was_resolved=was_resolved
        )
        assert entry.original_name == original
        assert entry.resolved_name == resolved
        assert entry.was_resolved is was_resolved

    def test_make_display_entry_auto_detection(self):
        """Test that make_display_entry auto-detects was_resolved."""
        entry1 = make_display_entry("gpt2", "openai-community/gpt2")
        assert entry1.was_resolved is True

        entry2 = make_display_entry("openai-community/gpt2", "openai-community/gpt2")
        assert entry2.was_resolved is False

        entry3 = make_display_entry("gpt2")
        assert entry3.resolved_name == "gpt2"
        assert entry3.was_resolved is False


class TestLogTokenizerValidationResults:
    """Tests for log_tokenizer_validation_results function."""

    def test_log_empty_results(self, mock_logger):
        """Test that empty results produce no output."""
        logger, messages = mock_logger
        log_tokenizer_validation_results([], logger)
        assert messages == []

    @pytest.mark.parametrize(
        ("original", "resolved", "was_resolved", "expected_in_output"),
        [
            (
                "gpt2",
                "openai-community/gpt2",
                True,
                ["✓", "openai-community/gpt2", "gpt2"],
            ),
            (
                "openai-community/gpt2",
                "openai-community/gpt2",
                False,
                ["✓", "openai-community/gpt2"],
            ),
        ],
        ids=["single_resolved", "single_canonical"],
    )
    def test_log_single_tokenizer(
        self, mock_logger, original, resolved, was_resolved, expected_in_output
    ):
        """Test logging a single tokenizer."""
        logger, messages = mock_logger
        entries = [
            TokenizerDisplayEntry(
                original_name=original,
                resolved_name=resolved,
                was_resolved=was_resolved,
            )
        ]
        log_tokenizer_validation_results(entries, logger)

        result = " ".join(messages)
        assert_output_contains(result, *expected_in_output)

    def test_log_multiple_tokenizers_summary(self, mock_logger):
        """Test logging multiple tokenizers shows correct summary."""
        logger, messages = mock_logger
        entries = [
            make_display_entry("gpt2", "openai-community/gpt2"),
            make_display_entry("meta-llama/Llama-3.1-8B"),
            make_display_entry("bert", "google-bert/bert-base-uncased"),
        ]
        log_tokenizer_validation_results(entries, logger)

        result = " ".join(messages)
        assert_output_contains_lowercase(result, "tokenizers validated", "resolved")


class TestDisplayTokenizerValidationError:
    """Tests for display_tokenizer_validation_error function."""

    @pytest.mark.parametrize(
        ("cause_chain", "error_message", "expected_title", "expected_in_output"),
        [
            # GatedRepoError - detected via cause_chain
            (
                ["TokenizerError", "OSError", "GatedRepoError"],
                "Access denied",
                "Gated Repository",
                ["gated", "huggingface-cli login"],
            ),
            # RepositoryNotFoundError
            (
                ["TokenizerError", "RepositoryNotFoundError"],
                "Model not found",
                "Repository Not Found",
                ["misspelled", "org-name/model-name"],
            ),
            # ModuleNotFoundError with module extraction
            (
                ["TokenizerError", "ModuleNotFoundError"],
                "No module named 'sentencepiece'",
                "Missing Package: sentencepiece",
                ["sentencepiece", "pip install sentencepiece"],
            ),
            # LocalEntryNotFoundError
            (
                ["TokenizerError", "LocalEntryNotFoundError"],
                "Files not cached",
                "Offline - Files Not Cached",
                ["cached", "HF_HUB_OFFLINE"],
            ),
            # RevisionNotFoundError
            (
                ["TokenizerError", "RevisionNotFoundError"],
                "Invalid revision",
                "Invalid Git Revision",
                ["revision", "--tokenizer-revision"],
            ),
            # PermissionError
            (
                ["TokenizerError", "PermissionError"],
                "Permission denied",
                "Cache Permission Error",
                ["cache", "chmod"],
            ),
            # TimeoutError
            (
                ["TokenizerError", "TimeoutError"],
                "Request timed out",
                "Network Timeout",
                ["timeout", "local"],
            ),
            # Fallback for unknown error
            (
                ["TokenizerError", "UnknownError"],
                "Something unexpected",
                "Tokenizer Configuration Error",
                ["unexpected"],
            ),
            # AmbiguousTokenizerNameError
            (
                ["AmbiguousTokenizerNameError"],
                "Ambiguous name",
                "Ambiguous Tokenizer Name",
                ["multiple", "org-name/model-name"],
            ),
            # No cause_chain - fallback
            (
                None,
                "Generic error",
                "Tokenizer Configuration Error",
                ["unexpected"],
            ),
        ],
        ids=[
            "gated_repo",
            "repo_not_found",
            "missing_module",
            "offline_not_cached",
            "invalid_revision",
            "permission_error",
            "timeout",
            "unknown_fallback",
            "ambiguous_name_error",
            "no_cause_chain_fallback",
        ],
    )
    def test_error_display_content(
        self,
        console_output,
        cause_chain,
        error_message,
        expected_title,
        expected_in_output,
    ):
        """Test that error displays show appropriate causes and fixes based on cause_chain."""
        console, output = console_output
        display_tokenizer_validation_error(
            "test-model",
            cause_chain=cause_chain,
            error_message=error_message,
            console=console,
        )

        result = output.getvalue()

        # Check structure
        assert_output_contains(
            result, expected_title, "test-model", "Possible Causes", "Suggested Fixes"
        )

        # Check expected content (case-insensitive)
        assert_output_contains_lowercase(result, *expected_in_output)

    def test_error_display_shows_explanation(self, console_output):
        """Test that the error panel shows why a tokenizer is needed."""
        console, output = console_output
        display_tokenizer_validation_error(
            "broken-model", cause_chain=["TokenizerError"], console=console
        )

        result = output.getvalue()
        assert "client-side token counting" in result
        assert "synthetic prompt generation" in result

    def test_reverse_cause_chain_priority(self, console_output):
        """Test that root cause (end of chain) takes priority over wrapper errors."""
        console, output = console_output
        # OSError is earlier but GatedRepoError is the root cause
        display_tokenizer_validation_error(
            "test-model",
            cause_chain=[
                "TokenizerError",
                "OSError",
                "GatedRepoError",
                "HTTPStatusError",
            ],
            console=console,
        )

        result = output.getvalue()
        # Should show Gated Repository, not Tokenizer Load Error (OSError)
        assert "Gated Repository" in result
        assert "Tokenizer Load Error" not in result


class TestDisplayTokenizerAmbiguousName:
    """Tests for display_tokenizer_ambiguous_name function."""

    def test_displays_ambiguous_name_panel(self, console_output):
        """Test that ambiguous name displays the warning panel."""
        console, output = console_output
        suggestions = [
            ("meta-llama/Llama-3.1-8B", 1_200_000),
            ("meta-llama/Llama-3.1-70B", 800_000),
        ]
        display_tokenizer_ambiguous_name("llama", suggestions, console)

        result = output.getvalue()
        assert_output_contains(
            result,
            "Ambiguous Tokenizer Name",
            "llama",
            "matched multiple",
            "Did you mean",
        )

    def test_displays_suggestions(self, console_output):
        """Test that suggestions are shown in output."""
        console, output = console_output
        suggestions = [
            ("meta-llama/Llama-3.1-8B", 1_200_000),
            ("meta-llama/Llama-3.1-70B", 800_000),
        ]
        display_tokenizer_ambiguous_name("llama", suggestions, console)

        result = output.getvalue()
        assert_output_contains(
            result, "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"
        )

    @pytest.mark.parametrize(
        ("downloads", "expected_format"),
        [(1_500_000, "1.5M"), (800_000, "800.0K"), (1_000, "1.0K"), (500, "500")],
        ids=["millions", "hundreds_of_thousands", "thousands", "hundreds"],
    )
    def test_formats_download_counts(self, console_output, downloads, expected_format):
        """Test that download counts are formatted correctly."""
        console, output = console_output
        suggestions = [("org/model", downloads)]
        display_tokenizer_ambiguous_name("model", suggestions, console)

        result = output.getvalue()
        assert expected_format in result

    def test_shows_fix_suggestion(self, console_output):
        """Test that fix suggestion shows the top model."""
        console, output = console_output
        suggestions = [("meta-llama/Llama-3.1-8B", 1_200_000)]
        display_tokenizer_ambiguous_name("llama", suggestions, console)

        result = output.getvalue()
        assert_output_contains(result, "--tokenizer meta-llama/Llama-3.1-8B")

    def test_limits_suggestions_to_five(self, console_output):
        """Test that only 5 suggestions are shown even if more provided."""
        console, output = console_output
        suggestions = [(f"org/model-{i}", 1000 * (10 - i)) for i in range(10)]
        display_tokenizer_ambiguous_name("model", suggestions, console)

        result = output.getvalue()
        for i in range(5):
            assert f"org/model-{i}" in result
        assert "org/model-5" not in result


class TestIsTokenizerError:
    """Tests for is_tokenizer_error function (cause_chain based)."""

    @pytest.mark.parametrize(
        ("cause_chain", "expected"),
        [
            # Tokenizer-related exception types
            (["TokenizerError"], True),
            (["TokenizerError", "OSError", "GatedRepoError"], True),
            (
                [
                    "LifecycleOperationError",
                    "TokenizerError",
                    "RepositoryNotFoundError",
                ],
                True,
            ),
            (["TokenizerError", "RevisionNotFoundError"], True),
            (["TokenizerError", "EntryNotFoundError"], True),
            (["TokenizerError", "LocalEntryNotFoundError"], True),
            (["TokenizerError", "HfHubHTTPError"], True),
            (["AmbiguousTokenizerNameError"], True),
            (["TokenizerError", "AmbiguousTokenizerNameError"], True),
            # Non-tokenizer errors
            (["RuntimeError"], False),
            (["ValueError", "KeyError"], False),
            (["ConnectionError", "TimeoutError"], False),
            # Empty/None
            (None, False),
            ([], False),
        ],
        ids=[
            "tokenizer_error_only",
            "gated_repo_in_chain",
            "repo_not_found_in_chain",
            "revision_not_found",
            "entry_not_found",
            "local_entry_not_found",
            "hf_hub_error",
            "ambiguous_name_only",
            "ambiguous_name_in_chain",
            "runtime_error",
            "value_key_error",
            "connection_timeout",
            "none_chain",
            "empty_chain",
        ],
    )
    def test_error_detection(self, cause_chain, expected):
        """Test tokenizer error detection based on cause_chain."""
        assert is_tokenizer_error(cause_chain) is expected


class TestExtractTokenizerNameFromError:
    """Tests for extract_tokenizer_name_from_error function."""

    @pytest.mark.parametrize(
        ("error_message", "expected_name"),
        [
            (
                "Can't load tokenizer for 'meta-llama/Llama-3.1-8B'",
                "meta-llama/Llama-3.1-8B",
            ),
            ('Can\'t load tokenizer for "gpt2"', "gpt2"),
            ("'my-model' is not a local folder and is not a valid model", "my-model"),
            ('"org/model-name" is not a local folder', "org/model-name"),
            ("Failed to load tokenizer: broken-model", "broken-model"),
            ("Failed to load tokenizer 'some-model'", "some-model"),
            ("AutoTokenizer.from_pretrained('openai/gpt-4')", "openai/gpt-4"),
            ('from_pretrained("bert-base-uncased") failed', "bert-base-uncased"),
            ("tokenizer 'my-tokenizer' could not be loaded", "my-tokenizer"),
            ("Generic error without tokenizer name", None),
            ("Something went wrong", None),
        ],
        ids=[
            "cant_load_single_quotes",
            "cant_load_double_quotes",
            "not_local_folder_single",
            "not_local_folder_double",
            "failed_to_load_no_quotes",
            "failed_to_load_with_quotes",
            "from_pretrained_single",
            "from_pretrained_double",
            "tokenizer_name_pattern",
            "no_match_generic",
            "no_match_simple",
        ],
    )
    def test_name_extraction(self, error_message, expected_name):
        """Test tokenizer name extraction from various error messages."""
        assert extract_tokenizer_name_from_error(error_message) == expected_name
