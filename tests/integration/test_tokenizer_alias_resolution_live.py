# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for tokenizer alias resolution.

These tests make REAL network calls to HuggingFace Hub to verify alias resolution.
Run with: uv run pytest tests/integration/test_tokenizer_alias_resolution_live.py -v
"""

import os

import pytest

from aiperf.common.tokenizer import Tokenizer

# =============================================================================
# Test Data
# =============================================================================

# Documented examples from docs/tutorials/tokenizer-alias-resolution.md
DOCUMENTED_ALIASES = [
    ("bert-base-uncased", "google-bert/bert-base-uncased"),
    ("roberta-large", "FacebookAI/roberta-large"),
    ("clip-vit-base-patch32", "openai/clip-vit-base-patch32"),
]

# LLM models - specific names that should resolve
LLM_SPECIFIC_NAMES = [
    # Meta Llama
    ("Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    ("CodeLlama-7b-hf", "codellama/CodeLlama-7b-hf"),
    # Mistral
    ("Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2"),
    # Qwen
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    # Google
    ("gemma-2b", "google/gemma-2b"),
    # Microsoft
    ("phi-2", "microsoft/phi-2"),
    # TII
    ("falcon-7b", "tiiuae/falcon-7b"),
]

# Lowercase variants - case-insensitive matching should work
LLM_LOWERCASE_NAMES = [
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B"),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("llama-3.1-8b", "meta-llama/Llama-3.1-8B"),
    ("llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"),
    ("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1"),
]

# Generic names that should NOT resolve (too ambiguous)
LLM_GENERIC_NAMES = ["llama", "mistral", "qwen", "gemma", "phi", "falcon"]

# Encoder/embedding models that resolve reliably
ENCODER_MODELS = [
    # GPT-2 family
    ("gpt2", "openai-community/gpt2"),
    ("gpt2-medium", "openai-community/gpt2-medium"),
    # BERT family
    ("bert-base-uncased", "google-bert/bert-base-uncased"),
    ("bert-base-cased", "google-bert/bert-base-cased"),
    # RoBERTa family
    ("roberta-base", "FacebookAI/roberta-base"),
    ("roberta-large", "FacebookAI/roberta-large"),
    # Other encoders
    ("distilbert-base-uncased", "distilbert/distilbert-base-uncased"),
    ("albert-base-v2", "albert/albert-base-v2"),
    # T5 family
    ("t5-small", "google-t5/t5-small"),
    ("t5-base", "google-t5/t5-base"),
    # Embedding models
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-large-en", "BAAI/bge-large-en"),
    ("e5-large", "intfloat/e5-large"),
]

# Full repository IDs that should pass through unchanged
FULL_REPOSITORY_IDS = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google-bert/bert-base-uncased",
    "openai-community/gpt2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Edge cases - paths, URLs, invalid inputs
EDGE_CASE_PATHS = ["../etc/passwd", "./local/path", "/absolute/path"]
EDGE_CASE_INVALID = ["", "a", "this-model-does-not-exist-xyz-123", "https://evil.com"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def ensure_online_mode(monkeypatch):
    """Ensure HuggingFace offline mode is disabled for these tests.

    Requires RUN_HF_INTEGRATION_TESTS=1 to opt in to live HF network calls.
    """
    if not os.getenv("RUN_HF_INTEGRATION_TESTS"):
        pytest.skip(
            "skipping HuggingFace live tests; set RUN_HF_INTEGRATION_TESTS=1 to enable"
        )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


# =============================================================================
# Helper Functions
# =============================================================================


def assert_resolves_to(alias: str, expected: str) -> None:
    """Assert that an alias resolves to the expected canonical ID."""
    result = Tokenizer.resolve_alias(alias)
    assert result.resolved_name == expected, (
        f"'{alias}' resolved to '{result.resolved_name}', expected '{expected}'"
    )


def assert_unchanged(name: str) -> None:
    """Assert that a name is returned unchanged (no resolution)."""
    result = Tokenizer.resolve_alias(name)
    assert result.resolved_name == name, (
        f"'{name}' unexpectedly resolved to '{result.resolved_name}'"
    )


def assert_ambiguous(name: str) -> None:
    """Assert that a name is ambiguous (unchanged with suggestions)."""
    result = Tokenizer.resolve_alias(name)
    assert result.resolved_name == name, (
        f"'{name}' unexpectedly resolved to '{result.resolved_name}'"
    )
    assert result.is_ambiguous, f"'{name}' should be ambiguous but has no suggestions"


# =============================================================================
# Tests
# =============================================================================


class TestDocumentedAliases:
    """Verify the examples documented in tokenizer-alias-resolution.md are accurate."""

    @pytest.mark.parametrize("alias,expected", DOCUMENTED_ALIASES)
    def test_documented_alias_resolves_correctly(self, alias: str, expected: str):
        """Each documented alias should resolve to its expected canonical ID."""
        assert_resolves_to(alias, expected)


class TestLLMModelResolution:
    """Test popular LLM model names - specific names work, generic don't."""

    @pytest.mark.parametrize("alias,expected", LLM_SPECIFIC_NAMES)
    def test_specific_llm_names_resolve(self, alias: str, expected: str):
        """Specific LLM model names with size/version should resolve correctly."""
        assert_resolves_to(alias, expected)

    @pytest.mark.parametrize("alias,expected", LLM_LOWERCASE_NAMES)
    def test_lowercase_llm_names_resolve(self, alias: str, expected: str):
        """Lowercase LLM model names should resolve via case-insensitive matching."""
        assert_resolves_to(alias, expected)

    @pytest.mark.parametrize("generic_name", LLM_GENERIC_NAMES)
    def test_generic_llm_names_are_ambiguous(self, generic_name: str):
        """Generic family names are too ambiguous and should return suggestions."""
        assert_ambiguous(generic_name)


class TestEncoderModels:
    """Test encoder/embedding models that work well with alias resolution."""

    @pytest.mark.parametrize("alias,expected", ENCODER_MODELS)
    def test_encoder_model_resolves(self, alias: str, expected: str):
        """Encoder and embedding models should resolve to their canonical IDs."""
        assert_resolves_to(alias, expected)


class TestFullRepositoryIDs:
    """Test that full repository IDs are returned as-is."""

    @pytest.mark.parametrize("full_id", FULL_REPOSITORY_IDS)
    def test_full_repository_id_unchanged(self, full_id: str):
        """Full repository IDs should be returned unchanged (direct lookup)."""
        assert_unchanged(full_id)


class TestEdgeCases:
    """Test edge cases and potential security issues."""

    @pytest.mark.parametrize("path", EDGE_CASE_PATHS)
    def test_local_paths_returned_unchanged(self, path: str):
        """Local filesystem paths should be returned unchanged (no network call)."""
        assert_unchanged(path)

    @pytest.mark.parametrize("invalid_input", EDGE_CASE_INVALID)
    def test_invalid_inputs_returned_unchanged(self, invalid_input: str):
        """Invalid inputs (empty, single char, URLs, non-existent) should return unchanged."""
        assert_unchanged(invalid_input)
