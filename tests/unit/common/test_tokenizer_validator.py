# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for early tokenizer validation."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)
from aiperf.common.tokenizer_validator import validate_tokenizer_early


@pytest.fixture
def mock_user_config() -> MagicMock:
    """Create a mock UserConfig with tokenizer requiring endpoints."""
    config = MagicMock()
    config.endpoint.type = "openai_chat"
    config.endpoint.model_names = ["gpt-4o", "gpt-4o-mini"]
    config.endpoint.use_server_token_count = False
    config.input.public_dataset = None
    config.input.custom_dataset_type = None
    config.input.file = None
    config.tokenizer.name = None
    config.tokenizer.trust_remote_code = False
    config.tokenizer.revision = "main"
    return config


@pytest.fixture
def mock_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture
def _mock_endpoint_meta() -> Iterator[None]:
    """Mock plugins.get_endpoint_metadata to return token-producing endpoint."""
    meta = MagicMock()
    meta.produces_tokens = True
    meta.tokenizes_input = True
    with patch(
        "aiperf.plugin.plugins.get_endpoint_metadata",
        return_value=meta,
    ):
        yield


@pytest.mark.usefixtures("_mock_endpoint_meta")
class TestValidatorTiktokenShortCircuit:
    def test_builtin_skips_alias_resolution(
        self, mock_user_config, mock_logger
    ) -> None:
        mock_user_config.tokenizer.name = BUILTIN_TOKENIZER_NAME

        with patch.object(Tokenizer, "resolve_alias") as mock_resolve:
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        mock_resolve.assert_not_called()
        assert result == {
            "gpt-4o": BUILTIN_TOKENIZER_NAME,
            "gpt-4o-mini": BUILTIN_TOKENIZER_NAME,
        }

    @pytest.mark.parametrize("encoding_name", sorted(TIKTOKEN_ENCODING_NAMES))
    def test_tiktoken_encoding_names_skip_alias_resolution(
        self, mock_user_config, mock_logger, encoding_name: str
    ) -> None:
        mock_user_config.tokenizer.name = encoding_name

        with patch.object(Tokenizer, "resolve_alias") as mock_resolve:
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        mock_resolve.assert_not_called()
        assert result == {
            "gpt-4o": encoding_name,
            "gpt-4o-mini": encoding_name,
        }


@pytest.mark.usefixtures("_mock_endpoint_meta")
class TestValidatorFakeModelFallback:
    """Placeholder model names default to builtin when --tokenizer is unset."""

    def test_all_fake_models_skip_alias_resolution(
        self, mock_user_config, mock_logger
    ) -> None:
        mock_user_config.tokenizer.name = None
        mock_user_config.endpoint.model_names = ["mock-llama", "test-model"]

        with patch.object(Tokenizer, "resolve_alias") as mock_resolve:
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        mock_resolve.assert_not_called()
        assert result == {
            "mock-llama": BUILTIN_TOKENIZER_NAME,
            "test-model": BUILTIN_TOKENIZER_NAME,
        }
        # tokenizer_cfg.name is mutated so downstream consumers see builtin.
        assert mock_user_config.tokenizer.name == BUILTIN_TOKENIZER_NAME
        # One warning per fake model name.
        assert mock_logger.warning.call_count == 2

    def test_mixed_fake_and_real_models_resolve_only_real(
        self, mock_user_config, mock_logger
    ) -> None:
        mock_user_config.tokenizer.name = None
        mock_user_config.endpoint.model_names = ["mock-llama", "Qwen/Qwen3-0.6B"]

        resolution = MagicMock()
        resolution.is_ambiguous = False
        resolution.resolved_name = "Qwen/Qwen3-0.6B"

        with patch.object(
            Tokenizer, "resolve_alias", return_value=resolution
        ) as mock_resolve:
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        # Only the real model is resolved; the fake one is skipped entirely.
        mock_resolve.assert_called_once_with("Qwen/Qwen3-0.6B")
        assert result == {
            "mock-llama": BUILTIN_TOKENIZER_NAME,
            "Qwen/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        }

    def test_explicit_tokenizer_overrides_fake_detection(
        self, mock_user_config, mock_logger
    ) -> None:
        """Explicit --tokenizer wins, even if --model is placeholder-shaped."""
        mock_user_config.tokenizer.name = "Qwen/Qwen3-0.6B"
        mock_user_config.endpoint.model_names = ["mock-llama"]

        resolution = MagicMock()
        resolution.is_ambiguous = False
        resolution.resolved_name = "Qwen/Qwen3-0.6B"

        with patch.object(
            Tokenizer, "resolve_alias", return_value=resolution
        ) as mock_resolve:
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        mock_resolve.assert_called_once_with("Qwen/Qwen3-0.6B")
        # No placeholder warning emitted.
        mock_logger.warning.assert_not_called()
        assert result == {"mock-llama": "Qwen/Qwen3-0.6B"}
