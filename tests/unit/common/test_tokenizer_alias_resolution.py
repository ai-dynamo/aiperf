# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace Hub alias resolution in Tokenizer class."""

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import Tokenizer


def _create_mock_response(status_code: int = 404) -> MagicMock:
    """Create a mock HTTP response for HuggingFace Hub exceptions."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return mock_response


@pytest.fixture
def mock_model_info():
    """Mock huggingface_hub.model_info."""
    with patch("huggingface_hub.model_info") as mock:
        yield mock


@pytest.fixture
def mock_list_models():
    """Mock huggingface_hub.list_models."""
    with patch("huggingface_hub.list_models") as mock:
        yield mock


class TestTokenizerAliasResolution:
    """Tests for HuggingFace Hub alias resolution."""

    def test_resolve_alias_successful(self, mock_model_info):
        """Test successful alias resolution via model_info keeps original name.

        When model_info() succeeds, the name is a valid HF identifier (redirect).
        We keep the original name so transformers caches under it directly.
        """
        mock_info = MagicMock()
        mock_info.id = "google-bert/bert-base-uncased"
        mock_model_info.return_value = mock_info

        result = Tokenizer.resolve_alias("bert-base-uncased")
        assert result.resolved_name == "bert-base-uncased"
        assert not result.is_ambiguous
        mock_model_info.assert_called_once_with("bert-base-uncased")

    def test_resolve_alias_not_found(self, mock_model_info, mock_list_models):
        """Test alias resolution when repository is not found."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError(
            "Not found", response=_create_mock_response(404)
        )
        mock_list_models.return_value = iter([])  # No search results

        result = Tokenizer.resolve_alias("nonexistent-model")
        assert result.resolved_name == "nonexistent-model"
        assert not result.is_ambiguous

    def test_resolve_alias_http_error(self, mock_model_info, mock_list_models):
        """Test alias resolution when HTTP error occurs."""
        from huggingface_hub.utils import HfHubHTTPError

        mock_model_info.side_effect = HfHubHTTPError(
            "HTTP error", response=_create_mock_response(500)
        )
        mock_list_models.return_value = iter([])  # No search results

        result = Tokenizer.resolve_alias("problematic-model")
        assert result.resolved_name == "problematic-model"

    def test_resolve_alias_generic_exception(self, mock_model_info):
        """Test alias resolution with unexpected exception."""
        mock_model_info.side_effect = Exception("Unexpected error")

        result = Tokenizer.resolve_alias("some-model")
        assert result.resolved_name == "some-model"

    def test_resolve_alias_with_search(self, mock_model_info, mock_list_models):
        """Test alias resolution using search when direct lookup fails."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError(
            "Not found", response=_create_mock_response(404)
        )

        # Search returns a matching model
        mock_model = MagicMock()
        mock_model.id = "meta-llama/Llama-3.1-8B"
        mock_list_models.return_value = iter([mock_model])

        result = Tokenizer.resolve_alias("Llama-3.1-8B")
        assert result.resolved_name == "meta-llama/Llama-3.1-8B"
        mock_list_models.assert_called_once_with(search="Llama-3.1-8B", limit=50)

    def test_resolve_alias_with_search_multiple_results(
        self, mock_model_info, mock_list_models
    ):
        """Test alias resolution chooses the correct model from multiple search results."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError(
            "Not found", response=_create_mock_response(404)
        )

        # Multiple search results - should pick suffix match
        mock_model1 = MagicMock()
        mock_model1.id = "other-org/roberta-large-variant"
        mock_model2 = MagicMock()
        mock_model2.id = "FacebookAI/roberta-large"
        mock_model3 = MagicMock()
        mock_model3.id = "another/model"
        mock_list_models.return_value = iter([mock_model1, mock_model2, mock_model3])

        result = Tokenizer.resolve_alias("roberta-large")
        # Should match the one ending with /roberta-large
        assert result.resolved_name == "FacebookAI/roberta-large"

    def test_resolve_alias_with_search_no_match_returns_ambiguous(
        self, mock_model_info, mock_list_models
    ):
        """Test alias resolution returns suggestions when no exact match found."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError(
            "Not found", response=_create_mock_response(404)
        )

        mock_model = MagicMock()
        mock_model.id = "other-org/different-model"
        mock_model.downloads = 1000
        mock_list_models.return_value = iter([mock_model])

        result = Tokenizer.resolve_alias("llama")
        # No suffix match found, returns ambiguous with suggestions
        assert result.resolved_name == "llama"
        assert result.is_ambiguous
        assert len(result.suggestions) == 1
        assert result.suggestions[0][0] == "other-org/different-model"

    def test_resolve_alias_skips_network_for_local_paths(
        self, mock_model_info, mock_list_models
    ):
        """Test that local paths skip network requests entirely."""
        # Test absolute path
        result = Tokenizer.resolve_alias("/home/user/models/my-tokenizer")
        assert result.resolved_name == "/home/user/models/my-tokenizer"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

        # Reset mocks
        mock_model_info.reset_mock()
        mock_list_models.reset_mock()

        # Test relative path with ./
        result = Tokenizer.resolve_alias("./local-model")
        assert result.resolved_name == "./local-model"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

        # Reset mocks
        mock_model_info.reset_mock()
        mock_list_models.reset_mock()

        # Test relative path with ../
        result = Tokenizer.resolve_alias("../another-model")
        assert result.resolved_name == "../another-model"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

    def test_resolve_alias_skips_in_offline_mode(
        self, mock_model_info, mock_list_models, monkeypatch
    ):
        """Test that offline mode skips network requests."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        result = Tokenizer.resolve_alias("bert-base-uncased")
        assert result.resolved_name == "bert-base-uncased"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

    def test_resolve_alias_skips_in_transformers_offline_mode(
        self, mock_model_info, mock_list_models, monkeypatch
    ):
        """Test that TRANSFORMERS_OFFLINE mode skips network requests."""
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        result = Tokenizer.resolve_alias("roberta-large")
        assert result.resolved_name == "roberta-large"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()
