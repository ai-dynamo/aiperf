# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for early tokenizer validation and preloading."""

import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)
from aiperf.common.tokenizer_validator import (
    _cache_tokenizer,
    _init_worker,
    validate_tokenizer_early,
)


@pytest.fixture
def mock_user_config() -> MagicMock:
    """Create a mock BenchmarkConfig with tokenizer-requiring endpoint.

    Mirrors the v2 BenchmarkConfig surface ``validate_tokenizer_early`` reads:
    ``endpoint.{type,use_server_token_count}``, ``tokenizer.{name,...}``, and
    the ``get_model_names()`` / ``get_default_dataset()`` accessors.
    """
    config = MagicMock()
    config.endpoint.type = "openai_chat"
    config.endpoint.use_server_token_count = False
    config.tokenizer.name = None
    config.tokenizer.trust_remote_code = False
    config.tokenizer.revision = "main"
    config.get_model_names.return_value = ["gpt-4o", "gpt-4o-mini"]
    # Default dataset is non-synthetic by default.
    config.get_default_dataset.return_value = MagicMock(type=None)
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


@pytest.fixture(autouse=True)
def _clean_hf_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove HF offline env vars before each test so assertions are reliable."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


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
        mock_user_config.get_model_names.return_value = ["mock-llama", "test-model"]

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
        mock_user_config.get_model_names.return_value = [
            "mock-llama",
            "Qwen/Qwen3-0.6B",
        ]

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
        mock_user_config.get_model_names.return_value = ["mock-llama"]

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


class TestInitWorker:
    """_init_worker must not raise and must configure the root logger."""

    @pytest.mark.parametrize(
        "level",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
    )  # fmt: skip
    def test_init_worker_does_not_raise(self, level: str) -> None:
        # Should import cleanly and run without error.
        _init_worker(level)

    def test_init_worker_configures_root_logger(self) -> None:
        root = logging.getLogger()
        original_level = root.level
        try:
            _init_worker("WARNING")
            assert root.level == logging.WARNING
        finally:
            root.setLevel(original_level)


@pytest.mark.network
class TestCacheTokenizerCauseChainPreservation:
    """cause_chain survives the ProcessPool boundary via the plain attribute."""

    def test_cause_chain_preserved_across_process_boundary(self) -> None:
        # _cache_tokenizer sets e.cause_chain before re-raising so that
        # concurrent.futures._RemoteTraceback doesn't strip the original chain.
        with ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_worker,
            initargs=("WARNING",),
        ) as pool:
            future = pool.submit(
                _cache_tokenizer, "clearly-not-a-real-model", False, "main"
            )
            with pytest.raises(Exception) as exc_info:  # noqa: BLE001
                future.result()

        e = exc_info.value
        cause_chain = getattr(e, "cause_chain", None)
        assert cause_chain is not None, (
            "cause_chain attribute must be set by the worker"
        )
        assert len(cause_chain) > 0, "cause_chain must be non-empty"
        # The chain must contain at least one real HF exception — not just _RemoteTraceback.
        real_hf_types = {
            "OSError",
            "LocalEntryNotFoundError",
            "RepositoryNotFoundError",
        }
        assert any(t in real_hf_types for t in cause_chain), (
            f"Expected a real HF exception type in cause_chain, got: {cause_chain}"
        )
