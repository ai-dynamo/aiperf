# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for early tokenizer validation."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)
from aiperf.common.tokenizer_validator import (
    _partition_cached_names,
    _prefetch_tokenizers,
    validate_tokenizer_early,
)


@pytest.fixture
def mock_user_config() -> MagicMock:
    """Create a mock BenchmarkConfig with tokenizer-requiring endpoint.

    Uses k8s's BenchmarkConfig surface: get_model_names() and
    get_default_dataset() are methods, not attributes; endpoint.type is a
    string; tokenizer is a sub-config.
    """
    config = MagicMock()
    config.endpoint.type = "openai_chat"
    config.endpoint.use_server_token_count = False
    config.get_model_names.return_value = ["gpt-4o", "gpt-4o-mini"]
    # Default to synthetic dataset so server-token-count skip does not trigger
    default_dataset = MagicMock()
    default_dataset.type = "synthetic"
    config.get_default_dataset.return_value = default_dataset
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


class TestPrefetchTokenizers:
    """Skip-branch coverage for _prefetch_tokenizers and _partition_cached_names.

    These tests pin the behaviors that PR #842 had locked through the now-removed
    preload_tokenizers helper: cache-hit short-circuit, dedup via set semantics,
    early return when nothing to fetch, and per-model failure handling.
    """

    @pytest.fixture
    def _logger(self) -> AIPerfLogger:
        return AIPerfLogger("test_prefetch_tokenizers")

    @pytest.fixture
    def _console(self) -> MagicMock:
        return MagicMock()

    def test_prefetch_skips_already_cached_with_revision(
        self, _logger: AIPerfLogger, _console: MagicMock
    ) -> None:
        """Cached names (revision-aware via _is_hf_cached) bypass the executor."""
        with (
            patch(
                "aiperf.common.tokenizer._is_hf_cached", return_value=True
            ) as mock_cached,
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool,
        ):
            _prefetch_tokenizers(
                {"meta-llama/Llama-2-7b-hf"},
                trust_remote_code=False,
                revision="v2.0",
                logger=_logger,
                console=_console,
            )

        mock_cached.assert_called_once_with("meta-llama/Llama-2-7b-hf", "v2.0")
        mock_pool.assert_not_called()

    def test_prefetch_returns_early_when_all_cached(
        self, _logger: AIPerfLogger, _console: MagicMock
    ) -> None:
        """When every input is cached, no ProcessPoolExecutor is constructed."""
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=True),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool,
        ):
            _prefetch_tokenizers(
                {"meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"},
                trust_remote_code=False,
                revision="main",
                logger=_logger,
                console=_console,
            )

        mock_pool.assert_not_called()

    def test_prefetch_dedups_repeated_names(
        self, _logger: AIPerfLogger, _console: MagicMock
    ) -> None:
        """Set input naturally dedups; partition is called with one unique name."""
        with patch(
            "aiperf.common.tokenizer_validator._partition_cached_names",
            return_value=(set(), set()),
        ) as mock_partition:
            _prefetch_tokenizers(
                {"meta-llama/Llama-2-7b-hf"},  # set semantics: cannot duplicate
                trust_remote_code=False,
                revision="main",
                logger=_logger,
                console=_console,
            )

        mock_partition.assert_called_once()
        names_arg = mock_partition.call_args.args[0]
        assert names_arg == {"meta-llama/Llama-2-7b-hf"}
        assert len(names_arg) == 1

    def test_prefetch_passes_revision_to_partition(
        self, _logger: AIPerfLogger, _console: MagicMock
    ) -> None:
        """Revision flows into _partition_cached_names so the cache check is revision-aware."""
        with patch(
            "aiperf.common.tokenizer_validator._partition_cached_names",
            return_value=(set(), set()),
        ) as mock_partition:
            _prefetch_tokenizers(
                {"meta-llama/Llama-2-7b-hf"},
                trust_remote_code=False,
                revision="release-1.2.3",
                logger=_logger,
                console=_console,
            )

        assert mock_partition.call_args.kwargs["revision"] == "release-1.2.3"

    def test_prefetch_exits_on_per_model_exception(
        self, _logger: AIPerfLogger, _console: MagicMock
    ) -> None:
        """A failed prefetch shows the rich diagnostic panel and calls sys.exit(1)."""
        future = MagicMock()
        future.result.side_effect = RuntimeError("network error")
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.submit.return_value = future

        with (
            patch(
                "aiperf.common.tokenizer_validator._partition_cached_names",
                return_value=(set(), {"meta-llama/Llama-2-7b-hf"}),
            ),
            patch(
                "concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ),
            patch(
                "concurrent.futures.as_completed",
                side_effect=lambda futures: iter(futures),
            ),
            patch(
                "aiperf.common.tokenizer_display.display_tokenizer_validation_error"
            ) as mock_display,
            pytest.raises(SystemExit) as excinfo,
        ):
            _prefetch_tokenizers(
                {"meta-llama/Llama-2-7b-hf"},
                trust_remote_code=False,
                revision="main",
                logger=_logger,
                console=_console,
            )

        assert excinfo.value.code == 1
        mock_display.assert_called_once()

    def test_partition_cached_names_separates_cached_and_uncached(
        self, _logger: AIPerfLogger
    ) -> None:
        """_partition_cached_names returns (already_cached, to_fetch) per name."""

        def _fake_cached(name: str, revision: str) -> bool:
            return name == "cached-model"

        with patch("aiperf.common.tokenizer._is_hf_cached", side_effect=_fake_cached):
            cached, to_fetch = _partition_cached_names(
                {"cached-model", "fresh-model"},
                revision="main",
                logger=_logger,
            )

        assert cached == {"cached-model"}
        assert to_fetch == {"fresh-model"}


@pytest.mark.usefixtures("_mock_endpoint_meta")
class TestValidatorSkipsPrefetchEntirely:
    """Top-level coverage: builtin/tiktoken short-circuit before _prefetch_tokenizers runs."""

    def test_builtin_tokenizer_skips_prefetch(
        self, mock_user_config, mock_logger
    ) -> None:
        mock_user_config.tokenizer.name = BUILTIN_TOKENIZER_NAME

        with (
            patch.object(Tokenizer, "resolve_alias"),
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(mock_user_config, mock_logger)

        mock_prefetch.assert_not_called()

    def test_tiktoken_tokenizer_skips_prefetch(
        self, mock_user_config, mock_logger
    ) -> None:
        # cl100k_base is a tiktoken encoding; validator must not prefetch it.
        mock_user_config.tokenizer.name = "cl100k_base"

        with (
            patch.object(Tokenizer, "resolve_alias"),
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(mock_user_config, mock_logger)

        mock_prefetch.assert_not_called()


@pytest.mark.usefixtures("_mock_endpoint_meta")
class TestValidatorFakeModelFallback:
    """Placeholder model names default to builtin when --tokenizer is unset."""

    def test_all_fake_models_skip_alias_resolution_and_prefetch(
        self, mock_user_config, mock_logger
    ) -> None:
        mock_user_config.tokenizer.name = None
        mock_user_config.get_model_names.return_value = ["mock-llama", "test-model"]

        with (
            patch.object(Tokenizer, "resolve_alias") as mock_resolve,
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        mock_resolve.assert_not_called()
        mock_prefetch.assert_not_called()
        assert result == {
            "mock-llama": BUILTIN_TOKENIZER_NAME,
            "test-model": BUILTIN_TOKENIZER_NAME,
        }
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

        with (
            patch.object(
                Tokenizer, "resolve_alias", return_value=resolution
            ) as mock_resolve,
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        # Only the real model is resolved; the fake one is skipped entirely.
        mock_resolve.assert_called_once_with("Qwen/Qwen3-0.6B")
        mock_prefetch.assert_called_once()
        prefetched = mock_prefetch.call_args.args[0]
        assert prefetched == {"Qwen/Qwen3-0.6B"}
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

        with (
            patch.object(
                Tokenizer, "resolve_alias", return_value=resolution
            ) as mock_resolve,
            patch("aiperf.common.tokenizer_validator._prefetch_tokenizers"),
        ):
            result = validate_tokenizer_early(mock_user_config, mock_logger)

        # The fake-detection branch is skipped because --tokenizer was set.
        mock_resolve.assert_called_once_with("Qwen/Qwen3-0.6B")
        # No placeholder warning emitted.
        mock_logger.warning.assert_not_called()
        assert result == {"mock-llama": "Qwen/Qwen3-0.6B"}
