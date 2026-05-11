# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for early tokenizer validation and preloading."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)
from aiperf.common.tokenizer_validator import (
    preload_tokenizers,
    preload_tokenizers_in_subprocess,
    validate_tokenizer_early,
)


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


@pytest.fixture(autouse=True)
def _clean_hf_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove HF offline env vars before each test so assertions are reliable."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


class TestPreloadTokenizers:
    """Tests for preload_tokenizers() — cache-warming step before child processes spawn."""

    @pytest.mark.asyncio
    async def test_skips_when_resolved_names_none(self) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers(None)
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_resolved_names_empty(self) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_builtin_name(self) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": BUILTIN_TOKENIZER_NAME})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("encoding_name", sorted(TIKTOKEN_ENCODING_NAMES))
    async def test_skips_tiktoken_encoding_names(self, encoding_name: str) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": encoding_name})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_already_cached(self) -> None:
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=True),
            patch.object(Tokenizer, "from_pretrained") as mock_load,
        ):
            await preload_tokenizers({"model": "meta-llama/Llama-2-7b-hf"})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_local_absolute_path(self, tmp_path) -> None:
        local_path = str(tmp_path)
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": local_path})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("local_name", ["./my-tokenizer", "../my-tokenizer"])
    async def test_skips_local_relative_path(self, local_name: str) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": local_name})
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_same_tokenizer_across_models(self) -> None:
        resolved = {
            "model-a": "meta-llama/Llama-2-7b-hf",
            "model-b": "meta-llama/Llama-2-7b-hf",
        }
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained") as mock_load,
        ):
            await preload_tokenizers(resolved)
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_from_pretrained_with_correct_params(self) -> None:
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained") as mock_load,
        ):
            await preload_tokenizers(
                resolved,
                trust_remote_code=True,
                revision="v1.0",
            )
        mock_load.assert_called_once_with(
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            revision="v1.0",
            resolve_alias=False,
        )

    @pytest.mark.asyncio
    async def test_swallows_exception_and_warns(self, mock_logger: MagicMock) -> None:
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(
                Tokenizer, "from_pretrained", side_effect=RuntimeError("network error")
            ),
        ):
            await preload_tokenizers(resolved, logger=mock_logger)  # must not raise

        mock_logger.warning.assert_called_once()
        assert "meta-llama/Llama-2-7b-hf" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_loads_multiple_distinct_tokenizers(self) -> None:
        resolved = {
            "model-a": "meta-llama/Llama-2-7b-hf",
            "model-b": "mistralai/Mistral-7B-v0.1",
        }
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained") as mock_load,
        ):
            await preload_tokenizers(resolved)
        assert mock_load.call_count == 2


class TestPreloadTokenizersInSubprocess:
    """Tests for the sync subprocess wrapper around preload_tokenizers."""

    def test_skips_subprocess_when_resolved_names_none(self) -> None:
        with patch("multiprocessing.get_context") as mock_get_context:
            preload_tokenizers_in_subprocess(None)
        mock_get_context.assert_not_called()

    def test_skips_subprocess_when_resolved_names_empty(self) -> None:
        with patch("multiprocessing.get_context") as mock_get_context:
            preload_tokenizers_in_subprocess({})
        mock_get_context.assert_not_called()

    def test_spawns_with_correct_args(self) -> None:
        import logging

        from aiperf.common import tokenizer_validator

        resolved = {"model-a": "meta-llama/Llama-2-7b-hf"}
        mock_proc = MagicMock()
        mock_queue = MagicMock()
        mock_queue.get_nowait.return_value = ("ok", None)
        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = mock_proc

        with patch(
            "multiprocessing.get_context", return_value=mock_ctx
        ) as mock_get_context:
            preload_tokenizers_in_subprocess(
                resolved, trust_remote_code=True, revision="v1.0"
            )

        mock_get_context.assert_called_once_with("spawn")
        mock_ctx.Process.assert_called_once_with(
            target=tokenizer_validator._preload_subprocess_main,
            kwargs={
                "resolved_names": resolved,
                "trust_remote_code": True,
                "revision": "v1.0",
                "log_level": logging.INFO,
                "result_queue": mock_queue,
            },
            name="aiperf-tokenizer-preload",
        )
        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called_once()

    def test_warns_on_subprocess_error_without_raising(self) -> None:
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        mock_queue = MagicMock()
        mock_queue.get_nowait.return_value = ("error", "RuntimeError('boom')")
        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        logger = MagicMock()

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            preload_tokenizers_in_subprocess(resolved, logger=logger)  # must not raise

        logger.warning.assert_called_once()
        assert "RuntimeError('boom')" in logger.warning.call_args[0][0]

    def test_warns_when_subprocess_exits_without_result(self) -> None:
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        mock_proc = MagicMock(exitcode=137)
        mock_queue = MagicMock()
        mock_queue.get_nowait.side_effect = RuntimeError("queue empty")
        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = mock_proc
        logger = MagicMock()

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            preload_tokenizers_in_subprocess(resolved, logger=logger)

        logger.warning.assert_called_once()
        assert "137" in logger.warning.call_args[0][0]


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
