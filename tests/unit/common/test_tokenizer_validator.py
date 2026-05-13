# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for early tokenizer validation and preloading."""

import asyncio
import concurrent.futures
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)
from aiperf.common.tokenizer_validator import (
    preload_tokenizers,
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
def _clean_hf_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Remove HF offline env vars and point tiktoken cache at an empty dir.

    Pointing ``TIKTOKEN_CACHE_DIR`` at a per-test tmp dir makes the
    ``_is_tiktoken_cached`` short-circuit in ``_partition_preload_names``
    return False so tests deterministically exercise the prefetch path
    regardless of whether the host has tiktoken's BPE file cached.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken-empty"))


class _SyncExecutor:
    """Synchronous in-process drop-in for ProcessPoolExecutor used in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_SyncExecutor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 - mirror real executor behavior
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass

    # Expose an empty _processes dict so the timeout-cleanup branch is benign.
    _processes: dict = {}


@pytest.fixture(autouse=True)
def _sync_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ProcessPoolExecutor with an in-process synchronous fake."""
    monkeypatch.setattr(
        "aiperf.common.tokenizer_validator.ProcessPoolExecutor",
        _SyncExecutor,
    )


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
    async def test_prewarms_builtin_tiktoken(self) -> None:
        # Pre-warming ensures macOS child processes (spawned fresh) find the
        # tiktoken encoding file on disk and make zero network calls.
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": BUILTIN_TOKENIZER_NAME})
        mock_load.assert_called_once_with(
            BUILTIN_TOKENIZER_NAME,
            trust_remote_code=False,
            revision="main",
            resolve_alias=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("encoding_name", sorted(TIKTOKEN_ENCODING_NAMES))
    async def test_prewarms_tiktoken_encoding_names(self, encoding_name: str) -> None:
        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": encoding_name})
        mock_load.assert_called_once_with(
            encoding_name,
            trust_remote_code=False,
            revision="main",
            resolve_alias=False,
        )

    @pytest.mark.asyncio
    async def test_tiktoken_prewarm_failure_logs_warning_continues(
        self, mock_logger: MagicMock
    ) -> None:
        with patch.object(
            Tokenizer, "from_pretrained", side_effect=RuntimeError("CDN blocked")
        ):
            await preload_tokenizers(
                {"model": BUILTIN_TOKENIZER_NAME}, logger=mock_logger
            )
        mock_logger.warning.assert_called_once()
        assert BUILTIN_TOKENIZER_NAME in mock_logger.warning.call_args[0][0]

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

    @pytest.mark.asyncio
    async def test_does_not_mutate_parent_env_after_successful_preload(self) -> None:
        # Children inherit os.environ at spawn time. Mutating it in the parent
        # poisons every subsequently spawned service (DatasetManager included),
        # breaking public HF dataset loading (OfflineModeIsEnabled).
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained"),
        ):
            await preload_tokenizers(resolved)
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    @pytest.mark.asyncio
    async def test_does_not_mutate_parent_env_when_all_already_cached(self) -> None:
        resolved = {"model": "meta-llama/Llama-2-7b-hf"}
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=True),
            patch.object(Tokenizer, "from_pretrained") as mock_load,
        ):
            await preload_tokenizers(resolved)
        mock_load.assert_not_called()
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    @pytest.mark.asyncio
    async def test_does_not_mutate_parent_env_when_tiktoken_prewarm_fails(self) -> None:
        with patch.object(
            Tokenizer, "from_pretrained", side_effect=RuntimeError("CDN blocked")
        ):
            await preload_tokenizers({"model": BUILTIN_TOKENIZER_NAME})
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    @pytest.mark.asyncio
    async def test_does_not_enable_offline_mode_when_skipped(self) -> None:
        await preload_tokenizers(None)
        assert os.environ.get("HF_HUB_OFFLINE") is None
        assert os.environ.get("TRANSFORMERS_OFFLINE") is None

    @pytest.mark.asyncio
    async def test_uses_spawn_context_with_correct_max_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Verify the executor is constructed with mp_context=spawn (so the
        # parent never inherits-into-fork the heavy native libs) and
        # max_workers matches the deduplicated name count.
        captured: dict[str, object] = {}

        def _capturing_executor(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _SyncExecutor(*args, **kwargs)

        monkeypatch.setattr(
            "aiperf.common.tokenizer_validator.ProcessPoolExecutor",
            _capturing_executor,
        )

        resolved = {
            "m1": "meta-llama/Llama-2-7b-hf",
            "m2": "meta-llama/Llama-2-7b-hf",  # duplicate
            "m3": "mistralai/Mistral-7B-v0.1",
        }
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained"),
        ):
            await preload_tokenizers(resolved)

        kwargs = captured.get("kwargs", {})
        mp_context = kwargs.get("mp_context")
        assert mp_context is not None
        assert type(mp_context).__name__ == "SpawnContext"
        assert kwargs.get("max_workers") == 2  # deduplicated

    @pytest.mark.asyncio
    async def test_timeout_logs_warning_and_continues(
        self, mock_logger: MagicMock
    ) -> None:
        # When asyncio.wait_for times out, preload_tokenizers must log a
        # warning, kill subprocesses, and return without raising.
        with (
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=False),
            patch.object(Tokenizer, "from_pretrained"),
            patch(
                "aiperf.common.tokenizer_validator.asyncio.wait_for",
                side_effect=asyncio.TimeoutError,
            ),
        ):
            await preload_tokenizers(
                {"m1": "meta-llama/Llama-2-7b-hf"}, logger=mock_logger
            )

        # Two warnings: one for the timeout itself, one per failed name.
        assert mock_logger.warning.call_count >= 1
        timeout_messages = [
            call.args[0]
            for call in mock_logger.warning.call_args_list
            if "AIPERF_TOKENIZER_PRELOAD_TIMEOUT" in call.args[0]
        ]
        assert timeout_messages, "expected a timeout warning message"


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


class TestOfflineConfigStub:
    """Stub config.json for tokenizer-only HF repos so offline child loads succeed.

    Some HF repos (e.g. hf-internal-testing/llama-tokenizer) ship only
    tokenizer files, no config.json. AutoTokenizer's offline path raises
    OSError('Couldn't connect') in that case even though the tokenizer
    files are fully cached. preload_tokenizers writes a stub config.json
    in the parent so child services find it during their offline load.
    """

    def _setup_snapshot(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        *,
        with_config: bool,
        with_tokenizer_config: bool = True,
        with_tokenizer_json: bool = False,
    ) -> Path:
        import json
        from pathlib import Path

        from huggingface_hub import constants as hf_const

        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(tmp_path))
        model_dir = tmp_path / f"models--{name.replace('/', '--')}"
        snapshot = model_dir / "snapshots" / "abc123"
        refs = model_dir / "refs"
        snapshot.mkdir(parents=True)
        refs.mkdir(parents=True)
        (refs / "main").write_text("abc123")
        if with_tokenizer_config:
            (snapshot / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "LlamaTokenizer"})
            )
        if with_tokenizer_json:
            (snapshot / "tokenizer.json").write_text("{}")
        if with_config:
            (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return Path(snapshot)

    @pytest.mark.asyncio
    async def test_writes_stub_for_already_cached_tokenizer_only_repo(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Already cached → no subprocess fires, but stub still gets written.
        name = "hf-internal-testing/llama-tokenizer"
        snapshot = self._setup_snapshot(tmp_path, monkeypatch, name, with_config=False)
        config_path = snapshot / "config.json"
        assert not config_path.exists()

        with patch.object(Tokenizer, "from_pretrained") as mock_load:
            await preload_tokenizers({"model": name})

        mock_load.assert_not_called()  # cache hit; no prefetch needed
        assert config_path.is_file()
        assert config_path.read_text() == "{}"

    @pytest.mark.asyncio
    async def test_writes_stub_after_fresh_prefetch(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No cache initially; subprocess "downloads" (mocked) and creates
        # the snapshot dir with tokenizer_config.json but no config.json.
        # Parent then writes the stub.
        import json
        from pathlib import Path

        from huggingface_hub import constants as hf_const

        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(tmp_path))
        name = "hf-internal-testing/llama-tokenizer"
        model_dir = tmp_path / f"models--{name.replace('/', '--')}"
        snapshot = model_dir / "snapshots" / "abc123"

        def fake_from_pretrained(*args, **kwargs):
            (model_dir / "refs").mkdir(parents=True, exist_ok=True)
            (model_dir / "refs" / "main").write_text("abc123")
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "LlamaTokenizer"})
            )
            return None

        with patch.object(
            Tokenizer, "from_pretrained", side_effect=fake_from_pretrained
        ):
            await preload_tokenizers({"model": name})

        config_path = Path(snapshot / "config.json")
        assert config_path.is_file()
        assert config_path.read_text() == "{}"

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_config(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Repo already has a real config.json; we must NOT overwrite it.
        name = "meta-llama/Llama-2-7b-hf"
        snapshot = self._setup_snapshot(tmp_path, monkeypatch, name, with_config=True)
        config_path = snapshot / "config.json"
        original_text = config_path.read_text()

        with patch.object(Tokenizer, "from_pretrained"):
            await preload_tokenizers({"model": name})

        assert config_path.read_text() == original_text

    @pytest.mark.asyncio
    async def test_writes_stub_for_fast_tokenizer_only_repo(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fast-tokenizer-only repos ship just tokenizer.json — no
        # tokenizer_config.json. Stub should still be written.
        name = "fast-tokenizer-only-repo"
        snapshot = self._setup_snapshot(
            tmp_path,
            monkeypatch,
            name,
            with_config=False,
            with_tokenizer_config=False,
            with_tokenizer_json=True,
        )
        config_path = snapshot / "config.json"
        assert not config_path.exists()

        with patch.object(Tokenizer, "from_pretrained"):
            await preload_tokenizers({"model": name})

        assert config_path.is_file()
        assert config_path.read_text() == "{}"

    @pytest.mark.asyncio
    async def test_does_not_write_stub_when_no_tokenizer_files_present(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Snapshot exists but has no tokenizer indicator file at all —
        # not a tokenizer repo, just broken/unrelated cache state. Don't
        # fabricate a config that transformers might pick up.
        name = "weird-incomplete-repo"
        snapshot = self._setup_snapshot(
            tmp_path,
            monkeypatch,
            name,
            with_config=False,
            with_tokenizer_config=False,
            with_tokenizer_json=False,
        )
        config_path = snapshot / "config.json"

        with patch.object(Tokenizer, "from_pretrained"):
            await preload_tokenizers({"model": name})

        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_does_not_write_stub_when_prefetch_failed(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the subprocess prefetch failed, the snapshot may not exist —
        # don't try to write a stub into a missing directory.
        from huggingface_hub import constants as hf_const

        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(tmp_path))
        name = "meta-llama/Llama-2-7b-hf"

        with patch.object(
            Tokenizer, "from_pretrained", side_effect=RuntimeError("download blew up")
        ):
            await preload_tokenizers({"model": name})

        # No cache dir was created; nothing to stub.
        assert not (tmp_path / f"models--{name.replace('/', '--')}").exists()

    @pytest.mark.asyncio
    async def test_oserror_on_stub_write_aborts_initialization(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If we can't write the stub (read-only filesystem, etc.), abort
        # startup rather than letting child services hang with a misleading
        # "Couldn't connect" error.
        import tempfile

        name = "hf-internal-testing/llama-tokenizer"
        snapshot = self._setup_snapshot(tmp_path, monkeypatch, name, with_config=False)
        original = tempfile.NamedTemporaryFile

        def selective_raise(*args, **kwargs):
            # Only raise for tmp files our stub-writer tries to create
            # inside the snapshot dir; leave other NamedTemporaryFile users
            # in the test machinery alone.
            if str(kwargs.get("dir", "")) == str(snapshot):
                raise OSError("read-only filesystem")
            return original(*args, **kwargs)

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", selective_raise)

        logger = MagicMock()
        with (
            patch.object(Tokenizer, "from_pretrained"),
            pytest.raises(OSError, match="read-only filesystem"),
        ):
            await preload_tokenizers({"model": name}, logger=logger)

        logger.error.assert_called_once()
        assert "config.json" in logger.error.call_args[0][0]
        # Atomic: no partial / leftover tmp file in the snapshot dir.
        assert list(snapshot.glob("config.aiperf-*.json")) == []
