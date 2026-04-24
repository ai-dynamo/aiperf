# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf.records._tokenizer_preload``.

These exercise the preload module's public surface in isolation — its
env-var parsing, soft-fail behavior, and the ``get_or_load`` fallback
path. Real HF tokenizer loading is mocked; the forkserver-helper CoW
savings themselves are covered by the e2e harness at
``tools/mem_validate_kind``.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from aiperf.records import _tokenizer_preload


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch):
    """Clear module state + env so each test starts from a clean slate."""
    _tokenizer_preload._LOADED.clear()
    monkeypatch.delenv("AIPERF_PRELOAD_TOKENIZERS", raising=False)
    monkeypatch.delenv("AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("AIPERF_PRELOAD_TOKENIZER_REVISION", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    yield
    _tokenizer_preload._LOADED.clear()


class TestEnvParsing:
    def test_env_models_empty(self):
        assert _tokenizer_preload._env_models() == []

    def test_env_models_single(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "Qwen/Qwen3-0.6B")
        assert _tokenizer_preload._env_models() == ["Qwen/Qwen3-0.6B"]

    def test_env_models_multiple_with_whitespace(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "AIPERF_PRELOAD_TOKENIZERS", "  Qwen/Qwen3-0.6B , openai/gpt-oss-120b ,"
        )
        assert _tokenizer_preload._env_models() == [
            "Qwen/Qwen3-0.6B",
            "openai/gpt-oss-120b",
        ]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            pytest.param("", False, id="unset-default"),
            pytest.param("false", False, id="false-literal"),
            pytest.param("FALSE", False, id="false-upper"),
            pytest.param("no", False, id="no"),
            pytest.param("0", False, id="zero"),
            pytest.param("true", True, id="true-literal"),
            pytest.param("TRUE", True, id="true-upper"),
            pytest.param("1", True, id="one"),
            pytest.param("yes", True, id="yes"),
        ],
    )
    def test_env_trust_remote_code(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ):
        if raw:
            monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE", raw)
        assert _tokenizer_preload._env_trust_remote_code() is expected

    def test_env_revision_default(self):
        assert _tokenizer_preload._env_revision() == "main"

    def test_env_revision_custom(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_REVISION", "v1.2.3")
        assert _tokenizer_preload._env_revision() == "v1.2.3"

    def test_env_revision_empty_string_falls_back_to_main(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_REVISION", "")
        assert _tokenizer_preload._env_revision() == "main"


class TestPreload:
    def test_preload_is_noop_when_env_unset(self, monkeypatch: pytest.MonkeyPatch):
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as loader:
            _tokenizer_preload._preload()
        loader.assert_not_called()
        assert _tokenizer_preload.preloaded_models() == []

    def test_preload_sets_tokenizers_parallelism_false_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "Qwen/Qwen3-0.6B")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=object()
        ):
            _tokenizer_preload._preload()
        import os as _os

        assert _os.environ.get("TOKENIZERS_PARALLELISM") == "false"

    def test_preload_does_not_override_user_set_parallelism(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "Qwen/Qwen3-0.6B")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=object()
        ):
            _tokenizer_preload._preload()
        import os as _os

        assert _os.environ.get("TOKENIZERS_PARALLELISM") == "true"

    def test_preload_loads_each_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "AIPERF_PRELOAD_TOKENIZERS", "Qwen/Qwen3-0.6B,openai/gpt-oss-120b"
        )
        fake_a = object()
        fake_b = object()
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained",
            side_effect=[fake_a, fake_b],
        ) as loader:
            _tokenizer_preload._preload()
        assert loader.call_count == 2
        assert _tokenizer_preload.preloaded_models() == [
            "Qwen/Qwen3-0.6B",
            "openai/gpt-oss-120b",
        ]
        assert _tokenizer_preload._LOADED["Qwen/Qwen3-0.6B"] is fake_a
        assert _tokenizer_preload._LOADED["openai/gpt-oss-120b"] is fake_b

    def test_preload_soft_fails_on_exception(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "broken/model,Qwen/Qwen3-0.6B")
        fake = object()
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained",
            side_effect=[RuntimeError("boom"), fake],
        ) as loader:
            _tokenizer_preload._preload()
        # broken model was skipped, second one succeeded
        assert loader.call_count == 2
        assert _tokenizer_preload.preloaded_models() == ["Qwen/Qwen3-0.6B"]
        assert "broken/model" not in _tokenizer_preload._LOADED

    def test_preload_respects_trust_remote_code_env(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "a")
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE", "true")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=object()
        ) as loader:
            _tokenizer_preload._preload()
        assert loader.call_args.kwargs["trust_remote_code"] is True

    def test_preload_respects_revision_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZERS", "a")
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_REVISION", "abc123")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=object()
        ) as loader:
            _tokenizer_preload._preload()
        assert loader.call_args.kwargs["revision"] == "abc123"


class TestGetPreloaded:
    def test_get_preloaded_returns_none_when_unknown(self):
        assert _tokenizer_preload.get_preloaded("missing/model") is None

    def test_get_preloaded_hit_default_config(self, monkeypatch: pytest.MonkeyPatch):
        fake = object()
        _tokenizer_preload._LOADED["Qwen/Qwen3-0.6B"] = fake
        result = _tokenizer_preload.get_preloaded("Qwen/Qwen3-0.6B")
        assert result is fake

    def test_get_preloaded_skips_on_trust_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Preload loaded with default trust=false; caller wants trust=true → miss.
        _tokenizer_preload._LOADED["a"] = object()
        assert _tokenizer_preload.get_preloaded("a", trust_remote_code=True) is None

    def test_get_preloaded_skips_on_revision_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Preload loaded with default revision=main; caller wants abc123 → miss.
        _tokenizer_preload._LOADED["a"] = object()
        assert _tokenizer_preload.get_preloaded("a", revision="abc123") is None

    def test_get_preloaded_hit_matching_trust_and_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE", "true")
        monkeypatch.setenv("AIPERF_PRELOAD_TOKENIZER_REVISION", "v2")
        fake = object()
        _tokenizer_preload._LOADED["a"] = fake
        result = _tokenizer_preload.get_preloaded(
            "a", trust_remote_code=True, revision="v2"
        )
        assert result is fake


class TestGetOrLoad:
    def test_get_or_load_uses_preloaded_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake = object()
        _tokenizer_preload._LOADED["Qwen/Qwen3-0.6B"] = fake
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as loader:
            result = _tokenizer_preload.get_or_load("Qwen/Qwen3-0.6B")
        assert result is fake
        loader.assert_not_called()

    def test_get_or_load_falls_back_on_miss(self, monkeypatch: pytest.MonkeyPatch):
        fresh = object()
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=fresh
        ) as loader:
            result = _tokenizer_preload.get_or_load(
                "missing/model",
                trust_remote_code=True,
                revision="v1",
                resolve_alias=False,
            )
        assert result is fresh
        loader.assert_called_once_with(
            "missing/model",
            trust_remote_code=True,
            revision="v1",
            resolve_alias=False,
        )

    def test_get_or_load_falls_back_on_config_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _tokenizer_preload._LOADED["a"] = object()
        fresh = object()
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained", return_value=fresh
        ) as loader:
            # Preload has trust=false; caller wants trust=true — must skip cache.
            result = _tokenizer_preload.get_or_load("a", trust_remote_code=True)
        assert result is fresh
        loader.assert_called_once()


class TestModuleImport:
    """Verify the module is safe to import without any env config set.

    The forkserver helper imports the module unconditionally, so a broken
    import path would crash every service startup. The module-level
    ``_preload()`` call must be a no-op when the env is unset.
    """

    def test_module_import_is_idempotent_and_noop_when_env_unset(self):
        # Reload while env is empty and confirm no tokenizers get loaded.
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as loader:
            importlib.reload(_tokenizer_preload)
        loader.assert_not_called()
        assert _tokenizer_preload.preloaded_models() == []
