# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus-loading tests for the AIPerf mock server."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiperf_mock_server import config, tokens
from aiperf_mock_server.config import MockServerConfig

from aiperf.common import random_generator as rng
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import PromptConfig
from aiperf.dataset.generator import prompt as prompt_module


@pytest.fixture(autouse=True)
def _reset_corpus_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokens, "CORPUS_TOKENS", None)


def test_load_corpus_uses_configured_tokenizer_and_prompt_generator_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(decode=lambda token_ids: f" token-{token_ids[0]}")
    from_pretrained = Mock(return_value=tokenizer)
    prompt_generator = Mock(return_value=SimpleNamespace(_tokenized_corpus=[7, 11]))
    server_config = MockServerConfig(
        tokenizer="/models/local-tokenizer",
        tokenizer_revision="test-revision",
        tokenizer_trust_remote_code=True,
    )

    monkeypatch.setattr(config, "server_config", server_config)
    monkeypatch.setattr(Tokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(prompt_module, "PromptGenerator", prompt_generator)

    assert tokens._load_corpus() == (" token-7", " token-11")
    from_pretrained.assert_called_once_with(
        "/models/local-tokenizer",
        trust_remote_code=True,
        revision="test-revision",
    )
    prompt_generator.assert_called_once()
    call_kwargs = prompt_generator.call_args.kwargs
    assert isinstance(call_kwargs["prompts"], PromptConfig)
    assert call_kwargs["prefix_prompts"] is None
    assert call_kwargs["tokenizer"] is tokenizer
    assert "config" not in call_kwargs


def test_load_corpus_does_not_hide_prompt_generator_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "server_config", MockServerConfig(tokenizer="builtin"))
    monkeypatch.setattr(
        Tokenizer,
        "from_pretrained",
        Mock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        prompt_module,
        "PromptGenerator",
        Mock(side_effect=TypeError("unsupported PromptGenerator arguments")),
    )

    with pytest.raises(TypeError, match="unsupported PromptGenerator arguments"):
        tokens._load_corpus()


def test_load_corpus_no_tokenizer_uses_character_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_pretrained = Mock(side_effect=AssertionError("tokenizer must not be loaded"))
    monkeypatch.setattr(
        config,
        "server_config",
        MockServerConfig(no_tokenizer=True),
    )
    monkeypatch.setattr(Tokenizer, "from_pretrained", from_pretrained)

    corpus = tokens._load_corpus()

    assert corpus
    from_pretrained.assert_not_called()


def test_load_corpus_initializes_prompt_generator_rng_for_standalone_server(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(config, "server_config", MockServerConfig(tokenizer="builtin"))
    rng.reset()

    with caplog.at_level("WARNING", logger=tokens.__name__):
        corpus = tokens._load_corpus()

    assert corpus
    assert "Tokenizer failed" not in caplog.text
