# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the llama.cpp tokenizer API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config import BenchmarkConfig


def _make_config(
    *,
    endpoint_urls: list[str],
    tokenizer: dict[str, object],
    prompt_corpus: str | None = None,
) -> BenchmarkConfig:
    prompts: dict[str, object] = {"isl": 128, "osl": 32}
    if prompt_corpus is not None:
        prompts["corpus"] = prompt_corpus
    return BenchmarkConfig(
        models=["qwen3.5"],
        endpoint={"urls": endpoint_urls, "type": "chat"},
        tokenizer=tokenizer,
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "prompts": prompts,
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 10,
                "concurrency": 1,
            }
        ],
    )


def test_llamacpp_without_tokenizer_url_reuses_inference_server() -> None:
    config = _make_config(
        endpoint_urls=["http://127.0.0.1:8081/v1/chat/completions"],
        tokenizer={"type": "llamacpp"},
    )

    assert config.tokenizer is not None
    assert config.tokenizer.name == "http://127.0.0.1:8081"


def test_llamacpp_explicit_tokenizer_url_overrides_inference_server() -> None:
    config = _make_config(
        endpoint_urls=["http://inference.test/v1/chat/completions"],
        tokenizer={
            "type": "llamacpp",
            "name": "http://tokenizer.test:8082/tokenize",
        },
    )

    assert config.tokenizer is not None
    assert config.tokenizer.name == "http://tokenizer.test:8082"


def test_llamacpp_multiple_inference_servers_require_explicit_tokenizer() -> None:
    with pytest.raises(ValidationError, match="pass --tokenizer <url> explicitly"):
        _make_config(
            endpoint_urls=["http://one.test", "http://two.test"],
            tokenizer={"type": "llamacpp"},
        )


def test_http_tokenizer_url_without_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="--tokenizer-type llamacpp"):
        _make_config(
            endpoint_urls=["http://inference.test"],
            tokenizer={"name": "http://tokenizer.test"},
        )


def test_llamacpp_chat_template_is_rejected() -> None:
    with pytest.raises(ValidationError, match="apply_chat_template is not supported"):
        _make_config(
            endpoint_urls=["http://inference.test"],
            tokenizer={"type": "llamacpp", "apply_chat_template": True},
        )


def test_llamacpp_random_prompt_corpus_is_rejected() -> None:
    with pytest.raises(
        ValidationError, match="prompt corpus 'random' is not supported"
    ):
        _make_config(
            endpoint_urls=["http://inference.test"],
            tokenizer={"type": "llamacpp"},
            prompt_corpus="random",
        )
