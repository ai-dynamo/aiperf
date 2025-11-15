# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.composer.custom import CustomDatasetComposer


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filename = None

    def _create_file(content_lines):
        nonlocal filename
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filename = f.name
        return filename

    yield _create_file

    # Cleanup all created files
    if filename:
        Path(filename).unlink(missing_ok=True)


@pytest.fixture
def create_user_config_and_composer(mock_tokenizer_cls):
    """Create a UserConfig and CustomDatasetComposer for testing."""

    def _create():
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                file="test_data.jsonl",
                conversation=ConversationConfig(num=5),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=10, stddev=2),
                ),
            ),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        composer = CustomDatasetComposer(config, tokenizer)
        return config, composer

    return _create


@pytest.fixture
def default_user_config() -> UserConfig:
    """Create a default UserConfig for testing."""
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
