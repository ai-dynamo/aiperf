# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import PosixPath

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    AudioConfig,
    ConversationConfig,
    ImageConfig,
    InputConfig,
    InputDefaults,
    PromptConfig,
)
from aiperf.common.enums import CustomDatasetType


def test_input_config_defaults():
    """
    Test the default values of the InputConfig class.

    This test verifies that an instance of InputConfig is initialized with the
    expected default values as defined in the InputDefaults class. Additionally,
    it checks that the `audio` attribute is an instance of the AudioConfig class.
    """

    config = InputConfig()
    assert config.extra == InputDefaults.EXTRA
    assert config.goodput == InputDefaults.GOODPUT
    assert config.headers == InputDefaults.HEADERS
    assert config.file == InputDefaults.FILE
    assert config.random_seed == InputDefaults.RANDOM_SEED
    assert config.custom_dataset_type == InputDefaults.CUSTOM_DATASET_TYPE
    assert isinstance(config.audio, AudioConfig)
    assert isinstance(config.image, ImageConfig)
    assert isinstance(config.prompt, PromptConfig)
    assert isinstance(config.conversation, ConversationConfig)


def test_input_config_custom_values():
    """
    Test the InputConfig class with custom values.

    This test verifies that the InputConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    config = InputConfig(
        extra={"key": "value"},
        goodput={"request_latency": 200},
        header={"Authorization": "Bearer token"},
        random_seed=42,
        custom_dataset_type=CustomDatasetType.MULTI_TURN,
    )

    assert config.extra == {"key": "value"}
    assert config.goodput == {"request_latency": 200}
    assert config.header == {"Authorization": "Bearer token"}
    assert config.file is None
    assert config.random_seed == 42
    assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN


def test_input_config_goodput_validation():
    """
    Test InputConfig goodput field with valid and invalid values.
    """
    valid_goodput = {"request_latency": 300, "output_token_throughput_per_user": 600}
    config = InputConfig(goodput=valid_goodput)
    assert config.goodput == valid_goodput

    with pytest.raises(ValidationError):
        InputConfig(goodput={"invalid_metric": "not_a_number"})  # Invalid goodput


def test_input_config_file_validation():
    """
    Test InputConfig file field with valid and invalid values.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(file=temp_file.name)
        assert config.file == PosixPath(temp_file.name)

    with pytest.raises(ValidationError):
        InputConfig(file=12345)  # Invalid file (non-string value)
