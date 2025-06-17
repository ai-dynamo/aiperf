#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import PosixPath

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    AudioConfig,
    ImageConfig,
    InputConfig,
    InputDefaults,
    PromptConfig,
    SessionsConfig,
)


def test_input_config_defaults():
    """
    Test the default values of the InputConfig class.

    This test verifies that an instance of InputConfig is initialized with the
    expected default values as defined in the InputDefaults class. Additionally,
    it checks that the `audio` attribute is an instance of the AudioConfig class.
    """

    config = InputConfig()
    assert config.batch_size == InputDefaults.BATCH_SIZE
    assert config.extra == InputDefaults.EXTRA
    assert config.goodput == InputDefaults.GOODPUT
    assert config.header == InputDefaults.HEADER
    assert config.file == InputDefaults.FILE
    assert config.num_dataset_entries == InputDefaults.NUM_DATASET_ENTRIES
    assert config.random_seed == InputDefaults.RANDOM_SEED
    assert isinstance(config.audio, AudioConfig)
    assert isinstance(config.image, ImageConfig)
    assert isinstance(config.prompt, PromptConfig)
    assert isinstance(config.sessions, SessionsConfig)


def test_input_config_custom_values():
    """
    Test the InputConfig class with custom values.

    This test verifies that the InputConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    config = InputConfig(
        batch_size=64,
        extra={"key": "value"},
        goodput={"request_latency": 200},
        header={"Authorization": "Bearer token"},
        file="synthetic:queries,passages",
        num_dataset_entries=10,
        random_seed=42,
    )

    assert config.batch_size == 64
    assert config.extra == {"key": "value"}
    assert config.goodput == {"request_latency": 200}
    assert config.header == {"Authorization": "Bearer token"}
    assert config.file == PosixPath("synthetic:queries,passages")
    assert config.num_dataset_entries == 10
    assert config.random_seed == 42


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
    valid_file = "synthetic:queries,passages"
    config = InputConfig(file=valid_file)
    assert config.file == PosixPath(valid_file)

    with pytest.raises(ValidationError):
        InputConfig(file=12345)  # Invalid file (non-string value)
