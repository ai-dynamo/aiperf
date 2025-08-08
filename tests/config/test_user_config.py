# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import mock_open, patch

from aiperf.common.config import (
    EndpointConfig,
    EndpointDefaults,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    TokenizerConfig,
    UserConfig,
)
from aiperf.common.enums import EndpointType


def test_user_config_serialization_to_file():
    """
    Test the serialization and deserialization of a UserConfig object to and from a file.

    This test verifies that a UserConfig instance can be serialized to JSON format,
    written to a file, and then accurately deserialized back into a UserConfig object.
    It ensures that the original configuration and the loaded configuration are identical.

    Steps:
    1. Create a UserConfig instance with predefined attributes.
    2. Serialize the UserConfig instance to JSON and write it to a mocked file.
    3. Read the JSON data from the mocked file and deserialize it back into a UserConfig instance.
    4. Assert that the original UserConfig instance matches the deserialized instance.

    Mocks:
    - `pathlib.Path.open` is mocked to simulate file operations without actual file I/O.
    """
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["model1", "model2"],
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            custom_endpoint="custom_endpoint",
            streaming=True,
            url="http://custom-url",
        ),
    )

    # Serialize to JSON and write to a mocked file
    mocked_file = mock_open()
    with patch("pathlib.Path.open", mocked_file):
        mocked_file().write(config.model_dump_json(indent=4))

    # Read the mocked file and deserialize back to UserConfig
    with patch("pathlib.Path.open", mocked_file):
        mocked_file().read.return_value = config.model_dump_json(indent=4)
        loaded_config = UserConfig.model_validate_json(mocked_file().read())

    # Ensure the original and loaded configs are identical
    assert config == loaded_config


def test_user_config_defaults():
    """
    Test the default values of the UserConfig class.
    This test verifies that the UserConfig instance is initialized with the expected
    default values as defined in the UserDefaults class. Additionally, it checks that
    the `endpoint` and `input` attributes are instances of their respective configuration
    classes.
    Assertions:
    - `model_names` matches `UserDefaults.MODEL_NAMES`.
    - `verbose` matches `UserDefaults.VERBOSE`.
    - `template_filename` matches `UserDefaults.TEMPLATE_FILENAME`.
    - `endpoint` is an instance of `EndpointConfig`.
    - `input` is an instance of `InputConfig`.
    - `output` is an instance of `OutputConfig`
    - `tokenizer` is an instance of `TokenizerConfig`.
    """

    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["model1", "model2"],
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            custom_endpoint="custom_endpoint",
        )
    )
    assert config.endpoint.model_names == ["model1", "model2"]
    assert config.endpoint.streaming == EndpointDefaults.STREAMING
    assert config.endpoint.url == EndpointDefaults.URL
    assert isinstance(config.endpoint, EndpointConfig)
    assert isinstance(config.input, InputConfig)
    assert isinstance(config.output, OutputConfig)
    assert isinstance(config.tokenizer, TokenizerConfig)


def test_user_config_custom_values():
    """
    Test the UserConfig class with custom values.
    This test verifies that the UserConfig instance correctly initializes
    with the provided custom values and that its attributes match the expected
    values.
    Assertions:
        - Checks that the `model_names` attribute is correctly set to "model1, model2".
        - Verifies that the `verbose` attribute is set to True.
        - Ensures that the `template_filename` attribute is set to "custom_template.yaml".
    """

    custom_values = {
        "endpoint": EndpointConfig(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            custom_endpoint="custom_endpoint",
            model_names=["model1", "model2"],
            streaming=True,
            url="http://custom-url",
        ),
    }
    config = UserConfig(**custom_values)
    assert config.endpoint.model_names == ["model1", "model2"]
    assert config.endpoint.streaming is True
    assert config.endpoint.url == "http://custom-url"
    assert isinstance(config.endpoint, EndpointConfig)
    assert isinstance(config.input, InputConfig)
    assert isinstance(config.output, OutputConfig)
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert isinstance(config.loadgen, LoadGeneratorConfig)
