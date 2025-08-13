# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from aiperf.common.config import EndpointConfig, EndpointDefaults
from aiperf.common.enums import EndpointType, ModelSelectionStrategy


def test_endpoint_config_defaults():
    """
    Test the default values of the EndpointConfig class.

    This test verifies that the default attributes of an EndpointConfig instance
    match the predefined constants in the EndpointDefaults class. It ensures that
    the configuration is initialized correctly with expected default values.
    """

    # NOTE: Model names must be filled out
    config = EndpointConfig(model_names=["gpt2"])

    assert config.model_selection_strategy == EndpointDefaults.MODEL_SELECTION_STRATEGY
    assert config.type == EndpointDefaults.TYPE
    assert config.custom_endpoint == EndpointDefaults.CUSTOM_ENDPOINT
    assert config.streaming == EndpointDefaults.STREAMING
    assert config.url == EndpointDefaults.URL


def test_endpoint_config_custom_values():
    """
    Test the `EndpointConfig` class with custom values.
    This test verifies that the `EndpointConfig` object correctly initializes
    its attributes when provided with a dictionary of custom values. It ensures
    that each attribute in the configuration matches the corresponding value
    from the input dictionary.

    Raises:
    - AssertionError: If any attribute value does not match the expected value.
    """

    custom_values = {
        "model_names": ["gpt2"],
        "model_selection_strategy": ModelSelectionStrategy.ROUND_ROBIN,
        "type": EndpointType.OPENAI_CHAT_COMPLETIONS,
        "custom_endpoint": "custom_endpoint",
        "streaming": True,
        "url": "http://custom-url",
        "timeout_seconds": 10,
        "api_key": "custom_api_key",
    }
    config = EndpointConfig(**custom_values)
    for key, value in custom_values.items():
        config_value = getattr(config, key)
        if isinstance(config_value, Enum):
            config_value = config_value.value.lower()

        assert config_value == value


def test_streaming_validation():
    """
    Test the validation of the `streaming` attribute in the `EndpointConfig` class.
    """

    config = EndpointConfig(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is disabled by default

    config = EndpointConfig(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS,
        streaming=False,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming was set to False

    config = EndpointConfig(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS,
        streaming=True,
        model_names=["gpt2"],
    )
    assert config.streaming  # Streaming was set to True

    config = EndpointConfig(
        type=EndpointType.OPENAI_EMBEDDINGS,
        streaming=True,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is not supported for embeddings
