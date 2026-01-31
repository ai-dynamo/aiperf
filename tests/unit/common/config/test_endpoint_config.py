# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pytest

from aiperf.common.config import EndpointConfig, EndpointDefaults
from aiperf.common.enums import ModelSelectionStrategy
from aiperf.plugin.enums import EndpointType, URLSelectionStrategy


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
        "type": EndpointType.CHAT,
        "custom_endpoint": "custom_endpoint",
        "streaming": True,
        "urls": ["http://custom-url"],
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
        type=EndpointType.CHAT,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is disabled by default

    config = EndpointConfig(
        type=EndpointType.CHAT,
        streaming=False,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming was set to False

    config = EndpointConfig(
        type=EndpointType.CHAT,
        streaming=True,
        model_names=["gpt2"],
    )
    assert config.streaming  # Streaming was set to True

    config = EndpointConfig(
        type=EndpointType.EMBEDDINGS,
        streaming=False,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is not supported for embeddings


class TestMultiURLSupport:
    """Tests for multi-URL load balancing support."""

    def test_single_url_default(self):
        """Single URL should be stored in urls list and accessible via url property."""
        config = EndpointConfig(model_names=["gpt2"])
        assert config.urls == [EndpointDefaults.URL]
        assert config.url == EndpointDefaults.URL

    def test_single_url_custom(self):
        """Custom single URL should work with backward-compatible url property."""
        config = EndpointConfig(
            model_names=["gpt2"], urls=["http://custom-server:8000"]
        )
        assert config.urls == ["http://custom-server:8000"]
        assert config.url == "http://custom-server:8000"

    def test_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        config = EndpointConfig(model_names=["gpt2"], urls=urls)
        assert config.urls == urls
        assert config.url == "http://server1:8000"  # First URL for backward compat

    def test_url_selection_strategy_default(self):
        """Default URL selection strategy should be ROUND_ROBIN."""
        config = EndpointConfig(model_names=["gpt2"])
        assert config.url_selection_strategy == URLSelectionStrategy.ROUND_ROBIN

    def test_url_selection_strategy_custom(self):
        """Custom URL selection strategy should be stored correctly."""
        config = EndpointConfig(
            model_names=["gpt2"],
            urls=["http://server1:8000", "http://server2:8000"],
            url_selection_strategy=URLSelectionStrategy.ROUND_ROBIN,
        )
        assert config.url_selection_strategy == URLSelectionStrategy.ROUND_ROBIN

    def test_urls_must_have_at_least_one(self):
        """URLs list must have at least one entry."""
        with pytest.raises(ValueError):
            EndpointConfig(model_names=["gpt2"], urls=[])
