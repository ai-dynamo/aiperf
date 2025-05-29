import pytest

from enum import Enum
from pydantic import ValidationError
from aiperf.common.models.config.endpoint_config import EndPointConfig
from aiperf.common.models.config.config_defaults import EndPointDefaults
from aiperf.common.models.config.config_validators import parse_str_or_list


def test_endpoint_config_defaults():
    """
    Test the default values of the EndPointConfig class.

    This test verifies that the default attributes of an EndPointConfig instance
    match the predefined constants in the EndPointDefaults class. It ensures that
    the configuration is initialized correctly with expected default values.
    """

    config = EndPointConfig()
    assert config.model_selection_strategy == EndPointDefaults.MODEL_SELECTION_STRATEGY
    assert config.backend == EndPointDefaults.BACKEND
    assert config.custom == EndPointDefaults.CUSTOM
    assert config.type == EndPointDefaults.TYPE
    assert config.streaming == EndPointDefaults.STREAMING
    assert config.server_metrics_urls == EndPointDefaults.SERVER_METRICS_URLS
    assert config.url == EndPointDefaults.URL
    assert config.grpc_method == EndPointDefaults.GRPC_METHOD


def test_endpoint_config_custom_values():
    """
    Test the `EndPointConfig` class with custom values.
    This test verifies that the `EndPointConfig` object correctly initializes
    its attributes when provided with a dictionary of custom values. It ensures
    that each attribute in the configuration matches the corresponding value
    from the input dictionary.

    Raises:
    - AssertionError: If any attribute value does not match the expected value.
    """

    custom_values = {
        "model_selection_strategy": "round_robin",
        "backend": "vllm",
        "custom": "custom_endpoint",
        "type": "custom_type",
        "streaming": True,
        "server_metrics_urls": ["http://custom-metrics-url"],
        "url": "http://custom-url",
        "grpc_method": "custom.package.Service/Method",
    }
    config = EndPointConfig(**custom_values)
    for key, value in custom_values.items():
        config_value = getattr(config, key)
        if isinstance(config_value, Enum):
            config_value = config_value.value.lower()

        assert config_value == value


def test_server_metrics_urls_validator():
    """
    Test the validation and assignment of the `server_metrics_urls` attribute
    in the `EndPointConfig` class.
    This test verifies the following scenarios:
    1. When a single URL string is provided, it is correctly converted into a list
    containing that URL.
    2. When a list of URL strings is provided, it is correctly assigned without modification.
    Assertions:
    - Ensure that `server_metrics_urls` is correctly set as a list in both cases.
    """

    config = EndPointConfig(server_metrics_urls="http://metrics-url")
    assert config.server_metrics_urls == ["http://metrics-url"]

    config = EndPointConfig(
        server_metrics_urls=["http://metrics-url1", "http://metrics-url2"]
    )
    assert config.server_metrics_urls == ["http://metrics-url1", "http://metrics-url2"]
