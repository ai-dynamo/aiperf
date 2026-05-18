# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.config.endpoint import EndpointDefaults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType, URLSelectionStrategy


def test_endpoint_config_defaults():
    """
    Test the default values of the EndpointConfig class.

    This test verifies that the default attributes of an EndpointConfig instance
    match the predefined constants in the EndpointDefaults class. It ensures that
    the configuration is initialized correctly with expected default values.
    """

    # NOTE: Model names must be filled out
    config = CLIConfig(model_names=["gpt2"])

    assert config.model_selection_strategy == EndpointDefaults.MODEL_SELECTION_STRATEGY
    assert config.endpoint_type == EndpointDefaults.TYPE
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
        "endpoint_type": EndpointType.CHAT,
        "custom_endpoint": "custom_endpoint",
        "streaming": True,
        "urls": ["http://custom-url"],
        "timeout_seconds": 10,
        "api_key": "custom_api_key",
    }
    config = CLIConfig(**custom_values)
    for key, value in custom_values.items():
        config_value = getattr(config, key)
        if isinstance(config_value, Enum):
            config_value = config_value.value.lower()

        assert config_value == value


def test_streaming_validation():
    """
    Test the validation of the `streaming` attribute in the `EndpointConfig` class.
    """

    config = CLIConfig(
        endpoint_type=EndpointType.CHAT,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is disabled by default

    config = CLIConfig(
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming was set to False

    config = CLIConfig(
        endpoint_type=EndpointType.CHAT,
        streaming=True,
        model_names=["gpt2"],
    )
    assert config.streaming  # Streaming was set to True

    config = CLIConfig(
        endpoint_type=EndpointType.EMBEDDINGS,
        streaming=False,
        model_names=["gpt2"],
    )
    assert not config.streaming  # Streaming is not supported for embeddings


class TestMultiURLSupport:
    """Tests for multi-URL load balancing support."""

    def test_single_url_default(self):
        """Single URL should be stored in urls list and accessible via url property."""
        config = CLIConfig(model_names=["gpt2"])
        assert config.urls == [EndpointDefaults.URL]
        assert config.url == EndpointDefaults.URL

    def test_single_url_custom(self):
        """Custom single URL should work with backward-compatible url property."""
        config = CLIConfig(model_names=["gpt2"], urls=["http://custom-server:8000"])
        assert config.urls == ["http://custom-server:8000"]
        assert config.url == "http://custom-server:8000"

    def test_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        config = CLIConfig(model_names=["gpt2"], urls=urls)
        assert config.urls == urls
        assert config.url == "http://server1:8000"  # First URL for backward compat

    def test_url_selection_strategy_default(self):
        """Default URL selection strategy should be ROUND_ROBIN."""
        config = CLIConfig(model_names=["gpt2"])
        assert config.url_selection_strategy == URLSelectionStrategy.ROUND_ROBIN

    def test_url_selection_strategy_custom(self):
        """Custom URL selection strategy should be stored correctly."""
        config = CLIConfig(
            model_names=["gpt2"],
            urls=["http://server1:8000", "http://server2:8000"],
            url_selection_strategy=URLSelectionStrategy.ROUND_ROBIN,
        )
        assert config.url_selection_strategy == URLSelectionStrategy.ROUND_ROBIN

    def test_urls_must_have_at_least_one(self):
        """URLs list must have at least one entry."""
        with pytest.raises(ValueError):
            CLIConfig(model_names=["gpt2"], urls=[])


class TestURLSchemeNormalization:
    """Regression tests for the readiness-probe scheme normalization bug.

    Previously, a URL like ``localhost:8000`` worked for benchmark requests
    (transport prepended ``http://``) but broke the readiness probe path,
    which passed the raw URL to aiohttp and got NonHttpUrlClientError.
    The fix centralizes scheme normalization in EndpointConfig so every
    consumer receives a well-formed URL.
    """

    def test_bare_host_port_gets_http_prepended(self):
        """`localhost:8000` should normalize to `http://localhost:8000`."""
        config = CLIConfig(model_names=["gpt2"], urls=["localhost:8000"])
        assert config.urls == ["http://localhost:8000"]
        assert config.url == "http://localhost:8000"

    def test_existing_http_url_unchanged(self):
        config = CLIConfig(model_names=["gpt2"], urls=["http://localhost:8000"])
        assert config.urls == ["http://localhost:8000"]

    def test_existing_https_url_unchanged(self):
        config = CLIConfig(model_names=["gpt2"], urls=["https://example.com:8443"])
        assert config.urls == ["https://example.com:8443"]

    def test_mixed_list_normalized_per_element(self):
        config = CLIConfig(
            model_names=["gpt2"],
            urls=["server1:8000", "https://server2:8443", "server3"],
        )
        assert config.urls == [
            "http://server1:8000",
            "https://server2:8443",
            "http://server3",
        ]

    def test_default_url_is_normalized(self):
        """The default `EndpointDefaults.URL` is scheme-less; with
        `validate_default=True` the AfterValidator runs on it too, so a
        config built without `--url` (e.g. just `--wait-for-model-timeout 30`)
        still yields a well-formed URL.
        """
        config = CLIConfig(model_names=["gpt2"])  # no urls= argument
        assert all(u.startswith(("http://", "https://")) for u in config.urls), (
            f"Default URLs were not normalized: {config.urls}"
        )

    def test_uppercase_scheme_not_corrupted(self):
        """Pre-existing schemes are preserved regardless of case (no
        ``http://`` is prepended to ``HTTP://host``)."""
        config = CLIConfig(model_names=["gpt2"], urls=["HTTP://host:8000"])
        assert config.urls == ["HTTP://host:8000"]

    def test_non_http_scheme_not_corrupted(self):
        """A non-http(s) scheme is left alone — the validator should not
        produce ``http://ftp://host``. aiohttp will reject it downstream,
        which is the correct error behavior."""
        config = CLIConfig(model_names=["gpt2"], urls=["ftp://host:21"])
        assert config.urls == ["ftp://host:21"]


class TestWaitForModelValidation:
    """Tests for the readiness-probe flag coherence + bounds validation.

    The probe is enabled by setting --wait-for-model-timeout to a positive
    value. Dependent flags (--wait-for-model-interval, --wait-for-model-mode)
    have no effect when disabled and should be rejected if set alone, so
    typos like `--wait-for-model-interval 1` (without a timeout) fail fast.
    """

    def test_default_probe_disabled(self):
        """With no probe flags set, the probe is disabled (timeout == 0)."""
        config = CLIConfig(model_names=["gpt2"])
        assert config.wait_for_model_timeout == 0.0

    def test_setting_timeout_enables_probe(self):
        """Setting --wait-for-model-timeout to a positive value is the
        one-and-only way to enable the probe."""
        config = CLIConfig(model_names=["gpt2"], wait_for_model_timeout=60.0)
        assert config.wait_for_model_timeout == 60.0

    def test_interval_without_timeout_no_longer_raises(self):
        """v1 EndpointConfig is validator-free; the interval-without-timeout
        coherence check moved to AIPerfConfig in v2."""
        config = CLIConfig(model_names=["gpt2"], wait_for_model_interval=1.0)
        assert config.wait_for_model_interval == 1.0

    def test_mode_without_timeout_no_longer_raises(self):
        """See test_interval_without_timeout_no_longer_raises."""
        config = CLIConfig(model_names=["gpt2"], wait_for_model_mode="inference")
        assert config.wait_for_model_mode == "inference"

    def test_interval_with_timeout_accepted(self):
        """With a positive timeout, interval can be customized freely."""
        config = CLIConfig(
            model_names=["gpt2"],
            wait_for_model_timeout=60.0,
            wait_for_model_interval=2.5,
        )
        assert config.wait_for_model_interval == 2.5

    def test_mode_with_timeout_accepted(self):
        """With a positive timeout, mode can be customized freely."""
        config = CLIConfig(
            model_names=["gpt2"],
            wait_for_model_timeout=60.0,
            wait_for_model_mode="both",
        )
        assert config.wait_for_model_mode == "both"

    def test_negative_timeout_now_rejected(self):
        """v1 EndpointConfig now rejects negative wait_for_model_timeout via
        ge=0; the v2 layer's downstream coherence check still runs but
        Pydantic catches the sign violation first."""
        with pytest.raises(ValueError):
            CLIConfig(model_names=["gpt2"], wait_for_model_timeout=-1.0)

    def test_zero_interval_rejected(self):
        """Zero interval would create a tight retry loop; rejected by gt=0.0
        validator (this fires even when paired with a positive timeout)."""
        with pytest.raises(ValueError):
            CLIConfig(
                model_names=["gpt2"],
                wait_for_model_timeout=60.0,
                wait_for_model_interval=0.0,
            )

    def test_invalid_mode_rejected(self):
        """Mode is a Literal; unknown values rejected by pydantic."""
        with pytest.raises(ValueError):
            CLIConfig(
                model_names=["gpt2"],
                wait_for_model_timeout=60.0,
                wait_for_model_mode="something-else",  # type: ignore[arg-type]
            )
