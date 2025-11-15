# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    EndpointDefaults,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    TokenizerConfig,
    TurnConfig,
    TurnDelayConfig,
    UserConfig,
)
from aiperf.common.enums import EndpointType, GPUTelemetryMode
from aiperf.common.enums.dataset_enums import DatasetSamplingStrategy
from aiperf.common.enums.timing_enums import TimingMode

"""
Test suite for the UserConfig class.
"""


class TestUserConfig:
    """Test suite for the UserConfig class."""

    def test_user_config_serialization_to_json_string(self):
        """Test the serialization and deserialization of a UserConfig object to and from a JSON string."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["model1", "model2"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
                streaming=True,
                url="http://custom-url",
                api_key="test_api_key",
                timeout_seconds=10,
            ),
            input=InputConfig(
                random_seed=42,
                dataset_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
                extra=[
                    ("key1", "value1"),
                    ("key2", "value2"),
                    ("key3", "value3"),
                ],
                headers=[
                    ("Authorization", "Bearer token"),
                    ("Content-Type", "application/json"),
                ],
                conversation=ConversationConfig(
                    num=10,
                    turn=TurnConfig(
                        mean=10,
                        stddev=10,
                        delay=TurnDelayConfig(
                            mean=10,
                            stddev=10,
                        ),
                    ),
                ),
            ),
            output=OutputConfig(
                artifact_directory="test_artifacts",
            ),
            tokenizer=TokenizerConfig(
                name="test_tokenizer",
                revision="test_revision",
            ),
            loadgen=LoadGeneratorConfig(
                concurrency=10,
                request_rate=10,
            ),
            cli_command="test_cli_command",
        )

        # NOTE: Currently, we have validation logic that uses the concept of whether a field was set by the user, so
        # exclude_unset must be used. exclude_defaults should also be able to work.
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_unset=True)
            )
            == config
        )
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_defaults=True)
            )
            == config
        )


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
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
            streaming=True,
            url="http://custom-url",
        ),
    )

    # Serialize to JSON and write to a mocked file
    mocked_file = mock_open()
    with patch("pathlib.Path.open", mocked_file):
        mocked_file().write(config.model_dump_json(indent=4, exclude_defaults=True))

    # Read the mocked file and deserialize back to UserConfig
    with patch("pathlib.Path.open", mocked_file):
        mocked_file().read.return_value = config.model_dump_json(
            indent=4, exclude_defaults=True
        )
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
            type=EndpointType.CHAT,
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
            type=EndpointType.CHAT,
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


def test_user_config_exclude_unset_fields():
    """
    Test that the UserConfig class correctly excludes unset fields when serializing to JSON.
    """
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["model1", "model2"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
            streaming=True,
            url="http://custom-url",
        ),
    )
    assert config.model_dump_json(exclude_unset=True) != config.model_dump_json()  # fmt: skip
    assert config.model_dump_json(exclude_defaults=True) != config.model_dump_json()  # fmt: skip
    assert config.model_dump_json(exclude_unset=True, exclude_defaults=True) != config.model_dump_json()  # fmt: skip
    assert config.model_dump_json(exclude_none=True) != config.model_dump_json()  # fmt: skip


@pytest.mark.parametrize(
    "model_names,endpoint_type,timing_mode,streaming,expected_dir",
    [
        (
            ["hf/model"],  # model name with slash
            EndpointType.CHAT,
            TimingMode.REQUEST_RATE,
            True,
            "/tmp/artifacts/hf_model-openai-chat-concurrency5-request_rate10.0",
        ),
        (
            ["model1", "model2"],  # multi-model
            EndpointType.COMPLETIONS,
            TimingMode.REQUEST_RATE,
            True,
            "/tmp/artifacts/model1_multi-openai-completions-concurrency5-request_rate10.0",
        ),
        (
            ["singlemodel"],  # single model
            EndpointType.EMBEDDINGS,
            TimingMode.FIXED_SCHEDULE,
            False,
            "/tmp/artifacts/singlemodel-openai-embeddings-fixed_schedule",
        ),
    ],
)
def test_compute_artifact_directory(
    monkeypatch, model_names, endpoint_type, timing_mode, streaming, expected_dir
):
    endpoint = EndpointConfig(
        model_names=model_names,
        type=endpoint_type,
        custom_endpoint="custom_endpoint",
        streaming=streaming,
        url="http://custom-url",
    )
    output = OutputConfig(artifact_directory=Path("/tmp/artifacts"))
    loadgen = LoadGeneratorConfig(concurrency=5, request_rate=10)

    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
    input_cfg = InputConfig(
        fixed_schedule=(timing_mode == TimingMode.FIXED_SCHEDULE),
        file="/tmp/dummy_input.txt",
    )
    config = UserConfig(
        endpoint=endpoint,
        output=output,
        loadgen=loadgen,
        input=input_cfg,
    )

    # Patch timing_mode property to return the desired timing_mode
    monkeypatch.setattr(UserConfig, "_timing_mode", property(lambda self: timing_mode))

    artifact_dir = config._compute_artifact_directory()
    assert artifact_dir == Path(expected_dir)


@pytest.mark.parametrize(
    "gpu_telemetry_input,expected_mode,expected_urls",
    [
        # No telemetry configured
        ([], GPUTelemetryMode.SUMMARY, []),
        # Dashboard mode only
        (["dashboard"], GPUTelemetryMode.REALTIME_DASHBOARD, []),
        # URLs only (no dashboard)
        (
            ["http://node1:9401/metrics"],
            GPUTelemetryMode.SUMMARY,
            ["http://node1:9401/metrics"],
        ),
        # Dashboard + URLs
        (
            ["dashboard", "http://node1:9401/metrics"],
            GPUTelemetryMode.REALTIME_DASHBOARD,
            ["http://node1:9401/metrics"],
        ),
        # Multiple URLs
        (
            ["http://node1:9401/metrics", "http://node2:9401/metrics"],
            GPUTelemetryMode.SUMMARY,
            ["http://node1:9401/metrics", "http://node2:9401/metrics"],
        ),
        # Dashboard + multiple URLs
        (
            [
                "dashboard",
                "http://node1:9401/metrics",
                "http://node2:9401/metrics",
            ],
            GPUTelemetryMode.REALTIME_DASHBOARD,
            ["http://node1:9401/metrics", "http://node2:9401/metrics"],
        ),
    ],
)
def test_parse_gpu_telemetry_config(gpu_telemetry_input, expected_mode, expected_urls):
    """Test parsing of gpu_telemetry list into mode and URLs."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=gpu_telemetry_input,
    )

    assert config.gpu_telemetry_mode == expected_mode
    assert config.gpu_telemetry_urls == expected_urls


def test_parse_gpu_telemetry_config_with_defaults():
    """Test that gpu_telemetry_mode and gpu_telemetry_urls have correct defaults."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        )
    )

    # Should have default values
    assert config.gpu_telemetry_mode == GPUTelemetryMode.SUMMARY
    assert config.gpu_telemetry_urls == []


def test_parse_gpu_telemetry_config_preserves_existing_fields():
    """Test that parsing GPU telemetry config doesn't affect other fields."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
            streaming=True,
        ),
        gpu_telemetry=["dashboard", "http://custom:9401/metrics"],
    )

    # Telemetry fields should be set
    assert config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
    assert config.gpu_telemetry_urls == ["http://custom:9401/metrics"]

    # Other fields should be unchanged
    assert config.endpoint.streaming is True
    assert config.endpoint.model_names == ["test-model"]


def test_gpu_telemetry_urls_extraction():
    """Test that only http URLs are extracted from gpu_telemetry list."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=[
            "dashboard",  # Not a URL
            "http://node1:9401/metrics",  # Valid URL
            "https://node2:9401/metrics",  # Valid URL
            "summary",  # Not a URL
        ],
    )

    # Should extract only http/https URLs
    assert len(config.gpu_telemetry_urls) == 2
    assert "http://node1:9401/metrics" in config.gpu_telemetry_urls
    assert "https://node2:9401/metrics" in config.gpu_telemetry_urls
    assert "dashboard" not in config.gpu_telemetry_urls
    assert "summary" not in config.gpu_telemetry_urls


def test_gpu_telemetry_mode_detection():
    """Test that dashboard mode is detected correctly in various positions."""
    # Dashboard at beginning
    config1 = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=["dashboard", "http://node1:9401/metrics"],
    )
    assert config1.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD

    # Dashboard at end
    config2 = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=["http://node1:9401/metrics", "dashboard"],
    )
    assert config2.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD

    # No dashboard
    config3 = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=["http://node1:9401/metrics"],
    )
    assert config3.gpu_telemetry_mode == GPUTelemetryMode.SUMMARY


def test_gpu_telemetry_url_normalization():
    """Test that URLs without http:// prefix are normalized correctly."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=[
            "localhost:9400",
            "node1:9401/metrics",
            "http://node2:9400",
            "https://node3:9401/metrics",
        ],
    )

    assert len(config.gpu_telemetry_urls) == 4
    assert "http://localhost:9400" in config.gpu_telemetry_urls
    assert "http://node1:9401/metrics" in config.gpu_telemetry_urls
    assert "http://node2:9400" in config.gpu_telemetry_urls
    assert "https://node3:9401/metrics" in config.gpu_telemetry_urls


def test_gpu_telemetry_mixed_formats():
    """Test that mixed URL formats (with and without http://) work correctly."""
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="test",
        ),
        gpu_telemetry=["dashboard", "localhost:9400", "http://node1:9401"],
    )

    assert config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
    assert len(config.gpu_telemetry_urls) == 2
    assert "http://localhost:9400" in config.gpu_telemetry_urls
    assert "http://node1:9401" in config.gpu_telemetry_urls


def test_gpu_telemetry_csv_file_not_found():
    """Test that GPU metrics CSV file validation raises error if file doesn't exist."""
    with pytest.raises(ValueError, match="GPU metrics file not found"):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="test",
            ),
            gpu_telemetry=["dashboard", "/nonexistent/path/metrics.csv"],
        )


def test_request_rate_mode_conflict_validation():
    """Test that CONCURRENCY_BURST mode with request_rate raises validation error."""
    from aiperf.common.enums.timing_enums import RequestRateMode

    with pytest.raises(
        ValueError,
        match="Request rate mode cannot be .* when a request rate is specified",
    ):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="test",
            ),
            loadgen=LoadGeneratorConfig(
                request_rate=10.0,
                request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            ),
        )


def test_benchmark_duration_and_count_conflict():
    """Test that both benchmark_duration and request_count raises validation error."""
    with pytest.raises(
        ValueError,
        match="Count-based and duration-based benchmarking cannot be used together",
    ):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="test",
            ),
            loadgen=LoadGeneratorConfig(
                benchmark_duration=60,
                request_count=100,
            ),
        )


def test_grace_period_without_duration_validation():
    """Test that grace period without duration raises validation error."""
    with pytest.raises(
        ValueError,
        match="--benchmark-grace-period can only be used with duration-based benchmarking",
    ):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="test",
            ),
            loadgen=LoadGeneratorConfig(
                benchmark_grace_period=10,
            ),
        )


def test_multi_turn_request_count_conflict():
    """Test that both request_count and conversation num raises validation error."""
    with pytest.raises(
        ValueError,
        match="Both a request-count and number of conversations are set",
    ):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="test",
            ),
            input=InputConfig(
                conversation=ConversationConfig(num=50),
            ),
            loadgen=LoadGeneratorConfig(
                request_count=100,
            ),
        )
