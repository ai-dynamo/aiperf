# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UserConfig core functionality (non-dataset-specific).

This module tests the fundamental UserConfig behavior:
- Object construction and validation
- Serialization/deserialization
- Default value handling
- Custom value validation
- Artifact directory computation
- Field exclusion logic

Does NOT test dataset-specific logic (that's in separate files).
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

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
from aiperf.common.enums.timing_enums import TimingMode


class TestUserConfigConstruction:
    """Test UserConfig object construction and validation."""

    def test_minimal_valid_config(self):
        """Test creating UserConfig with minimal required fields."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            )
        )

        # Should have sensible defaults
        assert config.endpoint.model_names == ["test-model"]
        assert config.endpoint.type == EndpointType.CHAT
        assert isinstance(config.input, InputConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.tokenizer, TokenizerConfig)
        assert isinstance(config.loadgen, LoadGeneratorConfig)

    def test_config_with_custom_values(self):
        """Test UserConfig with custom values."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["model1", "model2"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
                streaming=True,
                url="http://custom-url",
            ),
            loadgen=LoadGeneratorConfig(
                request_count=50,
                concurrency=5,
            ),
        )

        assert config.endpoint.model_names == ["model1", "model2"]
        assert config.endpoint.streaming is True
        assert config.endpoint.url == "http://custom-url"
        assert config.loadgen.request_count == 50
        assert config.loadgen.concurrency == 5

    def test_config_defaults_applied(self):
        """Test that default values are properly applied."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
            )
        )

        # Check endpoint defaults
        assert config.endpoint.streaming == EndpointDefaults.STREAMING
        assert config.endpoint.url == EndpointDefaults.URL

        # Check that sub-configs are instances of correct types
        assert isinstance(config.endpoint, EndpointConfig)
        assert isinstance(config.input, InputConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.tokenizer, TokenizerConfig)
        assert isinstance(config.loadgen, LoadGeneratorConfig)


class TestUserConfigSerialization:
    """Test UserConfig serialization and deserialization."""

    def test_json_serialization_roundtrip(self):
        """Test that UserConfig can be serialized to JSON and back."""
        original_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["model1", "model2"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
                streaming=True,
                url="http://custom-url",
            ),
        )

        # Serialize to JSON
        json_str = original_config.model_dump_json(indent=4, exclude_defaults=True)

        # Deserialize back to UserConfig
        loaded_config = UserConfig.model_validate_json(json_str)

        # Should be identical
        assert original_config == loaded_config

    def test_file_serialization_roundtrip(self):
        """Test serialization to file and back."""
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

    def test_field_exclusion_options(self):
        """Test various field exclusion options during serialization."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["model1", "model2"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
                streaming=True,
                url="http://custom-url",
            ),
        )

        # Different exclusion options should produce different output
        full_json = config.model_dump_json()
        exclude_unset_json = config.model_dump_json(exclude_unset=True)
        exclude_defaults_json = config.model_dump_json(exclude_defaults=True)
        exclude_none_json = config.model_dump_json(exclude_none=True)

        # They should all be different
        assert exclude_unset_json != full_json
        assert exclude_defaults_json != full_json
        assert exclude_none_json != full_json


class TestArtifactDirectoryComputation:
    """Test artifact directory path computation logic."""

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
    def test_artifact_directory_computation(
        self,
        monkeypatch,
        model_names,
        endpoint_type,
        timing_mode,
        streaming,
        expected_dir,
    ):
        """Test artifact directory computation with various configurations."""
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
        monkeypatch.setattr(
            UserConfig, "_timing_mode", property(lambda self: timing_mode)
        )

        artifact_dir = config._compute_artifact_directory()
        assert artifact_dir == Path(expected_dir)

    def test_artifact_directory_special_characters(self, monkeypatch):
        """Test artifact directory computation with special characters in model names."""
        endpoint = EndpointConfig(
            model_names=["microsoft/DialoGPT-medium"],  # Contains slash and dash
            type=EndpointType.CHAT,
            streaming=False,
            url="http://test-url",
        )
        output = OutputConfig(artifact_directory=Path("/tmp/artifacts"))
        loadgen = LoadGeneratorConfig(concurrency=1, request_rate=5)

        monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
        input_cfg = InputConfig(file="/tmp/dummy_input.txt")
        config = UserConfig(
            endpoint=endpoint,
            output=output,
            loadgen=loadgen,
            input=input_cfg,
        )

        monkeypatch.setattr(
            UserConfig, "_timing_mode", property(lambda self: TimingMode.REQUEST_RATE)
        )

        artifact_dir = config._compute_artifact_directory()
        # Should properly escape special characters
        assert "microsoft_DialoGPT-medium" in str(artifact_dir)


class TestUserConfigValidation:
    """Test UserConfig validation logic."""

    def test_cli_args_validation(self):
        """Test that CLI args are properly set during validation."""
        # Mock sys.argv
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["aiperf", "--model", "test-model", "--endpoint", "chat"]

            config = UserConfig(
                endpoint=EndpointConfig(
                    model_names=["test-model"],
                    type=EndpointType.CHAT,
                )
            )

            # Should have CLI command set
            assert config.cli_command is not None
            assert "aiperf" in config.cli_command

        finally:
            sys.argv = original_argv

    def test_timing_mode_validation_sets_defaults(self):
        """Test that timing mode validation sets appropriate defaults."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            )
        )

        # Should have timing mode set
        assert hasattr(config, "_timing_mode")
        assert config._timing_mode == TimingMode.REQUEST_RATE  # Default

    def test_loadgen_validation_sets_concurrency(self):
        """Test that loadgen validation sets concurrency when needed."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            ),
            loadgen=LoadGeneratorConfig(
                # No explicit concurrency - should be set by validation
            ),
        )

        # Should have concurrency set
        assert config.loadgen.concurrency is not None
