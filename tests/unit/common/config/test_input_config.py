# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    SynthesisConfig,
)
from aiperf.common.enums import MetricFlags, MetricTimeUnit, MetricType
from aiperf.common.exceptions import MetricTypeError
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.plugin.enums import CustomDatasetType


def test_input_config_defaults():
    """
    Test the default values of the InputConfig class.

    This test verifies that an instance of InputConfig is initialized with the
    expected default values as defined in the InputDefaults class. Additionally,
    it checks that the `audio` attribute is an instance of the AudioConfig class.
    """

    config = InputConfig()
    assert config.extra == InputDefaults.EXTRA
    assert config.headers == InputDefaults.HEADERS
    assert config.file == InputDefaults.FILE
    assert config.random_seed == InputDefaults.RANDOM_SEED
    assert config.custom_dataset_type == InputDefaults.CUSTOM_DATASET_TYPE
    assert config.goodput == InputDefaults.GOODPUT
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
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            extra={"key": "value"},
            headers={"Authorization": "Bearer token"},
            random_seed=42,
            custom_dataset_type=CustomDatasetType.MULTI_TURN,
            file=temp_file.name,
        )

        assert config.extra == [("key", "value")]
        assert config.headers == [("Authorization", "Bearer token")]
        assert config.file == PosixPath(temp_file.name)
        assert config.random_seed == 42
        assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN


def test_input_config_file_validation():
    """
    Test InputConfig file field with valid and invalid values.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(file=temp_file.name)
        assert config.file == PosixPath(temp_file.name)

    with pytest.raises(ValidationError):
        InputConfig(file=12345)  # Invalid file (non-string value)


def test_input_config_goodput_success():
    cfg = InputConfig(goodput="request_latency:250 inter_token_latency:10")
    assert cfg.goodput == {"request_latency": 250.0, "inter_token_latency": 10.0}


def test_input_config_goodput_validation_raises_error():
    with pytest.raises(ValidationError):
        InputConfig(goodput=123)  # not a string


@pytest.mark.parametrize(
    "goodput_str, unknown_tag",
    [
        ("foo:1", "foo"),
        ("request_latency:250 bar:10", "bar"),
    ],
)
def test_goodput_unknown_raises(monkeypatch, goodput_str, unknown_tag):
    def get_class(tag):
        if tag == "request_latency":
            return type(
                "MockRequestLatencyMetric",
                (),
                {
                    "tag": RequestLatencyMetric.tag,
                    "unit": MetricTimeUnit.MILLISECONDS,
                    "display_unit": None,
                    "flags": MetricFlags.NONE,
                    "type": MetricType.RECORD,
                },
            )
        raise MetricTypeError(f"Metric class with tag '{tag}' not found")

    monkeypatch.setattr(MetricRegistry, "get_class", get_class)

    with pytest.raises(ValidationError) as exc:
        InputConfig(goodput=goodput_str)

    assert f"Unknown metric tag in --goodput: {unknown_tag}" in str(exc.value)


def test_goodput_derived_metric_raises_error(monkeypatch):
    monkeypatch.setattr(
        MetricRegistry, "get_class", {"mock_derived": BaseDerivedMetric}.__getitem__
    )

    with pytest.raises(ValidationError) as exc:
        InputConfig(goodput="mock_derived:1")

    assert (
        "Metric 'mock_derived' is a Derived metric and cannot be used for --goodput."
        in str(exc.value)
    )


def test_custom_dataset_type_without_file_raises_error():
    """
    Test that setting custom_dataset_type without a file raises ValidationError.

    This validates the validate_custom_dataset_file model validator.
    """
    with pytest.raises(ValidationError) as exc:
        InputConfig(custom_dataset_type=CustomDatasetType.SINGLE_TURN, file=None)

    assert "Custom dataset type requires --input-file to be provided" in str(exc.value)


def test_custom_dataset_type_with_file_succeeds():
    """
    Test that setting custom_dataset_type with a file succeeds.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            custom_dataset_type=CustomDatasetType.MULTI_TURN, file=temp_file.name
        )
        assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN
        assert config.file == PosixPath(temp_file.name)


def test_file_without_custom_dataset_type_succeeds():
    """
    Test that providing a file without custom_dataset_type succeeds (allows auto-inference).
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(file=temp_file.name, custom_dataset_type=None)
        assert config.file == PosixPath(temp_file.name)
        assert config.custom_dataset_type is None


@pytest.mark.parametrize(
    "dataset_type",
    [
        CustomDatasetType.SINGLE_TURN,
        CustomDatasetType.MULTI_TURN,
        CustomDatasetType.RANDOM_POOL,
        CustomDatasetType.MOONCAKE_TRACE,
    ],
)
def test_all_custom_dataset_types_require_file(dataset_type):
    """
    Test that all custom dataset types require a file.
    """
    with pytest.raises(ValidationError) as exc:
        InputConfig(custom_dataset_type=dataset_type, file=None)

    assert "Custom dataset type requires --input-file to be provided" in str(exc.value)


# ============================================================================
# Synthesis Validation Tests
# ============================================================================


def test_synthesis_with_mooncake_trace_succeeds():
    """Test that synthesis options with mooncake_trace dataset type succeeds."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            file=temp_file.name,
            synthesis=SynthesisConfig(speedup_ratio=2.0),
        )
        assert config.synthesis.speedup_ratio == 2.0
        assert config.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE


@pytest.mark.parametrize(
    "dataset_type",
    [
        CustomDatasetType.SINGLE_TURN,
        CustomDatasetType.MULTI_TURN,
        CustomDatasetType.RANDOM_POOL,
    ],
)  # fmt: skip
def test_synthesis_with_non_mooncake_trace_raises_error(dataset_type):
    """Test that synthesis options with non-mooncake_trace dataset type raises error."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        with pytest.raises(ValidationError) as exc:
            InputConfig(
                custom_dataset_type=dataset_type,
                file=temp_file.name,
                synthesis=SynthesisConfig(speedup_ratio=2.0),
            )

        assert "require --custom-dataset-type mooncake_trace" in str(exc.value)


def test_synthesis_with_auto_detect_dataset_type_succeeds():
    """Test that synthesis options with auto-detect (None) dataset type succeeds.

    When custom_dataset_type is None, the type will be auto-detected at runtime.
    Synthesis validation is deferred until the actual type is known.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            custom_dataset_type=None,
            file=temp_file.name,
            synthesis=SynthesisConfig(prefix_len_multiplier=2.0),
        )
        assert config.synthesis.prefix_len_multiplier == 2.0
        assert config.custom_dataset_type is None


@pytest.mark.parametrize(
    "synthesis_config",
    [
        SynthesisConfig(speedup_ratio=2.0),
        SynthesisConfig(prefix_len_multiplier=2.0),
        SynthesisConfig(prefix_root_multiplier=2),
        SynthesisConfig(prompt_len_multiplier=2.0),
        SynthesisConfig(speedup_ratio=0.5, prefix_len_multiplier=1.5),
    ],
)  # fmt: skip
def test_synthesis_various_options_require_mooncake_trace(synthesis_config):
    """Test that various synthesis option combinations require mooncake_trace."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        with pytest.raises(ValidationError) as exc:
            InputConfig(
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
                file=temp_file.name,
                synthesis=synthesis_config,
            )

        assert "require --custom-dataset-type mooncake_trace" in str(exc.value)


def test_synthesis_defaults_with_any_dataset_type_succeeds():
    """Test that default synthesis options (no synthesis) work with any dataset type."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        # Default synthesis config (should_synthesize() returns False)
        config = InputConfig(
            custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            file=temp_file.name,
            synthesis=SynthesisConfig(),  # All defaults
        )
        assert not config.synthesis.should_synthesize()
        assert config.custom_dataset_type == CustomDatasetType.SINGLE_TURN


def test_synthesis_max_isl_alone_does_not_trigger_synthesis():
    """Test that max_isl alone doesn't trigger should_synthesize().

    max_isl is a filter, not a synthesis transformation, so it shouldn't
    trigger should_synthesize() even though it requires mooncake_trace.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            file=temp_file.name,
            synthesis=SynthesisConfig(max_isl=4096),
        )
        # max_isl alone doesn't trigger should_synthesize()
        assert not config.synthesis.should_synthesize()
        assert config.synthesis.max_isl == 4096


def test_synthesis_max_osl_alone_does_not_trigger_synthesis():
    """Test that max_osl alone doesn't trigger should_synthesize().

    max_osl is a cap, not a synthesis transformation, so it shouldn't
    trigger should_synthesize() even though it requires mooncake_trace.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = InputConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            file=temp_file.name,
            synthesis=SynthesisConfig(max_osl=2048),
        )
        # max_osl alone doesn't trigger should_synthesize()
        assert not config.synthesis.should_synthesize()
        assert config.synthesis.max_osl == 2048


@pytest.mark.parametrize(
    "dataset_type",
    [
        CustomDatasetType.SINGLE_TURN,
        CustomDatasetType.MULTI_TURN,
        CustomDatasetType.RANDOM_POOL,
    ],
)  # fmt: skip
def test_synthesis_max_isl_requires_mooncake_trace(dataset_type):
    """Test that max_isl requires mooncake_trace dataset type."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        with pytest.raises(ValidationError) as exc:
            InputConfig(
                custom_dataset_type=dataset_type,
                file=temp_file.name,
                synthesis=SynthesisConfig(max_isl=4096),
            )

        assert "require --custom-dataset-type mooncake_trace" in str(exc.value)


@pytest.mark.parametrize(
    "dataset_type",
    [
        CustomDatasetType.SINGLE_TURN,
        CustomDatasetType.MULTI_TURN,
        CustomDatasetType.RANDOM_POOL,
    ],
)  # fmt: skip
def test_synthesis_max_osl_requires_mooncake_trace(dataset_type):
    """Test that max_osl requires mooncake_trace dataset type."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        with pytest.raises(ValidationError) as exc:
            InputConfig(
                custom_dataset_type=dataset_type,
                file=temp_file.name,
                synthesis=SynthesisConfig(max_osl=2048),
            )

        assert "require --custom-dataset-type mooncake_trace" in str(exc.value)
