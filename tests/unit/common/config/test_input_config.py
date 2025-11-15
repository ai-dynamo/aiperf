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
from aiperf.common.enums import (
    CustomDatasetType,
    MetricFlags,
    MetricTimeUnit,
    MetricType,
)
from aiperf.common.exceptions import MetricTypeError
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


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


def test_rankings_passages_defaults_and_custom_values():
    cfg_default = InputConfig()
    assert cfg_default.rankings_passages_mean == 1
    assert cfg_default.rankings_passages_stddev == 0

    cfg_custom = InputConfig(rankings_passages_mean=5, rankings_passages_stddev=2)
    assert cfg_custom.rankings_passages_mean == 5
    assert cfg_custom.rankings_passages_stddev == 2


def test_rankings_passages_validation_errors():
    with pytest.raises(ValidationError):
        InputConfig(rankings_passages_mean=0)

    with pytest.raises(ValidationError):
        InputConfig(rankings_passages_stddev=-1)
