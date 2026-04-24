# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for AIPerfConfig basic functionality.

The old BaseConfig class has been removed. These tests verify the new
AIPerfConfig provides proper Pydantic model behavior including
serialization and nested model handling.
"""

from aiperf.common.models import AIPerfBaseModel
from aiperf.config import AIPerfConfig


class NestedConfig(AIPerfBaseModel):
    field1: str
    field2: int


_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def test_aiperf_config_serialization():
    """
    Tests that AIPerfConfig can be serialized to dict and contains expected fields.
    """
    config = AIPerfConfig(**_BASE)

    config_dict = config.model_dump()

    assert "models" in config_dict
    assert "endpoint" in config_dict
    assert "datasets" in config_dict
    assert "phases" in config_dict


def test_aiperf_config_json_serialization():
    """
    Tests that AIPerfConfig can be serialized to JSON string and contains expected values.
    """
    config = AIPerfConfig(**_BASE)

    json_output = config.model_dump_json()

    assert "test-model" in json_output
    assert "localhost" in json_output


def test_aiperf_config_nested_access():
    """
    Tests that nested configuration objects are properly accessible.
    """
    config = AIPerfConfig(**_BASE)

    # Verify nested dataset config is accessible
    dataset = config.get_default_dataset()
    assert dataset is not None
    assert dataset.entries == 100


def test_aiperf_config_model_names():
    """
    Tests that model names are properly extracted from configuration.
    """
    config = AIPerfConfig(**_BASE)

    model_names = config.get_model_names()
    assert model_names == ["test-model"]


def test_nested_pydantic_model():
    """
    Tests that nested AIPerfBaseModel works correctly with dict serialization.
    """
    nested = NestedConfig(field1="value1", field2=42)
    nested_dict = nested.model_dump()

    assert nested_dict["field1"] == "value1"
    assert nested_dict["field2"] == 42
    assert isinstance(nested_dict, dict)
