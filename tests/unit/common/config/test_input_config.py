# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 InputConfig DTO.

The v1 InputConfig is a CLI-only input DTO. The model/field validators that
previously enforced cross-field rules (custom_dataset_type requires file,
synthesis options require trace datasets, goodput tag-resolution against
the metric registry, etc.) have moved to AIPerfConfig in v2; the v1 layer
no longer raises on these inputs.

The tests below are limited to behavior that still belongs on the DTO:
field defaults, structural typing, and the BeforeValidator coercers that
parse CLI input shapes (extra/headers tuples, goodput dict-from-string).
"""

import tempfile
from pathlib import PosixPath

import pytest
from pydantic import ValidationError

from aiperf.config.dataset.defaults import InputDefaults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType


def test_input_config_defaults():
    """Default values match InputDefaults / hoisted modality flat fields."""
    config = CLIConfig()
    assert config.extra_inputs == InputDefaults.EXTRA
    assert config.headers == InputDefaults.HEADERS
    assert config.input_file == InputDefaults.FILE
    assert config.random_seed == InputDefaults.RANDOM_SEED
    assert config.custom_dataset_type == InputDefaults.CUSTOM_DATASET_TYPE
    assert config.goodput == InputDefaults.GOODPUT
    # Modality fields are flat post-Task-13; smoke-check a representative
    # field per modality to confirm they exist and carry their defaults.
    assert config.audio_batch_size == 1
    assert config.image_batch_size == 1
    assert config.prompt_batch_size == 1
    assert config.conversation_num is None


def test_input_config_custom_values():
    """Custom values flow through the DTO with the expected coercions."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            extra_inputs={"key": "value"},
            headers={"Authorization": "Bearer token"},
            random_seed=42,
            custom_dataset_type=CustomDatasetType.MULTI_TURN,
            input_file=temp_file.name,
        )

        assert config.extra_inputs == [("key", "value")]
        assert config.headers == [("Authorization", "Bearer token")]
        assert config.input_file == PosixPath(temp_file.name)
        assert config.random_seed == 42
        assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN


def test_input_config_file_validation():
    """File field accepts a path string but rejects non-string scalars."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(input_file=temp_file.name)
        assert config.input_file == PosixPath(temp_file.name)

    with pytest.raises(ValidationError):
        CLIConfig(input_file=12345)


def test_input_config_goodput_success():
    cfg = CLIConfig(goodput="request_latency:250 inter_token_latency:10")
    assert cfg.goodput == {"request_latency": 250.0, "inter_token_latency": 10.0}


def test_input_config_goodput_validation_raises_error():
    with pytest.raises(ValidationError):
        CLIConfig(goodput=123)


def test_custom_dataset_type_with_file_succeeds():
    """custom_dataset_type + file passes through unchanged at the DTO layer."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            custom_dataset_type=CustomDatasetType.MULTI_TURN, input_file=temp_file.name
        )
        assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN
        assert config.input_file == PosixPath(temp_file.name)


def test_file_without_custom_dataset_type_succeeds():
    """File without custom_dataset_type is allowed (auto-inference at runtime)."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(input_file=temp_file.name, custom_dataset_type=None)
        assert config.input_file == PosixPath(temp_file.name)
        assert config.custom_dataset_type is None


def test_synthesis_with_trace_dataset_succeeds():
    """Synthesis fields flow flat onto CLIConfig regardless of dataset type at
    the DTO layer (cross-field validation moved to AIPerfConfig)."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            input_file=temp_file.name,
            synthesis_speedup_ratio=2.0,
        )
        assert config.synthesis_speedup_ratio == 2.0
        assert config.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE


def test_synthesis_output_len_with_trace_dataset_succeeds():
    """Synthesis output length multiplier flows through unchanged at the DTO layer."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            input_file=temp_file.name,
            synthesis_output_len_multiplier=2.0,
        )
        assert config.synthesis_output_len_multiplier == 2.0


def test_synthesis_max_isl_with_trace_dataset_succeeds():
    """Synthesis max_isl flows through unchanged at the DTO layer."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            input_file=temp_file.name,
            synthesis_max_isl=4096,
        )
        assert config.synthesis_max_isl == 4096


def test_synthesis_max_osl_with_trace_dataset_succeeds():
    """Synthesis max_osl flows through unchanged at the DTO layer."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
        config = CLIConfig(
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            input_file=temp_file.name,
            synthesis_max_osl=2048,
        )
        assert config.synthesis_max_osl == 2048
