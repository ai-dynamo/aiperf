# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.common.config import OutputConfig, OutputDefaults


def test_output_config_defaults():
    """
    Test the default values of the OutputConfig class.

    This test verifies that the OutputConfig object is initialized with the correct
    default values as defined in the OutputDefaults class.
    """
    config = OutputConfig()
    assert config.artifact_directory == OutputDefaults.ARTIFACT_DIRECTORY
    assert config.slice_duration == OutputDefaults.SLICE_DURATION


def test_output_config_custom_values():
    """
    Test the OutputConfig class with custom values.

    This test verifies that the OutputConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "artifact_directory": Path("/custom/artifact/directory"),
        "slice_duration": 1000,
    }
    config = OutputConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
