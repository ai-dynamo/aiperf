# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.config.artifacts import OutputDefaults
from aiperf.config.flags import CLIConfig


def test_output_config_defaults():
    """
    Test the default values of the output fields on CLIConfig.

    This test verifies that CLIConfig is initialized with the correct
    default values as defined in the OutputDefaults class.
    """
    config = CLIConfig()
    assert config.artifact_directory == OutputDefaults.ARTIFACT_DIRECTORY
    assert config.slice_duration == OutputDefaults.SLICE_DURATION


def test_output_config_custom_values():
    """
    Test the output fields on CLIConfig with custom values.

    This test verifies that CLIConfig correctly initializes its output
    fields when provided with a dictionary of custom values.
    """
    custom_values = {
        "artifact_directory": Path("/custom/artifact/directory"),
        "slice_duration": 1.0,
    }
    config = CLIConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


def test_profile_export_prefix_set_on_dto():
    """profile_export_prefix is stored verbatim on the v1 DTO; the resolved
    ``profile_export_<fmt>_file`` properties live on the resolved config,
    not on the v1 CLIConfig DTO."""
    config = CLIConfig(
        artifact_directory=Path("/results"),
        profile_export_prefix=Path("my_bench"),
    )
    assert config.profile_export_prefix == Path("my_bench")
    assert config.artifact_directory == Path("/results")
