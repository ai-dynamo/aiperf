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
        "slice_duration": 1.0,
    }
    config = OutputConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


def test_profile_export_prefix_relative():
    """Relative prefix is joined with artifact_directory as-is."""
    config = OutputConfig(
        artifact_directory=Path("/results"),
        profile_export_prefix=Path("my_bench"),
    )
    assert config.profile_export_json_file == Path("/results/my_bench.json")
    assert config.profile_export_csv_file == Path("/results/my_bench.csv")


def test_profile_export_prefix_absolute_uses_name_only():
    """Absolute prefix: only the name component is used so artifact_directory controls the directory.

    This prevents the pathlib 'absolute wins' rule from bypassing per-run artifact isolation
    in multi-run mode (orchestrator sets artifact_directory per run).
    """
    config = OutputConfig(
        artifact_directory=Path("/results"),
        profile_export_prefix=Path("/tmp/aiperf_h2_test"),
    )
    assert config.profile_export_json_file == Path("/results/aiperf_h2_test.json")
    assert config.profile_export_csv_file == Path("/results/aiperf_h2_test.csv")


def test_profile_export_prefix_absolute_per_run_isolation():
    """Changing artifact_directory after construction re-routes output files per run."""
    config = OutputConfig(
        artifact_directory=Path("/results/run_0001"),
        profile_export_prefix=Path("/tmp/aiperf_h2_test"),
    )
    assert config.profile_export_json_file == Path(
        "/results/run_0001/aiperf_h2_test.json"
    )

    config.artifact_directory = Path("/results/run_0002")
    assert config.profile_export_json_file == Path(
        "/results/run_0002/aiperf_h2_test.json"
    )
