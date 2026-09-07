# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube profile`'s sweep-config hand-off helper."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_profile_check_no_sweep_keys_errors_on_sweep() -> None:
    """`sweep:` in a profile config triggers SystemExit with a hand-off message."""
    from aiperf.cli_commands.kube.profile import _check_no_sweep_keys

    config_dict = {"models": ["x"], "sweep": {"type": "grid"}}
    with pytest.raises(SystemExit):
        _check_no_sweep_keys(config_dict, source="my-config.yaml")


def test_profile_check_no_sweep_keys_errors_on_multi_run() -> None:
    """`multi_run:` in a profile config also triggers SystemExit."""
    from aiperf.cli_commands.kube.profile import _check_no_sweep_keys

    config_dict = {"multi_run": {"trials": 5}}
    with pytest.raises(SystemExit):
        _check_no_sweep_keys(config_dict, source="x.yaml")


def test_profile_check_no_sweep_keys_errors_on_camelcase_multiRun() -> None:
    """`multiRun:` (camelCase) is also detected as a sweep key."""
    from aiperf.cli_commands.kube.profile import _check_no_sweep_keys

    config_dict = {"multiRun": {"trials": 3}}
    with pytest.raises(SystemExit):
        _check_no_sweep_keys(config_dict, source="x.yaml")


def test_profile_check_no_sweep_keys_passes_clean_config() -> None:
    """A plain benchmark config (no sweep/multi_run keys) is allowed."""
    from aiperf.cli_commands.kube.profile import _check_no_sweep_keys

    _check_no_sweep_keys({"models": ["x"]}, source="x.yaml")


def test_profile_check_config_file_skips_aiperfjob_cr(tmp_path: Path) -> None:
    """An AIPerfJob CR YAML is not subjected to sweep-key checks."""
    from aiperf.cli_commands.kube.profile import _check_config_file_for_sweep_keys

    cr = tmp_path / "job.yaml"
    cr.write_text(
        """
apiVersion: aiperf.nvidia.com/v1
kind: AIPerfJob
metadata: {name: x}
spec:
  benchmark:
    sweep: {type: grid}
"""
    )
    # Should NOT raise — sweep keys nested inside an AIPerfJob CR's spec are
    # the operator's domain, not the profile CLI's.
    _check_config_file_for_sweep_keys(cr)


def test_profile_check_config_file_flags_plain_yaml_with_sweep(tmp_path: Path) -> None:
    """Plain (non-CR) YAML config with sweep: triggers SystemExit."""
    from aiperf.cli_commands.kube.profile import _check_config_file_for_sweep_keys

    plain = tmp_path / "plain.yaml"
    plain.write_text(
        """
models: [x]
sweep:
  type: grid
  parameters: {random_seed: [1, 2]}
"""
    )
    with pytest.raises(SystemExit):
        _check_config_file_for_sweep_keys(plain)


def test_profile_check_config_file_no_file_is_noop() -> None:
    """No config file = nothing to check."""
    from aiperf.cli_commands.kube.profile import _check_config_file_for_sweep_keys

    _check_config_file_for_sweep_keys(None)
