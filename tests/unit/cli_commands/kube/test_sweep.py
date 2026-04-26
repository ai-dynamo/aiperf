# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube sweep` CR-builder helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.cli_commands.kube import sweep as sweep_cmd


def _kube_options_mock() -> MagicMock:
    """Stub that satisfies the attribute access in `_build_sweep_cr_dict`."""
    m = MagicMock()
    m.image = "x:latest"
    m.namespace = "ns"
    m.name = None
    m.kubeconfig = None
    m.kube_context = None
    deployment = MagicMock()
    deployment.model_dump = MagicMock(return_value={})
    m.to_deployment_config = MagicMock(return_value=deployment)
    return m


def test_build_sweep_cr_dict_emits_aiperfsweep_kind(tmp_path: Path) -> None:
    """YAML with sweep:+multi_run: produces an AIPerfSweep CR with hoisted spec."""
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """
models: [Qwen/Qwen3-0.6B]
endpoint:
  urls: [http://localhost:8000/v1/chat/completions]
  type: chat
  streaming: true
datasets:
  main: {type: synthetic}
phases:
  - name: profiling
    type: concurrency
    duration: 5
    concurrency: 1
sweep:
  type: grid
  variables:
    random_seed: [1, 2]
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=2,
        cooldown_seconds=10,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["kind"] == "AIPerfSweep"
    assert cr["apiVersion"] == "aiperf.nvidia.com/v1"
    assert "sweep" in cr["spec"]
    assert cr["spec"]["multiRun"]["trials"] == 2
    assert cr["spec"]["multiRun"]["cooldownSeconds"] == 10
    # benchmark in template should NOT have sweep:
    assert "sweep" not in cr["spec"]["template"]["spec"]["benchmark"]


def test_build_sweep_cr_dict_with_convergence(tmp_path: Path) -> None:
    """--convergence-metric populates spec.convergence with min/max/threshold."""
    config_file = tmp_path / "conf.yaml"
    config_file.write_text(
        """
models: [m]
endpoint:
  urls: [http://x]
  type: chat
datasets:
  main: {type: synthetic}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
multi_run:
  cooldown_seconds: 30
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric="ttft_p99",
        convergence_min_runs=3,
        convergence_max_runs=7,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["convergence"]["metric"] == "ttft_p99"
    assert cr["spec"]["convergence"]["maxRuns"] == 7
    # multi_run from yaml is preserved (cooldown_seconds present in original)
    assert "multiRun" in cr["spec"]


def test_build_sweep_cr_dict_requires_config_file() -> None:
    """No --config <file> raises a helpful ValueError."""
    with pytest.raises(ValueError, match="--config <file>"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=None,
            kube_options=_kube_options_mock(),
            multi_run_trials=None,
            cooldown_seconds=0,
            convergence_metric=None,
            convergence_min_runs=3,
            convergence_max_runs=10,
            convergence_threshold=0.05,
        )


def test_build_sweep_cr_dict_default_name_from_config_stem(tmp_path: Path) -> None:
    """When no --name given, derive ``<stem>-sweep`` from the config file stem."""
    config_file = tmp_path / "concurrency_grid.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: {main: {type: synthetic}}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
sweep:
  type: grid
  variables: {random_seed: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["metadata"]["name"] == "concurrency-grid-sweep"


def test_build_sweep_cr_dict_no_sweep_or_multirun(tmp_path: Path) -> None:
    """Plain config (no sweep:/multi_run:) still builds a valid CR (sweep-only)."""
    config_file = tmp_path / "plain.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: {main: {type: synthetic}}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["kind"] == "AIPerfSweep"
    assert "sweep" not in cr["spec"]
    assert "multiRun" not in cr["spec"]
    assert "convergence" not in cr["spec"]


def test_name_from_config_file_sanitizes_and_truncates() -> None:
    """Stem is lowercased, sanitized to [a-z0-9-], capped at 30 chars + '-sweep'."""
    out = sweep_cmd._name_from_config_file(Path("My_Crazy.Config.YAML"))
    assert out.endswith("-sweep")
    assert all(c.islower() or c.isdigit() or c == "-" for c in out)


def test_camelcase_multiRun_key_also_hoisted(tmp_path: Path) -> None:
    """Top-level ``multiRun:`` (camelCase) is honored as a fallback for ``multi_run:``."""
    config_file = tmp_path / "cc.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: {main: {type: synthetic}}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
multiRun:
  trials: 4
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["multiRun"]["trials"] == 4
    assert "multiRun" not in cr["spec"]["template"]["spec"]["benchmark"]
