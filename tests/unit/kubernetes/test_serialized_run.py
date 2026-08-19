# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for safe serialized BenchmarkRun reads."""

from pathlib import Path

import pytest

from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.serialized_run import read_serialized_run_json


def test_read_serialized_run_json_reads_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"benchmark_id":"run"}', encoding="utf-8")

    assert read_serialized_run_json(path) == '{"benchmark_id":"run"}'


def test_read_serialized_run_json_accepts_configmap_atomic_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount = tmp_path / "config"
    data = mount / "..2026_08_04_00_00_00"
    data.mkdir(parents=True)
    (data / "run_config.json").write_text('{"benchmark_id":"run"}', encoding="utf-8")
    (mount / "..data").symlink_to(data.name)
    path = mount / "run_config.json"
    path.symlink_to("..data/run_config.json")
    monkeypatch.setattr(K8sEnvironment.JOBSET, "CONFIG_MOUNT_PATH", str(mount))

    assert read_serialized_run_json(path) == '{"benchmark_id":"run"}'


def test_read_serialized_run_json_rejects_configmap_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount = tmp_path / "config"
    data = mount / "..2026_08_04_00_00_00"
    data.mkdir(parents=True)
    (mount / "..data").symlink_to(data.name)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    path = mount / "run_config.json"
    path.symlink_to(outside)
    monkeypatch.setattr(K8sEnvironment.JOBSET, "CONFIG_MOUNT_PATH", str(mount))

    assert read_serialized_run_json(path) is None
