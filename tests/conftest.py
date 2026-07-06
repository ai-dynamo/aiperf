# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test configuration and fixtures for all test types.

ONLY ADD FIXTURES HERE THAT ARE USED IN ALL TEST TYPES.
DO NOT ADD FIXTURES THAT ARE ONLY USED IN A SPECIFIC TEST TYPE.
"""

import math
import os
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

PYTEST_AUTO_WORKER_CPU_FRACTION_ENV = "AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION"
PYTEST_XDIST_AUTO_WORKERS_ENV = "PYTEST_XDIST_AUTO_NUM_WORKERS"
DEFAULT_PYTEST_AUTO_WORKER_CPU_FRACTION = 0.75


def _read_cgroup_v2_cpu_capacity(cpu_max_path: Path) -> float | None:
    try:
        quota_text, period_text = cpu_max_path.read_text().strip().split()[:2]
        if quota_text == "max":
            return None
        quota = int(quota_text)
        period = int(period_text)
    except (OSError, ValueError):
        return None

    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _read_cgroup_v1_cpu_capacity(quota_path: Path, period_path: Path) -> float | None:
    try:
        quota = int(quota_path.read_text().strip())
        period = int(period_path.read_text().strip())
    except (OSError, ValueError):
        return None

    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _cgroup_relative_path(cgroup_path: str) -> Path:
    return Path(cgroup_path.lstrip("/"))


def _cgroup_v1_cpu_mount_dir_candidates(
    controller_field: str | None = None,
) -> list[str]:
    candidates: list[str] = []
    if controller_field:
        candidates.append(controller_field)
    for mount_dir in ("cpu", "cpu,cpuacct"):
        if mount_dir not in candidates:
            candidates.append(mount_dir)
    return candidates


def _cgroup_v2_cpu_max_path_candidates(
    cgroup_v2_root: Path, cgroup_path: str
) -> list[Path]:
    relative_path = _cgroup_relative_path(cgroup_path)
    candidates: list[Path] = []
    while True:
        candidates.append(cgroup_v2_root / relative_path / "cpu.max")
        if relative_path == Path("."):
            break
        relative_path = relative_path.parent
    return candidates


def _read_proc_self_cgroup_paths(
    proc_self_cgroup_path: Path,
) -> list[tuple[list[str], str, str]]:
    try:
        lines = proc_self_cgroup_path.read_text().splitlines()
    except OSError:
        return []

    entries = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controller_field = fields[1]
        controllers = [
            controller for controller in controller_field.split(",") if controller
        ]
        entries.append((controllers, controller_field, fields[2]))
    return entries


def _read_cgroup_cpu_capacity(
    proc_self_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_v2_root: Path = Path("/sys/fs/cgroup"),
    cgroup_v1_root: Path = Path("/sys/fs/cgroup"),
) -> float | None:
    for controllers, _controller_field, cgroup_path in _read_proc_self_cgroup_paths(
        proc_self_cgroup_path
    ):
        if controllers:
            continue
        cgroup_v2_capacities = [
            cgroup_v2_capacity
            for cpu_max_path in _cgroup_v2_cpu_max_path_candidates(
                cgroup_v2_root, cgroup_path
            )
            if (cgroup_v2_capacity := _read_cgroup_v2_cpu_capacity(cpu_max_path))
            is not None
        ]
        if cgroup_v2_capacities:
            return min(cgroup_v2_capacities)

    for controllers, controller_field, cgroup_path in _read_proc_self_cgroup_paths(
        proc_self_cgroup_path
    ):
        if "cpu" not in controllers:
            continue
        for mount_dir in _cgroup_v1_cpu_mount_dir_candidates(controller_field):
            cgroup_v1_path = (
                cgroup_v1_root / mount_dir / _cgroup_relative_path(cgroup_path)
            )
            cgroup_v1_capacity = _read_cgroup_v1_cpu_capacity(
                cgroup_v1_path / "cpu.cfs_quota_us",
                cgroup_v1_path / "cpu.cfs_period_us",
            )
            if cgroup_v1_capacity is not None:
                return cgroup_v1_capacity

    cgroup_v2_capacity = _read_cgroup_v2_cpu_capacity(cgroup_v2_root / "cpu.max")
    if cgroup_v2_capacity is not None:
        return cgroup_v2_capacity

    for mount_dir in _cgroup_v1_cpu_mount_dir_candidates():
        cgroup_v1_capacity = _read_cgroup_v1_cpu_capacity(
            cgroup_v1_root / mount_dir / "cpu.cfs_quota_us",
            cgroup_v1_root / mount_dir / "cpu.cfs_period_us",
        )
        if cgroup_v1_capacity is not None:
            return cgroup_v1_capacity

    return None


def _detect_pytest_cpu_capacity() -> float:
    affinity_cpu_count = None
    with suppress(AttributeError, OSError):
        affinity_cpu_count = len(os.sched_getaffinity(0))

    cgroup_capacity = _read_cgroup_cpu_capacity()
    if cgroup_capacity is not None:
        if affinity_cpu_count is not None and affinity_cpu_count > 0:
            return min(cgroup_capacity, float(affinity_cpu_count))
        return cgroup_capacity

    physical_cpu_count = psutil.cpu_count(logical=False)
    if physical_cpu_count is not None and physical_cpu_count > 0:
        if affinity_cpu_count is not None and affinity_cpu_count > 0:
            return float(min(physical_cpu_count, affinity_cpu_count))
        return float(physical_cpu_count)

    if affinity_cpu_count is not None and affinity_cpu_count > 0:
        return float(affinity_cpu_count)

    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 0:
        return float(cpu_count)
    return 1.0


def _get_pytest_auto_worker_cpu_fraction() -> float:
    value = os.environ.get(PYTEST_AUTO_WORKER_CPU_FRACTION_ENV)
    if value is None:
        return DEFAULT_PYTEST_AUTO_WORKER_CPU_FRACTION

    try:
        fraction = float(value)
    except ValueError:
        return DEFAULT_PYTEST_AUTO_WORKER_CPU_FRACTION

    if not math.isfinite(fraction) or fraction <= 0:
        return DEFAULT_PYTEST_AUTO_WORKER_CPU_FRACTION
    return fraction


def _calculate_pytest_auto_workers(cpu_capacity: float, fraction: float) -> int:
    return max(1, math.floor(cpu_capacity * fraction))


def _get_explicit_pytest_auto_workers() -> int | None:
    env_num_workers = os.environ.get(PYTEST_XDIST_AUTO_WORKERS_ENV)
    if env_num_workers is None:
        return None

    try:
        num_workers = int(env_num_workers)
    except ValueError:
        return None

    if num_workers <= 0:
        return None
    return num_workers


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    numprocesses = getattr(getattr(config, "option", None), "numprocesses", "auto")
    if numprocesses == "logical":
        return None

    env_num_workers = _get_explicit_pytest_auto_workers()
    if env_num_workers is not None:
        return env_num_workers

    return _calculate_pytest_auto_workers(
        _detect_pytest_cpu_capacity(),
        _get_pytest_auto_worker_cpu_fraction(),
    )
