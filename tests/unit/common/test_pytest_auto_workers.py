# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest


def test_explicit_xdist_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "7")

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 7


@pytest.mark.parametrize("value", ["many", "0", "-2"])
def test_invalid_explicit_xdist_env_override_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", value)
    monkeypatch.setattr(conftest, "_detect_pytest_cpu_capacity", lambda: 8.0)

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 6


def test_logical_xdist_mode_delegates_to_xdist_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)

    config = SimpleNamespace(option=SimpleNamespace(numprocesses="logical"))

    assert conftest.pytest_xdist_auto_num_workers(config) is None


def test_xdist_auto_num_workers_hook_is_optional() -> None:
    assert conftest.pytest_xdist_auto_num_workers.pytest_impl["optionalhook"] is True


def test_cgroup_v2_quota_drives_capacity(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("250000 100000\n")

    assert conftest._read_cgroup_v2_cpu_capacity(cpu_max) == 2.5


def test_cgroup_v2_max_has_no_capacity_limit(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000\n")

    assert conftest._read_cgroup_v2_cpu_capacity(cpu_max) is None


def test_cgroup_v2_proc_self_path_resolves_nested_cpu_max(tmp_path: Path) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    nested = cgroup_root / "user.slice" / "session.scope"
    nested.mkdir(parents=True)
    proc_self_cgroup.write_text("0::/user.slice/session.scope\n")
    (nested / "cpu.max").write_text("350000 100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v2_root=cgroup_root,
        )
        == 3.5
    )


def test_cgroup_v2_proc_self_path_uses_ancestor_quota(tmp_path: Path) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    ancestor = cgroup_root / "pod"
    nested = ancestor / "container"
    nested.mkdir(parents=True)
    proc_self_cgroup.write_text("0::/pod/container\n")
    (nested / "cpu.max").write_text("max 100000\n")
    (ancestor / "cpu.max").write_text("200000 100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v2_root=cgroup_root,
        )
        == 2.0
    )


def test_cgroup_v2_proc_self_path_uses_minimum_ancestor_quota(
    tmp_path: Path,
) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    ancestor = cgroup_root / "pod"
    nested = ancestor / "container"
    nested.mkdir(parents=True)
    proc_self_cgroup.write_text("0::/pod/container\n")
    (nested / "cpu.max").write_text("800000 100000\n")
    (ancestor / "cpu.max").write_text("200000 100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v2_root=cgroup_root,
        )
        == 2.0
    )


def test_cgroup_v1_quota_drives_capacity(tmp_path: Path) -> None:
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("300000\n")
    period.write_text("100000\n")

    assert conftest._read_cgroup_v1_cpu_capacity(quota, period) == 3.0


def test_cgroup_v1_negative_quota_has_no_capacity_limit(tmp_path: Path) -> None:
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("-1\n")
    period.write_text("100000\n")

    assert conftest._read_cgroup_v1_cpu_capacity(quota, period) is None


def test_cgroup_v1_proc_self_path_resolves_nested_quota(tmp_path: Path) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_v1_root = tmp_path / "sys" / "fs" / "cgroup"
    nested = cgroup_v1_root / "cpu" / "user.slice" / "session.scope"
    nested.mkdir(parents=True)
    proc_self_cgroup.write_text("2:cpu,cpuacct:/user.slice/session.scope\n")
    (nested / "cpu.cfs_quota_us").write_text("450000\n")
    (nested / "cpu.cfs_period_us").write_text("100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v1_root=cgroup_v1_root,
        )
        == 4.5
    )


def test_cgroup_v1_combined_cpu_controller_mount_resolves_quota(tmp_path: Path) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_v1_root = tmp_path / "sys" / "fs" / "cgroup"
    nested = cgroup_v1_root / "cpu,cpuacct" / "docker" / "abc"
    nested.mkdir(parents=True)
    proc_self_cgroup.write_text("2:cpu,cpuacct:/docker/abc\n")
    (nested / "cpu.cfs_quota_us").write_text("250000\n")
    (nested / "cpu.cfs_period_us").write_text("100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v1_root=cgroup_v1_root,
        )
        == 2.5
    )


def test_cgroup_v1_root_fallback_checks_combined_cpu_controller_mount(
    tmp_path: Path,
) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    cgroup_v1_root = tmp_path / "sys" / "fs" / "cgroup"
    cgroup_v2_root = tmp_path / "empty-cgroup-v2"
    combined_root = cgroup_v1_root / "cpu,cpuacct"
    cgroup_v2_root.mkdir()
    combined_root.mkdir(parents=True)
    proc_self_cgroup.write_text("")
    (combined_root / "cpu.cfs_quota_us").write_text("550000\n")
    (combined_root / "cpu.cfs_period_us").write_text("100000\n")

    assert (
        conftest._read_cgroup_cpu_capacity(
            proc_self_cgroup_path=proc_self_cgroup,
            cgroup_v2_root=cgroup_v2_root,
            cgroup_v1_root=cgroup_v1_root,
        )
        == 5.5
    )


def test_cgroup_capacity_takes_precedence_over_psutil_physical_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: 2.5)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: 24)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )

    assert conftest._detect_pytest_cpu_capacity() == 2.5


def test_cgroup_capacity_is_capped_by_restricted_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: 16.0)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: 24)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: set(range(4)), raising=False
    )

    assert conftest._detect_pytest_cpu_capacity() == 4.0


def test_cgroup_capacity_used_when_affinity_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_oserror(pid: int) -> set[int]:
        raise OSError("sched_getaffinity unavailable")

    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: 16.0)
    monkeypatch.setattr(conftest.os, "sched_getaffinity", raise_oserror, raising=False)

    assert conftest._detect_pytest_cpu_capacity() == 16.0


def test_psutil_physical_cores_precede_larger_logical_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: None)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: 24)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )

    assert conftest._detect_pytest_cpu_capacity() == 24.0


def test_psutil_physical_cores_are_capped_by_restricted_logical_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: None)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: 24)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: set(range(4)), raising=False
    )

    assert conftest._detect_pytest_cpu_capacity() == 4.0


def test_logical_affinity_used_when_psutil_physical_cores_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: None)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: None)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )

    assert conftest._detect_pytest_cpu_capacity() == 32.0


def test_cpu_affinity_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: None)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: None)
    monkeypatch.setattr(
        conftest.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
    )
    monkeypatch.setattr(conftest.os, "cpu_count", lambda: 64)

    assert conftest._detect_pytest_cpu_capacity() == 3.0


def test_os_cpu_count_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "_read_cgroup_cpu_capacity", lambda: None)
    monkeypatch.setattr(conftest.psutil, "cpu_count", lambda logical=False: None)
    monkeypatch.delattr(conftest.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(conftest.os, "cpu_count", lambda: 12)

    assert conftest._detect_pytest_cpu_capacity() == 12.0


def test_default_fraction_caps_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION", raising=False)
    monkeypatch.setattr(conftest, "_detect_pytest_cpu_capacity", lambda: 10.0)

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 7


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "nan", "inf", "-inf"])
def test_invalid_fraction_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setenv("AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION", value)
    monkeypatch.setattr(conftest, "_detect_pytest_cpu_capacity", lambda: 8.0)

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 6


def test_valid_fraction_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setenv("AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION", "0.5")
    monkeypatch.setattr(conftest, "_detect_pytest_cpu_capacity", lambda: 8.0)

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 4


def test_worker_count_minimum_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setenv("AIPERF_PYTEST_AUTO_WORKER_CPU_FRACTION", "0.25")
    monkeypatch.setattr(conftest, "_detect_pytest_cpu_capacity", lambda: 1.0)

    assert conftest.pytest_xdist_auto_num_workers(SimpleNamespace()) == 1
