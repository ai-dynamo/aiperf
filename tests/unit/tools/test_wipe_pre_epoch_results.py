# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools.wipe_pre_epoch_results import scan_pre_epoch, wipe_pre_epoch


def _make(p: Path, body: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_scan_identifies_pre_epoch_job_dirs(tmp_path: Path) -> None:
    # Pre-epoch shape: profile_export_aiperf.json directly under <ns>/<name>/
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    # Epoch shape: <ns>/<name>/<epoch>/profile_export_aiperf.json
    _make(tmp_path / "bench" / "new-job" / "1714069323" / "profile_export_aiperf.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("old-job" in p for p in paths)
    assert not any("new-job" in p for p in paths)


def test_scan_identifies_pre_epoch_sweep_dirs(tmp_path: Path) -> None:
    # Pre-epoch sweep: aggregate.json directly under <ns>/sweeps/<name>/
    _make(tmp_path / "bench" / "sweeps" / "old-sweep" / "aggregate.json")
    # Epoch shape:
    _make(tmp_path / "bench" / "sweeps" / "new-sweep" / "1714069323" / "aggregate.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("old-sweep" in p for p in paths)
    assert not any("new-sweep" in p for p in paths)


def test_scan_identifies_legacy_subdir(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "mig-job" / "legacy" / "profile_export_aiperf.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("mig-job" in p for p in paths)


def test_wipe_apply_actually_deletes(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    n = wipe_pre_epoch(tmp_path, dry_run=False)
    assert n >= 1
    assert not (tmp_path / "bench" / "old-job").exists()


def test_wipe_dry_run_keeps_files(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    n = wipe_pre_epoch(tmp_path, dry_run=True)
    assert n >= 1
    assert (tmp_path / "bench" / "old-job").exists()
