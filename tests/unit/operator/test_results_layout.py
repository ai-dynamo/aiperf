# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.results_layout.

Covers the full public API: write_latest/resolve_latest (atomic pointer file),
resolve_run_dir (latest + explicit epoch + missing-epoch None), enforce_retention
(mtime ordering, keep count, protect_epoch guarantee), migrate_legacy_layout
(relocates pre-migration files, idempotent, mixed layouts), epoch_key_from_body.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from aiperf.operator.results_layout import (
    LATEST_POINTER,
    enforce_retention,
    epoch_key_from_body,
    job_dir,
    list_run_epochs,
    migrate_legacy_layout,
    resolve_latest,
    resolve_run_dir,
    run_dir,
    write_latest,
)

EPOCH_A = "1714064523"
EPOCH_B = "1714064589"
EPOCH_C = "1714150923"


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_write_latest_atomic(tmp_path: Path) -> None:
    write_latest(tmp_path, "ns", "job", EPOCH_A)
    assert resolve_latest(tmp_path, "ns", "job") == EPOCH_A
    write_latest(tmp_path, "ns", "job", EPOCH_B)
    assert resolve_latest(tmp_path, "ns", "job") == EPOCH_B


def test_resolve_latest_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_latest(tmp_path, "ns", "job") is None


def test_resolve_run_dir_epoch_none_uses_latest(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", EPOCH_A).mkdir(parents=True)
    write_latest(tmp_path, "ns", "job", EPOCH_A)
    assert resolve_run_dir(tmp_path, "ns", "job") == run_dir(
        tmp_path, "ns", "job", EPOCH_A
    )


def test_resolve_run_dir_explicit_epoch(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", EPOCH_A).mkdir(parents=True)
    run_dir(tmp_path, "ns", "job", EPOCH_B).mkdir(parents=True)
    write_latest(tmp_path, "ns", "job", EPOCH_B)
    assert resolve_run_dir(tmp_path, "ns", "job", epoch=EPOCH_A) == run_dir(
        tmp_path, "ns", "job", EPOCH_A
    )


def test_resolve_run_dir_epoch_not_on_disk_returns_none(tmp_path: Path) -> None:
    assert resolve_run_dir(tmp_path, "ns", "job", epoch=EPOCH_A) is None


def test_resolve_run_dir_latest_points_at_missing_epoch_returns_none(
    tmp_path: Path,
) -> None:
    write_latest(tmp_path, "ns", "job", EPOCH_A)
    assert resolve_run_dir(tmp_path, "ns", "job") is None


def test_list_run_epochs_lists_only_epoch_shaped_dirs(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", EPOCH_A).mkdir(parents=True)
    run_dir(tmp_path, "ns", "job", EPOCH_B).mkdir(parents=True)
    (job_dir(tmp_path, "ns", "job") / "not-epoch-dir").mkdir()
    (job_dir(tmp_path, "ns", "job") / LATEST_POINTER).write_text(EPOCH_A)
    epochs = set(list_run_epochs(tmp_path, "ns", "job"))
    assert epochs == {EPOCH_A, EPOCH_B}


def test_list_run_epochs_includes_legacy(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", "legacy").mkdir(parents=True)
    assert "legacy" in list_run_epochs(tmp_path, "ns", "job")


def test_enforce_retention_keeps_n_newest(tmp_path: Path) -> None:
    base_time = time.time()
    epochs = [str(1714000000 + i * 60) for i in range(15)]
    for idx, e in enumerate(epochs):
        d = run_dir(tmp_path, "ns", "job", e)
        d.mkdir(parents=True)
        mtime = base_time - (idx * 60)
        os.utime(d, (mtime, mtime))
    deleted = enforce_retention(tmp_path, "ns", "job", keep=10, protect_epoch=epochs[0])
    assert len(deleted) == 5
    survivors = set(list_run_epochs(tmp_path, "ns", "job"))
    assert len(survivors) == 10
    assert epochs[0] in survivors


def test_enforce_retention_protects_epoch_even_if_oldest(tmp_path: Path) -> None:
    base_time = time.time()
    epochs = ["1714000001", "1714000002", "1714000003"]
    for idx, e in enumerate(epochs):
        d = run_dir(tmp_path, "ns", "job", e)
        d.mkdir(parents=True)
        mtime = base_time - (idx * 60)
        os.utime(d, (mtime, mtime))
    enforce_retention(tmp_path, "ns", "job", keep=1, protect_epoch=epochs[2])
    survivors = set(list_run_epochs(tmp_path, "ns", "job"))
    assert epochs[0] in survivors
    assert epochs[2] in survivors


def test_enforce_retention_empty_dir_noop(tmp_path: Path) -> None:
    assert (
        enforce_retention(tmp_path, "ns", "job", keep=10, protect_epoch=EPOCH_A) == []
    )


def test_migrate_legacy_layout_relocates_files(tmp_path: Path) -> None:
    _touch(tmp_path / "ns" / "job" / "foo.json", b'{"ok": true}')
    _touch(tmp_path / "ns" / "job" / "checkpoints" / "chk.json", b"{}")
    migrated = migrate_legacy_layout(tmp_path)
    assert migrated == [("ns", "job")]
    assert (tmp_path / "ns" / "job" / "legacy" / "foo.json").is_file()
    assert (tmp_path / "ns" / "job" / "legacy" / "checkpoints" / "chk.json").is_file()
    assert resolve_latest(tmp_path, "ns", "job") == "legacy"


def test_migrate_legacy_layout_idempotent(tmp_path: Path) -> None:
    _touch(tmp_path / "ns" / "job" / "foo.json")
    migrate_legacy_layout(tmp_path)
    assert migrate_legacy_layout(tmp_path) == []


def test_migrate_legacy_layout_skips_already_migrated(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", EPOCH_A).mkdir(parents=True)
    _touch(run_dir(tmp_path, "ns", "job", EPOCH_A) / "foo.json")
    write_latest(tmp_path, "ns", "job", EPOCH_A)
    assert migrate_legacy_layout(tmp_path) == []
    assert resolve_latest(tmp_path, "ns", "job") == EPOCH_A


def test_migrate_legacy_layout_mixed_epoch_and_legacy(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", EPOCH_A).mkdir(parents=True)
    _touch(tmp_path / "ns" / "other" / "bar.json")
    migrated = migrate_legacy_layout(tmp_path)
    assert migrated == [("ns", "other")]
    assert (tmp_path / "ns" / "job" / EPOCH_A).is_dir()
    assert (tmp_path / "ns" / "other" / "legacy" / "bar.json").is_file()


def test_migrate_legacy_layout_empty_name_dir_noop(tmp_path: Path) -> None:
    (tmp_path / "ns" / "job").mkdir(parents=True)
    assert migrate_legacy_layout(tmp_path) == []
    assert resolve_latest(tmp_path, "ns", "job") is None


def test_epoch_key_from_body_parses_iso_timestamp() -> None:
    body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}
    result = epoch_key_from_body(body)
    assert result.isdigit()
    assert 9 <= len(result) <= 11


def test_epoch_key_from_body_stable_across_calls() -> None:
    body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}
    assert epoch_key_from_body(body) == epoch_key_from_body(body)
