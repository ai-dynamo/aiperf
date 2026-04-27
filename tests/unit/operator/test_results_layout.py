# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.results_layout.

Covers the full public API: write_latest/resolve_latest (atomic pointer file),
resolve_run_dir (latest + explicit epoch + missing-epoch None), enforce_retention
(mtime ordering, keep count, protect_epoch guarantee), epoch_key_from_body.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from aiperf.operator.results_layout import (
    EPOCH_RE,
    LATEST_POINTER,
    enforce_retention,
    epoch_key_from_body,
    job_dir,
    list_run_epochs,
    resolve_latest,
    resolve_run_dir,
    resolve_sweep_dir,
    run_dir,
    write_latest,
)

EPOCH_A = "1714064523"
EPOCH_B = "1714064589"
EPOCH_C = "1714150923"


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


def test_epoch_re_no_longer_matches_legacy() -> None:
    assert EPOCH_RE.match("legacy") is None
    assert EPOCH_RE.match("1714069323") is not None


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


def test_epoch_key_from_body_parses_iso_timestamp() -> None:
    body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}
    result = epoch_key_from_body(body)
    assert result.isdigit()
    assert 9 <= len(result) <= 11


def test_epoch_key_from_body_stable_across_calls() -> None:
    body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}
    assert epoch_key_from_body(body) == epoch_key_from_body(body)


def test_enforce_retention_age_and_count_both_apply(tmp_path: Path) -> None:
    now = time.time()
    old_epoch, recent1, recent2 = "1700000000", "1714000000", "1714100000"
    for epoch, age_days in [(old_epoch, 100), (recent1, 1), (recent2, 0)]:
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        os.utime(d, (now - age_days * 86400, now - age_days * 86400))
    # keep=10 (everything in count window) AND retain_days=30 (only old eligible)
    # Intersection: only old_epoch is deleted.
    deleted = enforce_retention(
        tmp_path,
        "ns",
        "job",
        keep=10,
        protect_epoch=recent2,
        retain_days=30,
    )
    assert deleted == [old_epoch]


def test_enforce_retention_age_only_doesnt_delete_within_count_window(
    tmp_path: Path,
) -> None:
    now = time.time()
    epoch = "1700000000"
    d = run_dir(tmp_path, "ns", "job", epoch)
    d.mkdir(parents=True)
    os.utime(d, (now - 100 * 86400, now - 100 * 86400))
    # keep=10 says "keep"; age says "too old". Intersection = keep (conservative).
    deleted = enforce_retention(
        tmp_path,
        "ns",
        "job",
        keep=10,
        protect_epoch=epoch,
        retain_days=30,
    )
    assert deleted == []
    assert epoch in list_run_epochs(tmp_path, "ns", "job")


def test_enforce_retention_retain_days_zero_disables_age_policy(
    tmp_path: Path,
) -> None:
    now = time.time()
    epochs = ["1710000000", "1711000000", "1712000000"]
    for i, epoch in enumerate(epochs):
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        # epochs[-1] is newest so it matches protect_epoch under count-only.
        age_days = len(epochs) - i
        os.utime(d, (now - age_days * 86400, now - age_days * 86400))
    # keep=1 forces reap of two; retain_days=0 = age policy off -> count alone.
    deleted = enforce_retention(
        tmp_path,
        "ns",
        "job",
        keep=1,
        protect_epoch=epochs[-1],
        retain_days=0,
    )
    assert len(deleted) == 2


def test_enforce_retention_dry_run_returns_candidates_without_deleting(
    tmp_path: Path,
) -> None:
    now = time.time()
    epochs = ["1710000000", "1711000000", "1712000000"]
    for i, epoch in enumerate(epochs):
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        age_days = len(epochs) - i
        os.utime(d, (now - age_days * 86400, now - age_days * 86400))

    deleted = enforce_retention(
        tmp_path,
        "ns",
        "job",
        keep=1,
        protect_epoch=epochs[-1],
        retain_days=0,
        dry_run=True,
    )
    assert len(deleted) == 2
    # No runs actually removed from disk.
    survivors = set(list_run_epochs(tmp_path, "ns", "job"))
    assert survivors == set(epochs)


def test_enforce_retention_dry_run_matches_live_candidates(
    tmp_path: Path,
) -> None:
    def _seed(base: Path) -> list[str]:
        now = time.time()
        epochs = ["1710000000", "1711000000", "1712000000", "1713000000"]
        for i, epoch in enumerate(epochs):
            d = run_dir(base, "ns", "job", epoch)
            d.mkdir(parents=True)
            age_days = len(epochs) - i
            os.utime(d, (now - age_days * 86400, now - age_days * 86400))
        return epochs

    dry_base = tmp_path / "dry"
    dry_base.mkdir()
    epochs = _seed(dry_base)
    dry = enforce_retention(
        dry_base,
        "ns",
        "job",
        keep=2,
        protect_epoch=epochs[-1],
        retain_days=0,
        dry_run=True,
    )

    live_base = tmp_path / "live"
    live_base.mkdir()
    _seed(live_base)
    live = enforce_retention(
        live_base,
        "ns",
        "job",
        keep=2,
        protect_epoch=epochs[-1],
        retain_days=0,
    )
    assert sorted(dry) == sorted(live)


def test_resolve_sweep_dir_returns_path_when_present(tmp_path: Path) -> None:
    base = tmp_path
    sweep_dir = base / "bench" / "sweeps" / "saturation-sweep"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text("{}")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") == sweep_dir


def test_resolve_sweep_dir_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_sweep_dir(tmp_path, "bench", "nope") is None


def test_resolve_sweep_dir_returns_none_when_not_a_directory(tmp_path: Path) -> None:
    base = tmp_path
    (base / "bench" / "sweeps").mkdir(parents=True)
    (base / "bench" / "sweeps" / "saturation-sweep").write_text("not a dir")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") is None
