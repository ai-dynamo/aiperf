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

import pytest

from aiperf.operator.results_layout import (
    EPOCH_RE,
    LATEST_POINTER,
    enforce_retention,
    epoch_key_from_body,
    job_dir,
    list_run_epochs,
    list_runs_async,
    list_sweep_epochs,
    list_sweep_epochs_async,
    resolve_latest,
    resolve_run_dir,
    resolve_sweep_dir,
    resolve_sweep_latest,
    run_dir,
    write_latest,
    write_sweep_latest,
)

EPOCH_A = "1714064523"
EPOCH_B = "1714064589"
EPOCH_C = "1714150923"
INT64_MAX = 9_223_372_036_854_775_807


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


def test_enforce_retention_requires_age_and_count_to_reap(tmp_path: Path) -> None:
    now = time.time()
    old_epoch, recent1, recent2 = "1700000000", "1714000000", "1714100000"
    for epoch, age_days in [(old_epoch, 100), (recent1, 1), (recent2, 0)]:
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        os.utime(d, (now - age_days * 86400, now - age_days * 86400))
    # keep=10 retains every run, so age alone cannot select old_epoch for
    # deletion under the conservative intersection policy.
    deleted = enforce_retention(
        tmp_path,
        "ns",
        "job",
        keep=10,
        protect_epoch=recent2,
        retain_days=30,
    )
    assert deleted == []
    assert set(list_run_epochs(tmp_path, "ns", "job")) == {
        old_epoch,
        recent1,
        recent2,
    }


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


def test_enforce_retention_age_enabled_dry_run_matches_live_candidates(
    tmp_path: Path,
) -> None:
    now = time.time()
    epochs_and_ages = [
        ("1710000000", 30),
        ("1711000000", 5),
        ("1712000000", 2),
        ("1713000000", 1),
    ]

    def _seed(base: Path) -> None:
        for epoch, age_days in epochs_and_ages:
            path = run_dir(base, "ns", "job", epoch)
            path.mkdir(parents=True)
            mtime = now - age_days * 86400
            os.utime(path, (mtime, mtime))

    dry_base = tmp_path / "dry-age"
    live_base = tmp_path / "live-age"
    _seed(dry_base)
    _seed(live_base)

    dry = enforce_retention(
        dry_base,
        "ns",
        "job",
        keep=2,
        protect_epoch="1713000000",
        retain_days=7,
        dry_run=True,
    )
    live = enforce_retention(
        live_base,
        "ns",
        "job",
        keep=2,
        protect_epoch="1713000000",
        retain_days=7,
    )

    assert dry == ["1710000000"]
    assert live == dry
    assert set(list_run_epochs(live_base, "ns", "job")) == {
        "1711000000",
        "1712000000",
        "1713000000",
    }


def test_resolve_sweep_dir_returns_path_when_present(tmp_path: Path) -> None:
    base = tmp_path
    epoch_dir = base / "bench" / "sweeps" / "saturation-sweep" / "1714069323"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_text("{}")
    write_sweep_latest(base, "bench", "saturation-sweep", "1714069323")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") == epoch_dir


def test_resolve_sweep_dir_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_sweep_dir(tmp_path, "bench", "nope") is None


def test_resolve_sweep_dir_returns_none_when_not_a_directory(tmp_path: Path) -> None:
    base = tmp_path
    (base / "bench" / "sweeps").mkdir(parents=True)
    (base / "bench" / "sweeps" / "saturation-sweep").write_text("not a dir")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") is None


def test_resolve_sweep_dir_with_epoch(tmp_path: Path) -> None:
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    p.mkdir(parents=True)
    assert resolve_sweep_dir(tmp_path, "bench", "s1", epoch="1714069323") == p


def test_resolve_sweep_dir_with_epoch_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_sweep_dir(tmp_path, "bench", "s1", epoch="9999999999") is None


def test_resolve_sweep_dir_no_epoch_resolves_via_latest(tmp_path: Path) -> None:
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    p.mkdir(parents=True)
    write_sweep_latest(tmp_path, "bench", "s1", "1714069323")
    assert resolve_sweep_dir(tmp_path, "bench", "s1") == p


def test_list_sweep_epochs_orders_by_epoch_asc(tmp_path: Path) -> None:
    base = tmp_path / "bench" / "sweeps" / "s1"
    (base / "1714069323").mkdir(parents=True)
    (base / "1714069324" / "aggregate.json").parent.mkdir(parents=True)
    (base / "1714069324" / "aggregate.json").write_text("{}")
    write_sweep_latest(tmp_path, "bench", "s1", "1714069324")
    epochs = list_sweep_epochs(tmp_path, "bench", "s1")
    assert [e.epoch for e in epochs] == ["1714069323", "1714069324"]
    assert epochs[-1].is_latest is True
    assert epochs[0].is_latest is False


def test_resolve_sweep_latest_returns_none_when_unset(tmp_path: Path) -> None:
    assert resolve_sweep_latest(tmp_path, "bench", "s1") is None


def test_epoch_key_from_body_preserves_subsecond_distinction() -> None:
    first = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03.123456Z"}}
    second = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03.654321Z"}}

    first_key = epoch_key_from_body(first)
    second_key = epoch_key_from_body(second)

    assert first_key != second_key
    assert EPOCH_RE.match(first_key) is not None
    assert EPOCH_RE.match(second_key) is not None


def test_epoch_key_from_body_keeps_integer_key_for_whole_seconds() -> None:
    body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03.000000Z"}}
    assert epoch_key_from_body(body) == "1714069323"


def test_epoch_key_from_body_disambiguates_same_second_k8s_uids() -> None:
    first = {
        "metadata": {
            "creationTimestamp": "2024-04-25T18:22:03Z",
            "uid": "11111111-1111-4111-8111-111111111111",
        }
    }
    second = {
        "metadata": {
            "creationTimestamp": "2024-04-25T18:22:03Z",
            "uid": "22222222-2222-4222-8222-222222222222",
        }
    }

    first_key = epoch_key_from_body(first)
    second_key = epoch_key_from_body(second)

    assert first_key != second_key
    assert first_key.startswith("1714069323")
    assert second_key.startswith("1714069323")
    assert int(first_key) <= INT64_MAX
    assert int(second_key) <= INT64_MAX
    assert EPOCH_RE.match(first_key) is not None
    assert EPOCH_RE.match(second_key) is not None


@pytest.mark.asyncio
async def test_list_runs_async_merges_disk_epochs_when_index_has_stale_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.operator.runs_index_models import RunIndexRow

    old = "1714064523"
    new = "1714150923"
    old_dir = run_dir(tmp_path, "bench", "job", old)
    new_dir = run_dir(tmp_path, "bench", "job", new)
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "old.json").write_text("{}")
    (new_dir / "new.json").write_text("{}")
    write_latest(tmp_path, "bench", "job", new)

    async def fake_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
        return [
            RunIndexRow(
                namespace=namespace,
                job_id=job_id,
                epoch=old,
                phase="Succeeded",
                is_latest=True,
                start_time=None,
                end_time=None,
                created_unix=int(old),
                mtime_epoch=1,
                error=None,
                model=None,
                endpoint=None,
                gpu_count=0,
                gpu_name=None,
                file_count=1,
                total_size_bytes=2,
                sweep_namespace=None,
                sweep_name=None,
                sweep_epoch=None,
                sweep_variation_idx=None,
            )
        ]

    monkeypatch.setattr("aiperf.operator.runs_index.list_runs_for_job", fake_index_rows)

    runs = await list_runs_async(tmp_path, "bench", "job")

    assert {run.epoch for run in runs} == {old, new}
    assert {run.epoch for run in runs if run.is_latest} == {new}


@pytest.mark.asyncio
async def test_list_sweep_epochs_async_merges_disk_epochs_when_index_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "1714064523"
    new = "1714150923"
    sweep_root = tmp_path / "bench" / "sweeps" / "s1"
    (sweep_root / old).mkdir(parents=True)
    (sweep_root / new).mkdir(parents=True)
    write_sweep_latest(tmp_path, "bench", "s1", new)

    async def fake_index_epochs(namespace: str, sweep_name: str) -> list[str]:
        return [old]

    monkeypatch.setattr(
        "aiperf.operator.runs_index.list_sweep_epochs_for_sweep", fake_index_epochs
    )

    epochs = await list_sweep_epochs_async(tmp_path, "bench", "s1")

    assert [entry.epoch for entry in epochs] == [old, new]
    assert {entry.epoch for entry in epochs if entry.is_latest} == {new}
