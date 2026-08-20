# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for runs_index.py.

Companion to test_runs_index.py — focuses on pathological inputs and rarely-
exercised paths the happy-path suite doesn't cover:

- Corruption recovery posture (rebuild-after-stomp, mid-write SQLite errors)
- Schema-version forward-incompatibility guard (only a guard exists; no
  migration ladder is in scope today, so we verify the guard, not migrations)
- Bootstrap tolerance to malformed filesystems (bad JSON, missing fields,
  unmatched epoch dirs, empty namespaces, mid-walk ENOENT)
- Concurrent reader-during-write semantics under WAL
- Malformed metrics_json blob (read path)
- Time-travel / out-of-order epoch updates and idempotent re-completion
- list_all_latest stale-row tolerance
- leaderboard / compare on empty / tied / missing-metric inputs
- _index_sweep_from_disk with malformed / partial aggregate.json
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import aiosqlite
import orjson
import pytest
import zstandard

from aiperf.kubernetes.results_sidecar import write_ready_marker
from aiperf.operator import runs_index

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
async def index_path(tmp_path: Path) -> Path:
    """Open a fresh runs_index DB rooted at tmp_path; close on teardown.

    Mirrors the fixture in test_runs_index.py so helpers stay interchangeable.
    """
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    yield path
    await runs_index.close()


def _zstd(payload: dict[str, Any]) -> bytes:
    return zstandard.ZstdCompressor().compress(orjson.dumps(payload))


# ============================================================
# Corruption Recovery
# ============================================================


class TestCorruptionRecovery:
    """Behavior when the on-disk SQLite file is damaged or the writer faults.

    runs_index.py does NOT auto-quarantine/rebuild — that policy lives in the
    caller (operator startup). What it MUST guarantee: detection is honest,
    and a force-rebuild bootstrap fully recovers a clobbered DB without
    leaving stale rows.
    """

    @pytest.mark.asyncio
    async def test_force_rebuild_after_corruption_clears_stale_rows(
        self, tmp_path: Path
    ) -> None:
        """Caller-driven recovery: integrity_check fails -> reopen + bootstrap(force=True)."""
        path = tmp_path / ".aiperf_index.sqlite"
        await runs_index.open(path)
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.set_latest("ns", "j", "100")
        rows_before = await runs_index.list_all_latest()
        assert len(rows_before) == 1
        await runs_index.close()

        # Verify integrity_check honestly reports breakage
        path.write_bytes(b"not a sqlite db")
        assert await runs_index.integrity_check(path) is False

        # Operator's recovery path: delete the file, reopen, bootstrap from disk.
        # The PVC has no run dirs, so the rebuild produces an empty index —
        # callers fall back to disk scans on miss (per module docstring contract).
        path.unlink()
        await runs_index.open(path)
        try:
            stats = await runs_index.bootstrap(tmp_path / "results", force=True)
            assert stats.runs_indexed == 0
            rows_after = await runs_index.list_all_latest()
            assert rows_after == [], (
                "stale pre-corruption rows must not survive rebuild"
            )
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_force_rebuild_drops_existing_tables(self, index_path: Path) -> None:
        """bootstrap(force=True) must wipe runs + sweep_variations before re-walking.

        Why: after corruption-or-drift, the operator triggers a force rebuild;
        rows that no longer exist on disk must not survive the rebuild.
        """
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_sweep_variation(
            "ns",
            "s",
            "100",
            0,
            variation_values={"c": 1},
            mode="INDEPENDENT",
            phase="Succeeded",
            metrics={},
            child_ref=None,
            metrics_blob=b"",
        )

        # base dir doesn't exist → bootstrap returns immediately, but force=True
        # path runs the DELETEs first.
        await runs_index.bootstrap(index_path.parent / "no_such_dir", force=True)

        # Both tables must be empty
        assert await runs_index.list_runs_for_job("ns", "j") == []
        assert await runs_index.list_sweep_variations("ns", "s", "100") == []

    @pytest.mark.asyncio
    async def test_mid_write_sqlite_error_does_not_leave_partial_latest(
        self, index_path: Path
    ) -> None:
        """If the inner UPDATE in set_latest faults, ROLLBACK must restore prior state.

        set_latest wraps two UPDATEs in BEGIN IMMEDIATE/COMMIT. If the second
        UPDATE raises, the prior is_latest pointer must still be valid (no
        dual-latest, no zero-latest).
        """
        for ep in ("100", "200", "300"):
            await runs_index.upsert_run_created("ns", "j", ep, spec={})
        await runs_index.set_latest("ns", "j", "200")

        # Patch the connection so the *second* statement in set_latest's
        # transaction explodes. set_latest issues: BEGIN, UPDATE clear, UPDATE set,
        # COMMIT. Fail on the second UPDATE.
        db = runs_index._conn()
        original_execute = db.execute
        call_count = {"n": 0}

        async def flaky_execute(sql: str, *args: Any, **kw: Any) -> Any:
            call_count["n"] += 1
            # 1: BEGIN, 2: UPDATE clear, 3: UPDATE set <-- raise
            if call_count["n"] == 3:
                raise aiosqlite.OperationalError("disk I/O error (simulated)")
            return await original_execute(sql, *args, **kw)

        with (
            patch.object(db, "execute", side_effect=flaky_execute),
            pytest.raises(aiosqlite.OperationalError, match="disk I/O error"),
        ):
            await runs_index.set_latest("ns", "j", "300")

        # Post-rollback: still exactly one is_latest row, and it's the pre-failure 200.
        rows = await runs_index.list_runs_for_job("ns", "j")
        latest = [r for r in rows if r.is_latest]
        assert len(latest) == 1
        assert latest[0].epoch == "200"


# ============================================================
# Schema Migration Guard
# ============================================================


class TestSchemaMigrationGuard:
    """Verify the forward-incompatibility guard in open().

    runs_index.py carries SCHEMA_VERSION=1 and does not yet have a migration
    ladder. The behavior to lock down today:
    - Opening an existing v1 DB is idempotent (no version-row dup).
    - Opening a DB stamped with a higher version raises a clear RuntimeError.
    """

    @pytest.mark.asyncio
    async def test_higher_schema_version_refuses_to_open(self, tmp_path: Path) -> None:
        path = tmp_path / ".aiperf_index.sqlite"
        await runs_index.open(path)
        # Forge a future schema_version stamp
        await runs_index.set_meta("schema_version", "999")
        await runs_index.close()

        with pytest.raises(RuntimeError, match="schema_version=999"):
            await runs_index.open(path)
        # open() raised before assigning to module state, so close is a noop
        await runs_index.close()

    @pytest.mark.asyncio
    async def test_reopen_does_not_duplicate_meta_row(self, tmp_path: Path) -> None:
        """Re-opening a v1 DB must not insert a second schema_version meta row."""
        path = tmp_path / ".aiperf_index.sqlite"
        await runs_index.open(path)
        await runs_index.close()
        await runs_index.open(path)
        try:
            cur = await runs_index._conn().execute(
                "SELECT COUNT(*) FROM meta WHERE key = 'schema_version'"
            )
            assert (await cur.fetchone())[0] == 1
            await cur.close()
        finally:
            await runs_index.close()


# ============================================================
# Bootstrap Edge Cases
# ============================================================


class TestBootstrapEdgeCases:
    """Bootstrap must tolerate junk on the PVC — never abort mid-walk.

    Contract: stray files, malformed JSON, missing summary, unmatched epoch
    dirnames all degrade to "skip + warn", never raise.
    """

    @pytest.mark.asyncio
    async def test_bootstrap_empty_base_returns_zero_counts(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        base = tmp_path / "results"
        base.mkdir()
        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 0
        assert stats.sweep_variations_indexed == 0

    @pytest.mark.asyncio
    async def test_bootstrap_nonexistent_base_returns_zero_counts(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        stats = await runs_index.bootstrap(tmp_path / "does_not_exist")
        assert stats.runs_indexed == 0

    @pytest.mark.asyncio
    async def test_bootstrap_skips_stray_files_in_namespace_dir(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Stray files at <ns>/<file> level must not crash the walk.

        Why: PVC sometimes accumulates kubelet-managed files / partial uploads
        at the namespace level; treating them as job dirs would explode.
        """
        base = tmp_path / "results"
        ns = base / "ns1"
        ns.mkdir(parents=True)
        (ns / "stray.txt").write_text("hello")
        # A real run alongside the stray file
        run = ns / "job-a" / "1714069323"
        run.mkdir(parents=True)
        (run / "profile_export_aiperf.json").write_bytes(orjson.dumps({}))
        write_ready_marker(run)
        (ns / "job-a" / "latest.txt").write_text("1714069323")

        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 1

    @pytest.mark.asyncio
    async def test_bootstrap_skips_unmatched_epoch_dirnames(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Epoch dirs not matching EPOCH_RE (^\\d{9,11}$) must be ignored."""
        base = tmp_path / "results"
        # "weird" doesn't match EPOCH_RE, "1714069323" does
        (base / "ns" / "job" / "weird").mkdir(parents=True)
        (base / "ns" / "job" / "weird" / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({})
        )
        (base / "ns" / "job" / "1714069323").mkdir(parents=True)
        (base / "ns" / "job" / "1714069323" / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({})
        )
        write_ready_marker(base / "ns" / "job" / "1714069323")
        (base / "ns" / "job" / "latest.txt").write_text("1714069323")

        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 1
        rows = await runs_index.list_runs_for_job("ns", "job")
        assert [r.epoch for r in rows] == ["1714069323"]

    @pytest.mark.asyncio
    async def test_bootstrap_tolerates_corrupted_summary_json(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """A malformed profile_export_aiperf.json must skip + warn, not raise."""
        base = tmp_path / "results"
        run = base / "ns" / "j" / "1714069323"
        run.mkdir(parents=True)
        (run / "profile_export_aiperf.json").write_bytes(b"{not valid json}")
        write_ready_marker(run)
        (base / "ns" / "j" / "latest.txt").write_text("1714069323")

        # Real run with valid summary alongside, to prove the walk continues
        run2 = base / "ns" / "j2" / "1714069324"
        run2.mkdir(parents=True)
        (run2 / "profile_export_aiperf.json").write_bytes(orjson.dumps({}))
        write_ready_marker(run2)
        (base / "ns" / "j2" / "latest.txt").write_text("1714069324")

        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 1, "j2 must be indexed; j must be skipped"
        rows = await runs_index.list_all_latest()
        assert {r.job_id for r in rows} == {"j2"}

    @pytest.mark.asyncio
    async def test_bootstrap_tolerates_corrupted_zst_summary(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """A non-zstd .json.zst blob must be skipped, not crash bootstrap."""
        base = tmp_path / "results"
        run = base / "ns" / "j" / "1714069323"
        run.mkdir(parents=True)
        (run / "profile_export_aiperf.json.zst").write_bytes(b"definitely not zstd")
        (base / "ns" / "j" / "latest.txt").write_text("1714069323")

        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 0

    @pytest.mark.asyncio
    async def test_bootstrap_skips_run_dir_without_summary(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Epoch dir present but no profile_export_aiperf.json → skip silently."""
        base = tmp_path / "results"
        run = base / "ns" / "j" / "1714069323"
        run.mkdir(parents=True)
        # only ready marker, no summary — mid-flight or aborted run
        (run / ".aiperf_results_ready.json").write_text("{}")
        (base / "ns" / "j" / "latest.txt").write_text("1714069323")

        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 0

    @pytest.mark.asyncio
    async def test_bootstrap_skips_malformed_aggregate_json(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """A malformed sweep aggregate.json must be skipped."""
        base = tmp_path / "results"
        epoch_dir = base / "ns" / "sweeps" / "satsweep" / "1714069323"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(b"{not json")

        stats = await runs_index.bootstrap(base)
        assert stats.sweep_variations_indexed == 0
        assert (
            await runs_index.list_sweep_variations("ns", "satsweep", "1714069323") == []
        )

    @pytest.mark.asyncio
    async def test_bootstrap_continues_after_one_run_raises(
        self, tmp_path: Path, index_path: Path, monkeypatch
    ) -> None:
        """One bad run dir (e.g. a `KeyError` from malformed metrics, or an
        unexpected `OSError` from a disk hiccup mid-walk) must NOT abort the
        whole bootstrap — the other runs MUST still get indexed. Without the
        per-iteration try/except, a single corrupt run dir hides every
        subsequent run from the API until the next bootstrap tick.
        """
        base = tmp_path / "results"
        # Three job dirs in the same namespace, each with a stub epoch.
        for job in ("job-good-1", "job-bad", "job-good-2"):
            (base / "ns" / job / "1714069323").mkdir(parents=True)

        # Wrap the real indexer so the middle job raises.
        real_indexer = runs_index._index_run_from_disk
        seen: list[str] = []

        async def _selective_failure(base_, ns, job, epoch, *, is_latest):
            seen.append(job)
            if job == "job-bad":
                raise RuntimeError("simulated mid-walk indexer error")
            return await real_indexer(base_, ns, job, epoch, is_latest=is_latest)

        monkeypatch.setattr(runs_index, "_index_run_from_disk", _selective_failure)

        stats = await runs_index.bootstrap(base)
        # All three were ATTEMPTED — bootstrap didn't abort early.
        assert sorted(seen) == ["job-bad", "job-good-1", "job-good-2"]
        # `_index_run_from_disk` returns False for empty stub dirs (no summary
        # file), so neither good dir actually indexes. The contract under test
        # is the no-abort invariant — verified by `seen` containing all three.
        assert stats.runs_indexed >= 0  # never raises

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_running_no_aggregate(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Sweep dir with no aggregate.json yet (in-flight sweep) → returns False."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)
        assert ok == 0

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_aggregate_export_without_children_indexes_zero(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """AggregateConfidenceJsonExporter output alone has no per-variation rows."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        aggregate_dir = epoch_dir / "sweep_aggregate"
        aggregate_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps(
                {
                    "phase": "Succeeded",
                    "totalVariations": 1,
                    "completedRuns": 1,
                    "failedRuns": 0,
                }
            )
        )
        (aggregate_dir / "profile_export_aiperf_aggregate.json").write_bytes(
            orjson.dumps(
                {
                    "schema_version": "1.0",
                    "metadata": {
                        "aggregation_type": "confidence",
                        "num_profile_runs": 1,
                        "num_successful_runs": 1,
                        "failed_runs": [],
                    },
                    "metrics": {
                        "request_throughput": {
                            "mean": 50.0,
                            "std": 0.0,
                            "min": 50.0,
                            "max": 50.0,
                            "cv": 0.0,
                            "se": 0.0,
                            "ci_low": 50.0,
                            "ci_high": 50.0,
                            "t_critical": 0.0,
                            "unit": "rps",
                        }
                    },
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)
        assert ok == 0
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert rows == []

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_prefers_sweep_export_when_both_exist(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Valid sweep export rows beat confidence aggregate metadata."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        aggregate_dir = epoch_dir / "sweep_aggregate"
        aggregate_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 1})
        )
        (aggregate_dir / "profile_export_aiperf_aggregate.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"aggregation_type": "confidence"},
                    "metrics": {
                        "request_throughput": {"mean": 50.0, "unit": "rps"},
                    },
                }
            )
        )
        (aggregate_dir / "profile_export_aiperf_sweep.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"mode": "INDEPENDENT"},
                    "per_combination_metrics": [
                        {
                            "variation_idx": 0,
                            "variation_values": {"concurrency": 10},
                            "metrics": {
                                "request_throughput": {
                                    "avg": 75.0,
                                    "p50": 70.0,
                                    "p99": 90.0,
                                    "unit": "rps",
                                }
                            },
                        }
                    ],
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 1
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert len(rows) == 1
        assert rows[0].variation_idx == 0
        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "s", 0),
        )
        assert (await cur.fetchone())[0] == 75.0
        await cur.close()

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_prefers_sweep_export_over_parent_rows(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Valid sweep-export rows beat parent aggregate per-combination rows."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        aggregate_dir = epoch_dir / "sweep_aggregate"
        aggregate_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"mode": "INDEPENDENT"},
                    "per_combination_metrics": [
                        {
                            "variation_idx": 0,
                            "variation_values": {"concurrency": 10},
                            "metrics": {
                                "request_throughput": {"avg": 11.0, "unit": "rps"}
                            },
                        }
                    ],
                }
            )
        )
        (aggregate_dir / "profile_export_aiperf_sweep.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"mode": "INDEPENDENT"},
                    "per_combination_metrics": [
                        {
                            "variation_idx": 0,
                            "variation_values": {"concurrency": 10},
                            "metrics": {
                                "request_throughput": {"avg": 88.0, "unit": "rps"}
                            },
                        }
                    ],
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 1
        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "s", 0),
        )
        assert (await cur.fetchone())[0] == 88.0
        await cur.close()

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_degrades_non_dict_row_metrics(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """A row with metrics='bad' does not abort valid sweep rows."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        aggregate_dir = epoch_dir / "sweep_aggregate"
        aggregate_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 2})
        )
        (aggregate_dir / "profile_export_aiperf_sweep.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"mode": "INDEPENDENT"},
                    "per_combination_metrics": [
                        {
                            "variation_idx": 0,
                            "variation_values": {"concurrency": 10},
                            "metrics": "bad",
                        },
                        {
                            "variation_idx": 1,
                            "variation_values": {"concurrency": 20},
                            "metrics": {
                                "request_throughput": {"avg": 22.0, "unit": "rps"}
                            },
                        },
                    ],
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 2
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert [r.variation_idx for r in rows] == [0, 1]
        cur = await runs_index._conn().execute(
            "SELECT variation_idx, request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? ORDER BY variation_idx",
            ("ns", "s"),
        )
        assert await cur.fetchall() == [(0, None), (1, 22.0)]
        await cur.close()

    @pytest.mark.parametrize(
        "children_doc", [[{"not": "a mapping envelope"}], "scalar"]
    )
    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_skips_nondict_children_json_top_level(
        self, tmp_path: Path, index_path: Path, children_doc: Any
    ) -> None:
        """children.json must be a mapping; list/scalar top-level shapes are ignored."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 1})
        )
        (epoch_dir / "children.json").write_bytes(orjson.dumps(children_doc))

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 0
        assert await runs_index.list_sweep_variations("ns", "s", "1714069323") == []

    @pytest.mark.parametrize(
        "children_value",
        [
            {"variation_index": 0, "name": "s-v00", "child_run_epoch": "1714069324"},
            "not-a-list",
            42,
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_requires_children_list(
        self, tmp_path: Path, index_path: Path, children_value: Any
    ) -> None:
        """children.json.children must be a list; other shapes degrade to zero rows."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 1})
        )
        (epoch_dir / "children.json").write_bytes(
            orjson.dumps({"children": children_value})
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 0
        assert await runs_index.list_sweep_variations("ns", "s", "1714069323") == []

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_skips_nondict_child_entries(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Non-mapping entries in children.json.children are skipped, not indexed."""
        base = tmp_path
        epoch_dir = base / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        child_run = base / "ns" / "s-v00" / "1714069324"
        child_run.mkdir(parents=True)
        (child_run / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(
                {
                    "metrics": {
                        "request_throughput": {
                            "avg": 11.0,
                            "p50": 10.0,
                            "p99": 12.0,
                            "unit": "rps",
                        }
                    }
                }
            )
        )
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 1})
        )
        (epoch_dir / "children.json").write_bytes(
            orjson.dumps(
                {
                    "children": [
                        "junk",
                        123,
                        ["not", "a", "mapping"],
                        {
                            "namespace": "ns",
                            "name": "s-v00",
                            "variation_index": 0,
                            "variation_label": "concurrency=10",
                            "child_run_epoch": "1714069324",
                        },
                    ]
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 1
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert len(rows) == 1
        assert rows[0].variation_idx == 0
        assert rows[0].child_job_id == "s-v00"
        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "s", 0),
        )
        assert (await cur.fetchone())[0] == 11.0
        await cur.close()

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_skips_one_bad_row_indexes_rest(
        self, tmp_path: Path, index_path: Path, monkeypatch
    ) -> None:
        """One row failing upsert (sqlite constraint, unencodable value, etc.)
        must not poison the loop — the other variations still get indexed.
        Without the per-iteration try/except, a single bad row aborts the
        whole sweep ingest and 'sweep summary endpoints return zero rows'
        until the next backfill tick.
        """
        epoch_dir = tmp_path / "ns" / "sweeps" / "sweep-x" / "1714069323"
        epoch_dir.mkdir(parents=True)
        agg = {
            "metadata": {"mode": "INDEPENDENT"},
            "per_combination_metrics": [
                {
                    "variation_idx": 0,
                    "variation_values": {"c": 10},
                    "metrics": {"request_throughput": {"avg": 1.0, "unit": "rps"}},
                },
                {
                    "variation_idx": 1,
                    "variation_values": {"c": 50},
                    "metrics": {"request_throughput": {"avg": 5.0, "unit": "rps"}},
                },
                {
                    "variation_idx": 2,
                    "variation_values": {"c": 100},
                    "metrics": {"request_throughput": {"avg": 9.0, "unit": "rps"}},
                },
            ],
        }
        (epoch_dir / "aggregate.json").write_bytes(orjson.dumps(agg))

        # Force idx=1 to raise sqlite3.IntegrityError on upsert; idx=0 and
        # idx=2 must still index. Wraps the real upsert so the surviving
        # rows actually land in the DB.
        real_upsert = runs_index.upsert_sweep_variation

        async def _failing_upsert(ns, sweep_name, sweep_epoch, variation_idx, **kw):
            if variation_idx == 1:
                raise sqlite3.IntegrityError(
                    "simulated UNIQUE constraint failed on (ns, sweep, epoch, idx)"
                )
            return await real_upsert(ns, sweep_name, sweep_epoch, variation_idx, **kw)

        monkeypatch.setattr(runs_index, "upsert_sweep_variation", _failing_upsert)

        ok = await runs_index._index_sweep_from_disk(
            "ns", "sweep-x", "1714069323", epoch_dir
        )

        assert ok == 2, "two variations succeeded → indexed"
        rows = await runs_index.list_sweep_variations("ns", "sweep-x", "1714069323")
        indices = sorted(r.variation_idx for r in rows)
        assert indices == [0, 2], (
            f"idx=1 raised → must be skipped; idx=0 and idx=2 must persist. got={indices}"
        )

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_skips_scalar_combination_rows(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """Malformed scalar per_combination_metrics entries are skipped."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        aggregate_dir = epoch_dir / "sweep_aggregate"
        aggregate_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"phase": "Succeeded", "completedRuns": 1})
        )
        (aggregate_dir / "profile_export_aiperf_sweep.json").write_bytes(
            orjson.dumps(
                {
                    "metadata": {"mode": "INDEPENDENT"},
                    "per_combination_metrics": [
                        "not-a-row",
                        ["also", "not", "a", "row"],
                        {
                            "variation_idx": 3,
                            "variation_values": {"concurrency": 30},
                            "metrics": {
                                "request_throughput": {
                                    "avg": 33.0,
                                    "p50": 30.0,
                                    "p99": 36.0,
                                    "unit": "rps",
                                }
                            },
                        },
                    ],
                }
            )
        )

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)

        assert ok == 1
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert [r.variation_idx for r in rows] == [3]
        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "s", 3),
        )
        assert (await cur.fetchone())[0] == 33.0
        await cur.close()

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_skips_variations_missing_idx(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """per_combination_metrics entries without variation_idx are skipped."""
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        agg = {
            "metadata": {"mode": "INDEPENDENT"},
            "per_combination_metrics": [
                {"variation_values": {"c": 10}, "metrics": {}},  # no variation_idx
                {"variation_idx": 7, "variation_values": {"c": 20}, "metrics": {}},
            ],
        }
        (epoch_dir / "aggregate.json").write_bytes(orjson.dumps(agg))

        await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert [r.variation_idx for r in rows] == [7]

    @pytest.mark.asyncio
    async def test_index_sweep_from_disk_with_sla_metadata_passthrough(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """SLA-filter metadata in aggregate.json is non-fatal — variations index normally.

        runs_index doesn't materialize SLA filter info; it just must not choke.
        """
        epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714069323"
        epoch_dir.mkdir(parents=True)
        agg = {
            "metadata": {
                "mode": "INDEPENDENT",
                "sla_filters": {"request_latency_p99": {"max": 0.5}},
            },
            "per_combination_metrics": [
                {"variation_idx": 0, "variation_values": {"c": 1}, "metrics": {}},
            ],
            "best_configurations": [{"variation_idx": 0}],
        }
        (epoch_dir / "aggregate.json").write_bytes(orjson.dumps(agg))

        ok = await runs_index._index_sweep_from_disk("ns", "s", "1714069323", epoch_dir)
        assert ok == 1
        rows = await runs_index.list_sweep_variations("ns", "s", "1714069323")
        assert len(rows) == 1
        assert rows[0].is_best is True


# ============================================================
# Concurrent Reader-During-Write
# ============================================================


class TestConcurrentReaders:
    """WAL-mode runs_index must let read-only connections proceed during writes."""

    @pytest.mark.asyncio
    async def test_open_readonly_uses_shared_cache_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production readonly opener must opt into SQLite shared cache."""
        await runs_index.close()
        captured: dict[str, Any] = {}

        class FakeCursor:
            async def fetchone(self) -> tuple[str]:
                return (str(runs_index.SCHEMA_VERSION),)

            async def close(self) -> None:
                return None

        class FakeDB:
            async def execute(self, sql: str) -> FakeCursor:
                return FakeCursor()

            async def close(self) -> None:
                return None

        async def fake_connect(database_uri: str, **kwargs: Any) -> FakeDB:
            captured["uri"] = database_uri
            captured["kwargs"] = kwargs
            return FakeDB()

        monkeypatch.setattr(aiosqlite, "connect", fake_connect)

        await runs_index.open_readonly(tmp_path / ".aiperf_index.sqlite")
        try:
            assert "?mode=ro&cache=shared" in captured["uri"]
            assert captured["kwargs"]["uri"] is True
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_reader_ro_connection_sees_committed_state(
        self, index_path: Path
    ) -> None:
        """A separate ro connection sees rows committed by the writer."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.set_latest("ns", "j", "100")

        # Open as the production code does: mode=ro&cache=shared via URI
        uri = f"file:{index_path}?mode=ro&cache=shared"
        async with aiosqlite.connect(uri, uri=True) as ro:
            cur = await ro.execute(
                "SELECT epoch, is_latest FROM runs WHERE namespace = ? AND job_id = ?",
                ("ns", "j"),
            )
            rows = await cur.fetchall()
            await cur.close()
        assert rows == [("100", 1)]

    @pytest.mark.asyncio
    async def test_multiple_concurrent_readers_do_not_block(
        self, index_path: Path
    ) -> None:
        """Two ro connections can fetch concurrently without blocking each other."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})

        async def read_once() -> int:
            uri = f"file:{index_path}?mode=ro&cache=shared"
            async with aiosqlite.connect(uri, uri=True) as ro:
                cur = await ro.execute("SELECT COUNT(*) FROM runs")
                row = await cur.fetchone()
                await cur.close()
            return row[0]

        results = await asyncio.gather(*[read_once() for _ in range(5)])
        assert results == [1] * 5


# ============================================================
# Malformed Metrics Payloads
# ============================================================


class TestMalformedMetrics:
    """Malformed metrics shouldn't crash either the upsert or downstream reads."""

    @pytest.mark.asyncio
    async def test_upsert_completed_with_empty_metrics_leaves_narrow_columns_null(
        self, index_path: Path
    ) -> None:
        """metrics={} → all 24 narrow columns NULL, but the row is created."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={},
            files=[],
            mtime_epoch=100,
        )

        narrow = await runs_index.get_run_narrow_metrics("ns", "j", "100")
        assert narrow is not None
        for name in runs_index._NARROW_METRICS:
            for stat in ("avg", "p50", "p99"):
                assert narrow[f"{name}_{stat}"] is None

    @pytest.mark.asyncio
    async def test_upsert_completed_with_partial_metrics_other_cols_null(
        self, index_path: Path
    ) -> None:
        """Only request_throughput populated → the other 5 metrics' columns are NULL."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={
                "metrics": {
                    "request_throughput": {
                        "avg": 1.0,
                        "p50": 1.0,
                        "p99": 1.0,
                        "unit": "rps",
                    },
                },
            },
            files=[],
            mtime_epoch=100,
        )
        narrow = await runs_index.get_run_narrow_metrics("ns", "j", "100")
        assert narrow["request_throughput_avg"] == 1.0
        assert narrow["request_latency_avg"] is None
        assert narrow["time_to_first_token_p99"] is None

    @pytest.mark.asyncio
    async def test_get_summary_blob_returns_none_when_blob_is_empty(
        self, index_path: Path
    ) -> None:
        """upsert with summary_blob=b'' must read back as None (treated as absent).

        Why: the read-side helper coerces empty bytes to None so router code
        can fall back to disk; if it returned b'' callers would try to zstd-
        decompress garbage.
        """
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={},
            files=[],
            mtime_epoch=100,
        )
        assert await runs_index.get_summary_blob("ns", "j", "100") is None

    @pytest.mark.asyncio
    async def test_get_run_spec_returns_none_when_spec_json_null(
        self, index_path: Path
    ) -> None:
        """A row created via upsert_run_phase has no spec_json yet → returns None."""
        await runs_index.upsert_run_phase("ns", "j", "100", phase="Running")
        assert await runs_index.get_run_spec("ns", "j", "100") is None

    @pytest.mark.asyncio
    async def test_zstd_decompress_invalid_blob_raises_zstderror(self) -> None:
        """zstd_decompress on garbage raises ZstdError (callers must guard).

        Why: documents the contract — the read path callers (e.g. job_union)
        must catch ZstdError and fall back to disk. If runs_index ever swallows
        this internally, that would silently mask corruption.
        """
        with pytest.raises(zstandard.ZstdError):
            runs_index.zstd_decompress(b"not a zstd frame at all")

    @pytest.mark.asyncio
    async def test_telemetry_data_with_non_dict_endpoints_is_safe(
        self, index_path: Path
    ) -> None:
        """telemetry_data.endpoints not a dict → gpu_count=0, gpu_name=None."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={"telemetry_data": {"endpoints": "garbage"}},
            files=[],
            mtime_epoch=100,
        )
        row = await runs_index.get_run("ns", "j", "100")
        assert row.gpu_count == 0
        assert row.gpu_name is None


# ============================================================
# Idempotency / Out-of-order Events
# ============================================================


class TestOutOfOrderEvents:
    """Two epochs for the same (ns, job) and idempotent re-completion."""

    @pytest.mark.asyncio
    async def test_two_epochs_same_job_each_have_own_row(
        self, index_path: Path
    ) -> None:
        """Different epochs → distinct rows; set_latest only flips one of them."""
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_created("ns", "j", "200", spec={})
        await runs_index.set_latest("ns", "j", "200")

        rows = await runs_index.list_runs_for_job("ns", "j")
        assert len(rows) == 2
        latest = [r for r in rows if r.is_latest]
        assert len(latest) == 1
        assert latest[0].epoch == "200"
        # And flipping back must keep cardinality
        await runs_index.set_latest("ns", "j", "100")
        rows = await runs_index.list_runs_for_job("ns", "j")
        latest = [r for r in rows if r.is_latest]
        assert len(latest) == 1
        assert latest[0].epoch == "100"

    @pytest.mark.asyncio
    async def test_create_after_completed_does_not_overwrite_metrics(
        self, index_path: Path
    ) -> None:
        """An out-of-order create event after completion must NOT clobber metrics.

        This is the COALESCE behavior in upsert_run_created — model/endpoint/
        spec_json are preserved when the row already has them.
        """
        await runs_index.upsert_run_created(
            "ns",
            "j",
            "100",
            spec={
                "benchmark": {
                    "models": {"items": [{"name": "real-model"}]},
                    "endpoint": {"urls": ["http://real:8000"]},
                }
            },
        )
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={
                "metrics": {
                    "request_throughput": {
                        "avg": 99.0,
                        "p50": 99.0,
                        "p99": 99.0,
                        "unit": "rps",
                    }
                }
            },
            files=[],
            mtime_epoch=100,
        )
        # Out-of-order replay of create with empty spec
        await runs_index.upsert_run_created("ns", "j", "100", spec={})

        row = await runs_index.get_run("ns", "j", "100")
        # Metrics survive
        assert row.phase == "Succeeded"
        assert row.model == "real-model"
        assert row.endpoint == "http://real:8000"
        narrow = await runs_index.get_run_narrow_metrics("ns", "j", "100")
        assert narrow["request_throughput_avg"] == 99.0

    @pytest.mark.asyncio
    async def test_recompletion_is_last_write_wins(self, index_path: Path) -> None:
        """Re-completing an already-completed epoch overwrites with new metrics.

        Documents the contract: ``upsert_run_completed`` on an existing row
        is last-wins (DO UPDATE SET excluded.*), not idempotent-no-op. This
        matches operator behavior where a retry pushes fresher metrics.
        """
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        for tput in (10.0, 20.0, 30.0):
            await runs_index.upsert_run_completed(
                "ns",
                "j",
                "100",
                summary_blob=b"",
                metrics={
                    "metrics": {
                        "request_throughput": {
                            "avg": tput,
                            "p50": tput,
                            "p99": tput,
                            "unit": "rps",
                        }
                    }
                },
                files=[],
                mtime_epoch=int(tput),
            )

        narrow = await runs_index.get_run_narrow_metrics("ns", "j", "100")
        assert narrow["request_throughput_avg"] == 30.0
        rows = await runs_index.list_runs_for_job("ns", "j")
        assert len(rows) == 1, "no duplicate row from repeated completion"
        assert rows[0].mtime_epoch == 30


# ============================================================
# Stale Index Entries
# ============================================================


class TestStaleIndexEntries:
    """list_all_latest is a cache-only query — it does NOT validate disk presence."""

    @pytest.mark.asyncio
    async def test_list_all_latest_returns_rows_even_when_disk_dir_missing(
        self, index_path: Path
    ) -> None:
        """A row whose run dir was deleted on disk still appears in list_all_latest.

        Per the module docstring: "the index is a cache, never a source of
        truth." The stale-entry filter happens at the read site (router), not
        in list_all_latest itself. Lock that contract in.
        """
        await runs_index.upsert_run_created("ns", "j", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            "100",
            summary_blob=b"",
            metrics={},
            files=["a.json"],
            mtime_epoch=100,
        )
        await runs_index.set_latest("ns", "j", "100")

        # No disk dir was ever created — index entry is "stale" by definition.
        rows = await runs_index.list_all_latest()
        assert len(rows) == 1
        assert rows[0].job_id == "j"


# ============================================================
# Leaderboard / Compare Edge Cases
# ============================================================


class TestLeaderboardCompareEdges:
    """leaderboard / compare on empty / tied / missing-metric inputs."""

    @pytest.mark.asyncio
    async def test_leaderboard_empty_db_returns_empty_list(
        self, index_path: Path
    ) -> None:
        rows = await runs_index.leaderboard(metric="request_throughput", stat="avg")
        assert rows == []

    @pytest.mark.asyncio
    async def test_leaderboard_skips_rows_with_null_metric(
        self, index_path: Path
    ) -> None:
        """Rows whose target metric is NULL must not appear in the leaderboard."""
        await runs_index.upsert_run_created("ns", "j1", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j1",
            "100",
            summary_blob=b"",
            metrics={},  # all narrow cols NULL
            files=[],
            mtime_epoch=100,
        )
        await runs_index.set_latest("ns", "j1", "100")

        await runs_index.upsert_run_created("ns", "j2", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j2",
            "100",
            summary_blob=b"",
            metrics={
                "metrics": {
                    "request_throughput": {
                        "avg": 5.0,
                        "p50": 5.0,
                        "p99": 5.0,
                        "unit": "rps",
                    }
                }
            },
            files=[],
            mtime_epoch=100,
        )
        await runs_index.set_latest("ns", "j2", "100")

        rows = await runs_index.leaderboard(metric="request_throughput", stat="avg")
        assert [r["job_id"] for r in rows] == ["j2"]

    @pytest.mark.asyncio
    async def test_leaderboard_no_such_column_returns_empty(
        self, index_path: Path
    ) -> None:
        """Unknown but identifier-valid metric → SQLite errors → empty list.

        Why: matches the legacy DuckDB read-path contract — analytics callers
        rely on empty list (not 500) when the metric column doesn't exist.
        """
        rows = await runs_index.leaderboard(metric="not_a_real_metric", stat="avg")
        assert rows == []

    @pytest.mark.parametrize(
        "bad_metric",
        [
            "DROP TABLE runs;--",
            "request_throughput; SELECT *",
            "request-throughput",
            "request throughput",
            "",
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_leaderboard_rejects_invalid_identifier(
        self, index_path: Path, bad_metric: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await runs_index.leaderboard(metric=bad_metric)

    @pytest.mark.asyncio
    async def test_compare_empty_job_ids_returns_empty(self, index_path: Path) -> None:
        assert await runs_index.compare([]) == []

    @pytest.mark.asyncio
    async def test_compare_returns_partial_metrics_when_run_missing_some(
        self, index_path: Path
    ) -> None:
        """If a job has only some metrics, compare returns those columns + NULL elsewhere."""
        await runs_index.upsert_run_created("ns", "j1", "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            "j1",
            "100",
            summary_blob=b"",
            metrics={
                "metrics": {
                    "request_throughput": {
                        "avg": 50.0,
                        "p50": 50.0,
                        "p99": 50.0,
                        "unit": "rps",
                    }
                }
            },
            files=[],
            mtime_epoch=100,
        )
        await runs_index.set_latest("ns", "j1", "100")

        rows = await runs_index.compare(
            ["j1"], metrics=["request_throughput", "request_latency"]
        )
        assert len(rows) == 1
        assert rows[0]["request_throughput_avg"] == 50.0
        assert rows[0]["request_latency_avg"] is None
        assert rows[0]["request_latency_unit"] is None

    @pytest.mark.asyncio
    async def test_compare_with_unknown_job_ids_returns_empty(
        self, index_path: Path
    ) -> None:
        rows = await runs_index.compare(["nonexistent-job"])
        assert rows == []

    @pytest.mark.asyncio
    async def test_leaderboard_tied_metrics_returns_all_rows(
        self, index_path: Path
    ) -> None:
        """Multiple rows with identical values both surface — ordering is stable."""
        for j, ep in [("j1", "100"), ("j2", "200"), ("j3", "300")]:
            await runs_index.upsert_run_created("ns", j, ep, spec={})
            await runs_index.upsert_run_completed(
                "ns",
                j,
                ep,
                summary_blob=b"",
                metrics={
                    "metrics": {
                        "request_throughput": {
                            "avg": 7.0,
                            "p50": 7.0,
                            "p99": 7.0,
                            "unit": "rps",
                        }
                    }
                },
                files=[],
                mtime_epoch=int(ep),
            )
            await runs_index.set_latest("ns", j, ep)

        rows = await runs_index.leaderboard(
            metric="request_throughput", stat="avg", limit=10
        )
        assert len(rows) == 3
        assert all(r["value"] == 7.0 for r in rows)


# ============================================================
# Module-state guards
# ============================================================


class TestModuleStateGuards:
    """Helpers must fail loudly when the DB is closed."""

    @pytest.mark.asyncio
    async def test_conn_raises_when_db_not_open(self) -> None:
        # ensure clean state
        await runs_index.close()
        with pytest.raises(RuntimeError, match="open\\(\\) has not been called"):
            runs_index._conn()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        await runs_index.close()
        await runs_index.close()  # second call must not raise
        assert runs_index.is_open() is False

    @pytest.mark.asyncio
    async def test_integrity_check_returns_false_when_unopened(self) -> None:
        """integrity_check with no path and no open DB returns False, not exception."""
        await runs_index.close()
        assert await runs_index.integrity_check() is False

    @pytest.mark.asyncio
    async def test_integrity_check_unreadable_path_returns_false(
        self, tmp_path: Path
    ) -> None:
        """A file that can't be opened as SQLite returns False (no raise)."""
        bogus = tmp_path / "not_a_db.bin"
        bogus.write_bytes(b"\x00" * 100)
        assert await runs_index.integrity_check(bogus) is False

    @pytest.mark.asyncio
    async def test_stats_after_bootstrap_records_last_bootstrap_unix(
        self, tmp_path: Path, index_path: Path
    ) -> None:
        """After bootstrap walks an extant base dir, stats() reports last_bootstrap_unix.

        Why: when ``base`` doesn't exist, bootstrap returns immediately WITHOUT
        stamping ``last_bootstrap_unix`` (early-return path). Only a real walk
        sets the meta key. Pin both contracts.
        """
        # Existing-dir path stamps the meta key
        base = tmp_path / "results"
        base.mkdir()
        await runs_index.bootstrap(base)
        s = await runs_index.stats(index_path)
        assert s["last_bootstrap_unix"] is not None
        assert s["schema_version"] == runs_index.SCHEMA_VERSION
        assert s["runs_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_before_bootstrap_returns_none_last_bootstrap(
        self, index_path: Path
    ) -> None:
        """Fresh DB → last_bootstrap_unix is None until a real bootstrap walk runs."""
        s = await runs_index.stats(index_path)
        assert s["last_bootstrap_unix"] is None


# ============================================================
# _select_dicts internals
# ============================================================


class TestSelectDictsInternals:
    """The select-helper has a special carve-out for missing columns."""

    @pytest.mark.asyncio
    async def test_select_dicts_swallows_no_such_column_returns_empty(
        self, index_path: Path
    ) -> None:
        rows = await runs_index._select_dicts(
            "SELECT not_a_column_anywhere FROM runs", ()
        )
        assert rows == []

    @pytest.mark.asyncio
    async def test_select_dicts_propagates_other_operational_errors(
        self, index_path: Path
    ) -> None:
        """Real syntax errors must NOT be swallowed — only 'no such column'."""
        with pytest.raises(sqlite3.OperationalError):
            await runs_index._select_dicts("SELECT FROM WHERE", ())
