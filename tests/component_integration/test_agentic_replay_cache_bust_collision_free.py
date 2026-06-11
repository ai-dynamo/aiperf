# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end uniqueness test for the cache-bust marker under sustained load.

Sibling: ``test_agentic_replay_cache_bust.py`` (covers position correctness,
per-target parametrization, recycle-rotation observation under tight
duration). This file pushes the duration up so 100+ recycles per trace
happen, and asserts that across the entire run the rid set has zero
duplicates -- the regression bar for the collision-free fix.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.test_agentic_replay_cache_bust import (
    _payload_dict,
    _system_content,
    _write_weka_fixture,
)
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _extract_rid(text: str) -> str | None:
    m = _RID_RE.search(text)
    return m.group(0) if m else None


@pytest.fixture
def weka_collision_fixture(tmp_path: Path) -> Path:
    """4-trace fixture, non-zero system tokens so the SYSTEM_* path is exercised."""
    return _write_weka_fixture(tmp_path / "weka_collision", num_traces=4)


def _build_cmd(weka_dir: Path, *, duration: int) -> str:
    """Build an aiperf command that drives many distinct recycled sessions.

    4 traces x concurrency=3 over a multi-second benchmark window forces
    continuous recycle of the small pool, so each completed session mints a
    fresh marker. The exact session count is wall-clock-dependent (it scales
    with machine speed); the assertion floor below is set well under what even
    a loaded machine produces so the zero-collision contract -- not throughput
    -- is what the test gates on.
    """
    return f"""
        aiperf profile
            --model claude-haiku-4-5-20251001
            --model claude-opus-4-5-20251101
            --endpoint-type chat
            --streaming
            --custom-dataset-type weka_trace
            --input-file {weka_dir}
            --no-fixed-schedule
            --benchmark-duration {duration}
            --concurrency 3
            --random-seed 42
            --tokenizer {defaults.tokenizer}
            --extra-inputs ignore_eos:true
            --workers-max {defaults.workers_max}
            --ui {defaults.ui}
            --scenario inferencex-agentx-mvp
            --unsafe-override
            --cache-bust system_prefix
            --export-level raw
    """


def test_no_marker_collisions_across_large_recycle_run(
    cli: AIPerfCLI,
    weka_collision_fixture: Path,
) -> None:
    """Sustained-load run with cache-bust=SYSTEM_PREFIX must produce zero rid
    duplicates across PROFILING sessions.

    WARMUP and the FIRST PROFILING dispatch for each trajectory share the
    same rid by design (pass=0, same lane, same trace_id, same benchmark_id)
    so the server's prefix cache hit transfers warmup work into the
    measurement window. This test scopes the uniqueness assertion to
    PROFILING records only; the warmup-coherent pair is covered by
    ``test_agentic_replay_marker_uniqueness.py``.

    Asserts (within PROFILING):
      1. Every session has exactly one rid (intra-session marker continuity).
      2. ``len(set(rids)) == len(rids)`` across all sessions (zero collisions).
      3. >=20 distinct rids observed -- a non-vacuity floor, set well below the
         session count a loaded machine produces so it does not flake on
         throughput. The zero-collision check (2) is the real regression bar:
         the pre-fix 33% collision rate is caught with ~99.9% probability even
         at 20 sessions, so this floor does not weaken detection.
    """
    cmd = _build_cmd(weka_collision_fixture, duration=10)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed: stderr=\n{result.stderr}"
        f"\nlog tail=\n{(result.log or '')[-2000:]}"
    )
    assert result.raw_records is not None and len(result.raw_records) > 0, (
        "raw records JSONL must be present and non-empty"
    )

    # Group records by x_correlation_id, scoped to PROFILING phase only.
    by_session: dict[str, list] = defaultdict(list)
    for rec in result.raw_records:
        if rec.metadata.benchmark_phase != "profiling":
            continue
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None:
            by_session[xcorr].append(rec)

    # Per-session rid extraction + intra-session consistency check.
    session_rids: list[str] = []
    for xcorr, records in by_session.items():
        rids_in_session: set[str] = set()
        for rec in records:
            payload = _payload_dict(rec)
            carrier = _system_content(payload) or ""
            rid = _extract_rid(carrier)
            if rid is not None:
                rids_in_session.add(rid)
        assert len(rids_in_session) == 1, (
            f"session={xcorr}: expected exactly one rid across "
            f"{len(records)} turns; got {rids_in_session}"
        )
        session_rids.append(next(iter(rids_in_session)))

    assert len(session_rids) >= 20, (
        f"Need >=20 sessions for a non-vacuous uniqueness test; "
        f"got {len(session_rids)}. Increase --benchmark-duration or shrink the "
        f"fixture if a slower machine is under-producing sessions."
    )

    # The hard contract: zero duplicates across the entire run.
    duplicates = len(session_rids) - len(set(session_rids))
    assert duplicates == 0, (
        f"Marker collision detected: {duplicates} duplicate rids across "
        f"{len(session_rids)} sessions. Pre-fix this run produced ~33% "
        f"collisions; post-fix must be exactly zero."
    )
