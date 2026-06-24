# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ship-gate stress tests for the agentx exact-cutoff hard-cap ("N means N").

The targeted exact-cutoff tests (``test_dag_full_topology`` /
``test_dag_spawn``) prove the mechanism on tiny topologies. This module
stresses the *hard* paths the mechanism has to survive in production.

Two engine facts shape these tests:

- AIPerf rejects ``--concurrency`` greater than ``--request-count`` (or
  ``--num-conversations``). So a cap-``C`` run can use at most ``C`` concurrent
  slots.
- Under ``--request-count`` the engine RECYCLES root conversations up to the
  concurrency level: with a single root in the dataset and ``--concurrency K``,
  up to ``K`` root trees run at once, each fanning out, and the cap counts
  roots + children across all of them. The exact-count guarantee holds across
  that whole concurrent, recycled, fanned-out mess: total wire requests ==
  ``C`` exactly.

Each test asserts the two ship guarantees:
  1. EXACT count -- ``len(raw_records) == cap`` (N means N, zero overshoot).
  2. ZERO deadlock -- the run exits 0 within the timeout (the harness turns a
     hang into a ``RuntimeError`` via SIGINT and a non-zero exit raises), so a
     clean return is itself the no-deadlock assertion.

The deterministic single-root tests (``--concurrency 1``) pin exact topology
behaviour (depth, drain, no-over-truncation); the high-concurrency tests pin
the overshoot race + no-deadlock under recycled concurrent fan-out.

Fixtures are generated per-test into ``tmp_path`` so the topology parameters
live next to the assertions that depend on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

MODEL = "Qwen3-0.6B"


@pytest.fixture(autouse=True)
def _isolate_mmap_cache(monkeypatch):
    """Disable the cross-run mmap cache for these tests.

    Each test builds its own tiny DAG dataset. The persistent content-addressed
    mmap cache (``~/.cache/aiperf/dataset_mmap``) is shared across all runs and,
    under parallel xdist, races/cross-contaminates between tests. Disabling it
    forces every run to rebuild its own mmap deterministically. The runner
    propagates ``os.environ`` to the aiperf subprocess, so this reaches it.
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_CACHE_ENABLED", "0")


# --- Fixture generators ----------------------------------------------------


def _turn(
    content: str,
    *,
    forks: list[str] | None = None,
    spawns: list[Any] | None = None,
    max_tokens: int = 8,
) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if forks:
        turn["forks"] = forks
    if spawns:
        turn["spawns"] = spawns
    return turn


def _session(session_id: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"session_id": session_id, "turns": turns}


def _write_dag(path: Path, sessions: list[dict[str, Any]]) -> None:
    with open(path, "wb") as f:
        for sess in sessions:
            f.write(orjson.dumps(sess))
            f.write(b"\n")


def _wide_fanout(width: int, mode: str = "forks") -> tuple[list[dict[str, Any]], int]:
    """Root with one turn that fans out to ``width`` single-turn children.

    Returns (sessions, single_tree_size) where single_tree_size = 1 root +
    ``width`` children.
    """
    child_ids = [f"c{i}" for i in range(width)]
    root = _session("root", [_turn("root user", **{mode: child_ids})])
    children = [_session(cid, [_turn(f"{cid} user")]) for cid in child_ids]
    return [root, *children], 1 + width


def _root_forks_multiturn_child(child_turns: int) -> tuple[list[dict[str, Any]], int]:
    """Root forks ONE child that has ``child_turns`` turns. DAG fan-out is a
    single level (a fork-child's own forks are not re-dispatched), so the real
    depth axis is a child advancing through many turns. tree_size == 1 root +
    ``child_turns``; the child reaches turn_index ``child_turns - 1``."""
    root = _session("root", [_turn("root user", forks=["child"])])
    child = _session("child", [_turn(f"child turn {i}") for i in range(child_turns)])
    return [root, child], 1 + child_turns


def _delayed_join(
    n_children: int, parent_turns: int, join_at: int
) -> tuple[list[dict[str, Any]], int]:
    """Root with ``parent_turns`` turns that spawns ``n_children`` on turn 0
    with a delayed ``join_at``. single_tree_size == parent_turns + n_children."""
    child_ids = [f"sp{i}" for i in range(n_children)]
    turns: list[dict[str, Any]] = []
    for t in range(parent_turns):
        if t == 0:
            turns.append(
                _turn("root t0", spawns=[{"children": child_ids, "join_at": join_at}])
            )
        else:
            turns.append(_turn(f"root t{t}"))
    root = _session("root", turns)
    children = [_session(cid, [_turn(f"{cid} user")]) for cid in child_ids]
    return [root, *children], parent_turns + n_children


def _cmd(
    *,
    server_url: str,
    fixture: Path,
    request_count: int | None = None,
    num_conversations: int | None = None,
    concurrency: int = 1,
    workers_max: int = 4,
) -> str:
    count_flag = (
        f"--request-count {request_count}"
        if request_count is not None
        else f"--num-conversations {num_conversations}"
    )
    return f"""
        aiperf profile \
            --model {MODEL} \
            --url {server_url} \
            --endpoint-type chat \
            --input-file {fixture} \
            --custom-dataset-type dag_jsonl \
            {count_flag} \
            --concurrency {concurrency} \
            --workers-max {workers_max} \
            --export-level raw \
            --ui simple
        """


# --- Deterministic single-root tests (--concurrency 1) ---------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wide_fanout_single_root_cap_drains(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """Single root forks 40 children; ``--request-count 20`` caps mid-fan-out.
    At ``--concurrency 1`` exactly one root tree runs (no recycle before the
    cap bites), so this pins the clean drain path: root + 19 children land, the
    rest are truncated, and every suspended-parent join drains (no hang)."""
    fixture = tmp_path / "wide_single.dag.jsonl"
    width, cap = 40, 20
    sessions, tree_size = _wide_fanout(width)
    _write_dag(fixture, sessions)
    assert cap < tree_size

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            request_count=cap,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == cap, (
        f"--request-count {cap} on a {width}-wide single root must cap at "
        f"EXACTLY {cap}; got {len(result.raw_records)}"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    assert bs.children_errored == 0, (
        f"FORK children must seed cleanly, zero errored (stats={bs.stats_dict()})"
    )
    assert bs.children_spawned > 0, (
        f"children must actually dispatch, else errored==0 is vacuous "
        f"(stats={bs.stats_dict()})"
    )
    assert bs.children_truncated > 0, (
        f"cap must truncate children (stats={bs.stats_dict()})"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "every emitted wire request must be error-free (no FORK-routing failures)"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_multiturn_fork_child_runs_all_turns(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """A FORK child with many turns advances through ALL of them, re-seeding
    from the parent at each turn. Run single-pass (``--num-conversations 1``,
    ``--concurrency 1``: no recycle, no cap) so the topology is deterministic.
    Complements ``full.dag`` (2-turn children) with a deeper 6-turn child and
    guards multi-turn FORK continuation + clean seeding end to end.

    (Truncating a child mid-continuation is NOT cleanly testable under
    ``--request-count``: recycle competes for the same capped budget and tends
    to start fresh roots instead of advancing an existing child's later turns.
    The cap-honours-children path is covered by the wide-fan-out tests and the
    unit-level ``test_child_turn_honors_request_count_cap``.)"""
    fixture = tmp_path / "multiturn.dag.jsonl"
    child_turns = 6
    sessions, tree_size = _root_forks_multiturn_child(child_turns)
    _write_dag(fixture, sessions)

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            num_conversations=1,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == tree_size, (
        f"single-pass run must emit root + all {child_turns} child turns "
        f"({tree_size}); got {len(result.raw_records)}"
    )
    # The child advanced through every one of its turns.
    max_turn = max(r.metadata.turn_index for r in result.raw_records)
    assert max_turn == child_turns - 1, (
        f"child must advance to turn_index {child_turns - 1}; got {max_turn}"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    assert bs.children_errored == 0, (
        f"the FORK child must seed cleanly, zero errored (stats={bs.stats_dict()})"
    )
    assert bs.children_completed == 1, (
        f"the child must complete all turns and join (stats={bs.stats_dict()})"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "every emitted wire request must be error-free"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cap_equal_to_tree_size_emits_whole_tree(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """Boundary at ``--concurrency 1``: ``--request-count`` == single-tree size
    emits EXACTLY that tree -- no off-by-one undershoot, no over-truncation, no
    recycle into a second root. Guards N-means-N from below."""
    fixture = tmp_path / "exact.dag.jsonl"
    width = 24
    sessions, tree_size = _wide_fanout(width)
    _write_dag(fixture, sessions)

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            request_count=tree_size,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == tree_size, (
        f"cap == tree size ({tree_size}) must emit exactly that many wire "
        f"requests (no undershoot from below); got {len(result.raw_records)}"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    # The FORK fan-out seeded cleanly: no child hit the sticky-routing invariant
    # ("parent session not found on this worker"). This is the regression guard
    # for the payload_bytes-x-FORK / FORK-pin fixes.
    assert bs.children_errored == 0, (
        f"FORK children must seed cleanly, zero errored (stats={bs.stats_dict()})"
    )
    assert bs.children_completed > 0, (
        f"children must complete and join (stats={bs.stats_dict()})"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "every emitted wire request must be error-free (no FORK-routing failures)"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delayed_join_resumes_when_uncapped(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """Delayed-join SPAWN, uncapped (``--num-conversations 1``, concurrency 1):
    the parent suspends at ``join_at``, all children complete, and the parent
    RESUMES its join turn. Proves the join machinery end-to-end before we
    stress it under the cap."""
    fixture = tmp_path / "join_full.dag.jsonl"
    n_children, parent_turns, join_at = 6, 4, 2
    sessions, tree_size = _delayed_join(n_children, parent_turns, join_at)
    _write_dag(fixture, sessions)

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            num_conversations=1,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == tree_size, (
        f"uncapped delayed-join must run the whole tree ({tree_size}); got "
        f"{len(result.raw_records)}"
    )
    assert result.json is not None and result.json.branch_stats is not None
    assert all(rec.error is None for rec in result.raw_records), (
        "no wire request should error in a clean uncapped run"
    )
    bs = result.json.branch_stats
    assert bs.children_completed == n_children
    assert bs.parents_resumed >= 1, (
        f"parent must resume its join turn after children complete "
        f"(stats={bs.stats_dict()})"
    )
    assert bs.joins_suppressed == 0
    # Drain invariant: every suspended parent resolved (resumed or suppressed),
    # i.e. none wedged. A clean exit alone can't prove this -- a cap-reached run
    # can shut down with a join still outstanding.
    assert bs.parents_suspended == bs.parents_resumed + bs.joins_suppressed, (
        f"every suspended parent must drain (stats={bs.stats_dict()})"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delayed_join_suppressed_under_cap_no_deadlock(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """Delayed-join SPAWN truncated by a cap that bites mid-tree
    (``--concurrency 1`` => single root): the parent suspends at ``join_at``
    while children are in flight, the cap then refuses further issuance, and the
    run STILL finishes -- the suspended-parent join must drain (via
    ``on_child_stopped`` truncation or ``joins_suppressed``) instead of
    deadlocking. EXACTLY ``cap`` wire requests, zero errored, and the cap
    provably forced truncation/suppression somewhere."""
    fixture = tmp_path / "join_capped.dag.jsonl"
    n_children, parent_turns, join_at = 6, 4, 2
    sessions, tree_size = _delayed_join(n_children, parent_turns, join_at)
    _write_dag(fixture, sessions)
    # Cap bites well below the full tree so the join region is truncated.
    cap = 4
    assert cap < tree_size

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            request_count=cap,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == cap, (
        f"delayed-join capped at {cap} must emit exactly {cap}; got "
        f"{len(result.raw_records)}"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "the cap stops new issuance; in-flight requests complete without error"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    # No spurious errors, and the cap forced the join region to drain cleanly
    # one way or another (children truncated and/or the parent join suppressed).
    assert bs.children_errored == 0, f"unexpected errored children: {bs.stats_dict()}"
    assert (bs.children_truncated + bs.joins_suppressed) >= 1, (
        f"the cap must force truncation or join-suppression (stats={bs.stats_dict()})"
    )
    # Drain invariant: no suspended parent wedged even though the cap truncated
    # the join region. The real no-deadlock proof (a cap-reached run can exit 0
    # with a join still outstanding; a clean exit alone wouldn't catch it).
    assert bs.parents_suspended == bs.parents_resumed + bs.joins_suppressed, (
        f"every suspended parent must drain under the cap (stats={bs.stats_dict()})"
    )


# --- High-concurrency recycle + fan-out race (overshoot stress) ------------


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "cap", "concurrency"),
    [
        (80, 40, 8),  # ~8 recycled roots + children racing to fill 40
        (64, 48, 24),  # heavier concurrency, mid-fan-out cap
        (50, 30, 16),  # smaller tree, 16-wide concurrent race
    ],
)
async def test_wide_fanout_cap_exact_under_high_concurrency(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
    width: int,
    cap: int,
    concurrency: int,
):
    """The overshoot race: ``--concurrency`` recycles multiple root trees that
    fan out concurrently, all racing the same ``--request-count`` cap. Dozens
    of children acquire concurrency slots at once; the post-acquire cap
    re-check + synchronous increment must let EXACTLY ``cap`` total land. Zero
    overshoot, and the truncated children's joins drain (no deadlock)."""
    fixture = tmp_path / "wide_hiconc.dag.jsonl"
    sessions, _tree_size = _wide_fanout(width)
    _write_dag(fixture, sessions)

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            request_count=cap,
            concurrency=concurrency,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == cap, (
        f"--request-count {cap} must cap at EXACTLY {cap} under concurrency "
        f"{concurrency} with recycled {width}-wide fan-out; got "
        f"{len(result.raw_records)} (overshoot => cap re-check race)"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    assert bs.children_errored == 0, (
        f"FORK children must seed cleanly, zero errored (stats={bs.stats_dict()})"
    )
    assert bs.children_spawned > 0, (
        f"children must actually dispatch, else errored==0 is vacuous "
        f"(stats={bs.stats_dict()})"
    )
    assert bs.children_truncated > 0, (
        f"cap crossed mid-fan-out must truncate children (stats={bs.stats_dict()})"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "every emitted wire request must be error-free (no FORK-routing failures)"
    )


# --- Regression: FORK is clean WITH the mmap cache enabled -----------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fork_dataset_clean_with_mmap_cache_enabled(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
    monkeypatch,
):
    """A single-turn-root-with-FORK dataset is exactly the shape that pre-fix
    slipped through the preformat's single-turn exception into the PAYLOAD_BYTES
    fast path and poisoned the mmap cache. Re-enable the cache (in an isolated
    per-test dir so it can't cross-contaminate) and confirm the cache-enabled
    build path selects CONVERSATION, so FORK children seed cleanly. Guards the
    miss-path branch guard + MANIFEST_VERSION bump against a regression that only
    shows up when the cache machinery is active (the rest of this module runs
    with the cache disabled)."""
    # Override the module's autouse cache-disable for this one test.
    monkeypatch.setenv("AIPERF_DATASET_MMAP_CACHE_ENABLED", "1")
    monkeypatch.setenv("AIPERF_DATASET_MMAP_CACHE_DIR", str(tmp_path / "mmcache"))

    fixture = tmp_path / "fork_cached.dag.jsonl"
    width = 8
    sessions, tree_size = _wide_fanout(width)
    _write_dag(fixture, sessions)

    result = await cli.run(
        _cmd(
            server_url=aiperf_mock_server.url,
            fixture=fixture,
            num_conversations=1,
            concurrency=1,
        ),
        timeout=300.0,
    )

    assert result.raw_records is not None
    assert len(result.raw_records) == tree_size, (
        f"cache-enabled FORK run must emit root + {width} children ({tree_size}); "
        f"got {len(result.raw_records)}"
    )
    assert all(rec.error is None for rec in result.raw_records), (
        "FORK children must seed cleanly through the cache-enabled path; a "
        "payload_bytes-poisoned format would error them ('parent session not "
        "found on this worker')"
    )
    assert result.json is not None and result.json.branch_stats is not None
    bs = result.json.branch_stats
    assert bs.children_errored == 0, f"stats={bs.stats_dict()}"
    assert bs.children_completed == width, (
        f"all {width} children must complete and join (stats={bs.stats_dict()})"
    )
