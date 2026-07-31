# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial end-to-end tests for weka flattened-agent LCP splitting: the loader must split untagged flat fan-outs into per-agent ``::fa:`` child conversations and the runtime must reproduce their recorded concurrency, asserted per play over duration-bounded ``aiperf profile`` subprocess runs."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

import orjson
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricRecordMetadata
from tests.harness.utils import AIPerfMockServer, AIPerfResults, AIPerfRunnerResult
from tests.integration.conftest import IntegrationTestDefaults, get_venv_python

MockServerFactory = Callable[..., AsyncIterator[AIPerfMockServer]]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

BLOCK_SIZE = 64
TRACE_MODEL = "m"
SPLIT_ENV_VAR = "AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS"

# Exact log shapes emitted by WekaTraceLoader. The regex non-greedily skips the
# aux/worker-group bracketed breakdown (disabled for these mechanics tests) so
# the captured (trace, agents, seams, spawned chains, empty-hash) tuple is stable.
DETECTED_LINE_RE = re.compile(
    r"Trace (\S+): detected (\d+) agents "
    r"\((\d+) seams merged, (\d+) spawned chains"
    r"(?: \[[^\]]*\])?, (\d+) empty-hash kept on main\)"
)
SPLIT_SUMMARY_RE = re.compile(
    r"flattened-agent detection split (\d+) trace\(s\) into "
    r"(\d+) extra agent chain\(s\)"
)

# trace_id -> (root turn count, {child session-id suffix: child turn count})
FANOUT_EXPECTED: dict[str, tuple[int, dict[str, int]]] = {
    "trace_alpha": (3, {"fa:000": 2, "fa:001": 1}),
    "trace_beta": (2, {"fa:000": 1, "fa:001": 1}),
    "trace_gamma": (3, {"fa:000": 2}),
}
# (trace_id, agents, seams, spawned chains, empty-hash) per detection line.
FANOUT_EXPECTED_DETECTION: set[tuple[str, int, int, int, int]] = {
    ("trace_alpha", 3, 0, 2, 0),
    ("trace_beta", 3, 0, 2, 0),
    ("trace_gamma", 2, 0, 1, 0),
}
# With the split disabled the legacy single stream chains every retained
# top-level request into one root conversation, in time order.
LEGACY_EXPECTED_ROOT_TURNS: dict[str, int] = {
    "trace_alpha": 6,
    "trace_beta": 4,
    "trace_gamma": 5,
}


def _req(
    t: float,
    hash_ids: list[int],
    *,
    api_time: float,
    out: int = 8,
    model: str = TRACE_MODEL,
    request_type: str = "n",
) -> dict:
    """One hash-aligned top-level normal request (in == len(hash_ids) * 64)."""
    return {
        "t": t,
        "type": request_type,
        "model": model,
        "in": len(hash_ids) * BLOCK_SIZE,
        "out": out,
        "hash_ids": hash_ids,
        "api_time": api_time,
    }


def _write_trace(target_dir: Path, trace_id: str, requests: list[dict]) -> None:
    """Write one weka trace file with requests sorted by recorded ``t`` so legacy chaining never produces negative inter-turn delays."""
    target_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "id": trace_id,
        "models": [TRACE_MODEL],
        "block_size": BLOCK_SIZE,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": sorted(requests, key=lambda r: r["t"]),
    }
    (target_dir / f"{trace_id}.json").write_bytes(orjson.dumps(trace))


def _write_fanout_corpus(target_dir: Path) -> Path:
    """Three fan-out traces (main chain + chaining-hash_id workers forking at a shared prefix) with varied per-trace timing/topology; see FANOUT_EXPECTED for the split structure."""
    # trace_alpha: main 3 turns; worker A 2 turns (joins main turn 2);
    # worker B 1 turn (joins main turn 1). Workers overlap in recorded time.
    _write_trace(
        target_dir,
        "trace_alpha",
        [
            _req(0.00, [1, 2, 3], api_time=0.12),  # main t0
            _req(0.25, [1, 2, 50, 51], api_time=0.75),  # worker A t0, ends 1.00
            _req(0.30, [1, 2, 60, 61], api_time=0.50),  # worker B, ends 0.80
            _req(1.05, [1, 2, 50, 51, 52], api_time=0.15),  # worker A t1, ends 1.20
            _req(1.10, [1, 2, 3, 4], api_time=0.12),  # main t1 (gated by B)
            _req(1.75, [1, 2, 3, 4, 5], api_time=0.12),  # main t2 (gated by A)
        ],
    )
    # trace_beta: main 2 turns; two single-turn workers that overlap in time
    # (worker A is still in flight when worker B starts, vetoing any seam
    # splice) and share one (spawn turn 0, join turn 1) branch group.
    _write_trace(
        target_dir,
        "trace_beta",
        [
            _req(0.00, [201, 202, 203], api_time=0.10),  # main t0
            _req(0.20, [201, 202, 210, 211], api_time=0.40),  # worker A, ends 0.60
            _req(0.25, [201, 202, 220, 221], api_time=0.40),  # worker B, ends 0.65
            _req(1.00, [201, 202, 203, 204], api_time=0.10),  # main t1 (gated)
        ],
    )
    # trace_gamma: main 3 turns; one 2-turn worker joining main turn 1.
    _write_trace(
        target_dir,
        "trace_gamma",
        [
            _req(0.00, [301, 302, 303], api_time=0.10),  # main t0
            _req(0.20, [301, 302, 310], api_time=0.20),  # worker t0, ends 0.40
            _req(0.50, [301, 302, 310, 311], api_time=0.20),  # worker t1, ends 0.70
            _req(1.00, [301, 302, 303, 304], api_time=0.10),  # main t1 (gated)
            _req(1.60, [301, 302, 303, 304, 305], api_time=0.10),  # main t2
        ],
    )
    return target_dir


def _write_join_trace(target_dir: Path) -> Path:
    """One trace whose 5-turn worker chain (ending before main turn 1's recorded t) must gate the second main turn via SPAWN_JOIN, verifiable because the mock server runs slower than recorded."""
    requests = [_req(0.0, [1, 2, 3], api_time=0.05)]
    chain = [1, 2, 9]
    for k in range(5):
        requests.append(_req(0.1 * (k + 1), list(chain), api_time=0.05))
        chain = [*chain, 10 + k]
    requests.append(_req(1.0, [1, 2, 3, 4], api_time=0.05))
    _write_trace(target_dir, "trace_join", requests)
    return target_dir


def _write_issue_1231_trace(target_dir: Path) -> Path:
    """Scaled form of issue #1231's root-child-root join timing.

    Source seconds are divided by 1000 so the integration test stays fast:
    the previous root ends at 31.042ms, the child ends at 47.658ms, and the
    gated root is due at 229.088ms. That preserves both the parent's 198.046ms
    end-to-start delay and the source-aligned 181.430ms post-child residual.
    """
    _write_trace(
        target_dir,
        "trace_issue_1231",
        [
            _req(
                0.0,
                [1, 2, 3],
                api_time=0.031042,
                out=1,
                request_type="s",
            ),
            _req(0.021181, [1, 2, 50, 51], api_time=0.026477, out=1),
            _req(
                0.229088,
                [1, 2, 3, 4],
                api_time=0.005489,
                out=1,
                request_type="s",
            ),
        ],
    )
    return target_dir


def _write_background_trace(target_dir: Path) -> Path:
    """One trace whose worker chain ends after the last main turn, so the loader emits an ``is_background=True`` branch (no SPAWN_JOIN) the runtime must still send and drain cleanly."""
    _write_trace(
        target_dir,
        "trace_bg",
        [
            _req(0.0, [1, 2, 3], api_time=0.05),  # main t0
            _req(0.2, [1, 2, 70], api_time=0.10),  # worker t0
            _req(0.4, [1, 2, 3, 4], api_time=0.05),  # main t1 (last main turn)
            _req(0.6, [1, 2, 70, 71], api_time=0.10),  # worker t1, ends 0.70
        ],
    )
    return target_dir


def _write_poisoned_corpus(target_dir: Path) -> Path:
    """Two traces of 10 mutually disjoint single-block-hash requests each, which LCP detection treats as independent zero-depth founder chains."""
    for tid, base in (("trace_poison_a", 1000), ("trace_poison_b", 2000)):
        _write_trace(
            target_dir,
            tid,
            [_req(0.05 * i, [base + i], api_time=0.01, out=4) for i in range(10)],
        )
    return target_dir


async def _run_weka_profile(
    *,
    input_dir: Path,
    artifact_dir: Path,
    url: str,
    duration: float,
    concurrency: int | None = None,
    random_seed: int = 42,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 200.0,
) -> AIPerfResults:
    """Run one full ``aiperf profile`` subprocess over a weka trace directory with per-run artifact dir and env; a timeout here signals an orchestration drain deadlock."""
    args = [
        "profile",
        "--model",
        "test-model",
        "--endpoint-type",
        "chat",
        "--url",
        url,
        "--custom-dataset-type",
        "weka_trace",
        "--input-file",
        str(input_dir),
        "--no-fixed-schedule",
        "--benchmark-duration",
        str(duration),
        # Bounded grace: normal drains finish in ~1-2s; the bound also caps
        # the cost of a worst-case DAG drain stall (a delay-scheduled child
        # turn cancelled by the deadline's cancel_all_pending leaves
        # has_pending_branch_work stuck until the grace timeout fires).
        "--benchmark-grace-period",
        "20",
        "--random-seed",
        str(random_seed),
        "--workers-max",
        "2",
        "--export-level",
        "records",
        "--ui",
        "simple",
        # These tests assert on WekaTraceLoader detection log lines, which are
        # emitted at DEBUG; the artifact log file captures at the run level, so
        # the run must be DEBUG for _combined_log_text to see them.
        "--log-level",
        "debug",
        "--tokenizer",
        IntegrationTestDefaults.tokenizer,
        "--artifact-dir",
        str(artifact_dir),
    ]
    if concurrency is not None:
        args += ["--concurrency", str(concurrency)]
    args += extra_args or []

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # These tests assert on WekaTraceLoader log lines (detection,
        # poison WARNING). A warm mmap dataset cache legitimately skips the
        # loader entirely (and across tests the fixture CONTENT can collide,
        # e.g. determinism vs fanout runs), so force a cold loader per run.
        "AIPERF_DATASET_MMAP_CACHE_ENABLED": "false",
        # Disable the aux/worker-group sub-classifiers so detected chains keep the
        # ``::fa:`` tags these mechanics tests assert on (v2 defaults would
        # reclassify the synthetic workers to ``::aux:``/``::wg:``).
        "AIPERF_DATASET_WEKA_AUX_MAX_REQUESTS": "0",
        "AIPERF_DATASET_WEKA_AUX_REDUCTION_OSL_MAX": "0",
        "AIPERF_DATASET_WEKA_WORKER_GROUP_MIN": "0",
        **(extra_env or {}),
    }
    process = await asyncio.create_subprocess_exec(
        get_venv_python(),
        "-m",
        "aiperf",
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise AssertionError(
            f"aiperf profile did not exit within {timeout}s for {input_dir} -- "
            "duration-bounded DAG runs must drain and shut down; this is an "
            "orchestration deadlock/hang finding"
        ) from None

    return AIPerfResults(
        AIPerfRunnerResult(
            exit_code=process.returncode or 0,
            output_dir=artifact_dir,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )
    )


def _assert_success(result: AIPerfResults, label: str) -> None:
    assert result.exit_code == 0, (
        f"{label}: aiperf exited {result.exit_code}\n"
        f"STDERR (tail):\n{result.stderr[-4000:]}\n"
        f"STDOUT (tail):\n{result.stdout[-4000:]}\n"
        f"LOG (tail):\n{result.log[-4000:]}"
    )
    assert result.json is not None, f"{label}: profile_export_aiperf.json missing"
    assert result.jsonl, f"{label}: profile_export.jsonl missing or empty"


def _combined_log_text(result: AIPerfResults) -> str:
    """All run logs: per-service artifact log files plus captured console."""
    parts = [result.stdout or "", result.stderr or ""]
    for path in sorted(result.artifacts_dir.glob("**/logs/*.log")):
        parts.append(path.read_text(errors="replace"))
    return "\n".join(parts)


def _profiling_metadata(result: AIPerfResults) -> list[MetricRecordMetadata]:
    """Successful PROFILING-phase record metadata (errors/cancels excluded)."""
    return [
        record.metadata
        for record in result.jsonl or []
        if record.error is None
        and not record.metadata.was_cancelled
        and record.metadata.benchmark_phase == CreditPhase.PROFILING
    ]


@dataclass
class _Play:
    """One root session play and the child sessions it spawned."""

    trace_id: str
    root_corr: str
    root: list[MetricRecordMetadata]
    children: dict[str, list[MetricRecordMetadata]]


def _collect_plays(result: AIPerfResults) -> list[_Play]:
    """Group records into plays via x_correlation_id / parent_correlation_id."""
    roots: dict[str, list[MetricRecordMetadata]] = defaultdict(list)
    children: dict[str, list[MetricRecordMetadata]] = defaultdict(list)
    for md in _profiling_metadata(result):
        if md.agent_depth == 0:
            assert md.x_correlation_id is not None
            roots[md.x_correlation_id].append(md)
        else:
            assert md.parent_correlation_id is not None, (
                f"child record without parent_correlation_id: {md.conversation_id}"
            )
            children[md.parent_correlation_id].append(md)

    plays: list[_Play] = []
    for corr, recs in roots.items():
        recs.sort(key=lambda m: m.turn_index or 0)
        by_child: dict[str, list[MetricRecordMetadata]] = defaultdict(list)
        for child_md in children.get(corr, []):
            assert child_md.conversation_id is not None
            by_child[child_md.conversation_id].append(child_md)
        for child_recs in by_child.values():
            child_recs.sort(key=lambda m: m.turn_index or 0)
        assert recs[0].conversation_id is not None
        plays.append(
            _Play(
                trace_id=recs[0].conversation_id,
                root_corr=corr,
                root=recs,
                children=dict(by_child),
            )
        )
    return plays


def _child_suffix(conversation_id: str) -> str:
    return conversation_id.split("::", 1)[1] if "::" in conversation_id else ""


def _is_complete_play(
    play: _Play, expected: dict[str, tuple[int, dict[str, int]]]
) -> bool:
    """True iff this play executed exactly the expected per-session turns."""
    if play.trace_id not in expected:
        return False
    root_turns, child_spec = expected[play.trace_id]
    if [m.turn_index for m in play.root] != list(range(root_turns)):
        return False
    got = {
        _child_suffix(cid): [m.turn_index for m in recs]
        for cid, recs in play.children.items()
    }
    want = {suffix: list(range(n)) for suffix, n in child_spec.items()}
    return got == want


def _play_signature(play: _Play) -> tuple:
    return (
        play.trace_id,
        len(play.root),
        tuple(
            sorted(
                (_child_suffix(cid), len(recs)) for cid, recs in play.children.items()
            )
        ),
    )


def _detection_tuples(log_text: str) -> set[tuple[str, int, int, int, int]]:
    return {
        (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        for m in DETECTED_LINE_RE.finditer(log_text)
    }


def _sibling_children_overlap(plays: list[_Play]) -> bool:
    """True if any two sibling child sessions had requests in flight at once."""
    for play in plays:
        recs = [md for child in play.children.values() for md in child]
        for a in recs:
            for b in recs:
                if (
                    a.conversation_id != b.conversation_id
                    and a.request_start_ns < b.request_end_ns
                    and b.request_start_ns < a.request_end_ns
                ):
                    return True
    return False


def _assert_complete_play_order_and_joins(
    plays: list[_Play], *, assert_timing: bool = False
) -> None:
    """Assert source order, SPAWN_JOIN frontiers, and optional replay delays."""
    join_specs = {
        "trace_alpha": {1: ("fa:001",), 2: ("fa:000",)},
        "trace_beta": {1: ("fa:000", "fa:001")},
        "trace_gamma": {1: ("fa:000",)},
    }
    residual_delay_ms = {
        "trace_alpha": {1: 980.0, 2: 530.0},
        "trace_beta": {1: 900.0},
        "trace_gamma": {1: 900.0},
    }
    checked_trace_ids: set[str] = set()
    for play in plays:
        if play.trace_id not in FANOUT_EXPECTED or not _is_complete_play(
            play, FANOUT_EXPECTED
        ):
            continue
        for records in (play.root, *play.children.values()):
            assert [m.turn_index for m in records] == list(range(len(records)))
            for previous, current in zip(records, records[1:], strict=False):
                assert current.request_start_ns >= previous.request_end_ns, (
                    f"{play.trace_id}: turn {current.turn_index} started before "
                    f"turn {previous.turn_index} completed in "
                    f"{current.conversation_id}"
                )
        children_by_suffix = {
            _child_suffix(cid): records for cid, records in play.children.items()
        }
        for root_turn_index, child_suffixes in join_specs[play.trace_id].items():
            gated = play.root[root_turn_index]
            for suffix in child_suffixes:
                child_last = children_by_suffix[suffix][-1]
                assert gated.request_start_ns >= child_last.request_end_ns, (
                    f"{play.trace_id}: root turn {root_turn_index} crossed "
                    f"SPAWN_JOIN before child {suffix} completed"
                )
            if assert_timing:
                previous = play.root[root_turn_index - 1]
                replay_gap_ms = (
                    gated.request_start_ns - previous.request_end_ns
                ) / 1_000_000.0
                assert replay_gap_ms >= (
                    residual_delay_ms[play.trace_id][root_turn_index] - 25.0
                ), (
                    f"{play.trace_id}: root turn {root_turn_index} replayed "
                    f"after {replay_gap_ms:.1f}ms, shorter than its recorded "
                    f"{residual_delay_ms[play.trace_id][root_turn_index]:.1f}ms "
                    "end-to-start delay"
                )
        checked_trace_ids.add(play.trace_id)
    assert checked_trace_ids == set(FANOUT_EXPECTED), (
        "expected a fully ordered, join-valid play for every synthetic trace; "
        f"checked {sorted(checked_trace_ids)}"
    )


async def test_fanout_dir_splits_and_replays_recorded_concurrency(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """Spec 4/5: a directory of flattened fan-out traces splits into per-agent ::fa: children that log detection, execute their exact chains, and run siblings concurrently."""
    corpus = _write_fanout_corpus(tmp_path / "traces")
    async with mock_server_factory(ttft=150.0, itl=2.0, workers=4) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=7.0,
            concurrency=3,
            timeout=240.0,
        )
    _assert_success(result, "fanout dir")

    log_text = _combined_log_text(result)
    assert _detection_tuples(log_text) >= FANOUT_EXPECTED_DETECTION, (
        "per-trace 'detected N agents' log lines missing or wrong; got "
        f"{_detection_tuples(log_text)}, want at least {FANOUT_EXPECTED_DETECTION}"
    )
    summary = SPLIT_SUMMARY_RE.search(log_text)
    assert summary is not None, (
        "'flattened-agent detection split' summary log line missing"
    )
    assert (int(summary.group(1)), int(summary.group(2))) == (3, 5), (
        f"split summary counts wrong: {summary.group(0)}"
    )

    fa_records = [
        md
        for md in _profiling_metadata(result)
        if md.conversation_id and "::fa:" in md.conversation_id
    ]
    assert fa_records, "no ::fa: child session executed any request"
    assert all(md.agent_depth == 1 for md in fa_records), (
        "::fa: records must carry agent_depth=1 (spawned children)"
    )

    plays = _collect_plays(result)
    complete = [p for p in plays if _is_complete_play(p, FANOUT_EXPECTED)]
    assert {p.trace_id for p in complete} == set(FANOUT_EXPECTED), (
        "every trace must have at least one play whose per-session record "
        "counts match main+worker turn counts; complete plays: "
        f"{sorted(_play_signature(p) for p in complete)}; all plays: "
        f"{sorted(_play_signature(p) for p in plays)}"
    )
    _assert_complete_play_order_and_joins(plays, assert_timing=True)

    # Recorded concurrency reproduced: two sibling worker sessions of one
    # play overlap in wall-clock (each request takes >= ttft=150ms and the
    # siblings are dispatched together at the spawning turn's return). A
    # serialized single-stream replay can never overlap its own requests.
    assert _sibling_children_overlap(plays), (
        "no two sibling ::fa: sessions were in flight concurrently -- "
        "the split did not reproduce the recorded fan-out concurrency"
    )

    branch_stats = result.json.branch_stats
    assert branch_stats is not None, "branch_stats missing for a DAG-shaped run"
    assert branch_stats.children_spawned >= 5, branch_stats
    assert branch_stats.children_completed >= 5, branch_stats
    assert branch_stats.children_errored == 0, branch_stats


@pytest.mark.parametrize("warmup_requests_per_lane", [1, 10, 100])
async def test_cache_warmup_handoff_preserves_order_and_spawn_joins(
    tmp_path: Path,
    mock_server_factory: MockServerFactory,
    warmup_requests_per_lane: int,
) -> None:
    """Warmup quotas of different sizes hand profiling a complete DAG, never an independently resumed parent."""
    corpus = _write_fanout_corpus(tmp_path / "traces")
    async with mock_server_factory(fast=True, workers=4) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=3.0,
            concurrency=3,
            extra_args=[
                "--scenario",
                "inferencex-agentx-mvp",
                "--unsafe-override",
                "--ignore-trace-delays",
                "--warmup-requests-per-lane",
                str(warmup_requests_per_lane),
            ],
            timeout=240.0,
        )
    _assert_success(
        result, f"cache warmup handoff ({warmup_requests_per_lane} per lane)"
    )

    plays = _collect_plays(result)
    _assert_complete_play_order_and_joins(plays)
    branch_stats = result.json.branch_stats
    assert branch_stats is not None
    assert branch_stats.children_errored == 0, branch_stats
    assert branch_stats.parents_suspended >= 1, branch_stats
    assert branch_stats.parents_resumed >= 1, branch_stats


async def test_spawn_join_gates_main_turn_until_worker_chain_completes(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """Spec 5.3: the gated main turn is not dispatched until the worker chain's final response completes (verified against a mock server slower than recorded timings)."""
    corpus = _write_join_trace(tmp_path / "traces")
    async with mock_server_factory(ttft=500.0, itl=2.0, workers=2) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=6.0,
            concurrency=1,
            timeout=240.0,
        )
    _assert_success(result, "spawn-join gating")

    log_text = _combined_log_text(result)
    assert ("trace_join", 2, 0, 1, 0) in _detection_tuples(log_text), (
        f"expected 'Trace trace_join: detected 2 agents' line; got "
        f"{_detection_tuples(log_text)}"
    )

    gated_plays_checked = 0
    for play in _collect_plays(result):
        if play.trace_id != "trace_join":
            continue
        worker = play.children.get("trace_join::fa:000")
        gated = next((m for m in play.root if m.turn_index == 1), None)
        if gated is None or worker is None:
            continue
        if [m.turn_index for m in worker] != list(range(5)):
            continue  # truncated by the duration deadline; skip partial play
        # Child turns serialize within the session.
        for prev, nxt in zip(worker, worker[1:], strict=False):
            assert nxt.request_start_ns >= prev.request_end_ns, (
                "worker chain turns overlapped within one session: "
                f"turn {nxt.turn_index} started before turn {prev.turn_index} ended"
            )
        worker_last = worker[-1]
        assert gated.request_start_ns >= worker_last.request_end_ns, (
            "SPAWN_JOIN violated: gated main turn 1 started at "
            f"{gated.request_start_ns} before the worker chain's final "
            f"response completed at {worker_last.request_end_ns} "
            f"(delta {(worker_last.request_end_ns - gated.request_start_ns) / 1e6:.1f}ms)"
        )
        gated_plays_checked += 1
    assert gated_plays_checked >= 1, (
        "no complete gated play executed within the benchmark duration; "
        "cannot verify SPAWN_JOIN ordering"
    )

    branch_stats = result.json.branch_stats
    assert branch_stats is not None
    assert branch_stats.parents_suspended >= 1, (
        f"the parent never suspended on the gate (join was a no-op): {branch_stats}"
    )
    assert branch_stats.parents_resumed >= 1, branch_stats
    assert branch_stats.children_errored == 0, branch_stats


async def test_issue_1231_shape_preserves_parent_replay_deadline(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """The exact #1231 timing shape waits on replay time and child completion."""
    corpus = _write_issue_1231_trace(tmp_path / "traces")
    async with mock_server_factory(ttft=30.0, itl=0.0, workers=2) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=1.0,
            concurrency=1,
        )
    _assert_success(result, "issue #1231 timing shape")

    complete = [
        play
        for play in _collect_plays(result)
        if play.trace_id == "trace_issue_1231"
        and len(play.root) == 2
        and sorted(len(records) for records in play.children.values()) == [1]
    ]
    assert complete, "no complete replay of the #1231 root-child-root shape"
    play = complete[0]
    previous, gated = play.root
    child_last = next(iter(play.children.values()))[-1]

    assert [record.source_outer_idx for record in play.root] == [0, 2]
    assert child_last.source_outer_idx == 1
    assert gated.request_start_ns >= child_last.request_end_ns
    parent_gap_ms = (gated.request_start_ns - previous.request_end_ns) / 1_000_000
    assert parent_gap_ms >= 173.0, (
        f"gated parent replayed after {parent_gap_ms:.1f}ms; expected the "
        "198.046ms recorded parent delay within scheduler tolerance"
    )
    post_child_gap_ms = (gated.request_start_ns - child_last.request_end_ns) / 1_000_000
    assert post_child_gap_ms >= 140.0, (
        f"gated parent resumed only {post_child_gap_ms:.1f}ms after the child; "
        "the issue #1231 residual delay was dropped"
    )


async def test_system_idle_cap_advances_issue_1231_join_without_bypassing_child(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """The global idle guard advances a gated replay deadline, never its join."""
    corpus = _write_issue_1231_trace(tmp_path / "traces")
    async with mock_server_factory(ttft=30.0, itl=0.0, workers=2) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=1.0,
            concurrency=1,
            extra_args=[
                "--scenario",
                "inferencex-agentx-mvp",
                "--unsafe-override",
                "--system-idle-gap-cap-seconds",
                "0.05",
            ],
        )
    _assert_success(result, "issue #1231 timing shape with global idle cap")

    complete = [
        play
        for play in _collect_plays(result)
        if play.trace_id == "trace_issue_1231"
        and len(play.root) == 2
        and sorted(len(records) for records in play.children.values()) == [1]
    ]
    assert complete, "no complete replay of the capped #1231 join shape"
    play = complete[0]
    previous, gated = play.root
    child_last = next(iter(play.children.values()))[-1]

    assert gated.request_start_ns >= child_last.request_end_ns, (
        "the system-idle cap bypassed the required child-completion gate"
    )
    parent_gap_ms = (gated.request_start_ns - previous.request_end_ns) / 1_000_000
    post_child_gap_ms = (gated.request_start_ns - child_last.request_end_ns) / 1_000_000
    assert 35.0 <= post_child_gap_ms <= 100.0, (
        f"capped join resumed {post_child_gap_ms:.1f}ms after the child; "
        "expected the 50ms whole-system idle guard within process jitter"
    )
    assert parent_gap_ms < 173.0, (
        f"capped join still waited {parent_gap_ms:.1f}ms from its parent; "
        "the replay deadline was not advanced"
    )


async def test_background_worker_chain_runs_after_root_and_drains_cleanly(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Spec 5.3 (background): a chain outliving the last main turn becomes a fire-and-forget child that still sends, drains cleanly at shutdown, and never suspends the parent."""
    corpus = _write_background_trace(tmp_path / "traces")
    result = await _run_weka_profile(
        input_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        url=aiperf_mock_server.url,
        duration=5.0,
        concurrency=1,
        timeout=200.0,
    )
    _assert_success(result, "background chain")

    assert ("trace_bg", 2, 0, 1, 0) in _detection_tuples(_combined_log_text(result))

    expected = {"trace_bg": (2, {"fa:000": 2})}
    plays = _collect_plays(result)
    complete = [p for p in plays if _is_complete_play(p, expected)]
    assert complete, (
        "no play completed both main turns and both background child turns; "
        f"plays: {sorted(_play_signature(p) for p in plays)}"
    )
    for play in complete:
        # The chain's first request (outer idx 1) follows only main turn 0,
        # so the loader anchors the background branch on turn 0: the child
        # dispatches at turn 0's return, never earlier.
        spawning_turn = play.root[0]
        child_first = play.children["trace_bg::fa:000"][0]
        assert child_first.request_start_ns >= spawning_turn.request_end_ns, (
            "background child must spawn at its anchoring turn's return, but "
            "its first request started before main turn 0's response completed"
        )

    branch_stats = result.json.branch_stats
    assert branch_stats is not None
    assert branch_stats.children_spawned >= 1, branch_stats
    assert branch_stats.children_completed >= 1, branch_stats
    assert branch_stats.children_errored == 0, branch_stats
    # Background semantics: no SPAWN_JOIN gate exists anywhere in this trace,
    # so the parent must never have suspended waiting on the chain.
    assert branch_stats.parents_suspended == 0, (
        f"background-only trace must never suspend the parent: {branch_stats}"
    )


async def test_split_disabled_env_restores_legacy_single_stream(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Spec 6: ``AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false`` fully disables detection so every trace replays as one serialized root conversation with all rows."""
    corpus = _write_fanout_corpus(tmp_path / "traces")
    result = await _run_weka_profile(
        input_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        url=aiperf_mock_server.url,
        duration=5.0,
        concurrency=3,
        extra_env={SPLIT_ENV_VAR: "false"},
        timeout=200.0,
    )
    _assert_success(result, "split disabled")

    log_text = _combined_log_text(result)
    # Sanity that loader INFO logs are captured at all, so the absence
    # assertions below cannot pass vacuously.
    assert "WekaTraceLoader" in log_text, (
        "loader INFO logs not captured; absence assertions would be vacuous"
    )
    assert not _detection_tuples(log_text), (
        f"detection ran despite {SPLIT_ENV_VAR}=false: {_detection_tuples(log_text)}"
    )
    assert SPLIT_SUMMARY_RE.search(log_text) is None, (
        f"split summary logged despite {SPLIT_ENV_VAR}=false"
    )

    metadata = _profiling_metadata(result)
    fa_records = [
        md for md in metadata if md.conversation_id and "::fa:" in md.conversation_id
    ]
    assert not fa_records, (
        f"::fa: sessions executed despite {SPLIT_ENV_VAR}=false: "
        f"{sorted({md.conversation_id for md in fa_records})}"
    )

    plays = _collect_plays(result)
    for trace_id, n_turns in LEGACY_EXPECTED_ROOT_TURNS.items():
        assert any(
            play.trace_id == trace_id
            and not play.children
            and [m.turn_index for m in play.root] == list(range(n_turns))
            for play in plays
        ), (
            f"{trace_id}: expected at least one legacy single-stream play with "
            f"{n_turns} root turns and no children; plays: "
            f"{sorted(_play_signature(p) for p in plays if p.trace_id == trace_id)}"
        )

    branch_stats = result.json.branch_stats
    if branch_stats is not None:
        assert branch_stats.children_spawned == 0, branch_stats


_POISON_RATE_ARGS = ["--request-rate", "4", "--arrival-pattern", "constant"]


async def test_poisoned_corpus_completes_without_deadlock(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Whatever detection decides about nonce-poisoned hashes, the run completes cleanly and every recorded request is still replayed."""
    corpus = _write_poisoned_corpus(tmp_path / "traces")
    # 2 traces x 10 turns at --request-rate 4 needs ~5s of credits; 8s of
    # duration lets BOTH plays finish so the >=10-per-trace assertion below
    # measures dropped rows, not expected deadline truncation of play 2.
    result = await _run_weka_profile(
        input_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        url=aiperf_mock_server.url,
        duration=8.0,
        concurrency=4,
        extra_args=_POISON_RATE_ARGS,
        timeout=200.0,
    )
    _assert_success(result, "poisoned corpus completion")

    # Count EXECUTED requests, errored included: the degenerate poison rows
    # (every turn exactly one 64-token block, no partial tail) emit
    # assistant-only deltas, so the accumulated request ends on an assistant
    # message and the mock server returns no content -> turns 1+ record
    # InvalidInferenceResultError. That is a response-validation outcome,
    # irrelevant to this test's no-deadlock / no-dropped-rows claim.
    executed = [
        record.metadata
        for record in result.jsonl or []
        if not record.metadata.was_cancelled
        and record.metadata.benchmark_phase == CreditPhase.PROFILING
    ]
    for trace_id in ("trace_poison_a", "trace_poison_b"):
        trace_records = [
            md
            for md in executed
            if md.conversation_id
            and (
                md.conversation_id == trace_id
                or md.conversation_id.startswith(f"{trace_id}::")
            )
        ]
        # 10 recorded rows per trace; >= 10 records regardless of whether the
        # rows replay as one root (guarded) or root + children (split).
        assert len(trace_records) >= 10, (
            f"{trace_id}: only {len(trace_records)} requests executed; "
            "the poisoned trace's rows were dropped"
        )

    branch_stats = result.json.branch_stats
    if branch_stats is not None:
        assert branch_stats.children_errored == 0, branch_stats


async def test_disjoint_corpus_splits_into_fa_sessions(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """With the nonce-poison guard removed, a mutually-disjoint single-block-hash corpus splits into per-agent ::fa: sessions and logs no nonce-poison WARNING."""
    corpus = _write_poisoned_corpus(tmp_path / "traces")
    result = await _run_weka_profile(
        input_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        url=aiperf_mock_server.url,
        duration=4.0,
        concurrency=4,
        extra_args=_POISON_RATE_ARGS,
        timeout=200.0,
    )
    _assert_success(result, "disjoint corpus split behavior")

    fa_records = [
        md
        for md in _profiling_metadata(result)
        if md.conversation_id and "::fa:" in md.conversation_id
    ]
    assert fa_records, "disjoint traces must split into per-agent ::fa: chains"
    # The trace ids contain "poison" but never "nonce"; the removed guard's
    # warning was the only source of "nonce" in the logs.
    log_text = _combined_log_text(result).lower()
    assert "nonce" not in log_text, (
        "the nonce-poison guard was removed; no nonce-poison warning expected"
    )


async def test_two_identical_runs_produce_identical_split_structure(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Detection is deterministic: two identical runs produce the same per-trace detection results and per-play session structure."""
    corpus = _write_fanout_corpus(tmp_path / "traces")
    results: list[AIPerfResults] = []
    for run_idx in (1, 2):
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / f"artifacts_{run_idx}",
            url=aiperf_mock_server.url,
            duration=5.0,
            concurrency=3,
            random_seed=7,
            timeout=130.0,
        )
        _assert_success(result, f"determinism run {run_idx}")
        results.append(result)

    detections = [_detection_tuples(_combined_log_text(r)) for r in results]
    assert detections[0] == detections[1], (
        f"detection differs across identical runs:\n{detections[0]}\nvs\n{detections[1]}"
    )
    assert detections[0] == FANOUT_EXPECTED_DETECTION, (
        f"detection does not match the spec-expected chains: {detections[0]}"
    )

    signatures: list[set[tuple]] = []
    conversation_ids: list[set[str]] = []
    for result in results:
        plays = _collect_plays(result)
        complete = [p for p in plays if _is_complete_play(p, FANOUT_EXPECTED)]
        signatures.append({_play_signature(p) for p in complete})
        conversation_ids.append(
            {
                md.conversation_id
                for md in _profiling_metadata(result)
                if md.conversation_id
            }
        )

    expected_signatures = {
        (trace_id, root_turns, tuple(sorted(children.items())))
        for trace_id, (root_turns, children) in FANOUT_EXPECTED.items()
    }
    assert signatures[0] == signatures[1] == expected_signatures, (
        f"play structure differs or deviates from spec:\n"
        f"run1={signatures[0]}\nrun2={signatures[1]}\nexpected={expected_signatures}"
    )
    assert conversation_ids[0] == conversation_ids[1], (
        f"executed session-id sets differ across identical runs:\n"
        f"{conversation_ids[0] ^ conversation_ids[1]}"
    )
