# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial end-to-end integration tests for weka flattened-agent LCP splitting."""

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

DETECTED_LINE_RE = re.compile(
    r"Trace (\S+): detected (\d+) agents "
    r"\((\d+) seams merged, (\d+) spawned chains"
    r"(?: \[[^\]]*\])?, (\d+) empty-hash kept on main\)"
)
SPLIT_SUMMARY_RE = re.compile(
    r"flattened-agent detection split (\d+) trace\(s\) into "
    r"(\d+) extra agent chain\(s\)"
)

FANOUT_EXPECTED: dict[str, tuple[int, dict[str, int]]] = {
    "trace_alpha": (3, {"fa:000": 2, "fa:001": 1}),
    "trace_beta": (2, {"fa:000": 1, "fa:001": 1}),
    "trace_gamma": (3, {"fa:000": 2}),
}
FANOUT_EXPECTED_DETECTION: set[tuple[str, int, int, int, int]] = {
    ("trace_alpha", 3, 0, 2, 0),
    ("trace_beta", 3, 0, 2, 0),
    ("trace_gamma", 2, 0, 1, 0),
}
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
) -> dict:
    """One hash-aligned top-level normal request (in == len(hash_ids) * 64)."""
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": len(hash_ids) * BLOCK_SIZE,
        "out": out,
        "hash_ids": hash_ids,
        "api_time": api_time,
    }


def _write_trace(target_dir: Path, trace_id: str, requests: list[dict]) -> None:
    """Write one weka trace file with its requests sorted by recorded ``t``."""
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
    """Three fan-out traces with overlapping same-namespace worker chains."""
    _write_trace(
        target_dir,
        "trace_alpha",
        [
            _req(0.00, [1, 2, 3], api_time=0.12),
            _req(0.25, [1, 2, 50, 51], api_time=0.75),
            _req(0.30, [1, 2, 60, 61], api_time=0.50),
            _req(1.05, [1, 2, 50, 51, 52], api_time=0.15),
            _req(1.10, [1, 2, 3, 4], api_time=0.12),
            _req(1.75, [1, 2, 3, 4, 5], api_time=0.12),
        ],
    )
    _write_trace(
        target_dir,
        "trace_beta",
        [
            _req(0.00, [201, 202, 203], api_time=0.10),
            _req(0.20, [201, 202, 210, 211], api_time=0.40),
            _req(0.25, [201, 202, 220, 221], api_time=0.40),
            _req(1.00, [201, 202, 203, 204], api_time=0.10),
        ],
    )
    _write_trace(
        target_dir,
        "trace_gamma",
        [
            _req(0.00, [301, 302, 303], api_time=0.10),
            _req(0.20, [301, 302, 310], api_time=0.20),
            _req(0.50, [301, 302, 310, 311], api_time=0.20),
            _req(1.00, [301, 302, 303, 304], api_time=0.10),
            _req(1.60, [301, 302, 303, 304, 305], api_time=0.10),
        ],
    )
    return target_dir


def _write_join_trace(target_dir: Path) -> Path:
    """One trace where a 5-turn worker chain must gate the second main turn."""
    requests = [_req(0.0, [1, 2, 3], api_time=0.05)]
    chain = [1, 2, 9]
    for k in range(5):
        requests.append(_req(0.1 * (k + 1), list(chain), api_time=0.05))
        chain = [*chain, 10 + k]
    requests.append(_req(1.0, [1, 2, 3, 4], api_time=0.05))
    _write_trace(target_dir, "trace_join", requests)
    return target_dir


def _write_background_trace(target_dir: Path) -> Path:
    """One trace whose worker chain ends after the last main turn."""
    _write_trace(
        target_dir,
        "trace_bg",
        [
            _req(0.0, [1, 2, 3], api_time=0.05),
            _req(0.2, [1, 2, 70], api_time=0.10),
            _req(0.4, [1, 2, 3, 4], api_time=0.05),
            _req(0.6, [1, 2, 70, 71], api_time=0.10),
        ],
    )
    return target_dir


def _write_poisoned_corpus(target_dir: Path) -> Path:
    """Two traces of 10 mutually disjoint single-block-hash requests each."""
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
    """Run one full ``aiperf profile`` subprocess over a weka trace directory."""
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
        "AIPERF_DATASET_MMAP_CACHE_ENABLED": "false",
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


async def test_fanout_dir_splits_and_replays_recorded_concurrency(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """Spec sections 4/5: a directory of flattened fan-out traces is split into"""
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

    assert _sibling_children_overlap(plays), (
        "no two sibling ::fa: sessions were in flight concurrently -- "
        "the split did not reproduce the recorded fan-out concurrency"
    )

    branch_stats = result.json.branch_stats
    assert branch_stats is not None, "branch_stats missing for a DAG-shaped run"
    assert branch_stats.children_spawned >= 5, branch_stats
    assert branch_stats.children_completed >= 5, branch_stats
    assert branch_stats.children_errored == 0, branch_stats


async def test_spawn_join_gates_main_turn_until_worker_chain_completes(
    tmp_path: Path, mock_server_factory: MockServerFactory
) -> None:
    """Spec section 5.3: the gated main turn must not be dispatched until the"""
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
            continue
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


async def test_background_worker_chain_runs_after_root_and_drains_cleanly(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Spec section 5.3 (background): a chain ending after the last main turn"""
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
    assert branch_stats.parents_suspended == 0, (
        f"background-only trace must never suspend the parent: {branch_stats}"
    )


async def test_split_disabled_env_restores_legacy_single_stream(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Spec section 6: AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false must"""
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
    """Whatever detection decides about nonce-poisoned hashes, the run must"""
    corpus = _write_poisoned_corpus(tmp_path / "traces")
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
    """With the nonce-poison guard removed, a corpus of mutually-disjoint"""
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
    log_text = _combined_log_text(result).lower()
    assert "nonce" not in log_text, (
        "the nonce-poison guard was removed; no nonce-poison warning expected"
    )


async def test_two_identical_runs_produce_identical_split_structure(
    tmp_path: Path, aiperf_mock_server: AIPerfMockServer
) -> None:
    """Spec: detection is deterministic. Two identical runs must produce the"""
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
