# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REAL integration tests for cache-bust markers through the full aiperf
subprocess (real ZMQ + real workers + real HTTP against the mock server).

The ``component_integration`` suite covers the same behaviors in-process via
FakeCommunication; these run the actual subprocess so they also exercise the
serialization and multiprocess session/worker paths. Covered here:

- Per-target marker injection on the wire (FIRST_TURN_* on the first user turn,
  SYSTEM_* on the system message), one marker per session, distinct across
  sessions (collision-free minting), never stacked.
- NONE target: no markers anywhere.
- SPAWN subagent fan-out: subagent children are independently busted with their
  OWN marker, distinct from the parent root's (the reachable production fan-out
  + cache-bust path; contrast with FORK, which inherits and is unreachable here).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from aiperf.common.enums import CacheBustTarget
from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_OPUS = "claude-opus-4-5-20251101"
_HAIKU = "claude-haiku-4-5-20251001"
_TOKENIZER = "openai/gpt-oss-120b"  # pre-cached + offline in integration conftest
_BLOCK_SIZE = 16
_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


# --- fixtures ---------------------------------------------------------------


def _req(t: float, hash_ids: list[int], in_tokens: int, *, model: str = _OPUS) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": in_tokens,
        "out": 8,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": 0.05,
        "think_time": 0.0,
    }


def _write_linear_fixture(target_dir: Path, *, num_traces: int = 6) -> Path:
    """Linear multi-turn weka traces with a system prefix (tool+system tokens)
    so SYSTEM_* targets have a system-role message to inject into."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, num_traces + 1):
        requests = []
        for k in range(max(2, n)):  # >=2 turns so agentic_replay can split
            user_blocks = k + 1
            in_tokens = (1 + user_blocks) * _BLOCK_SIZE + 4
            hash_ids = list(range(1, 1 + 1 + user_blocks))  # 1 sys + N user
            requests.append(_req(k * 1.0, hash_ids, in_tokens))
        trace = {
            "id": f"lin_trace_{n:02d}",
            "models": [_OPUS],
            "block_size": _BLOCK_SIZE,
            "hash_id_scope": "local",
            "tool_tokens": 8,
            "system_tokens": 8,
            "requests": requests,
        }
        (target_dir / f"lin_trace_{n:02d}.json").write_text(json.dumps(trace))
    return target_dir


def _write_subagent_fixture(target_dir: Path, *, num_traces: int = 5) -> Path:
    """Weka traces each carrying a ``type:subagent`` entry -> a SPAWN child."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_traces + 1):
        base = i * 10
        trace = {
            "id": f"sa_trace_{i:02d}",
            "models": [_OPUS],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": [
                _req(0.0, [base + 1, base + 2, base + 3], 200),
                {
                    "t": 2.0,
                    "type": "subagent",
                    "agent_id": f"agent_{i:03d}",
                    "subagent_type": "Explore",
                    "duration_ms": 3000,
                    "total_tokens": 500,
                    "tool_use_count": 2,
                    "status": "completed",
                    "requests": [
                        _req(0.0, [base + 100, base + 101], 100, model=_HAIKU),
                    ],
                    "models": [_HAIKU],
                    "tool_tokens": 20,
                    "system_tokens": 10,
                },
                _req(6.0, [base + 1, base + 2, base + 3, base + 4, base + 5], 400),
            ],
        }
        # subagent inner request stop must be a tool-using stop on the parent turn 0
        trace["requests"][0]["stop"] = "tool_use"
        trace["requests"][2]["input_types"] = ["tool_result"]
        (target_dir / f"sa_trace_{i:02d}.json").write_text(json.dumps(trace))
    return target_dir


# --- helpers ----------------------------------------------------------------


def _payload_dict(record) -> dict:
    if record.payload is not None:
        return record.payload
    if record.payload_bytes is not None:
        return json.loads(record.payload_bytes)
    return {}


def _content_of(payload: dict, role: str) -> str | None:
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == role:
            c = msg.get("content")
            return c if isinstance(c, str) else None
    return None


def _carrier_text(payload: dict, target: CacheBustTarget) -> str | None:
    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.SYSTEM_SUFFIX):
        return _content_of(payload, "system")
    return _content_of(payload, "user")


def _max_rids_in_any_message(payload: dict) -> int:
    counts = [0]
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            counts.append(len(_RID_RE.findall(msg["content"])))
    return max(counts)


def _build_cmd(weka_dir: Path, url: str, cache_bust: str) -> str:
    return f"""
        aiperf profile \
            --model {_HAIKU} \
            --model {_OPUS} \
            --url {url} \
            --endpoint-type chat \
            --streaming \
            --custom-dataset-type weka_trace \
            --input-file {weka_dir} \
            --no-fixed-schedule \
            --benchmark-duration 8 \
            --concurrency 3 \
            --random-seed 42 \
            --tokenizer {_TOKENIZER} \
            --extra-inputs ignore_eos:true \
            --workers-max 2 \
            --scenario inferencex-agentx-mvp \
            --unsafe-override \
            --cache-bust {cache_bust} \
            --export-level raw \
            --ui simple
    """


def _profiling_records(result) -> list:
    return [
        r
        for r in (result.raw_records or [])
        if r.metadata.benchmark_phase == "profiling"
    ]


# --- tests ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
    ],
)
async def test_marker_in_wire_payload_real_subprocess(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
    target: CacheBustTarget,
):
    """Every profiling session carries exactly one marker in the target's
    carrier, distinct across sessions (collision-free), never stacked."""
    weka_dir = _write_linear_fixture(tmp_path / f"lin_{target.value}")
    result = await cli.run(
        _build_cmd(weka_dir, aiperf_mock_server.url, target.value), timeout=300.0
    )
    records = _profiling_records(result)
    assert records, f"no profiling records\n{(result.log or '')[-1500:]}"

    by_session: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        payload = _payload_dict(rec)
        assert _max_rids_in_any_message(payload) <= 1, (
            f"target={target}: stacked markers in conv={rec.metadata.conversation_id}"
        )
        carrier = _carrier_text(payload, target) or ""
        m = _RID_RE.search(carrier)
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None and m:
            by_session[xcorr].add(m.group(0))

    assert by_session, f"target={target}: no markers found in any session carrier"
    for xcorr, rids in by_session.items():
        assert len(rids) == 1, f"target={target} session={xcorr}: multiple rids {rids}"
    all_rids = [next(iter(v)) for v in by_session.values()]
    assert len(set(all_rids)) == len(all_rids), (
        f"target={target}: marker collision across sessions: {all_rids}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_none_target_has_no_markers_real_subprocess(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """With --cache-bust none, no rid marker appears anywhere on the wire."""
    weka_dir = _write_linear_fixture(tmp_path / "lin_none")
    result = await cli.run(
        _build_cmd(weka_dir, aiperf_mock_server.url, "none"), timeout=300.0
    )
    records = _profiling_records(result)
    assert records, f"no profiling records\n{(result.log or '')[-1500:]}"
    for rec in records:
        for role, content in (
            (m.get("role"), m.get("content"))
            for m in _payload_dict(rec).get("messages", [])
            if isinstance(m, dict)
        ):
            if isinstance(content, str):
                assert not _RID_RE.search(content), (
                    f"NONE target leaked a marker into {role} content: {content[:80]!r}"
                )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spawn_subagent_children_busted_with_own_marker_real_subprocess(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    """SPAWN subagent children get their OWN cache-bust marker (busted), distinct
    from the parent root's marker -- the reachable production fan-out path. This
    exercises the non-FORK branch of the worker's cache-bust guard end-to-end.
    """
    weka_dir = _write_subagent_fixture(tmp_path / "sa", num_traces=5)
    result = await cli.run(
        _build_cmd(weka_dir, aiperf_mock_server.url, "first_turn_prefix"),
        timeout=300.0,
    )
    records = _profiling_records(result)
    assert records, f"no profiling records\n{(result.log or '')[-1500:]}"

    # rids seen on root (depth==0) sessions, grouped by base conversation id.
    root_rids_by_base: dict[str, set[str]] = defaultdict(set)
    child_records: list = []
    for rec in records:
        payload = _payload_dict(rec)
        assert _max_rids_in_any_message(payload) <= 1, (
            f"stacked markers in conv={rec.metadata.conversation_id}"
        )
        m = _RID_RE.search(_content_of(payload, "user") or "")
        conv = rec.metadata.conversation_id or ""
        if rec.metadata.agent_depth and "::sa:" in conv:
            child_records.append((rec, m.group(0) if m else None))
        elif m:
            root_rids_by_base[conv].add(m.group(0))

    assert child_records, (
        "expected SPAWN subagent child records (agent_depth>0, '::sa:' in conv id) "
        "-- the subagent fan-out did not materialize"
    )
    for rec, child_rid in child_records:
        conv = rec.metadata.conversation_id or ""
        base = conv.split("::sa:")[0]
        assert child_rid is not None, (
            f"SPAWN child {conv} carries no cache-bust marker (not busted)"
        )
        # The child's marker must be its OWN, not inherited from the parent root.
        assert child_rid not in root_rids_by_base.get(base, set()), (
            f"SPAWN child {conv} reused a parent-root marker {child_rid} "
            f"(should be independently busted): root rids={root_rids_by_base.get(base)}"
        )
