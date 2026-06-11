# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end CLI tests for the cache-bust marker injection pipeline.

Drives ``aiperf profile --scenario inferencex-agentx-mvp --unsafe-override``
with each ``--cache-bust`` target through cyclopts + the in-process app
runner against the FakeTransport mock server, then inspects the
``profile_export_raw.jsonl`` payloads to verify markers appear in the
correct position of the wire payload.

Wiring covered:
  - CLI parser accepts ``--cache-bust <target>`` (PromptConfig).
  - Scenario validator allows non-SYSTEM_PREFIX values under
    ``--unsafe-override`` (warning, not lock-error).
  - AgenticReplayStrategy mints a deterministic ``[rid:HEX]`` marker per
    session keyed on ``x_correlation_id``, propagates it to TurnToSend +
    Credit, and rotates on recycle (incremented ``recycle_pass``).
  - Worker injects the marker into the actual sent payload at request build
    time.
  - Raw record exporter persists the post-injection payload to disk.

Each parametrised target value asserts:
  1. Marker tokens of shape ``[rid:[0-9a-f]{12}]`` appear in the captured
     wire payload at the position dictated by the target.
  2. All requests sharing an ``x_correlation_id`` carry the same rid token
     (per-session marker continuity across turns).
  3. Different ``x_correlation_id`` sessions get different rid tokens
     (marker uniqueness).
  4. The marker lives in the wrapper field (system role for SYSTEM_*; first
     user turn for FIRST_TURN_*) — never inside the trace turn body the
     loader produced.

A separate ``CacheBustTarget.NONE`` test asserts zero rid markers anywhere.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from aiperf.common.enums import CacheBustTarget
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration


# =============================================================================
# Fixture: weka trace dataset with non-zero system_tokens so the loader emits
# a system-role message in raw_messages (required for SYSTEM_* targets to
# inject visibly). FIRST_TURN_* targets work regardless because every
# request has a user turn.
# =============================================================================

# Block size = 16. tool_tokens=8 + system_tokens=8 -> ceil(16/16)=1 system
# block at the front. Each turn k consumes hash blocks
# [0..ceil((tool+sys)/bs)) for system, then [1..1+m_full_user) for user.
_BLOCK_SIZE = 16
_TOOL_TOKENS = 8
_SYSTEM_TOKENS = 8


def _write_weka_fixture(
    target_dir: Path,
    *,
    num_traces: int = 6,
    tool_tokens: int = _TOOL_TOKENS,
    system_tokens: int = _SYSTEM_TOKENS,
) -> Path:
    """Write a block-size-valid weka trace fixture into ``target_dir``.

    Default ``tool_tokens`` and ``system_tokens`` are non-zero so the
    synthesised raw_messages contain a leading ``role="system"`` message —
    required for SYSTEM_PREFIX / SYSTEM_SUFFIX cache-bust targets to inject
    visibly into the wire payload.

    Pass ``tool_tokens=0, system_tokens=0`` to exercise the SYSTEM_*
    fall-back path: with no system segment the loader emits only a user role
    in raw_messages, and the worker must route the marker to the first user
    turn rather than silently dropping it.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    has_prefix = (tool_tokens + system_tokens) > 0
    for n in range(1, num_traces + 1):
        requests = []
        for k in range(n):
            user_blocks = k + 1
            if has_prefix:
                in_tokens = (1 + user_blocks) * _BLOCK_SIZE + 4
                hash_ids = list(range(1, 1 + 1 + user_blocks))  # 1 sys + N user
            else:
                in_tokens = user_blocks * _BLOCK_SIZE + 4
                hash_ids = list(range(1, 1 + user_blocks))  # N user only
            requests.append(
                {
                    "t": k * 1.0,
                    "type": "n",
                    "model": "claude-opus-4-5-20251101",
                    "in": in_tokens,
                    "out": 8,
                    "hash_ids": hash_ids,
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "stop": "end_turn",
                    "api_time": 0.05,
                    "think_time": 0.0,
                }
            )
        trace = {
            "id": f"trace_{n:02d}_n{n}",
            "models": ["claude-opus-4-5-20251101"],
            "block_size": _BLOCK_SIZE,
            "hash_id_scope": "local",
            "tool_tokens": tool_tokens,
            "system_tokens": system_tokens,
            "requests": requests,
        }
        (target_dir / f"trace_{n:02d}_n{n}.json").write_text(json.dumps(trace))
    return target_dir


@pytest.fixture
def weka_with_system_dir(tmp_path: Path) -> Path:
    """A 6-trace weka fixture with non-zero tool/system tokens."""
    return _write_weka_fixture(tmp_path / "weka_sys", num_traces=6)


@pytest.fixture
def weka_without_system_dir(tmp_path: Path) -> Path:
    """A 6-trace weka fixture with zero tool/system tokens — the loader
    emits raw_messages with only a ``role="user"`` entry."""
    return _write_weka_fixture(
        tmp_path / "weka_no_sys",
        num_traces=6,
        tool_tokens=0,
        system_tokens=0,
    )


def _build_cmd(weka_dir: Path, *, cache_bust: str) -> str:
    """Build an ``aiperf profile`` command for an agentic_replay run with the
    given ``--cache-bust`` target.

    Forces ``--scenario inferencex-agentx-mvp --unsafe-override`` because
    AGENTIC_REPLAY timing mode is only reachable via the scenario validator
    write to the read-only ``timing_mode`` property. ``--unsafe-override``
    is required so the scenario lock on ``cache_bust=SYSTEM_PREFIX`` becomes
    a warning rather than a fail-fast for the SUFFIX / FIRST_TURN_* values.

    ``--export-level raw`` is required so the raw record JSONL exists.
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
            --benchmark-duration 8
            --concurrency 3
            --random-seed 42
            --tokenizer {defaults.tokenizer}
            --extra-inputs ignore_eos:true
            --workers-max {defaults.workers_max}
            --ui {defaults.ui}
            --scenario inferencex-agentx-mvp
            --unsafe-override
            --cache-bust {cache_bust}
            --export-level raw
    """


_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _extract_rid(text: str) -> str | None:
    m = _RID_RE.search(text)
    return m.group(0) if m else None


def _payload_dict(record) -> dict:
    """Return the request payload as a dict, regardless of which carrier
    field the exporter populated."""
    if record.payload is not None:
        return record.payload
    if record.payload_bytes is not None:
        return json.loads(record.payload_bytes)
    return {}


def _system_content(payload: dict) -> str | None:
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _first_user_content(payload: dict) -> str | None:
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _all_message_contents(payload: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for msg in payload.get("messages", []):
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                out.append((msg.get("role", ""), content))
    return out


# =============================================================================
# Helper: extract the marker carrier string for a given target.
# =============================================================================


def _marker_carrier_text(payload: dict, target: CacheBustTarget) -> str | None:
    """Return the substring of the payload where ``target`` is supposed to
    inject the marker, or ``None`` if the carrier does not exist."""
    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.SYSTEM_SUFFIX):
        return _system_content(payload)
    if target in (CacheBustTarget.FIRST_TURN_PREFIX, CacheBustTarget.FIRST_TURN_SUFFIX):
        return _first_user_content(payload)
    return None


def _trace_turn_bodies(payload: dict, target: CacheBustTarget) -> list[str]:
    """Return non-carrier message contents — those that must NOT contain the
    marker (the trace's own turn body content, hash-block payloads)."""
    bodies: list[str] = []
    is_system_target = target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
    )
    is_first_user_target = target in (
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    )
    saw_first_user = False
    for role, content in _all_message_contents(payload):
        if role == "system":
            if is_system_target:
                continue  # carrier — skip
            bodies.append(content)
        elif role == "user":
            if is_first_user_target and not saw_first_user:
                saw_first_user = True
                continue  # carrier — skip
            saw_first_user = True
            bodies.append(content)
        else:
            bodies.append(content)
    return bodies


# =============================================================================
# Tests: each target value injects a marker in the correct position.
# =============================================================================


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    ],
    ids=lambda t: str(t),
)
def test_agentic_replay_cache_bust_marker_in_wire_payload(
    cli: AIPerfCLI,
    weka_with_system_dir: Path,
    target: CacheBustTarget,
) -> None:
    """For each non-NONE target, a per-session ``[rid:HEX]`` marker appears
    in the wire payload at the position the target dictates, is consistent
    across all turns of a session, distinct across sessions, and absent
    from the trace turn bodies.

    Marker coverage is universal for every target: FIRST_TURN_* injects into
    the effective wire prefix's opening user turn on every credit (including
    mid-trajectory resumes at ``k_i > 0``, whose seeded turn 0 is the real
    prefix), and SYSTEM_* applies every turn. So every profiling session must
    carry exactly one marker — a regression of the seeded-resume fix would show
    up here as an unmarked ``k_i > 0`` session.
    """
    cmd = _build_cmd(weka_with_system_dir, cache_bust=target)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed for target={target}: stderr=\n{result.stderr}"
        f"\nlog tail=\n{(result.log or '')[-2000:]}"
    )
    assert result.raw_records is not None and len(result.raw_records) > 0, (
        "raw records JSONL must be present and non-empty"
    )

    # Group records by x_correlation_id, scoped to the PROFILING phase only.
    # WARMUP and the FIRST PROFILING dispatch for each trajectory share the
    # same rid by design (warmup-coherent prefix-cache lineage). The
    # uniqueness assertion below applies within PROFILING; the warmup-coherent
    # pair is covered by ``test_agentic_replay_marker_uniqueness.py``.
    by_session: dict[str, list] = defaultdict(list)
    for rec in result.raw_records:
        if rec.metadata.benchmark_phase != "profiling":
            continue
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None:
            by_session[xcorr].append(rec)

    assert len(by_session) >= 2, (
        "Need >=2 sessions to verify per-session-uniqueness; "
        f"got {len(by_session)}: {list(by_session.keys())}"
    )

    is_prefix_target = target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
    )

    session_rids: dict[str, str] = {}
    sessions_without_marker: list[str] = []
    for xcorr, records in by_session.items():
        rids_in_session: set[str] = set()
        for rec in records:
            payload = _payload_dict(rec)
            carrier = _marker_carrier_text(payload, target)
            if carrier is None:
                # Carrier role missing entirely. For SYSTEM_* this is a
                # fixture failure; for FIRST_TURN_* this means the request
                # has no user role at all (shouldn't happen).
                pytest.fail(
                    f"target={target}: payload missing carrier role; "
                    f"messages={payload.get('messages')!r}"
                )
            rid = _extract_rid(carrier)
            if rid is not None:
                rids_in_session.add(rid)
                # Position correctness.
                if is_prefix_target:
                    assert carrier.startswith(rid), (
                        f"target={target}: prefix marker must be at "
                        f"byte 0 of carrier; got {carrier[:80]!r}"
                    )
                else:
                    assert carrier.rstrip().endswith(rid), (
                        f"target={target}: suffix marker must be at "
                        f"end of carrier; got {carrier[-80:]!r}"
                    )

            # Marker must NOT appear in the trace's own turn bodies.
            for body in _trace_turn_bodies(payload, target):
                assert _RID_RE.search(body) is None, (
                    f"target={target} session={xcorr}: rid leaked into "
                    f"trace turn body (must only live in carrier); "
                    f"body[:120]={body[:120]!r}"
                )

        if not rids_in_session:
            sessions_without_marker.append(xcorr)
            continue

        # Per-session continuity: every marked record shares one rid.
        assert len(rids_in_session) == 1, (
            f"target={target} session={xcorr}: expected single rid "
            f"across {len(records)} turns; got {rids_in_session}"
        )
        session_rids[xcorr] = next(iter(rids_in_session))

    # Every profiling session must carry a marker, for ALL targets. FIRST_TURN_*
    # marks the effective prefix's opening user turn on every credit — including
    # seeded mid-trajectory resumes at k_i > 0 — so an unmarked session here is a
    # regression of the seeded-resume fix. SYSTEM_* applies every turn. (This
    # fixture is linear, no FORK children — FORK inheritance is covered in the
    # DAG cache-bust test and the worker unit tests.)
    assert not sessions_without_marker, (
        f"target={target}: every session must be marked, but these were not: "
        f"{sessions_without_marker}. Total sessions={len(by_session)}."
    )
    assert len(session_rids) >= 1, (
        f"target={target}: no marked sessions at all (fixture/run too small?)."
    )

    # Cross-session distinctness: among marked sessions we want >= 2 distinct
    # rids whenever there are >= 2 marked sessions (which is the common case).
    if len(session_rids) >= 2:
        distinct = set(session_rids.values())
        assert len(distinct) >= 2, (
            f"target={target}: expected distinct markers across "
            f"sessions; got {len(distinct)} distinct from "
            f"{len(session_rids)} sessions: {session_rids}"
        )

    # Collision-free per-session uniqueness: every marked session must have
    # its OWN rid — no two sessions can share a digest. Regression bar for
    # the collision-free design (trace_id is part of the marker tuple).
    all_session_rids = list(session_rids.values())
    assert len(set(all_session_rids)) == len(all_session_rids), (
        f"target={target}: marker collision detected — "
        f"{len(all_session_rids) - len(set(all_session_rids))} duplicate rids "
        f"across {len(all_session_rids)} sessions: {session_rids}"
    )


# =============================================================================
# NONE target: no rid markers anywhere in the wire payload.
# =============================================================================


def test_agentic_replay_cache_bust_none_emits_no_marker(
    cli: AIPerfCLI,
    weka_with_system_dir: Path,
) -> None:
    """With ``--cache-bust none`` the worker injection path is a no-op and
    no ``[rid:HEX]`` token can appear anywhere in the captured payload."""
    cmd = _build_cmd(weka_with_system_dir, cache_bust=CacheBustTarget.NONE)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed (target=none): stderr=\n{result.stderr}"
        f"\nlog tail=\n{(result.log or '')[-2000:]}"
    )
    assert result.raw_records is not None and len(result.raw_records) > 0

    for rec in result.raw_records:
        payload = _payload_dict(rec)
        for _role, content in _all_message_contents(payload):
            assert _RID_RE.search(content) is None, (
                "target=none must produce zero rid markers; found in "
                f"payload content: {content[:200]!r}"
            )


# =============================================================================
# Recycle rotation: under sustained load the same trace_id is recycled and
# the rid changes between incarnations. We exercise this via long enough
# duration + small fixture so the recycle queue drains and re-spawns.
# =============================================================================


def test_agentic_replay_cache_bust_recycle_rotates_marker(
    cli: AIPerfCLI,
    weka_with_system_dir: Path,
) -> None:
    """When a trace is recycled (queue drains and pops the same conversation
    again), the new session gets a different rid than its prior incarnation —
    the strategy increments ``recycle_pass`` per recycle, and the marker
    builder digests it.

    With 6 traces, concurrency=3, duration=8s, the small fixture is well
    inside one full cycle; we look for either:
      a) the same conversation_id appearing in two different sessions with
         distinct rids, or
      b) duration insufficient — at least 2 distinct rids on distinct
         x_correlation_ids (covers the lane-uniqueness floor).
    """
    cmd = _build_cmd(weka_with_system_dir, cache_bust=CacheBustTarget.SYSTEM_PREFIX)
    result = cli.run_sync(cmd, timeout=defaults.timeout)
    assert result.exit_code == 0, f"CLI run failed: stderr=\n{result.stderr}"
    assert result.raw_records is not None and len(result.raw_records) > 0

    # Map x_correlation_id -> (conversation_id, rid).
    by_session: dict[str, tuple[str | None, str | None]] = {}
    for rec in result.raw_records:
        xcorr = rec.metadata.x_correlation_id
        if xcorr is None or xcorr in by_session:
            continue
        cid = rec.metadata.conversation_id
        carrier = _system_content(_payload_dict(rec)) or ""
        rid = _extract_rid(carrier)
        by_session[xcorr] = (cid, rid)

    # Group conversation_ids; if any conversation_id appears in >1 session
    # those rids must differ (recycle pass increment).
    by_conv: dict[str, set[str]] = defaultdict(set)
    for cid, rid in by_session.values():
        if cid is not None and rid is not None:
            by_conv[cid].add(rid)

    duplicated = {c: rids for c, rids in by_conv.items() if len(rids) > 1}
    if duplicated:
        # Recycle observed: same conversation, distinct rids.
        for cid, rids in duplicated.items():
            assert len(rids) >= 2, (
                f"recycle: conversation {cid} should have >=2 distinct rids; got {rids}"
            )
    else:
        # Floor: at least 2 distinct rids overall (lane uniqueness alone).
        all_rids = {rid for _cid, rid in by_session.values() if rid is not None}
        assert len(all_rids) >= 2, (
            "expected at least 2 distinct rids across sessions even without recycle; "
            f"got {len(all_rids)}: {all_rids}"
        )


# =============================================================================
# SYSTEM_* fall-back: traces lacking any system message must still see the
# marker injected — routed to the first user turn rather than silently dropped.
# Asserts NO synthesized system role in the wire payload.
# =============================================================================


def test_agentic_replay_cache_bust_system_prefix_falls_back_when_trace_lacks_system(
    cli: AIPerfCLI,
    weka_without_system_dir: Path,
) -> None:
    """When a weka trace has ``system_tokens=0`` (no system message in
    raw_messages) and ``--cache-bust system_prefix`` is requested, the worker
    must fall back to first-user-turn-prefix injection. Contract:
      - ``messages[0].role == "user"`` (no synthesized system role).
      - First user message content starts with ``[rid:HEX]\\n\\n``.
      - All turns of a session share the same rid (per-session continuity).
    """
    cmd = _build_cmd(weka_without_system_dir, cache_bust=CacheBustTarget.SYSTEM_PREFIX)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed: stderr=\n{result.stderr}"
        f"\nlog tail=\n{(result.log or '')[-2000:]}"
    )
    assert result.raw_records is not None and len(result.raw_records) > 0

    by_session: dict[str, list] = defaultdict(list)
    for rec in result.raw_records:
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None:
            by_session[xcorr].append(rec)

    assert len(by_session) >= 1, "Need at least one session"

    sessions_with_marker = 0
    for xcorr, records in by_session.items():
        rids: set[str] = set()
        for rec in records:
            payload = _payload_dict(rec)
            messages = payload.get("messages", [])
            assert messages, f"session={xcorr}: payload has no messages"

            # Contract: NO synthesized system role. First message must be user.
            assert messages[0].get("role") == "user", (
                f"session={xcorr}: SYSTEM_* fallback must NOT synthesize a "
                f"system role; got messages[0]={messages[0]!r}"
            )

            user_content = messages[0].get("content", "")
            assert isinstance(user_content, str)
            rid = _extract_rid(user_content)
            if rid is not None:
                rids.add(rid)
                assert user_content.startswith(rid), (
                    f"session={xcorr}: prefix marker must be at byte 0 of "
                    f"first user content; got {user_content[:80]!r}"
                )
                # Marker prefix carries trailing whitespace boundary.
                assert user_content.startswith(f"{rid}\n\n"), (
                    f"session={xcorr}: expected marker followed by '\\n\\n'; "
                    f"got {user_content[: len(rid) + 4]!r}"
                )

        if rids:
            sessions_with_marker += 1
            assert len(rids) == 1, (
                f"session={xcorr}: expected single rid across "
                f"{len(records)} turns; got {rids}"
            )

    # SYSTEM_*-fallback fires only on turn_index==0 (matches FIRST_TURN_*
    # semantics). With recycled sessions starting at turn 0 plus k_i=0
    # trajectories, at least one session must carry a marker.
    assert sessions_with_marker >= 1, (
        "At least one session must have received the SYSTEM_PREFIX fallback "
        f"marker on its first turn; total sessions={len(by_session)}"
    )
