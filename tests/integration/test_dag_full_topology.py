# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real end-to-end integration test for a full two-branch DAG topology.

This test spins up the full aiperf subprocess against the shared mock server,
runs a single root conversation through the DAG loader, and validates:

1. Count + session identity (5 requests, correlation ids line up with the
   topology: root, branch-a sibling, branch-b sibling).
2. Ordering: root completes before either child starts; siblings fire in
   parallel after the root.
3. Payload merge correctness under the pure-append + one-system-at-root rule:
   each wire-payload ``messages`` array is the parent accumulator followed
   verbatim by the turn's authored messages, with captured assistant turns
   interleaved between turns.
4. ``branch_stats`` lands in ``profile_export_aiperf.json`` with the expected
   children-spawned/completed/errored counts.

The shared ``aiperf_mock_server`` fixture in ``tests/integration/conftest.py``
drives all I/O; no orchestrator or credit-issuer mocking happens here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dag" / "full.dag.jsonl"


# --- Fixture content (literal strings the assertions grep for) -------------

ROOT_SYS = "root system prompt"
ROOT_USER = "root user prompt"

A0_USER_A = "branch-a turn-0 user message A"
A0_USER_B = "branch-a turn-0 user message B"

A1_USER_A = "branch-a turn-1 user message A"
A1_USER_B = "branch-a turn-1 user message B"

B0_USER = "branch-b turn-0 user message"
B1_USER = "branch-b turn-1 user message"


# --- Helpers ---------------------------------------------------------------


def _text_of(msg: dict) -> str | None:
    """Extract a string representation of a message content."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts) if parts else None
    return None


def _roles_contents(messages: list[dict]) -> list[tuple[str, str | None]]:
    return [(m.get("role"), _text_of(m)) for m in messages]


def _classify(record) -> str:
    """Identify a request by matching a unique literal from its payload."""
    msgs = record.payload.get("messages", [])
    joined = " || ".join(_text_of(m) or "" for m in msgs)
    if A1_USER_A in joined:
        return "branch-a-turn-1"
    if A0_USER_A in joined:
        return "branch-a-turn-0"
    if B1_USER in joined:
        return "branch-b-turn-1"
    if B0_USER in joined:
        return "branch-b-turn-0"
    if ROOT_USER in joined and A0_USER_A not in joined and B0_USER not in joined:
        return "root"
    raise AssertionError(f"Unclassifiable record payload: {joined!r}")


# --- Test ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestDagFullTopologyEndToEnd:
    """End-to-end DAG benchmark through the real aiperf subprocess."""

    async def test_full_dag_payload_merge_and_stats(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ):
        """Run the two-branch DAG topology and validate merges + stats."""
        assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen3-0.6B \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {FIXTURE} \
                --custom-dataset-type dag_jsonl \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
        )

        # -------------------------------------------------------------------
        # A. Count + session identity
        # -------------------------------------------------------------------
        assert result.raw_records is not None, (
            "profile_export_raw.jsonl must exist when --export-level raw is set"
        )
        assert len(result.raw_records) == 5, (
            f"Expected 5 raw records, got {len(result.raw_records)}: "
            f"{[r.payload.get('messages', [])[0] for r in result.raw_records]}"
        )

        by_kind: dict[str, list] = {}
        for rec in result.raw_records:
            by_kind.setdefault(_classify(rec), []).append(rec)

        assert set(by_kind) == {
            "root",
            "branch-a-turn-0",
            "branch-a-turn-1",
            "branch-b-turn-0",
            "branch-b-turn-1",
        }, f"Unexpected record kinds: {set(by_kind)}"

        root_rec = by_kind["root"][0]
        a0 = by_kind["branch-a-turn-0"][0]
        a1 = by_kind["branch-a-turn-1"][0]
        b0 = by_kind["branch-b-turn-0"][0]
        b1 = by_kind["branch-b-turn-1"][0]

        root_corr = root_rec.metadata.x_correlation_id
        branch_a_corr = a0.metadata.x_correlation_id
        branch_b_corr = b0.metadata.x_correlation_id

        assert root_corr is not None
        assert branch_a_corr is not None
        assert branch_b_corr is not None
        assert len({root_corr, branch_a_corr, branch_b_corr}) == 3

        assert a1.metadata.x_correlation_id == branch_a_corr
        assert b1.metadata.x_correlation_id == branch_b_corr

        assert root_rec.metadata.parent_correlation_id is None
        for rec in (a0, a1, b0, b1):
            assert rec.metadata.parent_correlation_id == root_corr

        assert root_rec.metadata.agent_depth == 0
        for rec in (a0, a1, b0, b1):
            assert rec.metadata.agent_depth == 1

        # -------------------------------------------------------------------
        # B. Ordering (fork after root)
        # -------------------------------------------------------------------
        assert root_rec.metadata.request_end_ns <= a0.metadata.request_start_ns
        assert root_rec.metadata.request_end_ns <= b0.metadata.request_start_ns
        assert a0.metadata.request_end_ns <= a1.metadata.request_start_ns
        assert b0.metadata.request_end_ns <= b1.metadata.request_start_ns

        sibling_skew_ns = abs(
            a0.metadata.request_start_ns - b0.metadata.request_start_ns
        )
        assert sibling_skew_ns < 2_000_000_000

        # -------------------------------------------------------------------
        # C. Payload merge correctness — pure append, one system at root
        # -------------------------------------------------------------------
        def _assert_messages(
            rec,
            expected: list[tuple[str, str | None]],
            label: str,
        ) -> None:
            got = _roles_contents(rec.payload.get("messages", []))
            assert len(got) == len(expected), (
                f"{label}: expected {len(expected)} messages, got {len(got)}: {got!r}"
            )
            for i, ((exp_role, exp_content), (g_role, g_content)) in enumerate(
                zip(expected, got, strict=True)
            ):
                assert g_role == exp_role, (
                    f"{label}[{i}] role: expected {exp_role!r}, got {g_role!r}"
                )
                if exp_content is None:
                    assert g_content is not None and len(g_content) > 0, (
                        f"{label}[{i}]: assistant content must be non-empty"
                    )
                else:
                    assert g_content == exp_content, (
                        f"{label}[{i}] content: expected {exp_content!r}, "
                        f"got {g_content!r}"
                    )

        # Root: verbatim from fixture (accumulator is empty).
        _assert_messages(
            root_rec,
            [("system", ROOT_SYS), ("user", ROOT_USER)],
            "root",
        )

        # branch-a turn 0: root accumulator + captured root response + A0 users.
        _assert_messages(
            a0,
            [
                ("system", ROOT_SYS),
                ("user", ROOT_USER),
                ("assistant", None),
                ("user", A0_USER_A),
                ("user", A0_USER_B),
            ],
            "branch-a turn 0",
        )

        # branch-a turn 1: a0 accumulator + captured a0 response + A1 users.
        _assert_messages(
            a1,
            [
                ("system", ROOT_SYS),
                ("user", ROOT_USER),
                ("assistant", None),
                ("user", A0_USER_A),
                ("user", A0_USER_B),
                ("assistant", None),
                ("user", A1_USER_A),
                ("user", A1_USER_B),
            ],
            "branch-a turn 1",
        )

        # branch-b turn 0: root accumulator + captured root response + B0 user.
        _assert_messages(
            b0,
            [
                ("system", ROOT_SYS),
                ("user", ROOT_USER),
                ("assistant", None),
                ("user", B0_USER),
            ],
            "branch-b turn 0",
        )

        # branch-b turn 1: b0 accumulator + captured b0 response + B1 user.
        _assert_messages(
            b1,
            [
                ("system", ROOT_SYS),
                ("user", ROOT_USER),
                ("assistant", None),
                ("user", B0_USER),
                ("assistant", None),
                ("user", B1_USER),
            ],
            "branch-b turn 1",
        )

        # -------------------------------------------------------------------
        # D. BranchStats in profile_export_aiperf.json
        # -------------------------------------------------------------------
        assert result.json is not None, "profile_export_aiperf.json must exist"
        assert result.json.branch_stats is not None
        assert result.json.branch_stats.children_spawned == 2
        assert result.json.branch_stats.children_completed == 2
        assert result.json.branch_stats.children_errored == 0

        # -------------------------------------------------------------------
        # E. Sticky routing: all 5 requests land on the same worker.
        # -------------------------------------------------------------------
        worker_ids = {rec.metadata.worker_id for rec in result.raw_records}
        assert len(worker_ids) == 1, (
            f"All 5 DAG requests must route to the same worker via sticky "
            f"routing; saw workers {worker_ids}"
        )
