# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic content slots over the full multiprocess stack.

The live E2E counterpart to `tests/unit/graph/test_dynamic_slots_e2e.py`:
a native two-node graph (`plan` -> `review`, where `review` splices `plan`'s
ACTUAL response via `@plan_out`) runs through real `aiperf profile` against the
in-repo mock server, with two workers. It proves the dynamic-content pipeline
wires end-to-end over real ZMQ / multiprocess -- capture on one credit, sticky
routing to the same worker, and splice on the next -- not just in-process.

Asserted on the raw export:

* the `review` request's wire payload contains `plan`'s exact response text
  (the capture -> pool -> splice round-trip crossed process boundaries);
* both requests were handled by the SAME worker (per-trace sticky routing --
  dynamic content depends on it, so a break shows here first).
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_FIX_DIR = Path(__file__).parent / "fixtures" / "native_dynamic"
FIXTURE = _FIX_DIR / "planner_reviewer.yaml"
CHAIN_FIXTURE = _FIX_DIR / "accumulate_chain.yaml"

_PLAN_PROMPT = "Make a plan for testing a cache."
_REVIEW_PREFIX = "Review this plan: "


def _user_content(record) -> str:
    messages = record.payload.get("messages", [])
    assert messages, f"record has no messages: {record.payload}"
    content = messages[-1].get("content")
    assert isinstance(content, str), f"expected string content, got {content!r}"
    return content


def _assistant_content(record) -> str:
    """Extract the assistant text the mock returned (non-streaming chat body)."""
    assert record.responses, f"record has no responses: {record.metadata}"
    body = orjson.loads(record.responses[0].text)
    return body["choices"][0]["message"]["content"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestDynamicSlotsMockE2E:
    async def test_review_prompt_contains_plan_response(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Dynamic slots are incompatible with the t* snapshot window; keep it off.
        # 0.0/0.0 is now the default for every run; pinned explicitly here so this
        # test never depends on the ambient default.

        result = await cli.run(
            f"""
            aiperf profile \
                --model mock-model \
                --url {aiperf_mock_server.url} \
                --input-file {FIXTURE} \
                --graph-format native \
                --endpoint-type chat \
                --tokenizer builtin \
                --random-seed 7 \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]

        raw = result.raw_records
        assert raw is not None and len(raw) == 2, (
            f"expected 2 raw records (plan + review), got "
            f"{None if raw is None else len(raw)}"
        )

        plan = next(r for r in raw if _PLAN_PROMPT in _user_content(r))
        review = next(r for r in raw if _user_content(r).startswith(_REVIEW_PREFIX))
        assert plan is not review

        # The producer actually returned non-empty content...
        plan_response = _assistant_content(plan)
        assert plan_response, "mock returned empty plan content"

        # ...and the consumer's wire prompt spliced it verbatim after the static
        # instruction: capture -> pool -> splice survived the process boundary.
        assert _user_content(review) == f"{_REVIEW_PREFIX}{plan_response}"

        # Dynamic content requires per-trace sticky routing: both credits of the
        # one trace instance must have hit the same worker (else the pool value
        # would have been missing and the trace would have errored).
        assert plan.metadata.worker_id is not None
        assert plan.metadata.worker_id == review.metadata.worker_id

        # No trace errored (a pool miss would surface as a record error).
        assert all(r.error is None for r in raw), [
            r.error for r in raw if r.error is not None
        ]

    async def test_accumulate_chain_reconstructs_full_alternation(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 3-turn accumulate chain: turn3's wire prompt reconstructs the full
        user/assistant alternation of turns 1-2, with the earlier assistant
        messages being their ACTUAL live replies -- over the real stack."""

        result = await cli.run(
            f"""
            aiperf profile \
                --model mock-model \
                --url {aiperf_mock_server.url} \
                --input-file {CHAIN_FIXTURE} \
                --graph-format native \
                --endpoint-type chat \
                --tokenizer builtin \
                --random-seed 11 \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]

        raw = result.raw_records
        assert raw is not None and len(raw) == 3, (
            f"expected 3 records, got {None if raw is None else len(raw)}"
        )
        by_last_user = {_user_content(r): r for r in raw}
        turn1 = by_last_user["Name a primary color."]
        turn2 = by_last_user["Name a farm animal."]
        turn3 = by_last_user["Combine your two answers."]

        r1 = _assistant_content(turn1)
        r2 = _assistant_content(turn2)
        assert r1 and r2, "mock returned empty content"

        # turn3's wire prompt is the full reconstructed conversation: the seed,
        # each authored user turn, and each prior turn's LIVE reply, in order.
        assert turn3.payload["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Name a primary color."},
            {"role": "assistant", "content": r1},
            {"role": "user", "content": "Name a farm animal."},
            {"role": "assistant", "content": r2},
            {"role": "user", "content": "Combine your two answers."},
        ]

        # All three credits pinned to one worker (per-trace sticky), no errors.
        workers = {r.metadata.worker_id for r in raw}
        assert len(workers) == 1 and None not in workers
        assert all(r.error is None for r in raw)
