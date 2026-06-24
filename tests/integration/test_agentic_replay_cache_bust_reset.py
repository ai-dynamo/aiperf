# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REAL integration test: cache-bust markers survive reset_context turns.

Unlike the ``component_integration`` counterpart (single process,
FakeCommunication), this spins up the full ``aiperf`` subprocess against the
shared mock server over real ZMQ + real workers + real HTTP, so it also covers
the serialization / multiprocess session paths.

A weka trace fixture is crafted so turn 1 is a non-monotonic LCP cut, which the
loader emits with ``reset_context=True``. Two-turn traces => agentic_replay
resumes profiling at turn 1, making the reset turn the profiling turn. The run
asserts that cache-bust markers are actually injected into the wire payload and
land on the post-reset prefix (the reset-semantics fix), with no stacking.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_MODEL = "claude-opus-4-5-20251101"
_TOKENIZER = "openai/gpt-oss-120b"  # pre-cached + offline in integration conftest
_BLOCK_SIZE = 16
_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _req(t: float, hash_ids: list[int], in_tokens: int) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": _MODEL,
        "in": in_tokens,
        "out": 8,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": 0.05,
        "think_time": 0.0,
    }


def _write_reset_fixture(target_dir: Path, *, num_traces: int = 6) -> Path:
    """Weka traces whose turn 1 is a non-monotonic LCP cut (reset_context=True).

    Turn 0 is a single 5-block user segment ``[1,2,3,4,5]``; turn 1 shares only
    ``[1,2]`` (LCP=2), landing inside the emitted segment -> reset. See the
    component-integration counterpart for the full rationale.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, num_traces + 1):
        cut_a, cut_b = 100 + 2 * n, 101 + 2 * n
        trace = {
            "id": f"reset_trace_{n:02d}",
            "models": [_MODEL],
            "block_size": _BLOCK_SIZE,
            "hash_id_scope": "local",
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                _req(0.0, [1, 2, 3, 4, 5], 5 * _BLOCK_SIZE),
                _req(1.0, [1, 2, cut_a, cut_b], 4 * _BLOCK_SIZE),
            ],
        }
        (target_dir / f"reset_trace_{n:02d}.json").write_text(json.dumps(trace))
    return target_dir


def _fixture_produces_reset(trace_file: Path) -> bool:
    """Deterministically confirm the fixture yields a reset_context turn."""
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = [_MODEL]
    uc.loadgen.inter_turn_delay_cap_seconds = None

    loader = WekaTraceLoader(filename=str(trace_file), user_config=uc)
    loader.synthesize_prompts_from_hash_ids = lambda rs: {r.key: "p" for r in rs}
    pg = MagicMock()
    pg._corpus_size = 10000
    pg._tokenized_corpus = list(range(10000))
    pg.tokenizer.decode = lambda tokens: f"decoded-{len(tokens)}"
    loader.prompt_generator = pg
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = _BLOCK_SIZE
    convs = loader.convert_to_conversations(loader.load_dataset())
    return any(t.reset_context for c in convs for t in c.turns)


def _payload_dict(record) -> dict:
    if record.payload is not None:
        return record.payload
    if record.payload_bytes is not None:
        return json.loads(record.payload_bytes)
    return {}


def _first_user_content(payload: dict) -> str | None:
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _max_rids_in_any_message(payload: dict) -> int:
    counts = [0]
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            counts.append(len(_RID_RE.findall(msg["content"])))
    return max(counts)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agentic_replay_cache_bust_marker_survives_reset_real_subprocess(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
):
    weka_dir = _write_reset_fixture(tmp_path / "weka_reset", num_traces=6)

    # Guard: the fixture genuinely produces a reset_context turn, so the
    # end-to-end assertions are not vacuous.
    trace_file = next(weka_dir.glob("*.json"))
    assert _fixture_produces_reset(trace_file), (
        "fixture no longer triggers reset_context; the reset path is not exercised"
    )

    result = await cli.run(
        f"""
        aiperf profile \
            --model claude-haiku-4-5-20251001 \
            --model {_MODEL} \
            --url {aiperf_mock_server.url} \
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
            --cache-bust first_turn_prefix \
            --export-level raw \
            --ui simple
        """,
        timeout=300.0,
    )

    assert result.raw_records is not None and len(result.raw_records) > 0, (
        "profile_export_raw.jsonl must exist and be non-empty\n"
        f"{(result.log or '')[-2000:]}"
    )

    by_session: dict[str, list] = defaultdict(list)
    reset_turn_records: list = []
    for rec in result.raw_records:
        if rec.metadata.benchmark_phase != "profiling":
            continue
        assert _max_rids_in_any_message(_payload_dict(rec)) <= 1, (
            f"stacked rid markers: conv={rec.metadata.conversation_id} "
            f"ti={rec.metadata.turn_index}"
        )
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None:
            by_session[xcorr].append(rec)
        if rec.metadata.turn_index == 1:
            reset_turn_records.append(rec)

    # Actual markers: one distinct rid per profiling session.
    session_rids: list[str] = []
    for xcorr, records in by_session.items():
        rids: set[str] = set()
        for rec in records:
            m = _RID_RE.search(_first_user_content(_payload_dict(rec)) or "")
            if m:
                rids.add(m.group(0))
        assert len(rids) == 1, f"session={xcorr}: expected exactly one rid; got {rids}"
        session_rids.append(next(iter(rids)))

    assert len(session_rids) >= 2, f"need >=2 marked sessions; got {len(session_rids)}"
    assert len(set(session_rids)) == len(session_rids), (
        f"marker collision across sessions: {session_rids}"
    )

    # Reset semantics: reset-turn (turn 1) requests carry the marker on the
    # post-reset prefix. Without the reset fix these would be unmarked.
    assert reset_turn_records, "expected at least one reset-turn (turn_index==1) record"
    for rec in reset_turn_records:
        fu = _first_user_content(_payload_dict(rec))
        assert fu is not None and _RID_RE.search(fu), (
            f"reset turn wire prefix unmarked (marker lost across reset): "
            f"conv={rec.metadata.conversation_id} first_user={fu!r}"
        )
