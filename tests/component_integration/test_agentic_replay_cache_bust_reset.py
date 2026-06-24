# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: cache-bust markers survive ``reset_context`` turns.

Validates the reset-semantics fix on the actual wire. A weka trace fixture is
crafted so that turn 1 is a non-monotonic LCP cut -> the loader emits it with
``reset_context=True``. Under ``reset_context`` the endpoint's ``build_messages``
discards the accumulated turn-0 prefix and restarts the wire payload from the
reset turn, so the reset turn becomes a brand-new prefix.

The test asserts two things together:

1. ACTUAL MARKERS: every profiling session carries exactly one ``[rid:HEX]``
   cache-bust marker in its wire payload (markers are really injected, not
   silently dropped), and no wire message ever carries more than one (no
   stacking).

2. RESET SEMANTICS: the reset-turn requests (``turn_index == 1`` for these
   two-turn traces) carry the marker on their first user message -- the new
   post-reset prefix. Before the reset fix the marker landed on the discarded
   turn 0, leaving the reset turn's wire prefix unmarked; this asserts the
   marker follows the effective prefix across the cut.

A deterministic loader-level check (no benchmark) first proves the fixture
genuinely produces a ``reset_context`` turn, so the end-to-end assertions are
exercising the reset path rather than passing vacuously.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.component_integration.conftest import ComponentIntegrationTestDefaults
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

defaults = ComponentIntegrationTestDefaults
_MODEL = "claude-opus-4-5-20251101"
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
    """Write weka traces whose turn 1 is a non-monotonic LCP cut (reset_context).

    Turn 0 establishes a single 5-block user segment ``[1,2,3,4,5]``. Turn 1's
    ``hash_ids`` share only ``[1,2]`` (LCP=2), landing inside the previously
    emitted segment -> the reconstructor records a disturbance on an emitted
    segment and flags ``reset_context=True`` (see ConversationReconstructor.
    turn_delta case 3). Two turns => agentic_replay picks k_i=0 and resumes
    profiling at turn 1, so the reset turn is the profiling turn.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, num_traces + 1):
        # Distinct post-cut blocks per trace keep hash_ids varied across traces.
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


def _loader_user_config() -> MagicMock:
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
    return uc


def _fixture_produces_reset(trace_file: Path) -> bool:
    """Deterministically load one trace file and report whether any turn carries
    ``reset_context=True`` (proves the fixture exercises the reset path)."""
    loader = WekaTraceLoader(
        filename=str(trace_file), user_config=_loader_user_config()
    )
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


def _build_cmd(weka_dir: Path) -> str:
    return f"""
        aiperf profile
            --model claude-haiku-4-5-20251001
            --model {_MODEL}
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
            --cache-bust first_turn_prefix
            --export-level raw
    """


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


@pytest.fixture
def weka_reset_dir(tmp_path: Path) -> Path:
    return _write_reset_fixture(tmp_path / "weka_reset", num_traces=6)


def test_reset_fixture_actually_produces_reset_context(weka_reset_dir: Path):
    """Guard: the crafted fixture genuinely yields a reset_context turn, so the
    end-to-end test below is exercising the reset path (not passing vacuously)."""
    trace_file = next(weka_reset_dir.glob("*.json"))
    assert _fixture_produces_reset(trace_file), (
        "fixture no longer triggers reset_context — the LCP-cut shape or the "
        "reconstructor's reset rule changed; the end-to-end test would be vacuous"
    )


def test_cache_bust_marker_survives_reset_context(
    cli: AIPerfCLI,
    weka_reset_dir: Path,
):
    cmd = _build_cmd(weka_reset_dir)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed: stderr=\n{result.stderr}"
        f"\nlog tail=\n{(result.log or '')[-2000:]}"
    )
    assert result.raw_records is not None and len(result.raw_records) > 0, (
        "raw records JSONL must be present and non-empty"
    )

    by_session: dict[str, list] = defaultdict(list)
    reset_turn_records: list = []
    for rec in result.raw_records:
        if rec.metadata.benchmark_phase != "profiling":
            continue
        # No wire message may carry a stacked marker, ever.
        assert _max_rids_in_any_message(_payload_dict(rec)) <= 1, (
            f"stacked rid markers in a single message: "
            f"conv={rec.metadata.conversation_id} ti={rec.metadata.turn_index}"
        )
        xcorr = rec.metadata.x_correlation_id
        if xcorr is not None:
            by_session[xcorr].append(rec)
        # turn 1 is the reset turn for these two-turn traces.
        if rec.metadata.turn_index == 1:
            reset_turn_records.append(rec)

    # ACTUAL MARKERS: every profiling session carries exactly one rid.
    session_rids: list[str] = []
    for xcorr, records in by_session.items():
        rids: set[str] = set()
        for rec in records:
            fu = _first_user_content(_payload_dict(rec)) or ""
            m = _RID_RE.search(fu)
            if m:
                rids.add(m.group(0))
        assert len(rids) == 1, (
            f"session={xcorr}: expected exactly one rid across "
            f"{len(records)} turns; got {rids}"
        )
        session_rids.append(next(iter(rids)))

    assert len(session_rids) >= 2, (
        f"need >=2 marked sessions for a meaningful run; got {len(session_rids)}"
    )
    # Cross-session distinctness (collision-free per play/lane/trace).
    assert len(set(session_rids)) == len(session_rids), (
        f"marker collision across sessions: {session_rids}"
    )

    # RESET SEMANTICS: the reset-turn (turn 1) requests carry the marker on the
    # post-reset prefix. Without the reset fix the marker would land on the
    # discarded turn 0 and these would be unmarked.
    assert reset_turn_records, "expected at least one reset-turn (turn_index==1) record"
    for rec in reset_turn_records:
        fu = _first_user_content(_payload_dict(rec))
        assert fu is not None and _RID_RE.search(fu), (
            f"reset turn wire prefix is unmarked (marker lost across reset): "
            f"conv={rec.metadata.conversation_id} first_user={fu!r}"
        )
