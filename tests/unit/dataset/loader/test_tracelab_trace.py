# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TraceLab trace loader and its conversion layer."""

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pytest import param

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader._tracelab_convert import (
    HashIdMinter,
    build_join_index,
    build_trace,
    group_children_by_parent,
    order_rounds,
    round_timing,
    safe_trace_id,
    session_span,
    synthesize_hash_ids,
)
from aiperf.dataset.loader.tracelab_trace import (
    DEFAULT_BLOCK_SIZE,
    TraceLabTraceDatasetLoader,
)
from aiperf.dataset.loader.weka_trace_models import WekaTrace

T0 = datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC)


def iso(offset_s: float) -> str:
    """Seconds after the fixture epoch as a TraceLab-style ISO-8601 stamp."""
    stamp = (T0 + timedelta(seconds=offset_s)).isoformat()
    return stamp.replace("+00:00", "Z")


def make_row(
    *,
    session_id: str = "claude:s1",
    round_index: int = 0,
    submitted: float = 0.0,
    responded: float | None = 1.0,
    input_tokens: int = 128,
    prefix_tokens: int = 0,
    output_tokens: int = 10,
    reasoning_tokens: int | None = None,
    model: str = "claude-opus-4-7",
    provider: str = "claude",
    user: str = "user_a",
    project: str = "project_a",
    first_input_event_type: str = "user_message",
    emits_tool_call: bool = False,
    tools: list[dict] | None = None,
    timing_events: list[dict] | None = None,
) -> dict:
    """One TraceLab JSONL row, with only the fields the loader reads."""
    if timing_events is None:
        timing_events = [
            {"event_type": first_input_event_type, "timestamp": iso(submitted)}
        ]
        if responded is not None:
            timing_events.append(
                {
                    "event_type": "tool_call" if emits_tool_call else "text",
                    "timestamp": iso(responded),
                }
            )
    return {
        "provider": provider,
        "project": project,
        "user": user,
        "session_id": session_id,
        "round_index": round_index,
        "model": model,
        "input_tokens_total": input_tokens,
        "prefix_tokens": prefix_tokens,
        "newly_append_tokens": max(input_tokens - prefix_tokens, 0),
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "first_input_event_type": first_input_event_type,
        "timing_events": timing_events,
        "tools": tools or [],
    }


def spawn_tool(*, emitted: float, result: float, name: str = "Agent") -> dict:
    return {
        "tool_name": name,
        "emitted_at": iso(emitted),
        "result_at": iso(result),
        "tool_wall_latency_ms": int((result - emitted) * 1000),
    }


def make_tracelab_run(
    *,
    block_size: int | None = None,
    entries: int | None = None,
    max_context_length: int | None = None,
):
    """A real ``BenchmarkRun`` whose default dataset is a TraceLab file dataset."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    dataset: dict = {
        "name": "default",
        "type": "file",
        "records": [{"text": "placeholder"}],
        "format": "tracelab",
    }
    if block_size is not None:
        dataset["block_size"] = block_size
    if entries is not None:
        dataset["entries"] = entries
    if max_context_length is not None:
        dataset["max_context_length"] = max_context_length

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [dataset],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                }
            ],
            "tokenizer": {"name": "test-tok", "trust_remote_code": False},
            "runtime": {"ui": "simple"},
        }
    )
    return BenchmarkRun(
        benchmark_id="test-tracelab-run",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=0,
    )


def write_jsonl(path: Path, rows: list[dict], *, gz: bool = False) -> Path:
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Hash-id synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeHashIds:
    def test_length_is_floor_not_ceil(self):
        """A partial trailing block is carried unhashed, so the list floors.

        A ceil-length list is silently accepted downstream and wrong twice: the
        trailing id is dropped as content while the cache metric still counts
        it, and the extra id perturbs agent-chain detection.
        """
        rows = [make_row(input_tokens=130, prefix_tokens=0)]
        (ids,) = synthesize_hash_ids(rows, 64, HashIdMinter())
        assert len(ids) == 130 // 64 == 2

    @pytest.mark.parametrize(
        "total,block,expected",
        [
            param(0, 64, 0, id="empty"),
            param(63, 64, 0, id="below_one_block"),
            param(64, 64, 1, id="exactly_one_block"),
            param(127, 64, 1, id="partial_second_block"),
            param(256, 64, 4, id="four_blocks"),
            param(100, 16, 6, id="small_block_size"),
        ],
    )  # fmt: skip
    def test_length_matches_floor_division(self, total, block, expected):
        rows = [make_row(input_tokens=total)]
        (ids,) = synthesize_hash_ids(rows, block, HashIdMinter())
        assert len(ids) == expected

    def test_prefix_is_reused_from_the_previous_round(self):
        rows = [
            make_row(round_index=0, input_tokens=256, prefix_tokens=0),
            make_row(round_index=1, input_tokens=384, prefix_tokens=256),
        ]
        first, second = synthesize_hash_ids(rows, 64, HashIdMinter())
        assert first == [1, 2, 3, 4]
        # Four reused blocks, then two freshly minted ones.
        assert second[:4] == first
        assert len(second) == 6
        assert second[4:] == [5, 6]

    def test_ids_are_unique_when_nothing_is_reused(self):
        rows = [
            make_row(round_index=0, input_tokens=128, prefix_tokens=0),
            make_row(round_index=1, input_tokens=128, prefix_tokens=0),
        ]
        first, second = synthesize_hash_ids(rows, 64, HashIdMinter())
        assert set(first).isdisjoint(second)

    def test_compaction_shrinks_the_reused_span(self):
        """A smaller reported prefix means the agent compacted its context."""
        rows = [
            make_row(round_index=0, input_tokens=640, prefix_tokens=0),
            make_row(round_index=1, input_tokens=704, prefix_tokens=640),
            make_row(round_index=2, input_tokens=256, prefix_tokens=128),
        ]
        chains = synthesize_hash_ids(rows, 64, HashIdMinter())
        assert len(chains[1]) == 11 and chains[1][:10] == chains[0]
        # Only two blocks survive the compaction; the rest are fresh.
        assert chains[2][:2] == chains[1][:2]
        assert len(chains[2]) == 4
        assert set(chains[2][2:]).isdisjoint(chains[1])

    def test_prefix_larger_than_input_is_clamped(self):
        """Never claim more prefix than the round's input actually holds."""
        rows = [
            make_row(round_index=0, input_tokens=640, prefix_tokens=0),
            make_row(round_index=1, input_tokens=128, prefix_tokens=999999),
        ]
        chains = synthesize_hash_ids(rows, 64, HashIdMinter())
        assert len(chains[1]) == 2
        assert chains[1] == chains[0][:2]

    def test_missing_token_fields_do_not_raise(self):
        rows = [{"input_tokens_total": None, "prefix_tokens": None}]
        assert synthesize_hash_ids(rows, 64, HashIdMinter()) == [[]]

    def test_parent_and_child_share_one_namespace(self):
        minter = HashIdMinter()
        parent = synthesize_hash_ids([make_row(input_tokens=128)], 64, minter)
        child = synthesize_hash_ids([make_row(input_tokens=128)], 64, minter)
        assert set(parent[0]).isdisjoint(child[0])


class TestHashIdMinter:
    def test_take_is_monotonic_and_non_overlapping(self):
        m = HashIdMinter()
        assert m.take(3) == [1, 2, 3]
        assert m.take(2) == [4, 5]

    def test_take_zero_or_negative_is_empty(self):
        m = HashIdMinter()
        assert m.take(0) == []
        assert m.take(-1) == []
        assert m.take(1) == [1]


# ---------------------------------------------------------------------------
# parse_ts
# ---------------------------------------------------------------------------


class TestParseTs:
    def test_utc_suffix_z(self):
        from aiperf.dataset.loader._tracelab_convert import parse_ts

        assert parse_ts("2026-01-01T00:00:00Z") == parse_ts("2026-01-01T00:00:00+00:00")

    def test_naive_timestamp_assumed_utc(self):
        """A timestamp without tzinfo must be treated as UTC, not local time."""
        from aiperf.dataset.loader._tracelab_convert import parse_ts

        naive = parse_ts("2026-06-15T12:00:00")
        utc = parse_ts("2026-06-15T12:00:00Z")
        assert naive == utc


# ---------------------------------------------------------------------------
# Round timing and ordering
# ---------------------------------------------------------------------------


class TestRoundTiming:
    def test_api_time_spans_submission_to_last_output(self):
        _, api = round_timing(make_row(submitted=0.0, responded=2.5))
        assert api == pytest.approx(2.5)

    def test_submission_is_the_latest_input_event(self):
        """Parallel tool results: the request cannot predate the last one."""
        row = make_row(
            timing_events=[
                {"event_type": "tool_result", "timestamp": iso(1.0)},
                {"event_type": "tool_result", "timestamp": iso(3.0)},
                {"event_type": "text", "timestamp": iso(5.0)},
            ]
        )
        submitted, api = round_timing(row)
        assert submitted == pytest.approx((T0 + timedelta(seconds=3)).timestamp())
        assert api == pytest.approx(2.0)

    def test_output_only_round_has_no_api_time_offset(self):
        row = make_row(timing_events=[{"event_type": "text", "timestamp": iso(4.0)}])
        submitted, api = round_timing(row)
        assert submitted == pytest.approx((T0 + timedelta(seconds=4)).timestamp())
        assert api == pytest.approx(0.0)

    def test_input_only_round_has_no_api_time(self):
        row = make_row(
            timing_events=[{"event_type": "user_message", "timestamp": iso(1.0)}]
        )
        _, api = round_timing(row)
        assert api is None

    def test_no_timing_events_is_undated(self):
        assert round_timing(make_row(timing_events=[])) == (None, None)

    def test_api_time_is_never_negative(self):
        row = make_row(
            timing_events=[
                {"event_type": "user_message", "timestamp": iso(9.0)},
                {"event_type": "text", "timestamp": iso(1.0)},
            ]
        )
        _, api = round_timing(row)
        assert api == 0.0


class TestOrderRounds:
    def test_orders_by_submission_time_not_round_index(self):
        """round_index genuinely disagrees with recorded timestamps here.

        Spawn position is read by array index while the join is placed by
        timestamp, so a request array that is not monotonic in ``t`` puts a
        join ahead of its spawn and the whole trace is discarded.
        """
        rows = [
            make_row(round_index=5, submitted=1.0),
            make_row(round_index=1, submitted=9.0),
            make_row(round_index=3, submitted=4.0),
        ]
        ordered = order_rounds(rows)
        assert [r[2]["round_index"] for r in ordered] == [5, 3, 1]
        assert [r[0] for r in ordered] == sorted(r[0] for r in ordered)

    def test_round_index_breaks_timestamp_ties(self):
        rows = [
            make_row(round_index=7, submitted=2.0),
            make_row(round_index=2, submitted=2.0),
        ]
        assert [r[2]["round_index"] for r in order_rounds(rows)] == [2, 7]

    def test_undated_rows_are_dropped(self):
        rows = [make_row(round_index=0), make_row(round_index=1, timing_events=[])]
        assert len(order_rounds(rows)) == 1

    def test_duplicate_session_and_round_index_do_not_crash(self):
        rows = [
            make_row(round_index=1, submitted=1.0),
            make_row(round_index=1, submitted=2.0),
        ]
        assert len(order_rounds(rows)) == 2


class TestSessionSpan:
    def test_span_covers_events_and_tool_calls(self):
        rows = [
            make_row(
                submitted=1.0,
                responded=2.0,
                tools=[spawn_tool(emitted=3.0, result=8.0)],
            )
        ]
        start, end = session_span(rows)
        assert start == pytest.approx((T0 + timedelta(seconds=1)).timestamp())
        assert end == pytest.approx((T0 + timedelta(seconds=8)).timestamp())

    def test_no_stamps_is_none(self):
        assert session_span([make_row(timing_events=[], tools=[])]) is None


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


class TestBuildTraceRequests:
    def test_basic_shape_validates_against_the_schema(self):
        rows = [
            make_row(round_index=i, submitted=i * 10.0, responded=i * 10.0 + 1.0)
            for i in range(3)
        ]
        blob = build_trace("claude:s1", rows, 64)
        trace = WekaTrace.model_validate(blob)
        assert trace.hash_id_scope == "local"
        assert trace.block_size == 64
        assert len(trace.requests) == 3
        assert blob["totals"]["rounds"] == 3
        assert blob["totals"]["source"] == "tracelab"

    def test_reasoning_tokens_fold_into_output_length(self):
        rows = [make_row(output_tokens=10, reasoning_tokens=90)]
        blob = build_trace("s", rows, 64)
        assert blob["requests"][0]["out"] == 100

    def test_think_time_is_the_gap_after_the_previous_response(self):
        rows = [
            make_row(round_index=0, submitted=0.0, responded=2.0),
            make_row(round_index=1, submitted=10.0, responded=11.0),
        ]
        blob = build_trace("s", rows, 64)
        assert blob["requests"][0]["think_time"] is None
        assert blob["requests"][1]["think_time"] == pytest.approx(8.0)

    def test_stop_reason_reflects_whether_a_tool_was_requested(self):
        tool_round = build_trace("s", [make_row(emits_tool_call=True)], 64)
        plain_round = build_trace("s", [make_row(emits_tool_call=False)], 64)
        assert tool_round["requests"][0]["stop"] == "tool_use"
        assert plain_round["requests"][0]["stop"] == "end_turn"

    def test_input_types_follow_the_first_input_event(self):
        cont = build_trace("s", [make_row(first_input_event_type="tool_result")], 64)
        fresh = build_trace("s", [make_row(first_input_event_type="user_message")], 64)
        assert cont["requests"][0]["input_types"] == ["tool_result"]
        assert fresh["requests"][0]["input_types"] == ["text"]

    def test_token_counts_floor_at_one(self):
        blob = build_trace("s", [make_row(input_tokens=0, output_tokens=0)], 64)
        assert blob["requests"][0]["in"] == 1
        assert blob["requests"][0]["out"] == 1

    def test_models_are_collected_in_first_use_order(self):
        rows = [
            make_row(round_index=0, submitted=0.0, model="model-b"),
            make_row(round_index=1, submitted=1.0, model="model-a"),
            make_row(round_index=2, submitted=2.0, model="model-b"),
        ]
        assert build_trace("s", rows, 64)["models"] == ["model-b", "model-a"]

    def test_all_undated_session_returns_none(self):
        assert build_trace("s", [make_row(timing_events=[])], 64) is None


# ---------------------------------------------------------------------------
# Subagent join
# ---------------------------------------------------------------------------


def parent_with_spawn(*, sid="claude:parent", emitted=5.0, result=60.0, **kw):
    return [
        make_row(session_id=sid, round_index=0, submitted=0.0, responded=1.0, **kw),
        make_row(
            session_id=sid,
            round_index=1,
            submitted=4.0,
            responded=4.5,
            tools=[spawn_tool(emitted=emitted, result=result)],
            **kw,
        ),
        make_row(session_id=sid, round_index=2, submitted=70.0, responded=71.0, **kw),
    ]


def child_session(*, sid="claude:child", start=10.0, n=3, **kw):
    return [
        make_row(
            session_id=sid,
            round_index=i,
            submitted=start + i * 2.0,
            responded=start + i * 2.0 + 1.0,
            **kw,
        )
        for i in range(n)
    ]


class TestSubagentJoin:
    def test_contained_child_is_matched_to_its_spawn(self):
        sessions = {
            "claude:parent": parent_with_spawn(),
            "claude:child": child_session(),
        }
        links, stats = build_join_index(sessions)
        assert links["claude:child"].parent_sid == "claude:parent"
        assert stats.matched == 1 and stats.matched_claude == 1
        assert stats.windows == 1 and stats.windows_matched == 1

    def test_child_outside_the_window_is_not_matched(self):
        sessions = {
            "claude:parent": parent_with_spawn(emitted=5.0, result=15.0),
            "claude:child": child_session(start=100.0),
        }
        links, stats = build_join_index(sessions)
        assert links == {} and stats.matched == 0

    def test_child_only_partly_contained_is_not_matched(self):
        """Containment is total: a child overrunning its window is not one."""
        sessions = {
            "claude:parent": parent_with_spawn(emitted=5.0, result=15.0),
            "claude:child": child_session(start=10.0, n=6),
        }
        links, _ = build_join_index(sessions)
        assert links == {}

    def test_a_different_user_or_project_never_matches(self):
        sessions = {
            "claude:parent": parent_with_spawn(),
            "claude:child": child_session(user="someone_else"),
        }
        assert build_join_index(sessions)[0] == {}

        sessions = {
            "claude:parent": parent_with_spawn(),
            "claude:child": child_session(project="other_project"),
        }
        assert build_join_index(sessions)[0] == {}

    def test_short_spawning_calls_are_ignored(self):
        """Short calls are overwhelmingly no-ops; admitting them over-captures."""
        child = [
            make_row(
                session_id="claude:child", round_index=0, submitted=5.1, responded=5.3
            )
        ]
        sessions = {
            "claude:parent": parent_with_spawn(emitted=5.0, result=5.5),
            "claude:child": child,
        }
        links, stats = build_join_index(sessions)
        assert stats.windows == 0 and links == {}
        # Same data, floor lowered: the window appears and matches.
        links, stats = build_join_index(sessions, min_spawn_ms=100)
        assert stats.windows == 1 and "claude:child" in links

    def test_tightest_window_wins_and_ambiguity_is_counted(self):
        parent = parent_with_spawn(sid="claude:p", emitted=5.0, result=90.0)
        parent[1]["tools"].append(spawn_tool(emitted=8.0, result=40.0))
        sessions = {"claude:p": parent, "claude:child": child_session(start=10.0)}
        links, stats = build_join_index(sessions)
        assert stats.ambiguous == 1
        # The 32s window, not the 85s one.
        link = links["claude:child"]
        assert link.duration_ms == 32000

    def test_a_session_never_matches_its_own_window(self):
        rows = parent_with_spawn(sid="claude:solo")
        links, _ = build_join_index({"claude:solo": rows})
        assert "claude:solo" not in links

    def test_task_tool_also_spawns(self):
        parent = parent_with_spawn()
        parent[1]["tools"] = [spawn_tool(emitted=5.0, result=60.0, name="Task")]
        sessions = {"claude:parent": parent, "claude:child": child_session()}
        assert "claude:child" in build_join_index(sessions)[0]

    def test_unrelated_tool_names_do_not_spawn(self):
        parent = parent_with_spawn()
        parent[1]["tools"] = [spawn_tool(emitted=5.0, result=60.0, name="Bash")]
        sessions = {"claude:parent": parent, "claude:child": child_session()}
        assert build_join_index(sessions)[0] == {}


class TestCodexJoin:
    @staticmethod
    def _codex_parent(sid="codex:parent"):
        return [
            make_row(
                session_id=sid,
                provider="codex",
                round_index=0,
                submitted=0.0,
                responded=1.0,
                tools=[{"tool_name": "spawn_agent", "emitted_at": iso(5.0)}],
            ),
            make_row(
                session_id=sid,
                provider="codex",
                round_index=1,
                submitted=70.0,
                responded=71.0,
                tools=[{"tool_name": "wait_agent", "result_at": iso(60.0)}],
            ),
        ]

    def test_codex_session_window_matches_a_child(self):
        sessions = {
            "codex:parent": self._codex_parent(),
            "codex:child": child_session(sid="codex:child", provider="codex"),
        }
        links, stats = build_join_index(sessions)
        assert links["codex:child"].kind == "codex"
        assert stats.matched_codex == 1 and stats.windows_codex == 1

    def test_codex_join_can_be_disabled_independently(self):
        """Codex handles are stripped, so its window cannot attribute a spawn."""
        sessions = {
            "codex:parent": self._codex_parent(),
            "codex:child": child_session(sid="codex:child", provider="codex"),
        }
        links, stats = build_join_index(sessions, enable_codex=False)
        assert links == {} and stats.windows_codex == 0

    def test_spawn_without_a_wait_yields_no_window(self):
        rows = self._codex_parent()
        rows[1]["tools"] = []
        _, stats = build_join_index({"codex:parent": rows})
        assert stats.windows == 0


class TestGrouping:
    def test_grandchild_is_kept_as_a_standalone_trace(self):
        """The schema nests one level; dropping the grandchild would lose the
        deepest structure the join exists to recover."""
        sessions = {
            "claude:root": parent_with_spawn(
                sid="claude:root", emitted=5.0, result=200.0
            ),
            "claude:mid": [
                make_row(
                    session_id="claude:mid",
                    round_index=0,
                    submitted=10.0,
                    responded=11.0,
                ),
                make_row(
                    session_id="claude:mid",
                    round_index=1,
                    submitted=12.0,
                    responded=13.0,
                    tools=[spawn_tool(emitted=20.0, result=80.0)],
                ),
                make_row(
                    session_id="claude:mid",
                    round_index=2,
                    submitted=100.0,
                    responded=101.0,
                ),
            ],
            "claude:leaf": child_session(sid="claude:leaf", start=30.0, n=2),
        }
        links, stats = build_join_index(sessions)
        grouped = group_children_by_parent(sessions, links, stats)
        assert "claude:mid" in grouped["claude:root"]
        assert stats.grandchildren == 1
        # The leaf is not nested anywhere, so it survives on its own.
        assert all("claude:leaf" not in kids for kids in grouped.values())


class TestNestedTrace:
    @staticmethod
    def _nested():
        sessions = {
            "claude:parent": parent_with_spawn(),
            "claude:child": child_session(),
        }
        links, stats = build_join_index(sessions)
        grouped = group_children_by_parent(sessions, links, stats)
        return build_trace(
            "claude:parent", sessions["claude:parent"], 64, grouped.get("claude:parent")
        )

    def test_subagent_entry_is_emitted_and_validates(self):
        blob = self._nested()
        trace = WekaTrace.model_validate(blob)
        subs = [r for r in trace.requests if r.type == "subagent"]
        assert len(subs) == 1
        assert subs[0].requests and subs[0].status == "completed"
        assert blob["totals"]["subagents"] == 1

    def test_request_array_is_monotonic_in_t(self):
        """The invariant that reconciles array-index spawns with timestamp joins."""
        times = [r["t"] for r in self._nested()["requests"]]
        assert times == sorted(times)

    def test_marker_lands_after_a_parent_request_not_first(self):
        reqs = self._nested()["requests"]
        marker = next(i for i, r in enumerate(reqs) if r["type"] == "subagent")
        assert marker > 0
        assert reqs[marker - 1]["type"] == "n"

    def test_inner_requests_never_precede_the_marker(self):
        """An inner t below the marker is re-read as relative and flung forward."""
        blob = self._nested()
        entry = next(r for r in blob["requests"] if r["type"] == "subagent")
        assert all(r["t"] >= entry["t"] for r in entry["requests"])

    def test_containment_makes_early_inner_requests_unreachable(self):
        """The join's own precondition supplies the invariant.

        A child only matches when its whole span is inside the spawn window,
        and the span mins over tool stamps as well as timing events, so a
        matched child can never carry a request before its own marker. This
        asserts the invariant on a child that reaches right to both edges of
        its window, where a violation would be likeliest.
        """
        child = [
            make_row(
                session_id="claude:child",
                round_index=0,
                submitted=5.0,
                responded=6.0,
                tools=[spawn_tool(emitted=6.0, result=7.0, name="Bash")],
            ),
            make_row(
                session_id="claude:child", round_index=1, submitted=8.0, responded=60.0
            ),
        ]
        sessions = {
            "claude:parent": parent_with_spawn(emitted=5.0, result=60.0),
            "claude:child": child,
        }
        links, stats = build_join_index(sessions)
        assert "claude:child" in links, "fixture must actually match"
        grouped = group_children_by_parent(sessions, links, stats)
        blob = build_trace(
            "claude:parent", sessions["claude:parent"], 64, grouped["claude:parent"]
        )
        entry = next(r for r in blob["requests"] if r["type"] == "subagent")
        assert min(r["t"] for r in entry["requests"]) >= entry["t"]

    def test_guard_clamps_an_inner_request_that_precedes_its_marker(self):
        """Drive the guard directly: containment cannot produce this input.

        Downstream reads an inner t below the marker as subagent-relative and
        rewrites it, which would scatter one request far into the future. The
        guard is only reachable by handing ``build_subagent_entry`` a spawn that
        starts after its child, which the join never does.
        """
        from aiperf.dataset.loader._tracelab_convert import Spawn, build_subagent_entry

        child = child_session(start=0.0, n=3)
        t0 = order_rounds(child)[0][0]
        late = Spawn(
            parent_sid="claude:parent",
            child_sid="claude:child",
            start=t0 + 100.0,
            end=t0 + 200.0,
            duration_ms=100000,
            kind="claude",
        )
        entry = build_subagent_entry(
            late, child_rows=child, block_size=64, minter=HashIdMinter(), t0=t0
        )
        assert entry["t"] == pytest.approx(100.0)
        assert all(r["t"] >= entry["t"] for r in entry["requests"])
        assert entry["requests"][0]["t"] == pytest.approx(100.0)

    def test_child_first_request_has_no_think_time(self):
        entry = next(r for r in self._nested()["requests"] if r["type"] == "subagent")
        assert entry["requests"][0]["think_time"] is None

    def test_agent_id_is_filesystem_and_schema_safe(self):
        entry = next(r for r in self._nested()["requests"] if r["type"] == "subagent")
        assert entry["agent_id"] == "claude_child"

    def test_child_hash_ids_do_not_collide_with_the_parent(self):
        blob = self._nested()
        parent_ids = {
            i for r in blob["requests"] if r["type"] == "n" for i in r["hash_ids"]
        }
        entry = next(r for r in blob["requests"] if r["type"] == "subagent")
        child_ids = {i for r in entry["requests"] for i in r["hash_ids"]}
        assert parent_ids and child_ids
        assert parent_ids.isdisjoint(child_ids)

    def test_totals_count_only_top_level_rounds(self):
        blob = self._nested()
        assert blob["totals"]["rounds"] == 3


class TestSafeTraceId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            param("claude:abc-123", "claude_abc-123", id="colon_and_hyphen"),
            param("a/b\\c", "a_b_c", id="slash_and_backslash"),
            param("plain.id_1", "plain.id_1", id="already_safe"),
        ],
    )  # fmt: skip
    def test_unsafe_characters_are_replaced(self, raw, expected):
        assert safe_trace_id(raw) == expected

    def test_long_ids_are_truncated(self):
        assert len(safe_trace_id("x" * 500)) == 150


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def build_loader(path, **run_kw):
    return TraceLabTraceDatasetLoader(
        filename=str(path),
        run=make_tracelab_run(**run_kw),
        default_block_size=DEFAULT_BLOCK_SIZE,
    )


class TestLoaderCanLoad:
    def test_recognizes_a_tracelab_row(self):
        assert TraceLabTraceDatasetLoader.can_load(data=make_row())

    @pytest.mark.parametrize(
        "row",
        [
            param({"timestamp": 1, "input_length": 5}, id="mooncake_shape"),
            param({"session_id": "s", "round_index": 0}, id="partial_tracelab"),
            param({}, id="empty"),
        ],
    )  # fmt: skip
    def test_rejects_other_shapes(self, row):
        assert not TraceLabTraceDatasetLoader.can_load(data=row)

    def test_recognizes_a_file_and_its_gzipped_twin(self, tmp_path):
        plain = write_jsonl(tmp_path / "t.jsonl", [make_row()])
        gz = write_jsonl(tmp_path / "t.jsonl.gz", [make_row()], gz=True)
        assert TraceLabTraceDatasetLoader.can_load(filename=str(plain))
        assert TraceLabTraceDatasetLoader.can_load(filename=str(gz))

    def test_rejects_directories_and_missing_files(self, tmp_path):
        assert not TraceLabTraceDatasetLoader.can_load(filename=str(tmp_path))
        assert not TraceLabTraceDatasetLoader.can_load(
            filename=str(tmp_path / "nope.jsonl")
        )

    def test_rejects_a_non_tracelab_jsonl(self, tmp_path):
        path = write_jsonl(
            tmp_path / "other.jsonl", [{"timestamp": 0, "input_length": 4}]
        )
        assert not TraceLabTraceDatasetLoader.can_load(filename=str(path))

    def test_none_arguments_are_safe(self):
        assert not TraceLabTraceDatasetLoader.can_load()

    def test_malformed_json_does_not_raise(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        assert TraceLabTraceDatasetLoader.can_load(filename=str(path)) is False


class TestLoaderLoadDataset:
    @staticmethod
    def _corpus():
        return (
            parent_with_spawn()
            + child_session()
            + [
                make_row(
                    session_id="claude:solo",
                    round_index=i,
                    submitted=i * 3.0,
                    responded=i * 3.0 + 1.0,
                )
                for i in range(2)
            ]
        )

    def test_loads_sessions_and_nests_the_recovered_child(self, tmp_path):
        path = write_jsonl(tmp_path / "c.jsonl", self._corpus())
        data = build_loader(path).load_dataset()
        assert set(data) == {"claude_parent", "claude_solo"}
        parent = data["claude_parent"][0]
        assert any(r.type == "subagent" for r in parent.requests)

    def test_gzip_and_plain_produce_identical_traces(self, tmp_path):
        rows = self._corpus()
        plain = build_loader(write_jsonl(tmp_path / "c.jsonl", rows)).load_dataset()
        gz = build_loader(
            write_jsonl(tmp_path / "c.jsonl.gz", rows, gz=True)
        ).load_dataset()
        assert plain.keys() == gz.keys()
        for key in plain:
            assert plain[key][0].model_dump() == gz[key][0].model_dump()

    def test_block_size_defaults_to_the_plugin_metadata_value(self, tmp_path):
        path = write_jsonl(tmp_path / "c.jsonl", self._corpus())
        data = build_loader(path).load_dataset()
        assert all(t[0].block_size == DEFAULT_BLOCK_SIZE for t in data.values())

    def test_isl_block_size_overrides_the_plugin_metadata_value(self, tmp_path):
        """Unlike Weka, TraceLab has no recorded block size: the user's wins."""
        path = write_jsonl(tmp_path / "c.jsonl", self._corpus())
        data = build_loader(path, block_size=128).load_dataset()
        assert all(t[0].block_size == 128 for t in data.values())
        # And it actually changes synthesis, not just the declared field.
        wide = data["claude_solo"][0].requests[0]
        narrow = build_loader(path).load_dataset()["claude_solo"][0].requests[0]
        assert len(wide.hash_ids) < len(narrow.hash_ids)

    def test_num_dataset_entries_caps_the_trace_count(self, tmp_path):
        path = write_jsonl(tmp_path / "c.jsonl", self._corpus())
        assert len(build_loader(path, entries=1).load_dataset()) == 1

    def test_subagent_join_can_be_disabled(self, tmp_path, monkeypatch):
        from aiperf.common import environment as env_mod

        monkeypatch.setattr(
            env_mod.Environment.DATASET, "TRACELAB_SUBAGENT_JOIN", False
        )
        path = write_jsonl(tmp_path / "c.jsonl", self._corpus())
        data = build_loader(path).load_dataset()
        assert "claude_child" in data
        assert not any(
            r.type == "subagent" for t in data.values() for r in t[0].requests
        )

    def test_child_that_fails_to_nest_is_emitted_standalone(self, tmp_path):
        """A child linked by join but that fails to nest (e.g. anchor lookup
        returns None) must appear as a standalone trace, not be silently dropped."""
        from unittest.mock import patch

        from aiperf.dataset.loader._tracelab_convert import Spawn

        parent_rows = parent_with_spawn()
        child_rows = child_session()
        loader = build_loader(
            write_jsonl(tmp_path / "c.jsonl", parent_rows + child_rows)
        )
        sessions = {"claude:parent": parent_rows, "claude:child": child_rows}
        spawn = Spawn(
            parent_sid="claude:parent",
            child_sid="claude:child",
            start=5.0,
            end=60.0,
            duration_ms=55000,
            kind="claude",
        )
        children_by_parent = {"claude:parent": {"claude:child": (spawn, child_rows)}}
        nested_ids = {"claude:child"}

        with patch(
            "aiperf.dataset.loader._tracelab_convert._anchor_index",
            return_value=None,
        ):
            data = loader._build_traces(sessions, children_by_parent, nested_ids)

        assert "claude_child" in data
        assert "claude_parent" in data
        assert not any(r.type == "subagent" for r in data["claude_parent"].requests)

    def test_fallback_sessions_emitted_in_file_order(self, tmp_path):
        """Fallback traces must follow sessions-dict insertion order, not set
        iteration order, so --num-dataset-entries is deterministic."""
        from unittest.mock import patch

        rows_a = child_session(sid="claude:a", start=10.0)
        rows_b = child_session(sid="claude:b", start=12.0)
        rows_c = child_session(sid="claude:c", start=14.0)
        # Build a loader with a file that provides the sessions dict ordering.
        loader = build_loader(
            write_jsonl(
                tmp_path / "c.jsonl",
                parent_with_spawn() + rows_a + rows_b + rows_c,
            )
        )
        sessions = {
            "claude:parent": parent_with_spawn(),
            "claude:a": rows_a,
            "claude:b": rows_b,
            "claude:c": rows_c,
        }
        nested_ids = {"claude:a", "claude:b", "claude:c"}

        with patch(
            "aiperf.dataset.loader._tracelab_convert._anchor_index",
            return_value=None,
        ):
            traces = loader._build_traces(sessions, {}, nested_ids)

        keys = list(traces.keys())
        # All three fallback sessions present (parent also emitted as a root).
        assert {"claude_a", "claude_b", "claude_c"}.issubset(set(keys))
        # Fallback entries must follow sessions-dict order (a, b, c), not set order.
        fallback_keys = [k for k in keys if k in {"claude_a", "claude_b", "claude_c"}]
        assert fallback_keys == ["claude_a", "claude_b", "claude_c"]

    def test_build_traces_wraps_conversion_error_with_session_id(self, tmp_path):
        """ValueError/TypeError from build_trace is re-raised as DatasetLoaderError
        with the session id attached."""
        from unittest.mock import patch

        loader = build_loader(write_jsonl(tmp_path / "c.jsonl", parent_with_spawn()))
        sessions = {"claude:parent": parent_with_spawn()}

        with (
            patch(
                "aiperf.dataset.loader.tracelab_trace.build_trace",
                side_effect=ValueError("bad timestamp"),
            ),
            pytest.raises(DatasetLoaderError, match="claude:parent"),
        ):
            loader._build_traces(sessions, {}, set())

    def test_merge_skips_child_with_no_timed_rounds(self, tmp_path):
        """When build_subagent_entry returns None (child has no timed rounds),
        _merge_subagent_entries skips it without crashing."""
        from aiperf.dataset.loader._tracelab_convert import (
            HashIdMinter,
            Spawn,
            _merge_subagent_entries,
            build_requests,
            synthesize_hash_ids,
        )

        parent_rows = parent_with_spawn()
        timed_parent = [
            (r[0], r[1], r[2])
            for r in [(0.0, 1.0, parent_rows[0]), (4.0, 0.5, parent_rows[1])]
        ]
        t0 = 0.0
        minter = HashIdMinter()
        hashes = synthesize_hash_ids([r[2] for r in timed_parent], 64, minter)
        requests, models = build_requests(timed_parent, hashes, t0)

        child_no_timing = [make_row(session_id="claude:child", timing_events=[])]
        spawn = Spawn(
            parent_sid="claude:parent",
            child_sid="claude:child",
            start=5.0,
            end=60.0,
            duration_ms=55000,
            kind="claude",
        )
        placed: set[str] = set()
        merged, n = _merge_subagent_entries(
            requests=requests,
            models=models,
            children={"claude:child": (spawn, child_no_timing)},
            block_size=64,
            minter=minter,
            t0=t0,
            placed_sids=placed,
        )
        assert n == 0
        assert placed == set()
        assert not any(r.get("type") == "subagent" for r in merged)

    def test_merge_adds_child_model_to_parent_models(self, tmp_path):
        """When the child uses a model not yet in the parent's list, it is appended."""
        from aiperf.dataset.loader._tracelab_convert import (
            HashIdMinter,
            Spawn,
            _merge_subagent_entries,
            build_requests,
            synthesize_hash_ids,
        )

        parent_rows = parent_with_spawn()
        child_rows = [
            make_row(session_id="claude:child", model="different-model", submitted=10.0)
        ]
        t0 = 0.0
        minter = HashIdMinter()
        timed_parent = [(0.0, 1.0, parent_rows[0]), (4.0, 0.5, parent_rows[1])]
        hashes = synthesize_hash_ids([r[2] for r in timed_parent], 64, minter)
        requests, models = build_requests(timed_parent, hashes, t0)

        spawn = Spawn(
            parent_sid="claude:parent",
            child_sid="claude:child",
            start=5.0,
            end=60.0,
            duration_ms=55000,
            kind="claude",
        )
        placed: set[str] = set()
        _merge_subagent_entries(
            requests=requests,
            models=models,
            children={"claude:child": (spawn, child_rows)},
            block_size=64,
            minter=minter,
            t0=t0,
            placed_sids=placed,
        )
        assert "different-model" in models
        assert "claude:child" in placed

    def test_rows_without_a_session_id_are_skipped(self, tmp_path):
        rows = self._corpus() + [{"round_index": 0, "timing_events": []}]
        path = write_jsonl(tmp_path / "c.jsonl", rows)
        assert set(build_loader(path).load_dataset()) == {
            "claude_parent",
            "claude_solo",
        }

    def test_missing_filename_is_a_clear_error(self):
        loader = TraceLabTraceDatasetLoader(inline_records=[], run=make_tracelab_run())
        with pytest.raises(DatasetLoaderError, match="requires --input-file"):
            loader.load_dataset()

    def test_empty_file_is_a_clear_error(self, tmp_path):
        path = write_jsonl(tmp_path / "empty.jsonl", [])
        with pytest.raises(DatasetLoaderError, match="No TraceLab sessions"):
            build_loader(path).load_dataset()

    def test_invalid_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(make_row()) + "\n{oops\n", encoding="utf-8")
        with pytest.raises(DatasetLoaderError, match="line 2"):
            build_loader(path).load_dataset()

    def test_truncated_gzip_raises_dataset_loader_error(self, tmp_path):
        """A partially-downloaded or corrupt .gz raises DatasetLoaderError, not
        a raw zlib.error traceback."""
        import gzip as _gzip

        gz = tmp_path / "bad.jsonl.gz"
        with _gzip.open(gz, "wt", encoding="utf-8") as f:
            for _ in range(50):
                f.write(json.dumps(make_row()) + "\n")
        data = gz.read_bytes()
        gz.write_bytes(data[: max(len(data) - 30, 2)])
        with pytest.raises(DatasetLoaderError, match="Cannot read TraceLab file"):
            build_loader(gz).load_dataset()

    def test_non_utf8_file_raises_dataset_loader_error(self, tmp_path):
        """A file with invalid UTF-8 bytes raises DatasetLoaderError, not a
        raw UnicodeDecodeError traceback."""
        bad = tmp_path / "bad.jsonl"
        bad.write_bytes(b'{"session_id": "s1"}\n\xff\xfe\n')
        with pytest.raises(DatasetLoaderError, match="Cannot read TraceLab file"):
            build_loader(bad).load_dataset()

    def test_undated_sessions_are_dropped_not_fatal(self, tmp_path):
        rows = self._corpus() + [
            make_row(session_id="claude:ghost", timing_events=[], tools=[])
        ]
        path = write_jsonl(tmp_path / "c.jsonl", rows)
        assert "claude_ghost" not in build_loader(path).load_dataset()

    def test_sampling_strategy_is_sequential(self):
        from aiperf.plugin.enums import DatasetSamplingStrategy

        assert (
            TraceLabTraceDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


class TestLoaderReconstruction:
    def test_conversations_include_the_subagent_child(self, tmp_path):
        from tests.unit.dataset.loader._shared_helpers import _stub_loader

        rows = parent_with_spawn() + child_session()
        loader = build_loader(write_jsonl(tmp_path / "c.jsonl", rows))
        data = loader.load_dataset()
        _stub_loader(loader._weka)
        conversations = loader.convert_to_conversations(data)
        ids = {c.session_id for c in conversations}
        assert "claude_parent" in ids
        # One root plus one child carrying the recovered agent id.
        assert any("::sa:" in sid for sid in ids)
        assert len(conversations) >= 2


class TestRegistration:
    def test_loader_resolves_from_the_plugin_registry(self):
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import CustomDatasetType, PluginType

        cls = plugins.get_class(
            PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.TRACELAB
        )
        assert cls is TraceLabTraceDatasetLoader

    def test_plugin_metadata_declares_our_own_block_size(self):
        """Declared explicitly so it becomes ours to change independently."""
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import CustomDatasetType

        meta = plugins.get_dataset_loader_metadata(CustomDatasetType.TRACELAB)
        assert meta.is_trace is True
        assert meta.default_block_size == DEFAULT_BLOCK_SIZE

    def test_isl_block_size_is_accepted_for_tracelab(self):
        """Without the format in the block-size set this raises."""
        from aiperf.config.flags._converter_dataset import _BLOCK_SIZE_TRACE_FORMATS

        assert "tracelab" in _BLOCK_SIZE_TRACE_FORMATS

    def test_mmap_cache_treats_tracelab_as_a_trace(self):
        from aiperf.dataset.mmap_cache import is_trace_or_verbatim_dataset

        assert is_trace_or_verbatim_dataset("tracelab", None)

    def test_isl_block_size_survives_the_config_converter(self):
        """The flag must reach FileDataset.block_size, not be rejected or dropped.

        Weka rejects it outright because its traces declare their own per-block
        sizes; TraceLab synthesizes its block ids, so the value is load-bearing
        and has to arrive.
        """
        from aiperf.common.enums import DatasetType
        from aiperf.config.flags._converter_dataset import _apply_block_size

        cli = type(
            "StubCLI",
            (),
            {
                "model_fields_set": {"prompt_input_tokens_block_size"},
                "prompt_input_tokens_block_size": 128,
            },
        )()
        d: dict = {"format": "tracelab", "type": DatasetType.FILE, "dataset": None}
        _apply_block_size(d, cli)
        assert d["block_size"] == 128

    def test_block_size_zero_raises_not_silently_defaults(self):
        """block_size=0 passed via default_block_size must raise, not silently
        fall back to DEFAULT_BLOCK_SIZE (regression: 'if block_size' coerces 0
        to False and skips the validation that follows)."""
        with pytest.raises(DatasetLoaderError, match="must be positive"):
            TraceLabTraceDatasetLoader(
                inline_records=[],
                run=make_tracelab_run(),
                default_block_size=0,
            )

    def test_first_record_has_timestamp_detects_tracelab(self, tmp_path):
        """_first_record_has_timestamp must return True for TraceLab rows so
        _maybe_auto_promote_trace can promote them to fixed-schedule."""
        from aiperf.config.flags._converter_profiling import _first_record_has_timestamp

        path = write_jsonl(tmp_path / "t.jsonl", [make_row()])
        assert _first_record_has_timestamp(path)

    def test_first_record_has_timestamp_detects_gzipped_tracelab(self, tmp_path):
        from aiperf.config.flags._converter_profiling import _first_record_has_timestamp

        gz = write_jsonl(tmp_path / "t.jsonl.gz", [make_row()], gz=True)
        assert _first_record_has_timestamp(gz)

    def test_fixed_schedule_accepts_tracelab_timing(self):
        """Timing is nested under timing_events, so the generic probe misses it."""
        from aiperf.config.dataset.resolver import DatasetResolver
        from aiperf.plugin.enums import CustomDatasetType

        assert DatasetResolver._check_timing_data(
            "ignored.jsonl", None, CustomDatasetType.TRACELAB
        )


class TestGzipPlumbing:
    """BaseFileLoader._iter_record_dicts transparently handles gzipped sources."""

    def test_base_file_loader_iterates_a_gzipped_source(self, tmp_path):
        gz = write_jsonl(tmp_path / "a.jsonl.gz", [make_row()], gz=True)
        records = list(build_loader(gz)._iter_record_dicts())
        assert len(records) == 1 and records[0]["session_id"] == "claude:s1"
