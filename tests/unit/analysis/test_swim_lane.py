# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the swim-lane session layout and renderers.

Focus: subagent sessions (agent_depth > 0) must nest as child tiers below
their root session's lane instead of occupying their own slots.
"""

from __future__ import annotations

import base64
import gzip
import re
from pathlib import Path

import matplotlib
import orjson
import pytest

from aiperf.analysis.swim_lane import (
    PROFILE_JSON,
    _group_into_sessions,
    _layout_groups,
    _load_bench_config,
    _root_map,
    plot_swim_lane,
    write_swim_lane_html,
)
from aiperf.cli_commands.swim_lane import swim_lane as cli_swim_lane

matplotlib.use("Agg", force=True)

NS = 1_000_000_000


def _rec(
    sid: str,
    turn: int,
    start_s: float,
    end_s: float,
    parent: str | None = None,
    depth: int = 0,
    root: str | None = None,
) -> dict:
    """Build a minimal profile_export.jsonl record.

    ``root`` sets ``root_correlation_id`` (the session-tree root id the runtime
    persists). When omitted the field is absent, exercising the legacy
    parent_correlation_id heuristic; when present, ``_root_map`` groups
    authoritatively on it.
    """
    meta = {
        "x_correlation_id": sid,
        "conversation_id": f"conv-{sid}",
        "turn_index": turn,
        "request_start_ns": int(start_s * NS),
        "request_end_ns": int(end_s * NS),
        "agent_depth": depth,
        "parent_correlation_id": parent,
    }
    if root is not None:
        meta["root_correlation_id"] = root
    return {
        "metadata": meta,
        "metrics": {
            "time_to_first_token": {"value": 50.0, "unit": "ms"},
            "request_latency": {"value": (end_s - start_s) * 1e3, "unit": "ms"},
            "input_sequence_length": {"value": 10, "unit": "tokens"},
            "output_sequence_length": {"value": 5, "unit": "tokens"},
        },
    }


def _sessions(records: list[dict]) -> dict[str, list[dict]]:
    return _group_into_sessions(records)


def _group_for(groups, root_sid):
    return next(g for g in groups if g.root_sid == root_sid)


def _member(group, sid):
    return next(m for m in group.members if m.sid == sid)


class TestRootMap:
    def test_subagent_resolves_to_parent(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 1.0),
                _rec("sub", 0, 0.2, 0.8, parent="root", depth=1),
            ]
        )
        assert _root_map(sessions) == {"root": "root", "sub": "root"}

    def test_nested_subagent_resolves_to_root_ancestor(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 3.0),
                _rec("mid", 0, 0.5, 2.5, parent="root", depth=1),
                _rec("leaf", 0, 1.0, 2.0, parent="mid", depth=2),
            ]
        )
        assert _root_map(sessions) == {"root": "root", "mid": "root", "leaf": "root"}

    def test_root_correlation_id_is_authoritative(self):
        # The persisted root_correlation_id groups every node of a tree directly
        # -- no parent-walk. A subagent whose root made no profiled request
        # (root all before t*, or rootless lane) still carries its tree root id.
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 1.0, root="root"),
                _rec("sub", 0, 0.2, 0.8, parent="root", depth=1, root="root"),
                _rec("orphan", 0, 0.3, 0.9, parent="lane-x", depth=1, root="lane-x"),
            ]
        )
        assert _root_map(sessions) == {
            "root": "root",
            "sub": "root",
            "orphan": "lane-x",
        }

    def test_missing_parent_falls_back_to_own_root_legacy(self):
        # Legacy export (no root_correlation_id): an orphan whose parent is
        # absent keys on that absent parent id (one phantom lane per lane).
        sessions = _sessions([_rec("orphan", 0, 0.0, 1.0, parent="gone", depth=1)])
        assert _root_map(sessions) == {"orphan": "gone"}

    def test_parent_cycle_detaches_both_sessions(self):
        sessions = _sessions(
            [
                _rec("a", 0, 0.0, 1.0, parent="b", depth=1),
                _rec("b", 0, 0.0, 1.0, parent="a", depth=1),
            ]
        )
        assert _root_map(sessions) == {"a": "a", "b": "b"}


class TestLayoutGroups:
    def test_subagent_nests_below_root_in_same_slot(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 1.0),
                _rec("root", 1, 2.0, 3.0),
                _rec("sub", 0, 1.1, 1.9, parent="root", depth=1),
            ]
        )
        groups = _layout_groups(sessions)
        assert len(groups) == 1
        g = groups[0]
        assert g.rows == 2
        root, sub = _member(g, "root"), _member(g, "sub")
        assert (root.row0, root.rows, root.is_sub) == (0, 1, False)
        assert (sub.row0, sub.rows, sub.is_sub) == (1, 1, True)

    def test_concurrent_subagents_stack_into_separate_tiers(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 4.0),
                _rec("s1", 0, 1.0, 3.0, parent="root", depth=1),
                _rec("s2", 0, 1.5, 3.5, parent="root", depth=1),
            ]
        )
        g = _layout_groups(sessions)[0]
        assert g.rows == 3
        assert {_member(g, "s1").row0, _member(g, "s2").row0} == {1, 2}

    def test_sequential_subagents_reuse_child_tier(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 5.0),
                _rec("s1", 0, 1.0, 2.0, parent="root", depth=1),
                _rec("s2", 0, 3.0, 4.0, parent="root", depth=1),
            ]
        )
        g = _layout_groups(sessions)[0]
        assert g.rows == 2
        assert _member(g, "s1").row0 == _member(g, "s2").row0 == 1

    def test_orphan_subagent_lanes_under_its_tree_root(self):
        # A subagent whose root made no profiled request nests under its
        # persisted tree root (a phantom lane: no records of its own), in a
        # different slot than an unrelated root.
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 2.0, root="root"),
                _rec("orphan", 0, 0.5, 1.5, parent="lane-x", depth=1, root="lane-x"),
            ]
        )
        groups = _layout_groups(sessions)
        assert len(groups) == 2
        phantom = _group_for(groups, "lane-x")
        assert phantom.members[0].is_sub
        assert phantom.slot != _group_for(groups, "root").slot

    def test_orphan_subs_sharing_a_root_share_one_lane(self):
        # Two background subagents of the same rootless lane share their tree's
        # root_correlation_id and so group into ONE phantom lane.
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 2.0, root="root"),
                _rec("sa0", 0, 0.5, 1.5, parent="lane-1", depth=1, root="lane-1"),
                _rec("sa1", 0, 1.0, 2.5, parent="lane-1", depth=1, root="lane-1"),
            ]
        )
        groups = _layout_groups(sessions)
        assert len(groups) == 2
        phantom = _group_for(groups, "lane-1")
        # zero-row synthetic root: only the subagents are members
        assert [m.sid for m in phantom.members] == ["sa0", "sa1"]
        assert all(m.is_sub for m in phantom.members)
        assert {m.row0 for m in phantom.members} == {0, 1}
        assert phantom.rows == 2
        assert phantom.span_ns == (int(0.5 * NS), int(2.5 * NS))
        assert phantom.slot != _group_for(groups, "root").slot

    def test_orphan_subs_from_different_roots_stay_separate(self):
        # Subagents carrying different root_correlation_ids are different trees
        # and never collapse into one lane.
        sessions = _sessions(
            [
                _rec("sa_a", 0, 0.0, 1.0, parent="lane-a", depth=1, root="lane-a"),
                _rec("sa_b", 0, 0.5, 1.5, parent="lane-b", depth=1, root="lane-b"),
            ]
        )
        groups = _layout_groups(sessions)
        assert {g.root_sid for g in groups} == {"lane-a", "lane-b"}

    def test_group_span_covers_subagent_overhang(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 1.0),
                _rec("sub", 0, 0.5, 4.0, parent="root", depth=1),
            ]
        )
        g = _layout_groups(sessions)[0]
        assert g.span_ns == (0, 4 * NS)

    def test_parallel_root_turns_keep_top_rows(self):
        sessions = _sessions(
            [
                _rec("root", 0, 0.0, 2.0),
                _rec("root", 1, 1.0, 3.0),
                _rec("sub", 0, 1.0, 2.0, parent="root", depth=1),
            ]
        )
        g = _layout_groups(sessions)[0]
        root, sub = _member(g, "root"), _member(g, "sub")
        assert (root.row0, root.rows) == (0, 2)
        assert sub.row0 == 2
        assert g.rows == 3

    def test_whole_groups_share_slot_after_retirement(self):
        sessions = _sessions(
            [
                _rec("r1", 0, 0.0, 1.0),
                _rec("s1", 0, 0.2, 0.8, parent="r1", depth=1),
                _rec("r2", 0, 2.0, 3.0),
                _rec("s2", 0, 2.2, 2.8, parent="r2", depth=1),
            ]
        )
        groups = _layout_groups(sessions)
        assert _group_for(groups, "r1").slot == _group_for(groups, "r2").slot == 0


@pytest.fixture
def agentic_run_dir(tmp_path: Path) -> Path:
    """Run dir with one main session, two subagents, and one plain session."""
    records = [
        _rec("main", 0, 0.0, 1.0),
        _rec("main", 1, 3.0, 4.0),
        _rec("subA", 0, 1.0, 2.5, parent="main", depth=1),
        _rec("subB", 0, 1.2, 2.8, parent="main", depth=1),
        _rec("plain", 0, 0.0, 4.0),
    ]
    (tmp_path / "profile_export.jsonl").write_bytes(
        b"\n".join(orjson.dumps(r) for r in records)
    )
    return tmp_path


def _extract_payload(html_path: Path) -> dict:
    # payload is embedded as a gzip+base64 blob inflated client-side via
    # DecompressionStream; mirror that decode here to inspect it.
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html_path.read_text())
    assert match, "embedded payload not found"
    return orjson.loads(gzip.decompress(base64.b64decode(match.group(1))))


class TestRenderers:
    def test_plot_swim_lane_writes_png(self, agentic_run_dir: Path):
        out = plot_swim_lane(agentic_run_dir)
        assert out == agentic_run_dir / "swim_lane.png"
        assert out.stat().st_size > 0

    def test_html_payload_nests_subagents_inline(self, agentic_run_dir: Path):
        out = write_swim_lane_html(agentic_run_dir)
        payload = _extract_payload(out)
        by_id = {s["id"]: s for s in payload["sessions"]}

        main, plain = by_id["main"], by_id["plain"]
        sub_a, sub_b = by_id["subA"], by_id["subB"]
        # subagents share the root's slot and color, tiered below row 0
        assert sub_a["slot"] == sub_b["slot"] == main["slot"]
        assert sub_a["ci"] == sub_b["ci"] == main["ci"]
        assert main["row0"] == 0
        assert {sub_a["row0"], sub_b["row0"]} == {1, 2}
        assert sub_a["sub"] is True and sub_a["root"] == "main"
        assert "sub" not in main and "root" not in main
        # the plain session is its own lane in a different slot
        assert plain["slot"] != main["slot"]
        assert plain["ci"] != main["ci"]
        # nSlots counts lane groups, not sessions
        assert payload["nSlots"] == 2
        assert payload["peaks"]["subActive"] == 2

    def test_html_payload_no_subagents_has_no_sub_fields(self, tmp_path: Path):
        records = [_rec("a", 0, 0.0, 1.0), _rec("b", 0, 0.5, 1.5)]
        (tmp_path / "profile_export.jsonl").write_bytes(
            b"\n".join(orjson.dumps(r) for r in records)
        )
        payload = _extract_payload(write_swim_lane_html(tmp_path))
        assert all("sub" not in s and "root" not in s for s in payload["sessions"])
        assert all(s["row0"] == 0 for s in payload["sessions"])


class TestLoadBenchConfigRamp:
    """_load_bench_config must read the concurrency ramp from the v2 profile
    export shape (phases[].concurrency_ramp.duration on the profiling phase).
    The port left it reading the v1 loadgen.concurrency_ramp_duration, which v2
    exports never carry, so the swim-lane ramp-done marker silently vanished."""

    def test_reads_ramp_from_v2_profiling_phase(self, tmp_path: Path) -> None:
        export = {
            "input_config": {
                "phases": [
                    # warmup: must be skipped
                    {
                        "type": "concurrency",
                        "exclude_from_results": True,
                        "concurrency_ramp": {"duration": 5.0},
                    },
                    # profiling: the ramp the marker should use
                    {
                        "type": "concurrency",
                        "exclude_from_results": False,
                        "concurrency_ramp": {"duration": 12.0},
                    },
                ]
            }
        }
        (tmp_path / PROFILE_JSON).write_bytes(orjson.dumps(export))
        ramp, _ = _load_bench_config(tmp_path)
        assert ramp == 12.0

    def test_falls_back_to_v1_loadgen_shape(self, tmp_path: Path) -> None:
        export = {"input_config": {"loadgen": {"concurrency_ramp_duration": 8.0}}}
        (tmp_path / PROFILE_JSON).write_bytes(orjson.dumps(export))
        ramp, _ = _load_bench_config(tmp_path)
        assert ramp == 8.0

    def test_no_ramp_returns_none(self, tmp_path: Path) -> None:
        export = {
            "input_config": {
                "phases": [{"type": "concurrency", "exclude_from_results": False}]
            }
        }
        (tmp_path / PROFILE_JSON).write_bytes(orjson.dumps(export))
        ramp, _ = _load_bench_config(tmp_path)
        assert ramp is None


class TestCli:
    def test_partial_failure_exits_1_and_writes_good(self, tmp_path: Path) -> None:
        good = tmp_path / "good"
        good.mkdir()
        (good / "profile_export.jsonl").write_bytes(
            b"\n".join(
                orjson.dumps(r)
                for r in [_rec("a", 0, 0.0, 1.0), _rec("b", 0, 0.5, 1.5)]
            )
        )
        with pytest.raises(SystemExit) as e:
            cli_swim_lane([good, tmp_path / "missing"])
        assert e.value.code == 1
        assert (good / "swim_lane.png").is_file()

    def test_all_invalid_run_dirs_exit_1(self, tmp_path: Path, capsys) -> None:
        with pytest.raises(SystemExit) as e:
            cli_swim_lane([tmp_path / "missing1", tmp_path / "missing2"])
        assert e.value.code == 1
        assert capsys.readouterr().err.count("skip") == 2

    def test_out_with_multiple_run_dirs_exits_2(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        for d in (a, b):
            (d / "profile_export.jsonl").write_bytes(
                orjson.dumps(_rec("x", 0, 0.0, 1.0)) + b"\n"
            )
        with pytest.raises(SystemExit) as e:
            cli_swim_lane([a, b], out=tmp_path / "x.png")
        assert e.value.code == 2
