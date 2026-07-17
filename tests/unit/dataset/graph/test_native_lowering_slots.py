# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-content slots in the native lowering.

Covers the slot composition rules: array-level splices expand to init
segments + completion-ordered writer slots (writers chained through reads);
block-level refs compose the single writer's value INTO the containing
message; producers get `capture`, readers get injected `ChannelRequirement`s;
and every legality gate (unordered writers, chain-read violations, anchored
producer edges, block multi-writer / init+writer) fails loudly.
"""

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.parser import parse_native
from aiperf.dataset.graph.segment_ir.envelope import read_prompt_segment_ids


def _parse(tmp_path: Path, text: str):
    p = tmp_path / "workload.yaml"
    p.write_text(text)
    return parse_native(p)


def _trie(node: LlmNode) -> dict:
    return (node.metadata or {}).get("trie") or {}


PLANNER_REVIEWER = """
graph:
  nodes:
    plan:
      prompt: [{role: user, content: "Make a plan."}]
      output: plan_out
    review:
      prompt:
        - role: user
          content: ["Review this plan: ", "@plan_out"]
      output: review_out
  edges:
    - {source: START, target: plan}
    - {source: plan, target: review}
    - {source: review, target: END}
traces:
  - id: t1
"""


class TestBlockSlots:
    def test_planner_reviewer_composed_message(self, tmp_path: Path) -> None:
        parsed = _parse(tmp_path, PLANNER_REVIEWER)
        graph = parsed.graphs["t1"]
        plan, review = graph.nodes["plan"], graph.nodes["review"]

        assert _trie(plan).get("capture") is True
        assert "assembly" not in _trie(plan)

        assembly = _trie(review)["assembly"]
        assert assembly == [
            {
                "m": {
                    "role": "user",
                    "parts": [{"t": "Review this plan: "}, {"sv": "plan"}],
                }
            }
        ]
        assert read_prompt_segment_ids(review) == []
        assert [(r.channel, r.count) for r in review.inputs] == [("plan_out", 1)]

    @pytest.mark.parametrize(
        ("yaml_text", "match"),
        [
            param(
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: qa}]
      output: t_out
    b:
      prompt: [{role: user, content: qb}]
      output: t_out
    c:
      prompt:
        - role: user
          content: ["@t_out"]
      output: c_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: c}
    - {source: c, target: END}
traces:
  - id: t1
""",
                "multiple writers",
                id="block_two_writers",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: qa}]
      output: t_out
    b:
      prompt:
        - role: user
          content: ["@t_out"]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
    initial_state:
      t_out: "seeded"
""",
                "both init-seeded and written",
                id="block_init_plus_writer",
            ),
        ],
    )  # fmt: skip
    def test_block_gates(self, tmp_path: Path, yaml_text: str, match: str) -> None:
        with pytest.raises(NotImplementedError, match=match):
            _parse(tmp_path, yaml_text)


ACCUMULATE_CHAIN = """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: ["@hist", {role: user, content: "turn one"}]
      output: hist
    b:
      prompt: ["@hist", {role: user, content: "turn two"}]
      output: hist
    c:
      prompt: ["@hist", {role: user, content: "turn three"}]
      output: c_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: c}
    - {source: c, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: system, content: "be brief"}
"""


class TestArraySlots:
    def test_accumulate_chain_composition(self, tmp_path: Path) -> None:
        parsed = _parse(tmp_path, ACCUMULATE_CHAIN)
        graph = parsed.graphs["t1"]
        a, b, c = graph.nodes["a"], graph.nodes["b"], graph.nodes["c"]

        # Producers referenced by a downstream slot are captured; c is a pure
        # reader (writes c_out) so it is not.
        assert _trie(a).get("capture") is True
        assert _trie(b).get("capture") is True
        assert "capture" not in _trie(c)

        # a reads @hist (= init only; b is a future/descendant writer it does
        # not see), so a is fully STATIC — no slots, no assembly key. Its own
        # prompt materializes [system, "turn one"].
        assert "assembly" not in _trie(a)
        a_path = read_prompt_segment_ids(a)
        assert parsed.segment_pool.materialize(a_path) == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "turn one"},
        ]

        # b reconstructs [init system, delta(a)="turn one", reply(a),
        # delta(b)="turn two"].
        b_assembly = _trie(b)["assembly"]
        assert b_assembly[0].keys() == {"seg"}  # init: system message
        assert b_assembly[1].keys() == {"seg"}  # delta(a): "turn one"
        assert b_assembly[2] == {"s": {"src": "a"}}  # a's reply
        assert b_assembly[3].keys() == {"seg"}  # delta(b): "turn two"
        assert [(r.channel, r.count) for r in b.inputs] == [("hist", 1)]

        # c reconstructs the full alternation of both prior turns.
        c_assembly = _trie(c)["assembly"]
        assert c_assembly[0].keys() == {"seg"}  # init system
        assert c_assembly[1].keys() == {"seg"}  # delta(a) "turn one"
        assert c_assembly[2] == {"s": {"src": "a"}}
        assert c_assembly[3].keys() == {"seg"}  # delta(b) "turn two"
        assert c_assembly[4] == {"s": {"src": "b"}}
        assert c_assembly[5].keys() == {"seg"}  # delta(c) "turn three"
        assert [(r.channel, r.count) for r in c.inputs] == [("hist", 2)]

    def test_first_writer_delta_is_its_whole_prompt(self, tmp_path: Path) -> None:
        # No init: the root writer `a` need not read @hist (Gate B only applies
        # to init-bearing channels). Its delta is its whole prompt.
        parsed = _parse(
            tmp_path,
            """
graph:
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: hist
    b:
      prompt: ["@hist", {role: user, content: next}]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
""",
        )
        assembly = _trie(parsed.graphs["t1"].nodes["b"])["assembly"]
        assert assembly[0].keys() == {"seg"}  # delta(a) = whole prompt "q"
        assert assembly[1] == {"s": {"src": "a"}}  # a's reply
        assert assembly[2].keys() == {"seg"}  # delta(b) = "next"

    def test_init_seeded_write_only_channel_not_gated(self, tmp_path: Path) -> None:
        # G4: hist is init-seeded and written by `a`, but NOTHING reconstructs
        # it (no downstream @hist reader with committed writers), so Gate B
        # must not fire — the divergence it guards against cannot occur.
        parsed = _parse(
            tmp_path,
            """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: [{role: user, content: "turn one"}]
      output: hist
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: system, content: "be brief"}
""",
        )
        a = parsed.graph.nodes["a"]
        assert parsed.segment_pool.materialize(read_prompt_segment_ids(a)) == [
            {"role": "user", "content": "turn one"}
        ]

    def test_rewrite_own_draft_block_ref_not_gated(self, tmp_path: Path) -> None:
        # G4 (review repro): a node block-refs @draft, writes draft, and the
        # trace seeds it. No node reconstructs the channel, so the workload
        # must lower (previously Gate B rejected it and the suggested fix
        # dead-ended on the messages-splice list gate).
        parsed = _parse(
            tmp_path,
            """
graph:
  nodes:
    a:
      prompt:
        - role: user
          content: ["Rewrite this draft: ", "@draft"]
      output: draft
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      draft: "first draft"
""",
        )
        a = parsed.graphs["t1"].nodes["a"]
        assert parsed.segment_pool.materialize(read_prompt_segment_ids(a)) == [
            {"role": "user", "content": "Rewrite this draft: first draft"}
        ]

    def test_init_bearing_root_must_read_channel(self, tmp_path: Path) -> None:
        # Gate B: hist is init-seeded and written by root `a`, but `a` doesn't
        # read @hist -> rejected (it would dispatch without the seed). `b`
        # reconstructs @hist (array slot on a committed writer), so the gate
        # still applies after the G4 narrowing.
        with pytest.raises(NotImplementedError, match="does not splice"):
            _parse(
                tmp_path,
                """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: [{role: user, content: "turn one"}]
      output: hist
    b:
      prompt: ["@hist", {role: user, content: "turn two"}]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: system, content: "be brief"}
""",
            )

    def test_duplicate_channel_splice_gated(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="appears 2 times"):
            _parse(
                tmp_path,
                """
graph:
  nodes:
    a: {prompt: [{role: user, content: q}], output: hist}
    b: {prompt: ["@hist", "@hist"], output: b_out}
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
""",
            )

    def test_reader_writer_nonleading_splice_gated(self, tmp_path: Path) -> None:
        # b writes hist AND reads @hist, but @hist is not the first item.
        with pytest.raises(NotImplementedError, match="must be the first prompt item"):
            _parse(
                tmp_path,
                """
graph:
  nodes:
    a: {prompt: [{role: user, content: q}], output: hist}
    b:
      prompt: [{role: user, content: pre}, "@hist"]
      output: hist
    c: {prompt: ["@hist"], output: c_out}
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: c}
    - {source: c, target: END}
traces:
  - id: t1
""",
            )

    def test_root_writer_nonleading_splice_gated(self, tmp_path: Path) -> None:
        # N6: w is the ROOT writer of init-seeded conv (no ancestor writers),
        # so the reader-side Gate A never sees it; its non-leading "@conv"
        # would survive to _delta_messages, which drops only a LEADING splice
        # and would re-expand the init seed — r would reconstruct
        # [SEED, SYS, SEED, Q1, ...], duplicating SEED and displacing SYS.
        with pytest.raises(NotImplementedError, match="must be the first prompt item"):
            _parse(
                tmp_path,
                """
graph:
  nodes:
    w:
      prompt: [{role: system, content: SYS}, "@conv", {role: user, content: Q1}]
      output: conv
    r:
      prompt: ["@conv", {role: user, content: Q2}]
      output: answer
  edges:
    - {source: START, target: w}
    - {source: w, target: r}
    - {source: r, target: END}
traces:
  - id: t1
    initial_state:
      conv:
        - {role: user, content: SEED}
""",
            )

    def test_root_writer_leading_splice_no_seed_duplication(
        self, tmp_path: Path
    ) -> None:
        # The leading-splice form of the graph above must still parse, and the
        # reader's reconstruction must contain the init seed exactly once, in
        # the same order the writer dispatched it.
        parsed = _parse(
            tmp_path,
            """
graph:
  nodes:
    w:
      prompt: ["@conv", {role: system, content: SYS}, {role: user, content: Q1}]
      output: conv
    r:
      prompt: ["@conv", {role: user, content: Q2}]
      output: answer
  edges:
    - {source: START, target: w}
    - {source: w, target: r}
    - {source: r, target: END}
traces:
  - id: t1
    initial_state:
      conv:
        - {role: user, content: SEED}
""",
        )
        graph = parsed.graphs["t1"]
        w, r = graph.nodes["w"], graph.nodes["r"]
        assert parsed.segment_pool.materialize(read_prompt_segment_ids(w)) == [
            {"role": "user", "content": "SEED"},
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q1"},
        ]
        assembly = _trie(r)["assembly"]
        assert assembly[3] == {"s": {"src": "w"}}  # w's live reply slot
        assert parsed.segment_pool.materialize(read_prompt_segment_ids(r)) == [
            {"role": "user", "content": "SEED"},
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q1"},
            {"role": "user", "content": "Q2"},
        ]

    @pytest.mark.parametrize(
        ("yaml_text", "match"),
        [
            param(
                """
graph:
  nodes:
    a: {prompt: [{role: user, content: q}], output: fan}
    b: {prompt: [{role: user, content: q}], output: hist}
    c: {prompt: [{role: user, content: q}], output: hist}
    d: {prompt: ["@hist"], output: d_out}
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: a, target: c}
    - {source: b, target: d}
    - {source: c, target: d}
    - {source: d, target: END}
traces:
  - id: t1
""",
                "mutually unordered",
                id="parallel_writers_gated",
            ),
            param(
                """
graph:
  nodes:
    a: {prompt: [{role: user, content: q}], output: hist}
    b: {prompt: [{role: user, content: q}], output: hist}
    c: {prompt: ["@hist"], output: c_out}
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: c}
    - {source: c, target: END}
traces:
  - id: t1
""",
                "must chain through reads",
                id="writer_chain_violation",
            ),
        ],
    )  # fmt: skip
    def test_array_gates(self, tmp_path: Path, yaml_text: str, match: str) -> None:
        with pytest.raises(NotImplementedError, match=match):
            _parse(tmp_path, yaml_text)


def test_anchored_producer_to_reader_edge_gated(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="contradictory timing intent"):
        _parse(
            tmp_path,
            """
graph:
  nodes:
    a: {prompt: [{role: user, content: q}], output: a_out}
    x: {prompt: [{role: user, content: q}], output: x_out}
    b:
      prompt:
        - role: user
          content: ["@a_out"]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: x}
    - {source: x, target: b}
    - {source: a, target: b, delay_after_predecessor_start_us: 1000}
    - {source: b, target: END}
traces:
  - id: t1
""",
        )


def test_tstar_gate_rejects_slot_workloads(tmp_path: Path) -> None:
    from aiperf.dataset.graph.workload_detect import (
        _gate_dynamic_slots_vs_tstar,
    )

    parsed = _parse(tmp_path, PLANNER_REVIEWER)

    with pytest.raises(ValueError, match="t\\* snapshot window"):
        _gate_dynamic_slots_vs_tstar(parsed, 0.75)

    _gate_dynamic_slots_vs_tstar(parsed, 0.0)


def test_static_workloads_pass_tstar_gate(tmp_path: Path) -> None:
    from aiperf.dataset.graph.workload_detect import (
        _gate_dynamic_slots_vs_tstar,
    )

    parsed = _parse(
        tmp_path,
        """
graph:
  system: sys
traces:
  - id: t1
    messages:
      - {role: user, content: hello}
""",
    )
    _gate_dynamic_slots_vs_tstar(parsed, 0.75)
