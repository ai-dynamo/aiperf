# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lowering tests: DagTree -> unified-segment-store ParsedGraph.

Every case drives the REAL loader + tree expansion against a fixture file
(tmp_path line file or ``tests/fixtures/dag/*.dag.jsonl``) and then asserts
the byte-parity core contract: verbatim raw-message interning with prefix
dedup, live-reply slots for lineage producers, legacy-ordered dispatch
overrides, AND-fan-in gates, and START/END edge conventions. Structural
soundness is proven through ``validator.validate()`` (repo gotcha: node-level
assertions alone let structural bugs slip).
"""

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dag_jsonl.lowering import lower_dag_trees
from aiperf.dataset.graph.adapters.dag_jsonl.tree import (
    expand_trees,
    load_dag_conversations,
)
from aiperf.dataset.graph.models import ChannelRequirement, LlmNode, ParsedGraph
from aiperf.dataset.graph.segment_ir.envelope import read_prompt_segment_ids
from aiperf.dataset.graph.validator import validate

FIXTURES_DIR = Path(__file__).parents[5] / "fixtures" / "dag"

ALL_DAG_FIXTURES = [
    "background_fork.dag.jsonl",
    "bg_fork_fanout.dag.jsonl",
    "bg_fork_nested.dag.jsonl",
    "bg_fork_with_spawn_join.dag.jsonl",
    "full.dag.jsonl",
    "multi_root_single_turn.dag.jsonl",
    "small.dag.jsonl",
    "spawn_minimal.dag.jsonl",
]


def _write_dag(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return path


def _lower(
    path: Path,
    *,
    default_model: str | None = None,
    run_streaming: bool = True,
    endpoint_extra: list[tuple[str, object]] | None = None,
) -> ParsedGraph:
    trees = expand_trees(load_dag_conversations(path, delay_cap_seconds=None))
    return lower_dag_trees(
        trees,
        default_model=default_model,
        run_streaming=run_streaming,
        endpoint_extra=endpoint_extra,
    )


def _trie(node: LlmNode) -> dict:
    return node.metadata["trie"]


# (a) FORK fixture: parsed shape, node order, edges, inputs, record fields ----


class TestForkFixtureStructure:
    def test_parsed_shape_per_tree_graph_and_trace(self):
        parsed = _lower(FIXTURES_DIR / "small.dag.jsonl")
        assert set(parsed.graphs) == {"root"}
        assert parsed.graph is parsed.graphs["root"]
        assert [(t.id, t.graph_ref) for t in parsed.traces] == [("root", "root")]
        assert parsed.segment_pool is not None

    def test_record_conventions_version_provenance_state_outputs(self):
        graph = _lower(FIXTURES_DIR / "small.dag.jsonl").graphs["root"]
        assert graph.version == "2.0"
        assert graph.provenance.source == "dag_jsonl"
        assert graph.provenance.tool not in ("", "manual")
        assert set(graph.state) == {f"{nid}_out" for nid in graph.nodes}
        for nid, node in graph.nodes.items():
            assert node.output == f"{nid}_out"
            assert node.arrival_offset_us == 0
            assert node.streaming is True

    def test_node_order_edges_and_fan_in_inputs(self):
        graph = _lower(FIXTURES_DIR / "small.dag.jsonl").graphs["root"]
        assert list(graph.nodes) == [
            "root:0",
            "branchA:0",
            "branchA:1",
            "branchB:0",
            "branchB:1",
        ]
        edges = {(e.source, e.target): e for e in graph.edges}
        assert set(edges) == {
            ("START", "root:0"),
            ("root:0", "branchA:0"),
            ("branchA:0", "branchA:1"),
            ("root:0", "branchB:0"),
            ("branchB:0", "branchB:1"),
            ("branchA:1", "END"),
            ("branchB:1", "END"),
        }
        assert all(e.delay_after_predecessor_us is None for e in graph.edges)
        assert graph.nodes["root:0"].inputs == []
        assert graph.nodes["branchA:0"].inputs == [
            ChannelRequirement(channel="root:0_out", count=1)
        ]
        assert graph.nodes["branchA:1"].inputs == [
            ChannelRequirement(channel="root:0_out", count=1),
            ChannelRequirement(channel="branchA:0_out", count=1),
        ]

    def test_sequential_delay_ms_lowered_to_edge_us(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "s",
                    "turns": [
                        {"messages": [{"role": "user", "content": "a"}]},
                        {"messages": [{"role": "user", "content": "b"}], "delay": 5.0},
                    ],
                }
            ],
        )
        graph = _lower(path).graphs["s"]
        edges = {(e.source, e.target): e for e in graph.edges}
        assert edges[("s:0", "s:1")].delay_after_predecessor_us == 5000.0


# (b) assembly: verbatim segments, slot placement, prefix parent-chaining -----


class TestAssembly:
    def test_fork_child_assembly_slots_and_verbatim_segments(self):
        parsed = _lower(FIXTURES_DIR / "full.dag.jsonl")
        graph = parsed.graphs["root"]
        node = graph.nodes["branch-a:0"]
        trie = _trie(node)
        sids = trie["prompt_segment_ids"]
        assert trie["assembly"] == [
            {"seg": sids[0]},
            {"seg": sids[1]},
            {"s": {"src": "root:0"}},
            {"seg": sids[2]},
            {"seg": sids[3]},
        ]
        assert parsed.segment_pool.materialize(sids) == [
            {"role": "system", "content": "root system prompt"},
            {"role": "user", "content": "root user prompt"},
            {"role": "user", "content": "branch-a turn-0 user message A"},
            {"role": "user", "content": "branch-a turn-0 user message B"},
        ]
        assert (
            parsed.segment_pool.get(sids[0]).wire_json
            == orjson.dumps(
                {"role": "system", "content": "root system prompt"}
            ).decode()
        ), "raw segment must intern the authored message verbatim"

    def test_parent_chain_threads_through_the_slot_token(self):
        parsed = _lower(FIXTURES_DIR / "full.dag.jsonl")
        sids = _trie(parsed.graphs["root"].nodes["branch-a:0"])["prompt_segment_ids"]
        pool = parsed.segment_pool
        assert pool.get(sids[0]).parent_id is None
        assert pool.get(sids[1]).parent_id == sids[0]
        assert pool.get(sids[2]).parent_id == sids[1]
        assert pool.get(sids[3]).parent_id == sids[2]

    def test_shared_root_prefix_dedups_across_sibling_children(self):
        parsed = _lower(FIXTURES_DIR / "full.dag.jsonl")
        graph = parsed.graphs["root"]
        sids_a = _trie(graph.nodes["branch-a:0"])["prompt_segment_ids"]
        sids_b = _trie(graph.nodes["branch-b:0"])["prompt_segment_ids"]
        assert sids_a[:2] == sids_b[:2]
        assert sids_a[2:] != sids_b[2:]

    def test_lineage_free_node_has_no_assembly_key(self):
        parsed = _lower(FIXTURES_DIR / "full.dag.jsonl")
        trie = _trie(parsed.graphs["root"].nodes["root:0"])
        assert "assembly" not in trie
        assert len(trie["prompt_segment_ids"]) == 2

    def test_spawn_child_starts_fresh_no_assembly_no_lineage_inputs(self):
        parsed = _lower(FIXTURES_DIR / "spawn_minimal.dag.jsonl")
        graph = parsed.graphs["root"]
        child = graph.nodes["spawned-child:0"]
        assert "assembly" not in _trie(child)
        assert child.inputs == []
        assert parsed.segment_pool.materialize(read_prompt_segment_ids(child)) == [
            {"role": "system", "content": "spawn-sys"},
            {"role": "user", "content": "spawn-u"},
        ]


# (c) capture stamps -----------------------------------------------------------


class TestCapture:
    def test_capture_marks_lineage_producers_only(self):
        graph = _lower(FIXTURES_DIR / "small.dag.jsonl").graphs["root"]
        assert _trie(graph.nodes["root:0"])["capture"] is True
        assert _trie(graph.nodes["branchA:0"])["capture"] is True
        assert _trie(graph.nodes["branchB:0"])["capture"] is True
        assert "capture" not in _trie(graph.nodes["branchA:1"])
        assert "capture" not in _trie(graph.nodes["branchB:1"])

    def test_join_gating_leaf_is_not_captured(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "t0"}],
                            "spawns": [{"children": ["kid"], "join_at": 1}],
                        },
                        {"messages": [{"role": "user", "content": "t1"}]},
                    ],
                },
                {
                    "session_id": "kid",
                    "turns": [{"messages": [{"role": "user", "content": "k"}]}],
                },
            ],
        )
        graph = _lower(path).graphs["root"]
        # The join gate needs kid:0's channel COMMIT, never its response text.
        assert "capture" not in _trie(graph.nodes["kid:0"])
        assert _trie(graph.nodes["root:0"])["capture"] is True


# (d) dispatch overrides + native body fields ------------------------------------


class TestDispatchOverrides:
    def test_native_fields_and_overrides_all_fields(self, tmp_path):
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "solo",
                    "turns": [
                        {
                            "model": "authored-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "tools": tools,
                            "max_tokens": 7,
                            "extra": {"temperature": 0.5, "seed": 3},
                        }
                    ],
                }
            ],
        )
        node = _lower(path, default_model="fallback").graphs["solo"].nodes["solo:0"]
        # Model / stream / token cap / tools ride the NATIVE fields (Turn
        # naming); extra_body carries only the merged vendor keys.
        assert node.model == "authored-model"
        assert node.streaming is True
        assert node.max_tokens == 7
        assert node.raw_tools == tools
        assert node.extra_body == {"temperature": 0.5, "seed": 3}

    def test_overrides_default_model_and_omitted_optionals(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "solo",
                    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                }
            ],
        )
        parsed = _lower(path, default_model="fallback-model", run_streaming=False)
        node = parsed.graphs["solo"].nodes["solo:0"]
        assert node.model == "fallback-model"
        assert node.streaming is False
        assert node.max_tokens is None
        assert node.extra_body == {}

    def test_endpoint_extra_lands_before_turn_extra(self, tmp_path):
        # Legacy precedence: ``payload.update(endpoint.extra)`` runs BEFORE
        # ``payload.update(turn extra)``, so the turn value wins on overlap.
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "solo",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 7,
                            "extra": {"top_p": 0.9, "seed": 3},
                        }
                    ],
                }
            ],
        )
        parsed = _lower(
            path,
            default_model="fallback",
            endpoint_extra=[("min_p", 0.05), ("vendor_tag", "run")],
        )
        node = parsed.graphs["solo"].nodes["solo:0"]
        assert node.max_tokens == 7
        overrides = node.extra_body
        assert list(overrides) == [
            "min_p",
            "vendor_tag",
            "top_p",
            "seed",
        ]
        assert overrides["min_p"] == 0.05
        assert overrides["vendor_tag"] == "run"

    def test_endpoint_extra_overlap_turn_extra_wins_at_endpoint_position(
        self, tmp_path
    ):
        # OVERLAP: the same key in endpoint_extra AND the turn's ``extra`` keeps
        # the endpoint_extra POSITION (dict.update first insertion) but takes
        # the TURN value (later update wins) -- exactly the legacy merge.
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "solo",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "extra": {"temperature": 0.5, "top_p": 0.9},
                        }
                    ],
                }
            ],
        )
        parsed = _lower(
            path,
            default_model="fallback",
            endpoint_extra=[("min_p", 0.05), ("temperature", 0.9)],
        )
        overrides = parsed.graphs["solo"].nodes["solo:0"].extra_body
        assert list(overrides) == [
            "min_p",
            "temperature",
            "top_p",
        ], "overlap key must keep its endpoint_extra insertion position"
        assert overrides["temperature"] == 0.5, "turn extra value must win"
        assert overrides["min_p"] == 0.05
        assert overrides["top_p"] == 0.9

    @pytest.mark.parametrize(
        "endpoint_extra",
        [
            param(None, id="no-extras"),
            param([("min_p", 0.05)], id="with-extras"),
        ],
    )  # fmt: skip
    def test_every_node_stamped_endpoint_extra_applied(self, endpoint_extra):
        # Parse-time folding is authoritative even when the run has NO extras:
        # every dag node carries the dispatch stamp so the worker never
        # re-merges ``endpoint.extra`` over the adapter-owned overrides.
        parsed = _lower(FIXTURES_DIR / "small.dag.jsonl", endpoint_extra=endpoint_extra)
        for graph in parsed.graphs.values():
            for node in graph.nodes.values():
                assert node.metadata["dispatch"]["endpoint_extra_applied"] is True


# (e) tools inheritance ----------------------------------------------------------


class TestToolsInheritance:
    def test_nearest_lineage_tools_win(self, tmp_path):
        t0 = [{"type": "function", "function": {"name": "t0"}}]
        t1 = [{"type": "function", "function": {"name": "t1"}}]
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "s",
                    "turns": [
                        {"messages": [{"role": "user", "content": "a"}], "tools": t0},
                        {"messages": [{"role": "user", "content": "b"}], "tools": t1},
                        {"messages": [{"role": "user", "content": "c"}]},
                    ],
                }
            ],
        )
        graph = _lower(path).graphs["s"]
        assert graph.nodes["s:0"].raw_tools == t0
        assert graph.nodes["s:1"].raw_tools == t1
        assert graph.nodes["s:2"].raw_tools == t1

    def test_fork_child_inherits_spawn_child_does_not(self, tmp_path):
        tools = [{"type": "function", "function": {"name": "root-tool"}}]
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "tools": tools,
                            "forks": ["forked"],
                            "spawns": ["spawned"],
                        }
                    ],
                },
                {
                    "session_id": "forked",
                    "turns": [{"messages": [{"role": "user", "content": "f0"}]}],
                },
                {
                    "session_id": "spawned",
                    "turns": [{"messages": [{"role": "user", "content": "s0"}]}],
                },
            ],
        )
        graph = _lower(path).graphs["root"]
        assert graph.nodes["forked:0"].raw_tools == tools
        assert graph.nodes["spawned:0"].raw_tools is None


# (f) structural soundness via the real validator --------------------------------


class TestValidate:
    @pytest.mark.parametrize(
        "fixture_name",
        [param(name, id=name) for name in ALL_DAG_FIXTURES],
    )  # fmt: skip
    def test_validate_no_issues_for_every_fixture(self, fixture_name):
        parsed = _lower(FIXTURES_DIR / fixture_name)
        assert validate(parsed) == []

    def test_validate_no_issues_prespawn_and_join(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "pre_session_spawns": ["helper"],
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "spawns": [{"children": ["kidA", "kidB"], "join_at": 1}],
                        },
                        {"messages": [{"role": "user", "content": "r1"}]},
                    ],
                },
                {
                    "session_id": "helper",
                    "turns": [{"messages": [{"role": "user", "content": "h0"}]}],
                },
                {
                    "session_id": "kidA",
                    "turns": [{"messages": [{"role": "user", "content": "a"}]}],
                },
                {
                    "session_id": "kidB",
                    "turns": [{"messages": [{"role": "user", "content": "b"}]}],
                },
            ],
        )
        assert validate(_lower(path)) == []


# (g) spawn-join gating ----------------------------------------------------------


class TestSpawnJoin:
    def test_join_turn_inputs_gate_both_leaves_and_sequential_edge_exists(
        self, tmp_path
    ):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "t0"}],
                            "spawns": [{"children": ["kidA", "kidB"], "join_at": 1}],
                        },
                        {"messages": [{"role": "user", "content": "t1"}], "delay": 5.0},
                    ],
                },
                {
                    "session_id": "kidA",
                    "turns": [{"messages": [{"role": "user", "content": "a"}]}],
                },
                {
                    "session_id": "kidB",
                    "turns": [{"messages": [{"role": "user", "content": "b"}]}],
                },
            ],
        )
        parsed = _lower(path)
        graph = parsed.graphs["root"]
        assert graph.nodes["root:1"].inputs == [
            ChannelRequirement(channel="kidA:0_out", count=1),
            ChannelRequirement(channel="kidB:0_out", count=1),
            ChannelRequirement(channel="root:0_out", count=1),
        ]
        edges = {(e.source, e.target): e for e in graph.edges}
        assert edges[("root:0", "root:1")].delay_after_predecessor_us == 5000.0
        assert edges[("root:0", "kidA:0")].delay_after_predecessor_us is None
        assert edges[("root:0", "kidB:0")].delay_after_predecessor_us is None
        assert validate(parsed) == []

    def test_cross_turn_spawn_fan_in_gates_both_leaves_on_join_turn(self, tmp_path):
        # Two SPAWN branches on two DIFFERENT turns (turn 0 spawns "a", turn 1
        # spawns "b"), both joining on turn 2. The join turn's AND-fan-in must
        # carry both post-spawn leaf channels plus its own lineage producers,
        # and the sequential edge from the previous turn must still exist.
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "t0"}],
                            "spawns": [{"children": ["a"], "join_at": 2}],
                        },
                        {
                            "messages": [{"role": "user", "content": "t1"}],
                            "spawns": [{"children": ["b"], "join_at": 2}],
                            "delay": 3.0,
                        },
                        {
                            "messages": [{"role": "user", "content": "t2"}],
                            "delay": 5.0,
                        },
                    ],
                },
                {
                    "session_id": "a",
                    "turns": [{"messages": [{"role": "user", "content": "a0"}]}],
                },
                {
                    "session_id": "b",
                    "turns": [{"messages": [{"role": "user", "content": "b0"}]}],
                },
            ],
        )
        parsed = _lower(path)
        graph = parsed.graphs["root"]
        # Both spawn leaves gate the join turn, followed by its sequential
        # lineage producers (root:0, root:1).
        assert graph.nodes["root:2"].inputs == [
            ChannelRequirement(channel="a:0_out", count=1),
            ChannelRequirement(channel="b:0_out", count=1),
            ChannelRequirement(channel="root:0_out", count=1),
            ChannelRequirement(channel="root:1_out", count=1),
        ]
        edges = {(e.source, e.target): e for e in graph.edges}
        # The sequential edge into the join turn carries the authored delay; the
        # two spawn-dispatch edges fire immediately (delay-free).
        assert edges[("root:1", "root:2")].delay_after_predecessor_us == 5000.0
        assert edges[("root:0", "a:0")].delay_after_predecessor_us is None
        assert edges[("root:1", "b:0")].delay_after_predecessor_us is None
        assert validate(parsed) == []

    def test_pre_session_spawn_child_gets_start_edge(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "pre_session_spawns": ["helper"],
                    "turns": [{"messages": [{"role": "user", "content": "r0"}]}],
                },
                {
                    "session_id": "helper",
                    "turns": [{"messages": [{"role": "user", "content": "h0"}]}],
                },
            ],
        )
        parsed = _lower(path)
        graph = parsed.graphs["root"]
        pairs = {(e.source, e.target) for e in graph.edges}
        assert pairs == {
            ("START", "root:0"),
            ("START", "helper:0"),
            ("root:0", "END"),
            ("helper:0", "END"),
        }
        assert validate(parsed) == []


# (g2) repeated SPAWN instances: distinct #n nodes, shared segment dedup ---------


class TestRepeatedSpawnInstances:
    def test_same_template_spawned_from_two_turns_gets_distinct_instance_nodes(
        self, tmp_path
    ):
        """One SPAWN template fired from two different turns => ``kid`` + ``kid#2``.

        The first instance is bare, the second carries the ``#2`` suffix (tree
        encounter order). Both are fresh-context spawn children with identical
        authored bytes, so their prompt segments dedup to the SAME pool ids -- the
        deferred ``#n``-channel proof for the multiset parity comparator.
        """
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "model": "Qwen3-0.6B",
                            "messages": [{"role": "user", "content": "plan"}],
                            "max_tokens": 32,
                            "spawns": ["kid"],
                        },
                        {
                            "model": "Qwen3-0.6B",
                            "messages": [{"role": "user", "content": "revise"}],
                            "max_tokens": 32,
                            "spawns": ["kid"],
                        },
                    ],
                },
                {
                    "session_id": "kid",
                    "turns": [
                        {
                            "model": "Qwen3-0.6B",
                            "messages": [{"role": "user", "content": "work"}],
                            "max_tokens": 16,
                        }
                    ],
                },
            ],
        )
        parsed = _lower(path)
        graph = parsed.graphs["root"]

        # Both spawn instances exist as distinct nodes.
        assert "kid:0" in graph.nodes
        assert "kid#2:0" in graph.nodes
        assert set(graph.nodes) == {"root:0", "root:1", "kid:0", "kid#2:0"}

        # Each instance owns a distinct per-node output channel keyed by node id.
        assert graph.nodes["kid:0"].output == "kid:0_out"
        assert graph.nodes["kid#2:0"].output == "kid#2:0_out"
        assert "kid:0_out" in graph.state
        assert "kid#2:0_out" in graph.state

        # Identical authored bytes + fresh context => the content-addressed pool
        # hands both instances the SAME prompt segment ids (shared dedup).
        kid_sids = read_prompt_segment_ids(graph.nodes["kid:0"])
        kid2_sids = read_prompt_segment_ids(graph.nodes["kid#2:0"])
        assert kid_sids == kid2_sids
        assert parsed.segment_pool.materialize(kid_sids) == [
            {"role": "user", "content": "work"}
        ]

        # Structural soundness through the real validator (node-level asserts
        # alone let structural bugs slip -- repo gotcha).
        assert validate(parsed) == []


# (g3) dag identity metadata stamp ----------------------------------------------


class TestDagIdentityMetadata:
    """Lowering stamps ``metadata["dag"]`` with the tree's instance identity
    (``agent_depth`` / ``parent_node``) on every node, coexisting with the
    ``dispatch`` and ``trie`` metadata stamps."""

    def test_fork_fixture_stamps_root_and_child_identity(self):
        graph = _lower(FIXTURES_DIR / "small.dag.jsonl").graphs["root"]
        assert graph.nodes["root:0"].metadata["dag"] == {
            "agent_depth": 0,
            "parent_node": None,
        }
        for nid in ("branchA:0", "branchA:1", "branchB:0", "branchB:1"):
            assert graph.nodes[nid].metadata["dag"] == {
                "agent_depth": 1,
                "parent_node": "root:0",
            }, nid

    def test_dag_stamp_coexists_with_dispatch_and_trie_stamps(self):
        graph = _lower(FIXTURES_DIR / "small.dag.jsonl").graphs["root"]
        for node in graph.nodes.values():
            assert node.metadata["dispatch"] == {"endpoint_extra_applied": True}
            assert "prompt_segment_ids" in node.metadata["trie"]
            assert "dag" in node.metadata

    def test_prespawn_child_stamped_depth_one_no_parent(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "pre_session_spawns": ["helper"],
                    "turns": [{"messages": [{"role": "user", "content": "r0"}]}],
                },
                {
                    "session_id": "helper",
                    "turns": [{"messages": [{"role": "user", "content": "h0"}]}],
                },
            ],
        )
        graph = _lower(path).graphs["root"]
        assert graph.nodes["helper:0"].metadata["dag"] == {
            "agent_depth": 1,
            "parent_node": None,
        }
        assert graph.nodes["root:0"].metadata["dag"] == {
            "agent_depth": 0,
            "parent_node": None,
        }

    def test_grandchild_stamped_depth_two(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "forks": ["A"],
                        }
                    ],
                },
                {
                    "session_id": "A",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "a0"}],
                            "spawns": ["kid"],
                        },
                        {"messages": [{"role": "user", "content": "a1"}]},
                    ],
                },
                {
                    "session_id": "kid",
                    "turns": [{"messages": [{"role": "user", "content": "k0"}]}],
                },
            ],
        )
        graph = _lower(path).graphs["root"]
        assert graph.nodes["kid:0"].metadata["dag"] == {
            "agent_depth": 2,
            "parent_node": "A:0",
        }


# (h) determinism + multi-root ---------------------------------------------------


class TestDeterminismAndMultiRoot:
    def test_two_fresh_parses_lower_identically(self):
        path = FIXTURES_DIR / "full.dag.jsonl"
        first = _lower(path)
        second = _lower(path)
        ga, gb = first.graphs["root"], second.graphs["root"]
        assert list(ga.nodes) == list(gb.nodes)
        for nid in ga.nodes:
            assert _trie(ga.nodes[nid]) == _trie(gb.nodes[nid])
            assert ga.nodes[nid].extra_body == gb.nodes[nid].extra_body
        assert ga.edges == gb.edges
        assert sorted(first.segment_pool.by_id) == sorted(second.segment_pool.by_id)

    def test_multi_root_per_tree_graphs_share_one_pool(self):
        parsed = _lower(FIXTURES_DIR / "multi_root_single_turn.dag.jsonl")
        assert list(parsed.graphs) == ["r1", "r2"]
        assert [(t.id, t.graph_ref) for t in parsed.traces] == [
            ("r1", "r1"),
            ("r2", "r2"),
        ]
        assert parsed.graph is parsed.graphs["r1"]
        for graph in parsed.graphs.values():
            for node in graph.nodes.values():
                for sid in read_prompt_segment_ids(node):
                    parsed.segment_pool.get(sid)
        assert validate(parsed) == []

    def test_empty_trees_raise_loc_prefixed(self):
        with pytest.raises(NotImplementedError, match=r"^dag_jsonl workload: "):
            lower_dag_trees([], default_model=None, run_streaming=True)
