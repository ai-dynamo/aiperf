# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology expansion tests: legacy Conversations -> per-root DagTree.

Every case drives the real ``dag_jsonl`` loader against a fixture file (either
a tmp_path line file or an existing ``tests/fixtures/dag/*.dag.jsonl``) — no
MagicMock for the loader or the Conversation objects. The trees are then
asserted against the lineage / predecessor / join_input contract from the
task brief.
"""

import re
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dag_jsonl.tree import (
    DagTree,
    expand_trees,
    load_dag_conversations,
)

FIXTURES_DIR = Path(__file__).parents[5] / "fixtures" / "dag"


def _write_dag(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return path


def _expand(path: Path) -> list[DagTree]:
    return expand_trees(load_dag_conversations(path, delay_cap_seconds=None))


def _one_tree(path: Path) -> DagTree:
    trees = _expand(path)
    assert len(trees) == 1, [t.trace_id for t in trees]
    return trees[0]


# (a) linear 3-turn single session ------------------------------------------


def test_linear_session_lineage_predecessors_and_delay_ms_to_us(tmp_path):
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "solo",
                "turns": [
                    {"messages": [{"role": "user", "content": "t0"}], "delay": 100.0},
                    {"messages": [{"role": "user", "content": "t1"}], "delay": 5.0},
                    {"messages": [{"role": "user", "content": "t2"}], "delay": 10.0},
                ],
            }
        ],
    )
    tree = _one_tree(path)
    assert tree.trace_id == "solo"
    assert list(tree.nodes) == ["solo:0", "solo:1", "solo:2"]

    n0 = tree.nodes["solo:0"]
    # Root turn-0 delay is NOT honored by the legacy rate loop: no predecessor.
    assert n0.predecessors == []
    assert n0.lineage == []

    n1 = tree.nodes["solo:1"]
    assert n1.predecessors == [("solo:0", 5000)]
    assert n1.lineage == ["solo:0"]
    assert n1.turn.delay == 5.0

    n2 = tree.nodes["solo:2"]
    assert n2.predecessors == [("solo:1", 10000)]
    assert n2.lineage == ["solo:0", "solo:1"]

    assert all(not n.join_inputs for n in tree.nodes.values())


# (b) dag.md minimal FORK example --------------------------------------------


def test_fork_children_instantiated_with_parent_lineage():
    tree = _one_tree(FIXTURES_DIR / "small.dag.jsonl")
    assert tree.trace_id == "root"
    assert set(tree.nodes) == {
        "root:0",
        "branchA:0",
        "branchA:1",
        "branchB:0",
        "branchB:1",
    }

    assert tree.nodes["root:0"].predecessors == []
    assert tree.nodes["root:0"].lineage == []

    a0 = tree.nodes["branchA:0"]
    # FORK child turn-0: predecessor is the forking parent node, delay 0.
    assert a0.predecessors == [("root:0", 0)]
    # child lineage = parent node's lineage + [parent node id].
    assert a0.lineage == ["root:0"]

    b0 = tree.nodes["branchB:0"]
    assert b0.predecessors == [("root:0", 0)]
    assert b0.lineage == ["root:0"]

    a1 = tree.nodes["branchA:1"]
    assert a1.predecessors == [("branchA:0", 0)]
    assert a1.lineage == ["root:0", "branchA:0"]

    # Parent has a single turn -> terminal fork, no join gates anywhere.
    assert all(not n.join_inputs for n in tree.nodes.values())


# (c) background fork ---------------------------------------------------------


def test_background_fork_parent_chain_continues_no_join():
    tree = _one_tree(FIXTURES_DIR / "background_fork.dag.jsonl")
    assert tree.trace_id == "root"
    assert set(tree.nodes) == {"root:0", "root:1", "root:2", "side:0", "side:1"}

    # Parent chain continues past the fork turn, purely sequential.
    assert tree.nodes["root:1"].predecessors == [("root:0", 0)]
    assert tree.nodes["root:2"].predecessors == [("root:1", 0)]

    side0 = tree.nodes["side:0"]
    assert side0.predecessors == [("root:0", 0)]
    assert side0.lineage == ["root:0"]

    side1 = tree.nodes["side:1"]
    assert side1.predecessors == [("side:0", 0)]
    assert side1.lineage == ["root:0", "side:0"]

    # Background fork emits no SPAWN_JOIN gate.
    assert all(not n.join_inputs for n in tree.nodes.values())


# (d) SPAWN with explicit join_at --------------------------------------------


def test_spawn_join_at_gates_only_join_turn(tmp_path):
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "t0"}],
                        "spawns": [{"children": ["kid"], "join_at": 2}],
                    },
                    {"messages": [{"role": "user", "content": "t1"}], "delay": 3.0},
                    {"messages": [{"role": "user", "content": "t2"}], "delay": 7.0},
                ],
            },
            {
                "session_id": "kid",
                "turns": [{"messages": [{"role": "user", "content": "c0"}]}],
            },
        ],
    )
    tree = _one_tree(path)
    assert set(tree.nodes) == {"root:0", "root:1", "root:2", "kid:0"}

    # Intermediate turn runs concurrently with the child: NOT gated, no child pred.
    assert tree.nodes["root:1"].predecessors == [("root:0", 3000)]
    assert tree.nodes["root:1"].join_inputs == []

    # Join turn gates on the child leaf; sequential predecessor is still root:1.
    assert tree.nodes["root:2"].predecessors == [("root:1", 7000)]
    assert tree.nodes["root:2"].join_inputs == ["kid:0"]

    kid = tree.nodes["kid:0"]
    assert kid.predecessors == [("root:0", 0)]
    assert kid.lineage == []
    assert tree.nodes["root:0"].join_inputs == []


# (e) SPAWN template referenced by two parents in one tree -------------------


def test_spawn_template_referenced_twice_instances_distinctly(tmp_path):
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "r0"}],
                        "forks": ["A", "B"],
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
                "session_id": "B",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "b0"}],
                        "spawns": ["kid"],
                    },
                    {"messages": [{"role": "user", "content": "b1"}]},
                ],
            },
            {
                "session_id": "kid",
                "turns": [{"messages": [{"role": "user", "content": "k0"}]}],
            },
        ],
    )
    tree = _one_tree(path)
    # Fresh instance per SPAWN reference: first "kid", second "kid#2".
    assert "kid:0" in tree.nodes
    assert "kid#2:0" in tree.nodes

    assert tree.nodes["kid:0"].predecessors == [("A:0", 0)]
    assert tree.nodes["kid#2:0"].predecessors == [("B:0", 0)]
    assert tree.nodes["kid:0"].lineage == []
    assert tree.nodes["kid#2:0"].lineage == []

    # Auto-join on the spawning parent's next turn resolves to the right instance.
    assert tree.nodes["A:1"].join_inputs == ["kid:0"]
    assert tree.nodes["B:1"].join_inputs == ["kid#2:0"]


# (f) pre_session_spawns ------------------------------------------------------


def test_pre_session_spawn_owner_and_empty_root_predecessors(tmp_path):
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
    tree = _one_tree(path)
    assert tree.trace_id == "root"
    assert set(tree.nodes) == {"root:0", "helper:0"}

    helper = tree.nodes["helper:0"]
    # Owner is the root: its turn-0 has no predecessors, so the copy is empty.
    assert helper.predecessors == []
    assert helper.lineage == []

    assert tree.nodes["root:0"].predecessors == []


# (g) session_id collides with correlation-id framing ------------------------


@pytest.mark.parametrize("bad_char", ["|", "#"])
def test_session_id_with_framing_char_raises_loc_prefixed(tmp_path, bad_char):
    sid = f"root{bad_char}x"
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": sid,
                "turns": [{"messages": [{"role": "user", "content": "r0"}]}],
            }
        ],
    )
    conversations = load_dag_conversations(path, delay_cap_seconds=None)
    with pytest.raises(NotImplementedError, match=rf"conversation '{re.escape(sid)}':"):
        expand_trees(conversations)


# (h) cross-turn SPAWN fan-in: two spawn turns join on one gated turn ----------


def test_two_spawn_turns_fan_in_to_one_join_turn(tmp_path):
    # root turn 0 spawns "a" (join_at 2) and turn 1 spawns "b" (join_at 2), so the
    # gated turn root:2 fans in BOTH post-spawn leaves through two SPAWN_JOIN
    # prerequisites resolved from distinct branches.
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
                    {"messages": [{"role": "user", "content": "t2"}], "delay": 5.0},
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
    tree = _one_tree(path)
    assert set(tree.nodes) == {"root:0", "a:0", "root:1", "b:0", "root:2"}

    # Each spawn child fires at its own spawning parent's credit return (delay 0).
    assert tree.nodes["a:0"].predecessors == [("root:0", 0)]
    assert tree.nodes["b:0"].predecessors == [("root:1", 0)]

    # Only the join turn gates; the intermediate spawn turn does not.
    assert tree.nodes["root:0"].join_inputs == []
    assert tree.nodes["root:1"].join_inputs == []

    # Join turn fans in both leaves (branch/prerequisite declaration order) and
    # still chains sequentially off root:1 carrying its authored delay.
    root2 = tree.nodes["root:2"]
    assert root2.join_inputs == ["a:0", "b:0"]
    assert root2.predecessors == [("root:1", 5000)]
    assert root2.lineage == ["root:0", "root:1"]


# (i) legacy identity: agent_depth / parent_node_id ---------------------------


class TestDagIdentity:
    """Legacy identity semantics mirrored EXACTLY from ``conversation_source``:
    roots ``(0, None)``; FORK/SPAWN child instances ``(owner depth + 1,
    triggering parent node)``; pre-session spawn children ``(1, None)``.
    Every turn of one child instance carries the instance's identity."""

    def test_root_turns_depth_zero_no_parent(self, tmp_path):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "solo",
                    "turns": [
                        {"messages": [{"role": "user", "content": "t0"}]},
                        {"messages": [{"role": "user", "content": "t1"}]},
                    ],
                }
            ],
        )
        tree = _one_tree(path)
        for node in tree.nodes.values():
            assert node.agent_depth == 0
            assert node.parent_node_id is None

    def test_fork_children_all_turns_depth_one_parent_is_forking_node(self):
        tree = _one_tree(FIXTURES_DIR / "small.dag.jsonl")
        assert tree.nodes["root:0"].agent_depth == 0
        assert tree.nodes["root:0"].parent_node_id is None
        for nid in ("branchA:0", "branchA:1", "branchB:0", "branchB:1"):
            assert tree.nodes[nid].agent_depth == 1, nid
            assert tree.nodes[nid].parent_node_id == "root:0", nid

    def test_spawn_child_depth_one_parent_is_spawning_node(self, tmp_path):
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
                    "turns": [{"messages": [{"role": "user", "content": "k0"}]}],
                },
            ],
        )
        tree = _one_tree(path)
        assert tree.nodes["kid:0"].agent_depth == 1
        assert tree.nodes["kid:0"].parent_node_id == "root:0"
        assert tree.nodes["root:1"].agent_depth == 0
        assert tree.nodes["root:1"].parent_node_id is None

    def test_grandchild_depth_two_parent_is_triggering_child_node(self, tmp_path):
        # root forks A; A's turn 0 spawns kid => kid is a grandchild instance.
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
                    "turns": [
                        {"messages": [{"role": "user", "content": "k0"}]},
                        {"messages": [{"role": "user", "content": "k1"}]},
                    ],
                },
            ],
        )
        tree = _one_tree(path)
        assert tree.nodes["root:0"].agent_depth == 0
        for nid in ("A:0", "A:1"):
            assert tree.nodes[nid].agent_depth == 1, nid
            assert tree.nodes[nid].parent_node_id == "root:0", nid
        # ALL turns of the grandchild instance share the instance identity.
        for nid in ("kid:0", "kid:1"):
            assert tree.nodes[nid].agent_depth == 2, nid
            assert tree.nodes[nid].parent_node_id == "A:0", nid

    def test_pre_session_spawn_child_depth_one_no_parent(self, tmp_path):
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
        tree = _one_tree(path)
        helper = tree.nodes["helper:0"]
        assert helper.agent_depth == 1
        assert helper.parent_node_id is None
        assert tree.nodes["root:0"].agent_depth == 0
        assert tree.nodes["root:0"].parent_node_id is None

    def test_repeated_spawn_instances_each_carry_their_own_spawning_parent(
        self, tmp_path
    ):
        path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "spawns": ["kid"],
                        },
                        {
                            "messages": [{"role": "user", "content": "r1"}],
                            "spawns": ["kid"],
                        },
                    ],
                },
                {
                    "session_id": "kid",
                    "turns": [{"messages": [{"role": "user", "content": "k0"}]}],
                },
            ],
        )
        tree = _one_tree(path)
        assert tree.nodes["kid:0"].agent_depth == 1
        assert tree.nodes["kid:0"].parent_node_id == "root:0"
        assert tree.nodes["kid#2:0"].agent_depth == 1
        assert tree.nodes["kid#2:0"].parent_node_id == "root:1"


# (j) non-root pre_session_spawns rejected at load ----------------------------


@pytest.mark.parametrize(
    "owner_lines",
    [
        param(
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "forks": ["owner"],
                        }
                    ],
                },
                {
                    "session_id": "owner",
                    "pre_session_spawns": ["bg"],
                    "turns": [{"messages": [{"role": "user", "content": "o0"}]}],
                },
                {
                    "session_id": "bg",
                    "turns": [{"messages": [{"role": "user", "content": "b0"}]}],
                },
            ],
            id="fork-child-owner",
        ),
        param(
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "r0"}],
                            "spawns": ["owner"],
                        }
                    ],
                },
                {
                    "session_id": "owner",
                    "pre_session_spawns": ["bg"],
                    "turns": [{"messages": [{"role": "user", "content": "o0"}]}],
                },
                {
                    "session_id": "bg",
                    "turns": [{"messages": [{"role": "user", "content": "b0"}]}],
                },
            ],
            id="spawn-child-owner",
        ),
    ],
)  # fmt: skip
def test_non_root_pre_session_spawns_rejected_at_load(tmp_path, owner_lines):
    # A non-root conversation declaring ``pre_session_spawns`` is rejected by the
    # v1 orchestrator gate ("pre-session dispatch requires a root conversation")
    # during load. This is why a pre-session child's copied predecessors are
    # always empty: only a root can own pre-session spawns and its turn-0 has no
    # predecessors, so the generic non-root-owner branch of ``_child_entry_context``
    # is unreachable through the real loader.
    path = _write_dag(tmp_path, owner_lines)
    with pytest.raises(
        NotImplementedError,
        match=r"conversation 'owner' branch 'owner:pre': "
        r"pre-session dispatch requires a root conversation",
    ):
        load_dag_conversations(path, delay_cap_seconds=None)
