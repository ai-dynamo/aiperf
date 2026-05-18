# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full adversarial coverage for ``DagJsonlLoader``.

Picks orthogonal cases not covered by the existing ``test_dag_jsonl_*.py``
suite: ``join_at`` boundary values, programmatic-bypass edge cases,
``pre_session_spawns`` reachability and cycle interactions, mixed-form
``spawns`` groups, branch_id collisions across conversations,
session_id and child_conversation_id surface oddities, file-format
strictness, and round-trip idempotency through ``Conversation.to_metadata``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.dag_jsonl_models import DagSpawn
from aiperf.plugin.enums import DatasetSamplingStrategy


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


# ---------------------------------------------------------------------------
# 1. join_at boundary values
# ---------------------------------------------------------------------------


def _root_with_join_at(join_at: int, num_turns: int = 4) -> list[dict]:
    return [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "u0"}],
                    "spawns": [{"children": ["child"], "join_at": join_at}],
                },
                *[
                    {"messages": [{"role": "user", "content": f"u{i}"}]}
                    for i in range(1, num_turns)
                ],
            ],
        },
        {
            "session_id": "child",
            "turns": [{"messages": [{"role": "user", "content": "c"}]}],
        },
    ]


def test_join_at_zero_self_gate_rejected(tmp_path: Path):
    """``join_at=0`` on the spawning turn 0 is a self-gate; rejected as not
    strictly-prior."""
    path = _write(tmp_path, _root_with_join_at(0))
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_join_at_equals_idx_rejected(tmp_path: Path):
    """``join_at=idx`` on the spawning turn is a self-gate; rejected."""
    # Spawn on turn 1, join_at=1 (self).
    data = [
        {
            "session_id": "root",
            "turns": [
                {"messages": [{"role": "user", "content": "u0"}]},
                {
                    "messages": [{"role": "user", "content": "u1"}],
                    "spawns": [{"children": ["child"], "join_at": 1}],
                },
                {"messages": [{"role": "user", "content": "u2"}]},
            ],
        },
        {
            "session_id": "child",
            "turns": [{"messages": [{"role": "user", "content": "c"}]}],
        },
    ]
    path = _write(tmp_path, data)
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_join_at_backward_rejected(tmp_path: Path):
    """``join_at = idx - 1`` (backwards reference) is rejected."""
    data = [
        {
            "session_id": "root",
            "turns": [
                {"messages": [{"role": "user", "content": "u0"}]},
                {
                    "messages": [{"role": "user", "content": "u1"}],
                    "spawns": [{"children": ["child"], "join_at": 0}],
                },
            ],
        },
        {
            "session_id": "child",
            "turns": [{"messages": [{"role": "user", "content": "c"}]}],
        },
    ]
    path = _write(tmp_path, data)
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_join_at_equals_num_turns_off_by_one_rejected(tmp_path: Path):
    """``join_at == num_turns`` (one past last index) is rejected."""
    path = _write(tmp_path, _root_with_join_at(4, num_turns=4))
    with pytest.raises(DagLoadError, match="out of range"):
        DagJsonlLoader(filename=path).load()


def test_join_at_equals_last_turn_accepted(tmp_path: Path):
    """``join_at == num_turns - 1`` (the last turn) is accepted; prereq
    appears on that turn."""
    path = _write(tmp_path, _root_with_join_at(3, num_turns=4))
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.turns[3].prerequisites) == 1
    assert root.turns[3].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN


def test_join_at_negative_one_rejected(tmp_path: Path):
    """``join_at = -1`` fails the strictly-greater check (always <= idx)."""
    path = _write(tmp_path, _root_with_join_at(-1))
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_join_at_large_negative_rejected(tmp_path: Path):
    """A grossly negative ``join_at`` is rejected."""
    path = _write(tmp_path, _root_with_join_at(-100))
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_join_at_huge_positive_rejected(tmp_path: Path):
    """A huge ``join_at`` past the conversation length is rejected."""
    path = _write(tmp_path, _root_with_join_at(999_999))
    with pytest.raises(DagLoadError, match="out of range"):
        DagJsonlLoader(filename=path).load()


def test_join_at_float_pydantic_rejects():
    """Float ``join_at`` is rejected by pydantic at model construction."""
    with pytest.raises(ValidationError):
        DagSpawn(children=["c"], join_at=1.5)  # type: ignore[arg-type]


def test_join_at_string_coerced_by_pydantic():
    """Pydantic coerces a numeric string ``join_at`` to ``int``. Document
    this behavior; loader semantics still apply (strictly-prior).

    This is a permissive-coercion gotcha: authors who pass ``"5"`` won't be
    yelled at, but a non-numeric string ('abc') is rejected.
    """
    s = DagSpawn(children=["c"], join_at="5")  # type: ignore[arg-type]
    assert s.join_at == 5
    with pytest.raises(ValidationError):
        DagSpawn(children=["c"], join_at="abc")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. pre_session_spawns boundary cases
# ---------------------------------------------------------------------------


def test_pre_session_spawns_default_no_branch(tmp_path: Path):
    """A conversation with no ``pre_session_spawns`` key emits no pre branch.

    Companion to the existing default test, this asserts no `:pre` branch
    is created and the conversation is still flagged as a root.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = convs[0]
    assert all(b.branch_id != "root:pre" for b in root.branches)
    assert root.is_root is True


def test_pre_session_spawns_transitive_chain_rejected(tmp_path: Path):
    """A's pre_session_spawns includes B; B's pre_session_spawns includes C.

    A pre-session branch attached to a non-root conversation would fire
    at phase start (before its parent's session exists) and bypass the
    intended DAG topology. ``validate_for_orchestrator_v1`` now rejects
    via ``is_root`` rather than the looser ``agent_depth > 0`` check —
    SPAWN children keep ``agent_depth == 0`` but carry ``is_root=False``,
    so the agent-depth predicate missed this case.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "A",
                "pre_session_spawns": ["B"],
                "turns": [{"messages": [{"role": "user", "content": "a"}]}],
            },
            {
                "session_id": "B",
                "pre_session_spawns": ["C"],
                "turns": [{"messages": [{"role": "user", "content": "b"}]}],
            },
            {
                "session_id": "C",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    with pytest.raises(NotImplementedError, match="requires a root conversation"):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_cycle_rejected(tmp_path: Path):
    """A.pre includes B, B.pre includes A — DFS cycle detected at load."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "A",
                "pre_session_spawns": ["B"],
                "turns": [{"messages": [{"role": "user", "content": "a"}]}],
            },
            {
                "session_id": "B",
                "pre_session_spawns": ["A"],
                "turns": [{"messages": [{"role": "user", "content": "b"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="cycle detected"):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_self_cycle_rejected(tmp_path: Path):
    """A.pre includes A — self-cycle detected at load."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "A",
                "pre_session_spawns": ["A"],
                "turns": [{"messages": [{"role": "user", "content": "a"}]}],
            }
        ],
    )
    with pytest.raises(DagLoadError, match="cycle detected"):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_unknown_child_rejected(tmp_path: Path):
    """``pre_session_spawns`` referencing an undeclared session_id is
    rejected with the same error path as a regular branch reference."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["nope"],
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            }
        ],
    )
    with pytest.raises(DagLoadError, match="branch target 'nope' not declared"):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_large_list_accepted(tmp_path: Path):
    """A 200-entry ``pre_session_spawns`` list loads cleanly. Sanity check
    that O(N) lookup paths don't choke on moderate fan-out."""
    n = 200
    children = [f"c{i}" for i in range(n)]
    lines = [
        {
            "session_id": "root",
            "pre_session_spawns": children,
            "turns": [{"messages": [{"role": "user", "content": "u"}]}],
        }
    ]
    for cid in children:
        lines.append(
            {
                "session_id": cid,
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            }
        )
    path = _write(tmp_path, lines)
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert len(root.branches[0].child_conversation_ids) == n


def test_turn0_with_both_pre_session_and_turn0_spawns_emits_two_branches(
    tmp_path: Path,
):
    """Turn 0 with BOTH ``pre_session_spawns`` and turn-0 ``spawns`` produces
    two separate branches: ``root:pre`` (dispatch_timing="pre", background)
    and ``root:0`` (dispatch_timing="post"). Both share turn 0's branch_ids."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["pre_c"],
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["post_c"],
                    },
                    {"messages": [{"role": "user", "content": "u2"}]},
                ],
            },
            {
                "session_id": "pre_c",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "post_c",
                "turns": [{"messages": [{"role": "user", "content": "y"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    by_id = {b.branch_id: b for b in root.branches}
    assert "root:pre" in by_id
    assert "root:0" in by_id
    assert by_id["root:pre"].dispatch_timing == "pre"
    assert by_id["root:0"].dispatch_timing == "post"
    assert root.turns[0].branch_ids == ["root:pre", "root:0"]


# ---------------------------------------------------------------------------
# 3. spawns mixing legacy strings and DagSpawn objects on same turn
# ---------------------------------------------------------------------------


def test_spawns_mixed_string_and_object_emit_distinct_branches(tmp_path: Path):
    """``spawns`` mixing a bare string and a ``DagSpawn`` object on the same
    turn emit two branches with distinct branch_ids."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": [
                            "child_a",
                            {"children": ["child_b"], "join_at": 2},
                        ],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                    {"messages": [{"role": "user", "content": "u2"}]},
                ],
            },
            {
                "session_id": "child_a",
                "turns": [{"messages": [{"role": "user", "content": "a"}]}],
            },
            {
                "session_id": "child_b",
                "turns": [{"messages": [{"role": "user", "content": "b"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    branch_ids = {b.branch_id for b in root.branches}
    assert "root:0:spawn" in branch_ids
    assert "root:0:spawn1" in branch_ids
    # Legacy string emits prereq on idx+1=1; object emits on join_at=2.
    p_turns_with_root_spawn = [
        idx for idx, t in enumerate(root.turns) if t.prerequisites
    ]
    assert sorted(p_turns_with_root_spawn) == [1, 2]


# ---------------------------------------------------------------------------
# 4. DagSpawn invariants
# ---------------------------------------------------------------------------


def test_dag_spawn_empty_children_rejected_by_pydantic():
    """``DagSpawn(children=[])`` is rejected (min_length=1)."""
    with pytest.raises(ValidationError):
        DagSpawn(children=[], join_at=1)


def test_dag_spawn_duplicate_children_rejected_by_loader(
    tmp_path: Path,
):
    """Duplicate child_conversation_ids in a DagSpawn are rejected.

    Without dedupe the orchestrator would dispatch the same child twice and
    the SPAWN_JOIN gate would expect two completions.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "A",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "a"}],
                        "spawns": [{"children": ["B", "B"], "join_at": 1}],
                    },
                    {"messages": [{"role": "user", "content": "a2"}]},
                ],
            },
            {
                "session_id": "B",
                "turns": [{"messages": [{"role": "user", "content": "b"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="duplicate child_conversation_id"):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 5. SPAWN child shared by multiple parents (allowed) vs FORK (rejected)
# ---------------------------------------------------------------------------


def test_spawn_child_shared_by_multiple_parents_accepted(tmp_path: Path):
    """SPAWN children may have multiple parents (fresh-context templates)."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "p1",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["shared"],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "p2",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["shared"],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "shared",
                "turns": [{"messages": [{"role": "user", "content": "s"}]}],
            },
        ],
    )
    DagJsonlLoader(filename=path).load()


def test_fork_child_shared_by_multiple_parents_rejected(tmp_path: Path):
    """FORK children may NOT have multiple parents (single-parent invariant)."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "p1",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["shared"],
                    }
                ],
            },
            {
                "session_id": "p2",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["shared"],
                    }
                ],
            },
            {
                "session_id": "shared",
                "turns": [{"messages": [{"role": "user", "content": "s"}]}],
            },
        ],
    )
    with pytest.raises(
        DagLoadError, match="FORK-mode children require a single parent"
    ):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 6. Branch_id surface oddities
# ---------------------------------------------------------------------------


def test_session_id_with_colons_branch_id_resolves(tmp_path: Path):
    """A session_id containing ':' still produces a parseable branch_id;
    ``_turn_idx_from_branch_id`` rsplit-anchors on the trailing numeric."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "ns:tenant:user",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["c"],
                    },
                    {"messages": [{"role": "user", "content": "u"}]},
                ],
            },
            {
                "session_id": "c",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    parent = next(c for c in convs if c.session_id == "ns:tenant:user")
    assert parent.branches[0].branch_id == "ns:tenant:user:0"


def test_session_id_unicode_accepted(tmp_path: Path):
    """Unicode session_ids and child references are accepted."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "セッション",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["子"],
                    },
                    {"messages": [{"role": "user", "content": "u"}]},
                ],
            },
            {
                "session_id": "子",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    parent = next(c for c in convs if c.session_id == "セッション")
    assert parent.branches[0].branch_id == "セッション:0"


# ---------------------------------------------------------------------------
# 7. JSONL file format strictness
# ---------------------------------------------------------------------------


def test_jsonl_with_blank_lines_skipped(tmp_path: Path):
    """Blank lines in a JSONL file are skipped (lenient to author whitespace)."""
    path = tmp_path / "dag.jsonl"
    payload = json.dumps(
        {"session_id": "A", "turns": [{"messages": [{"role": "user", "content": "u"}]}]}
    )
    path.write_text(f"\n\n{payload}\n\n")
    convs = DagJsonlLoader(filename=path).load()
    assert [c.session_id for c in convs] == ["A"]


def test_jsonl_with_trailing_whitespace_strict_strip(tmp_path: Path):
    """Trailing whitespace on each line is stripped before JSON parse."""
    path = tmp_path / "dag.jsonl"
    payload = json.dumps(
        {"session_id": "A", "turns": [{"messages": [{"role": "user", "content": "u"}]}]}
    )
    path.write_text(payload + "   \t\r\n")
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].session_id == "A"


def test_jsonl_bom_accepted(tmp_path: Path):
    """A leading UTF-8 BOM is stripped before parsing so files saved by
    Windows/Excel tools load cleanly."""
    path = tmp_path / "dag.jsonl"
    body = json.dumps(
        {"session_id": "A", "turns": [{"messages": [{"role": "user", "content": "u"}]}]}
    ).encode()
    path.write_bytes(b"\xef\xbb\xbf" + body + b"\n")
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].session_id == "A"


# ---------------------------------------------------------------------------
# 8. Conversation/turn shape boundary cases
# ---------------------------------------------------------------------------


def test_conversation_zero_turns_pydantic_rejects(tmp_path: Path):
    """A conversation with ``turns: []`` is rejected by pydantic
    (``min_length=1``)."""
    path = _write(
        tmp_path,
        [{"session_id": "A", "turns": []}],
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(filename=path).load()


def test_forks_and_spawns_pointing_at_same_child_emits_two_branches(
    tmp_path: Path,
):
    """When ``forks`` and ``spawns`` on the SAME turn name the same child id,
    both branches are emitted with disambiguated suffixes.

    The child is registered as both a FORK target (sticky-routed) and a
    SPAWN target (fresh-context). Loader does not currently warn; test
    documents the silent acceptance.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["X"],
                        "spawns": ["X"],
                    },
                ],
            },
            {
                "session_id": "X",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    modes = {b.branch_id: b.mode for b in root.branches}
    assert modes == {
        "root:0:fork": ConversationBranchMode.FORK,
        "root:0:spawn": ConversationBranchMode.SPAWN,
    }


# ---------------------------------------------------------------------------
# 9. System message placement (footgun must still trigger after refactor)
# ---------------------------------------------------------------------------


def test_system_message_on_fork_child_turn0_still_rejected(tmp_path: Path):
    """The Qwen3-VL footgun: a FORK child's turn 0 may not contain a system
    message because pure-append puts it at index > 0 in the wire body. Rule
    must persist post-Phase-3."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "parent",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["child"],
                    }
                ],
            },
            {
                "session_id": "child",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "you are helpful"},
                            {"role": "user", "content": "u"},
                        ]
                    }
                ],
            },
        ],
    )
    with pytest.raises(
        DagLoadError, match="non-root turns may not contain a 'system' message"
    ):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 10. Multi-feature mix on a single turn (Phase 3 acceptance)
# ---------------------------------------------------------------------------


def test_turn_with_prerequisites_via_explicit_metadata_phase3_accepted():
    """A turn carrying multiple SPAWN_JOIN prerequisites (fan-in from
    earlier turns) is accepted post-Phase-3.

    Built directly through ConversationMetadata because the loader's
    shorthand only auto-generates one prereq per branch.
    """
    branches = [
        ConversationBranchInfo(
            branch_id=f"r:{i}",
            child_conversation_ids=[f"c{i}"],
            mode=ConversationBranchMode.SPAWN,
        )
        for i in range(5)
    ]
    turns = []
    # Turn 0..4 each declare a SPAWN branch.
    for i in range(5):
        turns.append(TurnMetadata(branch_ids=[f"r:{i}"]))
    # Turn 5 fans in all five.
    turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=f"r:{i}")
                for i in range(5)
            ]
        )
    )
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(conversation_id="r", turns=turns, branches=branches),
            *(
                ConversationMetadata(conversation_id=f"c{i}", turns=[TurnMetadata()])
                for i in range(5)
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )
    # Phase 3 lifted the multi-source-gate restriction.
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 11. extra deeply nested round-trip
# ---------------------------------------------------------------------------


def test_extra_deeply_nested_round_trips_through_loader(tmp_path: Path):
    """Deeply nested ``extra`` survives the loader unchanged."""
    extras = {
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "logit_bias": {"123": -100, "456": 50},
        },
        "vendor": {
            "ignore_eos": True,
            "min_tokens": 4,
            "stop": ["<|eot|>", "<|end|>"],
        },
        "nested": [{"a": [1, 2, [3, [4]]]}],
    }
    path = _write(
        tmp_path,
        [
            {
                "session_id": "A",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "extra": extras,
                    }
                ],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].extra_body == extras


# ---------------------------------------------------------------------------
# 12. Idempotent JSON round-trip of complex DatasetMetadata
# ---------------------------------------------------------------------------


def test_dataset_metadata_round_trip_idempotent():
    """A complex DatasetMetadata serializes, deserializes, and re-validates
    bit-identically. Guards against drift in field defaults that would break
    on-the-wire schemas."""
    branches = [
        ConversationBranchInfo(
            branch_id="r:pre",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        ),
        ConversationBranchInfo(
            branch_id="r:0",
            child_conversation_ids=["c2"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:pre", "r:0"], has_forks=True),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:pre"
                            )
                        ]
                    ),
                ],
                branches=branches,
            ),
            ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c2", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )
    blob = md.model_dump(mode="json")
    md2 = DatasetMetadata.model_validate(blob)
    blob2 = md2.model_dump(mode="json")
    assert blob == blob2
    # Specifically assert that dispatch_timing="pre" survives the round-trip
    # (Phase 2b field).
    pre_branch = next(
        b for b in md2.conversations[0].branches if b.branch_id == "r:pre"
    )
    assert pre_branch.dispatch_timing == "pre"
