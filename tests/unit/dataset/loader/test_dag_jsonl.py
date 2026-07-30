# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError


def write_lines(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return p


def _turn(content: str, **extras) -> dict:
    """Build a minimal flat turn with a single user message + structural extras."""
    t: dict = {"messages": [{"role": "user", "content": content}]}
    t.update(extras)
    return t


def _msg(content: str) -> dict:
    """A single user message object."""
    return {"role": "user", "content": content}


def test_loads_simple_fork(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [_turn("p1", forks=["a", "b"])],
            },
            {"session_id": "a", "turns": [_turn("pa")]},
            {"session_id": "b", "turns": [_turn("pb")]},
        ],
    )
    loader = DagJsonlLoader(path)
    conversations = loader.load()
    by_id = {c.session_id: c for c in conversations}
    assert set(by_id) == {"root", "a", "b"}
    root = by_id["root"]
    assert root.context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
    assert root.turns[0].branch_ids == ["root:0"]
    assert root.branches[0].child_conversation_ids == ["a", "b"]
    assert root.branches[0].branch_id == "root:0"
    assert loader.root_session_ids() == {"root"}


def test_missing_messages_raises(tmp_path):
    path = write_lines(
        tmp_path,
        [{"session_id": "root", "turns": [{"forks": ["a"]}]}],
    )
    with pytest.raises(DagLoadError, match=r"messages.*Field required"):
        DagJsonlLoader(path).load()


def test_messages_not_list_raises(tmp_path):
    path = write_lines(
        tmp_path,
        [{"session_id": "root", "turns": [{"messages": "oops"}]}],
    )
    with pytest.raises(DagLoadError, match=r"messages"):
        DagJsonlLoader(path).load()


def test_empty_messages_raises(tmp_path):
    path = write_lines(
        tmp_path,
        [{"session_id": "root", "turns": [{"messages": []}]}],
    )
    with pytest.raises(DagLoadError, match=r"messages"):
        DagJsonlLoader(path).load()


def test_unknown_top_level_turn_key_rejected(tmp_path):
    """extra='forbid' on DagTurn catches typos like max_token (no s)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_token": 300,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError, match=r"max_token"):
        DagJsonlLoader(path).load()


def test_unknown_top_level_conversation_key_rejected(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
                "not_a_real_field": True,
            }
        ],
    )
    with pytest.raises(DagLoadError, match=r"not_a_real_field"):
        DagJsonlLoader(path).load()


def test_message_missing_role_rejected(tmp_path):
    """Each message dict must have a 'role' key (matches MooncakeTrace)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [{"messages": [{"content": "x"}]}],
            }
        ],
    )
    with pytest.raises(DagLoadError, match=r"role"):
        DagJsonlLoader(path).load()


def test_extra_stored_on_turn(tmp_path):
    """Non-native fields (sampling params, vendor knobs) live on Turn.extra_body
    and are merged into the top of the wire body at dispatch time by the
    endpoint."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": 100,
                        "extra": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "seed": 42,
                            "ignore_eos": True,
                            "min_tokens": 50,
                        },
                    }
                ],
            }
        ],
    )
    conv = DagJsonlLoader(path).load()[0]
    turn = conv.turns[0]
    assert turn.max_tokens == 100
    assert turn.extra_body == {
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "ignore_eos": True,
        "min_tokens": 50,
    }


def test_extra_body_rejected(tmp_path):
    """DagTurn uses ``extra='forbid'``, so a typo'd ``extra_body`` surfaces as
    a load-time Pydantic error rather than silently shadowing the supported
    ``extra`` field."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "extra_body": {"temperature": 0.7},
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError, match="extra_body"):
        DagJsonlLoader(path).load()


def test_flat_extras_populate_native_turn_fields(tmp_path):
    """AIPerf-native fields land on Turn's native attributes (raw_messages,
    max_tokens, model, raw_tools) so the endpoint's normal construction path
    handles them without any merge-mode detour."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": 123,
                        "model": "my-model",
                        "tools": [{"type": "function"}],
                    }
                ],
            }
        ],
    )
    conv = DagJsonlLoader(path).load()[0]
    turn = conv.turns[0]
    assert turn.raw_messages == [{"role": "user", "content": "u"}]
    assert turn.max_tokens == 123
    assert turn.model == "my-model"
    assert turn.raw_tools == [{"type": "function"}]
    # Structural DAG fields must not leak onto the Turn.
    assert turn.raw_payload is None


def test_non_native_field_at_top_level_rejected(tmp_path):
    """Sampling params like `temperature` are not aiperf-native Turn fields;
    they must go in extra, not at the top level of a turn."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "temperature": 0.7,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError, match=r"temperature"):
        DagJsonlLoader(path).load()


def test_unresolved_child_raises(tmp_path):
    path = write_lines(
        tmp_path,
        [{"session_id": "root", "turns": [_turn("u", forks=["nope"])]}],
    )
    with pytest.raises(DagLoadError, match=r"branch target 'nope' not declared"):
        DagJsonlLoader(path).load()


def test_cycle_is_hard_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPERF_DAG_VALIDATION_STRICT", "false")
    path = write_lines(
        tmp_path,
        [
            {"session_id": "a", "turns": [_turn("u", forks=["b"])]},
            {"session_id": "b", "turns": [_turn("u", forks=["a"])]},
        ],
    )
    with pytest.raises(DagLoadError, match=r"cycle detected"):
        DagJsonlLoader(path).load()


def test_multiple_parents_rejected(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {"session_id": "p1", "turns": [_turn("u", forks=["c"])]},
            {"session_id": "p2", "turns": [_turn("u", forks=["c"])]},
            {"session_id": "c", "turns": [_turn("u")]},
        ],
    )
    with pytest.raises(DagLoadError, match=r"forked by both"):
        DagJsonlLoader(path).load()


def test_spawn_on_non_terminal_turn_rejected(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [_turn("u", forks=["a"]), _turn("u2")],
            },
            {"session_id": "a", "turns": [_turn("u")]},
        ],
    )
    with pytest.raises(DagLoadError, match=r"branches but is not the last turn"):
        DagJsonlLoader(path).load()


def test_spawns_shorthand_produces_spawn_mode_branches(tmp_path):
    from aiperf.common.enums import ConversationBranchMode

    path = write_lines(
        tmp_path,
        [
            {"session_id": "root", "turns": [_turn("root", spawns=["agent-a"])]},
            {"session_id": "agent-a", "turns": [_turn("u")]},
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    root = by_id["root"]
    assert len(root.branches) == 1
    assert root.branches[0].mode == ConversationBranchMode.SPAWN
    assert root.branches[0].child_conversation_ids == ["agent-a"]
    assert root.branches[0].branch_id == "root:0"


def test_forks_and_spawns_on_same_turn_disambiguate_branch_ids(tmp_path):
    from aiperf.common.enums import ConversationBranchMode

    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [_turn("root", forks=["f1"], spawns=["s1"])],
            },
            {"session_id": "f1", "turns": [_turn("u")]},
            {"session_id": "s1", "turns": [_turn("u")]},
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    root = by_id["root"]
    branches_by_mode = {b.mode: b for b in root.branches}
    assert branches_by_mode[ConversationBranchMode.FORK].branch_id == "root:0:fork"
    assert branches_by_mode[ConversationBranchMode.SPAWN].branch_id == "root:0:spawn"
    assert set(root.turns[0].branch_ids) == {"root:0:fork", "root:0:spawn"}


def test_empty_child_conversation_ids_rejected(tmp_path):
    from aiperf.common.enums import ConversationBranchMode
    from aiperf.common.models.branch import ConversationBranchInfo

    path = write_lines(
        tmp_path,
        [
            {"session_id": "root", "turns": [_turn("u", forks=["a"])]},
            {"session_id": "a", "turns": [_turn("u")]},
        ],
    )
    loader = DagJsonlLoader(path)
    loader._parse_lines()
    loader._desugar_forks()
    root = loader._conversations["root"]
    root.branches = [
        ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=[],
            mode=ConversationBranchMode.FORK,
        )
    ]
    root.turns[0].branch_ids = ["root:0"]
    with pytest.raises(DagLoadError, match=r"declares no child_conversation_ids"):
        loader._resolve_and_validate()


def test_spawn_child_allows_multiple_parents(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {"session_id": "p1", "turns": [_turn("u", spawns=["shared"])]},
            {"session_id": "p2", "turns": [_turn("u", spawns=["shared"])]},
            {"session_id": "shared", "turns": [_turn("u")]},
        ],
    )
    conversations = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert "shared" in conversations


def test_fork_child_rejects_multiple_parents(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {"session_id": "p1", "turns": [_turn("u", forks=["c"])]},
            {"session_id": "p2", "turns": [_turn("u", forks=["c"])]},
            {"session_id": "c", "turns": [_turn("u")]},
        ],
    )
    with pytest.raises(
        DagLoadError, match=r"FORK-mode children require a single parent"
    ):
        DagJsonlLoader(path).load()


def test_system_on_fork_child_turn0_rejected(tmp_path):
    """FORK child's turn 0 inherits the parent accumulator, so a `system`
    entry there would land past index 0 on the wire and be silently dropped
    by Qwen3-VL-style chat templates."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "root-sys"},
                            {"role": "user", "content": "root-u"},
                        ],
                        "forks": ["c"],
                    }
                ],
            },
            {
                "session_id": "c",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "c-sys"},
                            {"role": "user", "content": "c-u"},
                        ]
                    }
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError, match=r"non-root turns may not contain"):
        DagJsonlLoader(path).load()


def test_system_on_non_root_turn_index_rejected(tmp_path):
    """Turn index > 0 of the root session is also a non-root turn for
    accumulator purposes."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "s"},
                            {"role": "user", "content": "u1"},
                        ]
                    },
                    {
                        "messages": [
                            {"role": "system", "content": "s2"},
                            {"role": "user", "content": "u2"},
                        ]
                    },
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError, match=r"non-root turns may not contain"):
        DagJsonlLoader(path).load()


def test_system_on_spawn_child_turn0_allowed(tmp_path):
    """SPAWN children start with an empty accumulator, so a `system` entry on
    their turn 0 lands at wire-position 0 and is valid."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "root-sys"},
                            {"role": "user", "content": "root-u"},
                        ],
                        "spawns": ["c"],
                    }
                ],
            },
            {
                "session_id": "c",
                "turns": [
                    {
                        "messages": [
                            {"role": "system", "content": "c-sys"},
                            {"role": "user", "content": "c-u"},
                        ]
                    }
                ],
            },
        ],
    )
    # Must not raise.
    DagJsonlLoader(path).load()


def test_orchestrator_line_synthesizes_no_request_turn(tmp_path):
    """The loader synthesizes ONE no-op turn for an orchestrator conversation
    from its conversation-level ``spawns``.

    Drives the full ``.load()`` path, which also runs the v1 orchestrator
    validator. The synthesized turn carries ``no_request=True`` and a
    post-timing SPAWN branch referencing the fan-out child, so a request-less
    orchestrator plus its spawned child loads end-to-end without raising.
    """
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "spawns": ["fan-out-a"],
            },
            {
                "session_id": "fan-out-a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    start = by_id["start"]
    assert start.is_orchestrator is True
    assert len(start.turns) == 1
    assert start.turns[0].no_request is True
    post_branches = [b for b in start.branches if b.dispatch_timing == "post"]
    assert any("fan-out-a" in b.child_conversation_ids for b in post_branches)
    # The synthesized turn declares the post-timing spawn branch.
    assert start.turns[0].branch_ids
    assert post_branches[0].branch_id in start.turns[0].branch_ids


def test_orchestrator_spawn_child_must_exist(tmp_path):
    """An orchestrator whose spawn child is absent from the dataset raises at
    load time (child-existence is still enforced through the normal path)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "spawns": ["missing-child"],
            },
        ],
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(path).load()


def test_orchestrator_rounds_synthesizes_gated_spine(tmp_path):
    """``rounds: N`` synthesizes a gated spine: N spawning request-free turns
    (each AND-waiting its round before the next) plus a terminal gate turn, so
    every round's branches complete before the next round fires."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": 2,
                "spawns": ["a", "b"],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "b",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    start = by_id["start"]
    assert start.is_orchestrator is True
    # N spawning turns + 1 terminal gate = N+1
    assert len(start.turns) == 3
    assert all(t.no_request is True for t in start.turns)
    # The two spawning turns each declare a SPAWN branch; the terminal gate does not.
    assert start.turns[0].branch_ids
    assert start.turns[1].branch_ids
    assert not start.turns[2].branch_ids


def test_orchestrator_rounds_list_authors_distinct_branches_per_round(tmp_path):
    """A per-round ``rounds`` list synthesizes a gated spine where each round
    fans out its OWN branch sessions (not one repeated template) and applies its
    OWN think-time. The terminal gate carries no branches and no think-time."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": [
                    {"spawns": ["t0-a", "t0-b"], "think_time_ms": 12000.0},
                    {"spawns": ["t1-a", "t1-b"], "think_time_ms": 31000.0},
                ],
            },
            {
                "session_id": "t0-a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "t0-b",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "t1-a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "t1-b",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    start = by_id["start"]
    assert start.is_orchestrator is True
    # 2 spawning turns + terminal gate.
    assert len(start.turns) == 3
    assert all(t.no_request is True for t in start.turns)
    # Per-round think-time rides the spawning turns; the terminal gate has none.
    assert [t.delay for t in start.turns[:-1]] == [12000.0, 31000.0]
    assert not start.turns[-1].delay
    # Each round fans out its OWN distinct branch sessions.
    children_by_branch = {b.branch_id: b.child_conversation_ids for b in start.branches}
    assert children_by_branch[start.turns[0].branch_ids[0]] == ["t0-a", "t0-b"]
    assert children_by_branch[start.turns[1].branch_ids[0]] == ["t1-a", "t1-b"]
    assert not start.turns[2].branch_ids


def test_orchestrator_rounds_list_missing_branch_rejected(tmp_path):
    """A per-round branch id that names no conversation in the file is rejected
    at load, like any other unknown spawn target."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": [{"spawns": ["t0-a"]}, {"spawns": ["t1-a"]}],
            },
            {
                "session_id": "t0-a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="t1-a"):
        DagJsonlLoader(path).load()


def test_branch_context_mode_opt_in_payload_isolation(tmp_path):
    """A branch may author ``context_mode: message_array_with_responses`` for
    payload isolation -- each turn is sent as its own authored message array with
    no accumulation of prior turns or live responses. Conversations that omit it
    keep the DAG default (DELTAS_WITHOUT_RESPONSES)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": [{"spawns": ["b0"]}],
            },
            {
                "session_id": "b0",
                "context_mode": "message_array_with_responses",
                "turns": [_turn("u")],
            },
            {"session_id": "plain", "turns": [_turn("x")]},
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert (
        by_id["b0"].context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
    assert (
        by_id["plain"].context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
    )


def test_isolation_mode_allows_per_turn_system_prompt(tmp_path):
    """Under message_array_with_responses each turn is sent as its own complete
    array, so a distinct system prompt on a LATER turn sits at index 0 of its own
    payload and is honored -- the non-root placement rule must not reject it
    (required for per-node authored system prompts)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": [{"spawns": ["a"]}],
            },
            {
                "session_id": "a",
                "context_mode": "message_array_with_responses",
                "turns": [
                    {"messages": [{"role": "system", "content": "sys0"}, _msg("u0")]},
                    {"messages": [{"role": "system", "content": "sys1"}, _msg("u1")]},
                ],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    t1 = by_id["a"].turns[1].raw_messages
    assert [m["role"] for m in t1] == ["system", "user"]
    assert next(m["content"] for m in t1 if m["role"] == "system") == "sys1"


def test_isolation_mode_rejects_system_not_first_in_turn(tmp_path) -> None:
    """Even under isolation, a system message must be FIRST in its turn's array:
    a later position is still mid-payload and dropped by chat templates, so it
    is rejected rather than silently ignored."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": [{"spawns": ["a"]}],
            },
            {
                "session_id": "a",
                "context_mode": "message_array_with_responses",
                "turns": [
                    {"messages": [_msg("u0"), {"role": "system", "content": "sys"}]},
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="must be the first message"):
        DagJsonlLoader(path).load()


def test_accumulating_mode_still_rejects_non_root_system_prompt(tmp_path):
    """The placement rule still applies to accumulating (deltas) conversations,
    where a later-turn system message lands mid-array after the append-merge and
    is silently dropped by chat templates."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {"messages": [{"role": "system", "content": "s0"}, _msg("u0")]},
                    {"messages": [{"role": "system", "content": "s1"}, _msg("u1")]},
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError, match="non-root turns may not contain a 'system'"):
        DagJsonlLoader(path).load()


def test_rounds_list_with_sigma_builds_sampled_think_time(tmp_path):
    """List-form rounds with conversation-level ``think_time_sigma`` produce a
    sampled think-time spec; each round's ``think_time_ms`` is its per-round
    median, stamped as that spawning turn's delay."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "think_time_ms": 100.0,
                "think_time_sigma": 0.5,
                "rounds": [
                    {"spawns": ["a"], "think_time_ms": 12000.0},
                    {"spawns": ["b"], "think_time_ms": 31000.0},
                ],
            },
            {"session_id": "a", "turns": [_turn("x")]},
            {"session_id": "b", "turns": [_turn("x")]},
        ],
    )
    start = {c.session_id: c for c in DagJsonlLoader(path).load()}["start"]
    assert start.think_time is not None
    assert start.think_time.sigma == 0.5
    assert [t.delay for t in start.turns[:-1]] == [12000.0, 31000.0]


def test_orchestrator_think_time_respects_delay_cap(tmp_path):
    """The configured inter-turn delay cap bounds orchestrator think-time: each
    stamped fixed/median delay is clamped, and the sampled distribution's bounds
    are folded to the cap (min stays <= max even when authored above it)."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "think_time_ms": 250.0,
                "think_time_sigma": 0.6,
                "think_time_min_ms": 10.0,
                "think_time_max_ms": 9999.0,
                "rounds": [
                    {"spawns": ["a"], "think_time_ms": 250.0},
                    {"spawns": ["b"], "think_time_ms": 800.0},
                ],
            },
            {"session_id": "a", "turns": [_turn("x")]},
            {"session_id": "b", "turns": [_turn("x")]},
        ],
    )
    loader = DagJsonlLoader(path)
    loader._delay_cap_tracker.cap_seconds = 0.1  # 100 ms cap
    start = {c.session_id: c for c in loader.load()}["start"]
    # Both stamped round delays clamped to the cap; terminal gate carries none.
    assert [t.delay for t in start.turns] == [100.0, 100.0, None]
    # Sampled distribution bounds folded to the cap.
    assert start.think_time.max_ms == 100.0
    assert start.think_time.min_ms == 10.0


def test_rounds_rejected_without_orchestrator(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "rounds": 2,
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(path).load()


def test_orchestrator_think_time_ms_rides_spawning_turns_only(tmp_path):
    """``think_time_ms`` is stamped as the per-round delay on the N spawning
    spine turns (0..N-1), so the orchestrator applies it before each round. The
    terminal gate turn (turn N) spawns no round and carries NO think-time, so it
    does not delay session completion by a spurious extra interval."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": 2,
                "think_time_ms": 50.0,
                "spawns": ["a"],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    start = by_id["start"]
    assert len(start.turns) == 3
    # spawning turns (0..N-1) carry the per-round think; terminal gate does not.
    assert [t.delay for t in start.turns[:-1]] == [50.0, 50.0]
    assert not start.turns[-1].delay


def test_think_time_ms_requires_rounds(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "think_time_ms": 50.0,
                "spawns": ["a"],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(path).load()


def test_orchestrator_think_time_sigma_sets_distribution(tmp_path):
    """``think_time_sigma`` attaches a lognormal ThinkTimeSpec (median = the
    per-round think_time_ms) to the synthesized orchestrator conversation."""
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": 2,
                "think_time_ms": 200.0,
                "think_time_sigma": 0.5,
                "think_time_min_ms": 10.0,
                "spawns": ["a"],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    start = by_id["start"]
    assert start.think_time is not None
    assert start.think_time.sigma == 0.5
    assert start.think_time.min_ms == 10.0
    # median rides the per-round delay on the spawning turns; terminal gate is 0.
    assert [t.delay for t in start.turns[:-1]] == [200.0, 200.0]
    assert not start.turns[-1].delay


def test_think_time_sigma_requires_think_time_ms(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "start",
                "orchestrator": True,
                "rounds": 2,
                "think_time_sigma": 0.5,
                "spawns": ["a"],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(path).load()


def _orch(**extra):
    base = {
        "session_id": "start",
        "orchestrator": True,
        "rounds": 2,
        "think_time_ms": 100.0,
        "spawns": ["a"],
    }
    base.update(extra)
    return base


def _child():
    return {
        "session_id": "a",
        "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
    }


def test_think_time_min_gt_max_rejected(tmp_path):
    path = write_lines(
        tmp_path,
        [
            _orch(
                think_time_sigma=0.5, think_time_min_ms=500.0, think_time_max_ms=10.0
            ),
            _child(),
        ],
    )
    with pytest.raises(DagLoadError, match=r"min_ms.*<=.*max_ms"):
        DagJsonlLoader(path).load()


def test_think_time_clamp_without_sigma_rejected(tmp_path):
    path = write_lines(tmp_path, [_orch(think_time_min_ms=50.0), _child()])
    with pytest.raises(DagLoadError, match=r"require 'think_time_sigma'"):
        DagJsonlLoader(path).load()


def test_think_time_fields_on_non_orchestrator_rejected(tmp_path):
    path = write_lines(
        tmp_path,
        [
            {
                "session_id": "s",
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                "think_time_sigma": 0.5,
            },
        ],
    )
    with pytest.raises(DagLoadError, match=r"only valid on an orchestrator"):
        DagJsonlLoader(path).load()


def test_think_time_sigma_infinite_rejected(tmp_path):
    # inf is JSON-illegal via orjson, but a huge finite sigma must be capped (le=10).
    path = write_lines(tmp_path, [_orch(think_time_sigma=1e300), _child()])
    with pytest.raises(DagLoadError):
        DagJsonlLoader(path).load()


def test_fork_target_with_payload_isolation_rejected(tmp_path):
    """A FORK child inherits parent context; payload isolation would replace it
    with the isolated authored array, silently dropping the parent prompt. Reject
    message_array_with_responses on FORK targets (SPAWN targets are fine)."""
    path = write_lines(
        tmp_path,
        [
            {"session_id": "root", "turns": [_turn("p", forks=["child"])]},
            {
                "session_id": "child",
                "context_mode": "message_array_with_responses",
                "turns": [_turn("c")],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="FORK target with context_mode"):
        DagJsonlLoader(path).load()


def test_spawn_target_with_payload_isolation_allowed(tmp_path):
    """SPAWN starts fresh, so payload isolation on a SPAWN target is valid."""
    path = write_lines(
        tmp_path,
        [
            {"session_id": "root", "turns": [_turn("p", spawns=["child"])]},
            {
                "session_id": "child",
                "context_mode": "message_array_with_responses",
                "turns": [_turn("c")],
            },
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert (
        by_id["child"].context_mode
        == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
