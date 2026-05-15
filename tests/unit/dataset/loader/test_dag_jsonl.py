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
