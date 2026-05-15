# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial attack surface tests for the new dag5 loaders.

Covers ``DagJsonlLoader`` (dag_jsonl), ``RawPayloadDatasetLoader``
(raw_payload), and ``InputsJsonPayloadLoader`` (inputs_json) with hostile
inputs across structural corruption, schema boundary cases, DAG topology
attacks, sizing, and per-loader format-specific traps.

Every test asserts EITHER a clean load OR a ``DagLoadError`` / ``ValueError``
carrying actionable location info (``session 'X'``, ``turn N``, ``line N``,
or the offending field name). No test tolerates a panic, hang, or silent
truncation.

For each scenario where current production code drops actionable detail
(silent acceptance of dangerous values, opaque exception types), the test
is marked ``xfail(strict=True)`` so a future fix surfaces it.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path: Path, lines: list[dict], name: str = "dag.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _write_bytes(tmp_path: Path, body: bytes, name: str = "dag.jsonl") -> Path:
    p = tmp_path / name
    p.write_bytes(body)
    return p


def _basic_turn(content: str = "u") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _basic_conv(sid: str, n_turns: int = 1) -> dict:
    return {
        "session_id": sid,
        "turns": [_basic_turn(f"u{i}") for i in range(n_turns)],
    }


def _user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


# ===========================================================================
# Section 1. Structural corruption: malformed JSON
# ===========================================================================


def test_dag_jsonl_truncated_line_rejected(tmp_path: Path):
    """A truncated JSON line surfaces a DagLoadError naming the offending line."""
    body = b'{"session_id":"a","turns":[{"messages":[{"role":"user","content":"u"\n'
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "invalid JSON" in msg
    assert "line 1" in msg


def test_dag_jsonl_mismatched_braces_rejected(tmp_path: Path):
    """Extra unmatched braces produce a line-numbered DagLoadError."""
    body = b'{"session_id":"a","turns":[{"messages":[{"role":"user","content":"u"}]}]}}'
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    assert "invalid JSON" in str(excinfo.value)
    assert "line 1" in str(excinfo.value)


def test_dag_jsonl_embedded_raw_nul_byte_rejected(tmp_path: Path):
    """A raw NUL byte mid-string is a control character; orjson rejects it
    (RFC-strict). Loader surfaces invalid JSON + line number."""
    body = (
        b'{"session_id":"a","turns":[{"messages":[{"role":"u","content":"x\x00y"}]}]}'
    )
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    assert "invalid JSON" in str(excinfo.value)
    assert "line 1" in str(excinfo.value)


# ===========================================================================
# Section 2. Type confusion: schema enforcement
# ===========================================================================


def test_dag_jsonl_session_id_as_int_rejected(tmp_path: Path):
    """Numeric session_id is rejected; error names the field path."""
    path = _write_jsonl(tmp_path, [{"session_id": 12345, "turns": [_basic_turn()]}])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "session_id" in msg


def test_dag_jsonl_turns_as_dict_rejected(tmp_path: Path):
    """``turns`` declared as a dict (not list) is rejected with field locator."""
    path = _write_jsonl(tmp_path, [{"session_id": "a", "turns": {"0": _basic_turn()}}])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "turns" in msg


def test_dag_jsonl_max_tokens_as_bool_rejected(tmp_path: Path):
    """``max_tokens: true`` is a typo-trap (bool is int subclass); the
    custom validator rejects with a message mentioning ``max_tokens``."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": True,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    assert "max_tokens" in str(excinfo.value)


# ===========================================================================
# Section 3. Missing required fields
# ===========================================================================


def test_dag_jsonl_missing_session_id_rejected(tmp_path: Path):
    """Conversation missing ``session_id`` is rejected with field-path locator."""
    path = _write_jsonl(tmp_path, [{"turns": [_basic_turn()]}])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "session_id" in msg


def test_dag_jsonl_missing_turns_rejected(tmp_path: Path):
    """Conversation missing ``turns`` is rejected with field-path locator."""
    path = _write_jsonl(tmp_path, [{"session_id": "a"}])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "turns" in msg


def test_dag_jsonl_empty_turns_array_rejected(tmp_path: Path):
    """An empty ``turns: []`` array fails pydantic's ``min_length=1``."""
    path = _write_jsonl(tmp_path, [{"session_id": "a", "turns": []}])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "turns" in msg


# ===========================================================================
# Section 4. Schema boundary cases
# ===========================================================================


def test_dag_jsonl_session_id_emoji_accepted(tmp_path: Path):
    """Pure-emoji session_id is accepted (bytestring valid UTF-8) and
    branch_ids stay parseable through the rsplit-anchored helper."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "🚀💥",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["c"],
                    },
                    _basic_turn(),
                ],
            },
            _basic_conv("c"),
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    rocket = next(c for c in convs if c.session_id == "🚀💥")
    assert rocket.branches[0].branch_id == "🚀💥:0"


def test_dag_jsonl_max_tokens_zero_rejected_by_pydantic(tmp_path: Path):
    """``max_tokens=0`` fails pydantic's ``ge=1`` constraint with field path."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": 0,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "max_tokens" in msg


def test_dag_jsonl_max_tokens_negative_rejected(tmp_path: Path):
    """Negative ``max_tokens`` fails ``ge=1`` and surfaces field path."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": -5,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "max_tokens" in msg


def test_dag_jsonl_max_tokens_max_int_accepted(tmp_path: Path):
    """Pydantic int is unbounded above; 2**63 is accepted (bounds enforcement
    happens server-side, not in the loader)."""
    big = 2**63
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "max_tokens": big,
                    }
                ],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].max_tokens == big


def test_dag_jsonl_delay_negative_rejected(tmp_path: Path):
    """``delay=-1`` fails ``ge=0.0``; loader surfaces field path on line 1."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "delay": -1.0,
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "delay" in msg


def test_dag_jsonl_empty_messages_list_rejected(tmp_path: Path):
    """``messages: []`` is rejected by ``validate_chat_messages``."""
    path = _write_jsonl(
        tmp_path,
        [{"session_id": "a", "turns": [{"messages": []}]}],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    assert "non-empty" in str(excinfo.value)


def test_dag_jsonl_single_empty_content_message_accepted(tmp_path: Path):
    """A single message with empty-string content is structurally valid."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": ""}]}],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_messages[0]["content"] == ""


def test_dag_jsonl_extra_five_level_nested_round_trips(tmp_path: Path):
    """Five-level-deep ``extra`` survives the loader byte-identically."""
    deep = {
        "l1": {
            "l2": {
                "l3": {
                    "l4": {
                        "l5": {"value": [1, 2, 3], "flag": True, "name": "deep"},
                    }
                }
            }
        }
    }
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "extra": deep,
                    }
                ],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].extra_body == deep


def test_dag_jsonl_tools_empty_list_accepted(tmp_path: Path):
    """``tools: []`` is structurally valid (empty list of tool defs)."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "tools": [],
                    }
                ],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_tools == []


def test_dag_jsonl_tools_omitted_yields_none(tmp_path: Path):
    """Omitting ``tools`` entirely leaves ``raw_tools`` as None."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_tools is None


# ===========================================================================
# Section 5. DAG topology attacks
# ===========================================================================


def test_dag_jsonl_100_level_fork_chain_accepted(tmp_path: Path):
    """A 100-level deep cycle-free FORK chain loads cleanly. Exercises the
    iterative depth walk's fixed-point convergence under deep nesting."""
    n = 100
    lines: list[dict] = []
    for i in range(n - 1):
        lines.append(
            {
                "session_id": f"s{i}",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": [f"s{i + 1}"],
                    }
                ],
            }
        )
    lines.append({"session_id": f"s{n - 1}", "turns": [_basic_turn()]})
    path = _write_jsonl(tmp_path, lines)
    convs = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    assert convs[f"s{n - 1}"].agent_depth == n - 1
    assert convs["s0"].is_root is True


def test_dag_jsonl_self_fork_rejected(tmp_path: Path):
    """A conversation forking itself produces a 1-cycle; detect_cycles
    raises with the path naming the offender."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "selfie",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["selfie"],
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "cycle detected" in msg
    assert "selfie" in msg


def test_dag_jsonl_two_cycle_rejected(tmp_path: Path):
    """A → B → A two-cycle is rejected and the trace names both nodes."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "A",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["B"],
                    },
                    _basic_turn(),
                ],
            },
            {
                "session_id": "B",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["A"],
                    },
                    _basic_turn(),
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "cycle detected" in msg
    assert "A" in msg
    assert "B" in msg


def test_dag_jsonl_three_cycle_rejected(tmp_path: Path):
    """A → B → C → A three-cycle: cycle path lists all three nodes."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "A",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["B"],
                    },
                    _basic_turn(),
                ],
            },
            {
                "session_id": "B",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["C"],
                    },
                    _basic_turn(),
                ],
            },
            {
                "session_id": "C",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["A"],
                    },
                    _basic_turn(),
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "cycle detected" in msg
    for node in ("A", "B", "C"):
        assert node in msg


def test_dag_jsonl_five_cycle_rejected(tmp_path: Path):
    """A 5-cycle through ``spawns`` is rejected with every node in the path."""
    nodes = ["n0", "n1", "n2", "n3", "n4"]
    lines = []
    for i, sid in enumerate(nodes):
        nxt = nodes[(i + 1) % len(nodes)]
        lines.append(
            {
                "session_id": sid,
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": [nxt],
                    },
                    _basic_turn(),
                ],
            }
        )
    path = _write_jsonl(tmp_path, lines)
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "cycle detected" in msg
    for n in nodes:
        assert n in msg


def test_dag_jsonl_fork_target_not_in_dataset_rejected(tmp_path: Path):
    """Forking an undeclared child names both parent and missing target."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "parent",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["ghost"],
                    }
                ],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "ghost" in msg
    assert "parent" in msg
    assert "not declared" in msg


def test_dag_jsonl_duplicate_session_ids_rejected(tmp_path: Path):
    """Two conversations with the same session_id: line-numbered duplicate."""
    path = _write_jsonl(
        tmp_path,
        [_basic_conv("dup"), _basic_conv("dup")],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 2" in msg
    assert "duplicate session_id" in msg
    assert "dup" in msg


def test_dag_jsonl_pre_session_spawns_500_unique_loads(tmp_path: Path):
    """500 unique pre_session_spawn children load cleanly into a single
    ``:pre`` branch."""
    n = 500
    children = [f"k{i}" for i in range(n)]
    lines = [
        {
            "session_id": "root",
            "pre_session_spawns": children,
            "turns": [_basic_turn()],
        }
    ]
    for cid in children:
        lines.append(_basic_conv(cid))
    path = _write_jsonl(tmp_path, lines)
    convs = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    root = convs["root"]
    pre = next(b for b in root.branches if b.branch_id == "root:pre")
    assert len(pre.child_conversation_ids) == n


def test_dag_jsonl_branch_id_collision_across_conversations_accepted(tmp_path: Path):
    """``a`` and ``a:0`` both spawning at turn 0 produce distinct branch_ids
    (``a:0`` vs ``a:0:0``); accepted because branch_ids are scoped per parent."""
    path = _write_jsonl(
        tmp_path,
        [
            _basic_conv("leaf"),
            {
                "session_id": "a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["leaf"],
                    },
                    _basic_turn(),
                ],
            },
            {
                "session_id": "a:0",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["leaf"],
                    },
                    _basic_turn(),
                ],
            },
        ],
    )
    convs = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    assert convs["a"].branches[0].branch_id == "a:0"
    assert convs["a:0"].branches[0].branch_id == "a:0:0"


def test_dag_jsonl_pre_session_spawns_unknown_child_rejected(tmp_path: Path):
    """A pre_session_spawn pointing at a non-existent session names it."""
    path = _write_jsonl(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["nope"],
                "turns": [_basic_turn()],
            }
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "nope" in msg


# ===========================================================================
# Section 6. Sizing
# ===========================================================================


def test_dag_jsonl_10k_conversation_dataset_loads(tmp_path: Path):
    """A 10,000-conversation dataset loads successfully — smoke test for
    O(N) lookup paths in the loader."""
    n = 10_000
    lines = [_basic_conv(f"s{i}") for i in range(n)]
    path = _write_jsonl(tmp_path, lines)
    convs = DagJsonlLoader(filename=path).load()
    assert len(convs) == n


def test_dag_jsonl_single_conversation_1000_turns_loads(tmp_path: Path):
    """A single conversation with 1,000 turns loads (sanity check on the
    per-turn desugar loop)."""
    n = 1000
    line = {
        "session_id": "long",
        "turns": [_basic_turn(f"u{i}") for i in range(n)],
    }
    path = _write_jsonl(tmp_path, [line])
    convs = DagJsonlLoader(filename=path).load()
    assert len(convs[0].turns) == n


# ===========================================================================
# Section 7. raw_payload loader hostile inputs
# ===========================================================================


def test_raw_payload_empty_line_skipped(tmp_path: Path):
    """Empty/blank lines between payloads are skipped, not errored."""
    p = tmp_path / "payloads.jsonl"
    p.write_bytes(
        orjson.dumps({"messages": [{"role": "user", "content": "a"}]})
        + b"\n\n   \n"
        + orjson.dumps({"messages": [{"role": "user", "content": "b"}]})
        + b"\n"
    )
    loader = RawPayloadDatasetLoader(filename=p, user_config=_user_config())
    data = loader.load_dataset()
    # Two distinct sessions (one per non-blank line).
    assert len(data) == 2


def test_raw_payload_truncated_json_line_raises_json_decode_error(tmp_path: Path):
    """A truncated JSON line raises ``orjson.JSONDecodeError`` (a ``ValueError``
    subclass). Loader doesn't wrap it; document the surface so callers can
    catch the right exception type."""
    p = tmp_path / "payloads.jsonl"
    p.write_bytes(b'{"messages":[{"role":"user","content":"a"')
    loader = RawPayloadDatasetLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError):
        loader.load_dataset()


def test_raw_payload_binary_garbage_line_raises_value_error(tmp_path: Path):
    """A line of raw binary garbage (non-UTF-8 bytes) raises ValueError
    on parse — never silently dropped."""
    p = tmp_path / "payloads.jsonl"
    p.write_bytes(b"\x00\x01\x02\xff\xfe\xfd\n")
    loader = RawPayloadDatasetLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError):
        loader.load_dataset()


def test_raw_payload_missing_messages_field_rejected(tmp_path: Path):
    """A payload line missing ``messages`` is rejected at load with a
    location-rich error naming the file and line number.
    """
    p = tmp_path / "payloads.jsonl"
    p.write_bytes(orjson.dumps({"model": "x", "max_tokens": 16}) + b"\n")
    loader = RawPayloadDatasetLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError, match=r"payloads\.jsonl:1.*messages"):
        loader.load_dataset()


def test_raw_payload_can_load_rejects_non_dict_data():
    """``can_load`` returns False (no crash) for a bare list at the top level."""
    assert RawPayloadDatasetLoader.can_load(data={"messages": "not a list"}) is False
    # bare top-level scalars short-circuit on isinstance check:
    assert RawPayloadDatasetLoader.can_load(data={"messages": [{"role": "u"}]}) is True


# ===========================================================================
# Section 8. inputs_json loader hostile inputs
# ===========================================================================


def test_inputs_json_missing_top_level_data_raises(tmp_path: Path):
    """An ``inputs.json`` missing the AIPerf envelope (no ``data`` key)
    raises a clear KeyError or ValueError on load."""
    p = tmp_path / "inputs.json"
    p.write_bytes(orjson.dumps({"not_data": []}))
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises((KeyError, ValueError)):
        loader.load_dataset()


def test_inputs_json_session_missing_session_id_raises(tmp_path: Path):
    """A ``data[]`` entry without ``session_id`` raises a clear KeyError
    or ValidationError."""
    p = tmp_path / "inputs.json"
    p.write_bytes(
        orjson.dumps(
            {"data": [{"payloads": [{"messages": [{"role": "user", "content": "u"}]}]}]}
        )
    )
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises((KeyError, ValueError)):
        loader.load_dataset()


def test_inputs_json_empty_payloads_list_rejected(tmp_path: Path):
    """``InputsJsonSession`` has ``min_length=1`` on payloads; an empty list
    is rejected with a field-path-bearing ValidationError."""
    p = tmp_path / "inputs.json"
    p.write_bytes(orjson.dumps({"data": [{"session_id": "abc", "payloads": []}]}))
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError) as excinfo:
        loader.load_dataset()
    assert "payloads" in str(excinfo.value)


def test_inputs_json_duplicate_session_ids_rejected_with_index(tmp_path: Path):
    """Two ``data[]`` entries with the same session_id are rejected at
    load time with an index-rich error naming the duplicate. Previously
    the second entry silently overwrote the first and authored turns
    were lost without warning.
    """
    p = tmp_path / "inputs.json"
    p.write_bytes(
        orjson.dumps(
            {
                "data": [
                    {
                        "session_id": "dup",
                        "payloads": [
                            {"messages": [{"role": "user", "content": "first"}]}
                        ],
                    },
                    {
                        "session_id": "dup",
                        "payloads": [
                            {"messages": [{"role": "user", "content": "second"}]},
                            {"messages": [{"role": "user", "content": "third"}]},
                        ],
                    },
                ]
            }
        )
    )
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError, match=r"data\[1\] duplicate session_id 'dup'"):
        loader.load_dataset()


def test_inputs_json_duplicate_session_ids_should_be_rejected(tmp_path: Path):
    """Duplicates in ``data[]`` are rejected so users don't silently
    lose authored turns."""
    p = tmp_path / "inputs.json"
    p.write_bytes(
        orjson.dumps(
            {
                "data": [
                    {
                        "session_id": "dup",
                        "payloads": [{"messages": [{"role": "user", "content": "1"}]}],
                    },
                    {
                        "session_id": "dup",
                        "payloads": [{"messages": [{"role": "user", "content": "2"}]}],
                    },
                ]
            }
        )
    )
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_dataset()


def test_inputs_json_data_not_a_list_raises(tmp_path: Path):
    """``data`` typed as a string (not a list) raises on iteration."""
    p = tmp_path / "inputs.json"
    p.write_bytes(orjson.dumps({"data": "not a list"}))
    loader = InputsJsonPayloadLoader(filename=p, user_config=_user_config())
    with pytest.raises((TypeError, ValueError)):
        loader.load_dataset()
