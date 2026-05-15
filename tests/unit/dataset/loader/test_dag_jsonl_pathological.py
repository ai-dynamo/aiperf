# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pathological JSON / JSONL inputs for ``DagJsonlLoader``.

This file targets a different surface than ``test_dag_jsonl_adversarial_full.py``
and the per-feature suites: parser-level encoding edge cases (CRLF / mixed line
endings / BOM-suffixed lines / control chars), numeric corner cases inherited
from orjson + pydantic (NaN / Infinity / scientific notation / int64 overflow /
float-coerce on int fields), JSON shape oddities (top-level non-object,
duplicate JSON keys, extra fields, deep nesting, large payloads), branch_id /
session_id collision attacks, and serialization round-trips between orjson and
the stdlib ``json`` module.

Where a bug-class is genuinely undefined (e.g. ``delay=Infinity`` accepted
programmatically), the test pins down current behavior and is marked
``xfail(strict=True)`` so a future fix surfaces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import orjson
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
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.dag_jsonl_models import DagConversation, DagSpawn, DagTurn
from aiperf.plugin.enums import DatasetSamplingStrategy


def _basic_turn(content: str = "u") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _basic_conv(sid: str = "a", n: int = 1) -> dict:
    return {"session_id": sid, "turns": [_basic_turn(f"u{i}") for i in range(n)]}


def _write_bytes(tmp_path: Path, body: bytes) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_bytes(body)
    return p


def _write_lines(tmp_path: Path, lines: list[dict], sep: bytes = b"\n") -> Path:
    body = sep.join(json.dumps(line).encode() for line in lines)
    return _write_bytes(tmp_path, body)


# ---------------------------------------------------------------------------
# 1. Line endings: CRLF, mixed, leading whitespace
# ---------------------------------------------------------------------------


def test_jsonl_crlf_line_endings_accepted(tmp_path: Path):
    """Pure-CRLF JSONL parses cleanly: ``raw.strip()`` strips ``\\r``."""
    path = _write_lines(tmp_path, [_basic_conv("a"), _basic_conv("b")], sep=b"\r\n")
    convs = DagJsonlLoader(filename=path).load()
    assert sorted(c.session_id for c in convs) == ["a", "b"]


def test_jsonl_mixed_lf_and_crlf_accepted(tmp_path: Path):
    """LF + CRLF mixed in the same file parse cleanly."""
    body = (
        json.dumps(_basic_conv("a")).encode()
        + b"\r\n"
        + json.dumps(_basic_conv("b")).encode()
        + b"\n"
        + json.dumps(_basic_conv("c")).encode()
        + b"\r\n\r\n"
    )
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert sorted(c.session_id for c in convs) == ["a", "b", "c"]


def test_jsonl_leading_whitespace_on_line_accepted(tmp_path: Path):
    """Leading tabs/spaces on a JSONL line are stripped before orjson parse."""
    body = b"\t\t " + json.dumps(_basic_conv("a")).encode() + b"\n"
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].session_id == "a"


def test_jsonl_whitespace_only_line_skipped(tmp_path: Path):
    """A line containing only whitespace is treated as blank and skipped."""
    body = (
        json.dumps(_basic_conv("a")).encode()
        + b"\n   \t  \n"
        + json.dumps(_basic_conv("b")).encode()
    )
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert sorted(c.session_id for c in convs) == ["a", "b"]


# ---------------------------------------------------------------------------
# 2. Top-level JSON shape: non-object, extra fields, deeply nested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"[1, 2, 3]",
        b'"a string"',
        b"42",
        b"null",
        b"true",
        b"3.14",
    ],
)
def test_jsonl_line_valid_json_but_not_object_rejected(tmp_path: Path, body: bytes):
    """Each non-object top-level JSON value triggers DagLoadError with line N."""
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    assert "line 1" in str(excinfo.value)


def test_jsonl_unknown_top_level_conversation_field_rejected_with_line_no(
    tmp_path: Path,
):
    """Unknown conversation-level keys land in ``extra="forbid"`` and the
    DagLoadError surfaces the offending line number."""
    path = _write_bytes(
        tmp_path,
        json.dumps(
            {
                "session_id": "a",
                "turns": [_basic_turn()],
                "definitely_not_a_field": 1,
            }
        ).encode(),
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 1" in msg
    assert "definitely_not_a_field" in msg or "Extra" in msg or "forbidden" in msg


def test_jsonl_extreme_nesting_in_extra_accepted(tmp_path: Path):
    """``extra`` holds an arbitrary JSON-shaped dict; orjson and pydantic
    handle deeply-nested dicts without recursion-error stack blow up."""
    deep: dict = {"v": 1}
    for _ in range(500):
        deep = {"nested": deep}
    line = {
        "session_id": "a",
        "turns": [{"messages": [{"role": "user", "content": "u"}], "extra": deep}],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    eb = convs[0].turns[0].extra_body
    assert eb is not None
    cur = eb
    for _ in range(500):
        cur = cur["nested"]
    assert cur == {"v": 1}


def test_jsonl_large_extra_string_accepted(tmp_path: Path):
    """A multi-megabyte string inside ``extra`` survives the loader."""
    blob = "x" * (2 * 1024 * 1024)  # 2 MiB
    line = {
        "session_id": "a",
        "turns": [
            {
                "messages": [{"role": "user", "content": "u"}],
                "extra": {"big": blob},
            }
        ],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    assert len(convs[0].turns[0].extra_body["big"]) == len(blob)


# ---------------------------------------------------------------------------
# 3. Numeric corner cases: NaN, Infinity, scientific, overflow, float-as-int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_jsonl_nan_and_infinity_literals_rejected_by_orjson(
    tmp_path: Path, token: bytes
):
    """orjson is strict-RFC: ``NaN`` / ``Infinity`` literals never decode.
    The loader surfaces this as DagLoadError(invalid JSON) on the offending
    line, never as a half-parsed conversation."""
    body = (
        b'{"session_id":"a","turns":[{"messages":[{"role":"u","content":'
        + token
        + b"}]}]}"
    )
    # The above has the NaN/Inf literal as the *content value* -> orjson rejects
    # at parse time, before pydantic ever sees the dict.
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError, match="invalid JSON"):
        DagJsonlLoader(filename=path).load()


def test_jsonl_overflow_float_rejected_by_orjson(tmp_path: Path):
    """``1e400`` overflows IEEE-754 double; orjson raises rather than emit
    ``Infinity``. Loader surfaces this as DagLoadError(invalid JSON)."""
    body = b'{"session_id":"a","turns":[{"messages":[{"role":"user","content":"u"}],"delay":1e400}]}'
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError, match="invalid JSON"):
        DagJsonlLoader(filename=path).load()


def test_dag_turn_delay_positive_infinity_accepted_programmatically():
    """Programmatic construction (bypassing JSON) accepts ``delay=+inf``
    because pydantic's ``ge=0.0`` constraint admits ``+inf``. Documents a
    quirk: callers building DagTurn directly should sanity-check finiteness.
    Path is unreachable via JSONL because orjson rejects ``Infinity`` at
    parse time."""
    t = DagTurn(messages=[{"role": "user", "content": "u"}], delay=math.inf)
    assert math.isinf(t.delay)


def test_dag_turn_delay_nan_rejected():
    """``delay=NaN`` fails pydantic's ``ge=0.0`` (NaN comparisons are False)."""
    with pytest.raises(ValidationError):
        DagTurn(messages=[{"role": "user", "content": "u"}], delay=math.nan)


def test_dag_spawn_join_at_float_with_zero_fraction_coerced_to_int():
    """Pydantic's default coercion accepts ``5.0`` for an ``int`` field
    (becomes ``5``), but rejects ``5.5``. Documents both branches."""
    s = DagSpawn(children=["c"], join_at=5.0)
    assert s.join_at == 5
    assert isinstance(s.join_at, int)
    with pytest.raises(ValidationError):
        DagSpawn(children=["c"], join_at=5.5)


def test_dag_spawn_join_at_scientific_notation_via_orjson(tmp_path: Path):
    """JSON ``5e0`` decodes to float 5.0 in orjson; pydantic then coerces to
    int 5. Result: ``join_at: 5e0`` is treated as ``join_at: 5``."""
    line = {
        "session_id": "p",
        "turns": [
            {
                "messages": [{"role": "user", "content": "u"}],
                "spawns": [{"children": ["c"], "join_at": 5e0}],
            },
            *[_basic_turn(f"u{i}") for i in range(1, 6)],
        ],
    }
    path = _write_bytes(
        tmp_path,
        json.dumps(line).encode() + b"\n" + json.dumps(_basic_conv("c")).encode(),
    )
    convs = DagJsonlLoader(filename=path).load()
    parent = next(c for c in convs if c.session_id == "p")
    assert parent.turns[5].prerequisites[0].branch_id == "p:0"


def test_dag_spawn_join_at_bool_coerced_to_int():
    """Python bools are int subclasses; pydantic accepts ``join_at=True``
    as ``1``. Documents a footgun for authors writing JSON ``true``."""
    s = DagSpawn(children=["c"], join_at=True)
    assert s.join_at == 1
    assert type(s.join_at) is int


def test_dag_spawn_join_at_extreme_int_accepted_pydantic_unbounded():
    """Pydantic ``int`` is unbounded (Python int). Validity is enforced by
    the loader's ``join_at >= num_turns`` range check, not by pydantic."""
    s = DagSpawn(children=["c"], join_at=2**63)
    assert s.join_at == 2**63
    s2 = DagSpawn(children=["c"], join_at=-(2**63))
    assert s2.join_at == -(2**63)


# ---------------------------------------------------------------------------
# 4. JSON object oddities: duplicate keys, extra fields
# ---------------------------------------------------------------------------


def test_jsonl_duplicate_keys_orjson_keeps_last_value(tmp_path: Path):
    """orjson follows the JSON-spec-permissive convention of keeping the
    LAST value for duplicate keys. Verify the loader inherits that and a
    duplicated ``session_id`` resolves to the second value."""
    body = (
        b'{"session_id": "first", "session_id": "second", '
        b'"turns": [{"messages": [{"role": "user", "content": "u"}]}]}'
    )
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].session_id == "second"


def test_dag_turn_extra_top_level_field_rejected():
    """``extra="forbid"`` on DagTurn rejects unknown top-level keys with a
    clear ValidationError pointing at the offending key."""
    with pytest.raises(ValidationError) as excinfo:
        DagTurn.model_validate(
            {
                "messages": [{"role": "user", "content": "u"}],
                "rogue_field": 1,
            }
        )
    assert "rogue_field" in str(excinfo.value)


def test_dag_conversation_forks_null_rejected():
    """``forks: null`` is rejected (declared ``list[str]`` non-Optional)
    even though missing-key uses the default-factory empty list."""
    with pytest.raises(ValidationError):
        DagTurn.model_validate(
            {"messages": [{"role": "user", "content": "u"}], "forks": None}
        )


def test_dag_conversation_spawns_null_rejected():
    """Symmetric to ``forks=null``: ``spawns=null`` rejected."""
    with pytest.raises(ValidationError):
        DagTurn.model_validate(
            {"messages": [{"role": "user", "content": "u"}], "spawns": None}
        )


def test_dag_turn_tools_null_accepted_as_default():
    """``tools`` is declared ``list[...] | None``; explicit ``null`` is
    accepted and stored as ``None``."""
    t = DagTurn.model_validate(
        {"messages": [{"role": "user", "content": "u"}], "tools": None}
    )
    assert t.tools is None


def test_dag_turn_empty_forks_and_spawns_emit_no_branches(tmp_path: Path):
    """Explicit ``forks: []``, ``spawns: []`` matches the default-factory
    behavior: no ConversationBranchInfo entries emitted."""
    line = {
        "session_id": "a",
        "turns": [
            {
                "messages": [{"role": "user", "content": "u"}],
                "forks": [],
                "spawns": [],
            }
        ],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].branches == []


# ---------------------------------------------------------------------------
# 5. Empty file
# ---------------------------------------------------------------------------


def test_jsonl_empty_file_raises(tmp_path: Path):
    """Empty file is rejected with a friendly DagLoadError (the runtime can
    never proceed with 0 conversations, so failing fast at load time is
    the actionable behavior)."""
    path = _write_bytes(tmp_path, b"")
    loader = DagJsonlLoader(filename=path)
    with pytest.raises(DagLoadError, match="empty|no conversations"):
        loader.load_dataset()


def test_jsonl_only_blank_lines_raises(tmp_path: Path):
    """File of only blank/whitespace lines is rejected with the same
    friendly DagLoadError as an empty file."""
    path = _write_bytes(tmp_path, b"\n\n\r\n   \n\t\n")
    with pytest.raises(DagLoadError, match="empty|no conversations"):
        DagJsonlLoader(filename=path).load()


def test_jsonl_utf8_bom_first_line_accepted(tmp_path: Path):
    """A UTF-8 BOM at the start of the file (common from Windows/Excel exports)
    is stripped instead of producing an opaque ``invalid JSON: unexpected
    character`` error."""
    body = b"\xef\xbb\xbf" + json.dumps(_basic_conv()).encode() + b"\n"
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert len(convs) == 1
    assert convs[0].session_id == "a"


def test_jsonl_huge_uncapped_delay_warns(tmp_path: Path, caplog):
    """An authored delay > 60s with no cap configured silently hangs the
    benchmark. Loader must surface a single load-time warning so the user
    sees the problem before the credit phase starts sleeping."""
    convs = [
        {
            "session_id": "a",
            "turns": [
                {"messages": [{"role": "user", "content": "x"}]},
                {"messages": [{"role": "user", "content": "x"}], "delay": 99_999_999.0},
            ],
        }
    ]
    path = _write_lines(tmp_path, convs)
    with caplog.at_level("WARNING", logger="aiperf.dataset.loader.dag_jsonl"):
        DagJsonlLoader(filename=path).load()
    matched = [r for r in caplog.records if "inter-turn delay" in r.message.lower()]
    assert matched, (
        f"expected a delay warning, got: {[r.message for r in caplog.records]}"
    )
    assert matched[0].levelname == "WARNING"


# ---------------------------------------------------------------------------
# 6. Unicode / control chars / null bytes / surrogate handling
# ---------------------------------------------------------------------------


def test_jsonl_message_with_escaped_null_byte_accepted(tmp_path: Path):
    r"""Escaped NUL (\\u0000) in a message string is round-tripped intact.
    The orjson parser accepts the escape and yields a real ``\x00`` byte;
    pydantic does not strip it."""
    body = b'{"session_id":"a","turns":[{"messages":[{"role":"user","content":"pre\\u0000post"}]}]}'
    path = _write_bytes(tmp_path, body)
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_messages[0]["content"] == "pre\x00post"


def test_jsonl_message_with_raw_unescaped_control_char_rejected(tmp_path: Path):
    """A *raw* unescaped control byte in a JSON string body is rejected by
    orjson (RFC-strict). Loader surfaces invalid-JSON cleanly."""
    body = (
        b'{"session_id":"a","turns":[{"messages":[{"role":"u","content":"a\x01b"}]}]}'
    )
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError, match="invalid JSON"):
        DagJsonlLoader(filename=path).load()


def test_jsonl_unpaired_surrogate_rejected_by_orjson(tmp_path: Path):
    r"""``\uD800`` without a low-surrogate pair is rejected by orjson."""
    body = (
        b'{"session_id":"a","turns":[{"messages":[{"role":"u","content":"\\uD800"}]}]}'
    )
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError, match="invalid JSON"):
        DagJsonlLoader(filename=path).load()


def test_jsonl_nfc_vs_nfd_unicode_session_ids_treated_distinct(tmp_path: Path):
    """NFC and NFD forms of the same visual string are distinct keys: the
    loader does not Unicode-normalize session_ids before deduplication."""
    nfc = "café"  # NFC: 4 codepoints
    nfd = "café"  # NFD: 5 codepoints (e + combining acute)
    assert nfc != nfd
    path = _write_lines(tmp_path, [_basic_conv(nfc), _basic_conv(nfd)])
    convs = DagJsonlLoader(filename=path).load()
    sids = {c.session_id for c in convs}
    assert sids == {nfc, nfd}


def test_jsonl_non_utf8_byte_sequence_rejected(tmp_path: Path):
    """A raw latin-1 byte (``0xff``) inside a JSON string body fails orjson's
    UTF-8 strictness."""
    body = b'{"session_id":"a","turns":[{"messages":[{"role":"u","content":"\xff"}]}]}'
    path = _write_bytes(tmp_path, body)
    with pytest.raises(DagLoadError, match="invalid JSON"):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 7. Messages array shape edge cases
# ---------------------------------------------------------------------------


def test_jsonl_messages_entry_string_instead_of_dict_rejected(tmp_path: Path):
    """Each message entry must be a dict; a bare string in the array is
    rejected by ``validate_chat_messages``."""
    line = {"session_id": "a", "turns": [{"messages": ["just a string"]}]}
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    with pytest.raises(DagLoadError):
        DagJsonlLoader(filename=path).load()


def test_jsonl_messages_dict_missing_role_rejected(tmp_path: Path):
    """Messages must carry a ``role`` key; missing-role rejection lives in
    ``validate_chat_messages``."""
    line = {"session_id": "a", "turns": [{"messages": [{"content": "u"}]}]}
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    with pytest.raises(DagLoadError):
        DagJsonlLoader(filename=path).load()


def test_jsonl_message_with_very_long_content_string_accepted(tmp_path: Path):
    """A 1 MiB message content string survives the loader without
    truncation or coercion."""
    blob = "α" * (1024 * 1024)  # 2 MiB UTF-8
    line = {
        "session_id": "a",
        "turns": [{"messages": [{"role": "user", "content": blob}]}],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_messages[0]["content"] == blob


def test_jsonl_message_with_empty_content_string_accepted(tmp_path: Path):
    """An empty ``content`` string is valid (downstream may flag it; the
    loader does not)."""
    line = {
        "session_id": "a",
        "turns": [{"messages": [{"role": "user", "content": ""}]}],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_messages[0]["content"] == ""


# ---------------------------------------------------------------------------
# 8. Tools field
# ---------------------------------------------------------------------------


def test_jsonl_tools_valid_openai_shape_passes_through(tmp_path: Path):
    """OpenAI-spec ``tools`` is stored verbatim on the materialized Turn."""
    tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object"}},
    }
    line = {
        "session_id": "a",
        "turns": [{"messages": [{"role": "user", "content": "u"}], "tools": [tool]}],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].turns[0].raw_tools == [tool]


def test_jsonl_tools_must_be_list_of_dicts(tmp_path: Path):
    """``tools`` typed ``list[dict]``: bare-string entries are rejected."""
    line = {
        "session_id": "a",
        "turns": [
            {"messages": [{"role": "user", "content": "u"}], "tools": ["not a dict"]}
        ],
    }
    path = _write_bytes(tmp_path, json.dumps(line).encode())
    with pytest.raises(DagLoadError):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 9. session_id surface oddities
# ---------------------------------------------------------------------------


def test_jsonl_empty_session_id_rejected_by_pydantic(tmp_path: Path):
    """``session_id`` has ``min_length=1``; ``""`` is rejected at parse."""
    path = _write_bytes(
        tmp_path, json.dumps({"session_id": "", "turns": [_basic_turn()]}).encode()
    )
    with pytest.raises(DagLoadError):
        DagJsonlLoader(filename=path).load()


def test_jsonl_whitespace_only_session_id_currently_accepted(tmp_path: Path):
    """No ``str.strip()`` validator on ``session_id``: whitespace-only ids
    pass pydantic and become live session keys. Pin current behavior so a
    future tightening surfaces the test."""
    path = _write_bytes(
        tmp_path, json.dumps({"session_id": "   ", "turns": [_basic_turn()]}).encode()
    )
    convs = DagJsonlLoader(filename=path).load()
    assert convs[0].session_id == "   "


def test_jsonl_session_id_with_branch_suffix_collision(tmp_path: Path):
    """Hostile authoring: one conversation has ``session_id="x:0"`` (which
    happens to *look* like a branch_id another conversation generates).
    These are independent namespaces and must not conflate; the loader
    resolves both correctly."""
    path = _write_lines(
        tmp_path,
        [
            {"session_id": "leaf", "turns": [_basic_turn()]},
            {
                "session_id": "x:0",  # session-id literally matches an emitted branch_id
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["leaf"],
                    },
                    _basic_turn(),
                ],
            },
            {
                "session_id": "x",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["x:0"],
                    },
                    _basic_turn(),
                ],
            },
        ],
    )
    convs = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    # Branch from 'x' targets the literal session_id 'x:0' and is named
    # 'x:0' (parent_session 'x' + turn 0). The conversation 'x:0' has its
    # own branch named 'x:0:0'. They do not alias.
    x_branch = convs["x"].branches[0]
    assert x_branch.branch_id == "x:0"
    assert x_branch.child_conversation_ids == ["x:0"]
    x0_branch = convs["x:0"].branches[0]
    assert x0_branch.branch_id == "x:0:0"
    assert x0_branch.child_conversation_ids == ["leaf"]


def test_jsonl_session_id_python_keyword_accepted(tmp_path: Path):
    """``class``, ``def``, ``if`` are bare strings to pydantic — accepted."""
    path = _write_lines(
        tmp_path,
        [
            {"session_id": "class", "turns": [_basic_turn()]},
            {"session_id": "def", "turns": [_basic_turn()]},
            {"session_id": "if", "turns": [_basic_turn()]},
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    assert sorted(c.session_id for c in convs) == ["class", "def", "if"]


# ---------------------------------------------------------------------------
# 10. pre_session_spawns cycle / self-cycle / chain
# ---------------------------------------------------------------------------


def test_pre_session_spawns_self_referential_rejected(tmp_path: Path):
    """A conversation listing its own session_id in ``pre_session_spawns``
    creates a self-edge in the DAG, which the cycle check catches."""
    path = _write_bytes(
        tmp_path,
        json.dumps(
            {
                "session_id": "a",
                "pre_session_spawns": ["a"],
                "turns": [_basic_turn()],
            }
        ).encode(),
    )
    with pytest.raises(DagLoadError, match="cycle detected"):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_long_cycle_rejected(tmp_path: Path):
    """A → B → C → ... → A through ``pre_session_spawns`` and per-turn
    spawns is rejected by the cycle detector. Tests N=8 to confirm the
    DFS depth handles realistic cycle depths."""
    chain = [f"node{i}" for i in range(8)]
    lines = []
    for i, sid in enumerate(chain):
        next_sid = chain[(i + 1) % len(chain)]  # last node points back to first
        lines.append(
            {
                "session_id": sid,
                "pre_session_spawns": [next_sid],
                "turns": [_basic_turn()],
            }
        )
    path = _write_lines(tmp_path, lines)
    with pytest.raises(DagLoadError, match="cycle detected"):
        DagJsonlLoader(filename=path).load()


# ---------------------------------------------------------------------------
# 11. Duplicate session_id (entire-conversation duplication attack)
# ---------------------------------------------------------------------------


def test_jsonl_every_line_same_session_id_first_duplicate_caught(tmp_path: Path):
    """Bombing the file with N copies of the same conversation: the loader
    flags the duplicate at line 2 with the offending session_id."""
    same = _basic_conv("only")
    path = _write_lines(tmp_path, [same, same, same, same])
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "line 2" in msg
    assert "only" in msg


# ---------------------------------------------------------------------------
# 12. Legacy spawns regression
# ---------------------------------------------------------------------------


def test_legacy_string_spawns_emits_phase0_branch_layout(tmp_path: Path):
    """Phase 1 introduced object-form ``spawns``; legacy bare-string entries
    must still produce the exact branch_id/prereq layout from before
    (``join_at = idx + 1``, single coalesced branch per turn)."""
    path = _write_lines(
        tmp_path,
        [
            {"session_id": "ca", "turns": [_basic_turn()]},
            {"session_id": "cb", "turns": [_basic_turn()]},
            {
                "session_id": "p",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["ca", "cb"],
                    },
                    _basic_turn(),
                ],
            },
        ],
    )
    convs = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    p = convs["p"]
    assert len(p.branches) == 1
    b = p.branches[0]
    assert b.branch_id == "p:0"
    assert b.mode == ConversationBranchMode.SPAWN
    assert b.child_conversation_ids == ["ca", "cb"]
    # Implicit auto-join on turn 1.
    prereqs = p.turns[1].prerequisites
    assert len(prereqs) == 1
    assert prereqs[0].kind == PrerequisiteKind.SPAWN_JOIN
    assert prereqs[0].branch_id == "p:0"


# ---------------------------------------------------------------------------
# 13. Round-trip serialization: orjson <-> stdlib json
# ---------------------------------------------------------------------------


def _make_full_dag_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
        conversations=[
            ConversationMetadata(
                conversation_id="root",
                is_root=True,
                agent_depth=0,
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=["child_a"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="root:pre",
                        child_conversation_ids=["bg_child"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="pre",
                    ),
                ],
                turns=[
                    TurnMetadata(branch_ids=["root:0", "root:pre"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="root:0",
                            )
                        ]
                    ),
                ],
            ),
            ConversationMetadata(
                conversation_id="child_a", is_root=False, agent_depth=1
            ),
            ConversationMetadata(
                conversation_id="bg_child", is_root=False, agent_depth=1
            ),
        ],
    )


def test_dataset_metadata_orjson_roundtrip_idempotent():
    """orjson.dumps(model_dump) then orjson.loads -> model_validate yields
    an equal DatasetMetadata."""
    m = _make_full_dag_metadata()
    raw = orjson.dumps(m.model_dump(mode="json"))
    parsed = orjson.loads(raw)
    m2 = DatasetMetadata.model_validate(parsed)
    assert m == m2


def test_dataset_metadata_stdlib_json_roundtrip_idempotent():
    """stdlib json.dumps -> json.loads -> model_validate yields an equal
    DatasetMetadata."""
    m = _make_full_dag_metadata()
    raw = json.dumps(m.model_dump(mode="json"))
    parsed = json.loads(raw)
    m2 = DatasetMetadata.model_validate(parsed)
    assert m == m2


def test_dataset_metadata_orjson_dump_loads_with_stdlib_json():
    """Encoding compatibility: orjson-serialized bytes must be parseable by
    stdlib json (and the reverse). Catches any orjson-only escape forms
    that would break interop."""
    m = _make_full_dag_metadata()
    orjson_bytes = orjson.dumps(m.model_dump(mode="json"))
    parsed = json.loads(orjson_bytes.decode())
    m2 = DatasetMetadata.model_validate(parsed)
    assert m == m2


def test_dataset_metadata_stdlib_dump_loads_with_orjson():
    """Reverse direction: stdlib JSON output must orjson-decode equally."""
    m = _make_full_dag_metadata()
    stdlib_bytes = json.dumps(m.model_dump(mode="json")).encode()
    parsed = orjson.loads(stdlib_bytes)
    m2 = DatasetMetadata.model_validate(parsed)
    assert m == m2


def test_dag_conversation_load_then_jsonl_roundtrip_through_models(tmp_path: Path):
    """Build a JSONL file, load through DagJsonlLoader, re-dump each
    conversation through the model, reload — terminal branches and
    prerequisites match across the round-trip."""
    line = {
        "session_id": "p",
        "turns": [
            {
                "messages": [{"role": "user", "content": "u0"}],
                "spawns": [{"children": ["c"], "join_at": 2}],
                "extra": {"temperature": 0.7, "ignore_eos": True},
                "tools": [{"type": "function", "function": {"name": "f"}}],
            },
            _basic_turn("u1"),
            _basic_turn("u2"),
        ],
    }
    path = _write_lines(tmp_path, [line, _basic_conv("c")])
    convs1 = {c.session_id: c for c in DagJsonlLoader(filename=path).load()}
    # Re-validate the parent's source dict round-trip through DagConversation
    # (the wire shape) to confirm the wire model is itself idempotent.
    dc = DagConversation.model_validate(line)
    assert dc.session_id == "p"
    re_dumped = dc.model_dump(mode="json")
    dc2 = DagConversation.model_validate(re_dumped)
    assert dc == dc2
    # Materialized loader output: implicit SPAWN_JOIN on turn 2.
    p = convs1["p"]
    assert p.turns[2].prerequisites[0].branch_id == "p:0"


# ---------------------------------------------------------------------------
# 14. branch_id collision attack — inline forks vs inline spawns suffixing
# ---------------------------------------------------------------------------


def test_branch_id_collision_two_conversations_emit_distinct_ids(tmp_path: Path):
    """Two parents 'a' and 'a:0' both spawn at turn 0; one emits branch_id
    'a:0' (parent='a', turn=0) and the other emits 'a:0:0' (parent='a:0',
    turn=0). Validator must accept both — branch_ids are local to a
    conversation but globally distinct here by construction."""
    path = _write_lines(
        tmp_path,
        [
            {"session_id": "leaf", "turns": [_basic_turn()]},
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
