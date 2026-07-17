# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
from pathlib import Path

import orjson

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentTraceRecord,
    iter_trace_records,
)


def _write(tmp_path, records) -> Path:
    p = tmp_path / "t.jsonl.gz"
    with gzip.open(p, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")
    return p


def _req(session, ts, hashes=None):
    rec = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "agent_context": {"session_id": session},
        "request": {
            "request_id": f"{session}-{ts}",
            "input_tokens": 32,
            "output_tokens": 16,
        },
    }
    if hashes is not None:
        rec["request"]["replay"] = {
            "trace_block_size": 16,
            "input_length": 32,
            "input_sequence_hashes": hashes,
        }
    return rec


def test_parse_current_schema(tmp_path):
    p = _write(tmp_path, [_req("s1", 1000, [11, 22])])
    recs = list(iter_trace_records(p))
    assert len(recs) == 1
    assert recs[0].agent_context.session_id == "s1"
    assert recs[0].request.replay.input_sequence_hashes == [11, 22]


def test_replay_only_record_without_agent_context_or_source(tmp_path):
    # New schema: event_source + agent_context may be absent; model optional.
    p = _write(
        tmp_path,
        [
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 1000,
                "request": {
                    "request_id": "r1",
                    "output_tokens": 4,
                    "replay": {
                        "trace_block_size": 2,
                        "input_length": 4,
                        "input_sequence_hashes": [1, 2],
                    },
                },
            }
        ],
    )
    recs = list(iter_trace_records(p))
    assert recs[0].agent_context is None
    assert recs[0].event_source is None
    assert recs[0].request.model is None


def test_session_id_filter(tmp_path):
    p = _write(tmp_path, [_req("s1", 1000), _req("s2", 1001)])
    recs = list(iter_trace_records(p, session_id="s2"))
    assert len(recs) == 1 and recs[0].agent_context.session_id == "s2"


def test_record_type_is_agent_trace_record(tmp_path):
    p = _write(tmp_path, [_req("s1", 1000)])
    recs = list(iter_trace_records(p))
    assert isinstance(recs[0], AgentTraceRecord)


def test_collect_records_groups_by_session_and_parent_link(tmp_path):
    from aiperf.dataset.graph.adapters.dynamo.trace import (
        _collect_records,
    )

    p = _write(
        tmp_path,
        [
            _req("root", 1000),
            {
                **_req("child", 1100),
                "agent_context": {
                    "session_id": "child",
                    "parent_session_id": "root",
                },
            },
        ],
    )
    by_session, parent_link, skipped_no_context = _collect_records(p, None)
    assert set(by_session) == {"root", "child"}
    assert parent_link == {"child": "root"}
    assert skipped_no_context == 0


def test_collect_records_interns_duplicate_replay_hash_objects(tmp_path):
    """Read-time interning: equal u64 hash values re-listed across turns AND
    across sessions collapse to ONE int object after ``_collect_records`` --
    values unchanged, ``id()``-set cardinality equal to the unique-value count.

    Uses values above CPython's small-int cache so ``orjson.loads`` allocates a
    fresh object per occurrence (small ints like 11/22 are cached singletons and
    would make the identity assertion vacuous).
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import (
        _collect_records,
    )

    a, b, c, d = 2**63 + 1, 2**63 + 2, 2**63 + 3, 2**63 + 4
    p = _write(
        tmp_path,
        [
            _req("s1", 1000, [a, b]),
            _req("s1", 1001, [a, b, c]),
            _req("s2", 1002, [a, d]),
        ],
    )
    by_session, _parent_link, _skipped = _collect_records(p, None)

    all_hashes = [
        h
        for recs in by_session.values()
        for r in recs
        for h in r.request.replay.input_sequence_hashes
    ]
    # Values are preserved exactly (interning shares objects, never mutates).
    assert sorted(all_hashes) == sorted([a, b, a, b, c, a, d])
    # Every occurrence of an equal value is now the SAME object; four unique
    # values -> four distinct ids across all seven slots.
    assert len({id(h) for h in all_hashes}) == len(set(all_hashes)) == 4
