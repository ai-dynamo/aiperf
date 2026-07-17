# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic acceptance gate for the dag_jsonl graph adapter: raw-export parity.

The in-process byte-parity golden gate
(``tests/component_integration/graph/test_dag_jsonl_byte_parity.py``) proves the
graph adapter emits byte-identical wire bodies to the legacy path at the
transport seam. This file exercises the offline diff tool
(:func:`tools.dag_jsonl_fidelity.prove_parity`) that the external real-run proof
depends on: it drives ``prove_parity`` on hand-crafted synthetic export
fixtures. A matching pair PASSES; corrupted content, a missing request, and
reordered messages each FAIL with a pointed diff (so the tool cannot pass
vacuously). Extra guards prove criterion 3 (body byte-equal) is distinct from
criterion 2 (payload parsed-equal), that an unparsable x_request_id cannot be
silently dropped, and that an empty overlap fails.

The REAL-subprocess half of this gate -- ONE shared ``aiperf-mock-server``
answering a legacy (``--custom-dataset-type dag_jsonl``) and a graph
(``--graph-format dag_jsonl``) ``aiperf profile --export-level raw`` run, then
``prove_parity`` on the two exports -- lives in the INTEGRATION lane at
``tests/integration/graph/test_dag_jsonl_fidelity_real.py`` (Linux-only CI,
real subprocesses), mirroring the weka split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from tools.dag_jsonl_fidelity import prove_parity

# --- synthetic export helpers ---------------------------------------------

# (session_id, turn_index) requests a faithful pair dispatches, keyed identically
# on both planes. root has a spliced assistant reply so a message-order mutation
# has a role transition to catch.
_SPEC: list[tuple[str, int]] = [
    ("root", 0),
    ("branch-a", 0),
    ("branch-a", 1),
]


def _canonical_payload(session: str, turn: int) -> dict[str, Any]:
    """The wire request a FAITHFUL run would export for one ``(session, turn)``.

    Two messages (user + an assistant reply, as a FORK/multi-turn child would
    carry) so a reversed-order mutation flips a role, and a stable top-level key
    order so criterion 3 (body byte-equal) has a fixed reference.
    """
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": f"prompt {session} #{turn}"},
            {"role": "assistant", "content": f"resp-{session}-{turn}"},
        ],
        "max_tokens": 64,
        "stream": True,
    }


def _legacy_row(session: str, turn: int, payload: dict[str, Any]) -> dict[str, Any]:
    """A legacy raw-export line: identity carried directly in metadata."""
    return {
        "metadata": {
            "session_num": 0,
            "conversation_id": session,
            "turn_index": turn,
            "x_correlation_id": f"legacy-{session}-uuid",
            "benchmark_phase": "profiling",
            "request_start_ns": 1_000_000_000_000,
        },
        "payload": payload,
    }


def _graph_row(
    session: str, turn: int, payload: dict[str, Any], *, fire: int = 0
) -> dict[str, Any]:
    """A graph raw-export line: per-node identity folded into x_request_id.

    ``x_correlation_id`` is now an OPAQUE per-trajectory id (``{conversation}::
    {nonce}``) carrying no node identity, so it deliberately does NOT encode the
    node -- the tool must recover identity from ``x_request_id`` (the worker's
    ``_mint_x_request_id`` folds ``{node_id}::{nonce}``). ``conversation_id`` is
    the ONE trajectory TEMPLATE shared by every node and ``metadata.turn_index``
    is a per-correlation fire counter -- the tool must ignore BOTH.
    """
    node_id = f"{session}:{turn}"
    return {
        "metadata": {
            "session_num": 0,
            "conversation_id": "traceX",
            "turn_index": fire,
            "x_correlation_id": f"traceX::corr{fire:08x}",
            "x_request_id": f"{node_id}::req{fire:08x}",
            "benchmark_phase": "profiling",
            "request_start_ns": 1_000_000_000_000,
        },
        "payload": payload,
    }


def _graph_instance_row(
    session: str, turn: int, instance: int, payload: dict[str, Any]
) -> dict[str, Any]:
    """A graph row for the ``instance``-th SPAWN instantiation of one template.

    Instance 1 keeps the bare ``session:turn`` node id; instance >= 2 carries the
    ``#<n>`` suffix the tree mints for repeated SPAWN instances (folded into
    ``x_request_id`` by the worker's ``_mint_x_request_id``). Every instance strips
    back to the SAME recovered ``(session, turn)`` key, so a template fired N times
    contributes multiplicity N to that key.
    """
    node_id = f"{session}:{turn}" if instance == 1 else f"{session}#{instance}:{turn}"
    return {
        "metadata": {
            "session_num": 0,
            "conversation_id": "traceX",
            "turn_index": instance - 1,
            "x_correlation_id": f"traceX::corr{instance:08x}",
            "x_request_id": f"{node_id}::req{instance:08x}",
            "benchmark_phase": "profiling",
            "request_start_ns": 1_000_000_000_000,
        },
        "payload": payload,
    }


def _legacy_rows(specs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [_legacy_row(s, t, _canonical_payload(s, t)) for s, t in specs]


def _graph_rows(specs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        _graph_row(s, t, _canonical_payload(s, t), fire=i)
        for i, (s, t) in enumerate(specs)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    body = b"\n".join(orjson.dumps(r) for r in rows)
    path.write_bytes(body + b"\n" if rows else b"")
    return path


def _write_pair(
    tmp_path: Path,
    legacy_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    return (
        _write_jsonl(tmp_path / "legacy.jsonl", legacy_rows),
        _write_jsonl(tmp_path / "graph.jsonl", graph_rows),
    )


# --- tests -----------------------------------------------------------------


@pytest.mark.component_integration
def test_prove_parity_passes_on_faithful_pair(tmp_path: Path) -> None:
    """A byte-identical faithful pair passes every criterion + coverage invariant."""
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), _graph_rows(_SPEC))
    report = prove_parity(legacy, graph)
    assert report.passed, report.render()
    assert report.messages.checked == len(_SPEC)
    assert report.payload.checked == len(_SPEC)
    assert report.body.checked == len(_SPEC)
    # Non-vacuous: every criterion actually matched a request.
    assert report.messages.passes == len(_SPEC)
    assert report.counts_match and report.keys_match


@pytest.mark.component_integration
def test_prove_parity_fails_on_corrupted_content(tmp_path: Path) -> None:
    """A single mutated message byte FAILS all three criteria for that key only."""
    graph_rows = _graph_rows(_SPEC)
    # Corrupt branch-a turn 0's first user message content.
    graph_rows[1]["payload"]["messages"][0]["content"] += "_CORRUPTED"
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert any(
        "'branch-a'" in m.where and m.where.endswith("turn=0")
        for m in report.messages.mismatches
    ), report.messages.render()
    assert any("content" in m.detail for m in report.messages.mismatches)
    # Only the corrupted request fails; the two untouched ones still pass.
    assert report.messages.passes == len(_SPEC) - 1


@pytest.mark.component_integration
def test_prove_parity_fails_on_missing_request(tmp_path: Path) -> None:
    """A request present in legacy but missing from graph is an unmatched-key fail."""
    graph_rows = _graph_rows(_SPEC)[:-1]  # drop branch-a turn 1
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert ("branch-a", 1) in report.only_legacy
    assert not report.counts_match


@pytest.mark.component_integration
def test_prove_parity_fails_on_reordered_messages(tmp_path: Path) -> None:
    """Reordering a request's messages (same set, wrong order) FAILS with a role diff."""
    graph_rows = _graph_rows(_SPEC)
    graph_rows[0]["payload"][
        "messages"
    ].reverse()  # root: [user, assistant] -> [assistant, user]
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert any("'root'" in m.where for m in report.messages.mismatches)
    assert any("role" in m.detail for m in report.messages.mismatches)
    assert report.messages.passes == len(_SPEC) - 1


@pytest.mark.component_integration
def test_prove_parity_key_order_drift_passes_all_criteria(tmp_path: Path) -> None:
    """Same keys/values in a different order: EVERY criterion passes.

    The graph plane authors model / stream / the token cap through native
    node fields, so wire key POSITION legitimately differs from legacy; the
    body criterion compares CANONICAL (sorted-keys) serializations and must
    not flag pure order drift.
    """
    graph_rows = _graph_rows(_SPEC)
    payload = graph_rows[0]["payload"]
    graph_rows[0]["payload"] = {k: payload[k] for k in reversed(list(payload))}
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert report.passed, report.render()
    assert report.body.passed, report.body.render()


@pytest.mark.component_integration
def test_prove_parity_type_drift_fails_body_not_payload(tmp_path: Path) -> None:
    """Value-TYPE drift Python ``==`` conflates: criterion 2 PASSES, canonical
    criterion 3 FAILS.

    ``stream: 1`` vs ``stream: True`` compare equal as parsed dicts but
    serialize differently, so the canonical body criterion stays a genuinely
    sharper check than parsed equality even without key-order comparison.
    """
    graph_rows = _graph_rows(_SPEC)
    payload = dict(graph_rows[0]["payload"])
    assert payload["stream"] is True
    payload["stream"] = 1
    graph_rows[0]["payload"] = payload
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert report.messages.passed, report.messages.render()
    assert report.payload.passed, report.payload.render()
    assert not report.body.passed, report.body.render()
    assert any("'root'" in m.where for m in report.body.mismatches)
    assert any("value/type drift" in m.detail for m in report.body.mismatches)


@pytest.mark.component_integration
def test_prove_parity_fails_on_unparsable_x_request_id(tmp_path: Path) -> None:
    """A graph row whose x_request_id does not parse cannot be silently dropped.

    Identity-scheme drift is exactly what this gate must catch: the row surfaces
    as an ONLY-GRAPH sentinel key while its legacy counterpart is ONLY-LEGACY, so
    the proof fails loudly instead of vacuously passing.
    """
    graph_rows = _graph_rows(_SPEC)
    graph_rows[0]["metadata"]["x_request_id"] = "garbage-no-delimiters"
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert ("root", 0) in report.only_legacy
    assert any(k[1] == -1 for k in report.only_graph), report.render()


@pytest.mark.component_integration
def test_prove_parity_empty_exports_fail_vacuously(tmp_path: Path) -> None:
    """Two empty exports checked nothing -> the proof must FAIL, not pass."""
    legacy, graph = _write_pair(tmp_path, [], [])
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert not report.counts_match


def _kid_variant(content: str) -> dict[str, Any]:
    """A ``("kid", 0)`` payload with a distinguishing user-message content.

    Two distinct variants keyed identically prove the multiset comparison pairs
    by SORTED body (order-insensitive), not by dispatch order.
    """
    payload = _canonical_payload("kid", 0)
    payload["messages"][0]["content"] = content
    return payload


@pytest.mark.component_integration
def test_prove_parity_passes_on_matching_multiset(tmp_path: Path) -> None:
    """A ``(session, turn)`` key with multiplicity 2 passes when the body multisets
    match -- even when the two instances are dispatched in OPPOSITE order.

    A repeated SPAWN template fires N times under distinct ``#n`` node ids that all
    strip to one ``(session, turn)`` key; the comparator sorts each side's bodies
    and compares element-wise, so order-permuted-but-equal multisets pass.
    """
    a = _kid_variant("kid instance A")
    b = _kid_variant("kid instance B")
    legacy_rows = [
        _legacy_row("root", 0, _canonical_payload("root", 0)),
        _legacy_row("kid", 0, a),
        _legacy_row("kid", 0, b),
    ]
    graph_rows = [
        _graph_row("root", 0, _canonical_payload("root", 0)),
        _graph_instance_row("kid", 0, 1, b),  # opposite order from legacy
        _graph_instance_row("kid", 0, 2, a),
    ]
    legacy, graph = _write_pair(tmp_path, legacy_rows, graph_rows)
    report = prove_parity(legacy, graph)
    assert report.passed, report.render()
    # Multiplicity 2 for ("kid", 0) => 3 comparisons per criterion (root + 2 kids).
    assert report.messages.checked == 3
    assert report.messages.passes == 3
    assert report.counts_match and report.keys_match
    assert report.multiplicity_mismatch == []


@pytest.mark.component_integration
def test_prove_parity_fails_on_corruption_within_multiplicity_key(
    tmp_path: Path,
) -> None:
    """Corrupting ONE of the two bodies under a multiplicity-2 key FAILS with a
    pointed, instance-indexed diff (the other instance still matches).

    Proves the multiset comparator cannot be fooled by a matching sibling: a
    single corrupted instance breaks the sorted element-wise equality.
    """
    a = _kid_variant("kid instance A")
    b = _kid_variant("kid instance B")
    b_corrupt = _kid_variant("kid instance B")
    b_corrupt["messages"][0]["content"] += "_CORRUPTED"
    legacy_rows = [
        _legacy_row("root", 0, _canonical_payload("root", 0)),
        _legacy_row("kid", 0, a),
        _legacy_row("kid", 0, b),
    ]
    graph_rows = [
        _graph_row("root", 0, _canonical_payload("root", 0)),
        _graph_instance_row("kid", 0, 1, a),
        _graph_instance_row("kid", 0, 2, b_corrupt),
    ]
    legacy, graph = _write_pair(tmp_path, legacy_rows, graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert any(
        "'kid'" in m.where and m.where.endswith("turn=0")
        for m in report.messages.mismatches
    ), report.messages.render()
    # The failing mismatch is instance-indexed (which of the N instances failed).
    assert any(
        m.instance is not None and "content" in m.detail
        for m in report.messages.mismatches
    ), report.messages.render()
    # Coverage stays non-vacuous: the matching kid + root still passed.
    assert report.messages.passes == 2


@pytest.mark.component_integration
def test_prove_parity_fails_on_multiplicity_mismatch(tmp_path: Path) -> None:
    """A per-key multiplicity mismatch (2 vs 1) FAILS even when TOTAL counts match.

    Legacy fires ``("kid", 0)`` twice and ``("root", 0)`` once; graph fires
    ``("root", 0)`` twice and ``("kid", 0)`` once. Total profiling counts are
    equal (3 == 3) and the key SETS match, so the failure is isolated to the
    multiplicity guard -- proving it is a distinct invariant, not a proxy for the
    count check.
    """
    a = _kid_variant("kid instance A")
    b = _kid_variant("kid instance B")
    legacy_rows = [
        _legacy_row("root", 0, _canonical_payload("root", 0)),
        _legacy_row("kid", 0, a),
        _legacy_row("kid", 0, b),
    ]
    graph_rows = [
        _graph_row("root", 0, _canonical_payload("root", 0)),
        _graph_instance_row("root", 0, 2, _canonical_payload("root", 0)),
        _graph_instance_row("kid", 0, 1, a),
    ]
    legacy, graph = _write_pair(tmp_path, legacy_rows, graph_rows)
    report = prove_parity(legacy, graph)
    assert not report.passed, report.render()
    assert report.counts_match, report.render()  # total counts DO match (3 == 3)
    assert not report.keys_match, report.render()
    assert (("kid", 0), 2, 1) in report.multiplicity_mismatch
    assert (("root", 0), 1, 2) in report.multiplicity_mismatch


@pytest.mark.component_integration
def test_prove_parity_ignores_warmup_records(tmp_path: Path) -> None:
    """PROFILING-only comparison: a stray graph warmup row is filtered, not compared.

    Graph auto-warmup can prime a node before profiling; those rows carry a
    ``warmup`` phase and must not inflate the graph count or create a spurious
    unmatched key.
    """
    graph_rows = _graph_rows(_SPEC)
    warmup = _graph_row("root", 0, _canonical_payload("root", 0), fire=99)
    warmup["metadata"]["benchmark_phase"] = "warmup"
    legacy, graph = _write_pair(tmp_path, _legacy_rows(_SPEC), [warmup, *graph_rows])
    report = prove_parity(legacy, graph)
    assert report.passed, report.render()
    assert report.graph_profiling == len(_SPEC)
