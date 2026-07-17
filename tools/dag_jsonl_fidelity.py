# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical-proof tool: legacy ``dag_jsonl`` vs graph ``dag_jsonl`` raw-export parity.

This is the Task-7 EXTERNAL acceptance gate for the dag_jsonl graph adapter. It
does NOT touch the live run path; it reads the deterministic offline artifacts
two profiling runs produced (each a ``profile_export_raw.jsonl``) -- one from the
LEGACY custom-dataset path (``--custom-dataset-type dag_jsonl``) and one from the
GRAPH adapter path (``--graph-format dag_jsonl``) -- and proves, per matching
request keyed by ``(session_id, turn_index)``, three independent fidelity
criteria plus two coverage invariants:

1. :attr:`ParityReport.messages` (criterion 1) -- the exported ``payload.messages``
   are byte-identical (``orjson.dumps`` equal) between the two planes.
2. :attr:`ParityReport.payload` (criterion 2) -- the full parsed request payloads
   are equal (order-insensitive dict equality: same keys, same values).
3. :attr:`ParityReport.body` (criterion 3) -- the CANONICAL re-serializations are
   byte-identical: ``orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)`` on each
   side. Key ORDER is deliberately NOT compared (the graph plane authors the
   stream flag / token cap / model through native node fields, so their wire
   positions differ from legacy); canonical bytes stay strict on VALUES AND
   TYPES -- e.g. ``1`` vs ``1.0`` or ``True`` vs ``1`` serialize differently
   even though Python ``==`` (criterion 2) treats them as equal.

Plus COUNT equality (both planes dispatched the same number of PROFILING requests)
and NO-UNMATCHED-KEYS (every ``(session, turn)`` present on one plane is present
on the other, with the SAME multiplicity on both). A proof that compared ZERO
requests FAILS -- an empty overlap must never pass vacuously.

Node-identity recovery (the linchpin): the two planes carry request identity
differently in the raw export.

* LEGACY rows carry identity directly: ``metadata.conversation_id`` is the dag
  ``session_id`` and ``metadata.turn_index`` is the turn within that session.
* GRAPH rows carry the trajectory TEMPLATE id in ``metadata.conversation_id``
  (shared across every node of a trajectory and duplicated across recycles), so
  the per-node identity is recovered from ``metadata.x_request_id``, which the
  worker's ``_mint_x_request_id`` folds as ``{node_id}::{nonce}`` (the nonce is a
  ``uuid4().hex`` keeping the id fresh per dispatch, and it never contains ``::``).
  The node id is ``"<session>[#n]:<turn_idx>"`` (the ``#n`` instance suffix appears
  only for repeated SPAWN instances); we split off the trailing ``::<nonce>`` from
  the RIGHT, strip the ``#n`` suffix, and split the trailing ``:<turn_idx>`` back
  out. Rows whose ``x_request_id`` lacks the ``::`` nonce delimiter (or a numeric
  trailing turn) are non-graph/malformed and keep a loud sentinel key so coverage
  failures never pass vacuously.

REPEATED SPAWN TEMPLATES -- multiset keying: stripping the ``#n`` suffix collapses
every instance of one SPAWN template onto the same ``(session, turn)`` key, so a
template instantiated N times per tree contributes multiplicity N to that key. The
proof keys each plane into ``dict[key, list[body]]`` and, per key, sorts both
lists by CANONICAL serialized body and compares them ELEMENT-WISE: parity therefore
means the two planes emitted the SAME multiset of bodies for that key, and the
NO-UNMATCHED-KEYS invariant additionally requires equal per-key MULTIPLICITY (a
key fired twice on one plane and once on the other fails). This works without
instance-level pairing because a repeated template's instances carry identical
authored bytes; a corrupted instance breaks the sorted element-wise equality and
surfaces with an instance-indexed pointed diff.

Runtime multiplicity assumption (verified by the ``spawn-repeat`` gate case in
``tests/component_integration/graph/test_dag_jsonl_byte_parity.py``): the LEGACY
plane keys each spawn-child dispatch by its TEMPLATE ``conversation_id`` (the dag
``session_id``), not an instance-unique id, so a template spawned N times emits N
rows under the same ``(session, turn)`` legacy key -- matching the graph plane's N
``#n`` node instances. If a future runtime change makes legacy conversation ids
instance-unique, :func:`_legacy_key` must strip that instance discriminator to
keep the multiplicities aligned.

Run offline against any pair of artifacts; the live runs that produce them are
driven by ``tests/component_integration/graph/test_dag_jsonl_fidelity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

_PROFILING = "profiling"

Key = tuple[str, int]


# --- report ---------------------------------------------------------------


@dataclass
class Mismatch:
    """One failing comparison: which request key and why (pointed, not a blob).

    ``instance`` is the position within a multiplicity>1 key's sorted body list
    (``None`` for a single-instance key), so a repeated SPAWN template's failing
    instance is pinpointed rather than blurred across its siblings.
    """

    where: str
    detail: str
    instance: int | None = None


@dataclass
class Criterion:
    """Structured pass/fail result of one parity criterion.

    ``checked`` counts every ``(session, turn)`` compared; ``passes`` counts the
    subset that matched. ``passed`` requires at least one comparison AND zero
    mismatches -- a criterion that checked nothing is not a pass.
    """

    name: str
    checked: int = 0
    passes: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.checked > 0

    def fail(self, where: str, detail: str, *, instance: int | None = None) -> None:
        self.mismatches.append(Mismatch(where=where, detail=detail, instance=instance))

    def render(self) -> str:
        head = (
            f"{self.name}: {'PASS' if self.passed else 'FAIL'} "
            f"(checked={self.checked}, passed={self.passes}, "
            f"mismatches={len(self.mismatches)})"
        )
        lines = [head]
        for m in self.mismatches:
            inst = f" instance#{m.instance}" if m.instance is not None else ""
            lines.append(f"  MISMATCH @ {m.where}{inst}: {m.detail}")
        return "\n".join(lines)


@dataclass
class ParityReport:
    """Full legacy-vs-graph parity result: three criteria + coverage invariants.

    ``passed`` is the conjunction of: matching PROFILING request COUNTS (and at
    least one request), NO unmatched keys AND equal per-key multiplicity, and
    every criterion passing. Any single failure fails the whole proof; the CLI
    exit code is ``0`` iff ``passed``.
    """

    legacy_profiling: int
    graph_profiling: int
    only_legacy: list[Key] = field(default_factory=list)
    only_graph: list[Key] = field(default_factory=list)
    multiplicity_mismatch: list[tuple[Key, int, int]] = field(default_factory=list)
    """``(key, legacy_count, graph_count)`` for each shared key whose per-key
    multiplicity differs between the planes (a repeated SPAWN template fired a
    different number of times on each side)."""
    messages: Criterion = field(
        default_factory=lambda: Criterion("messages_byte_equal")
    )
    payload: Criterion = field(
        default_factory=lambda: Criterion("payload_parsed_equal")
    )
    body: Criterion = field(default_factory=lambda: Criterion("body_canonical_equal"))

    @property
    def counts_match(self) -> bool:
        """Both planes dispatched the same NONZERO number of PROFILING requests."""
        return (
            self.legacy_profiling == self.graph_profiling and self.legacy_profiling > 0
        )

    @property
    def keys_match(self) -> bool:
        """Same key SET on both planes AND the same per-key multiplicity."""
        return not (self.only_legacy or self.only_graph or self.multiplicity_mismatch)

    @property
    def passed(self) -> bool:
        return (
            self.counts_match
            and self.keys_match
            and self.messages.passed
            and self.payload.passed
            and self.body.passed
        )

    def render(self) -> str:
        head = (
            f"dag_jsonl parity: {'PASS' if self.passed else 'FAIL'} "
            f"(legacy_profiling={self.legacy_profiling}, "
            f"graph_profiling={self.graph_profiling})"
        )
        lines = [head]
        if not self.counts_match:
            lines.append(
                f"  COUNT MISMATCH: legacy={self.legacy_profiling} "
                f"graph={self.graph_profiling}"
            )
        for key in self.only_legacy:
            lines.append(f"  ONLY-LEGACY key {key} (missing from graph export)")
        for key in self.only_graph:
            lines.append(f"  ONLY-GRAPH key {key} (missing from legacy export)")
        for key, legacy_n, graph_n in self.multiplicity_mismatch:
            lines.append(
                f"  MULTIPLICITY MISMATCH key {key}: "
                f"legacy fired {legacy_n}x, graph fired {graph_n}x"
            )
        lines.append(self.messages.render())
        lines.append(self.payload.render())
        lines.append(self.body.render())
        return "\n".join(lines)


# --- raw export record ----------------------------------------------------


@dataclass
class _ExportRecord:
    """The parity-relevant projection of one raw-export JSONL line.

    ``payload`` is the FULL wire request dict (criteria 2 and 3 need every key,
    not just ``messages``). ``conversation_id`` / ``turn_index`` carry the legacy
    identity directly; ``x_request_id`` carries the graph node identity folded by
    the worker's ``_mint_x_request_id`` as ``{node_id}::{nonce}``.
    """

    conversation_id: str
    turn_index: int | None
    x_request_id: str | None
    phase: str
    payload: dict[str, Any]

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.payload.get("messages", []) or []


def load_raw_export(raw_jsonl: Path) -> list[_ExportRecord]:
    """Parse a ``profile_export_raw.jsonl`` into parity projections.

    Each line is one dispatched request. ``metadata`` carries ``conversation_id``
    / ``turn_index`` / ``x_request_id`` / ``benchmark_phase``; ``payload`` is the
    full wire request dict. Blank lines are skipped.
    """
    records: list[_ExportRecord] = []
    for line in Path(raw_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {}) or {}
        payload = obj.get("payload", {}) or {}
        records.append(
            _ExportRecord(
                conversation_id=str(meta.get("conversation_id") or ""),
                turn_index=meta.get("turn_index"),
                x_request_id=meta.get("x_request_id"),
                phase=str(meta.get("benchmark_phase") or ""),
                payload=payload,
            )
        )
    return records


def _legacy_key(rec: _ExportRecord) -> Key:
    """Legacy identity straight off the record: ``(conversation_id, turn_index)``."""
    return (rec.conversation_id, int(rec.turn_index or 0))


def _graph_key(rec: _ExportRecord) -> Key | None:
    """Recover ``(session_id, turn_index)`` from the graph ``x_request_id``.

    Per-dispatch node identity rides ``x_request_id`` as ``{node_id}::{nonce}``
    (the worker's ``_mint_x_request_id``); the node id is the dag coordinate
    ``"<session>[#n]:<turn>"`` and the nonce is a ``uuid4().hex`` with no ``::``.
    We split off the trailing ``::<nonce>`` from the RIGHT, then split the node
    id's trailing ``:<turn>`` and strip any ``#n`` SPAWN-instance suffix from the
    session. Returns ``None`` for any x_request_id lacking the ``::`` nonce
    delimiter or a numeric trailing turn (non-graph/malformed rows).
    """
    xreq = rec.x_request_id
    if not xreq or "::" not in xreq:
        return None
    node_id = xreq.rsplit("::", 1)[0]
    session_part, sep, turn_str = node_id.rpartition(":")
    if not sep or not turn_str.isdigit():
        return None
    session = session_part.split("#", 1)[0]
    return (session, int(turn_str))


def _index_legacy(
    records: list[_ExportRecord],
) -> dict[Key, list[dict[str, Any]]]:
    """Index legacy PROFILING payloads by ``(session, turn)`` into a multiset.

    A template spawned N times keys N payloads under the same ``(session, turn)``
    (legacy carries the TEMPLATE ``conversation_id``, not an instance-unique id),
    so the value is a LIST -- one entry per dispatch under that key.
    """
    keyed: dict[Key, list[dict[str, Any]]] = {}
    for rec in records:
        keyed.setdefault(_legacy_key(rec), []).append(rec.payload)
    return keyed


def _index_graph(
    records: list[_ExportRecord],
) -> dict[Key, list[dict[str, Any]]]:
    """Index graph PROFILING payloads by recovered ``(session, turn)`` into a multiset.

    Repeated SPAWN instances (distinct ``#n`` node ids) strip to the same key and
    accumulate in the key's payload list. A record whose ``x_request_id`` does not
    parse is keyed under a UNIQUE sentinel so it surfaces as an ONLY-GRAPH
    unmatched key rather than being silently dropped (an identity-scheme drift
    must not pass vacuously).
    """
    keyed: dict[Key, list[dict[str, Any]]] = {}
    for i, rec in enumerate(records):
        key = _graph_key(rec)
        if key is None:
            key = (f"<unparsable x_request_id #{i}>", -1)
        keyed.setdefault(key, []).append(rec.payload)
    return keyed


# --- pointed diffs --------------------------------------------------------


def _snip(value: Any, limit: int = 80) -> str:
    """A short repr of a value for a pointed (non-blob) mismatch detail."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _messages_diff(legacy: list[dict[str, Any]], graph: list[dict[str, Any]]) -> str:
    """A compact description of the first differing message."""
    if len(legacy) != len(graph):
        return f"message count graph={len(graph)} != legacy={len(legacy)}"
    for idx, (le, gr) in enumerate(zip(legacy, graph, strict=False)):
        if le != gr:
            if le.get("role") != gr.get("role"):
                return (
                    f"msg[{idx}] role graph={gr.get('role')!r} != "
                    f"legacy={le.get('role')!r}"
                )
            return (
                f"msg[{idx}] role={gr.get('role')!r} content "
                f"graph={_snip(gr.get('content'))} != "
                f"legacy={_snip(le.get('content'))}"
            )
    return "messages differ under byte compare but no positional field diff found"


def _payload_diff(legacy: dict[str, Any], graph: dict[str, Any]) -> str:
    """A compact description of the first payload key that differs in value/presence."""
    only_legacy = sorted(set(legacy) - set(graph))
    only_graph = sorted(set(graph) - set(legacy))
    if only_legacy:
        return f"key {only_legacy[0]!r} present in legacy, absent in graph"
    if only_graph:
        return f"key {only_graph[0]!r} present in graph, absent in legacy"
    for key in legacy:
        if legacy[key] != graph[key]:
            if key == "messages":
                return f"messages differ: {_messages_diff(legacy[key], graph[key])}"
            return (
                f"key {key!r}: graph={_snip(graph[key])} != legacy={_snip(legacy[key])}"
            )
    return "payloads differ under equality but no key diff found"


def _canonical(payload: dict[str, Any]) -> bytes:
    """Canonical order-insensitive serialization: sorted keys at every level."""
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)


def _body_diff(legacy: dict[str, Any], graph: dict[str, Any]) -> str:
    """Describe a parsed-equal but canonically byte-different body (value/type
    drift Python ``==`` conflates, e.g. 1 vs 1.0 or True vs 1)."""
    return (
        f"parsed-equal but CANONICALLY byte-different (value/type drift): "
        f"graph={_canonical(graph)!r} != legacy={_canonical(legacy)!r}"
    )


# --- the proof ------------------------------------------------------------


def prove_parity(legacy_export: Path, graph_export: Path) -> ParityReport:
    """Prove legacy-vs-graph dag_jsonl raw-export parity, keyed ``(session, turn)``.

    Loads both exports, keeps only PROFILING records (the comparable subset --
    warmup dispatch counts may legitimately differ), keys each plane into a
    ``dict[key, list[body]]`` multiset (:func:`_index_legacy` / :func:`_index_graph`),
    then for every common key sorts both bodies lists by serialized bytes and
    checks the three criteria ELEMENT-WISE across the aligned pairs. Coverage
    invariants (count equality, no unmatched keys, equal per-key multiplicity) are
    recorded on the returned :class:`ParityReport`.

    Repeated SPAWN templates are supported: their ``#n`` instances strip to one
    ``(session, turn)`` key and are compared as a multiset. A per-key multiplicity
    difference (fired N times on one plane, M on the other) is a
    MULTIPLICITY-MISMATCH failure; the criteria are compared only for keys whose
    multiplicities agree, so a corrupted instance under an agreeing key still fails
    loudly with an instance-indexed pointed diff.
    """
    legacy_recs = [r for r in load_raw_export(legacy_export) if r.phase == _PROFILING]
    graph_recs = [r for r in load_raw_export(graph_export) if r.phase == _PROFILING]

    legacy_by_key = _index_legacy(legacy_recs)
    graph_by_key = _index_graph(graph_recs)

    only_legacy = sorted(set(legacy_by_key) - set(graph_by_key))
    only_graph = sorted(set(graph_by_key) - set(legacy_by_key))

    report = ParityReport(
        legacy_profiling=len(legacy_recs),
        graph_profiling=len(graph_recs),
        only_legacy=only_legacy,
        only_graph=only_graph,
    )

    for key in sorted(set(legacy_by_key) & set(graph_by_key)):
        legacy_bodies = sorted(legacy_by_key[key], key=_canonical)
        graph_bodies = sorted(graph_by_key[key], key=_canonical)
        if len(legacy_bodies) != len(graph_bodies):
            report.multiplicity_mismatch.append(
                (key, len(legacy_bodies), len(graph_bodies))
            )
            continue

        multi = len(legacy_bodies) > 1
        where = f"session={key[0]!r} turn={key[1]}"
        for idx, (legacy_payload, graph_payload) in enumerate(
            zip(legacy_bodies, graph_bodies, strict=True)
        ):
            instance = idx if multi else None

            # Criterion 1: messages byte-equal.
            report.messages.checked += 1
            legacy_msgs = legacy_payload.get("messages", []) or []
            graph_msgs = graph_payload.get("messages", []) or []
            if orjson.dumps(legacy_msgs) != orjson.dumps(graph_msgs):
                report.messages.fail(
                    where, _messages_diff(legacy_msgs, graph_msgs), instance=instance
                )
            else:
                report.messages.passes += 1

            # Criterion 2: full payload parsed-equal.
            report.payload.checked += 1
            if legacy_payload != graph_payload:
                report.payload.fail(
                    where,
                    _payload_diff(legacy_payload, graph_payload),
                    instance=instance,
                )
            else:
                report.payload.passes += 1

            # Criterion 3: canonical (sorted-keys) body byte-equal.
            report.body.checked += 1
            if _canonical(legacy_payload) != _canonical(graph_payload):
                report.body.fail(
                    where, _body_diff(legacy_payload, graph_payload), instance=instance
                )
            else:
                report.body.passes += 1

    return report


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``python tools/dag_jsonl_fidelity.py <legacy.jsonl> <graph.jsonl>``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "legacy", type=Path, help="legacy (--custom-dataset-type dag_jsonl) raw export"
    )
    parser.add_argument(
        "graph", type=Path, help="graph (--graph-format dag_jsonl) raw export"
    )
    args = parser.parse_args(argv)
    report = prove_parity(args.legacy, args.graph)
    print(report.render())
    return 0 if report.passed else 1


__all__ = [
    "Criterion",
    "Mismatch",
    "ParityReport",
    "load_raw_export",
    "prove_parity",
]


if __name__ == "__main__":
    raise SystemExit(_main())
