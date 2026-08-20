# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archived spec-summary contract between the sweep-controller and the reader.

The producer (``sweep_controller.main._write_sweep_parent_aggregate``) writes
``spec_summary_snapshot(spec)`` under ``SPEC_SUMMARY_KEY`` into
``aggregate.json``; ``spec_summary_from_record`` consumes exactly that shape
once the CR is TTL-reaped. These tests pin the round-trip, the legacy
full-dump fallback, and graceful degradation for corrupt archives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson
import pytest
from aiperf.operator.routers._sweeps_spec import (
    _LATE_ADDED_SNAPSHOT_KEYS as _LATE_ADDED,
)
from aiperf.operator.routers._sweeps_spec import (
    LEGACY_SPEC_SNAPSHOT_KEY,
    SPEC_SUMMARY_KEY,
    spec_summary_from_record,
    spec_summary_snapshot,
)
from pytest import param

from aiperf.kubernetes.crd_models import AIPerfSweepSpec

_RAW_SPEC: dict[str, Any] = {
    "benchmark": {
        "models": {"items": [{"name": "llama-3"}]},
        "endpoint": {"urls": ["http://server:8000/v1/chat/completions"]},
        "datasets": [{"name": "main", "type": "synthetic"}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 1,
            }
        ],
    },
    "sweep": {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [8, 32]},
    },
}


@dataclass
class _Record:
    """Minimal SweepRecord stand-in with the fields the reader touches."""

    namespace: str = "bench"
    name: str = "latency-sweep"
    raw_spec: dict[str, Any] = field(default_factory=dict)
    aggregate_doc: dict[str, Any] | None = None


def _json_roundtrip(doc: dict[str, Any]) -> dict[str, Any]:
    """Serialize + reparse, matching what disk persistence does to the dict."""
    return orjson.loads(orjson.dumps(doc))


def _spec() -> AIPerfSweepSpec:
    return AIPerfSweepSpec.model_validate(_RAW_SPEC)


def test_spec_summary_snapshot_shape() -> None:
    snap = spec_summary_snapshot(_spec())

    assert snap["sweep_type"] == "grid"
    assert snap["dimensions"] == [{"name": "concurrency", "values": [8, 32]}]
    assert isinstance(snap["multi_run"], dict)
    assert snap["convergence"] is None


def test_reader_round_trips_producer_snapshot() -> None:
    """What the producer writes under SPEC_SUMMARY_KEY, the reader consumes."""
    snap = _json_roundtrip(spec_summary_snapshot(_spec()))
    rec = _Record(aggregate_doc={SPEC_SUMMARY_KEY: snap})

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert [d.name for d in summary.dimensions] == ["concurrency"]
    assert [d.values for d in summary.dimensions] == [[8, 32]]
    assert summary.multi_run == snap["multi_run"]
    assert summary.convergence is None


def test_reader_derives_summary_from_legacy_full_spec_dump() -> None:
    """Old archives carry only the full spec dump under specSnapshot; the
    reader re-validates the sweep/multi_run sub-blocks and derives the same
    summary the new key would have carried."""
    legacy_dump = _json_roundtrip(_spec().model_dump(mode="json"))
    rec = _Record(aggregate_doc={LEGACY_SPEC_SNAPSHOT_KEY: legacy_dump})

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert [d.name for d in summary.dimensions] == ["concurrency"]
    assert [d.values for d in summary.dimensions] == [[8, 32]]


def test_reader_prefers_new_key_over_legacy() -> None:
    snap = _json_roundtrip(spec_summary_snapshot(_spec()))
    bogus_legacy = {"sweep": {"type": "zip", "parameters": {"x": [1]}}}
    rec = _Record(
        aggregate_doc={
            SPEC_SUMMARY_KEY: snap,
            LEGACY_SPEC_SNAPSHOT_KEY: bogus_legacy,
        }
    )

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"


@pytest.mark.parametrize(
    "aggregate_doc",
    [
        param({}, id="no-spec-keys"),
        param({LEGACY_SPEC_SNAPSHOT_KEY: {"sweep": {"type": "bogus"}}}, id="bad-sweep-type"),
        param({LEGACY_SPEC_SNAPSHOT_KEY: {"no_sweep_block": True}}, id="missing-sweep-block"),
        param({SPEC_SUMMARY_KEY: "not-a-dict"}, id="summary-wrong-type"),
        param({SPEC_SUMMARY_KEY: {}}, id="summary-empty"),
    ],
)  # fmt: skip
def test_reader_degrades_gracefully_on_unusable_archives(
    aggregate_doc: dict[str, Any],
) -> None:
    rec = _Record(aggregate_doc=aggregate_doc)

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert summary.dimensions == []
    assert summary.multi_run is None
    assert summary.convergence is None


def test_reader_degrades_on_bare_valueerror_from_model_validate(monkeypatch) -> None:
    """A BARE ValueError from ``model_validate`` must degrade, not 500 the route.

    ``pydantic.ValidationError`` subclasses ``ValueError``, but a malformed
    distribution value makes ``AIPerfSweepSpec.model_validate`` raise a plain
    ``builtins.ValueError`` that is not a ``ValidationError``. The old
    ``except ValidationError`` missed it and let it escape, 500ing the summary
    route; the broadened ``except ValueError`` catches it (using ``str(exc)``
    since a bare ValueError has no structured ``.errors()``) and falls back.
    """

    class _RaisesBareValueError:
        @staticmethod
        def model_validate(_raw):
            raise ValueError("could not coerce distribution value '512' to a float")

    monkeypatch.setattr(
        "aiperf.operator.routers._sweeps_spec.AIPerfSweepSpec",
        _RaisesBareValueError,
    )
    rec = _Record(raw_spec=_RAW_SPEC)  # truthy raw_spec, no archived aggregate

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert summary.dimensions == []


_ADAPTIVE_RAW_SPEC: dict[str, Any] = {
    **_RAW_SPEC,
    "sweep": {
        "type": "adaptive_search",
        "max_iterations": 14,
        "search_space": [{"path": "phases.profiling.concurrency", "lo": 1, "hi": 512}],
        "objectives": [
            {
                "metric": "output_token_throughput",
                "stat": "avg",
                "direction": "maximize",
            }
        ],
        "sla_filters": [
            {
                "metric_tag": "time_to_first_token",
                "stat": "p99",
                "op": "lt",
                "threshold": 500.0,
            }
        ],
    },
}


def _adaptive_spec() -> AIPerfSweepSpec:
    return AIPerfSweepSpec.model_validate(_ADAPTIVE_RAW_SPEC)


def test_snapshot_carries_objectives_and_sla_filters() -> None:
    snap = spec_summary_snapshot(_adaptive_spec())

    assert [o["metric"] for o in snap["objectives"]] == ["output_token_throughput"]
    assert [f["metricTag"] for f in snap["sla_filters"]] == ["time_to_first_token"]


def test_reader_backfills_objectives_from_legacy_dump_on_older_archive() -> None:
    """A real archived adaptive sweep carries BOTH keys, and the newer one is stale.

    ``objectives`` / ``sla_filters`` were added to ``spec_summary_snapshot``
    after archives were already being written, so an archive from an older
    sweep-controller has a structurally valid four-key ``specSummary`` next to
    a ``specSnapshot`` that still holds the full sweep block. Preferring the
    newer key wholesale returned null for both -- ``pickObjectiveWinner`` then
    returned null and the page ranked by its chart metric instead, which is
    exactly what these fields exist to prevent.
    """
    snap = _json_roundtrip(spec_summary_snapshot(_adaptive_spec()))
    older_snapshot = {k: v for k, v in snap.items() if k not in _LATE_ADDED}
    rec = _Record(
        aggregate_doc={
            SPEC_SUMMARY_KEY: older_snapshot,
            LEGACY_SPEC_SNAPSHOT_KEY: _json_roundtrip(
                _adaptive_spec().model_dump(mode="json")
            ),
        }
    )

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "adaptive_search"
    assert [o["metric"] for o in summary.objectives or []] == [
        "output_token_throughput"
    ]
    assert [f["metricTag"] for f in summary.sla_filters or []] == [
        "time_to_first_token"
    ]


def test_backfill_never_overwrites_a_populated_snapshot() -> None:
    """Only the missing keys are topped up; the producer's own values stand."""
    snap = _json_roundtrip(spec_summary_snapshot(_adaptive_spec()))
    contradicting_legacy = _json_roundtrip(_spec().model_dump(mode="json"))
    rec = _Record(
        aggregate_doc={
            SPEC_SUMMARY_KEY: snap,
            LEGACY_SPEC_SNAPSHOT_KEY: contradicting_legacy,
        }
    )

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "adaptive_search"
    assert [o["metric"] for o in summary.objectives or []] == [
        "output_token_throughput"
    ]


def test_backfill_is_a_noop_when_the_legacy_dump_declares_no_objective() -> None:
    """Grid archives legitimately have neither key; nothing is invented."""
    snap = _json_roundtrip(spec_summary_snapshot(_spec()))
    rec = _Record(
        aggregate_doc={
            SPEC_SUMMARY_KEY: snap,
            LEGACY_SPEC_SNAPSHOT_KEY: _json_roundtrip(_spec().model_dump(mode="json")),
        }
    )

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert summary.objectives is None
    assert summary.sla_filters is None


def test_reader_live_path_still_wins_over_archive() -> None:
    """A validatable live raw_spec takes precedence over the archived doc."""
    rec = _Record(
        raw_spec=_RAW_SPEC,
        aggregate_doc={
            SPEC_SUMMARY_KEY: {
                "sweep_type": "zip",
                "dimensions": [{"name": "stale", "values": [0]}],
            }
        },
    )

    summary = spec_summary_from_record(rec)

    assert summary.sweep_type == "grid"
    assert [d.name for d in summary.dimensions] == ["concurrency"]
