# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType
from aiperf.sweep_controller.main import (
    AGGREGATE_READY_MARKER,
    _adaptive_search_log_summary,
    _load_aggregate_for_cr,
    aggregate_marker_exists,
    write_aggregate_marker,
)


def test_adaptive_search_log_summary_uses_objectives_list() -> None:
    sweep = AdaptiveSearchSweep(
        planner=SearchPlannerType.BAYESIAN,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=40, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=5,
        n_initial_points=2,
    )

    summary = _adaptive_search_log_summary(sweep)

    assert summary == (
        "planner=bayesian, max_iterations=5, "
        "objectives=output_token_throughput:avg:maximize"
    )


def test_aggregate_marker_lifecycle(tmp_path: Path):
    base = tmp_path / "results"
    base.mkdir()
    assert aggregate_marker_exists(base) is False
    write_aggregate_marker(base)
    assert aggregate_marker_exists(base) is True
    assert (base / AGGREGATE_READY_MARKER).exists()


def test_aggregate_marker_atomic_rename(tmp_path: Path):
    """Marker is written via .tmp + rename; partial writes don't appear ready."""
    base = tmp_path
    write_aggregate_marker(base)
    assert (base / AGGREGATE_READY_MARKER).exists()
    # No leftover .tmp
    assert not (base / (AGGREGATE_READY_MARKER + ".tmp")).exists()


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc))


def test_load_aggregate_for_cr_loads_all_three_keys(tmp_path: Path):
    """Small bundle: parent + children + confidence all inlined."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"parent": "ok"})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", {"k": "v"}
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert bundle["parent"] == {"parent": "ok"}
    assert bundle["children"] == [{"name": "c1"}]
    assert bundle["confidence"] == {"k": "v"}


def test_load_aggregate_for_cr_drops_confidence_when_over_size_cap(
    tmp_path: Path, monkeypatch
):
    """Bundle over the inline cap drops `confidence` to keep CR patch < 1MB.

    K8s rejects CR patches over ~1 MiB with HTTP 413. The aggregator
    docstring says confidence grows linearly with cells x metrics x
    percentiles — on big sweeps it dominates. We bound the inlined size:
    parent + children stay (small, structural metadata); confidence is
    served via the disk-backed results sidecar instead.
    """
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"summary": "small"})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    # ~50 KB confidence payload, well above the test cap below.
    big_confidence = {f"row_{i}": list(range(50)) for i in range(500)}
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", big_confidence
    )

    # Lower the cap to force the drop branch.
    monkeypatch.setattr(
        "aiperf.sweep_controller.main._AGGREGATE_INLINE_MAX_BYTES", 1000
    )
    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert "parent" in bundle
    assert "children" in bundle
    assert "confidence" not in bundle, (
        "confidence must be dropped when bundle exceeds inline cap"
    )


def test_load_aggregate_for_cr_keeps_confidence_under_cap(tmp_path: Path):
    """Default cap is generous enough that small confidence payloads stay inlined."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"a": 1})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", {"small": 1}
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")
    assert "confidence" in bundle


def test_load_aggregate_for_cr_skips_malformed_pareto_keeps_others(tmp_path: Path):
    """One corrupt aggregate file (truncated bytes -> orjson.JSONDecodeError)
    must NOT abort the whole bundle — the other artifacts still need to land
    on the CR. The pre-fix except-clause caught only FileNotFoundError so a
    corrupt sibling crashed the controller pod with a non-zero exit, losing
    all three artifacts.
    """
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "ajc-sweep-x" / "1778027124"
    sweep_dir.mkdir(parents=True)
    aggregate_dir = base_dir / "aggregate"
    aggregate_dir.mkdir(parents=True)

    # Valid parent + valid confidence; truncated children.
    _write_json(sweep_dir / "aggregate.json", {"sweep": "x", "epoch": 1778027124})
    (sweep_dir / "children.json").write_bytes(b'[{"name":"c1","status":')  # truncated
    _write_json(
        aggregate_dir / "profile_export_aiperf_aggregate.json",
        {"metadata": {"num_successful_runs": 6}},
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "ajc-sweep-x", "1778027124")

    assert "parent" in bundle
    assert "confidence" in bundle
    assert "children" not in bundle, (
        "malformed children.json must be skipped, not poison the bundle"
    )
