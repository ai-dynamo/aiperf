# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import orjson

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType
from aiperf.sweep_controller.main import (
    AGGREGATE_READY_MARKER,
    _adaptive_search_log_summary,
    _load_aggregate_for_cr,
    _mirror_strategy_aggregate_to_sweep_dir,
    _write_sweep_parent_aggregate,
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


def test_mirror_strategy_aggregate_to_sweep_dir_copies_files_only(tmp_path: Path):
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "profile_export_aiperf_aggregate.json").write_text("{}")
    (aggregate_dir / "profile_export_aiperf_aggregate.csv").write_text("metric,value\n")
    (aggregate_dir / "nested").mkdir()
    (aggregate_dir / "nested" / "skip.json").write_text("{}")

    _mirror_strategy_aggregate_to_sweep_dir(
        base_dir=tmp_path,
        aggregate_dir=aggregate_dir,
        namespace="ns",
        sweep_name="s",
        sweep_run_epoch="1234",
    )

    target = tmp_path / "ns" / "sweeps" / "s" / "1234" / "sweep_aggregate"
    assert sorted(p.name for p in target.iterdir()) == [
        "profile_export_aiperf_aggregate.csv",
        "profile_export_aiperf_aggregate.json",
    ]


def _real_sweep_spec():
    """Validated AIPerfSweepSpec matching the shape the controller reads from the CR."""
    from aiperf.operator.models import AIPerfSweepSpec

    return AIPerfSweepSpec.model_validate(
        {
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
                "variables": {"phases.profiling.concurrency": [8, 32]},
            },
        }
    )


def test_write_sweep_parent_aggregate_writes_spec_summary_contract(
    tmp_path: Path,
) -> None:
    """The archived aggregate.json carries the purpose-built ``specSummary``
    (sweep_type/dimensions/multi_run/convergence) that the operator's
    archived-sweep API consumes verbatim, alongside the full ``specSnapshot``
    dump kept for forensics."""
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=_real_sweep_spec(),
        results=[
            SimpleNamespace(
                label="cell-0",
                success=True,
                error=None,
                variation_values={},
                variation_label="concurrency=8",
                variation_index=0,
                trial_index=0,
                child_run_epoch="1714000042",
            )
        ],
        plan=SimpleNamespace(configs=[object(), object()]),
        sweep_run_epoch="1714000000",
        with_trial_suffix=False,
    )

    aggregate_path = tmp_path / "ns" / "sweeps" / "s" / "1714000000" / "aggregate.json"
    doc = orjson.loads(aggregate_path.read_bytes())
    summary = doc["specSummary"]
    assert summary["sweep_type"] == "grid"
    assert summary["dimensions"] == [{"name": "concurrency", "values": [8, 32]}]
    assert isinstance(summary["multi_run"], dict)
    assert summary["convergence"] is None
    # Full dump stays for forensics / legacy readers.
    assert doc["specSnapshot"]["sweep"]["type"] == "grid"


def test_write_sweep_parent_aggregate_non_model_spec_writes_empty_summary(
    tmp_path: Path,
) -> None:
    """A non-AIPerfSweepSpec spec object degrades to an empty summary dict
    rather than crashing the archive write."""
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=SimpleNamespace(model_dump=lambda mode: {}),
        results=[
            SimpleNamespace(
                label="cell-0",
                success=True,
                error=None,
                variation_values={},
                variation_label="v0",
                variation_index=0,
                trial_index=0,
                child_run_epoch="1714000042",
            )
        ],
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1714000000",
        with_trial_suffix=False,
    )

    aggregate_path = tmp_path / "ns" / "sweeps" / "s" / "1714000000" / "aggregate.json"
    doc = orjson.loads(aggregate_path.read_bytes())
    assert doc["specSummary"] == {}


def test_write_sweep_parent_aggregate_uses_child_run_epoch(tmp_path: Path) -> None:
    result = SimpleNamespace(
        label="cell-0",
        success=True,
        error=None,
        variation_values={},
        variation_label="search_iter_0000",
        variation_index=0,
        trial_index=0,
        child_run_epoch="1714000042",
    )

    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=SimpleNamespace(model_dump=lambda mode: {}),
        results=[result],
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1714000000",
        with_trial_suffix=False,
    )

    children_path = tmp_path / "ns" / "sweeps" / "s" / "1714000000" / "children.json"
    doc = orjson.loads(children_path.read_bytes())
    assert doc["children"][0]["child_run_epoch"] == "1714000042"


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
    assert len(orjson.dumps(bundle)) <= 1000


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


def test_load_aggregate_for_cr_omits_children_when_still_over_cap(
    tmp_path: Path, monkeypatch
):
    """Children must not make the terminal CR aggregate exceed its budget."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"summary": "small"})
    _write_json(
        sweep_dir / "children.json",
        {
            "sweep_run_epoch": "1234",
            "children": [
                {
                    "namespace": "ns",
                    "name": f"s-v{i:04d}",
                    "variation_index": i,
                    "variation_label": "x" * 80,
                    "trial_index": 0,
                    "child_run_epoch": "1234",
                }
                for i in range(250)
            ],
        },
    )
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", {"small": 1}
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.main._AGGREGATE_INLINE_MAX_BYTES", 1000
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert len(orjson.dumps(bundle)) <= 1000
    assert "children" not in bundle
    assert bundle["childrenTruncated"] == {
        "reason": "inline_status_budget_exceeded",
        "total": 250,
        "included": 0,
        "sweep_run_epoch": "1234",
    }


def test_load_aggregate_for_cr_drops_confidence_then_omits_children(
    tmp_path: Path, monkeypatch
):
    """The post-confidence-drop bundle is rechecked before patching status."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"summary": "small"})
    _write_json(
        sweep_dir / "children.json",
        {
            "sweep_run_epoch": "1234",
            "children": [
                {"name": f"child-{i}", "payload": "y" * 50} for i in range(200)
            ],
        },
    )
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json",
        {f"row_{i}": list(range(20)) for i in range(200)},
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.main._AGGREGATE_INLINE_MAX_BYTES", 1000
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert len(orjson.dumps(bundle)) <= 1000
    assert "confidence" not in bundle
    assert "children" not in bundle
    assert bundle["childrenTruncated"]["total"] == 200


def test_load_aggregate_for_cr_skips_malformed_pareto_keeps_others(tmp_path: Path):
    """One corrupt aggregate file (truncated bytes -> orjson.JSONDecodeError)
    must NOT abort the whole bundle — the other artifacts still need to land
    on the CR. The pre-fix except-clause caught only FileNotFoundError so a
    corrupt sibling crashed the controller pod with a non-zero exit, losing
    all three artifacts.
    """
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "sweep-x" / "1778027124"
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

    bundle = _load_aggregate_for_cr(base_dir, "ns", "sweep-x", "1778027124")

    assert "parent" in bundle
    assert "confidence" in bundle
    assert "children" not in bundle, (
        "malformed children.json must be skipped, not poison the bundle"
    )
