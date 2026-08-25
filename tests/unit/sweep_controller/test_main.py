# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType
from aiperf.sweep_controller.k8s_executor import ChildRunRef
from aiperf.sweep_controller.main import (
    AGGREGATE_READY_MARKER,
    _adaptive_search_log_summary,
    _archive_sampling_design,
    _archive_search_history,
    _load_aggregate_for_cr,
    _load_ready_terminal_phase,
    _mark_sweep_aggregate_ready,
    _mirror_strategy_aggregate_to_sweep_dir,
    _prune_noncanonical_sweep_artifacts,
    _recover_cancelled_terminal_results,
    _terminal_phase_from_aggregate_bundle,
    _write_sweep_parent_aggregate,
    aggregate_marker_exists,
    write_aggregate_marker,
)


@pytest.mark.asyncio
async def test_recover_cancelled_terminal_results_merges_recovered_children() -> None:
    recovered = [SimpleNamespace(label="recovered")]
    executor = SimpleNamespace(
        recover_terminal_results=AsyncMock(return_value=recovered)
    )
    results = [SimpleNamespace(label="current-process")]

    await _recover_cancelled_terminal_results(
        cancel_requested=True,
        executor=executor,
        plan=SimpleNamespace(),
        results=results,
    )

    assert [result.label for result in results] == ["current-process", "recovered"]


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
    """Sweep readiness uses the canonical durable marker transaction."""
    base = tmp_path
    write_aggregate_marker(base)
    marker = base / AGGREGATE_READY_MARKER
    assert orjson.loads(marker.read_bytes()) == {
        "ready": True,
        "was_cancelled": False,
    }
    assert not list(base.glob(f".{AGGREGATE_READY_MARKER}.*.tmp"))


@pytest.mark.parametrize(
    "phase",
    ["Succeeded", "Failed", "PartiallyFailed", "Cancelled"],
)
def test_terminal_phase_from_aggregate_bundle_accepts_terminal_phase(
    phase: str,
) -> None:
    assert _terminal_phase_from_aggregate_bundle({"parent": {"phase": phase}}) == phase


@pytest.mark.parametrize(
    "bundle",
    [{}, {"parent": {}}, {"parent": {"phase": "Running"}}],
)
def test_terminal_phase_from_aggregate_bundle_rejects_incomplete_bundle(
    bundle: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="no valid parent terminal phase"):
        _terminal_phase_from_aggregate_bundle(bundle)


def test_load_ready_terminal_phase_uses_raw_parent_when_inline_bundle_truncates(
    tmp_path: Path,
) -> None:
    parent = (
        tmp_path / "bench" / "sweeps" / "large-sweep" / "1770000000" / "aggregate.json"
    )
    parent.parent.mkdir(parents=True)
    parent.write_bytes(orjson.dumps({"phase": "PartiallyFailed"}))

    assert (
        _load_ready_terminal_phase(tmp_path, "bench", "large-sweep", "1770000000")
        == "PartiallyFailed"
    )


@pytest.mark.asyncio
async def test_sweep_auto_plot_completes_before_aggregate_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    auto_plot = AsyncMock()
    monkeypatch.setattr("aiperf.plot.auto_plot.run_auto_plot_async", auto_plot)
    monkeypatch.setattr(
        "aiperf.sweep_controller.main.asyncio.to_thread",
        AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)),
    )
    sweep_dir = tmp_path / "ns" / "sweeps" / "demo" / "1234567890"
    sweep_dir.mkdir(parents=True)

    await _mark_sweep_aggregate_ready(
        base_dir=tmp_path,
        namespace="ns",
        sweep_name="demo",
        sweep_run_epoch="1234567890",
        auto_plot=True,
        plot_required=True,
        plot_envelope=None,
    )

    auto_plot.assert_awaited_once_with(
        artifact_dir=sweep_dir,
        input_paths=[tmp_path],
        output_dir=sweep_dir / "plots",
        plot_required=True,
        plot_envelope=None,
    )
    assert aggregate_marker_exists(tmp_path) is True


@pytest.mark.asyncio
async def test_sweep_required_auto_plot_failure_withholds_aggregate_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "aiperf.plot.auto_plot.run_auto_plot_async",
        AsyncMock(side_effect=RuntimeError("plot failed")),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        await _mark_sweep_aggregate_ready(
            base_dir=tmp_path,
            namespace="ns",
            sweep_name="demo",
            sweep_run_epoch="1234567890",
            auto_plot=True,
            plot_required=True,
            plot_envelope=None,
        )

    assert aggregate_marker_exists(tmp_path) is False


@pytest.mark.asyncio
async def test_sweep_explicit_auto_plot_false_writes_marker_without_plotting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    auto_plot = AsyncMock()
    monkeypatch.setattr("aiperf.plot.auto_plot.run_auto_plot_async", auto_plot)
    monkeypatch.setattr(
        "aiperf.sweep_controller.main.asyncio.to_thread",
        AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)),
    )
    (tmp_path / "ns" / "sweeps" / "demo" / "1234567890").mkdir(parents=True)

    await _mark_sweep_aggregate_ready(
        base_dir=tmp_path,
        namespace="ns",
        sweep_name="demo",
        sweep_run_epoch="1234567890",
        auto_plot=False,
        plot_required=True,
        plot_envelope=object(),
    )

    auto_plot.assert_not_awaited()
    assert aggregate_marker_exists(tmp_path) is True


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc))


def _child_ref(
    *,
    variation_index: int = 0,
    variation_label: str = "v0",
    trial_index: int | None = None,
    child_run_epoch: str = "1714000042",
) -> ChildRunRef:
    return ChildRunRef(
        namespace="ns",
        name=f"s-v{variation_index:02d}",
        variation_index=variation_index,
        variation_label=variation_label,
        trial_index=trial_index,
        child_run_epoch=child_run_epoch,
        label="cell-0",
        status="Succeeded",
        error="",
    )


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


def test_archive_search_history_moves_file_into_sweep_epoch(tmp_path: Path) -> None:
    source = tmp_path / "search_history.json"
    source.write_bytes(orjson.dumps({"iterations": [{"iteration_idx": 0}]}))

    archived = _archive_search_history(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="adaptive-demo",
        sweep_run_epoch="1770000000",
    )

    expected = (
        tmp_path
        / "bench"
        / "sweeps"
        / "adaptive-demo"
        / "1770000000"
        / "search_history.json"
    )
    assert archived == expected
    assert not source.exists()
    assert orjson.loads(expected.read_bytes()) == {"iterations": [{"iteration_idx": 0}]}


def test_archive_search_history_non_adaptive_is_noop(tmp_path: Path) -> None:
    assert (
        _archive_search_history(
            base_dir=tmp_path,
            namespace="bench",
            sweep_name="grid-demo",
            sweep_run_epoch="1770000000",
        )
        is None
    )


def test_archive_sampling_design_moves_file_into_sweep_epoch(tmp_path: Path) -> None:
    source = tmp_path / "sweep_aggregate" / "sampling_design.json"
    source.parent.mkdir()
    source.write_bytes(orjson.dumps({"type": "sobol", "samples": 2}))

    archived = _archive_sampling_design(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="sobol-demo",
        sweep_run_epoch="1770000000",
    )

    expected = (
        tmp_path
        / "bench"
        / "sweeps"
        / "sobol-demo"
        / "1770000000"
        / "sweep_aggregate"
        / "sampling_design.json"
    )
    assert archived == expected
    assert not source.exists()
    assert orjson.loads(expected.read_bytes()) == {"type": "sobol", "samples": 2}


def test_prune_noncanonical_sweep_artifacts_keeps_only_epoch_bundle(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "bench" / "sweeps" / "demo" / "1770000000"
    canonical.mkdir(parents=True)
    (canonical / "aggregate.json").write_bytes(orjson.dumps({"phase": "Succeeded"}))
    (canonical / "sweep_aggregate").mkdir()
    (canonical / "sweep_aggregate" / "summary.json").write_bytes(b"{}")

    stale_paths = [
        tmp_path / "aggregate" / "summary.json",
        tmp_path / "sweep_aggregate" / "summary.json",
        tmp_path / "bench" / "demo-v00" / "profile_export.jsonl",
        tmp_path / "other-namespace" / "unscoped.json",
        tmp_path / "bench" / "sweeps" / "other-sweep" / "aggregate.json",
        tmp_path / "bench" / "sweeps" / "demo" / "old-epoch" / "aggregate.json",
    ]
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    _prune_noncanonical_sweep_artifacts(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="demo",
        sweep_run_epoch="1770000000",
    )

    assert sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
    ) == [
        "bench",
        "bench/sweeps",
        "bench/sweeps/demo",
        "bench/sweeps/demo/1770000000",
        "bench/sweeps/demo/1770000000/aggregate.json",
        "bench/sweeps/demo/1770000000/sweep_aggregate",
        "bench/sweeps/demo/1770000000/sweep_aggregate/summary.json",
    ]


def test_prune_noncanonical_sweep_artifacts_refuses_ready_tree(tmp_path: Path) -> None:
    canonical = tmp_path / "bench" / "sweeps" / "demo" / "1770000000"
    canonical.mkdir(parents=True)
    stale = tmp_path / "aggregate" / "summary.json"
    stale.parent.mkdir()
    stale.write_bytes(b"do not mutate after ready")
    write_aggregate_marker(tmp_path)

    with pytest.raises(RuntimeError, match="artifact tree marked ready"):
        _prune_noncanonical_sweep_artifacts(
            base_dir=tmp_path,
            namespace="bench",
            sweep_name="demo",
            sweep_run_epoch="1770000000",
        )

    assert stale.read_bytes() == b"do not mutate after ready"


def test_prune_noncanonical_sweep_artifacts_requires_epoch_bundle(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="canonical sweep artifact directory"):
        _prune_noncanonical_sweep_artifacts(
            base_dir=tmp_path,
            namespace="bench",
            sweep_name="demo",
            sweep_run_epoch="1770000000",
        )


def _real_sweep_spec():
    """Validated AIPerfSweepSpec matching the shape the controller reads from the CR."""
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec

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
                "parameters": {"phases.profiling.concurrency": [8, 32]},
            },
            "resultsTtlDays": 17,
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
            )
        ],
        child_runs=[_child_ref(variation_label="concurrency=8")],
        plan=SimpleNamespace(configs=[object(), object()]),
        sweep_run_epoch="1714000000",
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
    assert doc["specSnapshot"]["resultsTtlDays"] == 17
    assert "results_ttl_days" not in doc["specSnapshot"]


def test_write_sweep_parent_aggregate_redacts_legacy_credential_axis(
    tmp_path: Path,
) -> None:
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec

    raw_spec = _real_sweep_spec().model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    raw_spec["benchmark"].setdefault("runtime", {})["ui"] = "none"
    raw_spec["sweep"]["parameters"] = {
        "endpoint.apiKey": ["axis-secret"],
        "endpoint.urls": ["https://user:pass@host/v1?token=query-secret"],
    }
    spec = AIPerfSweepSpec.model_validate(raw_spec)
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=spec,
        results=[
            SimpleNamespace(
                label="run_0001",
                success=True,
                error=None,
                variation_values={},
                variation_label="endpoint.apiKey=axis-secret",
                variation_index=0,
                trial_index=0,
            )
        ],
        child_runs=[_child_ref(variation_label="endpoint.apiKey=axis-secret")],
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1714000000",
    )

    epoch_dir = tmp_path / "ns" / "sweeps" / "s" / "1714000000"
    aggregate = (epoch_dir / "aggregate.json").read_text()
    children = (epoch_dir / "children.json").read_text()
    for secret in ("axis-secret", "user:pass", "query-secret"):
        assert secret not in aggregate
        assert secret not in children
    assert "<redacted>" in aggregate


def test_write_sweep_parent_aggregate_non_model_spec_writes_empty_summary(
    tmp_path: Path,
) -> None:
    """A non-AIPerfSweepSpec spec object degrades to an empty summary dict
    rather than crashing the archive write."""
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=SimpleNamespace(model_dump=lambda mode, by_alias: {}),
        results=[
            SimpleNamespace(
                label="cell-0",
                success=True,
                error=None,
                variation_values={},
                variation_label="v0",
                variation_index=0,
                trial_index=0,
            )
        ],
        child_runs=[_child_ref()],
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1714000000",
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
    )

    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "ns", "name": "s"}},
        spec=SimpleNamespace(model_dump=lambda mode, by_alias: {}),
        results=[result],
        child_runs=[_child_ref(variation_label="search_iter_0000")],
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1714000000",
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
