# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-source memoized graph workload resolution.

Graph-ness is derived AT MOST ONCE per process via
``workload_detect.resolve_graph_workload``: the config resolver chain
populates ``run.resolved.graph_workload`` (+ the ``graph_workload_resolved``
marker) eagerly in single-run mode, and the accessor derives-and-memoizes
lazily for runs that never pass the chain (``aiperf service`` processes,
test-built runs). These tests pin:

* (a) a weka file resolves a ``GraphWorkloadRef`` at resolver-chain time;
* (b) a weka HF ``org/name`` id resolves a ref with the RAW path (never
  ``.resolve()``\\d -- HF ids are not filesystem paths) via the resolver's
  HF early-return branch;
* (c) a synthetic run resolves ``None`` WITH the marker set (distinguishes
  "not a graph run" from "never checked");
* (d) the accessor on a chain-less run derives exactly once across repeated
  calls (memoization);
* (e) an explicit ``--graph-format`` override wins with detection never
  called;
* (f) the timing/config veto: a graph input plus an explicit non-graph
  ``--custom-dataset-type`` stays NON-graph for timing (the veto composes
  OVER the accessor; pinned so the migration cannot absorb or drop it).
"""

from __future__ import annotations

from pathlib import Path

from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli

WEKA_MIN = Path(__file__).resolve().parents[1] / "graph" / "fixtures" / "weka_min.json"

# Weka-marked HF repo id (org/name shape, non-existent local path) -- routed
# through the resolver's HF early-return branch, never the file-existence gate.
HF_WEKA_ID = "semianalysisai/cc-traces-weka-000000"


def _run(**cli_overrides):
    """Build an un-chained ``BenchmarkRun`` from CLI flags (no resolver chain)."""
    cfg = CLIConfig(model_names=["test-model"], **cli_overrides)
    return make_run_from_cli(cfg)


def test_weka_file_resolves_ref_at_resolver_chain_time() -> None:
    run = _run(input_file=str(WEKA_MIN))
    DatasetResolver().resolve(run)
    assert run.resolved.graph_workload_resolved is True
    ref = run.resolved.graph_workload
    assert ref is not None
    assert ref.format == "weka_trace"
    assert ref.path == WEKA_MIN


def test_hf_id_resolves_ref_with_raw_path() -> None:
    run = _run(input_file=HF_WEKA_ID)
    DatasetResolver().resolve(run)
    assert run.resolved.graph_workload_resolved is True
    ref = run.resolved.graph_workload
    assert ref is not None
    assert ref.format == "weka_trace"
    # RAW org/name id verbatim: never .resolve()d into a cwd-anchored path.
    assert ref.path == Path(HF_WEKA_ID)
    assert not ref.path.is_absolute()


def test_synthetic_run_resolves_none_with_marker() -> None:
    run = _run()
    DatasetResolver().resolve(run)
    assert run.resolved.graph_workload_resolved is True
    assert run.resolved.graph_workload is None


def test_accessor_derives_once_and_memoizes(monkeypatch) -> None:
    from aiperf.dataset.graph import workload_detect

    run = _run(input_file=str(WEKA_MIN))
    calls: list[Path] = []
    real_detect = workload_detect._detect_graph_workload_format

    def spy(path: Path):
        calls.append(path)
        return real_detect(path)

    monkeypatch.setattr(workload_detect, "_detect_graph_workload_format", spy)
    first = workload_detect.resolve_graph_workload(run)
    second = workload_detect.resolve_graph_workload(run)
    assert first is not None
    assert first.format == "weka_trace"
    assert second is first
    assert len(calls) == 1, "the second accessor call must read the memo"


def test_graph_format_override_wins_without_detection(monkeypatch, tmp_path) -> None:
    from aiperf.dataset.graph import workload_detect

    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    run = _run(input_file=str(plain), graph_format="native")
    calls: list[Path] = []
    monkeypatch.setattr(
        workload_detect, "_detect_graph_workload_format", lambda p: calls.append(p)
    )
    ref = workload_detect.resolve_graph_workload(run)
    assert ref is not None
    assert ref.format == "native"
    assert ref.path == plain
    assert calls == [], "--graph-format must short-circuit registry detection"


def test_explicit_non_graph_format_vetoes_graph_timing() -> None:
    from aiperf.common.enums import DatasetFormat
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.config import TimingConfig

    run = _run(
        input_file=str(WEKA_MIN),
        custom_dataset_type=DatasetFormat.MULTI_TURN,
        warmup_request_count=2,
        request_count=3,
    )
    tc = TimingConfig.from_run(run)
    assert tc.phase_configs
    assert all(p.timing_mode != TimingMode.GRAPH_IR for p in tc.phase_configs), (
        "an explicit graph-incompatible --custom-dataset-type must veto graph "
        "timing even though the input file sniffs as a graph workload"
    )
