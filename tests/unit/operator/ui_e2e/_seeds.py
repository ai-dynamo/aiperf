# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for seeding the on-PVC layout consumed by the operator results-server.

Designed to be the ONE source of truth that every adversarial-test agent uses,
so the shape of the seeded data stays consistent across test files. Helpers
return paths so the caller can layer extra fixtures on top.

Every helper is a thin wrapper around the disk layout — no fancy state, no
hidden globals. Authoring an adversarial test should feel like::

    def test_partial_summary_renders_phase_unknown(harness):
        seed_run(
            harness.results_dir, ns="ns1", name="j", epoch="1714069323",
            summary={"status": "Cancelled"},
        )
        page = harness.goto_job_detail("ns1", "j", epoch="1714069323")
        assert "Operator API unreachable" not in page.locator("body").inner_text()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def seed_run(
    base: Path,
    *,
    ns: str,
    name: str,
    epoch: str,
    summary: dict[str, Any] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    extra_files: dict[str, bytes] | None = None,
    is_latest: bool = False,
    sweep_marker: dict[str, Any] | None = None,
) -> Path:
    """Create ``<base>/<ns>/<name>/<epoch>/`` with the requested artifacts.

    Args:
        base: Session-shared results directory.
        ns: Namespace component of the on-disk path.
        name: AIPerfJob name component of the on-disk path.
        epoch: Decimal-seconds run epoch (validated against EPOCH_RE upstream;
            must be 10 digits like "1714069323").
        summary: Body to write at ``profile_export_aiperf.json``. ``None``
            seeds a run dir with NO summary (the bug-repro case).
        conditions: Optional list to write at ``conditions.json``.
        extra_files: Map of relative filename -> bytes. Useful for seeding
            run-scoped artifacts the results router serves (e.g.,
            ``profile_export_genai_perf.csv``).
        is_latest: If True, also write ``<base>/<ns>/<name>/latest.txt``
            pointing at this epoch.
        sweep_marker: Optional dict written to ``<base>/<ns>/<name>/sweep.json``
            for sweep-linkage tests. Reused across epochs of the same name.

    Returns:
        The run directory path.
    """
    run = base / ns / name / epoch
    run.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (run / "profile_export_aiperf.json").write_text(json.dumps(summary))
    if conditions is not None:
        (run / "conditions.json").write_text(json.dumps(conditions))
    if extra_files:
        for rel, data in extra_files.items():
            target = run / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if is_latest:
        (base / ns / name / "latest.txt").write_text(epoch)
    if sweep_marker is not None:
        (base / ns / name / "sweep.json").write_text(json.dumps(sweep_marker))
    return run


def seed_results_ready(run: Path) -> None:
    """Drop the ``.aiperf_results_ready.json`` marker so the results router
    will serve top-level files. Without this marker the sidecar refuses
    artifact GETs (see CLAUDE.md "Results-ready marker")."""
    (run / ".aiperf_results_ready.json").write_text(
        json.dumps({"ready_at": int(time.time())})
    )


def seed_sweep_aggregate(
    base: Path,
    *,
    ns: str,
    sweep: str,
    epoch: str,
    aggregate: dict[str, Any] | None = None,
    children: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    runs: list[dict[str, Any]] | None = None,
    is_latest: bool = False,
) -> Path:
    """Create ``<base>/<ns>/sweeps/<sweep>/<epoch>/`` with aggregate artifacts.

    Mirrors the layout the sweep-controller harvester writes — the operator
    reads ``aggregate.json`` and ``children.json`` (see
    ``src/aiperf/operator/sweep_union.py`` ``_AGGREGATE_FILE`` constant).

    Args:
        aggregate: Body for ``aggregate.json`` — the cross-variation rollup the
            operator consumes through ``find_any_sweep``.
        children: Body for ``children.json`` — per-variation slim summaries
            (``runs`` schema).
        summary, runs: Back-compat aliases for callers from the original
            harness rev. ``summary`` writes ``profile_export_aiperf.json`` and
            ``runs`` writes ``runs.json``. Prefer ``aggregate``/``children``
            for new tests — those are what the operator actually reads.
        is_latest: Also write ``latest.txt`` pointing at this epoch.
    """
    sweep_dir = base / ns / "sweeps" / sweep / epoch
    sweep_dir.mkdir(parents=True, exist_ok=True)
    if aggregate is not None:
        (sweep_dir / "aggregate.json").write_text(json.dumps(aggregate))
    if children is not None:
        (sweep_dir / "children.json").write_text(json.dumps(children))
    if summary is not None:
        (sweep_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
    if runs is not None:
        (sweep_dir / "runs.json").write_text(json.dumps(runs))
    if is_latest:
        (base / ns / "sweeps" / sweep / "latest.txt").write_text(epoch)
    return sweep_dir


@dataclass(slots=True)
class FakeLiveCR:
    """Stand-in for a live AIPerfJob CR. Tests register one of these per
    (ns, name) to simulate cluster state without touching kubernetes_asyncio.

    Only the fields the operator UI reads through ``find_aiperf_job`` are
    populated; everything else defaults to a sensible empty value.
    """

    name: str
    namespace: str
    phase: str = "Running"
    job_id: str | None = None
    workers_ready: int = 0
    workers_total: int = 0
    current_phase: str | None = None
    error: str | None = None
    start_time: str | None = None
    completion_time: str | None = None
    # ``AIPerfJobInfo.created`` is typed ``str`` (not ``str | None``) and the
    # operator's list-pages drop any CR that fails ``model_validate`` here, so
    # leave this as a non-None default to keep live-only CRs visible in
    # ``GET /api/v1/jobs``.
    created: str = ""
    progress_percent: float | None = None
    throughput_rps: float | None = None
    latency_p99_ms: float | None = None
    model: str | None = None
    endpoint: str | None = None
    sweep_name: str | None = None
    variation_index: int | None = None
    variation_label: str | None = None
    # ``raw_cr``: optional full CR dict matching the apiserver shape, used by
    # ``get_raw_aiperfjob`` on the operator side (``/api/v1/config/{ns}/{name}``
    # and ``/api/v1/jobs/{ns}/{name}/events``). ``None`` is fine when the test
    # doesn't exercise those routes — the patched lookup returns ``None`` and
    # the routes render the appropriate "unavailable" fallback.
    raw_cr: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_info(self):
        """Build an actual ``AIPerfJobInfo`` for the patched find_aiperf_job."""
        from aiperf.kubernetes.models import AIPerfJobInfo

        return AIPerfJobInfo(
            name=self.name,
            namespace=self.namespace,
            phase=self.phase,
            job_id=self.job_id or self.name,
            workers_ready=self.workers_ready,
            workers_total=self.workers_total,
            current_phase=self.current_phase,
            error=self.error,
            start_time=self.start_time,
            completion_time=self.completion_time,
            created=self.created,
            progress_percent=self.progress_percent,
            throughput_rps=self.throughput_rps,
            latency_p99_ms=self.latency_p99_ms,
            model=self.model,
            endpoint=self.endpoint,
            sweep_name=self.sweep_name,
            variation_index=self.variation_index,
            variation_label=self.variation_label,
        )


# Canonical successful summary used as the baseline for "happy" tests.
def good_summary(
    *,
    throughput_rps: float = 100.0,
    latency_p99_ms: float = 50.0,
    model: str = "llama3-8b",
    endpoint: str = "http://srv:8000/v1",
    request_count: int = 1000,
) -> dict[str, Any]:
    """Return a profile_export-shaped dict that exercises all the KPI paths
    the dashboard reads. Use as a starting point and mutate from there.
    """
    return {
        "status": "Succeeded",
        "start_time": "2026-05-19T00:00:00Z",
        "end_time": "2026-05-19T00:05:00Z",
        "request_throughput": {"avg": throughput_rps, "unit": "requests/sec"},
        "request_latency": {"avg": latency_p99_ms / 2, "p99": latency_p99_ms},
        "time_to_first_token": {"avg": 30.0, "p99": 80.0},
        "output_token_throughput": {"avg": 800.0},
        "inter_token_latency": {"avg": 5.0},
        "request_count": {"avg": request_count, "unit": "count"},
        "input_config": {
            "models": {"items": [{"name": model}]},
            "endpoint": {"urls": [endpoint]},
        },
    }
