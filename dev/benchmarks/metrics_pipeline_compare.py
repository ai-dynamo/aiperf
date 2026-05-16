#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver: run ``metrics_pipeline_worker.py`` against both worktrees and compare.

Compares the new MetricsAccumulator + ColumnStore + RaggedSeries pipeline
(this worktree, ``ajc/k8s-metrics``) against the old MetricResultsProcessor +
MetricArray + TDigestListMetricAggregator pipeline
(``../new-config-kube``, branch ``ajc/k8s``).

Two sweeps:
  1. n_records sweep at fixed avg_icl_chunks (memory + throughput vs scale)
  2. avg_icl_chunks sweep at fixed n_records (ICL ragged-vs-tdigest growth)

Outputs JSON, PNG plots, and a markdown table to
``dev/benchmarks/results/metrics_pipeline_<timestamp>/``.

Usage:
    uv run python dev/benchmarks/metrics_pipeline_compare.py
    uv run python dev/benchmarks/metrics_pipeline_compare.py --quick
    uv run python dev/benchmarks/metrics_pipeline_compare.py \\
        --old-worktree /path/to/old --no-plots
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKTREE_NEW = Path(__file__).resolve().parents[2]
WORKTREE_OLD_DEFAULT = Path("/home/anthony/nvidia/projects/aiperf/ajc/new-config-kube")
WORKER_REL = Path("dev/benchmarks/metrics_pipeline_worker.py")


@dataclass(frozen=True)
class Case:
    """One worker invocation. Synthetic mode: ``n_records``+``avg_icl_chunks``.
    File-replay mode: ``records_file``+``repeat`` (n_records is then derived as
    ``source_lines * repeat`` and surfaced post-run; avg_icl_chunks is read off
    the source).
    """

    n_records: int
    avg_icl_chunks: int
    slice_duration: float | None = None
    records_file: Path | None = None
    repeat: int = 1

    @property
    def label(self) -> str:
        if self.records_file is not None:
            return f"file:{self.records_file.name}_x{self.repeat}"
        s = f"n{self.n_records}_icl{self.avg_icl_chunks}"
        if self.slice_duration is not None:
            s += f"_slice{self.slice_duration}"
        return s


def _build_n_records_sweep(quick: bool) -> list[Case]:
    ns = [10_000, 50_000] if quick else [10_000, 50_000, 100_000, 500_000, 1_000_000]
    return [Case(n_records=n, avg_icl_chunks=100) for n in ns]


def _build_icl_sweep(quick: bool) -> list[Case]:
    icls = [10, 50, 200] if quick else [10, 50, 100, 200, 500]
    return [Case(n_records=100_000, avg_icl_chunks=k) for k in icls]


def _build_file_replay_sweep(records_file: Path, repeats: list[int]) -> list[Case]:
    return [
        Case(
            n_records=0,
            avg_icl_chunks=0,
            records_file=records_file,
            repeat=r,
        )
        for r in repeats
    ]


def _run_worker(
    worktree: Path,
    case: Case,
    *,
    timeout_s: int,
    skip_breakdown: bool,
) -> dict[str, Any]:
    """Invoke the worker in ``worktree`` and return parsed JSON."""
    args: list[str] = ["uv", "run", "python", str(WORKER_REL)]
    if case.records_file is not None:
        args += [
            "--records-file",
            str(case.records_file),
            "--repeat",
            str(case.repeat),
        ]
    else:
        args += [
            "--n-records",
            str(case.n_records),
            "--avg-icl-chunks",
            str(case.avg_icl_chunks),
        ]
    if case.slice_duration is not None:
        args += ["--slice-duration", str(case.slice_duration)]
    if skip_breakdown:
        args.append("--no-asizeof")

    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
        "AIPERF_LOG_LEVEL": "WARNING",
    }
    proc = subprocess.run(  # noqa: S603 - benchmark driver, args are constructed from typed fields
        args,
        cwd=str(worktree),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed in {worktree} for {case.label}: rc={proc.returncode}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    # The worker prints a single JSON line; some runs emit per-record warnings on
    # stderr (and a few escape to stdout via the AIPerf logger). Find the JSON line.
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(
            f"no JSON line in worker stdout for {worktree} / {case.label}\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads(lines[-1])


def _ensure_worker_in_old(old_worktree: Path) -> None:
    src = WORKTREE_NEW / WORKER_REL
    dst = old_worktree / WORKER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _format_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def _run_all(
    cases: list[Case],
    old_worktree: Path,
    *,
    timeout_s: int,
    skip_breakdown: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(cases) * 2
    idx = 0
    for case in cases:
        for tree, side in ((WORKTREE_NEW, "new"), (old_worktree, "old")):
            idx += 1
            t0 = time.perf_counter()
            print(
                f"[{idx}/{total}] {side:3s} | {case.label} ...",
                file=sys.stderr,
                flush=True,
            )
            try:
                payload = _run_worker(
                    tree, case, timeout_s=timeout_s, skip_breakdown=skip_breakdown
                )
            except Exception as e:  # noqa: BLE001 - report and continue across the grid
                print(f"  FAILED: {e}", file=sys.stderr, flush=True)
                continue
            elapsed = time.perf_counter() - t0
            # In file-replay mode the case kwargs are zero-placeholders; back-fill
            # them from the worker's source summary so the pivot key is meaningful.
            if case.records_file is not None:
                src = payload.get("source", {})
                payload["case"] = {
                    "n_records": int(src.get("total_records", payload["n_records"])),
                    "avg_icl_chunks": int(
                        round(src.get("avg_chunks_per_record_in_source", 0))
                    ),
                    "slice_duration": case.slice_duration,
                    "records_file": str(case.records_file),
                    "repeat": case.repeat,
                }
            else:
                payload["case"] = {
                    "n_records": case.n_records,
                    "avg_icl_chunks": case.avg_icl_chunks,
                    "slice_duration": case.slice_duration,
                }
            payload["expected_side"] = side
            payload["wall_clock_total_s"] = elapsed
            results.append(payload)
            ingest_ms = payload["ingest"]["wall_time_s"] * 1000
            sum_ms = payload["summarize"]["wall_time_s"] * 1000
            peak_mb = payload["ingest"]["tracemalloc_peak_bytes"] / 1024 / 1024
            rss_mb = payload["final_rss_bytes"] / 1024 / 1024
            print(
                f"  -> ingest {ingest_ms:7.1f} ms ({payload['ingest']['records_per_second']:>8.0f} rec/s) | "
                f"summarize {sum_ms:7.1f} ms | peak {peak_mb:6.2f} MB | rss {rss_mb:6.1f} MB",
                file=sys.stderr,
                flush=True,
            )
    return results


def _pivot(results: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, dict]]:
    """Group results by (case n_records, case avg_icl_chunks) -> {pipeline: payload}.

    Keys come from ``r["case"]`` (the requested target), not ``r["avg_icl_chunks"]``
    (the worker's sample-mean readback) — the latter drifts from the target because
    chunk counts are Poisson-sampled.
    """
    out: dict[tuple[int, int], dict[str, dict]] = {}
    for r in results:
        key = (r["case"]["n_records"], r["case"]["avg_icl_chunks"])
        out.setdefault(key, {})[r["pipeline"]] = r
    return out


def _scalar_arrays_bytes(p: dict[str, Any]) -> int:
    """Return the scalar-RECORD metric storage in bytes for a worker payload.

    New pipeline: ColumnStore._numeric (NaN-sparse float64 arrays, capacity-doubled).
    Old pipeline: sum of MetricArray sizes across RECORD-typed scalar metrics.
    """
    parts = p["breakdown"]["parts"]
    if p["pipeline"] == "new":
        return int(parts.get("column_store_numeric_arrays", 0))
    return int(parts.get("metric_array_total", 0))


def _ragged_or_tdigest_bytes(p: dict[str, Any]) -> int:
    parts = p["breakdown"]["parts"]
    if p["pipeline"] == "new":
        return int(parts.get("column_store_ragged", 0))
    return int(parts.get("tdigest_total", 0))


def _new_only_overhead_bytes(p: dict[str, Any]) -> int:
    """Bytes the new ColumnStore tracks that the old pipeline doesn't store at all:
    timestamps, metadata_numeric, metadata_string, running sums."""
    if p["pipeline"] != "new":
        return 0
    parts = p["breakdown"]["parts"]
    return int(
        parts.get("column_store_timestamps", 0)
        + parts.get("column_store_metadata_numeric", 0)
        + parts.get("column_store_metadata_string", 0)
        + parts.get("column_store_running_sums", 0)
    )


def _markdown_table(results: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    pivot = _pivot(results)

    rows.append("### Headline: scalar metric arrays + throughput")
    rows.append("")
    rows.append(
        "| n_records | avg_icl | side | scalar arrays (KiB) | per-tag (KiB) | "
        "ingest ms | rec/s | summarize ms |"
    )
    rows.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for key in sorted(pivot.keys()):
        for side, p in sorted(pivot[key].items()):
            scalar_kib = _scalar_arrays_bytes(p) / 1024
            n_tags = (
                len(p["breakdown"]["parts"].get("per_tag_numeric", {}))
                if p["pipeline"] == "new"
                else len(p["breakdown"]["parts"].get("per_tag_metric_array", {}))
            )
            per_tag = scalar_kib / max(1, n_tags)
            ingest_ms = p["ingest"]["wall_time_s"] * 1000
            rps = p["ingest"]["records_per_second"]
            sum_ms = p["summarize"]["wall_time_s"] * 1000
            rows.append(
                f"| {key[0]} | {key[1]} | {side} | {scalar_kib:.1f} | "
                f"{per_tag:.1f} ({n_tags} tags) | "
                f"{ingest_ms:.1f} | {rps:.0f} | {sum_ms:.1f} |"
            )

    rows.append("")
    rows.append("### Per-tag scalar arrays (largest case, both sides present)")
    rows.append("")
    # Prefer the largest n_records that has BOTH pipelines present, otherwise the
    # largest one available — works for synthetic sweeps and file-replay alike.
    both_keys = [k for k in pivot if {"new", "old"} <= set(pivot[k])]
    target_pool = both_keys or list(pivot.keys())
    if target_pool:
        target = max(target_pool, key=lambda k: (k[0], k[1]))
        rows.append(f"_Sampled at n_records={target[0]}, avg_icl_chunks={target[1]}_")
        rows.append("")
        new_p = pivot[target].get("new")
        old_p = pivot[target].get("old")
        if new_p and old_p:
            new_per_tag = new_p["breakdown"]["parts"].get("per_tag_numeric", {})
            old_per_tag = old_p["breakdown"]["parts"].get("per_tag_metric_array", {})
            all_tags = sorted(set(new_per_tag) | set(old_per_tag))
            rows.append("| metric tag | new (KiB) | old (KiB) | new / old |")
            rows.append("|---|---:|---:|---:|")
            for tag in all_tags:
                n_kib = new_per_tag.get(tag, 0) / 1024
                o_kib = old_per_tag.get(tag, 0) / 1024
                ratio = (n_kib / o_kib) if o_kib else float("inf")
                rows.append(f"| {tag} | {n_kib:.1f} | {o_kib:.1f} | {ratio:.2f}x |")

    rows.append("")
    rows.append("### Per-field metadata (new pipeline only — old discards metadata)")
    rows.append("")
    if target_pool:
        target = max(target_pool, key=lambda k: (k[0], k[1]))
        new_p = pivot[target].get("new")
        if new_p:
            parts = new_p["breakdown"]["parts"]
            md_str: dict[str, int] = parts.get("per_field_metadata_string", {})
            md_num: dict[str, int] = parts.get("per_field_metadata_numeric", {})
            ts: dict[str, int] = parts.get("per_field_timestamps", {})
            running_sums = parts.get("column_store_running_sums", 0)
            rows.append(f"_Sampled at n_records={target[0]}_")
            rows.append("")
            rows.append("| group | field | KiB |")
            rows.append("|---|---|---:|")
            for tag, sz in sorted(md_str.items(), key=lambda kv: -kv[1]):
                rows.append(f"| metadata_string | {tag} | {sz / 1024:.1f} |")
            for tag, sz in sorted(md_num.items(), key=lambda kv: -kv[1]):
                rows.append(f"| metadata_numeric | {tag} | {sz / 1024:.1f} |")
            for tag, sz in sorted(ts.items(), key=lambda kv: -kv[1]):
                rows.append(f"| timestamps | {tag} | {sz / 1024:.1f} |")
            rows.append(
                f"| running_sums | _sums + _counts_ | {running_sums / 1024:.1f} |"
            )
            grand = (
                sum(md_str.values())
                + sum(md_num.values())
                + sum(ts.values())
                + running_sums
            )
            rows.append(f"| **total** | | **{grand / 1024:.1f}** |")

    rows.append("")
    rows.append("### Total memory + ICL + new-only overhead")
    rows.append("")
    rows.append(
        "| n_records | avg_icl | side | scalar (KiB) | ICL store | new-only extras (KiB) | "
        "total MB | peak MB | rss MB |"
    )
    rows.append("|---:|---:|---|---:|---|---:|---:|---:|---:|")
    for key in sorted(pivot.keys()):
        for side, p in sorted(pivot[key].items()):
            scalar_kib = _scalar_arrays_bytes(p) / 1024
            icl_b = _ragged_or_tdigest_bytes(p)
            extras_kib = _new_only_overhead_bytes(p) / 1024
            total_mb = p["breakdown"]["total_bytes"] / 1024 / 1024
            peak_mb = p["ingest"]["tracemalloc_peak_bytes"] / 1024 / 1024
            rss_mb = p["final_rss_bytes"] / 1024 / 1024
            if p["pipeline"] == "new":
                icl_label = f"{icl_b / 1024 / 1024:.2f} MB ragged"
            else:
                icl_label = f"{icl_b:.0f} B tdigest"
            extras_str = f"{extras_kib:.1f}" if side == "new" else "n/a"
            rows.append(
                f"| {key[0]} | {key[1]} | {side} | {scalar_kib:.1f} | {icl_label} | "
                f"{extras_str} | {total_mb:.2f} | {peak_mb:.2f} | {rss_mb:.1f} |"
            )
    return "\n".join(rows)


def _try_plots(results: list[dict[str, Any]], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)
    pivot = _pivot(results)

    # Detect which avg_icl_chunks value to use for the n_records-axis plots.
    # Synthetic sweep uses 100; file-replay uses whatever the source's avg is —
    # in either case there's typically one dominant value across the n-sweep.
    n_records_plot_icl: int | None = None
    icl_counts: dict[int, int] = {}
    for k in pivot:
        icl_counts[k[1]] = icl_counts.get(k[1], 0) + 1
    if icl_counts:
        n_records_plot_icl = max(icl_counts, key=lambda v: icl_counts[v])

    # n_records sweep at the dominant avg_icl_chunks value
    n_keys = sorted(k for k in pivot if k[1] == n_records_plot_icl)
    if n_keys:
        ns = [k[0] for k in n_keys]

        def _plot_metric(
            getter, ylabel: str, title: str, fname: str, *, log_y: bool = False
        ) -> None:
            fig, ax = plt.subplots(figsize=(8, 5))
            for side, marker in (("new", "o"), ("old", "s")):
                ys = [getter(pivot[k].get(side)) for k in n_keys]
                if all(y is not None for y in ys):
                    ax.plot(ns, ys, marker=marker, label=side)
            ax.set_xlabel("n_records")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / fname, dpi=110)
            plt.close(fig)

        _plot_metric(
            lambda p: None if p is None else _scalar_arrays_bytes(p) / 1024,
            "scalar metric arrays (KiB)",
            f"Scalar metric arrays vs n_records (avg_icl_chunks={n_records_plot_icl})",
            "scalar_arrays_vs_records.png",
            log_y=True,
        )
        _plot_metric(
            lambda p: None
            if p is None
            else p["breakdown"]["total_bytes"] / 1024 / 1024,
            "total pipeline structure size (MB)",
            f"Pipeline memory vs n_records (avg_icl_chunks={n_records_plot_icl})",
            "memory_total_vs_records.png",
            log_y=True,
        )
        _plot_metric(
            lambda p: None if p is None else p["ingest"]["records_per_second"],
            "ingest throughput (records/s)",
            f"Ingest throughput vs n_records (avg_icl_chunks={n_records_plot_icl})",
            "throughput_vs_records.png",
        )
        _plot_metric(
            lambda p: None if p is None else p["summarize"]["wall_time_s"] * 1000,
            "summarize wall time (ms)",
            f"Summarize time vs n_records (avg_icl_chunks={n_records_plot_icl})",
            "summarize_vs_records.png",
        )
        _plot_metric(
            lambda p: None if p is None else p["final_rss_bytes"] / 1024 / 1024,
            "final RSS (MB)",
            f"Final RSS vs n_records (avg_icl_chunks={n_records_plot_icl})",
            "rss_vs_records.png",
        )

        # Stacked-component plot: scalar arrays + ICL storage + new-only extras
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        for ax, side, marker in ((axes[0], "new", "o"), (axes[1], "old", "s")):
            scalars = [
                _scalar_arrays_bytes(pivot[k][side]) / 1024 / 1024
                if side in pivot[k]
                else None
                for k in n_keys
            ]
            icl = [
                _ragged_or_tdigest_bytes(pivot[k][side]) / 1024 / 1024
                if side in pivot[k]
                else None
                for k in n_keys
            ]
            extras = [
                _new_only_overhead_bytes(pivot[k][side]) / 1024 / 1024
                if side in pivot[k]
                else None
                for k in n_keys
            ]
            for ys, label in (
                (scalars, "scalar arrays"),
                (icl, "ICL storage"),
                (extras, "new-only extras (timestamps + metadata)"),
            ):
                if all(y is not None for y in ys) and any(y and y > 0 for y in ys):
                    ax.plot(ns, ys, marker=marker, label=label)
            ax.set_xlabel("n_records")
            ax.set_ylabel("bytes (MB)")
            ax.set_title(f"{side} pipeline (avg_icl_chunks={n_records_plot_icl})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "memory_components.png", dpi=110)
        plt.close(fig)

        # Metadata-field stacked-area plot (new pipeline only)
        # Sums per-field bytes across the n_records sweep so we can see which
        # metadata column grows the fastest as the workload scales.
        new_keys = [k for k in n_keys if "new" in pivot[k]]
        if new_keys:
            ns_new = [k[0] for k in new_keys]
            # Union of field names across all cases
            md_str_fields: set[str] = set()
            md_num_fields: set[str] = set()
            ts_fields: set[str] = set()
            for k in new_keys:
                parts = pivot[k]["new"]["breakdown"]["parts"]
                md_str_fields.update(parts.get("per_field_metadata_string", {}))
                md_num_fields.update(parts.get("per_field_metadata_numeric", {}))
                ts_fields.update(parts.get("per_field_timestamps", {}))

            def _series(group_key: str, field: str) -> list[float]:
                return [
                    pivot[k]["new"]["breakdown"]["parts"]
                    .get(group_key, {})
                    .get(field, 0)
                    / 1024
                    for k in new_keys
                ]

            series: list[tuple[str, list[float]]] = []
            for f in sorted(md_str_fields):
                series.append((f"string:{f}", _series("per_field_metadata_string", f)))
            for f in sorted(md_num_fields):
                series.append((f"num:{f}", _series("per_field_metadata_numeric", f)))
            for f in sorted(ts_fields):
                series.append((f"ts:{f}", _series("per_field_timestamps", f)))

            # Drop fields that are zero everywhere
            series = [(lbl, ys) for lbl, ys in series if any(ys)]
            if series:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [s[0] for s in series]
                ys_stack = [s[1] for s in series]
                ax.stackplot(ns_new, *ys_stack, labels=labels, alpha=0.85)
                ax.set_xlabel("n_records")
                ax.set_ylabel("metadata storage (KiB)")
                ax.set_title(
                    "New pipeline metadata fields (per-field, stacked) — old has none"
                )
                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(loc="upper left", fontsize="small", ncol=2)
                fig.tight_layout()
                fig.savefig(plots_dir / "metadata_per_field.png", dpi=110)
                plt.close(fig)

    # ICL sweep at n_records=100_000
    icl_keys = sorted(k for k in pivot if k[0] == 100_000)
    if icl_keys:
        chunks = [k[1] for k in icl_keys]
        fig, ax = plt.subplots(figsize=(8, 5))
        new_y = [
            pivot[k]["new"]["breakdown"]["parts"].get("column_store_ragged", 0) / 1024
            for k in icl_keys
            if "new" in pivot[k]
        ]
        old_y = [
            pivot[k]["old"]["breakdown"]["parts"].get("tdigest_total", 0) / 1024
            for k in icl_keys
            if "old" in pivot[k]
        ]
        if len(new_y) == len(chunks):
            ax.plot(chunks, new_y, marker="o", label="new: RaggedSeries (KiB)")
        if len(old_y) == len(chunks):
            ax.plot(chunks, old_y, marker="s", label="old: TDigest (KiB)")
        ax.set_xlabel("avg_icl_chunks per record")
        ax.set_ylabel("ICL storage (KiB)")
        ax.set_yscale("log")
        ax.set_title("ICL storage: ragged exact vs t-digest sketch (n_records=100k)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "icl_ragged_vs_tdigest.png", dpi=110)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--old-worktree",
        type=Path,
        default=WORKTREE_OLD_DEFAULT,
        help="Path to the worktree containing MetricResultsProcessor (old pipeline).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller grid for fast iteration.",
    )
    p.add_argument(
        "--records-file",
        type=Path,
        default=None,
        help="Replay a real profile_export.jsonl file instead of running the synthetic "
        "sweeps. Each --repeats value becomes one case.",
    )
    p.add_argument(
        "--repeats",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of replay multipliers to use with --records-file.",
    )
    p.add_argument(
        "--no-asizeof",
        action="store_true",
        help="Skip the pympler.asizeof breakdown in workers.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib PNG generation.",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=900,
        help="Per-worker subprocess timeout (default 900 s = 15 min).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=WORKTREE_NEW / "dev/benchmarks/results",
        help="Where the timestamped output directory is created.",
    )
    args = p.parse_args()

    if not (
        args.old_worktree / "src/aiperf/post_processors/metric_results_processor.py"
    ).exists():
        sys.exit(
            f"old worktree missing MetricResultsProcessor: {args.old_worktree}\n"
            "use --old-worktree to point at the correct path."
        )

    _ensure_worker_in_old(args.old_worktree)

    if args.records_file is not None:
        if not args.records_file.exists():
            sys.exit(f"--records-file does not exist: {args.records_file}")
        repeats = [int(x) for x in args.repeats.split(",") if x.strip()]
        unique_cases: list[Case] = _build_file_replay_sweep(args.records_file, repeats)
    else:
        cases = _build_n_records_sweep(args.quick) + _build_icl_sweep(args.quick)
        # de-dup (n_records=100k, avg_icl_chunks=100) appears in both sweeps
        seen: set[tuple[int, int, float | None]] = set()
        unique_cases = []
        for c in cases:
            key = (c.n_records, c.avg_icl_chunks, c.slice_duration)
            if key in seen:
                continue
            seen.add(key)
            unique_cases.append(c)

    print(
        f"Running {len(unique_cases)} cases x 2 pipelines = {len(unique_cases) * 2} workers",
        file=sys.stderr,
    )

    results = _run_all(
        unique_cases,
        args.old_worktree,
        timeout_s=args.timeout_s,
        skip_breakdown=args.no_asizeof,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out_root / f"metrics_pipeline_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "results.json").write_text(
        json.dumps(
            {
                "results": results,
                "args": vars(args)
                | {
                    "out_root": str(args.out_root),
                    "old_worktree": str(args.old_worktree),
                },
            },
            indent=2,
            default=str,
        )
    )

    md = _markdown_table(results)
    (outdir / "summary.md").write_text(md + "\n")
    print("\n=== Summary ===\n", file=sys.stderr)
    print(md)

    if not args.no_plots:
        _try_plots(results, outdir)

    print(f"\nResults: {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
