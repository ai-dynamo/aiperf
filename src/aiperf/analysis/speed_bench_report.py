# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble per-category SPEED-Bench results into a matrix report.

Run ``aiperf profile`` once per SPEED-Bench category, then invoke
``aiperf speed-bench-report`` against the output directories to produce an
acceptance-length matrix matching the SPEED-Bench paper format.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Literal

import orjson

MetricType = Literal["accept_length", "accept_rate", "throughput"]
OutputFormat = Literal["csv", "table", "both"]


class SpeedBenchReportError(Exception):
    """Raised when the report cannot be assembled from the given paths."""


QUALITATIVE_CATEGORIES = [
    "coding",
    "humanities",
    "math",
    "multilingual",
    "qa",
    "rag",
    "reasoning",
    "roleplay",
    "stem",
    "summarization",
    "writing",
]

THROUGHPUT_TIERS = ["low_entropy", "mixed", "high_entropy"]

# Server metric names that represent acceptance length, in priority order.
# Different engines expose this under different names.
ACCEPT_LENGTH_METRICS = [
    "sglang:spec_accept_length",
    "vllm:spec_decode_mean_accepted_length",
    "trtllm:spec_accept_length",
]

ACCEPT_RATE_METRICS = [
    "sglang:spec_accept_rate",
    "vllm:spec_decode_draft_acceptance_rate",
    "trtllm:spec_accept_rate",
]

PROFILE_JSON = "profile_export_aiperf.json"
SERVER_METRICS_JSON = "server_metrics_export.json"


def find_run_dirs(paths: list[Path]) -> list[Path]:
    """Discover aiperf run directories from the given paths.

    Each path can be either a run directory (containing profile_export_aiperf.json)
    or a parent directory whose children are run directories.
    """
    run_dirs: list[Path] = []
    for p in paths:
        if not p.is_dir():
            print(f"Warning: {p} is not a directory, skipping", file=sys.stderr)
            continue
        if (p / PROFILE_JSON).exists():
            run_dirs.append(p)
        else:
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / PROFILE_JSON).exists():
                    run_dirs.append(child)
    return run_dirs


def load_profile(run_dir: Path) -> dict | None:
    """Load the profile JSON export."""
    path = run_dir / PROFILE_JSON
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except (OSError, orjson.JSONDecodeError) as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None


def load_server_metrics(run_dir: Path) -> dict | None:
    """Load the server metrics JSON export."""
    path = run_dir / SERVER_METRICS_JSON
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except (OSError, orjson.JSONDecodeError) as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None


def extract_category(profile: dict) -> str | None:
    """Extract the SPEED-Bench category from the input config."""
    try:
        dataset = profile["input_config"]["input"]["public_dataset"]
    except (KeyError, TypeError):
        return None
    if not isinstance(dataset, str) or not dataset.startswith("speed_bench_"):
        return None
    return dataset.removeprefix("speed_bench_")


def extract_model(profile: dict) -> str:
    """Extract model name from the input config."""
    try:
        names = profile["input_config"]["endpoint"]["model_names"]
        if names:
            return names[0]
    except (KeyError, TypeError):
        pass
    return "unknown"


def _get_metric_stat(metrics: dict, name: str, stat: str) -> float | None:
    """Get a stat value from a named metric's first series."""
    metric = metrics.get(name)
    if not metric:
        return None
    series = metric.get("series", [])
    if not series:
        return None
    return series[0].get("stats", {}).get(stat)


def extract_accept_length(server_metrics: dict) -> float | None:
    """Extract acceptance length from server metrics.

    Handles multiple engine types:
    - SGLang: directly exposes ``spec_accept_length`` gauge
    - vLLM: exposes counters for accepted tokens and drafts, compute ratio
    """
    metrics = server_metrics.get("metrics", {})

    # SGLang: direct gauge
    for name in ACCEPT_LENGTH_METRICS:
        val = _get_metric_stat(metrics, name, "avg")
        if val is not None:
            return val

    # vLLM: compute from counters (accepted_tokens / num_drafts)
    # Each draft step produces 1 verification token + accepted draft tokens,
    # so acceptance length = (accepted / drafts) + 1
    accepted = _get_metric_stat(
        metrics, "vllm:spec_decode_num_accepted_tokens", "total"
    )
    drafts = _get_metric_stat(metrics, "vllm:spec_decode_num_drafts", "total")
    if accepted is not None and drafts and drafts > 0:
        return (accepted / drafts) + 1.0

    # Fuzzy fallback for engines we don't know by name yet. Require all three
    # of "spec", "accept", "length" in the metric name so we don't pick up
    # unrelated metrics like "request_acceptance_total_length".
    for metric_name, metric_data in metrics.items():
        lower = metric_name.lower()
        if "spec" in lower and "accept" in lower and "length" in lower:
            series = metric_data.get("series", [])
            if series:
                val = series[0].get("stats", {}).get("avg")
                if val is not None:
                    return val

    return None


def extract_accept_rate(server_metrics: dict) -> float | None:
    """Extract acceptance rate from server metrics."""
    metrics = server_metrics.get("metrics", {})

    # SGLang: direct gauge
    for name in ACCEPT_RATE_METRICS:
        val = _get_metric_stat(metrics, name, "avg")
        if val is not None:
            return val

    # vLLM: compute from counters (accepted_tokens / draft_tokens)
    accepted = _get_metric_stat(
        metrics, "vllm:spec_decode_num_accepted_tokens", "total"
    )
    draft_tokens = _get_metric_stat(
        metrics, "vllm:spec_decode_num_draft_tokens", "total"
    )
    if accepted is not None and draft_tokens and draft_tokens > 0:
        return accepted / draft_tokens

    return None


def extract_throughput(profile: dict) -> float | None:
    """Extract output token throughput from profile metrics."""
    otp = profile.get("output_token_throughput")
    if otp and otp.get("avg") is not None:
        return otp["avg"]
    return None


def build_report(
    run_dirs: list[Path],
    metric_type: MetricType = "accept_length",
) -> dict[str, dict[str, float | None]]:
    """Build a {model: {category: value}} matrix from run directories.

    Returns:
        Nested dict mapping model name -> category -> metric value.
    """
    results: dict[str, dict[str, float | None]] = {}

    for run_dir in run_dirs:
        profile = load_profile(run_dir)
        if not profile:
            print(f"Warning: no {PROFILE_JSON} in {run_dir}, skipping", file=sys.stderr)
            continue

        category = extract_category(profile)
        if not category:
            print(
                f"Warning: cannot determine category from {run_dir}, skipping",
                file=sys.stderr,
            )
            continue

        model = extract_model(profile)
        if model not in results:
            results[model] = {}

        if metric_type in ("accept_length", "accept_rate"):
            server_metrics = load_server_metrics(run_dir)
            if server_metrics is not None:
                if metric_type == "accept_length":
                    value = extract_accept_length(server_metrics)
                else:
                    value = extract_accept_rate(server_metrics)
            else:
                value = None
                print(
                    f"Warning: no {SERVER_METRICS_JSON} in {run_dir}",
                    file=sys.stderr,
                )
        elif metric_type == "throughput":
            value = extract_throughput(profile)
        else:
            print(f"Unknown metric type: {metric_type}", file=sys.stderr)
            value = None

        results[model][category] = value

    return results


def detect_columns(results: dict[str, dict[str, float | None]]) -> list[str]:
    """Detect which column set to use based on the categories present."""
    all_cats: set[str] = set()
    for model_data in results.values():
        all_cats.update(model_data.keys())

    if all_cats <= set(QUALITATIVE_CATEGORIES):
        return [c for c in QUALITATIVE_CATEGORIES if c in all_cats]
    if all_cats <= set(THROUGHPUT_TIERS):
        return [c for c in THROUGHPUT_TIERS if c in all_cats]
    return sorted(all_cats)


def write_csv(
    results: dict[str, dict[str, float | None]],
    columns: list[str],
    output: Path,
) -> None:
    """Write the matrix as a CSV file."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *columns, "Overall"])
        for model, data in sorted(results.items()):
            row = [model]
            values = []
            for col in columns:
                v = data.get(col)
                row.append(f"{v:.2f}" if v is not None else "")
                if v is not None:
                    values.append(v)
            overall = mean(values) if values else None
            row.append(f"{overall:.2f}" if overall is not None else "")
            writer.writerow(row)
    print(f"CSV written to {output}")


def print_table(
    results: dict[str, dict[str, float | None]],
    columns: list[str],
    metric_type: MetricType,
) -> None:
    """Print a rich console table, falling back to plain text."""
    try:
        from rich.console import Console
        from rich.table import Table

        title_map = {
            "accept_length": "SPEED-Bench Acceptance Length Report",
            "accept_rate": "SPEED-Bench Acceptance Rate Report",
            "throughput": "SPEED-Bench Throughput Report (tokens/sec)",
        }
        table = Table(
            title=title_map.get(metric_type, "SPEED-Bench Report"),
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        for col in columns:
            table.add_column(col, justify="right", style="green")
        table.add_column("Overall", justify="right", style="bold green")

        for model, data in sorted(results.items()):
            row = [model]
            values = []
            for col in columns:
                v = data.get(col)
                row.append(f"{v:.2f}" if v is not None else "-")
                if v is not None:
                    values.append(v)
            overall = mean(values) if values else None
            row.append(f"{overall:.2f}" if overall is not None else "-")
            table.add_row(*row)

        Console().print(table)

    except ImportError:
        header = ["Model", *columns, "Overall"]
        widths = [max(len(h), 8) for h in header]
        widths[0] = max(widths[0], max((len(m) for m in results), default=8))

        print("  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True)))
        print("  ".join("-" * w for w in widths))
        for model, data in sorted(results.items()):
            values = []
            cells = [model]
            for col in columns:
                v = data.get(col)
                cells.append(f"{v:.2f}" if v is not None else "-")
                if v is not None:
                    values.append(v)
            overall = mean(values) if values else None
            cells.append(f"{overall:.2f}" if overall is not None else "-")
            print("  ".join(c.rjust(w) for c, w in zip(cells, widths, strict=True)))


def generate_report(
    paths: list[Path],
    output: Path = Path("speed_bench_report.csv"),
    output_format: OutputFormat = "both",
    metric: MetricType = "accept_length",
) -> None:
    """Discover run directories, build a SPEED-Bench matrix report, and emit it.

    Raises:
        SpeedBenchReportError: if no run directories are found under ``paths``,
            or if no SPEED-Bench results could be extracted from them.
    """
    run_dirs = find_run_dirs(paths)
    if not run_dirs:
        raise SpeedBenchReportError("no aiperf run directories found")

    print(f"Found {len(run_dirs)} run directories.")
    results = build_report(run_dirs, metric_type=metric)

    for model_data in results.values():
        for cat, v in model_data.items():
            if isinstance(v, float) and math.isnan(v):
                model_data[cat] = None

    has_value = any(
        v is not None for model_data in results.values() for v in model_data.values()
    )
    if not has_value:
        raise SpeedBenchReportError("no SPEED-Bench results extracted")

    columns = detect_columns(results)

    if output_format in ("table", "both"):
        print_table(results, columns, metric)

    if output_format in ("csv", "both"):
        write_csv(results, columns, output)
