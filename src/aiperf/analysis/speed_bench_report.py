# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble per-category SPEED-Bench results into a matrix report.

Point ``aiperf speed-bench-report`` at one or more ``aiperf profile`` output
directories to produce an acceptance matrix in the SPEED-Bench paper format.

Acceptance numbers come from one of three sources, tried in this order
(``--source`` pins one explicitly):

1. **Per-request records** (``profile_export.jsonl``) -- the engine-neutral
   ``spec_decode_acceptance`` struct on each request, written at the default
   ``--export-level records``. Highest fidelity, and the only source that can
   split a *single* run into several columns: SPEED-Bench runs stamp each
   record with its row's category via ``metadata.source_kind``, so one run over
   an aggregate split (``speed_bench_qualitative``) yields the full matrix.
2. **Summarized per-request metrics** (``profile_export_aiperf.json``) -- the
   same per-request data, already reduced by AIPerf to run-level scalars. Same
   provenance as ``records``, coarser granularity. One column per run; the only
   source available when the run was exported at ``--export-level summary``.
3. **Server scrape** (``server_metrics_export.json``) -- Prometheus counters
   sampled from the server during the run. One column per run, whole-server
   scope. The portable path: the only one that works for engines with no
   per-request acceptance reporting.

Every source computes the same token-weighted quantities, so columns stay
comparable across them: acceptance length is ``1 + accepted / steps`` and
acceptance rate is ``accepted / drafted``, both summed over the run rather than
averaged per request.
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import orjson

MetricType = Literal["accept_length", "accept_rate", "throughput"]
MetricSource = Literal["auto", "records", "summary", "server"]
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

# spec_al_* acceptance-length benchmarks, in a curated order so the report
# columns read math -> chat -> code rather than alphabetically.
SPEC_AL_BENCHMARKS = ["gsm8k", "math500", "mtbench", "humaneval", "mbpp"]

# Dataset-selector prefixes that mark an acceptance-length benchmark run. The
# category is the selector value with the prefix stripped (e.g.
# "speed_bench_coding" -> "coding", "spec_al_gsm8k" -> "gsm8k").
CATEGORY_PREFIXES = ("speed_bench_", "spec_al_")

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
PROFILE_JSONL = "profile_export.jsonl"
SERVER_METRICS_JSON = "server_metrics_export.json"

# Run-level metric tags in the summary export that already carry the
# token-weighted per-request acceptance, used when the run was exported at
# --export-level summary (no per-request JSONL to read).
SUMMARY_ACCEPT_METRICS: dict[MetricType, str] = {
    "accept_length": "spec_decode_token_weighted_acceptance_length",
    "accept_rate": "spec_decode_overall_draft_acceptance_rate",
}


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
    """Extract the acceptance-length benchmark category from the input config.

    The exporter writes ``input_config`` as a dump of the v2 ``BenchmarkConfig``.
    Custom/file datasets (e.g. SPEED-Bench) serialize their selector under
    ``datasets[].format``; public datasets (e.g. the spec_al_* HuggingFace
    benchmarks) serialize it under ``datasets[].dataset``. Returns the suffix of
    the first entry whose selector starts with a recognized prefix
    (see ``CATEGORY_PREFIXES``).
    """
    try:
        datasets = profile["input_config"]["datasets"]
    except (KeyError, TypeError):
        return None
    if not isinstance(datasets, list):
        return None
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("format") or entry.get("dataset")
        if not isinstance(name, str):
            continue
        for prefix in CATEGORY_PREFIXES:
            if name.startswith(prefix):
                return name.removeprefix(prefix)
    return None


def extract_model(profile: dict) -> str:
    """Extract model name from the input config.

    Reads ``input_config.models.items[0].name`` from the v2 ``BenchmarkConfig``
    dump. Falls back to ``"unknown"`` when absent or malformed.
    """
    try:
        items = profile["input_config"]["models"]["items"]
    except (KeyError, TypeError):
        return "unknown"
    if not isinstance(items, list):
        return "unknown"
    for entry in items:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                return name
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


def extract_summary_acceptance(profile: dict, metric_type: MetricType) -> float | None:
    """Extract run-level per-request acceptance from the summary export.

    Reads the token-weighted scalars AIPerf derives from the per-request
    records, so the value matches what the records path computes for the same
    run. Acceptance rate is stored as a percentage and returned as a 0..1
    fraction to match the server-scrape path.
    """
    tag = SUMMARY_ACCEPT_METRICS.get(metric_type)
    if tag is None:
        return None
    entry = profile.get(tag)
    if not isinstance(entry, dict):
        return None
    value = entry.get("avg")
    if value is None:
        return None
    return value / 100.0 if metric_type == "accept_rate" else value


@dataclass
class AcceptanceTotals:
    """Summed acceptance counters for one category within one run."""

    accepted: int = 0
    """Accepted draft tokens, excluding the always-accepted bonus token."""

    drafted: int = 0
    """Proposed draft tokens counted toward acceptance."""

    steps: int = 0
    """Speculative verification steps."""

    def add(self, spec: dict) -> None:
        """Fold one request's acceptance record into the totals.

        Reads all three counters before mutating, so a record missing the last
        one cannot leave the totals half-updated and skew the category.
        """
        accepted = spec["num_accepted_draft_tokens"]
        drafted = spec["num_draft_tokens"]
        steps = spec["num_spec_steps"]
        self.accepted += accepted
        self.drafted += drafted
        self.steps += steps

    def value(self, metric_type: MetricType) -> float | None:
        """Token-weighted acceptance length or rate; None when undefined."""
        if metric_type == "accept_length":
            return 1.0 + self.accepted / self.steps if self.steps else None
        if metric_type == "accept_rate":
            return self.accepted / self.drafted if self.drafted else None
        return None


def iter_records(run_dir: Path) -> Iterator[dict]:
    """Yield each parsed line of the per-request records export.

    Silently skips unparseable lines: a truncated final line is normal for a
    run that was interrupted, and one bad record should not sink the report.
    """
    path = run_dir / PROFILE_JSONL
    if not path.exists():
        return
    try:
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)


def acceptance_from_records(
    run_dir: Path,
    metric_type: MetricType,
    fallback_category: str | None,
) -> dict[str, float]:
    """Build a {category: value} mapping from one run's per-request records.

    Records are grouped by ``metadata.source_kind``, which SPEED-Bench runs set
    to the row's category -- so a single run over an aggregate split produces
    one entry per category. Runs whose loader sets no ``source_kind`` (the
    ``spec_al_*`` public datasets, for instance) fall back to the run's own
    dataset-derived category and produce a single entry.

    Warmup records are excluded: they are written to the same JSONL but are
    excluded from every other reported number.
    """
    totals: dict[str, AcceptanceTotals] = {}
    for record in iter_records(run_dir):
        spec = record.get("spec_decode_acceptance")
        if not isinstance(spec, dict):
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if "warmup" in (metadata.get("benchmark_phase"), metadata.get("phase_kind")):
            continue
        category = metadata.get("source_kind") or fallback_category
        if not isinstance(category, str) or not category:
            continue
        try:
            totals.setdefault(category, AcceptanceTotals()).add(spec)
        except (KeyError, TypeError):
            continue

    return {
        category: value
        for category, total in totals.items()
        if (value := total.value(metric_type)) is not None
    }


def build_report(
    run_dirs: list[Path],
    metric_type: MetricType = "accept_length",
    source: MetricSource = "auto",
) -> dict[str, dict[str, float | None]]:
    """Build a {model: {category: value}} matrix from run directories.

    A run directory usually contributes one column, but a SPEED-Bench run read
    from per-request records contributes one column per category present in it.

    Returns:
        Nested dict mapping model name -> category -> metric value.
    """
    results: dict[str, dict[str, float | None]] = {}

    for run_dir in run_dirs:
        profile = load_profile(run_dir)
        if not profile:
            print(f"Warning: no {PROFILE_JSON} in {run_dir}, skipping", file=sys.stderr)
            continue

        run_category = extract_category(profile)

        values: dict[str, float | None] = {}
        if metric_type == "throughput":
            # Throughput is a rate over the whole run, so it cannot be split
            # per category inside a mixed run -- every category shares the same
            # window and the same server. One column per run, always.
            if run_category:
                values = {run_category: extract_throughput(profile)}
        elif metric_type in ("accept_length", "accept_rate"):
            values = _acceptance_values(
                run_dir,
                profile,
                metric_type=metric_type,
                source=source,
                run_category=run_category,
            )
        else:
            print(f"Unknown metric type: {metric_type}", file=sys.stderr)
            if run_category:
                values = {run_category: None}

        if not values:
            print(
                f"Warning: cannot determine category from {run_dir}, skipping",
                file=sys.stderr,
            )
            continue

        model_data = results.setdefault(extract_model(profile), {})

        model_data.update(values)

    return results


def _acceptance_values(
    run_dir: Path,
    profile: dict,
    *,
    metric_type: MetricType,
    source: MetricSource,
    run_category: str | None,
) -> dict[str, float | None]:
    """Resolve one run's acceptance columns from the highest-fidelity source.

    ``auto`` walks per-request records, then the run-level per-request metrics,
    then the server scrape, taking the first that yields anything. A run whose
    category is known but has no acceptance data anywhere still contributes an
    empty cell, so the matrix shows which runs are missing numbers instead of
    dropping them.
    """
    values: dict[str, float | None] = {}
    if source in ("auto", "records"):
        values = dict(acceptance_from_records(run_dir, metric_type, run_category))
    if not values and source in ("auto", "summary"):
        summarized = extract_summary_acceptance(profile, metric_type)
        if summarized is not None and run_category:
            values = {run_category: summarized}
    if not values and source in ("auto", "server"):
        values = _acceptance_from_server(run_dir, metric_type, run_category)
    if not values and run_category:
        values = {run_category: None}
    return values


def _acceptance_from_server(
    run_dir: Path,
    metric_type: MetricType,
    run_category: str | None,
) -> dict[str, float | None]:
    """Read whole-run acceptance from the scraped server metrics export."""
    if not run_category:
        return {}
    server_metrics = load_server_metrics(run_dir)
    if server_metrics is None:
        print(f"Warning: no {SERVER_METRICS_JSON} in {run_dir}", file=sys.stderr)
        return {}
    value = (
        extract_accept_length(server_metrics)
        if metric_type == "accept_length"
        else extract_accept_rate(server_metrics)
    )
    return {run_category: value} if value is not None else {}


def detect_columns(results: dict[str, dict[str, float | None]]) -> list[str]:
    """Detect which column set to use based on the categories present."""
    all_cats: set[str] = set()
    for model_data in results.values():
        all_cats.update(model_data.keys())

    if all_cats <= set(QUALITATIVE_CATEGORIES):
        return [c for c in QUALITATIVE_CATEGORIES if c in all_cats]
    if all_cats <= set(THROUGHPUT_TIERS):
        return [c for c in THROUGHPUT_TIERS if c in all_cats]
    if all_cats <= set(SPEC_AL_BENCHMARKS):
        return [c for c in SPEC_AL_BENCHMARKS if c in all_cats]
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
            "accept_length": "Acceptance Length Report",
            "accept_rate": "Acceptance Rate Report",
            "throughput": "Throughput Report (tokens/sec)",
        }
        table = Table(
            title=title_map.get(metric_type, "Speculative Decoding Report"),
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
    *,
    output: Path = Path("speed_bench_report.csv"),
    output_format: OutputFormat = "both",
    metric: MetricType = "accept_length",
    source: MetricSource = "auto",
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
    results = build_report(run_dirs, metric_type=metric, source=source)

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
