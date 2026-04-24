# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging utilities for Kubernetes CLI commands."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from collections.abc import Generator

    from aiperf.kubernetes.models import AIPerfJobInfo

# Logger for kube CLI output, configured with RichHandler for colored output
logger = AIPerfLogger("aiperf.kube")

if not logger.handlers:
    _handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
        log_time_format="%H:%M:%S",
        omit_repeated_times=False,
    )
    _handler.setLevel(logging.DEBUG)
    logger.addHandler(_handler)
    logger.set_level(logging.DEBUG)
    logger._logger.propagate = False

# Console kept only for structured output (tables)
console = Console()
stderr_console = Console(stderr=True)

# Separate logger for stderr output (deployment summary when piped)
_stderr_logger = AIPerfLogger("aiperf.kube.stderr")
if not _stderr_logger.handlers:
    _stderr_handler = RichHandler(
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
        console=stderr_console,
    )
    _stderr_handler.setLevel(logging.DEBUG)
    _stderr_logger.addHandler(_stderr_handler)
    _stderr_logger.set_level(logging.DEBUG)
    _stderr_logger._logger.propagate = False

# Path to store the last benchmark info
_LAST_BENCHMARK_FILE = Path.home() / ".aiperf" / "last_kube_benchmark.json"


@dataclass(slots=True)
class LastBenchmarkInfo:
    """Information about the last used benchmark."""

    job_id: str
    """The AIPerf job identifier."""

    namespace: str
    """The Kubernetes namespace where the job was deployed."""

    name: str | None = None
    """Optional human-readable name for the benchmark."""


def save_last_benchmark(job_id: str, namespace: str, name: str | None = None) -> None:
    """Save the last used benchmark info.

    Args:
        job_id: The job ID.
        namespace: The Kubernetes namespace.
        name: Optional human-readable name.
    """
    _LAST_BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {"job_id": job_id, "namespace": namespace}
    if name:
        data["name"] = name
    _LAST_BENCHMARK_FILE.write_bytes(orjson.dumps(data))


def get_last_benchmark() -> LastBenchmarkInfo | None:
    """Get the last used benchmark info.

    Returns:
        LastBenchmarkInfo if available, None otherwise.
    """
    if not _LAST_BENCHMARK_FILE.exists():
        return None
    try:
        data = orjson.loads(_LAST_BENCHMARK_FILE.read_bytes())
        return LastBenchmarkInfo(
            job_id=data["job_id"],
            namespace=data["namespace"],
            name=data.get("name"),
        )
    except (orjson.JSONDecodeError, KeyError, OSError):
        return None


def clear_last_benchmark() -> None:
    """Clear the last benchmark info (e.g., after deletion)."""
    if _LAST_BENCHMARK_FILE.exists():
        _LAST_BENCHMARK_FILE.unlink()


def print_step(message: str, step: int | None = None, total: int | None = None) -> None:
    """Log a step message with optional step counter.

    Args:
        message: The message to display.
        step: Current step number (optional).
        total: Total number of steps (optional).
    """
    if step is not None and total is not None:
        logger.info(f"[dim cyan]\\[{step}/{total}][/dim cyan] [cyan]{message}[/cyan]")
    else:
        logger.info(f"[cyan]{message}[/cyan]")


def print_success(message: str) -> None:
    """Log a success message with checkmark."""
    logger.info(f"[green]✓ {message}[/green]")


def print_info(message: str) -> None:
    """Log an info message."""
    logger.info(f"[dim]{message}[/dim]")


def print_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(f"[yellow]! {message}[/yellow]")


def print_error(message: str) -> None:
    """Log an error message."""
    logger.error(f"[red]✗ {message}[/red]")


def print_action(message: str) -> None:
    """Log an action/instruction message."""
    logger.info(f"[blue]→ {message}[/blue]")


def print_header(title: str, style: str = "bold cyan") -> None:
    """Log a section header.

    Args:
        title: The header text.
        style: Rich markup style for the header text.
    """
    logger.info("")
    logger.info(f"[{style}]{title}[/{style}]")
    logger.info(f"[dim]{'─' * len(title)}[/dim]")


@contextmanager
def status_log(message: str) -> Generator[None, None, None]:
    """Context manager that logs a status message before an operation.

    Args:
        message: The message to display.

    Yields:
        None
    """
    logger.info(f"[cyan]... {message}[/cyan]")
    yield


def _print_command_hints(
    job_id: str,
    namespace: str,
    *,
    header: str | None = None,
    header_style: str = "green bold",
    attach_label: str = "Monitor progress",
) -> None:
    """Log attach/results command hints.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
        header: Optional header message.
        header_style: Rich markup style for the header.
        attach_label: Label for the attach command.
    """
    if header:
        print_header(header, style=header_style)
    else:
        logger.info("")
    logger.info(f"  [dim]{attach_label}:[/dim]")
    logger.info(f"    aiperf kube watch {job_id} --namespace {namespace}")
    logger.info("")
    logger.info("  [dim]Retrieve results:[/dim]")
    logger.info(f"    aiperf kube results {job_id} --namespace {namespace}")


def print_detach_info(job_id: str, namespace: str, name: str | None = None) -> None:
    """Log detach information with helpful commands.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
        name: Human-readable benchmark name (optional).
    """
    label = f"Benchmark '{name}'" if name else "Benchmark"
    _print_command_hints(job_id, namespace, header=f"{label} deployed successfully!")


def print_interrupt_info(job_id: str, namespace: str) -> None:
    """Log info after keyboard interrupt with helpful commands.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
    """
    print_header("Detached from benchmark", style="yellow bold")
    logger.info("[dim]Benchmark continues running in Kubernetes[/dim]")
    _print_command_hints(
        job_id,
        namespace,
        attach_label="Reattach",
    )
    logger.info("")
    logger.info("  [dim]Cancel benchmark:[/dim]")
    logger.info(f"    aiperf kube delete {job_id} --namespace {namespace}")


def print_results_summary(output_dir: str) -> None:
    """Log results summary and display console export if available.

    Args:
        output_dir: Path to the output directory.
    """
    import sys
    from pathlib import Path

    print_header("Results", style="bold green")
    logger.info(f"  [dim]Saved to:[/dim]  {output_dir}/")

    output_path = Path(output_dir)
    ansi_file = output_path / "profile_export_console.ansi"
    txt_file = output_path / "profile_export_console.txt"

    if sys.stdout.isatty() and ansi_file.exists():
        logger.info("")
        sys.stdout.write(ansi_file.read_text(encoding="utf-8"))
        sys.stdout.flush()
    elif txt_file.exists():
        logger.info("")
        sys.stdout.write(txt_file.read_text(encoding="utf-8"))
        sys.stdout.flush()


def print_benchmark_complete() -> None:
    """Log benchmark completion message."""
    logger.info("")
    logger.info("[bold green]Benchmark complete![/bold green]")


def _human_size(size_bytes: int) -> str:
    """Format byte count as human-readable string (e.g. 2.5 MiB)."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size_bytes) < 1024 or unit == "GiB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GiB"


def print_file_table(files: list[tuple[str, int]], *, verb: str = "Downloaded") -> None:
    """Log a structured list of files with sizes.

    Args:
        files: List of (filename, size_bytes) tuples.
        verb: Action verb for the title (e.g., "Downloaded", "Copied").
    """
    print_header(f"{verb} {len(files)} file(s)")
    for name, size in sorted(files):
        print_info(f"{name}: {_human_size(size)}")


def print_metrics_summary(data: dict) -> None:
    """Log a summary of key metrics."""
    key_metrics = [
        "throughput",
        "latency_p50",
        "latency_p99",
        "latency_mean",
        "ttft_p50",
        "ttft_p99",
    ]
    found = [
        (
            m.get("tag", ""),
            f"{m.get('value', 0):.2f}",
            m.get("display_unit", m.get("unit", "")),
        )
        for m in data.get("metrics", [])
        if any(k in m.get("tag", "").lower() for k in key_metrics)
    ]

    if found:
        print_header("Benchmark Metrics")
        for tag, value, unit in found[:10]:
            logger.info(f"  [dim]{tag:<30}[/dim]  {value} {unit}")

    if "benchmark_id" in data:
        print_info(f"Benchmark ID: {data['benchmark_id']}")


def print_cr_submission_summary(
    name: str,
    namespace: str,
    image: str,
    *,
    endpoint_url: str | None = None,
    model_names: list[str] | None = None,
    connections_per_worker: int | None = None,
    to_stderr: bool = False,
) -> None:
    """Log a summary of an AIPerfJob CR submission.

    Args:
        name: The AIPerfJob CR name.
        namespace: Kubernetes namespace.
        image: Container image.
        endpoint_url: Target LLM endpoint URL (optional).
        model_names: Model names being benchmarked (optional).
        connections_per_worker: Connections per worker (optional).
        to_stderr: If True, log to stderr instead of stdout.
    """
    target = _stderr_logger if to_stderr else logger

    header = "AIPerf Kubernetes Benchmark"
    target.info("")
    target.info(f"[bold cyan]{header}[/bold cyan]")
    target.info(f"[dim]{'─' * len(header)}[/dim]")

    fields: list[tuple[str, str]] = [
        ("Name", name),
        ("Namespace", namespace),
    ]
    if endpoint_url:
        fields.append(("Endpoint", endpoint_url))
    if model_names:
        fields.append(("Model", ", ".join(model_names)))
    fields.append(("Image", image))
    if connections_per_worker is not None:
        fields.append(("Connections/Worker", str(connections_per_worker)))

    label_width = max(len(k) for k, _ in fields)
    for label, value in fields:
        target.info(f"  [dim]{label:<{label_width}}[/dim]  {value}")


def print_aiperfjob_table(
    jobs: list[AIPerfJobInfo],
    wide: bool = False,
) -> None:
    """Print a formatted table of AIPerfJob CRs.

    Args:
        jobs: List of AIPerfJobInfo objects.
        wide: Show additional columns (model, endpoint, error).
    """
    from aiperf.kubernetes.cli_helpers import format_age
    from aiperf.operator.status import Phase

    _phase_styles: dict[str, str] = {
        Phase.RUNNING: "bold green",
        Phase.COMPLETED: "green",
        Phase.FAILED: "red",
        Phase.CANCELLED: "yellow",
        Phase.PENDING: "dim",
        Phase.QUEUED: "dim cyan",
        Phase.INITIALIZING: "cyan",
    }

    table = Table(show_header=True, header_style="bold", box=None)

    table.add_column("NAME", style="cyan")
    table.add_column("NAMESPACE", style="dim")
    table.add_column("PHASE")
    table.add_column("WORKERS", justify="right")
    table.add_column("PROGRESS", justify="right")
    table.add_column("THROUGHPUT", justify="right")
    table.add_column("LATENCY", justify="right")
    table.add_column("AGE", style="dim")
    if wide:
        table.add_column("MODEL", style="dim")
        table.add_column("ENDPOINT", style="dim")
        table.add_column("ERROR", style="red")

    for job in jobs:
        phase_style = _phase_styles.get(job.phase, "")
        phase_text = Text(job.phase, style=phase_style)
        workers_text = Text(
            job.workers_str, style="dim" if job.workers_ready == 0 else ""
        )

        if job.progress_percent is not None:
            progress_text = Text(f"{job.progress_percent:.0f}%", style="bold")
        else:
            progress_text = Text("-", style="dim")

        if job.throughput_rps is not None:
            throughput_text = Text(f"{job.throughput_rps:.1f} rps")
        else:
            throughput_text = Text("-", style="dim")

        if job.latency_p99_ms is not None:
            latency_text = Text(f"{job.latency_p99_ms:.1f} ms")
        else:
            latency_text = Text("-", style="dim")

        age = format_age(job.created)

        row: list[str | Text] = [
            job.name,
            job.namespace,
            phase_text,
            workers_text,
            progress_text,
            throughput_text,
            latency_text,
            age,
        ]
        if wide:
            row.append(job.model or "")
            row.append(job.endpoint or "")
            row.append(job.error or "")
        table.add_row(*row)

    console.print(table)
