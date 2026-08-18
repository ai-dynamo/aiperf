# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering and dry-run annotation for ``aiperf kube results list-runs``.

Split from ``results.py`` to keep that file under the 500-line ergonomics
ceiling. The two helpers here are the read-side of ``list-runs``:

- :func:`annotate_preview` mirrors the server's retention policy so users can
  see *which runs would be reaped* without actually reaping anything.
- :func:`print_runs_table` formats the ``RunHistoryListResponse`` payload as a
  Rich table for the default ``-o text`` output.
"""

from __future__ import annotations

from datetime import UTC


def annotate_preview(payload: dict, retention: dict) -> None:
    """Stamp each run with ``would_delete`` replicating ``enforce_retention`` dry-run.

    Mirrors the server-side policy: a run is marked for deletion only if it
    falls outside the count-keepers AND (when retain_days > 0) is older than
    the cutoff. ``latest_epoch`` is always protected. Also embeds the raw
    retention config on the payload so JSON consumers get both views.
    """
    import time

    runs = payload.get("runs", []) or []
    retain_runs = int(retention.get("retain_runs", 0))
    retain_days = int(retention.get("retain_days", 0))
    latest_epoch = payload.get("latest_epoch")

    sorted_runs = sorted(runs, key=lambda r: int(r.get("mtime_epoch", 0)), reverse=True)
    count_keepers = {r.get("epoch") for r in sorted_runs[:retain_runs]}
    age_cutoff = time.time() - retain_days * 86400 if retain_days > 0 else None

    for run in runs:
        if run.get("epoch") == latest_epoch:
            run["would_delete"] = False
            continue
        count_reap = run.get("epoch") not in count_keepers
        age_reap = age_cutoff is None or int(run.get("mtime_epoch", 0)) < age_cutoff
        run["would_delete"] = count_reap and age_reap

    payload["retention"] = {
        "retain_runs": retain_runs,
        "retain_days": retain_days,
    }


def print_runs_table(payload: dict, *, preview: bool = False) -> None:
    """Render a ``RunHistoryListResponse`` payload as a rich table.

    When ``preview=True``, includes a ``WOULD DELETE`` column driven by the
    per-run ``would_delete`` flag populated by :func:`annotate_preview`, plus a
    footer line summarizing the active retention policy.
    """
    from datetime import datetime

    from rich.table import Table

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.console import _human_size

    runs = payload.get("runs", [])
    namespace = payload.get("namespace", "")
    job_id = payload.get("job_id", "")

    if not runs:
        kube_console.print_info(f"No runs found for {namespace}/{job_id}")
        return

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("EPOCH", style="cyan")
    table.add_column("TIMESTAMP", style="dim")
    table.add_column("FILES", justify="right")
    table.add_column("SIZE", justify="right")
    table.add_column("LATEST", justify="center")
    if preview:
        table.add_column("WOULD DELETE", justify="center")

    for run in runs:
        ts = datetime.fromtimestamp(run.get("mtime_epoch", 0), tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        latest = "[green]✓[/green]" if run.get("is_latest") else ""
        row = [
            str(run.get("epoch", "")),
            ts,
            str(run.get("file_count", 0)),
            _human_size(int(run.get("total_size_bytes", 0))),
            latest,
        ]
        if preview:
            row.append("[red]✓[/red]" if run.get("would_delete") else "")
        table.add_row(*row)

    kube_console.console.print(table)
    kube_console.console.print(
        "Pass --run <epoch> to `aiperf kube results` to pin a historical download."
    )

    if preview:
        retention = payload.get("retention") or {}
        retain_runs = retention.get("retain_runs", 0)
        retain_days = retention.get("retain_days", 0)
        age_desc = f"{retain_days}" if retain_days else "0 (age policy disabled)"
        kube_console.print_info(
            f"Retention: RETAIN_RUNS={retain_runs}, RETAIN_DAYS={age_desc}"
        )
