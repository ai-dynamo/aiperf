#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple example showing how to load and read profile_export.jsonl files
using native AIPerf Pydantic models in both sync and async mode.
"""

from pathlib import Path

import cyclopts

from aiperf.common.models.record_models import MetricRecordInfo
from aiperf.metrics.metric_registry import MetricRegistry

app = cyclopts.App(name=Path(__file__).name, help=__doc__)


def load_records(file_path: Path) -> list[MetricRecordInfo]:
    """Load profile_export.jsonl file into structured Pydantic models in sync mode."""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records


async def load_records_async(file_path: Path) -> list[MetricRecordInfo]:
    """Load profile_export.jsonl file into structured Pydantic models in async mode."""
    import aiofiles

    records = []
    async with aiofiles.open(file_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records


def print_record_info(file_path: Path, records: list[MetricRecordInfo]) -> None:
    """Print the records to the console."""
    from rich.console import Console

    console = Console()
    console.print(
        f"\n✓ Loaded [bold cyan]{len(records)}[/] records from [bold]{file_path.name}[/]\n"
    )

    record = next(record for record in records if not record.metadata.error)
    if record:
        console.rule("[bold]Example Metadata[/]")
        for key, value in record.metadata.model_dump().items():
            console.print(f"[bold cyan]{key}[/]: [green]{value}[/]")

        console.rule("[bold]Example Metrics[/]")
        for metric_tag, metric_value in record.metrics.items():
            metric_cls = MetricRegistry.get_class(metric_tag)
            console.print(
                f"[bold cyan]{metric_cls.header} ({metric_tag})[/]: [green]{metric_value.value} ({metric_value.unit})[/]"
            )

    error_record = next(record for record in records if record.metadata.error)
    if error_record:
        console.rule("[bold]Example Error[/]")
        console.print_json(data=error_record.metadata.error.model_dump(), indent=2)


@app.default
def main(
    file_path: Path = Path(__file__).parent
    / "artifacts"
    / "run1"
    / "profile_export.jsonl",
    _async: bool = False,
) -> None:
    if _async:
        import asyncio

        records = asyncio.run(load_records_async(file_path))
    else:
        records = load_records(file_path)

    print_record_info(file_path, records)


if __name__ == "__main__":
    app()
