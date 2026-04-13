# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for anonymizing conversation traces."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="anonymize-trace")


@app.default
def anonymize_trace(
    input_file: Path,
    model: str,
    output_file: Path | None = None,
    block_size: int = 512,
) -> None:
    """Anonymize raw chat logs into privacy-preserving Mooncake traces.

    Converts OpenAI-compatible conversation logs into traces with block-hashed
    prefix patterns. Strips all text content, preserving only token counts and
    hash ID sequences for prefix-cache-aware benchmarking.

    The --model argument specifies the TARGET model you intend to benchmark
    against, not the model that generated the original logs. The target model's
    tokenizer and chat template are used to produce accurate token counts and
    prefix patterns.

    Args:
        input_file: Path to input JSONL with raw conversation logs.
        model: HuggingFace model name for tokenizer and chat template (target model).
        output_file: Path to output Mooncake trace JSONL. Defaults to <input>_anonymized.jsonl.
        block_size: Tokens per block for hashing (default: 512).
    """
    from rich.console import Console

    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.synthesis.anonymize import anonymize_trace as _anonymize

    console = Console(width=120)

    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise SystemExit(1)

    if output_file is None:
        output_file = input_file.with_name(f"{input_file.stem}_anonymized.jsonl")

    console.print(f"Loading tokenizer: {model}")
    tokenizer = Tokenizer.from_pretrained(model)

    console.print(f"Processing: {input_file}")
    result = _anonymize(
        input_file=input_file,
        output_file=output_file,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    console.print()
    console.print("[bold]Anonymization Summary[/bold]")
    console.print(f"  Requests processed: {result.total_processed:,}")
    if result.total_skipped > 0:
        console.print(f"  Requests skipped:   {result.total_skipped:,}")
    if result.sessions_detected > 0:
        console.print(f"  Sessions detected:  {result.sessions_detected:,}")
    console.print(f"  Unique hash IDs:    {result.unique_hash_ids:,}")
    console.print(f"  Output file:        {result.output_file}")

    if result.no_timestamps_warning:
        console.print()
        console.print(
            "[yellow]Warning: No timestamps found in input. "
            "The output trace will not support --fixed-schedule replay. "
            "Consider adding timestamps or using --request-rate during replay.[/yellow]"
        )
