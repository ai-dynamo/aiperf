# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for synthesizing mooncake traces."""

import json
from pathlib import Path

from cyclopts import App

from aiperf.dataset.synthesis import Synthesizer
from aiperf.dataset.synthesis.models import SynthesisParams

synthesize_app = App(
    name="synthesize-trace", help="Generate synthetic trace with prefix patterns"
)


@synthesize_app.default
def synthesize_trace(
    input_file: Path,
    output_file: Path,
    block_size: int = 512,
    speedup_ratio: float = 1.0,
    prefix_len_multiplier: float = 1.0,
    prefix_root_multiplier: int = 1,
    prompt_len_multiplier: float = 1.0,
    max_isl: int | None = None,
) -> None:
    """Generate synthetic trace with controlled prefix patterns.

    Args:
        input_file: Path to input mooncake trace JSONL file
        output_file: Path for synthesized output trace JSONL file
        block_size: KV cache block size (default: 512)
        speedup_ratio: Timestamp scaling multiplier (default: 1.0)
        prefix_len_multiplier: Core prefix length multiplier (default: 1.0)
        prefix_root_multiplier: Tree replication factor (default: 1)
        prompt_len_multiplier: Leaf prompt length multiplier (default: 1.0)
        max_isl: Maximum input sequence length filter (optional)
    """
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print("\nStarting trace synthesis...")
    print(f"Input file:                {input_file}")
    print(f"Output file:               {output_file}")
    print(f"Block size:                {block_size}")
    print(f"Speedup ratio:             {speedup_ratio:.2f}")
    print(f"Prefix length multiplier:  {prefix_len_multiplier:.2f}")
    print(f"Prefix root multiplier:    {prefix_root_multiplier}")
    print(f"Prompt length multiplier:  {prompt_len_multiplier:.2f}")
    if max_isl:
        print(f"Max ISL:                   {max_isl}")
    print()

    params = SynthesisParams(
        speedup_ratio=speedup_ratio,
        prefix_len_multiplier=prefix_len_multiplier,
        prefix_root_multiplier=prefix_root_multiplier,
        prompt_len_multiplier=prompt_len_multiplier,
        max_isl=max_isl,
        block_size=block_size,
    )

    synthesizer = Synthesizer(params=params)
    synthesized_traces = synthesizer.synthesize_from_file(input_file)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for trace in synthesized_traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Generated {len(synthesized_traces)} synthetic traces")
    print(f"Output saved to {output_file}")
    print()

    # Print stats
    stats = synthesizer.get_stats()
    print("Synthesis Statistics:")
    print(f"  Tree nodes: {stats['tree_nodes']}")
    print(f"  Tree depth: {stats['tree_depth']}")
    print()
