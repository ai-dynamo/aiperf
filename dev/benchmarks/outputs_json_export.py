# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the real outputs.json exporter in a fresh process per fixture."""

import argparse
import asyncio
import math
import resource
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import orjson

from aiperf.common.constants import IS_MACOS
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.outputs_json_exporter import OutputsJsonExporter


def generate(root: Path, records: int, response_bytes: int, shards: int) -> None:
    """Write deterministic, unsorted, interleaved fragment files."""
    fragments = root / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
    fragments.mkdir(parents=True, exist_ok=True)
    handles = [
        (fragments / f"output_fragments_{i:03}.jsonl").open("wb") for i in range(shards)
    ]
    stride = 7919
    while math.gcd(stride, records) != 1:
        stride += 1
    try:
        for i in range(records):
            index = (i * stride) % records
            row = {
                "session_num": index // 3,
                "turn_index": index % 3,
                "conversation_id": f"conversation-{index // 3}",
                "x_request_id": f"request-{index}",
                "benchmark_phase": "warmup" if index % 13 == 0 else "profiling",
                "request_start_ns": index * 1000,
                "request_end_ns": index * 1000 + 200,
                "metrics": {"output_token_count": 20, "request_latency": 0.2},
                "response_text": "x" * response_bytes,
            }
            handles[i % shards].write(orjson.dumps(row) + b"\n")
    finally:
        for handle in handles:
            handle.close()


async def measure(fixture: Path, work: Path) -> dict[str, int | float | str]:
    """Copy inputs before timing and report process peak RSS."""
    shutil.copytree(fixture, work)
    cfg = MagicMock()
    cfg.cfg.artifacts.export_outputs_json = True
    cfg.cfg.artifacts.outputs_json_file = work / "outputs.json"
    cfg.cfg.artifacts.artifact_directory = work
    exporter = OutputsJsonExporter(cfg)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    await exporter.export()
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, KiB on Linux.
    scale = 1 if IS_MACOS else 1024
    return {
        "fixture": fixture.name,
        "seconds": elapsed,
        "pre_export_peak_rss_bytes": before * scale,
        "peak_rss_bytes": after * scale,
        "input_bytes": sum(file.stat().st_size for file in fixture.rglob("*.jsonl")),
        "output_bytes": (work / "outputs.json").stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generator = sub.add_parser("generate")
    generator.add_argument("root", type=Path)
    generator.add_argument("--records", type=int, required=True)
    generator.add_argument("--response-bytes", type=int, required=True)
    generator.add_argument("--shards", type=int, default=8)
    runner = sub.add_parser("run")
    runner.add_argument("fixture", type=Path)
    runner.add_argument("work", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        if args.records <= 0 or args.shards <= 0 or args.response_bytes < 0:
            parser.error(
                "records and shards must be positive; response bytes must be nonnegative"
            )
        generate(args.root, args.records, args.response_bytes, args.shards)
    else:
        print(orjson.dumps(asyncio.run(measure(args.fixture, args.work))).decode())


if __name__ == "__main__":
    main()
