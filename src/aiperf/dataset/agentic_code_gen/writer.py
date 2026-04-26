# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write synthesized sessions to Mooncake-compatible JSONL and metadata files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import orjson

from aiperf.dataset.agentic_code_gen.models import (
    DatasetManifest,
    SessionDistributionConfig,
    SynthesizedSession,
)
from aiperf.dataset.agentic_code_gen.reporting.metrics import (
    compute_quality_report,
)
from aiperf.dataset.agentic_code_gen.reporting.report import write_generated_reports
from aiperf.dataset.agentic_code_gen.schedule import expand_schedule


def write_dataset(
    sessions: list[SynthesizedSession],
    output_dir: Path,
    config: SessionDistributionConfig,
    seed: int,
    config_name: str | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    """Write JSONL dataset, manifest, quality report, cache explorer, and
    (optionally) a concurrency schedule into *output_dir*.

    Returns:
        Tuple of (jsonl_path, manifest_path, quality_path, schedule_path).
        schedule_path is None when config.concurrency_schedule is not set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality.json"

    # Write JSONL
    _write_jsonl(sessions, jsonl_path, config.block_size)

    # Write manifest
    manifest = DatasetManifest(
        seed=seed,
        num_sessions=len(sessions),
        config_name=config_name,
        generation_params=config,
    )
    manifest_path.write_bytes(
        orjson.dumps(manifest.model_dump(), option=orjson.OPT_INDENT_2)
    )

    # Write quality report
    report = compute_quality_report(sessions, config)
    quality_dict = report.model_dump()
    quality_path.write_bytes(orjson.dumps(quality_dict, option=orjson.OPT_INDENT_2))

    write_generated_reports(sessions, manifest, quality_dict, output_dir)

    schedule_path: Path | None = None
    if config.concurrency_schedule is not None:
        schedule_path = output_dir / "schedule.jsonl"
        # Dedicated RNG so schedule reproducibility doesn't perturb session
        # sampling or vice-versa.
        schedule_rng = np.random.default_rng(seed)
        ticks = expand_schedule(config.concurrency_schedule, schedule_rng)
        _write_schedule_jsonl(ticks, schedule_path)

    return jsonl_path, manifest_path, quality_path, schedule_path


def _write_schedule_jsonl(ticks: list[tuple[float, int]], path: Path) -> None:
    """Write the dense tick stream as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\n".join(orjson.dumps({"time_sec": t, "concurrency": c}) for t, c in ticks)
        + b"\n"
    )


def _write_jsonl(
    sessions: list[SynthesizedSession], path: Path, block_size: int
) -> None:
    """Write Mooncake-compatible JSONL rows with incremental values.

    Each row's input_length = new_tokens for that turn, and hash_ids contains
    exactly ceil(new_tokens / block_size) block IDs so that:

        input_length = (len(hash_ids) - 1) * block_size + final_block_size
        1 <= final_block_size <= block_size

    Turn 0 uses the synthesizer's prefix hash_ids (L1 + L1.5 + L2).
    Turns 1+ allocate fresh sequential IDs from a dataset-wide cursor,
    guaranteeing uniqueness across sessions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    next_hash_id = 0
    with path.open("wb") as f:
        for session in sessions:
            for turn in session.turns:
                n_blocks = math.ceil(turn.new_tokens / block_size)
                if turn.turn_index == 0:
                    hash_ids = turn.hash_ids
                    if hash_ids:
                        next_hash_id = max(next_hash_id, max(hash_ids) + 1)
                else:
                    hash_ids = list(range(next_hash_id, next_hash_id + n_blocks))
                    next_hash_id += n_blocks
                # Mooncake uses input_length for per-row incremental tokens;
                # SynthesizedTurn.input_length remains cumulative in memory.
                row: dict = {
                    "session_id": session.session_id,
                    "input_length": turn.new_tokens,
                    "output_length": turn.output_length,
                    "hash_ids": hash_ids,
                }
                if turn.turn_index == 0:
                    row["timestamp"] = round(turn.timestamp_ms, 1)
                    row["group_id"] = session.group_id
                    if session.is_restart_continuation:
                        row["is_restart"] = True
                else:
                    row["delay"] = round(turn.delay_ms, 1)
                f.write(orjson.dumps(row))
                f.write(b"\n")
