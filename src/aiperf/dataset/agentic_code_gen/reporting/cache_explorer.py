# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache structure extraction and D3 cache explorer rendering."""

from __future__ import annotations

import math
from pathlib import Path

import orjson

from aiperf.dataset.agentic_code_gen.models import CacheLayerConfig, DatasetManifest
from aiperf.dataset.agentic_code_gen.reporting.templates import (
    render_template,
    script_safe_json,
)
from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn


def _classify_turn_blocks(
    hash_ids: list[int],
    prev_hash_id_set: set[int] | None,
    l1_blocks: int,
    l15_blocks: int = 0,
    turn_index: int = 0,
) -> list[dict]:
    """Classify each block in a turn by layer and cache status."""
    del turn_index
    prefix_blocks = l1_blocks + l15_blocks
    blocks: list[dict] = []
    for pos, hid in enumerate(hash_ids):
        if pos < l1_blocks:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "L1", "status": "cached"}
            )
        elif pos < prefix_blocks:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "L1.5", "status": "cached"}
            )
        elif prev_hash_id_set is None:
            blocks.append({"pos": pos, "hash_id": hid, "layer": "L2", "status": "new"})
        elif hid in prev_hash_id_set:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "L2", "status": "cached"}
            )
        else:
            blocks.append({"pos": pos, "hash_id": hid, "layer": "L3", "status": "new"})
    return blocks


def write_cache_structure(
    sessions: dict[str, list[ParsedTurn]],
    manifest: DatasetManifest | None,
    output_dir: Path,
) -> dict:
    """Generate cache_structure.json with per-session block classification."""
    default_cache = CacheLayerConfig()
    l1_tokens = default_cache.layer1_tokens
    l15_tokens = default_cache.layer1_5_tokens
    block_size = 512
    if manifest:
        block_size = manifest.generation_params.block_size
        l1_tokens = manifest.generation_params.cache.layer1_tokens
        l15_tokens = manifest.generation_params.cache.layer1_5_tokens
    l1_blocks = math.ceil(l1_tokens / block_size) if block_size > 0 else 0
    l15_blocks_count = math.ceil(l15_tokens / block_size) if block_size > 0 else 0

    session_data: list[dict] = []
    for i, (sid, turns) in enumerate(sessions.items()):
        if i >= 50:
            break
        turn_data: list[dict] = []
        prev_hash_id_set: set[int] | None = None
        for turn_index, turn in enumerate(turns):
            classified = _classify_turn_blocks(
                turn.hash_ids,
                prev_hash_id_set,
                l1_blocks,
                l15_blocks_count,
                turn_index,
            )
            segments = _run_length_encode_segments(classified)
            turn_data.append(
                {
                    "turn_index": turn_index,
                    "input_length": turn.input_length,
                    "output_length": turn.output_length,
                    "num_blocks": len(turn.hash_ids),
                    "segments": segments,
                }
            )
            prev_hash_id_set = set(turn.hash_ids)

        session_data.append({"session_id": sid, "turns": turn_data})

    payload = {
        "block_size": block_size,
        "l1_blocks": l1_blocks,
        "l15_blocks": l15_blocks_count,
        "sessions": session_data,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cache_structure.json"
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return payload


def render_cache_explorer(output_dir: Path, cache_payload: dict) -> Path:
    """Write the standalone D3.js cache explorer HTML with inlined data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cache_explorer.html"
    html = render_template(
        "cache_explorer.html",
        INLINE_DATA=script_safe_json(cache_payload),
    )
    out_path.write_text(html)
    return out_path


def _run_length_encode_segments(classified: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for block in classified:
        key = (block["layer"], block["status"])
        if segments and (segments[-1]["layer"], segments[-1]["status"]) == key:
            segments[-1]["count"] += 1
            continue
        segments.append(
            {
                "start": block["pos"],
                "count": 1,
                "layer": block["layer"],
                "status": block["status"],
            }
        )
    return segments
