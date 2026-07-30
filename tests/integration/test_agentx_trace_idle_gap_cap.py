# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end coverage for the AgentX per-trace idle-gap cap."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer, AIPerfResults


def _write_trace(path: Path, trace_id: str, starts: list[float]) -> None:
    """Write one root-only Weka trace with the requested start schedule."""
    block_size = 16
    requests = [
        {
            "t": start,
            "type": "n",
            "model": "mock-model",
            "in": (turn_index + 1) * block_size + 8,
            "out": 2,
            "hash_ids": list(range(1, turn_index + 2)),
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 0.05,
            "think_time": max(0.0, start - starts[turn_index - 1] - 0.05)
            if turn_index
            else 0.0,
        }
        for turn_index, start in enumerate(starts)
    ]
    trace = {
        "id": trace_id,
        "models": ["mock-model"],
        "block_size": block_size,
        "hash_id_scope": "local",
        "requests": requests,
    }
    path.write_text(json.dumps(trace), encoding="utf-8")


@pytest.fixture
def varied_weka_traces(tmp_path: Path) -> Path:
    """Create several traces with different mixtures of short and long gaps."""
    trace_dir = tmp_path / "weka"
    trace_dir.mkdir()
    schedules = {
        "alternating_2s": [0.0, 0.2, 0.4, 2.4, 2.6, 4.6, 4.8, 6.8],
        "alternating_1_5s": [0.0, 0.15, 1.65, 1.85, 3.35, 3.55, 5.05, 5.25],
        "alternating_2_5s": [0.0, 0.3, 0.6, 3.1, 3.4, 5.9, 6.2, 8.7],
        "mixed_1s_3s": [0.0, 0.1, 0.2, 1.2, 1.3, 4.3, 4.4, 7.4],
    }
    for trace_id, starts in schedules.items():
        _write_trace(trace_dir / f"{trace_id}.json", trace_id, starts)
    return trace_dir


def _tree_idle_gaps_seconds(result: AIPerfResults) -> list[float]:
    """Return periods with no request active in each profiling trajectory tree."""
    assert result.jsonl is not None
    by_root = defaultdict(list)
    for record in result.jsonl:
        root_id = record.metadata.root_correlation_id
        assert root_id is not None
        by_root[root_id].append(record.metadata)

    gaps: list[float] = []
    for records in by_root.values():
        records.sort(key=lambda record: record.request_start_ns)
        latest_end_ns = records[0].request_end_ns
        for record in records[1:]:
            if record.request_start_ns > latest_end_ns:
                gaps.append((record.request_start_ns - latest_end_ns) / 1e9)
            latest_end_ns = max(latest_end_ns, record.request_end_ns)
    return gaps


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cap_seconds", [None, 0.1, 0.25, 0.5])
async def test_agentx_trace_idle_gap_cap_controls_replay_timing(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    varied_weka_traces: Path,
    cap_seconds: float | None,
) -> None:
    """Replay varied traces through the mock server and verify the configured cap."""
    cap_arg = (
        "" if cap_seconds is None else f"--trace-idle-gap-cap-seconds {cap_seconds}"
    )
    result = await cli.run(
        f"""
        aiperf profile
            --scenario inferencex-agentx-mvp
            --unsafe-override
            --model mock-model
            --url {aiperf_mock_server.url}
            --endpoint-type chat
            --streaming
            --extra-inputs ignore_eos:true
            --custom-dataset-type weka_trace
            --input-file {varied_weka_traces}
            --no-fixed-schedule
            --concurrency 4
            --benchmark-duration 6
            --trajectory-start-min-ratio 0.25
            --trajectory-start-max-ratio 0.25
            --random-seed 42
            --workers-max 4
            --ui simple
            {cap_arg}
        """,
        timeout=300.0,
    )

    assert result.exit_code == 0, (
        f"AIPerf failed; stderr=\n{result.stderr}\n\nstdout=\n{result.stdout}"
    )
    assert result.jsonl is not None
    source_traces = {
        record.metadata.source_trace_id
        for record in result.jsonl
        if record.metadata.source_trace_id is not None
    }
    assert source_traces == {
        "alternating_2s",
        "alternating_1_5s",
        "alternating_2_5s",
        "mixed_1s_3s",
    }

    gaps = _tree_idle_gaps_seconds(result)
    assert len(gaps) >= 8
    if cap_seconds is None:
        assert max(gaps) > 1.0
    else:
        assert max(gaps) <= cap_seconds + 0.35
