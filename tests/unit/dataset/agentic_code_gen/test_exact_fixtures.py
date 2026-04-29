# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-value end-to-end fixtures for tiny generated datasets."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.agentic_code_gen.models import (
    CacheLayerConfig,
    Layer15GroupConfig,
    LognormalParams,
    MixtureDelayConfig,
    NewTokensPerTurnConfig,
    SessionDistributionConfig,
    TurnCountConfig,
)
from aiperf.dataset.agentic_code_gen.reporting.report import generate_report
from aiperf.dataset.agentic_code_gen.reporting.simulation import load_sessions
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import write_dataset


def _fixed_lognormal(value: int) -> LognormalParams:
    return LognormalParams(mean=value, median=value, min=value, max=value)


def _fixed_config(
    turns: int,
    initial_tokens: int = 1,
    new_tokens: int = 1,
    output_tokens: int = 10,
    block_size: int = 8,
) -> SessionDistributionConfig:
    return SessionDistributionConfig(
        new_tokens_per_turn=NewTokensPerTurnConfig(
            mean=new_tokens,
            median=new_tokens,
            min=new_tokens,
            max=new_tokens,
        ),
        generation_length=_fixed_lognormal(output_tokens),
        inter_turn_delay=MixtureDelayConfig(
            agentic_fraction=1.0,
            agentic_delay=_fixed_lognormal(5),
            human_delay=_fixed_lognormal(5),
        ),
        reset=None,
        turns=TurnCountConfig(
            mean=turns,
            median=turns,
            min=turns,
            max=turns,
            max_session_attempts=1,
        ),
        max_prompt_tokens=1_000_000,
        block_size=block_size,
        cache=CacheLayerConfig(
            layer1_tokens=0,
            layer1_5_tokens=0,
            layer2=_fixed_lognormal(initial_tokens),
            layer1_5_groups=Layer15GroupConfig(num_groups=1, zipf_alpha=0.0),
        ),
    )


def _generate_trace(
    tmp_path: Path,
    config: SessionDistributionConfig,
) -> tuple[Path, list[dict]]:
    sessions = SessionSynthesizer(config, seed=7).synthesize_sessions(1)
    run_dir = tmp_path / "run"
    jsonl_path, _, _ = write_dataset(sessions, run_dir, config, seed=7)
    rows = [
        orjson.loads(line)
        for line in jsonl_path.read_bytes().splitlines()
        if line.strip()
    ]
    return run_dir, rows


class TestExactE2EFixtures:
    def test_one_turn_one_token_trace_is_exact(self, tmp_path: Path) -> None:
        config = _fixed_config(turns=1)
        run_dir, rows = _generate_trace(tmp_path, config)

        assert len(rows) == 1
        session_id = rows[0]["session_id"]
        assert rows == [
            {
                "session_id": session_id,
                "input_length": 1,
                "output_length": 10,
                "hash_ids": [200_000],
                "timestamp": 0.0,
                "group_id": 0,
            }
        ]

        sim_sessions = load_sessions(run_dir / "dataset.jsonl")
        assert sim_sessions[0]["turns"][0]["cumulative_input_length"] == 1

    def test_three_turn_fixed_trace_is_exact(self, tmp_path: Path) -> None:
        config = _fixed_config(turns=3)
        run_dir, rows = _generate_trace(tmp_path, config)

        assert len(rows) == 3
        assert [row["input_length"] for row in rows] == [1, 1, 1]
        assert [row["output_length"] for row in rows] == [10, 10, 10]
        assert [row["hash_ids"] for row in rows] == [[200_000], [200_001], [200_002]]
        assert "delay" not in rows[0]
        assert [row["delay"] for row in rows[1:]] == [5.0, 5.0]

        sim_turns = load_sessions(run_dir / "dataset.jsonl")[0]["turns"]
        assert [turn["cumulative_input_length"] for turn in sim_turns] == [1, 12, 23]

        report = generate_report(run_dir, fmt="text")
        comparisons = {c.metric_name: c for c in report.comparisons}
        assert comparisons["Initial Context (tokens)"].observed.mean == pytest.approx(
            1.0
        )
        assert comparisons["New Tokens/Turn"].observed.mean == pytest.approx(1.0)
        assert comparisons["Generation Length (tokens)"].observed.mean == pytest.approx(
            10.0
        )
