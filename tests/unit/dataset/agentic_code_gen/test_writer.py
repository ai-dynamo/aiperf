# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the writer module."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.agentic_code_gen.models import (
    SessionDistributionConfig,
    SessionEndReason,
    SynthesizedSession,
    SynthesizedTurn,
)
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import compute_quality_report, write_dataset
from aiperf.dataset.loader.models import MooncakeTrace


@pytest.fixture
def sessions(coding_config: SessionDistributionConfig):
    synth = SessionSynthesizer(coding_config, seed=42)
    return synth.synthesize_sessions(10)


@pytest.fixture
def turn_mode_sessions(turns_config: SessionDistributionConfig):
    synth = SessionSynthesizer(turns_config, seed=42)
    return synth.synthesize_sessions(10)


class TestWriteDataset:
    def test_creates_three_files(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        jsonl_path, manifest_path, quality_path = write_dataset(
            sessions, run_dir, coding_config, seed=42, config_name="default"
        )
        assert jsonl_path.exists()
        assert manifest_path.exists()
        assert quality_path.exists()
        assert jsonl_path.name == "dataset.jsonl"
        assert manifest_path.name == "manifest.json"
        assert quality_path.name == "quality.json"

    def test_jsonl_rows_are_mooncake_compatible(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = orjson.loads(line)
                MooncakeTrace(**data)

    def test_manifest_contains_seed_and_config_name(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        _, manifest_path, _ = write_dataset(
            sessions, run_dir, coding_config, seed=42, config_name="default"
        )
        manifest = orjson.loads(manifest_path.read_bytes())
        assert manifest["seed"] == 42
        assert manifest["config_name"] == "default"
        assert manifest["num_sessions"] == 10

    def test_first_turn_has_no_delay_field(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            first_line = orjson.loads(f.readline())
            assert "delay" not in first_line

    def test_session_ids_present_in_output(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        session_ids = set()
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                session_ids.add(data["session_id"])
        assert len(session_ids) == 10

    def test_jsonl_hash_ids_no_duplicates_within_turn(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        """Verify each row's hash_ids has no duplicates and IDs are non-negative."""
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"

        with jsonl.open("rb") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                data = orjson.loads(line)
                hids = data["hash_ids"]
                assert len(hids) == len(set(hids)), (
                    f"line {line_num}: duplicate hash_ids"
                )
                assert all(h >= 0 for h in hids), f"line {line_num}: negative hash_id"

    def test_jsonl_post_turn_hash_ids_are_dataset_unique(
        self, tmp_path: Path, coding_config: SessionDistributionConfig
    ) -> None:
        sessions = [
            SynthesizedSession(
                session_id="s0",
                group_id=0,
                end_reason=SessionEndReason.TARGET_TURN_COUNT,
                turns=[
                    SynthesizedTurn(
                        turn_index=0,
                        input_length=64,
                        output_length=1,
                        new_tokens=64,
                        delay_ms=0,
                        timestamp_ms=0,
                        hash_ids=[10],
                    ),
                    SynthesizedTurn(
                        turn_index=1,
                        input_length=129,
                        output_length=1,
                        new_tokens=64,
                        delay_ms=1,
                        timestamp_ms=1,
                        hash_ids=[10, 11],
                    ),
                ],
            ),
            SynthesizedSession(
                session_id="s1",
                group_id=0,
                end_reason=SessionEndReason.TARGET_TURN_COUNT,
                turns=[
                    SynthesizedTurn(
                        turn_index=0,
                        input_length=64,
                        output_length=1,
                        new_tokens=64,
                        delay_ms=0,
                        timestamp_ms=0,
                        hash_ids=[20],
                    ),
                    SynthesizedTurn(
                        turn_index=1,
                        input_length=129,
                        output_length=1,
                        new_tokens=64,
                        delay_ms=1,
                        timestamp_ms=1,
                        hash_ids=[20, 21],
                    ),
                ],
            ),
        ]
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        rows = [
            orjson.loads(line)
            for line in (run_dir / "dataset.jsonl").read_bytes().splitlines()
            if line.strip()
        ]
        post_turn_ids = [
            hash_id for row in rows if "delay" in row for hash_id in row["hash_ids"]
        ]
        assert len(post_turn_ids) == len(set(post_turn_ids))


class TestGroupIdInJsonl:
    def test_turn0_rows_have_group_id(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 rows must include group_id."""
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        turn0_count = 0
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "timestamp" in data:
                    assert "group_id" in data
                    turn0_count += 1
        assert turn0_count == 10

    def test_group_id_matches_session(
        self, tmp_path: Path, coding_config: SessionDistributionConfig
    ) -> None:
        """group_id in JSONL must match the synthesized session's group_id."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(10)
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"

        expected_group_ids = {s.session_id: s.group_id for s in sessions}
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "group_id" in data:
                    sid = data["session_id"]
                    assert data["group_id"] == expected_group_ids[sid]

    def test_non_turn0_rows_have_no_group_id(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        """Only turn 0 rows (with timestamp) should have group_id."""
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "delay" in data:
                    assert "group_id" not in data

    def test_group_ids_in_valid_range(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        """All group_id values must be in [0, num_groups)."""
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        num_groups = coding_config.cache.layer1_5_groups.num_groups
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "group_id" in data:
                    assert 0 <= data["group_id"] < num_groups


class TestQualityReport:
    def test_report_has_expected_metrics(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert "initial_context" in report.observed_vs_target
        assert "generation_length" in report.observed_vs_target
        assert "new_tokens_per_turn" in report.observed_vs_target
        assert "inter_turn_delay_ms" in report.observed_vs_target
        assert "turns_per_session" in report.observed_vs_target

    def test_report_observed_has_percentile_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        ic = report.observed_vs_target["initial_context"]
        assert ic.observed.count == 10
        assert ic.observed.mean > 0
        assert ic.observed.median > 0
        assert ic.observed.p05 <= ic.observed.p25 <= ic.observed.median
        assert ic.observed.median <= ic.observed.p75 <= ic.observed.p95

    def test_report_has_config_summary(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert report.config_summary["initial_context_mean"] == 62000
        assert report.config_summary["max_prompt_tokens"] == 200000

    def test_report_has_session_end_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        stats = report.session_end_stats
        assert stats.total_sessions == 10
        assert stats.forced_retires + stats.probabilistic_resets == 10
        assert stats.retire_fraction + stats.reset_fraction == pytest.approx(1.0)
        assert stats.final_context_utilization.count == 10

    def test_report_session_stats_is_percentile_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert report.session_stats.count == 10
        assert report.session_stats.mean > 0

    def test_report_pct_error_mean_is_non_negative(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        for metric in report.observed_vs_target.values():
            if metric.pct_error_mean is not None:
                assert metric.pct_error_mean >= 0

    def test_report_pct_error_median_is_non_negative(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        for metric in report.observed_vs_target.values():
            if metric.pct_error_median is not None:
                assert metric.pct_error_median >= 0

    def test_report_target_values_from_config(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        ic = report.observed_vs_target["initial_context"]
        cache = coding_config.cache
        assert (
            ic.target_mean
            == cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.mean
        )
        assert (
            ic.target_median
            == cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.median
        )
        delay = report.observed_vs_target["inter_turn_delay_ms"]
        assert delay.target_mean is None
        assert delay.target_median is None

    def test_config_summary_has_layer_fields(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        cs = report.config_summary
        assert cs["layer1_tokens"] == coding_config.cache.layer1_tokens
        assert cs["layer1_5_tokens"] == coding_config.cache.layer1_5_tokens
        assert cs["layer2_mean"] == coding_config.cache.layer2.mean
        assert cs["num_groups"] == coding_config.cache.layer1_5_groups.num_groups
        assert cs["zipf_alpha"] == coding_config.cache.layer1_5_groups.zipf_alpha

    def test_config_summary_initial_context_is_derived(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        """initial_context_mean in config_summary must equal L1 + L1.5 + L2.mean."""
        report = compute_quality_report(sessions, coding_config)
        cache = coding_config.cache
        expected = cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.mean
        assert report.config_summary["initial_context_mean"] == expected

    def test_turn_mode_report_tracks_target_turn_completion(
        self, turn_mode_sessions, turns_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(turn_mode_sessions, turns_config)
        stats = report.session_end_stats
        assert stats.target_turn_completions == 10
        assert stats.target_turn_fraction == pytest.approx(1.0)
        assert stats.forced_retires == 0
        assert stats.probabilistic_resets == 0

    def test_turn_mode_report_exposes_turn_targets(
        self, turn_mode_sessions, turns_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(turn_mode_sessions, turns_config)
        turns_metric = report.observed_vs_target["turns_per_session"]
        assert turns_metric.target_mean == float(turns_config.turns.mean)
        assert turns_metric.target_median == float(turns_config.turns.median)
        assert report.config_summary["turn_mode_enabled"] == 1
        assert report.config_summary["turns_mean"] == turns_config.turns.mean
        assert report.config_summary["turns_allow_truncation"] == 0
        assert report.config_summary["turns_max_session_attempts"] == 20
