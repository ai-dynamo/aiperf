# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest

from aiperf.dataset.agentic_code_gen.models import (
    DatasetManifest,
    ResetConfig,
    SessionDistributionConfig,
)
from aiperf.dataset.agentic_code_gen.reporting.cache_explorer import (
    _classify_turn_blocks,
)
from aiperf.dataset.agentic_code_gen.reporting.report import (
    ParsedTurn,
    build_report_data,
    extract_cache_metrics,
    extract_metrics,
    generate_report,
    group_sessions,
    load_jsonl,
    render_cache_explorer,
    render_text_report,
    write_cache_structure,
)
from aiperf.dataset.agentic_code_gen.reporting.trace import (
    synthesized_sessions_to_parsed,
    validate_mooncake_trace,
)
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import write_dataset


@pytest.fixture
def run_dir(tmp_path: Path, coding_config: SessionDistributionConfig) -> Path:
    """Create a run directory with a small dataset."""
    synth = SessionSynthesizer(coding_config, seed=42)
    sessions = synth.synthesize_sessions(20)
    d = tmp_path / "run"
    write_dataset(sessions, d, coding_config, seed=42, config_name="default")
    return d


class TestLoadJsonl:
    def test_loads_all_turns(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        assert len(turns) > 20  # 20 sessions, each with multiple turns

    def test_parsed_turn_fields(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        t = turns[0]
        assert isinstance(t.session_id, str)
        assert t.input_length > 0
        assert t.output_length > 0
        assert isinstance(t.hash_ids, list)


class TestGroupSessions:
    def test_groups_by_session_id(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        assert len(sessions) == 20

    def test_preserves_turn_order(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        for _sid, session_turns in sessions.items():
            # First turn should have delay_ms == 0
            assert session_turns[0].delay_ms == 0.0


class TestTraceValidation:
    def test_validate_mooncake_trace_reports_json_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not-json}\n")

        line_count, errors = validate_mooncake_trace(path)

        assert line_count == 1
        assert errors
        assert "Line 1" in errors[0]

    def test_synthesized_sessions_to_parsed_preserves_restart_metadata(self) -> None:
        config = SessionDistributionConfig(
            reset=ResetConfig(base_probability=0.0, context_scaling=1.0),
            restart_initial_probability=1.0,
            restart_turn_range=[1, 2],
        )
        sessions = SessionSynthesizer(config, seed=42).synthesize_session(
            inject_restart=True
        )
        parsed = synthesized_sessions_to_parsed(sessions)
        continuation = sessions[1]
        first_turn = parsed[continuation.session_id][0]

        assert first_turn.group_id == continuation.group_id
        assert first_turn.is_restart is True


class TestExtractMetrics:
    def test_returns_all_metric_keys(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        expected_keys = {
            "initial_context",
            "new_tokens_per_turn",
            "generation_length",
            "inter_turn_delay_s",
            "turns_per_session",
            "total_isl",
            "total_osl",
            "hash_id_block_count",
            "request_latency_ms",
            "request_latency_s",
            "session_duration_min",
        }
        assert set(metrics.keys()) == expected_keys

    def test_initial_context_count_matches_sessions(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        assert len(metrics["initial_context"]) == 20

    def test_request_latency_is_positive(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        assert np.all(metrics["request_latency_ms"] > 0)

    def test_custom_tps_affects_latency(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        m_fast = extract_metrics(sessions, prefill_tps=100_000, decode_tps=400)
        m_slow = extract_metrics(sessions, prefill_tps=25_000, decode_tps=100)
        assert np.mean(m_fast["request_latency_ms"]) < np.mean(
            m_slow["request_latency_ms"]
        )

    def test_new_tokens_use_incremental_jsonl_input_length(self) -> None:
        sessions = {
            "s1": [
                ParsedTurn(
                    session_id="s1",
                    input_length=1000,
                    output_length=20,
                    hash_ids=[],
                    delay_ms=0.0,
                ),
                ParsedTurn(
                    session_id="s1",
                    input_length=75,
                    output_length=10,
                    hash_ids=[],
                    delay_ms=1.0,
                ),
                ParsedTurn(
                    session_id="s1",
                    input_length=125,
                    output_length=10,
                    hash_ids=[],
                    delay_ms=1.0,
                ),
            ]
        }

        metrics = extract_metrics(sessions)

        assert metrics["initial_context"].tolist() == [1000.0]
        assert metrics["new_tokens_per_turn"].tolist() == [75.0, 125.0]


class TestBuildReportData:
    def test_comparisons_include_target_metrics(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        data = build_report_data(metrics)
        names = [c.metric_name for c in data.comparisons]
        assert "Initial Context (tokens)" in names
        assert "Generation Length (tokens)" in names

    def test_session_count_matches(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        data = build_report_data(metrics)
        assert data.session_count == 20

    def test_pct_error_is_non_negative(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        data = build_report_data(metrics)
        for c in data.comparisons:
            if c.pct_error_mean is not None:
                assert c.pct_error_mean >= 0


class TestRenderTextReport:
    def test_produces_non_empty_string(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        data = build_report_data(metrics)
        text = render_text_report(data)
        assert len(text) > 100
        assert "Target vs Observed" in text

    def test_contains_summary_table(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        data = build_report_data(metrics)
        text = render_text_report(data)
        assert "Summary Statistics" in text


class TestGenerateReport:
    def test_text_only(self, run_dir: Path) -> None:
        data = generate_report(run_dir, fmt="text")
        assert data.session_count == 20
        assert not (run_dir / "report").exists()

    def test_plot_only(self, run_dir: Path) -> None:
        generate_report(run_dir, fmt="plot")
        assert (run_dir / "report.html").exists()

    def test_both(self, run_dir: Path) -> None:
        data = generate_report(run_dir, fmt="both")
        assert data.session_count == 20
        assert (run_dir / "report.html").exists()

    def test_missing_jsonl_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_report(tmp_path)

    def test_report_matches_quality_metrics_from_generated_dataset(
        self,
        run_dir: Path,
    ) -> None:
        quality = orjson.loads((run_dir / "quality.json").read_bytes())
        report = generate_report(run_dir, fmt="text")
        comparisons = {c.metric_name: c for c in report.comparisons}

        checks = {
            "Initial Context (tokens)": "initial_context",
            "New Tokens/Turn": "new_tokens_per_turn",
            "Generation Length (tokens)": "generation_length",
        }
        for report_name, quality_name in checks.items():
            quality_observed = quality["observed_vs_target"][quality_name]["observed"]
            comparison = comparisons[report_name]
            assert comparison.observed.mean == pytest.approx(
                quality_observed["mean"], rel=1e-3
            )
            assert comparison.observed.median == pytest.approx(
                quality_observed["median"], rel=1e-3
            )


class TestExtractCacheMetrics:
    def test_returns_all_cache_metric_keys(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        cache = extract_cache_metrics(sessions, block_size=512)
        expected_keys = {
            "prefix_length",
            "unique_prompt_length",
            "prefix_ratio",
            "sequential_cache_hit_rate",
            "per_session_cache_hit_rate",
        }
        assert set(cache.keys()) == expected_keys

    def test_prefix_length_le_input_length(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        metrics = extract_metrics(sessions)
        cache = extract_cache_metrics(sessions, block_size=512)
        assert np.all(cache["prefix_length"] <= metrics["total_isl"])

    def test_cache_hit_rate_bounded_zero_one(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        cache = extract_cache_metrics(sessions, block_size=512)
        for key in ("sequential_cache_hit_rate", "per_session_cache_hit_rate"):
            assert np.all(cache[key] >= 0.0)
            assert np.all(cache[key] <= 1.0)

    def test_per_session_first_turn_zero(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        cache = extract_cache_metrics(sessions, block_size=512)
        # First turn of each session should have 0 hit rate (nothing cached yet)
        idx = 0
        for session_turns in sessions.values():
            assert cache["per_session_cache_hit_rate"][idx] == 0.0
            idx += len(session_turns)


class TestRenderCacheExplorer:
    def test_produces_html_file(self, run_dir: Path) -> None:
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        cache_payload = write_cache_structure(sessions, None, run_dir)
        path = render_cache_explorer(run_dir, cache_payload)
        assert path.exists()
        content = path.read_text()
        assert "d3" in content
        assert "svg" in content

    def test_escapes_inline_json_script_end_tags(self, run_dir: Path) -> None:
        cache_payload = {
            "block_size": 512,
            "l1_blocks": 0,
            "l15_blocks": 0,
            "sessions": [{"session_id": "</script><div>", "turns": []}],
        }
        path = render_cache_explorer(run_dir, cache_payload)
        content = path.read_text()
        assert "<\\/script>" in content
        assert "</script><div>" not in content

    def test_block_classification_l1_always_cached(self) -> None:
        hash_ids = list(range(10))
        blocks = _classify_turn_blocks(hash_ids, prev_hash_id_set=None, l1_blocks=5)
        l1_blocks = [b for b in blocks if b["layer"] == "L1"]
        assert len(l1_blocks) == 5
        assert all(b["status"] == "cached" for b in l1_blocks)

    def test_block_classification_turn0_no_l3(self) -> None:
        hash_ids = list(range(12))
        blocks = _classify_turn_blocks(
            hash_ids, prev_hash_id_set=None, l1_blocks=5, l15_blocks=2
        )
        l3_blocks = [b for b in blocks if b["layer"] == "L3"]
        assert len(l3_blocks) == 0
        l15_blocks = [b for b in blocks if b["layer"] == "L1.5"]
        assert len(l15_blocks) == 2
        l2_blocks = [b for b in blocks if b["layer"] == "L2"]
        assert len(l2_blocks) == 5
        assert all(b["status"] == "new" for b in l2_blocks)

    def test_l3_appears_on_turn1(self) -> None:
        l1_blocks = 5
        l15_blocks = 2
        turn0_ids = list(range(12))
        prev_set = set(turn0_ids)
        turn1_ids = list(range(17))
        blocks = _classify_turn_blocks(
            turn1_ids,
            prev_hash_id_set=prev_set,
            l1_blocks=l1_blocks,
            l15_blocks=l15_blocks,
            turn_index=1,
        )
        l3_blocks = [b for b in blocks if b["layer"] == "L3"]
        assert len(l3_blocks) == 5
        assert all(b["status"] == "new" for b in l3_blocks)
        l2_cached = [
            b for b in blocks if b["layer"] == "L2" and b["status"] == "cached"
        ]
        assert len(l2_cached) == 5


class TestClassifierMatchesSynthesizer:
    """Verify that _classify_turn_blocks produces correct labels for
    hash_ids actually generated by the synthesizer + allocator pipeline."""

    def test_turn0_all_session_new_no_l3(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 from real synthesized data should be L1(cached) + L1.5(cached) + L2(new), no L3."""
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        t0 = session.turns[0]
        l1_blocks = synth.allocator.l1_blocks
        l15_blocks = synth.allocator.l15_blocks

        blocks = _classify_turn_blocks(
            t0.hash_ids,
            prev_hash_id_set=None,
            l1_blocks=l1_blocks,
            l15_blocks=l15_blocks,
        )

        layers = {b["layer"] for b in blocks}
        assert "L3" not in layers
        assert all(b["status"] == "cached" for b in blocks if b["layer"] == "L1")
        assert all(b["status"] == "cached" for b in blocks if b["layer"] == "L1.5")
        assert all(b["status"] == "new" for b in blocks if b["layer"] == "L2")

        l1_count = sum(1 for b in blocks if b["layer"] == "L1")
        l15_count = sum(1 for b in blocks if b["layer"] == "L1.5")
        l2_count = sum(1 for b in blocks if b["layer"] == "L2")
        assert l1_count == l1_blocks
        assert l15_count == l15_blocks
        assert l1_count + l15_count + l2_count == len(t0.hash_ids)

    def test_turn1_has_cached_session_and_new_l3(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Turn 1 should carry forward L2 as cached, and add L3 new blocks."""
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        assert len(session.turns) >= 2

        t0 = session.turns[0]
        t1 = session.turns[1]
        l1_blocks = synth.allocator.l1_blocks
        l15_blocks = synth.allocator.l15_blocks
        prev_set = set(t0.hash_ids)

        blocks = _classify_turn_blocks(
            t1.hash_ids,
            prev_hash_id_set=prev_set,
            l1_blocks=l1_blocks,
            l15_blocks=l15_blocks,
            turn_index=1,
        )

        l1 = [b for b in blocks if b["layer"] == "L1"]
        l15 = [b for b in blocks if b["layer"] == "L1.5"]
        l2_cached = [
            b for b in blocks if b["layer"] == "L2" and b["status"] == "cached"
        ]
        l3_new = [b for b in blocks if b["layer"] == "L3" and b["status"] == "new"]

        assert len(l1) == l1_blocks
        assert all(b["status"] == "cached" for b in l1)
        assert len(l15) == l15_blocks
        # L2 prefix from turn 0 is carried forward
        t0_l2_count = len(t0.hash_ids) - l1_blocks - l15_blocks
        assert len(l2_cached) == t0_l2_count
        # L3 accounts for the growth
        assert len(l3_new) == len(t1.hash_ids) - l1_blocks - l15_blocks - t0_l2_count
        assert len(l3_new) > 0

    def test_all_blocks_classified_every_turn(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Every block in every turn should be classified (no gaps)."""
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        l1_blocks = synth.allocator.l1_blocks

        l15_blocks = synth.allocator.l15_blocks

        prev_set: set[int] | None = None
        for i, turn in enumerate(session.turns):
            blocks = _classify_turn_blocks(
                turn.hash_ids, prev_set, l1_blocks, l15_blocks, turn_index=i
            )
            assert len(blocks) == len(turn.hash_ids)
            assert [b["pos"] for b in blocks] == list(range(len(turn.hash_ids)))
            prev_set = set(turn.hash_ids)


class TestSessionPrefixSizeInvariant:
    """The core invariant: on turn 0, session prefix blocks = total_blocks - prefix_blocks."""

    def test_turn0_session_prefix_equals_remainder(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        import math

        synth = SessionSynthesizer(coding_config, seed=42)
        alloc = synth.allocator
        sessions = synth.synthesize_sessions(20)
        for session in sessions:
            t0 = session.turns[0]
            total_blocks = math.ceil(t0.input_length / alloc.block_size)
            session_ids = alloc.extract_session_ids(t0.hash_ids)
            assert len(session_ids) == total_blocks - alloc.prefix_blocks

    def test_initial_context_always_exceeds_l1(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """The floor guarantees initial_ctx > layer1_tokens for every session."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(100)
        l1_tokens = small_config.cache.layer1_tokens
        for session in sessions:
            assert session.turns[0].input_length > l1_tokens


class TestCacheStructureJson:
    """Verify write_cache_structure output is consistent with the synthesized data."""

    def test_segment_counts_sum_to_num_blocks(self, run_dir: Path) -> None:
        manifest = DatasetManifest(
            **orjson.loads((run_dir / "manifest.json").read_bytes())
        )
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        data = write_cache_structure(sessions, manifest, run_dir)

        for sess in data["sessions"]:
            for turn in sess["turns"]:
                seg_total = sum(s["count"] for s in turn["segments"])
                assert seg_total == turn["num_blocks"]

    def test_l1_blocks_matches_config(self, run_dir: Path) -> None:
        import math

        manifest = DatasetManifest(
            **orjson.loads((run_dir / "manifest.json").read_bytes())
        )
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        data = write_cache_structure(sessions, manifest, run_dir)

        expected_l1 = math.ceil(
            manifest.generation_params.cache.layer1_tokens
            / manifest.generation_params.block_size
        )
        assert data["l1_blocks"] == expected_l1
        assert data["block_size"] == manifest.generation_params.block_size

    def test_turn0_segments_are_l1_l15_l2_only(self, run_dir: Path) -> None:
        manifest = DatasetManifest(
            **orjson.loads((run_dir / "manifest.json").read_bytes())
        )
        turns = load_jsonl(run_dir / "dataset.jsonl")
        sessions = group_sessions(turns)
        data = write_cache_structure(sessions, manifest, run_dir)

        for sess in data["sessions"]:
            t0 = sess["turns"][0]
            layers = {s["layer"] for s in t0["segments"]}
            assert "L3" not in layers
            assert layers <= {"L1", "L1.5", "L2"}

    def test_sessions_capped_at_50(self, tmp_path: Path) -> None:
        """write_cache_structure should emit at most 50 sessions."""
        # Build fake sessions dict with 60 entries
        fake_sessions: dict[str, list[ParsedTurn]] = {}
        for i in range(60):
            sid = f"sess-{i:04d}"
            fake_sessions[sid] = [
                ParsedTurn(
                    session_id=sid,
                    input_length=1024,
                    output_length=100,
                    hash_ids=list(range(10)),
                    delay_ms=0.0,
                )
            ]
        data = write_cache_structure(fake_sessions, None, tmp_path)
        assert len(data["sessions"]) == 50


class TestGenerateReportCacheExplorer:
    def test_plot_format_produces_cache_explorer_files(self, run_dir: Path) -> None:
        generate_report(run_dir, fmt="plot")
        assert (run_dir / "cache_structure.json").exists()
        assert (run_dir / "cache_explorer.html").exists()

    def test_cache_explorer_created_by_writer(self, run_dir: Path) -> None:
        """write_dataset always produces cache explorer files."""
        assert (run_dir / "cache_structure.json").exists()
        assert (run_dir / "cache_explorer.html").exists()
