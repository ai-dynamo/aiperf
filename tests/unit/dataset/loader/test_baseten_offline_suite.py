# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline replay-correctness suite for the baseten_trace loader.

FAST (no server, no GPU, mock tokenizer), fully in-process. This is the
standing characterization net that must stay green as the loader is refactored.

Run just this suite:   pytest tests/unit/dataset/loader/test_baseten_offline_suite.py

It exercises the REAL code path at three levels:
  1. BasetenTraceDatasetLoader.load_dataset()       -> grouped BasetenTrace rows
  2. loader.convert_to_conversations(data)          -> Conversation/Turn objects
  3. CompletionsEndpoint.format_payload(...)         -> the on-the-wire payload

Every check is a hard assertion: a failure means a real regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.baseten_trace import BasetenTraceDatasetLoader
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import CustomDatasetType, EndpointType
from tests.unit.conftest import make_run_from_cli
from tests.unit.endpoints.conftest import create_request_info

BLOCK_SIZE = 64
GAP_CAP_MS = 5_000  # max idle gap used when exercising the gap-cap


def _fixture_rows() -> list[dict]:
    """Small but structurally faithful: 3 sessions, multi-turn, growing hash
    prefixes (KV reuse), realistic gaps (incl. one > GAP_CAP), a canceled row,
    and the recorded-outcome columns."""
    return [
        # session A: 3 turns, growing shared prefix [10,11] -> [10,11,12] -> [10,11,12,13]
        dict(
            timestamp_start_unix_ms=1_000,
            prompt="A-1 reconstructed prompt text",
            input_tokens=128,
            output_tokens=50,
            total_hashes=[10, 11],
            provided_session_id="A",
            poor_man_session_id=1,
            block_size=BLOCK_SIZE,
            request_canceled=0,
            duration_e2e_ms=800,
            duration_ttft_ms=120,
            cached_tokens_reference=0,
        ),
        dict(
            timestamp_start_unix_ms=3_000,
            prompt="A-2 reconstructed prompt text",
            input_tokens=192,
            output_tokens=40,
            total_hashes=[10, 11, 12],
            provided_session_id="A",
            poor_man_session_id=1,
            block_size=BLOCK_SIZE,
            request_canceled=0,
            duration_e2e_ms=700,
            duration_ttft_ms=110,
            cached_tokens_reference=128,
        ),
        dict(
            timestamp_start_unix_ms=6_000,
            prompt="A-3 reconstructed prompt text",
            input_tokens=256,
            output_tokens=60,
            total_hashes=[10, 11, 12, 13],
            provided_session_id="A",
            poor_man_session_id=1,
            block_size=BLOCK_SIZE,
            request_canceled=0,
            duration_e2e_ms=900,
            duration_ttft_ms=130,
            cached_tokens_reference=192,
        ),
        # session B: 2 turns with a deliberate 7.5s idle gap (> GAP_CAP_MS)
        dict(
            timestamp_start_unix_ms=1_500,
            prompt="B-1 reconstructed prompt text",
            input_tokens=128,
            output_tokens=30,
            total_hashes=[20, 21],
            provided_session_id="B",
            poor_man_session_id=2,
            block_size=BLOCK_SIZE,
            request_canceled=0,
            duration_e2e_ms=600,
            duration_ttft_ms=100,
            cached_tokens_reference=0,
        ),
        dict(
            timestamp_start_unix_ms=9_000,
            prompt="B-2 reconstructed prompt text",
            input_tokens=192,
            output_tokens=45,
            total_hashes=[20, 21, 22],
            provided_session_id="B",
            poor_man_session_id=2,
            block_size=BLOCK_SIZE,
            request_canceled=0,
            duration_e2e_ms=750,
            duration_ttft_ms=115,
            cached_tokens_reference=128,
        ),
        # session C: single-turn, canceled, far in the future (creates an 11s
        # global dead-air gap after B-2 to exercise the idle-gap cap)
        dict(
            timestamp_start_unix_ms=20_000,
            prompt="C-1 reconstructed prompt text",
            input_tokens=64,
            output_tokens=20,
            total_hashes=[30],
            provided_session_id="C",
            poor_man_session_id=3,
            block_size=BLOCK_SIZE,
            request_canceled=1,
            duration_e2e_ms=300,
            duration_ttft_ms=90,
            cached_tokens_reference=0,
        ),
    ]


def _mock_prompt_generator() -> Mock:
    gen = Mock()
    gen._decoded_cache = {}
    gen.tokenizer.resolved_name = "test-tokenizer"
    return gen


def _make_run(input_file: Path, **cli):
    cli.setdefault("input_file", str(input_file))
    cli.setdefault("custom_dataset_type", CustomDatasetType.BASETEN_TRACE)
    return make_run_from_cli(CLIConfig(model_names=["test-model"], **cli))


def _load(path: Path, **cli):
    loader = BasetenTraceDatasetLoader(
        filename=str(path),
        run=_make_run(path, **cli),
        prompt_generator=_mock_prompt_generator(),
    )
    return loader, loader.load_dataset()


def _completions_model_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.RANDOM,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.COMPLETIONS,
            base_url="http://localhost:8000",
            custom_endpoint="/v1/completions",
        ),
    )


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    path = tmp_path / "baseten_fixture.parquet"
    pq.write_table(pa.Table.from_pylist(_fixture_rows()), path)
    return path


class TestSchemaAndLoad:
    def test_groups_three_sessions_by_provided_id(self, fixture_path):
        _, data = _load(fixture_path)
        assert set(data.keys()) == {"A", "B", "C"}

    def test_turns_sorted_within_session(self, fixture_path):
        _, data = _load(fixture_path)
        ts = [t.timestamp for t in data["A"]]
        assert ts == sorted(ts)

    def test_timestamps_normalized_to_zero_origin(self, fixture_path):
        _, data = _load(fixture_path)
        all_ts = [t.timestamp for traces in data.values() for t in traces]
        assert min(all_ts) == 0


class TestRequestBodyContract:
    def test_request_body_min_tokens_and_cache_meta(self, fixture_path):
        _, data = _load(fixture_path)
        a1 = data["A"][0]
        assert a1.request_body["min_tokens"] == a1.output_length == 50
        assert a1.request_body["hash_ids"] == [10, 11]
        assert a1.request_body["block_size"] == BLOCK_SIZE

    def test_completions_payload_prompt_is_bare_string(self, fixture_path):
        # Baseten's gateway rejects list[str]; a single prompt must be a bare str.
        loader, data = _load(fixture_path)
        model_endpoint = _completions_model_endpoint()
        endpoint = CompletionsEndpoint(model_endpoint)
        for conv in loader.convert_to_conversations(data):
            for turn in conv.turns:
                payload = endpoint.format_payload(
                    create_request_info(model_endpoint=model_endpoint, turns=[turn])
                )
                assert isinstance(payload["prompt"], str), (
                    f"prompt not a str: {payload['prompt']!r}"
                )

    def test_completions_payload_carries_replay_metadata(self, fixture_path):
        loader, data = _load(fixture_path)
        model_endpoint = _completions_model_endpoint()
        endpoint = CompletionsEndpoint(model_endpoint)
        conv = next(c for c in loader.convert_to_conversations(data) if c.turns)
        turn = conv.turns[0]
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["min_tokens"] == turn.max_tokens
        assert "hash_ids" in payload and "block_size" in payload


class TestDeterminism:
    def test_session_sampling_is_deterministic(self, fixture_path):
        _, d1 = _load(fixture_path, trace_session_sample_ratio=0.4, random_seed=123)
        _, d2 = _load(fixture_path, trace_session_sample_ratio=0.4, random_seed=123)
        assert d1.keys() == d2.keys()
        assert d1, "sampling must keep at least one session"
        assert set(d1.keys()) < {"A", "B", "C"}, (
            "0.4 sampling should keep a non-empty strict subset"
        )
        assert [[t.prompt for t in traces] for traces in d1.values()] == [
            [t.prompt for t in traces] for traces in d2.values()
        ]


class TestTimingNoHang:
    def test_global_idle_gap_capped(self, fixture_path):
        # With the cap, no gap between consecutive requests (across all sessions)
        # exceeds the cap, so fixed-schedule replay never idles longer than that.
        _, data = _load(fixture_path, max_idle_gap_cap_seconds=GAP_CAP_MS / 1000)
        ts = sorted(t.timestamp for traces in data.values() for t in traces)
        gaps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        assert gaps, "fixture should have multiple events"
        assert max(gaps) <= GAP_CAP_MS, f"global gaps exceed cap: {gaps}"

    def test_gap_cap_is_opt_in(self, fixture_path):
        # Default (no cap): the large dead-air gap is preserved verbatim.
        _, data = _load(fixture_path)
        ts = sorted(t.timestamp for traces in data.values() for t in traces)
        gaps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        assert max(gaps) > GAP_CAP_MS

    def test_gap_cap_only_collapses_oversized_gap(self, fixture_path):
        # The cap collapses ONLY the one oversized gap; smaller structure is intact.
        _, capped = _load(fixture_path, max_idle_gap_cap_seconds=GAP_CAP_MS / 1000)
        _, uncapped = _load(fixture_path)
        cap_ts = sorted(t.timestamp for tr in capped.values() for t in tr)
        unc_ts = sorted(t.timestamp for tr in uncapped.values() for t in tr)
        assert cap_ts[-1] == unc_ts[-1] - (11_000 - GAP_CAP_MS)
        assert cap_ts[:5] == unc_ts[:5]

    def test_back_pressure_turn0_absolute_continuation_delay(self, fixture_path):
        # Closed-loop multi-turn replay: turn 0 keeps an absolute timestamp
        # (session start); continuation turns fire on completion via a delay.
        # Companion to test_open_loop_default_keeps_absolute_timestamps_no_delay:
        # together they prove open_loop_replay is what flips the behavior.
        loader, data = _load(fixture_path, open_loop_replay=False)
        multi = [c for c in loader.convert_to_conversations(data) if len(c.turns) > 1]
        assert multi, "fixture should have a multi-turn session"
        for conv in multi:
            assert conv.turns[0].timestamp is not None
            assert conv.turns[0].delay is None
            for turn in conv.turns[1:]:
                assert turn.delay is not None
                assert turn.timestamp is None

    def test_back_pressure_subtracts_prior_service_time(self, fixture_path):
        # A continuation turn's delay = recorded start-to-start gap MINUS the
        # prior turn's recorded e2e. fixed_schedule applies the delay AFTER the
        # prior turn completes in replay, so using the raw gap would double-count
        # server time (replay inter-arrival = replay_service + recorded_service +
        # think). Fixture (zero-origin): A gaps 2000,3000 w/ prev e2e 800,700 ->
        # 1200,2300; B gap 7500 w/ prev e2e 600 -> 6900.
        loader, data = _load(fixture_path, open_loop_replay=False)
        delays = sorted(
            turn.delay
            for conv in loader.convert_to_conversations(data)
            for turn in conv.turns[1:]
        )
        assert delays == [1200.0, 2300.0, 6900.0]
        # regression guard: must NOT be the raw gaps (which double-count service)
        assert not ({2000.0, 3000.0, 7500.0} & set(delays))

    def test_back_pressure_falls_back_to_raw_gap_without_duration(self, tmp_path):
        # When the prior turn's duration_e2e_ms is absent, the delay falls back
        # to the raw start-to-start gap (nothing to subtract).
        rows = [
            dict(
                timestamp_start_unix_ms=1_000,
                prompt="Z-1",
                input_tokens=10,
                output_tokens=5,
                total_hashes=[1],
                provided_session_id="Z",
                poor_man_session_id=9,
                block_size=BLOCK_SIZE,
                request_canceled=0,
                duration_e2e_ms=None,
                duration_ttft_ms=None,
                cached_tokens_reference=0,
            ),
            dict(
                timestamp_start_unix_ms=4_000,
                prompt="Z-2",
                input_tokens=12,
                output_tokens=6,
                total_hashes=[1, 2],
                provided_session_id="Z",
                poor_man_session_id=9,
                block_size=BLOCK_SIZE,
                request_canceled=0,
                duration_e2e_ms=500,
                duration_ttft_ms=90,
                cached_tokens_reference=0,
            ),
        ]
        path = tmp_path / "nulldur.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        loader, data = _load(path, open_loop_replay=False)
        conts = [
            t for conv in loader.convert_to_conversations(data) for t in conv.turns[1:]
        ]
        assert conts and conts[0].delay == 3000.0  # raw gap, no subtraction

    def test_closed_loop_gap_cap_preserves_recorded_think_time(self, tmp_path):
        # Closed-loop think time derives from the RECORDED start-to-start gap,
        # not the gap-cap-reflowed one. Fixture: 150s recorded gap with a 120s
        # prior e2e -> 30s think time, exactly at a 30s gap cap. Reflowing
        # first would compress the gap to 30s and zero the delay
        # (max(0, 30_000 - 120_000)); think-time bounding belongs to
        # inter_turn_delay_cap_seconds alone.
        rows = [
            dict(
                timestamp_start_unix_ms=1_000,
                prompt="G-1",
                input_tokens=10,
                output_tokens=5,
                provided_session_id="G",
                duration_e2e_ms=120_000,
            ),
            dict(
                timestamp_start_unix_ms=151_000,
                prompt="G-2",
                input_tokens=12,
                output_tokens=6,
                provided_session_id="G",
                duration_e2e_ms=500,
            ),
        ]
        path = tmp_path / "gapcap.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        loader, data = _load(
            path, open_loop_replay=False, max_idle_gap_cap_seconds=30.0
        )
        conts = [
            t for conv in loader.convert_to_conversations(data) for t in conv.turns[1:]
        ]
        assert conts and conts[0].delay == 30_000.0

    def test_closed_loop_gap_cap_still_collapses_dead_air_between_sessions(
        self, tmp_path
    ):
        # In closed-loop mode the gap cap still applies to the absolute
        # session-start timestamps (the only ones left on the schedule).
        rows = [
            dict(
                timestamp_start_unix_ms=1_000,
                prompt="S1",
                input_tokens=10,
                output_tokens=5,
                provided_session_id="S1",
                duration_e2e_ms=500,
            ),
            dict(
                timestamp_start_unix_ms=201_000,
                prompt="S2",
                input_tokens=10,
                output_tokens=5,
                provided_session_id="S2",
                duration_e2e_ms=500,
            ),
        ]
        path = tmp_path / "deadair.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        loader, data = _load(
            path, open_loop_replay=False, max_idle_gap_cap_seconds=30.0
        )
        starts = sorted(
            conv.turns[0].timestamp for conv in loader.convert_to_conversations(data)
        )
        assert starts == [0, 30_000]

    def test_inter_turn_delay_cap_clamps_continuation_delays(self, fixture_path):
        # The existing inter_turn_delay_cap_seconds knob clamps think-time so a
        # session with long gaps stays runnable.
        loader, data = _load(
            fixture_path, inter_turn_delay_cap_seconds=1.0, open_loop_replay=False
        )
        capped = [
            turn
            for conv in loader.convert_to_conversations(data)
            for turn in conv.turns[1:]
        ]
        assert capped, "fixture should have continuation turns"
        assert all(t.delay is not None and t.delay <= 1000 for t in capped)

    def test_open_loop_default_keeps_absolute_timestamps_no_delay(self, fixture_path):
        # Open-loop replay (the default): back-pressure is skipped, so EVERY turn
        # across every session keeps its absolute timestamp and no delay is set.
        loader, data = _load(fixture_path)
        conversations = loader.convert_to_conversations(data)
        assert conversations, "fixture should produce conversations"
        all_turns = [turn for conv in conversations for turn in conv.turns]
        assert any(len(conv.turns) > 1 for conv in conversations), (
            "fixture should have a multi-turn session"
        )
        for turn in all_turns:
            assert turn.timestamp is not None
            assert turn.delay is None


class TestFidelityCarryThrough:
    def test_recorded_outcomes_present(self, fixture_path):
        # Recorded ground truth must survive load so a later comparison is possible.
        _, data = _load(fixture_path)
        a1 = data["A"][0]
        assert a1.duration_e2e_ms == 800
        assert a1.duration_ttft_ms == 120
        assert a1.cached_tokens_reference is not None


class TestTimeCompression:
    def test_speedup_compresses_timestamps_not_hashes(self, fixture_path):
        # replay_speedup shrinks the wall-clock by the factor; hash_ids untouched.
        _, base = _load(fixture_path)
        _, fast = _load(fixture_path, replay_speedup=10.0)
        b = max(t.timestamp for tr in base.values() for t in tr)
        f = max(t.timestamp for tr in fast.values() for t in tr)
        assert f == b / 10
        assert (
            fast["A"][2].request_body["hash_ids"]
            == base["A"][2].request_body["hash_ids"]
            == [10, 11, 12, 13]
        )


class TestHashIntegrity:
    def test_hash_ids_not_rewritten(self, fixture_path):
        _, data = _load(fixture_path)
        assert data["A"][0].request_body["hash_ids"] == [10, 11]
        assert data["A"][2].request_body["hash_ids"] == [10, 11, 12, 13]
