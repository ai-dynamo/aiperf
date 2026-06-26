# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline replay-correctness suite for the baseten_trace loader.

FAST (no server, no GPU, mock tokenizer), fully in-process. This is the
standing characterization net that must stay green as the loader is refactored
(see traffic-replay/plan.md, traffic-replay/MODULES.md).

Run just this suite:   pytest tests/unit/dataset/loader/test_baseten_offline_suite.py

It exercises the REAL code path at three levels:
  1. BasetenTraceDatasetLoader.load_dataset()       -> grouped BasetenTrace rows
  2. loader.convert_to_conversations(data)          -> Conversation/Turn objects
  3. CompletionsEndpoint.format_payload(...)         -> the on-the-wire payload

Convention: checks for features that are NOT YET IMPLEMENTED are marked
`xfail(strict=False)` with a reason naming the plan item. When that work lands
the test xpasses -> drop the xfail and make it strict. Everything else is a hard
assertion: a failure means a real regression.
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
GAP_CAP_MS = 5_000  # target max idle gap once the P1.2 gap-cap lands


def _fixture_rows() -> list[dict]:
    """Small but structurally faithful: 3 sessions, multi-turn, growing hash
    prefixes (KV reuse), realistic gaps (incl. one > GAP_CAP), a canceled row,
    and the recorded-outcome columns."""
    return [
        # session A: 3 turns, growing shared prefix [10,11] -> [10,11,12] -> [10,11,12,13]
        dict(timestamp_start_unix_ms=1_000, prompt="A-1 reconstructed prompt text", input_tokens=128, output_tokens=50,
             total_hashes=[10, 11], provided_session_id="A", poor_man_session_id=1, block_size=BLOCK_SIZE,
             request_canceled=0, duration_e2e_ms=800, duration_ttft_ms=120, cached_tokens_reference=0),
        dict(timestamp_start_unix_ms=3_000, prompt="A-2 reconstructed prompt text", input_tokens=192, output_tokens=40,
             total_hashes=[10, 11, 12], provided_session_id="A", poor_man_session_id=1, block_size=BLOCK_SIZE,
             request_canceled=0, duration_e2e_ms=700, duration_ttft_ms=110, cached_tokens_reference=128),
        dict(timestamp_start_unix_ms=6_000, prompt="A-3 reconstructed prompt text", input_tokens=256, output_tokens=60,
             total_hashes=[10, 11, 12, 13], provided_session_id="A", poor_man_session_id=1, block_size=BLOCK_SIZE,
             request_canceled=0, duration_e2e_ms=900, duration_ttft_ms=130, cached_tokens_reference=192),
        # session B: 2 turns with a deliberate 7.5s idle gap (> GAP_CAP_MS)
        dict(timestamp_start_unix_ms=1_500, prompt="B-1 reconstructed prompt text", input_tokens=128, output_tokens=30,
             total_hashes=[20, 21], provided_session_id="B", poor_man_session_id=2, block_size=BLOCK_SIZE,
             request_canceled=0, duration_e2e_ms=600, duration_ttft_ms=100, cached_tokens_reference=0),
        dict(timestamp_start_unix_ms=9_000, prompt="B-2 reconstructed prompt text", input_tokens=192, output_tokens=45,
             total_hashes=[20, 21, 22], provided_session_id="B", poor_man_session_id=2, block_size=BLOCK_SIZE,
             request_canceled=0, duration_e2e_ms=750, duration_ttft_ms=115, cached_tokens_reference=128),
        # session C: single-turn, canceled
        dict(timestamp_start_unix_ms=12_000, prompt="C-1 reconstructed prompt text", input_tokens=64, output_tokens=20,
             total_hashes=[30], provided_session_id="C", poor_man_session_id=3, block_size=BLOCK_SIZE,
             request_canceled=1, duration_e2e_ms=300, duration_ttft_ms=90, cached_tokens_reference=0),
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
        filename=str(path), run=_make_run(path, **cli), prompt_generator=_mock_prompt_generator()
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
    def test_can_load(self, fixture_path):
        assert BasetenTraceDatasetLoader.can_load(filename=fixture_path) is True

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
                assert isinstance(payload["prompt"], str), f"prompt not a str: {payload['prompt']!r}"

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
        _, d1 = _load(fixture_path, trace_session_sample_ratio=0.67, random_seed=123)
        _, d2 = _load(fixture_path, trace_session_sample_ratio=0.67, random_seed=123)
        assert set(d1.keys()) == set(d2.keys())
        assert set(d1.keys()).issubset({"A", "B", "C"})


class TestTimingNoHang:
    def test_first_turn_has_timestamp(self, fixture_path):
        # fixed_schedule raises on a None first-turn timestamp.
        loader, data = _load(fixture_path)
        for conv in loader.convert_to_conversations(data):
            assert conv.turns[0].timestamp is not None

    @pytest.mark.xfail(reason="P1.2 max-idle-gap-cap not implemented (session B has a 7.5s gap)", strict=False)
    def test_no_idle_gap_exceeds_cap(self, fixture_path):
        _, data = _load(fixture_path)
        for sid, traces in data.items():
            ts = sorted(t.timestamp for t in traces)
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            assert all(g <= GAP_CAP_MS for g in gaps), f"session {sid}: gaps {gaps} exceed cap {GAP_CAP_MS}"

    @pytest.mark.xfail(reason="P2 back-pressure not wired: turns>0 should carry delay, not absolute timestamp", strict=False)
    def test_continuation_turns_use_delay(self, fixture_path):
        loader, data = _load(fixture_path)
        for conv in loader.convert_to_conversations(data):
            for turn in conv.turns[1:]:
                assert turn.delay is not None


class TestFidelityCarryThrough:
    def test_recorded_outcomes_present(self, fixture_path):
        # Recorded ground truth must survive load so a later comparison is possible.
        _, data = _load(fixture_path)
        a1 = data["A"][0]
        assert a1.duration_e2e_ms == 800
        assert a1.duration_ttft_ms == 120
        assert a1.cached_tokens_reference is not None


class TestHashIntegrity:
    def test_hash_ids_not_rewritten(self, fixture_path):
        _, data = _load(fixture_path)
        assert data["A"][0].request_body["hash_ids"] == [10, 11]
        assert data["A"][2].request_body["hash_ids"] == [10, 11, 12, 13]


class TestRepresentativenessTier2:
    @pytest.mark.skip(reason="Tier 2: runs against a real dataset slice; not part of the fast suite")
    def test_sample_matches_full_distribution(self):
        """Placeholder: load a real contiguous multi-session slice and assert the
        sampled ISL/OSL/session distributions + recomputed KV-hit track the full trace."""
