# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory-estimator calibration script.

Focuses on pure-logic surface that doesn't require pympler measurements:
- ``_mib`` / ``_fmt`` formatting helpers (KiB/MiB/B switching, units).
- ``_next_pow2`` rounding.
- Object factories (``_make_prompt_text``, ``_make_turn``,
  ``_make_streaming_response``, ``_make_text_response``,
  ``_make_request_record``) — verify shape and counts only.
- ``Scenario`` defaults / field construction.
- ``measure_records_manager`` extrapolation logic (real MetricArrays, just
  small N) — confirms scaling math.

The pympler-driven branches (``measure_single_request``,
``measure_inflight_set``, ``measure_record_processor``,
``print_object_reference``, ``run_scenario``, ``main``) are
integration/measurement-side and intentionally skipped.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.analysis.calibrate_memory_estimates import (
    _CHARS_PER_TOKEN,
    SCENARIOS,
    Scenario,
    _fmt,
    _make_prompt_text,
    _make_request_record,
    _make_streaming_response,
    _make_text_response,
    _make_turn,
    _mib,
    _next_pow2,
    measure_records_manager,
)
from aiperf.common.enums import SSEFieldType

# ============================================================
# Numeric helpers
# ============================================================


class TestMib:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (1024 * 1024, 1.0),
            (0, 0.0),
            param(1024 * 1024 * 100, 100.0, id="100-MiB"),
            param(2_097_152, 2.0, id="exact-2-MiB"),
        ],
    )  # fmt: skip
    def test_mib_converts_bytes_to_mib(self, raw: int, expected: float) -> None:
        assert _mib(raw) == expected


class TestFmt:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0, "0 B"),
            (1023, "1,023 B"),
            param(2 * 1024, "2.0 KiB", id="2-KiB"),
            param(int(1.5 * 1024 * 1024), "1.5 MiB", id="1.5-MiB"),
            param(1024 * 1024, "1.0 MiB", id="exact-1-MiB"),
        ],
    )  # fmt: skip
    def test_fmt_picks_unit_by_magnitude(self, raw: int, expected: str) -> None:
        assert _fmt(raw) == expected


class TestNextPow2:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 4),
            (5, 8),
            (1024, 1024),
            param(1025, 2048, id="just-over-pow2"),
            param(-5, 1, id="negative-clamps-to-one"),
        ],
    )  # fmt: skip
    def test_next_pow2_rounds_up(self, n: int, expected: int) -> None:
        assert _next_pow2(n) == expected


# ============================================================
# Object factories
# ============================================================


class TestMakePromptText:
    @pytest.mark.parametrize("isl", [0, 1, 64, 512])
    def test_make_prompt_text_length_scales_with_isl(self, isl: int) -> None:
        text = _make_prompt_text(isl)
        assert len(text) == isl * _CHARS_PER_TOKEN


class TestMakeTurn:
    def test_make_turn_returns_user_role(self) -> None:
        turn = _make_turn(isl=10)
        assert turn.role == "user"

    def test_make_turn_contains_single_text_with_one_content_string(self) -> None:
        turn = _make_turn(isl=8)
        assert len(turn.texts) == 1
        assert len(turn.texts[0].contents) == 1
        assert len(turn.texts[0].contents[0]) == 8 * _CHARS_PER_TOKEN


class TestMakeStreamingResponse:
    @pytest.mark.parametrize("osl", [1, 8, 64])
    def test_make_streaming_response_one_packet_per_token(self, osl: int) -> None:
        msg = _make_streaming_response(osl)
        assert len(msg.packets) == osl
        for pkt in msg.packets:
            assert pkt.name == SSEFieldType.DATA

    def test_make_streaming_response_perf_ns_set(self) -> None:
        msg = _make_streaming_response(osl=1)
        assert msg.perf_ns > 0


class TestMakeTextResponse:
    @pytest.mark.parametrize("osl", [1, 64, 256])
    def test_make_text_response_body_grows_with_osl(self, osl: int) -> None:
        resp = _make_text_response(osl)
        assert "completion_tokens" in resp.text
        # Body must contain at least osl * _CHARS_PER_TOKEN content chars.
        assert resp.text.count("y") >= osl * _CHARS_PER_TOKEN


class TestMakeRequestRecord:
    def test_make_request_record_streaming_uses_sse_response(self) -> None:
        rec = _make_request_record(isl=16, osl=4, streaming=True)
        assert len(rec.responses) == 1
        # Streaming branch sets recv_start_perf_ns; non-streaming leaves None.
        assert rec.recv_start_perf_ns is not None

    def test_make_request_record_non_streaming_returns_text_response(self) -> None:
        rec = _make_request_record(isl=16, osl=4, streaming=False)
        assert len(rec.responses) == 1
        assert rec.recv_start_perf_ns is None

    @pytest.mark.parametrize("turns", [1, 3, 5])
    def test_make_request_record_turn_count_matches_param(self, turns: int) -> None:
        rec = _make_request_record(isl=16, osl=4, streaming=True, turns=turns)
        assert len(rec.turns) == turns

    def test_make_request_record_status_200(self) -> None:
        rec = _make_request_record(isl=16, osl=4, streaming=True)
        assert rec.status == 200


# ============================================================
# Scenario dataclass
# ============================================================


class TestScenario:
    def test_scenario_required_fields_only(self) -> None:
        s = Scenario(
            name="t",
            isl=128,
            osl=64,
            streaming=True,
            concurrency=10,
            total_requests=100,
        )
        assert s.turns == 1
        assert s.total_workers == 10
        assert s.workers_per_pod == 10
        assert s.num_models == 1
        assert s.duration_s == 300.0
        assert s.num_gpus == 0

    def test_scenario_module_default_list_non_empty(self) -> None:
        # Sanity: the script ships a populated default scenario list.
        assert len(SCENARIOS) > 0
        for s in SCENARIOS:
            assert isinstance(s, Scenario)
            assert s.total_requests > 0
            assert s.concurrency > 0


# ============================================================
# measure_records_manager (small-N path; pympler is real but cheap here)
# ============================================================


class TestMeasureRecordsManager:
    def test_measure_records_manager_no_extrapolation_below_cap(self) -> None:
        # Below the 200K cap, actual_filled == extrapolated_to.
        s = Scenario(
            name="tiny",
            isl=8,
            osl=4,
            streaming=True,
            concurrency=2,
            total_requests=100,
        )
        result = measure_records_manager(s)
        assert result["actual_filled"] == 100
        assert result["extrapolated_to"] == 100
        assert result["records_manager_bytes"] > 0

    def test_measure_records_manager_extrapolates_when_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the cap to be tiny so we exercise the extrapolation branch.
        # We also stub asizeof.asizeof to a fixed value for determinism.
        import aiperf.analysis.calibrate_memory_estimates as mod

        monkeypatch.setattr(mod.asizeof, "asizeof", lambda _obj: 1_000_000)

        # total_requests > 200_000 cap forces extrapolation.
        s = Scenario(
            name="huge",
            isl=8,
            osl=4,
            streaming=True,
            concurrency=2,
            total_requests=400_000,
        )
        result = measure_records_manager(s)
        assert result["actual_filled"] == 200_000
        assert result["extrapolated_to"] == 400_000
        # 1_000_000 * (400_000 / 200_000) == 2_000_000
        assert result["records_manager_bytes"] == 2_000_000
