# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-request memory constant re-baseline script.

Focuses on pure-logic surface:
- Object factories (``_make_turn``, ``_make_sse_message_unique``,
  ``_make_text_response``, ``_make_record``, ``_make_empty_record``).
- ``_linear_fit`` regression helper — verifies base + per_token slope math
  using a stub factory (no pympler dependency).
- ``Calibration`` dataclass: drift_pct math, OK / DRIFT classification,
  and the rendered output format.

Pympler-driven branches (``collect_calibrations``,
``validate_full_record``, ``validate_time_series``, ``main``,
``emit_constants_block``) are integration / measurement-side and skipped.
"""

from __future__ import annotations

import pytest
from pytest import param

from tools.rebaseline_memory_constants import (
    _CHARS_PER_TOKEN,
    Calibration,
    _linear_fit,
    _make_empty_record,
    _make_record,
    _make_sse_message_unique,
    _make_text_response,
    _make_turn,
)
from aiperf.common.enums import SSEFieldType

# ============================================================
# Object factories
# ============================================================


class TestMakeTurn:
    def test_make_turn_zero_isl_has_empty_texts(self) -> None:
        turn = _make_turn(isl=0)
        assert turn.role == "user"
        assert turn.texts == []

    @pytest.mark.parametrize("isl", [1, 32, 256])
    def test_make_turn_positive_isl_one_text_with_content(self, isl: int) -> None:
        turn = _make_turn(isl=isl)
        assert len(turn.texts) == 1
        contents = turn.texts[0].contents
        assert len(contents) == 1
        assert len(contents[0]) == isl * _CHARS_PER_TOKEN


class TestMakeSseMessageUnique:
    def test_make_sse_message_unique_one_packet_per_token(self) -> None:
        msg = _make_sse_message_unique(osl=5)
        assert len(msg.packets) == 5

    def test_make_sse_message_unique_chunk_values_are_unique(self) -> None:
        # Distinct values defeat string interning — this is the whole point
        # of having a "unique" variant of the factory.
        msg = _make_sse_message_unique(osl=8)
        values = [pkt.value for pkt in msg.packets]
        assert len(set(values)) == len(values)
        for pkt in msg.packets:
            assert pkt.name == SSEFieldType.DATA

    def test_make_sse_message_unique_zero_osl_empty_packets(self) -> None:
        msg = _make_sse_message_unique(osl=0)
        assert msg.packets == []


class TestMakeTextResponse:
    def test_make_text_response_zero_osl_empty_text(self) -> None:
        resp = _make_text_response(osl=0)
        assert resp.text == ""

    @pytest.mark.parametrize("osl", [1, 64, 1024])
    def test_make_text_response_grows_with_osl(self, osl: int) -> None:
        resp = _make_text_response(osl=osl)
        assert resp.text.count("y") >= osl * _CHARS_PER_TOKEN
        assert "completion_tokens" in resp.text


class TestMakeEmptyRecord:
    def test_make_empty_record_has_no_responses_or_turns(self) -> None:
        rec = _make_empty_record()
        assert rec.responses == []
        assert rec.turns == []
        assert rec.status == 200


class TestMakeRecord:
    def test_make_record_streaming_uses_sse_message_response(self) -> None:
        rec = _make_record(isl=16, osl=4, streaming=True)
        assert len(rec.responses) == 1
        # Streaming branch sets recv_start_perf_ns.
        assert rec.recv_start_perf_ns is not None

    def test_make_record_non_streaming_no_recv_start(self) -> None:
        rec = _make_record(isl=16, osl=4, streaming=False)
        assert rec.recv_start_perf_ns is None

    @pytest.mark.parametrize("turns", [1, 3])
    def test_make_record_turn_count(self, turns: int) -> None:
        rec = _make_record(isl=16, osl=4, streaming=True, turns=turns)
        assert len(rec.turns) == turns


# ============================================================
# _linear_fit
# ============================================================


class TestLinearFit:
    def test_linear_fit_perfect_line_recovers_slope_and_intercept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub pympler.asizeof.asizeof to return n itself so the implied
        # "size" function is size(n) = 100 + 7*n. We model the factory's
        # output as a wrapper carrying n; asizeof returns 100 + 7*n.
        import tools.rebaseline_memory_constants as mod

        def fake_asizeof(obj: int) -> int:
            return 100 + 7 * obj

        monkeypatch.setattr(mod.asizeof, "asizeof", fake_asizeof)

        # The factory just returns the integer N — _linear_fit uses asizeof
        # on whatever the factory yields.
        base, per_token = _linear_fit(lambda n: n, [0, 100, 1000])
        assert base == 100
        assert per_token == 7.0

    def test_linear_fit_uses_at_zero_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tools.rebaseline_memory_constants as mod

        # Make asizeof return a constant — irrelevant for at_zero=42 override.
        monkeypatch.setattr(mod.asizeof, "asizeof", lambda _obj: 200)

        base, per_token = _linear_fit(lambda n: n, [100], at_zero=42)
        assert base == 42
        # per_token = (200 - 42) / 100 = 1.58
        assert per_token == pytest.approx(1.58)

    def test_linear_fit_inserts_zero_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tools.rebaseline_memory_constants as mod

        # asizeof returns 50 + 3*n.
        def fake_asizeof(obj: int) -> int:
            return 50 + 3 * obj

        monkeypatch.setattr(mod.asizeof, "asizeof", fake_asizeof)

        # 0 not in sizes; helper must call factory(0) on its own.
        base, per_token = _linear_fit(lambda n: n, [200])
        assert base == 50
        assert per_token == pytest.approx(3.0)


# ============================================================
# Calibration
# ============================================================


class TestCalibration:
    def test_drift_pct_zero_current_returns_zero(self) -> None:
        cal = Calibration(name="x", current=0.0, measured=100.0, unit="B")
        assert cal.drift_pct == 0.0

    @pytest.mark.parametrize(
        "current,measured,expected",
        [
            (100.0, 110.0, 10.0),
            (100.0, 90.0, -10.0),
            (200.0, 200.0, 0.0),
            param(50.0, 75.0, 50.0, id="50pct-up"),
        ],
    )  # fmt: skip
    def test_drift_pct_signed_percentage(
        self, current: float, measured: float, expected: float
    ) -> None:
        cal = Calibration(name="x", current=current, measured=measured, unit="B")
        assert cal.drift_pct == pytest.approx(expected)

    @pytest.mark.parametrize(
        "current,measured,tolerance,expected",
        [
            (100.0, 105.0, 10.0, "OK"),
            (100.0, 110.0, 10.0, "OK"),  # exactly at boundary (abs(10) <= 10)
            (100.0, 90.0, 10.0, "OK"),
            param(100.0, 111.0, 10.0, "DRIFT", id="just-over-positive"),
            param(100.0, 85.0, 10.0, "DRIFT", id="negative-drift"),
        ],
    )  # fmt: skip
    def test_status_classifies_against_tolerance(
        self,
        current: float,
        measured: float,
        tolerance: float,
        expected: str,
    ) -> None:
        cal = Calibration(name="x", current=current, measured=measured, unit="B")
        assert cal.status(tolerance) == expected

    def test_render_marks_ok_when_within_tolerance(self) -> None:
        cal = Calibration(name="_FOO_BYTES", current=100.0, measured=105.0, unit="B")
        rendered = cal.render(tolerance_pct=10.0)
        assert "[OK" in rendered
        assert "_FOO_BYTES" in rendered
        # The percentage value '5.0%' is formatted with width padding, so the
        # exact substring '+5.0%' does not appear; instead the leading '+'
        # sits separately from the right-justified number.
        assert "drift=+" in rendered
        assert "5.0%" in rendered

    def test_render_marks_drift_outside_tolerance(self) -> None:
        cal = Calibration(name="_FOO_BYTES", current=100.0, measured=200.0, unit="B")
        rendered = cal.render(tolerance_pct=10.0)
        assert "[DRIFT" in rendered
        assert "+100.0%" in rendered

    def test_render_negative_drift_has_no_explicit_plus_sign(self) -> None:
        cal = Calibration(name="_FOO_BYTES", current=100.0, measured=80.0, unit="B")
        rendered = cal.render(tolerance_pct=5.0)
        # Negative drift renders without a leading '+'.
        assert "+-" not in rendered
        assert "-20.0%" in rendered
