# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MooncakeTrace per-row HTTP headers."""

import pytest
from pydantic import ValidationError

from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader


class TestMooncakeHeadersValidation:
    """Pydantic-level validation of the new optional ``headers`` field."""

    def test_headers_default_is_none(self):
        trace = MooncakeTrace(text_input="hi")
        assert trace.headers is None

    def test_headers_accepts_str_str_dict(self):
        headers = {"x-session-token": "tok-1", "baggage": "k=v,a=b"}
        trace = MooncakeTrace(text_input="hi", headers=headers)
        assert trace.headers == headers

    def test_headers_rejects_non_string_value(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(text_input="hi", headers={"x": 1})  # type: ignore[arg-type]

    def test_headers_round_trip_through_json(self):
        raw = '{"text_input":"hi","headers":{"x-session-token":"tok-1"}}'
        trace = MooncakeTrace.model_validate_json(raw)
        assert trace.headers == {"x-session-token": "tok-1"}


class TestMooncakeBuildTurnForwardsHeaders:
    """``MooncakeTraceDatasetLoader._build_turn`` carries headers onto the Turn."""

    def _loader(self) -> MooncakeTraceDatasetLoader:
        # Bypass __init__ so we can call _build_turn without a real config.
        # The base _build_turn clamps delay through _delay_cap_tracker, so
        # supply an uncapped one; these tests don't exercise delay.
        loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
        loader._delay_cap_tracker = DelayCapTracker(cap_seconds=None)
        return loader

    def test_build_turn_text_input_path_carries_headers(self):
        loader = self._loader()
        trace = MooncakeTrace(
            text_input="hi",
            output_length=4,
            headers={"x-session-token": "tok-A"},
        )
        turn = loader._build_turn(trace, "hi")
        assert turn.extra_headers == {"x-session-token": "tok-A"}

    def test_build_turn_messages_path_carries_headers(self):
        loader = self._loader()
        trace = MooncakeTrace(
            messages=[{"role": "user", "content": "hi"}],
            output_length=4,
            headers={"baggage": "userId=alice"},
        )
        turn = loader._build_turn(trace, "")
        assert turn.extra_headers == {"baggage": "userId=alice"}

    def test_build_turn_payload_path_carries_headers(self):
        loader = self._loader()
        trace = MooncakeTrace(
            payload={"prompt": "hi", "max_tokens": 4},
            output_length=4,
            headers={"x-session-token": "tok-C"},
        )
        turn = loader._build_turn(trace, "")
        assert turn.extra_headers == {"x-session-token": "tok-C"}

    def test_build_turn_no_headers_yields_none(self):
        loader = self._loader()
        trace = MooncakeTrace(text_input="hi", output_length=4)
        turn = loader._build_turn(trace, "hi")
        assert turn.extra_headers is None


class TestMooncakeSynthesisPreservesHeaders:
    """Synthesis must not mutate or drop the ``headers`` field."""

    def test_synthesis_excludes_headers(self):
        loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
        assert "headers" in loader._synthesis_exclude_fields()

    def test_reconstruct_traces_reattaches_original_headers(self):
        loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
        originals = [
            MooncakeTrace(
                text_input="hi", output_length=4, headers={"x-session-token": "tok-A"}
            ),
            MooncakeTrace(
                text_input="bye", output_length=4, headers={"x-session-token": "tok-B"}
            ),
        ]
        # Simulate post-synthesis dicts (no ``headers`` key — it was excluded).
        synth_dicts = [
            {"text_input": "hi-syn", "output_length": 4},
            {"text_input": "bye-syn", "output_length": 4},
        ]
        rebuilt = loader._reconstruct_traces(originals, synth_dicts)
        assert rebuilt[0].headers == {"x-session-token": "tok-A"}
        assert rebuilt[1].headers == {"x-session-token": "tok-B"}

    def test_reconstruct_traces_tail_reuses_last_original_headers(self):
        # Synthesis may emit more traces than originals (e.g., speedup
        # expansion); the tail traces reuse the last original's headers.
        loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
        originals = [
            MooncakeTrace(
                text_input="hi", output_length=4, headers={"x-session-token": "tok-A"}
            ),
            MooncakeTrace(
                text_input="bye", output_length=4, headers={"x-session-token": "tok-B"}
            ),
        ]
        synth_dicts = [
            {"text_input": "hi-syn", "output_length": 4},
            {"text_input": "bye-syn", "output_length": 4},
            {"text_input": "tail-syn", "output_length": 4},
        ]
        rebuilt = loader._reconstruct_traces(originals, synth_dicts)
        assert rebuilt[0].headers == {"x-session-token": "tok-A"}
        assert rebuilt[1].headers == {"x-session-token": "tok-B"}
        assert rebuilt[2].headers == {"x-session-token": "tok-B"}

    def test_reconstruct_traces_without_originals_keeps_none(self):
        loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
        synth_dicts = [{"text_input": "hi", "output_length": 4}]
        rebuilt = loader._reconstruct_traces([], synth_dicts)
        assert rebuilt[0].headers is None
