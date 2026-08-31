# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the inter-turn delay clamp (`_clamp_delay_ms`): boundary, sign, NaN/Inf, zero/None cap, parent vs subagent paths, think-time-only (spec 8.4.4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader, _clamp_delay_ms

# Helper-level adversarial cases (operate directly on `_clamp_delay_ms`).


def test_clamp_at_cap_is_inclusive_unchanged():
    # Boundary: exactly at cap is *not* clamped (preserves original float identity
    # when no rewrite is needed).
    assert _clamp_delay_ms(60_000.0, cap_seconds=60.0) == 60_000.0


def test_clamp_one_microsecond_above_cap_clamps():
    # `60_000.001 ms` is `60s + 1us`; must be clamped down to exactly `cap_ms`.
    assert _clamp_delay_ms(60_000.001, cap_seconds=60.0) == 60_000.0


def test_clamp_negative_passes_through_corrupt_trace():
    # Pinned behavior: clamp only enforces the upper bound. Negative `delay_ms`
    # (corrupt trace) is intentionally left untouched so other validation layers
    # can flag it explicitly. Documented in the helper docstring.
    assert _clamp_delay_ms(-100.0, cap_seconds=60.0) == -100.0


def test_clamp_nan_maps_to_none():
    # NaN comparisons are always false, so without an isfinite gate the
    # `delay_ms > cap_ms` branch never fires. Scrub to None (absent delay).
    assert _clamp_delay_ms(float("nan"), cap_seconds=60.0) is None
    assert _clamp_delay_ms(float("nan"), cap_seconds=None) is None


def test_clamp_non_finite_inf_maps_to_none():
    # ±Inf are non-finite; scrub to None rather than clamping or passing through.
    assert _clamp_delay_ms(float("inf"), cap_seconds=60.0) is None
    assert _clamp_delay_ms(float("-inf"), cap_seconds=60.0) is None
    assert _clamp_delay_ms(float("inf"), cap_seconds=None) is None


def test_clamp_zero_cap_clamps_everything_to_zero():
    # Legal but unusual: cap=0 effectively disables inter-turn delays.
    assert _clamp_delay_ms(1.0, cap_seconds=0.0) == 0.0
    assert _clamp_delay_ms(0.0, cap_seconds=0.0) == 0.0
    assert _clamp_delay_ms(86_400_000.0, cap_seconds=0.0) == 0.0


def test_clamp_none_cap_passes_through_24h_delay():
    # Default: no cap -> even pathologically large delays survive.
    assert _clamp_delay_ms(86_400_000.0, cap_seconds=None) == 86_400_000.0


# Parameterized integration tests: parent path (line ~400) and subagent path
# (line ~527) must clamp identically. Spec 8.4.4 calls for "a parameterized
# test that runs the same scenarios on both code paths".

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(
    *,
    cap_seconds: float | None,
    think_time_only: bool = False,
    **overrides,
):
    from tests.unit.dataset.loader.conftest import make_weka_run

    overrides.setdefault(
        "model_names", ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]
    )
    return make_weka_run(
        tokenizer_name="test-tok",
        inter_turn_delay_cap_seconds=cap_seconds,
        use_think_time_only=think_time_only,
        **overrides,
    )


def _stub_prompt_generator(loader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _make_two_turn_parent_trace(
    *,
    second_turn_t: float,
    second_turn_think_time: float | None = 0.0,
) -> dict:
    """Parent trace with two normal requests: turn[1].delay = (t1-t0)*1000."""
    return {
        "id": "trace_clamp_parent",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 100,
                "out": 10,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 1.0,
                "think_time": 0.0,
            },
            {
                "t": second_turn_t,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 200,
                "out": 20,
                # Extends turn 0's [1, 2] prefix: consecutive same-agent turns
                # must chain or flattened-agent detection will split them.
                "hash_ids": [1, 2, 3],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 1.0,
                "think_time": second_turn_think_time,
            },
        ],
    }


def _make_subagent_trace_with_two_child_turns(
    *,
    child_second_t: float,
    child_second_think_time: float | None = 0.0,
) -> dict:
    """Parent normal request plus a t=0.0 subagent block with two child requests, so the child path computes a delay on absolute root-timeline timestamps."""
    return {
        "id": "trace_clamp_child",
        "models": ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 100,
                "out": 10,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "tool_use",
                "api_time": 1.0,
                "think_time": 0.0,
            },
            {
                "t": 0.0,
                "type": "subagent",
                "agent_id": "agent_clamp",
                "subagent_type": "Explore",
                "duration_ms": 5000,
                "total_tokens": 500,
                "tool_use_count": 2,
                "status": "completed",
                "models": ["claude-haiku-4-5-20251001"],
                "tool_tokens": 20,
                "system_tokens": 10,
                "requests": [
                    {
                        "t": 0.0,
                        "type": "n",
                        "model": "claude-haiku-4-5-20251001",
                        "in": 100,
                        "out": 30,
                        "hash_ids": [10, 11],
                        "input_types": ["text"],
                        "output_types": ["text"],
                        "stop": "end_turn",
                        "api_time": 0.5,
                        "think_time": 0.0,
                    },
                    {
                        "t": child_second_t,
                        "type": "n",
                        "model": "claude-haiku-4-5-20251001",
                        "in": 150,
                        "out": 40,
                        # Extends the first request's [10, 11] prefix so LCP
                        # chain detection keeps both requests in ONE chain
                        # (a disjoint hash list would split them into two
                        # one-turn children and there would be no delay).
                        "hash_ids": [10, 11, 12, 13],
                        "input_types": ["text"],
                        "output_types": ["text"],
                        "stop": "end_turn",
                        "api_time": 0.5,
                        "think_time": child_second_think_time,
                    },
                ],
            },
        ],
    }


def _build_loader(tmp_path, trace: dict, uc, monkeypatch) -> WekaTraceLoader:
    f = tmp_path / f"{trace['id']}.json"
    f.write_bytes(orjson.dumps(trace))
    loader = WekaTraceLoader(filename=str(f), run=uc)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"prompt-{r.key}" for r in rs},
    )
    _stub_prompt_generator(loader)
    return loader


# (cap_seconds, second_turn_t_seconds, parent_expected_ms, child_expected_ms)
# The emitted delay is the end-to-start idle gap (start-to-start minus the
# previous turn's api_time), then clamped to cap. The parent fixture uses
# api_time=1.0s and the child fixture uses api_time=0.5s, so the two paths
# diverge below the cap even though they share the same start-to-start matrix.
_PARAM_CASES = [
    # at-cap inclusive: 60s start-to-start -> 59.0s/59.5s end-to-start, under cap
    pytest.param(60.0, 60.0, 59_000.0, 59_500.0, id="at_cap_inclusive"),
    # just over cap on start-to-start, still under cap after end-to-start
    pytest.param(60.0, 60.001, 59_001.0, 59_501.0, id="just_above_cap_clamps"),
    # well over cap: 24h -> end-to-start still clamped to 60_000ms
    pytest.param(60.0, 86_400.0, 60_000.0, 60_000.0, id="huge_delay_clamps"),
    # zero cap -> any positive delay clamps to 0
    pytest.param(0.0, 5.0, 0.0, 0.0, id="zero_cap_clamps_to_zero"),
    # None cap -> end-to-start 24h passes through
    pytest.param(
        None, 86_400.0, 86_399_000.0, 86_399_500.0, id="none_cap_24h_passthrough"
    ),
]


@pytest.mark.parametrize(
    "cap_seconds,second_t,parent_expected_ms,child_expected_ms", _PARAM_CASES
)
def test_parent_turn_delay_clamp_matrix(
    tmp_path, monkeypatch, cap_seconds, second_t, parent_expected_ms, child_expected_ms
):
    """Parent path (`weka_trace.py:~400`) clamps with `cap_seconds`."""
    uc = _mk_user_config(cap_seconds=cap_seconds)
    trace = _make_two_turn_parent_trace(second_turn_t=second_t)
    loader = _build_loader(tmp_path, trace, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_clamp_parent")
    assert parent.turns[0].delay is None  # first turn always
    assert parent.turns[1].delay == pytest.approx(parent_expected_ms)


@pytest.mark.parametrize(
    "cap_seconds,second_t,parent_expected_ms,child_expected_ms", _PARAM_CASES
)
def test_subagent_child_turn_delay_clamp_matrix(
    tmp_path, monkeypatch, cap_seconds, second_t, parent_expected_ms, child_expected_ms
):
    """Subagent child path (`weka_trace.py:~527`) clamps with the same `cap_seconds` matrix as the parent path."""
    uc = _mk_user_config(cap_seconds=cap_seconds)
    trace = _make_subagent_trace_with_two_child_turns(child_second_t=second_t)
    loader = _build_loader(tmp_path, trace, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id.endswith("::sa:agent_clamp"))
    assert child.turns[0].delay is None
    assert child.turns[1].delay == pytest.approx(child_expected_ms)


# Cap interaction with `--use-think-time-only` (spec 8.4.4 bullet 8).


def test_think_time_only_path_also_clamps_when_think_time_exceeds_cap(
    tmp_path, monkeypatch
):
    """With `use_think_time_only=True` and `think_time > cap`, the think_time-derived `delay_ms` is also clamped (cap applies to the active delay source)."""
    uc = _mk_user_config(cap_seconds=60.0, think_time_only=True)
    # Wall-clock delta would be 1s, but think_time=120s drives the delay.
    trace = _make_two_turn_parent_trace(
        second_turn_t=1.0,
        second_turn_think_time=120.0,
    )
    loader = _build_loader(tmp_path, trace, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_clamp_parent")
    # think_time=120s -> 120_000ms, clamped to 60_000ms by the cap.
    assert parent.turns[1].delay == pytest.approx(60_000.0)


def test_think_time_only_below_cap_passes_through(tmp_path, monkeypatch):
    """Sanity: think_time below cap is emitted unchanged even with the cap set."""
    uc = _mk_user_config(cap_seconds=60.0, think_time_only=True)
    trace = _make_two_turn_parent_trace(
        second_turn_t=1.0,
        second_turn_think_time=7.0,
    )
    loader = _build_loader(tmp_path, trace, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_clamp_parent")
    assert parent.turns[1].delay == pytest.approx(7000.0)
