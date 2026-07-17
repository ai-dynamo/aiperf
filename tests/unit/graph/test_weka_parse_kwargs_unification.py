# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.dataset.graph.adapters.weka import trace as weka_trace

FIX_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_resolve_parse_kwargs_keys_match_parse_trace_dict_signature() -> None:
    # Drift guard: the ONE kwargs dict must cover exactly the keyword set of
    # the shared per-trace core (minus the per-item ``source`` label).
    kwargs = weka_trace._resolve_parse_kwargs(
        tag="t",
        idle_gap_cap_seconds=weka_trace._USE_DEFAULT,
        content_root_seed=None,
        content_tokenizer=None,
        prompt_corpus=None,
        max_osl=None,
    )
    sig = inspect.signature(weka_trace._parse_trace_dict)
    expected = {
        name
        for name, p in sig.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and name != "source"
    }
    assert set(kwargs) == expected
    assert "delay_cap_seconds" not in kwargs


def test_resolve_parse_kwargs_resolves_sentinel_and_seed() -> None:
    rng.reset()
    rng.init(777)
    kwargs = weka_trace._resolve_parse_kwargs(
        tag="t",
        idle_gap_cap_seconds=weka_trace._USE_DEFAULT,
        content_root_seed=None,
        content_tokenizer=None,
        prompt_corpus=None,
        max_osl=None,
    )
    assert kwargs["idle_gap_cap_seconds"] == weka_trace._DEFAULT_IDLE_GAP_CAP_SECONDS
    assert kwargs["content_root_seed"] == 777


def test_from_weka_trace_threads_resolved_kwargs_to_core(monkeypatch) -> None:
    rng.reset()
    rng.init(777)
    captured: list[dict[str, Any]] = []
    real = weka_trace._parse_trace_dict

    def spy(raw: dict[str, Any], *, source: str, **kwargs: Any):
        captured.append(dict(kwargs))
        return real(raw, source=source, **kwargs)

    monkeypatch.setattr(weka_trace, "_parse_trace_dict", spy)
    weka_trace.from_weka_trace(str(FIX_MIN))
    assert len(captured) == 1
    assert captured[0]["content_root_seed"] == 777
    assert (
        captured[0]["idle_gap_cap_seconds"] == weka_trace._DEFAULT_IDLE_GAP_CAP_SECONDS
    )
