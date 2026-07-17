# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF streaming store build threads the run content knobs.

The eager parse (``parse_graph_workload``) resolves ``content_tokenizer`` /
``prompt_corpus`` / ``max_osl`` / ``idle_gap_cap_seconds`` from the run config
and threads them into the weka parse; the HF STREAMING store build must thread
the SAME knobs into its per-row workers or the streamed store synthesizes
builtin+"coding" bytes instead of the run-resolved content. Knobs resolve from
the run config so every parse of the same run agrees. These tests pin both
wiring seams:

* ``stream_weka_trace_segment_payloads`` forwards the knobs into
  ``parse_kwargs``;
* ``GraphStoreBuilder._build_graph_store_streaming`` resolves them from the run
  exactly like the eager parse (the ONE ``resolve_graph_parse_context``
  resolution, spread verbatim into the stream entry).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli

WEKA_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_stream_segment_payloads_threads_content_knobs(monkeypatch) -> None:
    """The streaming entry point forwards every run content knob to workers."""
    from aiperf.dataset.graph.adapters.weka import trace as weka_trace
    from aiperf.dataset.graph.adapters.weka import trace_parallel

    captured: dict[str, Any] = {}

    def fake_iter(items, *, source_label, item_count, workers, parse_kwargs):  # noqa: ANN001, ARG001
        captured.update(parse_kwargs)
        yield "payload"

    monkeypatch.setattr(trace_parallel, "iter_item_segment_payloads", fake_iter)
    monkeypatch.setattr(weka_trace, "_load_hf_rows", lambda *a, **k: iter(()))  # noqa: ARG005

    payloads = list(
        weka_trace.stream_weka_trace_segment_payloads(
            "org/weka-corpus",
            content_root_seed=7,
            content_tokenizer="run-tok",
            prompt_corpus="sonnet",
            max_osl=64,
            idle_gap_cap_seconds=30.0,
        )
    )

    assert payloads == ["payload"]
    assert captured["content_root_seed"] == 7
    assert captured["content_tokenizer"] == "run-tok"
    assert captured["prompt_corpus"] == "sonnet"
    assert captured["max_osl"] == 64
    assert captured["idle_gap_cap_seconds"] == 30.0


@pytest.mark.asyncio
async def test_build_graph_store_streaming_resolves_run_knobs(monkeypatch) -> None:
    """The streaming store build resolves knobs from the run like the eager plane."""
    from aiperf.dataset.graph.adapters.weka import trace as weka_trace
    from aiperf.dataset.graph.store_build import GraphStoreBuilder
    from aiperf.timing.config import (
        resolve_graph_content_seed,
        resolve_graph_content_tokenizer,
    )

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(WEKA_MIN),
            tokenizer_name="builtin",
            prompt_corpus="sonnet",
            synthesis_max_osl=64,
            synthesis_idle_gap_cap=30.0,
        )
    )

    captured: dict[str, Any] = {}

    def fake_stream(path, **kwargs):  # noqa: ANN001
        captured["path"] = path
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(weka_trace, "stream_weka_trace_segment_payloads", fake_stream)

    sentinel_catalog = {"trace": {"r_0": 0}}

    async def fake_trie(payloads, base_path):  # noqa: ANN001, ARG001
        return sentinel_catalog, None

    stub = SimpleNamespace(run=run, _build_graph_store_streaming_trie=fake_trie)

    catalog, merged = await GraphStoreBuilder._build_graph_store_streaming(
        stub, Path("org/weka-corpus"), Path("/tmp/unused"), "weka_trace"
    )

    assert catalog is sentinel_catalog
    assert merged is None
    assert captured["content_root_seed"] == resolve_graph_content_seed(run)
    assert captured["content_tokenizer"] == resolve_graph_content_tokenizer(run)
    assert captured["prompt_corpus"] == "sonnet"
    assert captured["max_osl"] == 64
    assert captured["idle_gap_cap_seconds"] == 30.0
