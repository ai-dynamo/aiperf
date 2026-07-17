# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""t* snapshot selection in the graph IR source (Task 12).

Per trace the source samples a wall-clock instant ``t*`` in
``[start_min_ratio, start_max_ratio] * trace_duration`` (deterministic under a
seed, mirroring agentx ``TrajectorySource``). ``t*==0`` (the default
full-replay ratio) means full native replay with no warmup history.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.timing.graph_ir_source import GraphIRConversationSource

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def _multi_trace_corpus(tmp_path: Path, n_traces: int = 3) -> Path:
    """A directory of ``n_traces`` weka traces with DISTINCT durations.

    Derived from the committed single-trace fixture by re-id'ing and scaling
    the recorded timestamps, so per-trace t* windows differ and cross-trace /
    cross-seed variation is actually observable.
    """
    base = orjson.loads(FIX.read_bytes())
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(n_traces):
        doc = dict(base)
        doc["id"] = f"trace_{i}"
        scale = float(i + 1)
        doc["requests"] = [{**req, "t": req["t"] * scale} for req in base["requests"]]
        (corpus / f"trace_{i}.json").write_bytes(orjson.dumps(doc))
    return corpus


def test_full_replay_default_yields_tstar_zero() -> None:
    parsed = from_weka_trace(str(FIX))
    src = GraphIRConversationSource(
        parsed=parsed, start_min_ratio=0.0, start_max_ratio=0.0
    )
    gt = next(iter(src.iter_traces()))
    # Ratios [0, 0] => t*=0 => full native replay, no warmup history.
    assert gt.t_star_us == 0


def test_positive_window_samples_nonzero_tstar() -> None:
    parsed = from_weka_trace(str(FIX))
    src = GraphIRConversationSource(
        parsed=parsed,
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        random_seed=42,
    )
    gt = next(iter(src.iter_traces()))
    # weka_min: 3 root turns at offsets 0, 1.5s, 3.0s. A 50% window over the
    # ~3s duration engages a positive t* (not the inert t*=0 identity replay).
    assert gt.t_star_us > 0


def test_tstar_is_deterministic_under_seed() -> None:
    parsed = from_weka_trace(str(FIX))
    kw = dict(start_min_ratio=0.0, start_max_ratio=0.7, random_seed=123)
    a = next(iter(GraphIRConversationSource(parsed=parsed, **kw).iter_traces()))
    b = next(iter(GraphIRConversationSource(parsed=parsed, **kw).iter_traces()))
    assert a.t_star_us == b.t_star_us


def test_tstar_varies_across_traces_and_seeds(tmp_path: Path) -> None:
    """Distinct seeds pick distinct t* profiles over a REAL multi-trace corpus.

    A single-trace fixture would make any per-trace-id disjunction vacuous;
    three traces with distinct durations force both axes to be observable.
    """
    parsed = from_weka_trace(str(_multi_trace_corpus(tmp_path)))
    assert len(parsed.traces) == 3
    kw = dict(start_min_ratio=0.0, start_max_ratio=0.9)

    seed_1 = {
        gt.trace_id: gt.t_star_us
        for gt in GraphIRConversationSource(
            parsed=parsed, random_seed=1, **kw
        ).iter_traces()
    }
    seed_2 = {
        gt.trace_id: gt.t_star_us
        for gt in GraphIRConversationSource(
            parsed=parsed, random_seed=2, **kw
        ).iter_traces()
    }

    assert set(seed_1) == set(seed_2) and len(seed_1) == 3
    # Across seeds: the sampled t* profile must differ.
    assert seed_1 != seed_2
    # Across traces (within one seed): distinct-duration traces must not all
    # collapse to one t*.
    assert len(set(seed_1.values())) > 1


def test_source_yields_one_graphtrace_per_trace() -> None:
    parsed = from_weka_trace(str(FIX))
    src = GraphIRConversationSource(parsed=parsed)
    traces = list(src.iter_traces())
    assert len(traces) == len(parsed.traces)
    assert traces[0].parsed_graph is parsed
