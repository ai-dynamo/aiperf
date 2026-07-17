# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog presence keys off the spliced ``__msgdelta`` node, not delta VALUES.

Phase 2 strips message TEXT out of ``replay_outputs``; the per-trace node-ordinal
catalog must survive that, because its ordinals are the dispatch<->store contract
(a shift => GraphEnvelopeMissing). This proves emptying every trace's
``replay_outputs`` VALUES leaves the catalog byte-identical and non-empty.
"""

from pathlib import Path

import msgspec

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog

WEKA_FIXTURE = Path(__file__).parents[2] / "graph" / "fixtures" / "weka_min.json"

# The fixture is COMMITTED; a missing path is repo drift and must fail loudly,
# never silently vanish the test via a skipif.
assert WEKA_FIXTURE.exists(), f"committed weka fixture missing: {WEKA_FIXTURE}"


def test_catalog_survives_emptied_replay_outputs() -> None:
    pg = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
    before = build_graph_path_catalog(pg)
    emptied_traces = [msgspec.structs.replace(t, replay_outputs={}) for t in pg.traces]
    pg2 = msgspec.structs.replace(pg, traces=emptied_traces)
    after = build_graph_path_catalog(pg2)
    assert after == before
    assert before  # non-empty
