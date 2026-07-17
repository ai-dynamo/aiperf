# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-seed determinism through ``parse_graph_workload``.

Content-seed determinism across in-process and spawn-started pool-worker parses
of the same run config. The weka content seed is
``resolve_graph_content_seed(run)`` -- the run ``--random-seed`` verbatim, with
NO weka-specific fallback. With an explicit seed every parse synthesizes
byte-identical content; with the seed unset (``None``) synthesis defers to the
process's ambient global RNG manager, and only the seed-independent
catalog/ordinal invariants are guaranteed. The TimingManager never parses -- it
ingests the graph_meta sidecar from the graph-typed dataset broadcast.

The byte-identity assertions here compare the REAL synthesized bytes -- the
segment pool's materialized ``(role, content, parent_id)`` per
content-addressed segment id -- on FULL parses. ``TraceRecord.replay_outputs``
is always empty on the weka path and proves nothing.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.timing.config import resolve_graph_content_seed
from tests.unit.conftest import make_run_from_cli

WEKA_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


def _graph_run(input_file: Path = WEKA_MIN, **cli_overrides):
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(input_file),
        # Content tokenizer now comes from the run config (no env override): pin
        # the offline builtin so the fake "test-model" doesn't trigger a HF load.
        # Mirrors the CLI's fake-model -> builtin substitution the full pipeline
        # applies before parsing.
        tokenizer_name="builtin",
        **cli_overrides,
    )
    return make_run_from_cli(cfg)


def _pool_contents(parsed: ParsedGraph) -> dict[str, tuple[str, str, str | None]]:
    """Materialized segment content keyed by content-addressed segment id.

    This is the real synthesized-content image of a full parse: comparing two
    of these compares the actual bytes workers materialize, not the always-empty
    ``TraceRecord.replay_outputs``.
    """
    pool = parsed.segment_pool
    assert pool is not None, "weka parse must surface the segment pool"
    assert pool._by_id, "weka parse must carry real content segments"
    return {sid: (s.role, s.content, s.parent_id) for sid, s in pool._by_id.items()}


def test_prompt_corpus_flows_to_synthesis_and_resolver() -> None:
    """``--prompt-corpus`` lands on ``synthesis.corpus`` and the graph resolver
    reads it; default (unset) resolves to ``coding``."""
    from aiperf.dataset.graph.workload_detect import _resolve_graph_corpus

    default_run = _graph_run()
    assert _resolve_graph_corpus(default_run) == "coding"

    sonnet_run = _graph_run(prompt_corpus="sonnet")
    assert sonnet_run.cfg.get_default_dataset().synthesis.corpus == "sonnet"
    assert _resolve_graph_corpus(sonnet_run) == "sonnet"


def test_idle_gap_cap_flows_to_synthesis_and_resolver() -> None:
    """``--synthesis-idle-gap-cap`` lands on ``synthesis.idle_gap_cap_seconds`` and
    the graph resolver reads it; default (unset) resolves to 60.0."""
    from aiperf.dataset.graph.workload_detect import _resolve_graph_idle_gap_cap

    assert _resolve_graph_idle_gap_cap(_graph_run()) == 60.0

    capped = _graph_run(synthesis_idle_gap_cap=30.0)
    assert capped.cfg.get_default_dataset().synthesis.idle_gap_cap_seconds == 30.0
    assert _resolve_graph_idle_gap_cap(capped) == 30.0


def test_parse_graph_workload_explicit_null_idle_gap_replays_raw_offsets(
    tmp_path: Path,
) -> None:
    """Explicit ``synthesis.idle_gap_cap_seconds: null`` replays RAW offsets.

    The regression both adversarial reviews of the GraphParseContext plan
    proved the original plan would ship silently: the run's explicit-null
    idle-gap disable collapsing into the 60s adapter default somewhere along
    resolver -> ctx -> adapter, silently warping the replay geometry. The
    fixture's second request starts at raw ``t=137124`` after the first ends at
    ``t=2`` (a 137122s idle gap >> any cap), so the two outcomes are far apart:
    unwarped replay keeps ``arrival_offset_us == 137_124_000_000``; a collapse
    into the 60s cap yields ``62_000_000``. The default-cap arm pins the warped
    value too, proving the fixture actually engages the cap (falsifiability).

    Construction note: the CLI converter DROPS ``None`` values
    (``_converter_dataset.py``: ``if value is not None``), so
    ``make_run_from_cli(synthesis_idle_gap_cap=None)`` yields the DEFAULT 60s
    cap, not explicit-null. Build the run, then set
    ``synthesis.idle_gap_cap_seconds = None`` on the default dataset directly
    (attaching a synthesis object if absent). Explicit null IS end-user
    reachable via a YAML config, so this pin covers a real configuration.
    """
    from aiperf.config.dataset.trace import SynthesisConfig

    trace_file = tmp_path / "idle_gap.json"
    trace_file.write_bytes(
        orjson.dumps(
            {
                "id": "trace_idle_gap",
                "models": ["M"],
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [
                    {"t": 0.0, "type": "n", "model": "M", "in": 128, "out": 64,
                     "hash_ids": [1, 2], "api_time": 2.0},
                    {"t": 137124.0, "type": "n", "model": "M", "in": 256, "out": 64,
                     "hash_ids": [1, 2, 3, 4], "api_time": 1.0},
                ],
            }
        )
    )  # fmt: skip

    # Falsifiability arm: the default (unset) cap DOES warp this fixture --
    # turn1 lands at end(2s) + capped idle (60s) = 62s.
    warped = parse_graph_workload(_graph_run(input_file=trace_file), trace_file)
    assert {
        nid: n.arrival_offset_us for nid, n in warped.graph.nodes.items()
    } == {"trace_idle_gap:0": 0, "trace_idle_gap:1": 62_000_000}  # fmt: skip

    null_run = _graph_run(input_file=trace_file)
    dataset = null_run.cfg.get_default_dataset()
    if dataset.synthesis is None:
        dataset.synthesis = SynthesisConfig()
    dataset.synthesis.idle_gap_cap_seconds = None

    unwarped = parse_graph_workload(null_run, trace_file)
    assert {
        nid: n.arrival_offset_us for nid, n in unwarped.graph.nodes.items()
    } == {"trace_idle_gap:0": 0, "trace_idle_gap:1": 137_124_000_000}  # fmt: skip


def test_parse_graph_workload_random_seed_unset_two_parses_identical() -> None:
    """Two full parses at random_seed=None match real content bytes + ordinals."""
    run = _graph_run()
    assert run.random_seed is None, "default single run must leave --random-seed unset"

    build_plane = parse_graph_workload(run, WEKA_MIN)
    timing_plane = parse_graph_workload(run, WEKA_MIN)

    # Content: the materialized segment bytes must be identical in-process
    # (both parses defer to the same ambient RNG manager).
    assert _pool_contents(build_plane) == _pool_contents(timing_plane)

    # Ordinals: the build/runtime addressing catalog must match too.
    assert build_graph_path_catalog(build_plane) == build_graph_path_catalog(
        timing_plane
    )


def test_parse_graph_workload_explicit_seed_threads_to_content_synthesis() -> None:
    """``parse_graph_workload`` threads exactly ``resolve_graph_content_seed(run)``.

    If the seed were dropped (or a different value substituted), the direct
    same-seed ``from_weka_trace`` parse would synthesize different bytes and
    this comparison would fail -- seed-dependence is proven by the divergence
    test below.
    """
    run = _graph_run(random_seed=7)
    assert resolve_graph_content_seed(run) == 7

    via_parse = parse_graph_workload(run, WEKA_MIN)
    direct_same_seed = from_weka_trace(str(WEKA_MIN), content_root_seed=7)
    assert _pool_contents(via_parse) == _pool_contents(direct_same_seed)


def test_parse_graph_workload_different_seeds_diverge_content_not_catalog() -> None:
    """Different seeds synthesize different bytes; catalog ordinals are seed-free.

    The divergence arm makes the byte-identity assertions falsifiable: if
    content were seed-independent (or the pools compared vacuously), this test
    would fail.
    """
    parsed_seed_7 = parse_graph_workload(_graph_run(random_seed=7), WEKA_MIN)
    parsed_seed_8 = parse_graph_workload(_graph_run(random_seed=8), WEKA_MIN)

    contents_7 = _pool_contents(parsed_seed_7)
    contents_8 = _pool_contents(parsed_seed_8)
    assert contents_7 != contents_8, "distinct seeds must synthesize distinct bytes"
    # The synthesized text itself differs, not just the token-derived ids.
    assert sorted(c for _, c, _ in contents_7.values()) != sorted(
        c for _, c, _ in contents_8.values()
    )

    # Addressing is seed-independent: same catalog regardless of content seed.
    assert build_graph_path_catalog(parsed_seed_7) == build_graph_path_catalog(
        parsed_seed_8
    )
