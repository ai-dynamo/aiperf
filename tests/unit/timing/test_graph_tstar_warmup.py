# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The t* snapshot warmup phase is reachable end-to-end on a real dynamo trace."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.adapters.dynamo import from_dynamo_trace
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph
from aiperf.graph.ids import chain_key
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.agent_graph_source import AgentGraphConversationSource
from aiperf.timing.config import TimingConfig
from aiperf.timing.strategies.graph_warmup import (
    _warmup_boundary_nodes,
    rewrite_for_warmup,
)
from tests.unit.conftest import make_run_from_cli

DYNAMO_TRACE = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "graph"
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)
SEED = 42


def _timing_config(**overrides: object) -> TimingConfig:
    """TimingConfig for a dynamo-trace run with the given t* window overrides."""
    cli = CLIConfig(
        model_names=["m"],
        tokenizer="builtin",
        input_file=str(DYNAMO_TRACE),
        **overrides,  # type: ignore[arg-type]
    )
    run = make_run_from_cli(cli)
    DatasetResolver().resolve(run)
    return TimingConfig.from_run(run)


@pytest.fixture(scope="module")
def parsed() -> ParsedGraph:
    """The real dynamo nested trace, parsed once for the whole module."""
    return from_dynamo_trace(
        DYNAMO_TRACE, content_root_seed=SEED, content_tokenizer="builtin"
    )


class TestGraphAutoWarmupGate:
    """``timing.config`` injects the graph WARMUP phase iff the t* window is on."""

    @pytest.mark.parametrize(
        ("overrides", "expected_phases"),
        [
            param({}, [CreditPhase.PROFILING], id="bare-no-warmup"),
            param(
                {
                    "trajectory_start_min_ratio": 0.0,
                    "trajectory_start_max_ratio": 0.0,
                },
                [CreditPhase.PROFILING],
                id="explicit-zero-no-warmup",
            ),
            param(
                {
                    "trajectory_start_min_ratio": 0.0,
                    "trajectory_start_max_ratio": 0.5,
                },
                [CreditPhase.WARMUP, CreditPhase.PROFILING],
                id="tstar-half",
            ),
            param(
                {
                    "trajectory_start_min_ratio": 0.0,
                    "trajectory_start_max_ratio": 1.0,
                },
                [CreditPhase.WARMUP, CreditPhase.PROFILING],
                id="tstar-full",
            ),
        ],
    )  # fmt: skip
    def test_warmup_injected_only_when_tstar_window_open(
        self, overrides: dict[str, object], expected_phases: list[CreditPhase]
    ) -> None:
        config = _timing_config(**overrides)
        assert [p.phase for p in config.phase_configs] == expected_phases

    def test_injected_warmup_phase_shape(self) -> None:
        """The auto-warmup phase is a AGENT_GRAPH burst carrying the t* window."""
        config = _timing_config(
            trajectory_start_min_ratio=0.0, trajectory_start_max_ratio=1.0
        )
        warmup = config.phase_configs[0]
        assert warmup.phase == CreditPhase.WARMUP
        assert warmup.timing_mode == TimingMode.AGENT_GRAPH
        assert warmup.arrival_pattern == ArrivalPattern.CONCURRENCY_BURST
        assert warmup.trajectory_start_min_ratio == 0.0
        assert warmup.trajectory_start_max_ratio == 1.0
        assert warmup.grace_period_sec == float("inf")
        assert warmup.seamless is False


class TestGraphWarmupRuntime:
    """The injected phase's t* window primes a non-empty boundary graph."""

    def _plans(self, parsed: ParsedGraph, max_ratio: float) -> list[tuple[str, float]]:
        source = AgentGraphConversationSource(
            parsed=parsed,
            start_min_ratio=0.0,
            start_max_ratio=max_ratio,
            random_seed=SEED,
        )
        return [(gt.trace_id, gt.t_star_us) for gt in source.iter_traces()]

    @pytest.mark.parametrize(
        "max_ratio", [param(0.5, id="half"), param(1.0, id="full")]
    )
    def test_open_window_primes_every_trace(
        self, parsed: ParsedGraph, max_ratio: float
    ) -> None:
        plans = self._plans(parsed, max_ratio)
        # Exact, not just non-empty: _warmup_boundary_nodes is also the oracle
        # below, so a change to the boundary rule would move both sides of that
        # comparison together and the test would still pass. The fixture is
        # 5 nodes across two sessions; only sess_A's first node is on the
        # boundary at either ratio.
        assert [t for t, _ in plans] == ["sess_A"]
        for trace_id, t_star_us in plans:
            assert t_star_us > 0, trace_id
            warmup = rewrite_for_warmup(parsed, t_star_us)
            boundary = _warmup_boundary_nodes(parsed.graph, t_star_us)
            assert sorted(boundary) == ["sess_A:0"], trace_id
            assert set(warmup.graph.nodes) == set(boundary)
            # Flat + START-rooted: one in-edge per surviving boundary node.
            assert len(warmup.graph.edges) == len(warmup.graph.nodes)
            assert all(not n.inputs for n in warmup.graph.nodes.values())
            assert all(
                n.min_start_delay_us is None for n in warmup.graph.nodes.values()
            )

    def test_closed_window_yields_empty_warmup_graph(self, parsed: ParsedGraph) -> None:
        assert not rewrite_for_warmup(parsed, 0).graph.nodes


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        param("parent:0", "parent", id="colon-ordinal"),
        param("sess_A:2", "sess_A", id="underscore-inside-session-id"),
        param("sess_B:11", "sess_B", id="multi-digit-ordinal"),
        param("bare", "bare", id="no-delimiter-singleton"),
        param("weird:tail", "weird:tail", id="non-numeric-tail-singleton"),
    ],
)  # fmt: skip
def test_chain_key_splits_on_the_delimiter_the_lowering_mints(
    node_id: str, expected: str
) -> None:
    """Chain keys come from the ``:``-delimited turn ordinal, never from ``_``."""
    # Splitting on ``_`` made every colon-only id (``parent:0``) a singleton
    # chain -- so a live chain looked absent and t* warmup primed nothing --
    # while merging unrelated sessions that shared an underscore prefix.
    assert chain_key(node_id) == expected


def test_colon_only_corpus_yields_one_boundary_per_session() -> None:
    """A ``{session}:{k}`` corpus with no underscores still finds live chains."""
    nodes = {
        f"{sid}:{k}": LlmNode(
            prompt=[{"role": "user", "content": "q"}],
            output=f"{sid}_{k}_out",
            arrival_offset_us=k * 1000,
        )
        for sid in ("parent", "child")
        for k in range(3)
    }
    graph = GraphRecord(nodes=nodes)

    boundary = _warmup_boundary_nodes(graph, t_star_us=1500)

    assert set(boundary) == {"parent:1", "child:1"}
