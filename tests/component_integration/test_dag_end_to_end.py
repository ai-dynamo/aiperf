# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the DAG subagent pipeline.

This test does not spawn the full ``aiperf`` subprocess. Instead it loads the
``small.dag.jsonl`` fixture via the plugin-registered
``DagJsonlLoader``, wires a ``BranchOrchestrator`` directly against a real
``ConversationSource`` + fake credit issuer + fake sticky router, and drives
the orchestrator by fabricating a root credit-return. The goal is to exercise
the genuine orchestrator-intercept path end-to-end from fixture -> metadata ->
spawn + sticky-routing side-effects, using only live (non-mock) collaborators
for the dataset + DAG loader + orchestrator.

Validated invariants
--------------------
- Fixture loads: 3 conversations, root has 2 children (branchA, branchB), each
  child has 2 turns. ``is_root`` is set only on ``root``.
- ``BranchOrchestrator.intercept(root_credit)`` returns ``True`` (short-circuits
  the default strategy dispatch) and triggers dispatch of both children.
- ``BranchStats`` post-spawn: ``children_spawned == 2``,
  ``children_completed == 0``, ``children_errored == 0``,
  ``parents_suspended == 0`` (no join turn in this topology).
- Sticky router refcount bumps by +2 on the parent's correlation id so both
  children route to the parent's worker (locality invariant).
- The fake credit issuer receives first-turn dispatches for both children with
  ``agent_depth == 1`` and ``parent_correlation_id == parent_corr``.

End-to-end cross-process validation (full ``aiperf`` subprocess run, request
transcript capture, sticky-routing assertion from ``profile_export*.json``)
is deferred because the mock server does not currently capture per-request
transcripts and the exporter does not currently emit ``branch_stats`` into
``profile_export_aiperf.json``. Both are separate gaps to close before a
full E2E assertion pass is possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationContextMode, CreditPhase
from aiperf.common.models import DatasetMetadata
from aiperf.credit.structs import Credit
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.conversation_source import ConversationSource, SampledSession

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dag" / "small.dag.jsonl"


# --- Fakes -----------------------------------------------------------------


@dataclass
class _FakeIssuer:
    dispatched: list[SampledSession] = field(default_factory=list)

    async def dispatch_first_turn(self, session: SampledSession) -> bool:
        self.dispatched.append(session)
        return True

    async def dispatch_join_turn(self, parent_corr: str, join_turn_index: int) -> None:
        raise AssertionError(
            "No-join topology should not call dispatch_join_turn "
            f"(parent={parent_corr}, idx={join_turn_index})"
        )


@dataclass
class _FakeStickyRouter:
    registers: list[str] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)

    def register_child_routing(self, parent_corr: str) -> None:
        self.registers.append(parent_corr)

    def release_child_routing(self, parent_corr: str) -> None:
        self.releases.append(parent_corr)


# --- Helpers ---------------------------------------------------------------


def _build_metadata(loader: DagJsonlLoader) -> DatasetMetadata:
    """Project loaded conversations to a DatasetMetadata analogous to the
    real DatasetManager pipeline."""
    conversations = loader.load()
    return DatasetMetadata(
        conversations=[c.metadata() for c in conversations],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
        default_context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
    )


class _IdentitySampler:
    """Trivial sampler that cycles through provided ids; unused here but satisfies
    ``ConversationSource`` construction invariants."""

    def __init__(self, conversation_ids: list[str]) -> None:
        self._ids = list(conversation_ids)
        self._i = 0

    def next_conversation_id(self) -> str:
        cid = self._ids[self._i % len(self._ids)]
        self._i += 1
        return cid


# --- Tests -----------------------------------------------------------------


@pytest.mark.component_integration
class TestDagEndToEndSmoke:
    """Smoke-level end-to-end DAG validation through the orchestrator seam."""

    def test_fixture_loads_and_declares_expected_topology(self) -> None:
        loader = DagJsonlLoader(FIXTURE)
        conversations = {c.session_id: c for c in loader.load()}

        assert set(conversations) == {"root", "branchA", "branchB"}

        root = conversations["root"]
        assert root.is_root is True
        assert len(root.turns) == 1
        assert len(root.branches) == 1
        assert root.branches[0].child_conversation_ids == ["branchA", "branchB"]
        assert root.turns[0].branch_ids == [root.branches[0].branch_id]
        assert root.context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES

        for child_id in ("branchA", "branchB"):
            child = conversations[child_id]
            assert child.is_root is False
            assert len(child.turns) == 2
            assert child.branches == []

    @pytest.mark.asyncio
    async def test_orchestrator_spawns_both_children_with_sticky_locality(self) -> None:
        """Fabricate a root credit-return and drive the orchestrator.

        Asserts:
        - intercept returns True (short-circuits strategy dispatch).
        - Both children dispatched via issuer.dispatch_first_turn, each with
          agent_depth=1 and parent_correlation_id=root_corr.
        - BranchStats.children_spawned == 2, errored == 0, suspended == 0.
        - Sticky router received 2 register_child_routing calls for the parent
          (locality invariant).
        """
        loader = DagJsonlLoader(FIXTURE)
        dataset_metadata = _build_metadata(loader)

        sampler = _IdentitySampler(
            [c.conversation_id for c in dataset_metadata.conversations if c.is_root]
        )
        conv_source = ConversationSource(dataset_metadata, sampler)

        issuer = _FakeIssuer()
        sticky = _FakeStickyRouter()
        orch = BranchOrchestrator(
            conversation_source=conv_source,
            credit_issuer=issuer,
            sticky_router=sticky,
        )

        # Fabricate the root's turn-0 completion credit (the would-be credit
        # return from the worker after the root's first (and only) turn).
        root_corr = "root-corr-xyz"
        root_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="root",
            x_correlation_id=root_corr,
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
            agent_depth=0,
            parent_correlation_id=None,
        )

        intercepted = await orch.intercept(root_credit)

        # Phase 1: intercept returns True only when the parent's next turn
        # is gated. This fixture has no join turn -> parent may continue ->
        # intercept returns False.
        assert intercepted is False
        assert orch.stats.children_spawned == 2
        assert orch.stats.children_completed == 0
        assert orch.stats.children_errored == 0
        assert orch.stats.parents_suspended == 0, (
            "This topology has no join turn, so no parent suspension"
        )

        dispatched_convs = {s.conversation_id for s in issuer.dispatched}
        assert dispatched_convs == {"branchA", "branchB"}
        for session in issuer.dispatched:
            assert session.agent_depth == 1
            assert session.parent_correlation_id == root_corr
            assert session.routing_key == root_corr

        assert sticky.registers == [root_corr, root_corr], (
            "Parent's sticky refcount must bump by +2 so both children pin to "
            "the parent's worker."
        )
        assert sticky.releases == [], "Releases only happen on child leaf completion"

    @pytest.mark.asyncio
    async def test_spawned_children_share_root_tree_marker(self) -> None:
        """The whole trajectory TREE shares ONE cache-bust marker.

        Spawned descendants (subagents / flat agents) must carry their
        tree-root's marker (resolved by ``root_correlation_id`` from the shared
        ledger), not a per-child marker that would fragment the tree's
        prefix-cache domain. Pre-fix each child minted its own digest of its
        child conversation id; here every dispatched child carries the seeded
        root marker verbatim.
        """
        from aiperf.common.enums import CacheBustTarget
        from aiperf.timing.trajectory_source import CacheBustLedger

        loader = DagJsonlLoader(FIXTURE)
        dataset_metadata = _build_metadata(loader)
        sampler = _IdentitySampler(
            [c.conversation_id for c in dataset_metadata.conversations if c.is_root]
        )
        conv_source = ConversationSource(dataset_metadata, sampler)

        issuer = _FakeIssuer()
        sticky = _FakeStickyRouter()
        root_corr = "root-corr-tree"
        root_marker = "[rid:deadbeefcafe]\n\n"
        ledger = CacheBustLedger()
        ledger.session_marker[root_corr] = root_marker

        orch = BranchOrchestrator(
            conversation_source=conv_source,
            credit_issuer=issuer,
            sticky_router=sticky,
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
            cache_bust_ledger=ledger,
        )

        root_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="root",
            x_correlation_id=root_corr,
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
            agent_depth=0,
            parent_correlation_id=None,
        )

        await orch.intercept(root_credit)

        assert orch.stats.children_spawned == 2
        assert {s.conversation_id for s in issuer.dispatched} == {"branchA", "branchB"}
        for session in issuer.dispatched:
            assert session.cache_bust_marker == root_marker, (
                f"child {session.conversation_id} must carry the tree-root "
                f"marker {root_marker!r}, got {session.cache_bust_marker!r}"
            )

    @pytest.mark.asyncio
    async def test_orchestrator_completes_after_both_children_reach_leaf(self) -> None:
        """Drive both children to their leaf terminations and verify the
        sticky-routing refcount drains and children_completed advances."""
        loader = DagJsonlLoader(FIXTURE)
        dataset_metadata = _build_metadata(loader)
        sampler = _IdentitySampler(
            [c.conversation_id for c in dataset_metadata.conversations if c.is_root]
        )
        conv_source = ConversationSource(dataset_metadata, sampler)

        issuer = _FakeIssuer()
        sticky = _FakeStickyRouter()
        orch = BranchOrchestrator(
            conversation_source=conv_source,
            credit_issuer=issuer,
            sticky_router=sticky,
        )

        root_corr = "root-corr-abc"
        root_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="root",
            x_correlation_id=root_corr,
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
        )
        await orch.intercept(root_credit)

        # Children's x_correlation_ids were assigned by start_branch_child
        child_corrs = [s.x_correlation_id for s in issuer.dispatched]
        assert len(child_corrs) == 2

        # Simulate both children reaching a leaf turn (as would happen when the
        # worker returns their final credit and the leaf-reach seam fires).
        for cc in child_corrs:
            await orch.on_child_leaf_reached(cc)

        assert orch.stats.children_completed == 2
        assert orch.stats.children_errored == 0
        # Sticky refcount released once per child.
        assert sticky.releases == [root_corr, root_corr]


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_orchestrator_runs_when_issuer_dispatch_fails_gracefully() -> None:
    """Regression: a child-dispatch failure must bump children_errored without
    crashing the orchestrator (asyncio.gather(..., return_exceptions=True))."""
    loader = DagJsonlLoader(FIXTURE)
    dataset_metadata = _build_metadata(loader)
    sampler = _IdentitySampler(
        [c.conversation_id for c in dataset_metadata.conversations if c.is_root]
    )
    conv_source = ConversationSource(dataset_metadata, sampler)

    class _ExplodingIssuer:
        async def dispatch_first_turn(self, session: SampledSession) -> bool:
            raise RuntimeError("synthetic dispatch failure")

    sticky = _FakeStickyRouter()
    orch = BranchOrchestrator(
        conversation_source=conv_source,
        credit_issuer=_ExplodingIssuer(),
        sticky_router=sticky,
    )

    root_credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="root-corr-err",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
    )

    # Should NOT raise; gather swallows individual task exceptions.
    intercepted = await orch.intercept(root_credit)

    # Phase 1: topology fixture has no join -> parent not suspended ->
    # intercept returns False. The explode path still rolls back
    # bookkeeping cleanly.
    assert intercepted is False
    # Both child sessions were created (spawn_id booked) before dispatch
    # attempted; when dispatch raises the orchestrator rolls back the
    # children_spawned increment and bumps children_errored.
    assert orch.stats.children_spawned == 0
    assert orch.stats.children_errored == 2


@pytest.mark.component_integration
def test_dataset_total_turn_count_matches_fixture() -> None:
    """Regression: the fixture declares exactly 5 turns (1 root + 2 + 2)."""
    loader = DagJsonlLoader(FIXTURE)
    metadata = _build_metadata(loader)
    assert metadata.total_turn_count == 5
    assert len(metadata.conversations) == 3
    assert sum(1 for c in metadata.conversations if c.is_root) == 1
