# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-side trajectory-corr minting + refusal taxonomy.

The dispatch adapter mints one ``x_correlation_id`` per TRAJECTORY
(``{conversation_id}::{nonce}`` -- the legacy per-session HTTP identity;
routing keys on the instance ``trace_id``) and stamps it on every
``TurnToSend`` -- on BOTH issuer arms: an accepted issue parks a Future and
resolves on the credit return; a refusal surfaces as the typed
``CreditIssueRefusedError`` so the executor treats it as a trace-stop
instead of sentinel-continuing.
"""

import asyncio
from dataclasses import dataclass

import pytest

from aiperf.dataset.graph.graph_path_catalog import CatalogContext
from aiperf.graph.credit_dispatch_adapter import (
    CreditDispatchAdapter,
    CreditIssueRefusedError,
    GraphDispatchError,
)
from aiperf.graph.placement import DispatchRequest, PlacementContext


class _RefusingIssuer:
    """Records the turn, then refuses (stop gate / caps) like the real issuer."""

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def issue_graph_credit(self, turn: object) -> bool:
        self.sent.append(turn)
        return False


class _AcceptingIssuer(_RefusingIssuer):
    """Records the turn and ACCEPTS (credit placed on the wire)."""

    async def issue_graph_credit(self, turn: object) -> bool:
        self.sent.append(turn)
        return True


@dataclass
class _FakeCredit:
    """Minimal Credit-like carrying the correlation identity the bridge keys on."""

    x_correlation_id: str
    turn_index: int
    trace_id: str
    node_ordinal: int
    phase_variant: str = "profiling"


@dataclass
class _FakeLlmNode:
    output: str = "out"


def _adapter(issuer: _RefusingIssuer) -> CreditDispatchAdapter:
    return CreditDispatchAdapter(
        credit_issuer=issuer,
        catalog_context=CatalogContext(
            catalog={"t-1": {"t-1:0": 0}},
        ),
        trace_id="t-1",
        phase_variant="profiling",
    )


@pytest.mark.asyncio
async def test_refusal_raises_typed_error_and_turn_carries_trajectory_corr() -> None:
    issuer = _RefusingIssuer()
    adapter = _adapter(issuer)

    with pytest.raises(CreditIssueRefusedError):
        await adapter.dispatch(
            _FakeLlmNode(),
            DispatchRequest(node_id="t-1:0"),
            PlacementContext(parent_trace_id="t-1", parent_node_id="t-1:0"),
        )

    assert len(issuer.sent) == 1
    turn = issuer.sent[0]
    # Per-trajectory affinity: the corr is ``{conversation_id}::{nonce}``.
    assert turn.x_correlation_id.startswith("t-1::")
    assert adapter.phase_variant == "profiling"


@pytest.mark.asyncio
async def test_accepted_issue_mints_trajectory_corr_and_resolves() -> None:
    """The issuing arm (issuer returns True): same corr mint, no refusal.

    The A/B counterpart to the refusal test above -- the trajectory corr must
    ride the ACCEPTED turn too, and the dispatch must complete when the
    correlated credit return resolves the parked Future.
    """
    issuer = _AcceptingIssuer()
    adapter = _adapter(issuer)

    task = asyncio.create_task(
        adapter.dispatch(
            _FakeLlmNode(),
            DispatchRequest(node_id="t-1:0"),
            PlacementContext(parent_trace_id="t-1", parent_node_id="t-1:0"),
        )
    )
    await asyncio.sleep(0)

    assert len(issuer.sent) == 1
    turn = issuer.sent[0]
    assert turn.x_correlation_id.startswith("t-1::")

    adapter.resolve(
        _FakeCredit(
            x_correlation_id=turn.x_correlation_id,
            turn_index=turn.turn_index,
            trace_id=turn.trace_id,
            node_ordinal=turn.node_ordinal,
            phase_variant=turn.phase_variant,
        ),
        error=None,
        cancelled=False,
    )
    assert isinstance(await task, str)


def test_refusal_error_is_a_dispatch_error_subclass() -> None:
    """Existing except-GraphDispatchError sites keep catching refusals."""
    assert issubclass(CreditIssueRefusedError, GraphDispatchError)
