# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the byte-exact replay-barrier-coordinator golden for the Rust parity test.

This drives the **real** Python interval-barrier coordinator
``aiperf.timing.replay_dependencies.ReplayBarrierCoordinator`` through a fixed,
deterministic scripted scenario over a small recorded predecessor graph spanning
three runtime roots (including a cross-stream, join-width-2 barrier), and records:

    * ``release_order``     -- the exact order keys were issued (release order),
    * ``completed_prefixes``-- ``completed_prefixes(root)`` at labelled checkpoints,
    * ``pending_turns_by_root`` -- a ``pending_turns_by_root()`` snapshot.

The coordinator's ``submit``/``complete`` are async and dispatch releases through
detached ``asyncio`` tasks, so we drive it under ``asyncio.run`` with an ``issue``
callback that appends ``(conversation_id, turn_index)`` and returns ``True``, and
``await asyncio.sleep(0)`` after each mutation so released tasks settle before the
next scripted op / snapshot.

The Rust ``ReplayGate`` is a single-central-driver port: it models a "release" as
synchronously pushing the key onto an ordered output list inside ``complete`` (no
asyncio), and must reproduce ``release_order`` / ``completed_prefixes`` /
``pending_turns_by_root`` byte-for-byte.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.models.dataset_models import ReplayTurnReference
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import (
    ReplayBarrierCoordinator,
    ReplayResumeBoundary,
)

# --- Runtime roots -----------------------------------------------------------
R1 = "root-1"
R2 = "root-2"
R3 = "root-3"


@dataclass(frozen=True)
class _Turn:
    """Minimal ``TurnToSend`` stand-in exposing exactly what the coordinator reads."""

    root: str
    conversation_id: str
    turn_index: int

    @property
    def effective_root_correlation_id(self) -> str:
        return self.root


@dataclass(frozen=True)
class _Credit:
    """Minimal ``Credit`` stand-in exposing exactly what ``complete`` reads."""

    root: str
    conversation_id: str
    turn_index: int

    @property
    def effective_root_correlation_id(self) -> str:
        return self.root


def _ref(conversation_id: str, turn_index: int) -> ReplayTurnReference:
    return ReplayTurnReference(conversation_id=conversation_id, turn_index=turn_index)


def build_dataset() -> DatasetMetadata:
    """A fixed recorded predecessor graph.

    R1 tree:
        A0 (no preds), A1 (no preds)
        B0 depends on [A0]                       (cross-stream edge)
        C0 depends on [A1, B0]                    (join width 2)
    R3 tree:
        F0 (no preds)
        G0 depends on [F0]
    (R2 carries no predecessor turns; it exercises seed/prefix only.)
    """
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="A",
                turns=[TurnMetadata(), TurnMetadata()],
            ),
            ConversationMetadata(
                conversation_id="B",
                turns=[TurnMetadata(replay_predecessors=[_ref("A", 0)])],
            ),
            ConversationMetadata(
                conversation_id="C",
                turns=[
                    TurnMetadata(
                        replay_predecessors=[_ref("A", 1), _ref("B", 0)]
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id="F",
                turns=[TurnMetadata()],
            ),
            ConversationMetadata(
                conversation_id="G",
                turns=[TurnMetadata(replay_predecessors=[_ref("F", 0)])],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


async def run() -> dict:
    coordinator = ReplayBarrierCoordinator(build_dataset())

    released: list[list] = []

    def make_issue(conversation_id: str, turn_index: int):
        async def issue() -> bool:
            released.append([conversation_id, turn_index])
            return True

        return issue

    async def submit(turn: _Turn) -> None:
        await coordinator.submit(
            turn, make_issue(turn.conversation_id, turn.turn_index)
        )
        await asyncio.sleep(0)

    def complete(credit: _Credit) -> None:
        coordinator.complete(credit)

    async def settle() -> None:
        # Let detached release tasks created by complete() run to completion.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    coordinator.activate()

    # --- R1: cross-stream barrier release chain (unpaused) -------------------
    # A0 ready immediately; A1 ready immediately; B0 retained (needs A0);
    # C0 retained (needs A1 + B0).
    await submit(_Turn(R1, "A", 0))
    await submit(_Turn(R1, "B", 0))
    await submit(_Turn(R1, "C", 0))
    await submit(_Turn(R1, "A", 1))

    # complete A0 -> releases B0.
    complete(_Credit(R1, "A", 0))
    await settle()
    # complete A1 -> C0 still needs B0 completion; nothing releases.
    complete(_Credit(R1, "A", 1))
    await settle()
    # complete B0 -> C0 now ready -> releases C0.
    complete(_Credit(R1, "B", 0))
    await settle()

    r1_prefixes = coordinator.completed_prefixes(R1)

    # --- R2: seed a resume prefix, then a ready submit -----------------------
    coordinator.seed_completed_prefixes(
        R2, (ReplayResumeBoundary("D", 2),)
    )
    r2_prefixes = coordinator.completed_prefixes(R2)

    # --- R3: pause then submit; newly-ready work is retained -----------------
    coordinator.pause_releases()
    await submit(_Turn(R3, "F", 0))  # ready but paused -> retained
    await submit(_Turn(R3, "G", 0))  # not ready -> retained
    complete(_Credit(R3, "F", 0))  # paused -> no release even though G0 ready
    await settle()

    pending_by_root = coordinator.pending_turns_by_root()

    def dump_prefixes(prefixes) -> list[list]:
        return [[b.conversation_id, b.next_turn_index] for b in prefixes]

    def dump_pending(turns) -> list[list]:
        return [[t.conversation_id, t.turn_index] for t in turns]

    return {
        "release_order": released,
        "completed_prefixes": {
            "r1_final": dump_prefixes(r1_prefixes),
            "r2_after_seed": dump_prefixes(r2_prefixes),
        },
        "pending_turns_by_root": {
            root: dump_pending(turns)
            for root, turns in sorted(pending_by_root.items())
        },
        "_provenance": (
            "aiperf.timing.replay_dependencies.ReplayBarrierCoordinator driven "
            "through a fixed scripted submit/complete/seed/pause scenario"
        ),
    }


def main() -> None:
    out = asyncio.run(run())
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
