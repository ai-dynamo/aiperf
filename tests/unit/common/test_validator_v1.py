# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.validators import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy


def _meta(conversations):
    return DatasetMetadata(
        conversations=conversations, sampling_strategy=DatasetSamplingStrategy.RANDOM
    )


def _convo(sid, *, branches=None, turns=None, is_root=True):
    # ``is_root`` is no longer a field on ``ConversationMetadata`` (it's
    # derived from ``agent_depth == 0``). The kw-arg here is preserved as
    # a test affordance: callers that want a non-root pass ``is_root=False``,
    # which translates to ``agent_depth=1``. The validator under test does
    # not consume agent_depth; it walks the topology by ``branches``, so
    # the actual integer value is immaterial as long as roots are 0.
    return ConversationMetadata(
        conversation_id=sid,
        turns=turns or [TurnMetadata()],
        branches=branches or [],
        agent_depth=0 if is_root else 1,
    )


def test_empty_dataset_passes():
    validate_for_orchestrator_v1(_meta([]))


def test_single_fork_passes():
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["root:0"])],
            ),
            _convo("c", is_root=False),
        ]
    )
    validate_for_orchestrator_v1(md)


def test_spawn_mode_rejected():
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            _convo("c", is_root=False),
        ]
    )
    with pytest.raises(NotImplementedError, match=r"branch mode .* not supported"):
        validate_for_orchestrator_v1(md)


def test_non_empty_prerequisites_rejected():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
    md = _meta([_convo("root", turns=[TurnMetadata(prerequisites=[p])])])
    with pytest.raises(NotImplementedError, match=r"prerequisites .* not supported"):
        validate_for_orchestrator_v1(md)


def test_duplicate_branch_id_within_turn_rejected():
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="b1",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["b1", "b1"])],
            ),
            _convo("c", is_root=False),
        ]
    )
    with pytest.raises(NotImplementedError, match=r"declared multiple times"):
        validate_for_orchestrator_v1(md)


def test_orphan_branch_id_reference_rejected():
    """A turn references a branch_id that's not declared in the conversation."""
    md = _meta(
        [_convo("root", branches=[], turns=[TurnMetadata(branch_ids=["ghost"])])]
    )
    with pytest.raises(NotImplementedError, match=r"references undeclared branch_id"):
        validate_for_orchestrator_v1(md)


def test_missing_child_conversation_rejected():
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=["ghost"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["root:0"])],
            )
        ]
    )
    with pytest.raises(NotImplementedError, match=r"does not reference"):
        validate_for_orchestrator_v1(md)


def test_empty_child_conversation_ids_rejected():
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=[],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["root:0"])],
            )
        ]
    )
    with pytest.raises(NotImplementedError, match=r"declares no child"):
        validate_for_orchestrator_v1(md)


def test_multi_fork_parent_rejected():
    md = _meta(
        [
            _convo(
                "p1",
                branches=[
                    ConversationBranchInfo(
                        branch_id="p1:0",
                        child_conversation_ids=["shared"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["p1:0"])],
            ),
            _convo(
                "p2",
                branches=[
                    ConversationBranchInfo(
                        branch_id="p2:0",
                        child_conversation_ids=["shared"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["p2:0"])],
            ),
            _convo("shared", is_root=False),
        ]
    )
    with pytest.raises(NotImplementedError, match=r"multiple FORK"):
        validate_for_orchestrator_v1(md)


def test_multi_root_forest_passes():
    """Multi-root files (independent root trees in one dataset) are valid:
    each root's tree is internally well-formed, so the orchestrator handles
    them independently.

    The old rejection clause was over-broad — it was masking a separate bug
    in ``DatasetManager._preformat_payloads`` that flipped storage to
    ``PAYLOAD_BYTES`` for all-single-turn fixtures, which dropped branch
    metadata and tripped the FORK-routing invariant at runtime. With that
    preformat path now skipped for FORK datasets (and the worker's
    cached ``Conversation`` retaining ``branches``), multi-root files run
    cleanly and the validator no longer needs to refuse them.
    """
    md = _meta(
        [
            _convo("r1"),
            _convo("r2"),
            _convo("r3"),
        ]
    )
    validate_for_orchestrator_v1(md)  # must not raise


def test_single_root_tree_with_children_still_passes():
    """Regression guard: the multi-root check must NOT fire on a
    valid single-root tree (1 root + N children)."""
    md = _meta(
        [
            _convo(
                "root",
                branches=[
                    ConversationBranchInfo(
                        branch_id="root:0",
                        child_conversation_ids=["c1", "c2"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
                turns=[TurnMetadata(branch_ids=["root:0"])],
            ),
            _convo("c1", is_root=False),
            _convo("c2", is_root=False),
        ]
    )
    validate_for_orchestrator_v1(md)  # must not raise
