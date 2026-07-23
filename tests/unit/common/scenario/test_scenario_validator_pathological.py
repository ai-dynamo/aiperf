# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the AgentX scenario + DAG validators.

Two surfaces are attacked here.

``validate_for_orchestrator_v1`` (src/aiperf/common/validators/orchestrator_v1.py)
    The validator's contract is to reject (``NotImplementedError`` /
    ``ValueError``) any construct the v1 BranchOrchestrator cannot honor at
    load time, so misconfigurations surface before credit is issued. These
    probes feed it graph shapes the orchestrator would mis-handle:

      * DAG cycles / self-spawn (rejected by ``_assert_spawn_graph_acyclic``
        with "spawn graph contains a cycle").
      * dangling ``branch_id`` declared in a turn with no matching
        ``ConversationBranchInfo`` (rejected by the prereq / branch checks).
      * duplicate ``branch_id`` across two branch descriptors (the v1
        ``branches_by_id`` dict-comprehension drops one).

    These tests use REAL aiperf models and port cleanly to v2.

``apply_scenario`` (src/aiperf/common/scenario/validator.py)
    Coercion + gate edge cases the basic/adversarial/advanced suites skip:
    NaN / float-zero ``ignore_eos`` truthiness, case-sensitive loader
    matching, and a negative ``--benchmark-duration`` slipping past the
    ``or 0.0`` short-circuit. Rebased from the v1 MagicMock UserConfig onto a
    REAL ``BenchmarkRun``.

DROP NOTE (v1 behavior with no v2 analog):
* ``test_scenario_extra_inputs_list_of_tuples_falsy_eos_violates`` exercised
  the v1 ``_extract_extra_inputs`` ``dict(raw)`` fallback (parsed -> extra ->
  dict(list-of-tuples)). v2 reads ``endpoint.extra`` (a real dict) directly;
  the fallback chain is gone, so this test is dropped. The falsy-value
  semantics it pinned are still covered by the float/string falsy tests here
  and in the sibling modules.

All tests are hermetic: real aiperf models, no network/services/sleep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.plugin.enums import DatasetSamplingStrategy

_WEKA_LOADER = "semianalysis_cc_traces_weka_with_subagents"

# =============================================================================
# Shared builders
# =============================================================================


def _child(cid: str, **kw) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()], **kw)


def _dataset(*convs: ConversationMetadata) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=list(convs),
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _build_run(
    *,
    scenario: str | None = "inferencex-agentx-mvp",
    dataset: dict[str, Any] | None = None,
    streaming: bool | None = True,
    extra: dict[str, Any] | None = None,
    duration: Any = 1800,
    detected_loader: str | None = None,
) -> BenchmarkRun:
    """Construct a BenchmarkRun for the scenario (mirrors the sibling modules).

    Defaults to the public weka dataset; pass ``dataset`` for a custom dict.
    ``detected_loader`` populates ``run.resolved.dataset_types`` (only relevant
    for FileDataset loader detection).
    """
    if dataset is None:
        dataset = {"name": "main", "type": "public", "dataset": _WEKA_LOADER}
    if extra is None:
        extra = {"ignore_eos": True}
    endpoint: dict[str, Any] = {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "type": "chat",
        "extra": extra,
    }
    if streaming is not None:
        endpoint["streaming"] = streaming

    body: dict[str, Any] = {
        "models": ["my-model"],
        "endpoint": endpoint,
        "datasets": [dataset],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 8,
                "duration": duration,
            }
        ],
    }
    if scenario is not None:
        body["scenario"] = scenario

    cfg = BenchmarkConfig.model_validate(body)
    run = BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=Path("/tmp/aiperf-scenario-test"),
    )
    if detected_loader is not None:
        run.resolved.dataset_types = {"main": detected_loader}
    return run


# =============================================================================
# orchestrator_v1: DAG cycles / self-spawn
# =============================================================================


def test_validator_self_spawn_child_equals_parent_should_reject() -> None:
    """A branch whose only child is the conversation that declares it forms a
    self-cycle in a graph documented as a DAG. v1 cannot honor a cyclic graph,
    so the load-time validator rejects it."""
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["r"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
            branches=[branch],
        )
    )
    with pytest.raises(NotImplementedError):
        validate_for_orchestrator_v1(md)


def test_validator_two_node_spawn_cycle_should_reject() -> None:
    """r -> c -> r is a directed cycle the v1 orchestrator cannot terminate."""
    b_r = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    b_c = ConversationBranchInfo(
        branch_id="c:0",
        child_conversation_ids=["r"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
            branches=[b_r],
        ),
        ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata(branch_ids=["c:0"]), TurnMetadata()],
            branches=[b_c],
        ),
    )
    with pytest.raises(NotImplementedError):
        validate_for_orchestrator_v1(md)


# =============================================================================
# orchestrator_v1: duplicate branch_id objects
# =============================================================================


def test_validator_duplicate_branch_id_across_branch_objects_should_reject() -> None:
    """Declaring two branches with the same branch_id but different children is
    an authoring bug: the dict-comprehension drops one and its children are
    never dispatched. The validator should reject the collision."""
    b1 = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    b2 = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c2"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
            branches=[b1, b2],
        ),
        _child("c1"),
        _child("c2"),
    )
    with pytest.raises((NotImplementedError, ValueError)):
        validate_for_orchestrator_v1(md)


# =============================================================================
# orchestrator_v1: dangling branch_id reference
# =============================================================================


def test_validator_dangling_branch_id_in_turn_should_reject() -> None:
    """Turn 0 declares branch_ids=['ghost'] but the conversation has no branch
    descriptor for 'ghost'. This dangling reference is an authoring error the
    load-time validator should surface, not silently drop at spawn time."""
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(branch_ids=["ghost"]), TurnMetadata()],
            branches=[],
        )
    )
    with pytest.raises(NotImplementedError):
        validate_for_orchestrator_v1(md)


# =============================================================================
# orchestrator_v1: characterizations of surprising-but-current behavior
# =============================================================================


def test_validator_two_spawn_parents_same_child_accepted_characterization() -> None:
    """CHARACTERIZATION: the global single-parent guard only fires for FORK
    branches. Two SPAWN branches in different conversations claiming the same
    child are accepted today. SPAWN children start fresh (no inherited parent
    context), so multi-parent SPAWN is not flagged."""
    b1 = ConversationBranchInfo(
        branch_id="r1:0",
        child_conversation_ids=["shared"],
        mode=ConversationBranchMode.SPAWN,
    )
    b2 = ConversationBranchInfo(
        branch_id="r2:0",
        child_conversation_ids=["shared"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r1",
            turns=[TurnMetadata(branch_ids=["r1:0"])],
            branches=[b1],
        ),
        ConversationMetadata(
            conversation_id="r2",
            turns=[TurnMetadata(branch_ids=["r2:0"])],
            branches=[b2],
        ),
        _child("shared"),
    )
    validate_for_orchestrator_v1(md)


def test_validator_duplicate_conversation_id_accepted_characterization() -> None:
    """CHARACTERIZATION: two conversations sharing one conversation_id are
    accepted. ``all_conversation_ids`` is a set, so the collision dedups and
    child-existence still passes; the validator performs no uniqueness check on
    conversation_id. The branch targets a distinct ``leaf`` child (a duplicate
    id that also self-references would be rejected by the acyclicity pass)."""
    branch = ConversationBranchInfo(
        branch_id="dup:0",
        child_conversation_ids=["leaf"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="dup",
            turns=[TurnMetadata(branch_ids=["dup:0"]), TurnMetadata()],
            branches=[branch],
        ),
        _child("dup"),
        _child("leaf"),
    )
    validate_for_orchestrator_v1(md)


def test_validator_child_agent_depth_shallower_than_parent_accepted_characterization() -> (
    None
):
    """CHARACTERIZATION: a child conversation whose stored ``agent_depth`` (0)
    is shallower than its spawning parent's (5) is accepted. The validator
    never cross-checks parent vs child agent_depth — at spawn time the
    orchestrator assigns ``agent_depth=parent_depth+1``, so the dataset field is
    advisory only."""
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
            branches=[branch],
            agent_depth=5,
        ),
        _child("c", agent_depth=0, parent_conversation_id="r"),
    )
    validate_for_orchestrator_v1(md)


def test_validator_orphan_branch_never_declared_by_turn_accepted_characterization() -> (
    None
):
    """CHARACTERIZATION: a non-background post-dispatch branch descriptor whose
    branch_id is never listed in any turn's branch_ids is accepted. Only the
    ``dispatch_timing='pre'`` path requires a declaring turn; a regular SPAWN
    branch that no turn triggers is simply dead — never spawned at runtime —
    and the validator does not flag it."""
    branch = ConversationBranchInfo(
        branch_id="orphan",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _dataset(
        ConversationMetadata(
            conversation_id="r",
            turns=[TurnMetadata(), TurnMetadata()],
            branches=[branch],
        ),
        _child("c"),
    )
    validate_for_orchestrator_v1(md)


# =============================================================================
# apply_scenario: coercion + gate edge cases (characterizations)
# =============================================================================


def test_scenario_ignore_eos_nan_treated_as_truthy_characterization() -> None:
    """CHARACTERIZATION: ``ignore_eos`` set to float NaN passes the lock.
    ``_is_falsy_extra_input`` returns False because ``nan == 0`` is False, so
    neither the injection nor the violation path fires — a pathological NaN
    reads as 'ignore_eos enabled'."""
    run = _build_run(extra={"ignore_eos": float("nan")})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_scenario_ignore_eos_float_zero_violates_characterization() -> None:
    """CHARACTERIZATION: ``ignore_eos=0.0`` (float, not int) is falsy via
    ``value == 0`` in ``_is_falsy_extra_input`` and trips the lock — float zero
    is treated identically to int zero."""
    run = _build_run(extra={"ignore_eos": 0.0})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc.value.violations)


def test_scenario_loader_match_is_case_sensitive_characterization() -> None:
    """CHARACTERIZATION: ``_detect_loader`` does no case-folding. For a
    FileDataset, the detected loader is the raw ``run.resolved.dataset_types``
    value; an upper-cased 'WEKA_TRACE' is not in the allowed lowercase tuple,
    so ``detected not in allowed`` fires a loader violation. (PublicDataset
    loaders can't exercise this — ``PublicDatasetType`` case-folds the enum
    value at config-build, so only the free-form FileDataset path can carry a
    differently-cased loader name to the lock.)"""
    run = _build_run(
        dataset={
            "name": "main",
            "type": "file",
            "format": "mooncake_trace",
            "records": [
                {
                    "timestamp": 0,
                    "input_length": 10,
                    "output_length": 5,
                    "hash_ids": [1],
                }
            ],
            "cache_bust": {"target": "first_turn_prefix"},
        },
        detected_loader="WEKA_TRACE",
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(v.flag == "--input-file (loader)" for v in exc.value.violations)


def test_scenario_negative_benchmark_duration_violates_characterization() -> None:
    """CHARACTERIZATION: a negative ``--benchmark-duration`` (-100s) is truthy,
    so the ``duration or 0.0`` short-circuit keeps -100, which is < the 900s
    floor and violates. The violation's current_value is the negative number
    verbatim — the validator clamps nothing, it just reports the floor
    breach."""
    # phase.duration has gt=0 at config-build; build valid then set the
    # negative value directly on the phase to reach the lock.
    run = _build_run()
    run.cfg.get_profiling_phases()[0].duration = -100.0
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    dur_violations = [
        v for v in exc.value.violations if v.flag == "--benchmark-duration"
    ]
    assert dur_violations
    assert dur_violations[0].current_value == -100.0
