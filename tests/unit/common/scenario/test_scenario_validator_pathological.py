# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the AgentX scenario + DAG validators.

Two surfaces are attacked here, deliberately targeting holes the existing
suites in this directory and under ``tests/unit/common/`` do NOT cover:

``validate_for_orchestrator_v1`` (src/aiperf/common/validators/orchestrator_v1.py)
    The validator's stated contract is to "raise NotImplementedError for any
    construct v1 cannot honor" at load time, so misconfigurations surface
    before credit is issued. These probes feed it graph shapes that the
    runtime orchestrator silently mis-handles:

      * DAG cycles / self-spawn (a branch whose child resolves to an ancestor
        or to its own declaring conversation) — the orchestrator recurses
        ``agent_depth=parent_depth+1`` with no cap, so a cycle never
        terminates. The validator only checks child *existence*, not acyclicity.
      * duplicate ``branch_id`` across two ``ConversationBranchInfo`` objects in
        one conversation — both ``branches_by_id`` dict-comprehensions (validator
        and orchestrator) silently drop all but the last, so a gate references a
        different branch than the author wrote.
      * a ``branch_id`` declared in a turn's ``branch_ids`` with no matching
        ``ConversationBranchInfo`` object — a dangling declaration that the
        orchestrator's ``branches_by_id.get(b_id)`` quietly skips at spawn time.

    Confirmed gaps are asserted as the *correct* invariant and marked
    ``xfail(strict=True)`` citing the offending source line.

``validate_scenario`` (src/aiperf/common/scenario/validator.py)
    Coercion + gate edge cases the basic/adversarial/advanced suites skip:
    NaN ``ignore_eos`` truthiness, the list-of-tuples wire shape reaching the
    ``dict(raw)`` fallback, case-sensitive loader matching, and a negative
    ``--benchmark-duration`` slipping past the ``or 0.0`` short-circuit.

All tests are hermetic: real aiperf models, no network/services/sleep.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario import ScenarioLockError, validate_scenario
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode

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


def _user_config(
    *,
    scenario: str | None = "inferencex-agentx-mvp",
    timing_mode: TimingMode | str = TimingMode.AGENTIC_REPLAY,
    extra_inputs: dict | None = None,
    loader: str | None = "weka_trace",
    benchmark_duration: float | None = 900.0,
    unsafe_override: bool = False,
) -> MagicMock:
    """MagicMock UserConfig pre-shaped for ``validate_scenario`` (mirrors the
    fixtures in the sibling scenario test modules)."""
    cfg = MagicMock()
    cfg.scenario = scenario
    cfg.unsafe_override = unsafe_override
    cfg.timing_mode = timing_mode
    cfg.input.extra_inputs_parsed = (
        extra_inputs if extra_inputs is not None else {"ignore_eos": True}
    )
    cfg.input.use_think_time_only = True
    cfg.input.ignore_trace_delays = False
    cfg.input.random_seed = 42
    cfg.input.synthesis.max_isl = None
    cfg.input.detected_loader = loader
    cfg.input.public_dataset = None
    cfg.input.hf_weka_dataset = None
    cfg.loadgen.benchmark_duration = benchmark_duration
    cfg.loadgen.inter_turn_delay_cap_seconds = None
    cfg.loadgen.trace_idle_gap_cap_seconds = 10.0
    cfg.loadgen.concurrency = 10
    cfg.input.prompt.cache_bust.target = CacheBustTarget.FIRST_TURN_PREFIX
    cfg.input._use_think_time_only_explicitly_set = False
    cfg.loadgen._inter_turn_delay_cap_explicitly_set = False
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False
    cfg.loadgen.trajectory_start_min_ratio = 0.0
    cfg.loadgen.trajectory_start_max_ratio = 1.0
    cfg.loadgen._trajectory_start_min_ratio_explicitly_set = False
    cfg.loadgen._trajectory_start_max_ratio_explicitly_set = False
    cfg.input.prompt.cache_bust._target_explicitly_set = False
    return cfg


# =============================================================================
# orchestrator_v1: DAG cycles / self-spawn (CONFIRMED BUGS)
# =============================================================================


def test_validator_self_spawn_child_equals_parent_should_reject() -> None:
    """A branch whose only child is the conversation that declares it forms a
    self-cycle in a graph documented as a DAG. v1 cannot honor a cyclic graph,
    so the load-time validator should reject it."""
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
    """r -> c -> r is a directed cycle. The validator has no reachability /
    acyclicity pass, so it accepts a graph the orchestrator cannot terminate."""
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
# orchestrator_v1: duplicate branch_id objects (CONFIRMED BUG)
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
# orchestrator_v1: dangling branch_id reference (CONFIRMED BUG)
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
    branches (orchestrator_v1.py:199 ``if branch.mode != FORK: continue``).
    Two SPAWN branches in different conversations claiming the same child are
    accepted today. FORK children inherit one parent context (hence the guard),
    while SPAWN children start fresh, so multi-parent SPAWN is not flagged. This
    documents that asymmetry; it is not asserted to be correct, only current."""
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
    accepted. ``all_conversation_ids`` is a set (orchestrator_v1.py:47), so the
    collision dedups to one entry and child-existence still passes; the
    validator performs no uniqueness check on conversation_id. Pinning this
    because a runtime ConversationSource keyed by conversation_id would resolve
    a branch child to whichever duplicate it indexed last.

    NOTE: the branch targets a distinct ``leaf`` child. A duplicate id that
    *also* self-references (branch child == its own conversation_id) is now
    rejected by the spawn-graph acyclicity pass -- see
    test_validator_self_spawn_child_equals_parent_should_reject."""
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
    orchestrator ignores the stored value and assigns
    ``agent_depth=parent_depth+1`` (branch_orchestrator.py:612), so the dataset
    field is advisory only. Pinned so a future depth-consistency check would
    fail loudly here."""
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
    ``dispatch_timing='pre'`` path requires a declaring turn
    (orchestrator_v1.py:123-129); a regular SPAWN branch that no turn triggers
    is simply dead — never spawned at runtime — and the validator does not flag
    it. This is the inverse of the dangling-branch_id bug and is currently
    tolerated."""
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
# validate_scenario: coercion + gate edge cases (characterizations)
# =============================================================================


def test_scenario_ignore_eos_nan_treated_as_truthy_characterization() -> None:
    """CHARACTERIZATION: ``ignore_eos`` set to float NaN passes the lock.
    ``_is_truthy_extra_input`` (validator.py:55-62) returns ``bool(nan) == True``
    and ``_is_falsy_extra_input`` (lines 65-74) returns False because
    ``nan == 0`` is False, so neither the injection nor the violation path
    fires. A pathological NaN therefore reads as 'ignore_eos enabled'."""
    cfg = _user_config(extra_inputs={"ignore_eos": float("nan")})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_scenario_ignore_eos_float_zero_violates_characterization() -> None:
    """CHARACTERIZATION: ``ignore_eos=0.0`` (float, not int) is falsy via
    ``value == 0`` in ``_is_falsy_extra_input`` (validator.py:73) and triggers
    the lock — confirming float zero is treated identically to int zero, which
    the existing suites only cover for the int and string forms."""
    cfg = _user_config(extra_inputs={"ignore_eos": 0.0})
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc.value.violations)


def test_scenario_extra_inputs_list_of_tuples_falsy_eos_violates_characterization() -> (
    None
):
    """CHARACTERIZATION: the wire shape ``extra`` is ``list[tuple[str, Any]]``.
    When ``extra_inputs_parsed`` is None and ``extra`` is a list of tuples,
    ``_extract_extra_inputs`` reaches ``dict(raw)`` (validator.py:49-50), which
    coerces ``[('ignore_eos', False)]`` into ``{'ignore_eos': False}`` and the
    falsy value then violates the lock. The advanced suite only exercised the
    non-coercible (int 42) branch of this fallback; this pins the coercible
    list-of-tuples branch with a falsy payload."""
    cfg = _user_config(extra_inputs=None)
    cfg.input.extra_inputs_parsed = None
    cfg.input.extra = [("ignore_eos", False)]
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc.value.violations)


def test_scenario_loader_match_is_case_sensitive_characterization() -> None:
    """CHARACTERIZATION: loader matching is exact/case-sensitive. An
    upper-cased detected loader 'WEKA_TRACE' is NOT in the allowed lowercase
    tuple, so ``detected not in allowed`` (validator.py:257) fires a loader
    violation. Detected loader names are produced internally in lowercase, so
    this is benign in practice, but pins that the validator does no
    case-folding on the loader name (unlike the ConversationBranchMode enum)."""
    cfg = _user_config(loader="WEKA_TRACE")
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(v.flag == "--input-file (loader)" for v in exc.value.violations)


def test_scenario_negative_benchmark_duration_violates_characterization() -> None:
    """CHARACTERIZATION: a negative ``--benchmark-duration`` (-100s) is truthy,
    so the ``duration or 0.0`` short-circuit (validator.py:331) keeps -100,
    which is < the 900s floor and violates. The violation's current_value is
    the negative number verbatim — the validator clamps nothing, it just
    reports the floor breach. Confirms negatives are rejected rather than
    silently treated as 'unset' the way 0/None are."""
    cfg = _user_config(benchmark_duration=-100.0)
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    dur_violations = [
        v for v in exc.value.violations if v.flag == "--benchmark-duration"
    ]
    assert dur_violations
    assert dur_violations[0].current_value == -100.0
