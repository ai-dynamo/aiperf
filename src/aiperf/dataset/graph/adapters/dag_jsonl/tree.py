# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-topology middle layer for the ``dag_jsonl`` graph adapter.

Reuses the legacy ``dag_jsonl`` loader output (``Conversation`` objects with
``branches``, per-turn ``prerequisites`` / ``branch_ids``, and ``is_root``) and
expands each root conversation into an instanced tree of :class:`DagNodeSpec`s.
Each node captures its lineage (message-context ancestors), completion-anchored
firing predecessors with microsecond delays, and SPAWN_JOIN fan-in gates. No
graph IR types are produced here; :mod:`.lowering` lowers these trees onto the
graph IR.

All dag.md load-time validation (cycles, single-parent FORK, system-message
placement, join_at bounds) is delegated to the legacy loader — this module
does NOT re-validate. The only additional gate is that a session id must not
contain the framing characters used by the graph correlation-id scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader


@dataclass(slots=True)
class DagNodeSpec:
    """One dispatchable turn instance inside one root tree."""

    node_id: str
    """Unique per tree: ``"<instance>:<turn_idx>"``; ``<instance>`` is the
    session_id, suffixed ``"#<n>"`` (n>=2) for repeated SPAWN instantiations of
    one template."""
    turn: Turn
    """The legacy Turn (raw_messages/raw_tools/model/max_tokens/extra_body/delay)."""
    lineage: list[str]
    """node_ids whose (messages + live reply) precede this node's own messages,
    oldest first. FORK child turn 0: parent's full lineage + the forking parent
    node. SPAWN child turn 0 and roots: []. Turn k>0: lineage(k-1) + [k-1's id]."""
    predecessors: list[tuple[str, int]]
    """(pred node_id, delay_us) completion-anchored firing deps. Sequential edge
    carries this turn's authored delay (ms -> us); fork/spawn child turn-0 edges
    carry 0 (legacy dispatches children immediately at parent credit return, and
    root turn-0 delay is NOT honored by the legacy rate loop - both preserved)."""
    join_inputs: list[str] = field(default_factory=list)
    """Leaf node_ids of SPAWN branches gating this turn (SPAWN_JOIN fan-in)."""
    agent_depth: int = 0
    """Legacy record identity: 0 for root-chain turns, owner depth + 1 for
    FORK/SPAWN child instances, 1 for pre-session spawn children (mirrors
    ``conversation_source.start_branch_child`` / ``start_pre_session_child``).
    Every turn of one child instance carries the instance's depth."""
    parent_node_id: str | None = None
    """The triggering parent node (the fork/spawn turn) for child instances;
    None for roots AND pre-session children (legacy: no parent session exists
    at pre-dispatch). Same value on every turn of the instance."""


@dataclass(slots=True)
class DagTree:
    """All instanced turn nodes reachable from one root conversation."""

    trace_id: str
    """Root session_id; namespaces this tree's node_ids."""
    nodes: dict[str, DagNodeSpec]
    """node_id -> spec, insertion-ordered by a deterministic depth-first walk."""


def load_dag_conversations(
    path: Path, *, delay_cap_seconds: float | None
) -> dict[str, list[Conversation]]:
    """Load a ``dag_jsonl`` file into the loader's session_id -> [Conversation] map.

    Wraps the legacy :class:`DagJsonlLoader` standalone construction so later
    tasks never touch loader internals. The standalone constructor hard-codes
    the delay cap to disabled (it has no ``run`` to read it from), so when a cap
    is requested we replace the tracker before the first ``load_dataset()`` call
    parses turns and applies the clamp.
    """
    loader = DagJsonlLoader(path)
    if delay_cap_seconds is not None:
        loader._delay_cap_tracker = DelayCapTracker(cap_seconds=delay_cap_seconds)
    return loader.load_dataset()


def expand_trees(conversations: dict[str, list[Conversation]]) -> list[DagTree]:
    """Expand each root conversation into an instanced :class:`DagTree`.

    Roots are the conversations with ``is_root=True``, walked in the loader's
    declaration (file) order; each yields one tree. SPAWN children are
    instantiated fresh per reference, FORK children exactly once (single-parent
    guaranteed by the loader).
    """
    conv_by_sid: dict[str, Conversation] = {}
    for convs in conversations.values():
        for conv in convs:
            conv_by_sid[conv.session_id] = conv

    for sid in conv_by_sid:
        if "|" in sid or "#" in sid:
            raise NotImplementedError(
                f"conversation '{sid}': session ids containing '|' or '#' "
                "collide with graph correlation-id framing"
            )

    trees: list[DagTree] = []
    for convs in conversations.values():
        for conv in convs:
            if conv.is_root:
                trees.append(_expand_tree(conv, conv_by_sid))
    return trees


def _resolve_join_inputs(
    *, turn: Turn, branch_leaves: dict[str, list[str]]
) -> list[str]:
    """SPAWN_JOIN fan-in leaves gating ``turn``.

    Each SPAWN_JOIN prerequisite naming an already-fired branch contributes
    that branch's post-spawn child-instance leaves, in prerequisite order.
    Branches that have not fired yet resolve to no leaves.
    """
    join_inputs: list[str] = []
    for prereq in turn.prerequisites:
        if prereq.kind == PrerequisiteKind.SPAWN_JOIN and prereq.branch_id is not None:
            join_inputs.extend(branch_leaves.get(prereq.branch_id, []))
    return join_inputs


def _child_entry_context(
    *,
    branch: ConversationBranchInfo,
    is_pre: bool,
    node_id: str,
    lineage: list[str],
    predecessors: list[tuple[str, int]],
) -> tuple[list[str], list[tuple[str, int]]]:
    """Turn-0 seed ``(lineage, predecessors)`` for a child branch.

    Pre-session spawns fire on the owner turn-0's own trigger, so they copy its
    predecessors and start context-free. Forks inherit the parent's message
    context and fire at parent credit return (delay 0). Plain spawns start
    context-free and also fire at parent credit return.
    """
    if is_pre:
        return [], list(predecessors)
    if branch.mode == ConversationBranchMode.FORK:
        return [*lineage, node_id], [(node_id, 0)]
    return [], [(node_id, 0)]


@dataclass(slots=True)
class _TreeBuilder:
    """Mutable per-tree expansion state for one root conversation.

    Holds the template lookup plus the growing node map and per-template
    instantiation counter so the recursive branch walk can stay a set of small
    methods rather than one deeply nested closure.
    """

    conv_by_sid: dict[str, Conversation]
    nodes: dict[str, DagNodeSpec] = field(default_factory=dict)
    # Per-tree instantiation counts: the first instance of a template session
    # id is bare, subsequent SPAWN references get a "#<n>" (n>=2) suffix.
    instance_counts: dict[str, int] = field(default_factory=dict)

    def alloc_instance(self, sid: str) -> str:
        """Mint this tree's instance id for template ``sid``.

        The first instance of a template is bare; subsequent SPAWN references
        get a ``"#<n>"`` (n>=2) suffix.
        """
        count = self.instance_counts.get(sid, 0) + 1
        self.instance_counts[sid] = count
        return sid if count == 1 else f"{sid}#{count}"

    def _turn_context(
        self,
        *,
        instance_id: str,
        turn: Turn,
        k: int,
        turn0_lineage: list[str],
        turn0_predecessors: list[tuple[str, int]],
    ) -> tuple[list[str], list[tuple[str, int]]]:
        """Lineage and firing predecessors for one turn node.

        Turn 0 inherits the caller-supplied seeds (fork/spawn/root entry); a
        later turn chains sequentially off ``k-1`` carrying its authored delay
        (ms -> us).
        """
        if k == 0:
            return list(turn0_lineage), list(turn0_predecessors)
        prev_id = f"{instance_id}:{k - 1}"
        lineage = [*self.nodes[prev_id].lineage, prev_id]
        predecessors = [(prev_id, int((turn.delay or 0) * 1000))]
        return lineage, predecessors

    def _expand_branches(
        self,
        *,
        turn: Turn,
        branch_by_id: dict[str, ConversationBranchInfo],
        branch_leaves: dict[str, list[str]],
        node_id: str,
        lineage: list[str],
        predecessors: list[tuple[str, int]],
        agent_depth: int,
    ) -> None:
        """Instantiate and recurse into every child branch fired by ``turn``.

        Records post-spawn branch leaves (so a later turn's SPAWN_JOIN can
        resolve its fan-in) before recursing with per-branch-kind entry context.
        Child instances carry the legacy identity: FORK/SPAWN children get
        ``agent_depth + 1`` with this firing node as their parent; pre-session
        children get depth 1 with no parent (``start_pre_session_child``).
        """
        for branch_id in turn.branch_ids:
            branch = branch_by_id.get(branch_id)
            if branch is None:
                raise ValueError(
                    f"node {node_id!r} fires branch {branch_id!r}, but the loader "
                    f"emitted no matching branch on this conversation "
                    f"(loader-consistency invariant violated)"
                )
            is_pre = branch.dispatch_timing == "pre"
            for child_sid in branch.child_conversation_ids:
                child_conv = self.conv_by_sid[child_sid]
                child_instance = self.alloc_instance(child_sid)
                child_leaf = f"{child_instance}:{len(child_conv.turns) - 1}"
                if branch.mode == ConversationBranchMode.SPAWN and not is_pre:
                    branch_leaves.setdefault(branch_id, []).append(child_leaf)

                child_lineage, child_predecessors = _child_entry_context(
                    branch=branch,
                    is_pre=is_pre,
                    node_id=node_id,
                    lineage=lineage,
                    predecessors=predecessors,
                )
                self.expand(
                    child_conv,
                    child_instance,
                    turn0_lineage=child_lineage,
                    turn0_predecessors=child_predecessors,
                    agent_depth=1 if is_pre else agent_depth + 1,
                    parent_node_id=None if is_pre else node_id,
                )

    def expand(
        self,
        conv: Conversation,
        instance_id: str,
        *,
        turn0_lineage: list[str],
        turn0_predecessors: list[tuple[str, int]],
        agent_depth: int,
        parent_node_id: str | None,
    ) -> None:
        """Emit every turn node for one conversation instance, depth-first.

        Turn nodes are created in order; each fired branch is expanded
        immediately after its owning turn so child node_ids interleave with the
        parent's remaining turns exactly as the legacy walk produced them.
        ``agent_depth`` / ``parent_node_id`` are the INSTANCE's legacy identity,
        stamped identically on every turn node the instance emits.
        """
        branch_by_id = {b.branch_id: b for b in conv.branches}
        # branch_id -> leaf node_ids of its post-SPAWN child instances, filled as
        # branches fire so a later turn's SPAWN_JOIN can resolve its fan-in.
        branch_leaves: dict[str, list[str]] = {}

        for k, turn in enumerate(conv.turns):
            node_id = f"{instance_id}:{k}"
            lineage, predecessors = self._turn_context(
                instance_id=instance_id,
                turn=turn,
                k=k,
                turn0_lineage=turn0_lineage,
                turn0_predecessors=turn0_predecessors,
            )
            self.nodes[node_id] = DagNodeSpec(
                node_id=node_id,
                turn=turn,
                lineage=lineage,
                predecessors=predecessors,
                join_inputs=_resolve_join_inputs(
                    turn=turn, branch_leaves=branch_leaves
                ),
                agent_depth=agent_depth,
                parent_node_id=parent_node_id,
            )
            self._expand_branches(
                turn=turn,
                branch_by_id=branch_by_id,
                branch_leaves=branch_leaves,
                node_id=node_id,
                lineage=lineage,
                predecessors=predecessors,
                agent_depth=agent_depth,
            )


def _expand_tree(root: Conversation, conv_by_sid: dict[str, Conversation]) -> DagTree:
    builder = _TreeBuilder(conv_by_sid=conv_by_sid)
    root_instance = builder.alloc_instance(root.session_id)
    # Recursive descent: expand -> _expand_branches -> expand adds ~3 stack frames
    # per fork/spawn nesting level, so recursion depth scales with authored nesting,
    # not tree width. Realistic DAGs nest a handful of levels; a pathological
    # >300-level chain would approach CPython's default recursion limit and want an
    # explicit iterative walk instead.
    builder.expand(
        root,
        root_instance,
        turn0_lineage=[],
        turn0_predecessors=[],
        agent_depth=0,
        parent_node_id=None,
    )
    return DagTree(trace_id=root.session_id, nodes=builder.nodes)
