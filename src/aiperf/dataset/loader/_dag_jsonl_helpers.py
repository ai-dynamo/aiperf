# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-level helpers for the DAG JSONL loader.

Split out from ``dag_jsonl.py`` to keep the loader file under the 500-line
ergonomics cap. Contains the loader's exception type, formatters, and the
pure-function shape transforms (no loader state) used during desugaring and
validation.
"""

from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import Conversation
from aiperf.dataset.loader.dag_jsonl_models import DagFork


class DagLoadError(ValueError):
    """Raised when a DAG JSONL file cannot be parsed."""


def normalize_fork_entry(entry: str | DagFork) -> DagFork:
    """Bare-string ``"<sid>"`` desugars to ``DagFork(child="<sid>",
    background=False)``; an object entry passes through. Normalizing at
    parse time means the desugar path only sees one shape.
    """
    if isinstance(entry, str):
        return DagFork(child=entry, background=False)
    return entry


def format_validation_error(lineno: int, err: ValidationError) -> str:
    """Render the first pydantic error as ``line N: <path>: <msg>``.

    Pydantic's default stringification produces multi-line output that is
    noisy in a single-line ``DagLoadError.message``. We surface the first
    error (usually the most actionable) with its dotted location so authors
    can jump straight to the bad field.
    """
    errors = err.errors()
    if not errors:
        return f"line {lineno}: invalid DAG conversation: {err}"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "validation error")
    return f"line {lineno}: {loc}: {msg}" if loc else f"line {lineno}: {msg}"


def group_spawn_entries(
    entries: list[Any],
) -> list[tuple[list[str], int | None]]:
    """Split a turn's ``spawns`` list into (children, join_at) groups.

    Consecutive bare strings collapse into one group with ``join_at=None``
    (single-branch shorthand). Each DagSpawn object becomes its own group
    carrying the authored ``join_at``.
    """
    groups: list[tuple[list[str], int | None]] = []
    bare_bucket: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            bare_bucket.append(entry)
        else:
            if bare_bucket:
                groups.append((bare_bucket, None))
                bare_bucket = []
            groups.append((list(entry.children), entry.join_at))
    if bare_bucket:
        groups.append((bare_bucket, None))
    return groups


def turn_idx_from_branch_id(branch_id: str) -> int:
    """Extract the spawning turn index from a generated branch_id.

    branch_id shapes: ``<sid>:<turn>``, ``<sid>:<turn>:<mode-suffix>``, or the
    pre-session marker ``<sid>:pre`` (always turn 0). ``sid`` itself can
    contain ``:`` so we anchor on the trailing numeric (with optional
    fork/spawn suffix) or the literal ``pre`` suffix.
    """
    if branch_id.endswith(":pre"):
        return 0
    parts = branch_id.rsplit(":", 2)
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    if len(parts) == 3 and parts[-2].isdigit():
        return int(parts[-2])
    raise DagLoadError(
        f"malformed branch_id '{branch_id}' (expected '<sid>:<turn>' "
        "or '<sid>:<turn>:<mode>')"
    )


def check_branch_duplicates(
    sid: str,
    idx: int,
    forks: list[Any],
    spawn_groups: list[tuple[list[str], int | None]],
) -> None:
    """Reject duplicate child_conversation_ids within fork/spawn groups.

    Duplicates within a group OR across multiple spawn groups on the same
    turn would silently double-dispatch the child and double-count the
    SPAWN_JOIN gate's expected counter. The orchestrator has no defense
    against this — the loader is the only line.

    A child appearing once with ``background=False`` AND once with
    ``background=True`` on the same turn is also rejected (different
    scheduling semantics, but the child still ends up dispatched twice
    with ambiguous parent-completion semantics).

    Fork-vs-spawn cross-pollination on the same turn is a distinct case
    (different modes, disambiguated branch_ids) and is intentionally allowed.
    """
    seen_in_fork: set[str] = set()
    for f in forks:
        if f.child in seen_in_fork:
            raise DagLoadError(
                f"session '{sid}' turn {idx}: duplicate "
                f"child_conversation_id '{f.child}' in fork entries"
            )
        seen_in_fork.add(f.child)
    seen_across_spawns: set[str] = set()
    for group_children, _ in spawn_groups:
        seen_in_group: set[str] = set()
        for child in group_children:
            if child in seen_in_group:
                raise DagLoadError(
                    f"session '{sid}' turn {idx}: duplicate "
                    f"child_conversation_id '{child}' in spawn group"
                )
            seen_in_group.add(child)
        cross = seen_in_group & seen_across_spawns
        if cross:
            dup = sorted(cross)[0]
            raise DagLoadError(
                f"session '{sid}' turn {idx}: duplicate "
                f"child_conversation_id '{dup}' across spawn groups"
            )
        seen_across_spawns |= seen_in_group


def validate_explicit_join_at(sid: str, idx: int, join_at: int, num_turns: int) -> None:
    """Author-supplied join_at must be strictly after the spawning turn and
    within the conversation."""
    if join_at <= idx:
        raise DagLoadError(
            f"session '{sid}' turn {idx}: spawn "
            f"join_at={join_at} must be strictly greater "
            f"than the spawning turn index"
        )
    if join_at >= num_turns:
        raise DagLoadError(
            f"session '{sid}' turn {idx}: spawn "
            f"join_at={join_at} is out of range "
            f"(conversation has {num_turns} turns)"
        )


def detect_cycles(conversations: dict[str, Conversation]) -> None:
    """Reject any cycles in the branch graph with a path-trace error.

    Walks ``Conversation.branches`` recursively and raises ``DagLoadError``
    with the offending cycle path on first hit.
    """
    visited: set[str] = set()
    path_stack: list[str] = []

    def dfs(node: str) -> None:
        if node in path_stack:
            cycle = " -> ".join(path_stack[path_stack.index(node) :] + [node])
            raise DagLoadError(f"cycle detected: {cycle}")
        if node in visited:
            return
        path_stack.append(node)
        for sp in conversations[node].branches:
            for child in sp.child_conversation_ids:
                dfs(child)
        path_stack.pop()
        visited.add(node)

    for sid in conversations:
        dfs(sid)


def validate_system_message_placement(
    conversations: dict[str, Any],
    parent_of: dict[str, tuple[str, int]],
) -> None:
    """Reject ``system`` messages on non-root turns.

    The accumulator-seeding turn is turn 0 IFF this session is a root (no
    FORK parent). Every other turn would place its ``system`` entry at
    position > 0 in the wire payload after the pure-append merge, which
    Qwen3-VL and similar chat templates silently drop.
    """
    for sid, conv in conversations.items():
        is_fork_child = sid in parent_of
        for idx, turn in enumerate(conv.turns):
            is_accumulator_root = idx == 0 and not is_fork_child
            if is_accumulator_root:
                continue
            for m in turn.raw_messages or []:
                if isinstance(m, dict) and m.get("role") == "system":
                    raise DagLoadError(
                        f"session '{sid}' turn {idx}: non-root turns may not "
                        "contain a 'system' message. Place the single system "
                        "prompt at the root turn only; popular chat templates "
                        "(e.g. Qwen3-VL) ignore system messages after index 0."
                    )


def validate_pre_session_spawns_disjoint_from_forks(
    inline_pre_session_spawns: dict[str, list[str]],
    parent_of: dict[str, tuple[str, int]],
) -> None:
    """``pre_session_spawns`` children are SPAWN-mode (no parent context to
    inherit) — silently letting a FORK target sit in pre_session_spawns
    produces an empty seed for the child."""
    for sid, children in inline_pre_session_spawns.items():
        for child in children:
            if child in parent_of:
                raise DagLoadError(
                    f"session '{child}' is referenced by '{sid}' "
                    f"pre_session_spawns but is also a FORK target; "
                    "pre-session children must be SPAWN-mode (no parent "
                    "context to inherit)"
                )


def validate_branch_targets_and_collect_parents(
    conversations: dict[str, Any],
    all_ids: set[str],
) -> dict[str, tuple[str, int]]:
    """Walk every branch, validate child existence + non-empty children,
    and build the FORK-only ``parent_of`` map (used for multi-parent +
    cycle detection downstream).

    Multi-parent constraint applies only to FORK edges: FORK children
    inherit context from a single parent, so two FORK parents would
    produce ambiguous seed messages. SPAWN children are fresh-context
    templates and may be instantiated from multiple parents.
    """
    parent_of: dict[str, tuple[str, int]] = {}
    for sid, conv in conversations.items():
        for sp in conv.branches:
            turn_idx = turn_idx_from_branch_id(sp.branch_id)
            if not sp.child_conversation_ids:
                raise DagLoadError(
                    f"session '{sid}' turn {turn_idx}: branch '{sp.branch_id}' "
                    "declares no child_conversation_ids; empty branches are rejected"
                )
            is_fork = sp.mode == ConversationBranchMode.FORK
            for child in sp.child_conversation_ids:
                if child not in all_ids:
                    known = sorted(all_ids)[:10]
                    raise DagLoadError(
                        f"session '{sid}' turn {turn_idx}: branch target '{child}' not declared. "
                        f"Known sessions: {known}"
                    )
                if is_fork:
                    if child in parent_of:
                        prev_parent, prev_turn = parent_of[child]
                        raise DagLoadError(
                            f"session '{child}' forked by both '{prev_parent}' "
                            f"turn {prev_turn} and '{sid}' turn {turn_idx}; "
                            "FORK-mode children require a single parent"
                        )
                    parent_of[child] = (sid, turn_idx)
    return parent_of


def validate_non_terminal_branches(conversations: dict[str, Any]) -> None:
    """SPAWN branches on non-terminal turns auto-join via a generated
    SPAWN_JOIN prerequisite. FORK branches inherit parent context; on a
    non-final turn they require either ``background=True`` (parent
    continues, no join) or — implicitly — terminal placement. Reject
    FORK + foreground (background=False) on non-terminal turns."""
    for sid, conv in conversations.items():
        branch_by_id = {b.branch_id: b for b in conv.branches}
        for idx, turn in enumerate(conv.turns):
            if not turn.branch_ids or idx == len(conv.turns) - 1:
                continue
            offending = [
                bid
                for bid in turn.branch_ids
                if (b := branch_by_id.get(bid)) is not None
                and b.mode == ConversationBranchMode.FORK
                and not b.background
            ]
            if offending:
                raise DagLoadError(
                    f"session '{sid}' turn {idx} has foreground FORK "
                    f"branches but is not the last turn and no join is "
                    f"declared (use ``background=True`` on the fork "
                    f"entry to keep the parent running, or move the "
                    f"forks to the final turn)"
                )
