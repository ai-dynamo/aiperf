# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    PrerequisiteKind,
)
from aiperf.common.models import DatasetMetadata, TurnPrerequisite
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.dag_jsonl_models import DagConversation
from aiperf.plugin.enums import DatasetSamplingStrategy


class DagLoadError(ValueError):
    """Raised when a DAG JSONL file cannot be parsed."""


def _format_validation_error(lineno: int, err: ValidationError) -> str:
    """Render the first pydantic error as ``line N: <path>: <msg>``.

    Pydantic's default stringification produces multi-line output that is
    noisy in a single-line ``DagLoadError.message``. We surface the first
    error (usually the most actionable) with its dotted location so authors
    can jump straight to the bad field.
    """
    errors = err.errors()
    if not errors:
        return f"line {lineno}: invalid DAG conversation"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "validation error")
    return f"line {lineno}: {loc}: {msg}" if loc else f"line {lineno}: {msg}"


class DagJsonlLoader(BaseFileLoader):
    """Plugin loader for DAG-shaped conversation JSONL files.

    One :class:`DagConversation` per line. Each turn is a flat
    :class:`DagTurn` object carrying a required ``messages`` array plus an
    explicit whitelist of OpenAI chat-completions fields (``max_tokens``,
    ``model``, ``tools``, ``temperature``, …); vendor-specific fields go in
    ``extra_body``. Unknown top-level keys on either a conversation or a turn
    are rejected at load time so typos surface immediately.

    Structural keys describe branching and scheduling (not sent on the wire):

    - ``forks: [session_id, ...]`` — FORK-mode branches. Children inherit the
      parent's accumulated message context and sticky-route to the parent's
      worker (prefix-cache locality).
    - ``spawns: [session_id, ...]`` — SPAWN-mode branches. Children start with
      a fresh context and route freely.

    Both shorthands may appear on the same turn; they desugar into separate
    ``ConversationBranchInfo`` entries with distinct ``branch_id``s.

    ``messages`` is concatenated onto the session's accumulator on each turn
    (pure append). Authors should place a single ``system`` entry on the
    root/seed turn only — ``system`` entries on non-root turns are rejected at
    load time because popular chat templates (e.g. Qwen3-VL) ignore system
    messages after position 0, which would silently misrepresent the
    benchmark.

    The loader supports two constructor shapes:
    - Plugin contract: ``DagJsonlLoader(filename=..., user_config=...)``
    - Legacy/standalone: ``DagJsonlLoader(path)`` (used by unit tests and tools)
    """

    def __init__(
        self,
        filename: str | Path | None = None,
        *,
        user_config: UserConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if filename is None:
            raise ValueError("DagJsonlLoader requires a filename/path")
        if user_config is not None:
            super().__init__(filename=str(filename), user_config=user_config, **kwargs)
            cap_seconds = user_config.loadgen.inter_turn_delay_cap_seconds
        else:
            # Legacy path: bypass BaseFileLoader (no user_config available).
            self.user_config = None
            self.filename = str(filename)
            cap_seconds = None
        self._path = Path(filename)
        self._delay_cap_tracker = DelayCapTracker(cap_seconds=cap_seconds)
        self._conversations: dict[str, Conversation] = {}
        self._inline_forks: dict[str, list[list[str]]] = {}
        # Each per-turn entry is a list of (children, join_at) groups. Legacy
        # string entries in the wire format collapse into a single group with
        # ``join_at=None``; explicit DagSpawn object entries become one group
        # per entry carrying the authored ``join_at``.
        self._inline_spawns: dict[str, list[list[tuple[list[str], int | None]]]] = {}
        # Per-session list of child session_ids flagged as pre-session
        # background spawns (dispatch_timing="pre"). Desugared into a
        # single SPAWN/background branch attached to turn 0.
        self._inline_pre_session_spawns: dict[str, list[str]] = {}
        self._roots: set[str] = set()
        self._loaded: bool = False

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True when data looks like a DAG conversation line.

        DAG lines have top-level ``session_id`` and ``turns`` where at least
        one turn carries a ``messages`` array, ``forks``, or ``spawns``.
        """
        if data is None:
            return False
        # Auto-detection feeds arbitrary first-record shapes; guard against
        # non-dict inputs before calling ``data.get`` so the probe returns
        # False cleanly instead of AttributeError.
        if not isinstance(data, dict):
            return False
        if not isinstance(data.get("session_id"), str):
            return False
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            return False
        for t in turns:
            if not isinstance(t, dict):
                return False
            if isinstance(t.get("messages"), list):
                return True
            if "forks" in t or "spawns" in t:
                return True
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.RANDOM

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        return ConversationContextMode.DELTAS_WITHOUT_RESPONSES

    # --- Plugin-facing API ---------------------------------------------------

    def load_dataset(self) -> dict[str, list[Conversation]]:
        """Parse the DAG JSONL file and return session_id -> [Conversation]."""
        if not self._loaded:
            self._parse_lines()
            self._desugar_forks()
            self._resolve_and_validate()
            self._roots = self._compute_roots()
            for sid, conv in self._conversations.items():
                conv.context_mode = ConversationContextMode.DELTAS_WITHOUT_RESPONSES
                conv.is_root = sid in self._roots
            # v1 orchestrator capability check - surface any unsupported
            # prereq/branch shapes before any credit is issued.
            validate_for_orchestrator_v1(
                DatasetMetadata(
                    conversations=[
                        c.to_metadata() for c in self._conversations.values()
                    ],
                    sampling_strategy=self.get_preferred_sampling_strategy(),
                )
            )
            self._delay_cap_tracker.log_summary(logger_name=__name__)
            self._loaded = True
        return {sid: [conv] for sid, conv in self._conversations.items()}

    def convert_to_conversations(
        self, data: dict[str, list[Conversation]]
    ) -> list[Conversation]:
        """Flatten the loader's intermediate dict into a list of Conversations."""
        out: list[Conversation] = []
        for convs in data.values():
            out.extend(convs)
        return out

    # --- Standalone API ------------------------------------------------------

    def load(self) -> list[Conversation]:
        """Helper used by tests and offline tooling."""
        data = self.load_dataset()
        return self.convert_to_conversations(data)

    def root_session_ids(self) -> set[str]:
        if not self._loaded:
            self.load_dataset()
        return self._roots

    # --- Internal parsing ----------------------------------------------------

    def _parse_lines(self) -> None:
        with self._path.open("rb") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError as e:
                    raise DagLoadError(f"line {lineno}: invalid JSON: {e}") from e
                try:
                    dag_conv = DagConversation.model_validate(obj)
                except ValidationError as e:
                    raise DagLoadError(_format_validation_error(lineno, e)) from e
                sid = dag_conv.session_id
                if sid in self._conversations:
                    raise DagLoadError(f"line {lineno}: duplicate session_id '{sid}'")
                turns: list[Turn] = []
                inline_forks_per_turn: list[list[str]] = []
                inline_spawns_per_turn: list[list[tuple[list[str], int | None]]] = []
                for t in dag_conv.turns:
                    turns.append(
                        Turn(
                            raw_messages=list(t.messages),
                            raw_tools=list(t.tools) if t.tools is not None else None,
                            model=t.model,
                            max_tokens=t.max_tokens,
                            extra_body=dict(t.extra_body)
                            if t.extra_body is not None
                            else None,
                            delay=self._delay_cap_tracker.clamp(t.delay),
                        )
                    )
                    inline_forks_per_turn.append(list(t.forks))
                    # Split a turn's ``spawns`` list into groups: consecutive
                    # legacy strings collapse into one group (preserves the
                    # single-branch legacy semantics); each DagSpawn object
                    # becomes its own group carrying the authored join_at.
                    groups: list[tuple[list[str], int | None]] = []
                    legacy_bucket: list[str] = []
                    for entry in t.spawns:
                        if isinstance(entry, str):
                            legacy_bucket.append(entry)
                        else:
                            if legacy_bucket:
                                groups.append((legacy_bucket, None))
                                legacy_bucket = []
                            groups.append((list(entry.children), entry.join_at))
                    if legacy_bucket:
                        groups.append((legacy_bucket, None))
                    inline_spawns_per_turn.append(groups)
                self._conversations[sid] = Conversation(session_id=sid, turns=turns)
                self._inline_forks[sid] = inline_forks_per_turn
                self._inline_spawns[sid] = inline_spawns_per_turn
                self._inline_pre_session_spawns[sid] = list(dag_conv.pre_session_spawns)

    def _desugar_forks(self) -> None:
        for sid in self._conversations:
            conv = self._conversations[sid]
            fork_per_turn = self._inline_forks.get(sid, [])
            spawn_per_turn = self._inline_spawns.get(sid, [])
            num_turns = len(conv.turns)
            # Phase 2b: pre-session background SPAWN branch attached to
            # turn 0. Emitted BEFORE the per-turn loop so its branch_id is
            # stable and doesn't collide with per-turn spawn suffixes.
            pre_session_children = self._inline_pre_session_spawns.get(sid, [])
            if pre_session_children:
                branch_id = f"{sid}:pre"
                conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=list(pre_session_children),
                        mode=ConversationBranchMode.SPAWN,
                        is_background=True,
                        dispatch_timing="pre",
                    )
                )
                conv.turns[0].branch_ids.append(branch_id)
            for idx in range(num_turns):
                fork_children = fork_per_turn[idx] if idx < len(fork_per_turn) else []
                spawn_groups = spawn_per_turn[idx] if idx < len(spawn_per_turn) else []
                if not fork_children and not spawn_groups:
                    continue
                # Reject duplicate child_conversation_ids per spawn group AND
                # across multiple spawn groups on the same turn (legacy
                # strings + DagSpawn objects materialize as separate groups).
                # Duplicates would silently double-dispatch the child and
                # double-count the SPAWN_JOIN gate's expected counter (the
                # gate would never fire or fire late). The orchestrator has
                # no defense against this — the loader is the only line.
                # Fork-vs-spawn cross-pollination on the same turn is a
                # distinct case (different modes, disambiguated branch_ids)
                # and is intentionally allowed; see
                # test_forks_and_spawns_pointing_at_same_child_emits_two_branches.
                seen_in_fork: set[str] = set()
                for child in fork_children:
                    if child in seen_in_fork:
                        raise DagLoadError(
                            f"session '{sid}' turn {idx}: duplicate "
                            f"child_conversation_id '{child}' in fork group"
                        )
                    seen_in_fork.add(child)
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
                # When both shorthands appear on the same turn, disambiguate
                # the branch_ids so the orchestrator can look up each
                # ConversationBranchInfo distinctly.
                mixed = bool(fork_children) and bool(spawn_groups)
                if fork_children:
                    branch_id = f"{sid}:{idx}:fork" if mixed else f"{sid}:{idx}"
                    conv.branches.append(
                        ConversationBranchInfo(
                            branch_id=branch_id,
                            child_conversation_ids=list(fork_children),
                            mode=ConversationBranchMode.FORK,
                        )
                    )
                    conv.turns[idx].branch_ids.append(branch_id)
                if spawn_groups:
                    # Multiple spawn groups on one turn get suffixed branch
                    # ids (:spawn, :spawn2, ...) so they resolve distinctly.
                    for group_idx, (children, join_at) in enumerate(spawn_groups):
                        if not children:
                            continue
                        if mixed or len(spawn_groups) > 1:
                            suffix = "spawn" if group_idx == 0 else f"spawn{group_idx}"
                            branch_id = f"{sid}:{idx}:{suffix}"
                        else:
                            branch_id = f"{sid}:{idx}"
                        # Determine join_at: explicit author value if
                        # provided, else legacy default of idx+1.
                        effective_join_at = join_at if join_at is not None else idx + 1
                        # is_terminal_spawn True when no legal join target
                        # exists (spawn on last turn and no author override).
                        is_terminal_spawn = effective_join_at >= num_turns
                        if join_at is not None:
                            # Author-supplied join_at must be strictly after
                            # the spawning turn and within the conversation.
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
                        conv.branches.append(
                            ConversationBranchInfo(
                                branch_id=branch_id,
                                child_conversation_ids=list(children),
                                mode=ConversationBranchMode.SPAWN,
                                is_background=is_terminal_spawn,
                            )
                        )
                        conv.turns[idx].branch_ids.append(branch_id)
                        # Implicit SPAWN_JOIN on the resolved join turn.
                        # Terminal spawns get no prereq and are marked
                        # background (fire-and-forget).
                        if not is_terminal_spawn:
                            conv.turns[effective_join_at].prerequisites.append(
                                TurnPrerequisite(
                                    kind=PrerequisiteKind.SPAWN_JOIN,
                                    branch_id=branch_id,
                                )
                            )

    def _resolve_and_validate(self) -> None:
        all_ids = set(self._conversations.keys())
        parent_of: dict[str, tuple[str, int]] = {}

        def _turn_idx_from_branch_id(branch_id: str) -> int:
            # branch_id shapes: "<sid>:<turn>", "<sid>:<turn>:<mode-suffix>",
            # or the pre-session marker "<sid>:pre" (always turn 0). ``sid``
            # itself can contain ':' so we must anchor on the trailing numeric
            # (with optional fork/spawn suffix) or the literal ``pre`` suffix.
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

        for sid, conv in self._conversations.items():
            for sp in conv.branches:
                turn_idx = _turn_idx_from_branch_id(sp.branch_id)
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
                    # Multi-parent constraint applies only to FORK edges:
                    # FORK children inherit context from a single parent, so
                    # two FORK parents would produce ambiguous seed messages.
                    # SPAWN children are fresh-context templates and may be
                    # instantiated from multiple parents.
                    if is_fork:
                        if child in parent_of:
                            prev_parent, prev_turn = parent_of[child]
                            raise DagLoadError(
                                f"session '{child}' forked by both '{prev_parent}' "
                                f"turn {prev_turn} and '{sid}' turn {turn_idx}; "
                                "FORK-mode children require a single parent"
                            )
                        parent_of[child] = (sid, turn_idx)
        for sid, conv in self._conversations.items():
            branch_mode_by_id = {b.branch_id: b.mode for b in conv.branches}
            for idx, turn in enumerate(conv.turns):
                if not turn.branch_ids or idx == len(conv.turns) - 1:
                    continue
                # SPAWN branches on non-terminal turns auto-join on the
                # immediately-following turn via a generated SPAWN_JOIN
                # prerequisite. FORK branches inherit parent context and
                # still must terminate the parent's script.
                non_spawn = [
                    bid
                    for bid in turn.branch_ids
                    if branch_mode_by_id.get(bid) != ConversationBranchMode.SPAWN
                ]
                if non_spawn:
                    raise DagLoadError(
                        f"session '{sid}' turn {idx} has branches but is not the last turn "
                        f"and no join is declared"
                    )
        # System-prompt placement: the accumulator-seeding turn for a session
        # is turn 0 IFF this session is a root (no FORK parent). Every other
        # turn would place its ``system`` entry at a position > 0 in the wire
        # payload after the pure-append merge, which Qwen3-VL and similar chat
        # templates silently drop. Reject early so authors catch the mistake.
        for sid, conv in self._conversations.items():
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
        visited: set[str] = set()
        path_stack: list[str] = []

        def dfs(node: str) -> None:
            if node in path_stack:
                cycle = " -> ".join(path_stack[path_stack.index(node) :] + [node])
                raise DagLoadError(f"cycle detected: {cycle}")
            if node in visited:
                return
            path_stack.append(node)
            for sp in self._conversations[node].branches:
                for child in sp.child_conversation_ids:
                    dfs(child)
            path_stack.pop()
            visited.add(node)

        for sid in self._conversations:
            dfs(sid)

    def _compute_roots(self) -> set[str]:
        referenced: set[str] = set()
        for c in self._conversations.values():
            for sp in c.branches:
                referenced.update(sp.child_conversation_ids)
        return set(self._conversations.keys()) - referenced
