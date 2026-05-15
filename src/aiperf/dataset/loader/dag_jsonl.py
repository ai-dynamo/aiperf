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
from aiperf.dataset.loader._dag_jsonl_helpers import (
    DagLoadError,
    check_branch_duplicates,
    detect_cycles,
    format_validation_error,
    group_spawn_entries,
    normalize_fork_entry,
    validate_branch_targets_and_collect_parents,
    validate_explicit_join_at,
    validate_non_terminal_branches,
    validate_pre_session_spawns_disjoint_from_forks,
    validate_system_message_placement,
)
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.dataset.loader.dag_jsonl_models import DagConversation, DagFork, DagTurn
from aiperf.plugin.enums import DatasetSamplingStrategy

__all__ = ["DagJsonlLoader", "DagLoadError"]


class DagJsonlLoader(BaseFileLoader):
    """Plugin loader for DAG-shaped conversation JSONL files.

    One :class:`DagConversation` per line. Each turn is a :class:`DagTurn`
    carrying a required ``messages`` array plus an explicit whitelist of
    OpenAI chat-completions fields; vendor-specific fields go in
    ``extra``. Unknown top-level keys are rejected at load time.

    Structural keys describe branching/scheduling (not sent on the wire):

    - ``forks``: FORK branches. Children inherit the parent's accumulated
      context and sticky-route to the parent's worker. Bare-string entries
      terminate the parent; object entries with ``background=True`` keep
      the parent running its remaining turns.
    - ``spawns``: SPAWN branches. Children start fresh and route freely;
      bare-string auto-joins on the next turn, ``DagSpawn`` objects carry
      an explicit ``join_at``.

    Both keywords may appear on the same turn; they desugar into separate
    ``ConversationBranchInfo`` entries with distinct ``branch_id``s.

    ``messages`` is concatenated onto the session's accumulator on each turn
    (pure append). Authors should place a single ``system`` entry on the
    root/seed turn only — ``system`` entries on non-root turns are rejected
    at load time because popular chat templates (e.g. Qwen3-VL) ignore
    system messages after position 0.

    Constructor shapes:
    - Plugin contract: ``DagJsonlLoader(filename=..., user_config=...)``
    - Standalone: ``DagJsonlLoader(path)`` (unit tests, offline tooling).

    Standalone Python usage (offline tooling, tests)::

        >>> loader = DagJsonlLoader("tests/fixtures/dag/small.dag.jsonl")
        >>> conversations = loader.load()  # list[Conversation]
        >>> roots = loader.root_session_ids()  # set[str]; not referenced as a child

    Plugin usage (the framework calls these, not user code)::

        loader = DagJsonlLoader(filename=path, user_config=cfg)
        per_session = loader.load_dataset()  # dict[session_id, list[Conversation]]
        conversations = loader.convert_to_conversations(per_session)
    """

    def __init__(
        self,
        filename: str | Path | None = None,
        *,
        user_config: UserConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if filename is None:
            raise ValueError(
                "DagJsonlLoader requires a 'filename' (or positional path); "
                "plugin callers pass filename=... + user_config=..., standalone "
                "callers pass the path positionally"
            )
        if user_config is not None:
            super().__init__(filename=str(filename), user_config=user_config, **kwargs)
            cap_seconds = user_config.loadgen.inter_turn_delay_cap_seconds
        else:
            # Standalone path: bypass BaseFileLoader (no user_config available).
            self.user_config = None
            self.filename = str(filename)
            cap_seconds = None
        self._path = Path(filename)
        self._delay_cap_tracker = DelayCapTracker(cap_seconds=cap_seconds)
        self._conversations: dict[str, Conversation] = {}
        # Per-session, per-turn list of normalized DagFork objects. Bare-string
        # entries normalize to DagFork(child=s, background=False) at parse time.
        self._inline_forks: dict[str, list[list[DagFork]]] = {}
        # Each per-turn entry is a list of (children, join_at) groups. Bare
        # string entries collapse into a single group with ``join_at=None``;
        # explicit DagSpawn object entries become one group per entry
        # carrying the authored ``join_at``.
        self._inline_spawns: dict[str, list[list[tuple[list[str], int | None]]]] = {}
        # Per-session list of child session_ids flagged as pre-session
        # background spawns (dispatch_timing="pre"). Desugared into a
        # single SPAWN/pre branch attached to turn 0.
        self._inline_pre_session_spawns: dict[str, list[str]] = {}
        self._roots: set[str] = set()
        self._loaded: bool = False

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
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
        """Return the sampling strategy this loader prefers when none is set."""
        return DatasetSamplingStrategy.RANDOM

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        """Return the default conversation context mode for DAG datasets."""
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
                    conversations=[c.metadata() for c in self._conversations.values()],
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
                # Tolerate a UTF-8 BOM on the first line (common from
                # Windows/Excel exports). Bare bytes everywhere else.
                if lineno == 1 and raw.startswith(b"\xef\xbb\xbf"):
                    raw = raw[3:]
                raw = raw.strip()
                if not raw:
                    continue
                self._parse_one_line(lineno, raw)
        if not self._conversations:
            raise DagLoadError(
                f"DAG JSONL file '{self._path}' is empty (no conversations "
                "parsed); supply at least one conversation line"
            )

    def _parse_one_line(self, lineno: int, raw: bytes) -> None:
        prefix = f"failed to parse DAG JSONL '{self._path}'"
        try:
            obj = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            raise DagLoadError(f"{prefix} line {lineno}: invalid JSON: {e}") from e
        try:
            dag_conv = DagConversation.model_validate(obj)
        except ValidationError as e:
            raise DagLoadError(f"{prefix}: {format_validation_error(lineno, e)}") from e
        sid = dag_conv.session_id
        if sid in self._conversations:
            raise DagLoadError(f"{prefix} line {lineno}: duplicate session_id '{sid}'")
        turns: list[Turn] = []
        inline_forks_per_turn: list[list[DagFork]] = []
        inline_spawns_per_turn: list[list[tuple[list[str], int | None]]] = []
        for t in dag_conv.turns:
            turns.append(self._build_turn(t))
            inline_forks_per_turn.append([normalize_fork_entry(e) for e in t.forks])
            inline_spawns_per_turn.append(group_spawn_entries(t.spawns))
        self._conversations[sid] = Conversation(session_id=sid, turns=turns)
        self._inline_forks[sid] = inline_forks_per_turn
        self._inline_spawns[sid] = inline_spawns_per_turn
        self._inline_pre_session_spawns[sid] = self._check_pre_session_duplicates(
            sid, prefix, lineno, dag_conv.pre_session_spawns
        )

    @staticmethod
    def _check_pre_session_duplicates(
        sid: str, prefix: str, lineno: int, children: list[str]
    ) -> list[str]:
        """Reject duplicate child_conversation_ids in ``pre_session_spawns``.

        Per-turn ``forks``/``spawns`` reject duplicates via
        ``check_branch_duplicates`` because the orchestrator would otherwise
        double-dispatch the same child and double-count its SPAWN_JOIN
        gate. ``pre_session_spawns`` desugars into a single SPAWN branch
        with the same dispatch semantics, so the same guarantee applies."""
        seen: set[str] = set()
        for child in children:
            if child in seen:
                raise DagLoadError(
                    f"{prefix} line {lineno}: session '{sid}' duplicate "
                    f"child_conversation_id '{child}' in pre_session_spawns"
                )
            seen.add(child)
        return list(children)

    def _build_turn(self, t: DagTurn) -> Turn:
        return Turn(
            raw_messages=list(t.messages),
            raw_tools=list(t.tools) if t.tools is not None else None,
            model=t.model,
            max_tokens=t.max_tokens,
            extra_body=dict(t.extra) if t.extra is not None else None,
            delay=self._delay_cap_tracker.clamp(t.delay),
        )

    def _desugar_forks(self) -> None:
        for sid in self._conversations:
            conv = self._conversations[sid]
            fork_per_turn = self._inline_forks.get(sid, [])
            spawn_per_turn = self._inline_spawns.get(sid, [])
            num_turns = len(conv.turns)
            self._apply_pre_session_spawns(sid, conv)
            for idx in range(num_turns):
                forks = fork_per_turn[idx] if idx < len(fork_per_turn) else []
                spawn_groups = spawn_per_turn[idx] if idx < len(spawn_per_turn) else []
                if not forks and not spawn_groups:
                    continue
                # Partition by background flag — FG and BG forks emit
                # separate ConversationBranchInfo entries (different parent-
                # termination semantics, so they need distinct branch_ids).
                fg_forks = [f for f in forks if not f.background]
                bg_forks = [f for f in forks if f.background]
                check_branch_duplicates(sid, idx, forks, spawn_groups)
                # Suffix branch_ids when multiple branches need to coexist
                # on this turn: fork+spawn coexistence OR FG+BG fork
                # coexistence.
                multiple_fork_classes = bool(fg_forks) and bool(bg_forks)
                mixed = (bool(fg_forks) or bool(bg_forks)) and bool(spawn_groups)
                fork_disambiguate = mixed or multiple_fork_classes
                if fg_forks:
                    self._apply_forks(
                        sid,
                        conv,
                        idx,
                        forks=fg_forks,
                        background=False,
                        disambiguate=fork_disambiguate,
                    )
                if bg_forks:
                    self._apply_forks(
                        sid,
                        conv,
                        idx,
                        forks=bg_forks,
                        background=True,
                        disambiguate=fork_disambiguate,
                    )
                if spawn_groups:
                    self._apply_spawns(
                        sid,
                        conv,
                        idx,
                        spawn_groups=spawn_groups,
                        mixed=mixed,
                        num_turns=num_turns,
                    )

    def _apply_pre_session_spawns(self, sid: str, conv: Conversation) -> None:
        """Emit a pre-session SPAWN branch on turn 0 if any are declared.

        Done BEFORE the per-turn loop so its branch_id is stable and doesn't
        collide with per-turn spawn suffixes.
        """
        pre_session_children = self._inline_pre_session_spawns.get(sid, [])
        if not pre_session_children:
            return
        branch_id = f"{sid}:pre"
        conv.branches.append(
            ConversationBranchInfo(
                branch_id=branch_id,
                child_conversation_ids=list(pre_session_children),
                mode=ConversationBranchMode.SPAWN,
                dispatch_timing="pre",
            )
        )
        conv.turns[0].branch_ids.append(branch_id)

    def _apply_forks(
        self,
        sid: str,
        conv: Conversation,
        idx: int,
        *,
        forks: list[DagFork],
        background: bool,
        disambiguate: bool,
    ) -> None:
        # ``:fork``/``:bg_fork`` mirrors ``:spawn``/``:spawn<N>`` from
        # ``_apply_spawns``: distinct branch_ids when multiple branches
        # coexist on one turn.
        if disambiguate:
            branch_id = f"{sid}:{idx}:{'bg_fork' if background else 'fork'}"
        else:
            branch_id = f"{sid}:{idx}"
        conv.branches.append(
            ConversationBranchInfo(
                branch_id=branch_id,
                child_conversation_ids=[f.child for f in forks],
                mode=ConversationBranchMode.FORK,
                background=background,
            )
        )
        conv.turns[idx].branch_ids.append(branch_id)

    def _apply_spawns(
        self,
        sid: str,
        conv: Conversation,
        idx: int,
        *,
        spawn_groups: list[tuple[list[str], int | None]],
        mixed: bool,
        num_turns: int,
    ) -> None:
        # Multiple spawn groups on one turn get suffixed branch ids
        # (:spawn, :spawn2, ...) so they resolve distinctly.
        for group_idx, (children, join_at) in enumerate(spawn_groups):
            if not children:
                continue
            if mixed or len(spawn_groups) > 1:
                suffix = "spawn" if group_idx == 0 else f"spawn{group_idx}"
                branch_id = f"{sid}:{idx}:{suffix}"
            else:
                branch_id = f"{sid}:{idx}"
            # Determine join_at: explicit author value if provided, else
            # default to idx+1 (auto-join on the next turn).
            effective_join_at = join_at if join_at is not None else idx + 1
            # Terminal SPAWN: no legal join target exists. Author must
            # explicitly supply a join_at strictly inside the conversation —
            # otherwise the spawn is fire-and-forget within the post-session
            # window.
            is_terminal_spawn = effective_join_at >= num_turns
            if join_at is not None:
                validate_explicit_join_at(sid, idx, join_at, num_turns)
            conv.branches.append(
                ConversationBranchInfo(
                    branch_id=branch_id,
                    child_conversation_ids=list(children),
                    mode=ConversationBranchMode.SPAWN,
                )
            )
            conv.turns[idx].branch_ids.append(branch_id)
            # Implicit SPAWN_JOIN on the resolved join turn. Terminal spawns
            # get no prereq (fire-and-forget on the last turn).
            if not is_terminal_spawn:
                conv.turns[effective_join_at].prerequisites.append(
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch_id,
                    )
                )

    def _resolve_and_validate(self) -> None:
        all_ids = set(self._conversations.keys())
        parent_of = validate_branch_targets_and_collect_parents(
            self._conversations, all_ids
        )
        validate_non_terminal_branches(self._conversations)
        validate_pre_session_spawns_disjoint_from_forks(
            self._inline_pre_session_spawns, parent_of
        )
        validate_system_message_placement(self._conversations, parent_of)
        self._stamp_topology(parent_of)

    def _stamp_topology(self, parent_of: dict[str, tuple[str, int]]) -> None:
        """Stamp parent_conversation_id, run cycle detection, then walk
        agent_depth iteratively. FORK children inherit parent_depth + 1; SPAWN
        children remain at depth 0 (fresh root-like context) per the DAG spec."""
        for child_sid, (parent_sid, _turn_idx) in parent_of.items():
            child = self._conversations.get(child_sid)
            if child is None:
                continue
            child.parent_conversation_id = parent_sid
        # Detect cycles BEFORE the depth walk — the iterative depth update
        # cannot terminate on cyclic input.
        self._detect_cycles()
        # Compute agent_depth iteratively: parent_depth + 1 for FORK children.
        # Acyclic by construction now (cycle check above raises first).
        depth_changed = True
        while depth_changed:
            depth_changed = False
            for child_sid, (parent_sid, _turn_idx) in parent_of.items():
                child = self._conversations.get(child_sid)
                parent = self._conversations.get(parent_sid)
                if child is None or parent is None:
                    continue
                new_depth = parent.agent_depth + 1
                if child.agent_depth < new_depth:
                    child.agent_depth = new_depth
                    depth_changed = True

    def _detect_cycles(self) -> None:
        detect_cycles(self._conversations)

    def _compute_roots(self) -> set[str]:
        referenced: set[str] = set()
        for c in self._conversations.values():
            for sp in c.branches:
                referenced.update(sp.child_conversation_ids)
        return set(self._conversations.keys()) - referenced
