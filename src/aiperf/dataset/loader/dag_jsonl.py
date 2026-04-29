# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationBranchMode, ConversationContextMode
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, DatasetMetadata, Turn
from aiperf.common.validators import validate_for_orchestrator_v1
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

    ``messages`` is concatenated onto the session's accumulator on each turn
    (pure append). Authors should place a single ``system`` entry on the
    root/seed turn only — ``system`` entries on non-root turns are rejected at
    load time because popular chat templates (e.g. Qwen3-VL) ignore system
    messages after position 0, which would silently misrepresent the
    benchmark.

    The loader supports two constructor shapes:
    - Plugin contract: ``DagJsonlLoader(filename=..., user_config=...)``
    - Legacy/standalone: ``DagJsonlLoader(path)`` (used by unit tests and tools)

    Example:
        Two-session DAG: a root with one user turn that forks into one child::

            {"session_id": "root", "turns": [
                {"messages": [{"role": "user", "content": "Hi"}],
                 "forks": ["child"]}
            ]}
            {"session_id": "child", "turns": [
                {"messages": [{"role": "user", "content": "Continue"}],
                 "max_tokens": 64,
                 "extra_body": {"temperature": 0.0, "ignore_eos": true}}
            ]}

        Authoring rules enforced at load time:
        - System messages are only valid on a root's turn 0 (chat-template
          placement; rejected elsewhere).
        - ``forks`` is only legal on the last turn (no joins yet in v1).
        - ``extra_body`` is the catch-all for non-native fields (temperature,
          top_p, seed, ignore_eos, ...) and is merged into the wire body at
          dispatch time, matching the OpenAI SDK ``extra_body=`` convention.

        See ``examples/dag_jsonl/example.dag.jsonl`` for a fuller example.
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
        else:
            # Legacy path: bypass BaseFileLoader (no user_config available).
            self.user_config = None
            self.filename = str(filename)
        self._path = Path(filename)
        self._conversations: dict[str, Conversation] = {}
        self._inline_forks: dict[str, list[list[str]]] = {}
        self._depths: dict[str, int] = {}
        self._loaded: bool = False

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True when data looks like a DAG conversation line.

        DAG lines have top-level ``session_id`` and ``turns`` where at least
        one turn carries a ``messages`` array or ``forks``.
        """
        if data is None:
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
            if "forks" in t:
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
            self._depths = self._compute_depths()
            for sid, conv in self._conversations.items():
                conv.context_mode = ConversationContextMode.DELTAS_WITHOUT_RESPONSES
                conv.agent_depth = self._depths[sid]
            metadata = DatasetMetadata(
                conversations=[c.metadata() for c in self._conversations.values()],
                sampling_strategy=DatasetSamplingStrategy.RANDOM,
            )
            validate_for_orchestrator_v1(metadata)
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

    # --- Legacy / standalone API ---------------------------------------------

    def load(self) -> list[Conversation]:
        """Legacy helper used by tests and offline tooling."""
        data = self.load_dataset()
        return self.convert_to_conversations(data)

    def root_session_ids(self) -> set[str]:
        if not self._loaded:
            self.load_dataset()
        return {sid for sid, depth in self._depths.items() if depth == 0}

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
                            delay=t.delay,
                        )
                    )
                    inline_forks_per_turn.append(list(t.forks))
                self._conversations[sid] = Conversation(session_id=sid, turns=turns)
                self._inline_forks[sid] = inline_forks_per_turn

    def _desugar_forks(self) -> None:
        for sid in self._conversations:
            conv = self._conversations[sid]
            fork_per_turn = self._inline_forks.get(sid, [])
            num_turns = len(conv.turns)
            for idx in range(num_turns):
                fork_children = fork_per_turn[idx] if idx < len(fork_per_turn) else []
                if not fork_children:
                    continue
                branch_id = f"{sid}:{idx}"
                conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=list(fork_children),
                        mode=ConversationBranchMode.FORK,
                    )
                )
                conv.turns[idx].branch_ids.append(branch_id)

    def _resolve_and_validate(self) -> None:
        all_ids = set(self._conversations.keys())
        parent_of: dict[str, tuple[str, int]] = {}

        def _turn_idx_from_branch_id(branch_id: str) -> int:
            # branch_id shape: "<sid>:<turn>" (sid may itself contain ':',
            # so anchor on the trailing numeric).
            tail = branch_id.rsplit(":", 1)
            if len(tail) == 2 and tail[-1].isdigit():
                return int(tail[-1])
            raise DagLoadError(
                f"malformed branch_id '{branch_id}' (expected '<sid>:<turn>')"
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
            for idx, turn in enumerate(conv.turns):
                if turn.branch_ids and idx != len(conv.turns) - 1:
                    raise DagLoadError(
                        f"session '{sid}' turn {idx} has branches but is not the last turn"
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

    def _compute_depths(self) -> dict[str, int]:
        """Compute the static DAG depth of every conversation in one pass.

        Depth is a topological property of the FORK-only single-parent
        invariant the loader enforces in ``_resolve_and_validate`` and
        ``validate_for_orchestrator_v1`` (no diamonds, no cycles, every
        fork-target declared as its own conversation, fork-targets have
        exactly one parent). Under those rules each conversation has a
        unique parent and therefore a unique distance from the nearest
        root, computable in a single BFS.

        Roots = conversations with no incoming fork edge. They seed the
        BFS at depth 0; each child gets ``parent_depth + 1``. Every
        conversation in the file must be reached — a miss indicates a
        topology validation gap upstream and raises here as a backstop.

        The result is stamped onto ``Conversation.agent_depth`` so the
        runtime ``BranchOrchestrator.intercept`` can look up a child's
        depth instead of recomputing ``parent_depth + 1`` on every spawn.
        """
        # Roots = nodes with no incoming fork edge. Compute and seed
        # the BFS in one pass so we don't walk the topology twice.
        referenced: set[str] = set()
        for c in self._conversations.values():
            for sp in c.branches:
                referenced.update(sp.child_conversation_ids)
        roots = set(self._conversations.keys()) - referenced

        depths: dict[str, int] = dict.fromkeys(roots, 0)
        frontier: list[str] = list(roots)
        while frontier:
            next_frontier: list[str] = []
            for parent_sid in frontier:
                parent_depth = depths[parent_sid]
                for branch in self._conversations[parent_sid].branches:
                    for child_sid in branch.child_conversation_ids:
                        if child_sid in depths:
                            # Single-parent invariant: a child should
                            # never be visited twice. A repeat here
                            # means an upstream validator missed a
                            # diamond.
                            continue
                        depths[child_sid] = parent_depth + 1
                        next_frontier.append(child_sid)
            frontier = next_frontier

        missing = set(self._conversations.keys()) - set(depths.keys())
        if missing:
            raise ValueError(
                f"DAG depth computation missed conversations (orphaned, "
                f"unreachable from any root): {sorted(missing)!r}. This "
                f"indicates a topology validation gap upstream."
            )
        return depths
