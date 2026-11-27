# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import orjson
from pydantic import ValidationError

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    PrerequisiteKind,
)
from aiperf.common.models import DatasetMetadata, TurnPrerequisite
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, ThinkTimeSpec, Turn
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
    validate_fork_targets_not_payload_isolated,
    validate_non_terminal_branches,
    validate_pre_session_spawns_disjoint_from_forks,
    validate_system_message_placement,
)
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.dataset.loader.dag_jsonl_media import (
    DagMultimodalMaterializer,
    MediaStagingSummary,
    prepare_materializer_for_conversations,
)
from aiperf.dataset.loader.dag_jsonl_models import DagConversation, DagFork, DagTurn
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

__all__ = ["DagJsonlLoader", "DagLoadError"]


_MULTI_TURN_MODALITY_KEYS: frozenset[str] = frozenset(
    {"text", "texts", "image", "images", "audio", "audios", "video", "videos"}
)


def _is_dag_turn_probe(turn_obj: dict[str, Any]) -> bool:
    """Return True when ``turn_obj`` carries any DAG-specific turn indicator."""
    if not _MULTI_TURN_MODALITY_KEYS.isdisjoint(turn_obj):
        return False
    if isinstance(turn_obj.get("messages"), list):
        return True
    prompt_spans = turn_obj.get("prompt_spans")
    if isinstance(prompt_spans, list) and len(prompt_spans) > 0:
        return True
    media_segments = turn_obj.get("media_segments")
    if isinstance(media_segments, list) and len(media_segments) > 0:
        return True
    in_tok = turn_obj.get("in_text_tokens")
    if not isinstance(in_tok, bool) and isinstance(in_tok, int) and in_tok > 0:
        return True
    return "forks" in turn_obj or "spawns" in turn_obj


def _resolve_delay_cap_seconds(run: BenchmarkRun | None) -> float | None:
    """Pull ``inter_turn_delay_cap_seconds`` off the active file dataset.

    DAG loading goes through the v2 plugin contract, which passes ``run``
    (not a config object). The cap lives on the matching
    :class:`aiperf.config.dataset.config.FileDataset` entry; standalone
    construction (unit tests, offline tooling) passes ``run=None`` and
    the cap defaults to disabled.
    """
    if run is None:
        return None
    from aiperf.config.dataset import FileDataset

    for ds in run.cfg.datasets:
        if isinstance(ds, FileDataset) and ds.inter_turn_delay_cap_seconds is not None:
            return ds.inter_turn_delay_cap_seconds
    return None


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
    - ``spawns``: SPAWN branches. Children start fresh on the parent's
      worker when its sticky entry is live (least-loaded otherwise), then
      stay pinned to that worker under their own id (no parent refcount
      bump). Bare-string auto-joins on the next turn, ``DagSpawn`` objects
      carry an explicit ``join_at``.

    Both keywords may appear on the same turn; they desugar into separate
    ``ConversationBranchInfo`` entries with distinct ``branch_id``s.

    ``messages`` is concatenated onto the session's accumulator on each turn
    (pure append). Authors should place a single ``system`` entry on the
    root/seed turn only — ``system`` entries on non-root turns are rejected
    at load time because popular chat templates (e.g. Qwen3-VL) ignore
    system messages after position 0.

    Constructor shapes:
    - Plugin contract: ``DagJsonlLoader(filename=..., run=...)``
    - Standalone: ``DagJsonlLoader(path)`` (unit tests, offline tooling).

    Standalone Python usage (offline tooling, tests)::

        >>> loader = DagJsonlLoader("tests/fixtures/dag/small.dag.jsonl")
        >>> conversations = loader.load()  # list[Conversation]
        >>> roots = loader.root_session_ids()  # set[str]; not referenced as a child

    Plugin usage (the framework calls these, not user code)::

        loader = DagJsonlLoader(filename=path, run=run)
        per_session = loader.load_dataset()  # dict[session_id, list[Conversation]]
        conversations = loader.convert_to_conversations(per_session)
    """

    def __init__(
        self,
        filename: str | Path | None = None,
        *,
        run: BenchmarkRun | None = None,
        materializer: DagMultimodalMaterializer | None = None,
        media_scratch_dir: str | Path | None = None,
        media_base_url: str | None = None,
        video_wire_mode: Literal["video_url", "image_frames"] = "video_url",
        emit_media_uuid: bool = False,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if filename is None:
            raise ValueError(
                "DagJsonlLoader requires a 'filename' (or positional path); "
                "plugin callers pass filename=... + run=..., standalone "
                "callers pass the path positionally"
            )
        if run is not None:
            super().__init__(filename=str(filename), run=run, **kwargs)
            cap_seconds = _resolve_delay_cap_seconds(run)
        else:
            # Standalone path: bypass BaseFileLoader (no run available).
            self.run = None
            self.filename = str(filename)
            cap_seconds = None
        self._path = Path(filename)
        self._delay_cap_tracker = DelayCapTracker(cap_seconds=cap_seconds)
        self._materializer_arg = materializer
        self._media_scratch_dir = (
            Path(media_scratch_dir) if media_scratch_dir is not None else None
        )
        self._media_base_url = media_base_url
        self._video_wire_mode: Literal["video_url", "image_frames"] = video_wire_mode
        self._emit_media_uuid = emit_media_uuid
        self._tokenizer = tokenizer
        self.media_staging_summary: MediaStagingSummary | None = None
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
        self._load_lock = threading.RLock()

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True when data looks like a DAG conversation line.

        DAG lines have top-level ``session_id`` and ``turns`` where at least
        one turn carries a ``messages`` array, structural synthesis fields
        (non-empty ``prompt_spans`` / ``media_segments`` or positive
        ``in_text_tokens``), ``forks``, or ``spawns``, and no turn carries
        ordinary ``MultiTurn`` modality keys.
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
        has_dag_turn = False
        for t in turns:
            if not isinstance(t, dict) or not _MULTI_TURN_MODALITY_KEYS.isdisjoint(t):
                return False
            if _is_dag_turn_probe(t):
                has_dag_turn = True
        return has_dag_turn

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
        with self._load_lock:
            if self._loaded:
                return {sid: [conv] for sid, conv in self._conversations.items()}
            self._conversations.clear()
            self._inline_forks.clear()
            self._inline_spawns.clear()
            self._inline_pre_session_spawns.clear()
            self._roots.clear()
            self.media_staging_summary = None
            self._parse_lines()
            self._desugar_forks()
            self._resolve_and_validate()
            self._roots = self._compute_roots()
            for sid, conv in self._conversations.items():
                # Respect an author-declared context mode (e.g.
                # MESSAGE_ARRAY_WITH_RESPONSES for payload-isolated turns that
                # send each turn's authored message array as-is, with no
                # accumulation of prior turns or live responses). Default to the
                # accumulating multi-turn-chat mode when unset.
                if conv.context_mode is None:
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

    async def load_dataset_async(self) -> dict[str, list[Conversation]]:
        """Parse and stage the DAG JSONL file off the event loop."""
        return await asyncio.to_thread(self.load_dataset)

    async def load_async(self) -> list[Conversation]:
        """Async standalone helper offloading file and media staging I/O."""
        return await asyncio.to_thread(self.load)

    def root_session_ids(self) -> set[str]:
        with self._load_lock:
            if not self._loaded:
                self.load_dataset()
            return set(self._roots)

    # --- Internal parsing ----------------------------------------------------

    def _parse_lines(self) -> None:
        parsed_convs: list[DagConversation] = []
        seen_sids: set[str] = set()
        prefix = f"failed to parse DAG JSONL '{self._path}'"
        with self._path.open("rb") as f:
            for lineno, raw in enumerate(f, start=1):
                # Tolerate a UTF-8 BOM on the first line (common from
                # Windows/Excel exports). Bare bytes everywhere else.
                if lineno == 1 and raw.startswith(b"\xef\xbb\xbf"):
                    raw = raw[3:]
                raw = raw.strip()
                if not raw:
                    continue
                dag_conv = self._validate_raw_line(lineno, raw, seen_sids, prefix)
                parsed_convs.append(dag_conv)
        if not parsed_convs:
            raise DagLoadError(
                f"DAG JSONL file '{self._path}' is empty (no conversations "
                "parsed); supply at least one conversation line"
            )
        try:
            mat, summary = prepare_materializer_for_conversations(
                parsed_convs,
                materializer=self._materializer_arg,
                media_scratch_dir=self._media_scratch_dir,
                media_base_url=self._media_base_url,
                video_wire_mode=self._video_wire_mode,
                emit_media_uuid=self._emit_media_uuid,
                tokenizer=self._tokenizer,
            )
        except (ValueError, RuntimeError, OSError) as e:
            raise DagLoadError(f"{prefix}: multimodal staging failed: {e}") from e
        self.media_staging_summary = summary
        for dag_conv in parsed_convs:
            self._populate_conversation(dag_conv, materializer=mat)

    def _validate_raw_line(
        self, lineno: int, raw: bytes, seen_sids: set[str], prefix: str
    ) -> DagConversation:
        try:
            obj = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            raise DagLoadError(f"{prefix} line {lineno}: invalid JSON: {e}") from e
        try:
            dag_conv = DagConversation.model_validate(obj)
        except ValidationError as e:
            raise DagLoadError(f"{prefix}: {format_validation_error(lineno, e)}") from e
        sid = dag_conv.session_id
        if sid in seen_sids:
            raise DagLoadError(f"{prefix} line {lineno}: duplicate session_id '{sid}'")
        seen_sids.add(sid)
        self._inline_pre_session_spawns[sid] = self._check_pre_session_duplicates(
            sid, prefix, lineno, dag_conv.pre_session_spawns
        )
        return dag_conv

    def _build_think_time_spec(
        self, dag_conv: DagConversation, cap_ms: float | None
    ) -> ThinkTimeSpec | None:
        """Build the sampled think-time spec, folding the delay cap into its
        bounds. Both bounds are clamped so ``min <= max`` holds even when an
        authored bound already exceeds the cap. None when no distribution."""
        if dag_conv.think_time_sigma is None:
            return None
        spec_min = dag_conv.think_time_min_ms
        spec_max = dag_conv.think_time_max_ms
        if cap_ms is not None:
            spec_max = cap_ms if spec_max is None else min(spec_max, cap_ms)
            if spec_min is not None:
                spec_min = min(spec_min, cap_ms)
        return ThinkTimeSpec(
            sigma=dag_conv.think_time_sigma, min_ms=spec_min, max_ms=spec_max
        )

    @staticmethod
    def _build_round_plans(
        dag_conv: DagConversation,
    ) -> list[tuple[list[str], float]]:
        """Normalize both ``rounds`` forms to a per-round ``(spawns, think_ms)``
        plan. INT form re-fires the shared conversation-level ``spawns``; LIST
        form authors each round's own ``spawns`` (and optional think-time)."""
        default_think_ms = dag_conv.think_time_ms or 0.0
        if isinstance(dag_conv.rounds, list):
            return [
                (
                    list(r.spawns),
                    r.think_time_ms
                    if r.think_time_ms is not None
                    else default_think_ms,
                )
                for r in dag_conv.rounds
            ]
        return [
            (list(dag_conv.spawns), default_think_ms) for _ in range(dag_conv.rounds)
        ]

    def _populate_conversation(
        self,
        dag_conv: DagConversation,
        *,
        materializer: DagMultimodalMaterializer | None = None,
    ) -> None:
        sid = dag_conv.session_id
        turns: list[Turn] = []
        inline_forks_per_turn: list[list[DagFork]] = []
        inline_spawns_per_turn: list[list[tuple[list[str], int | None]]] = []
        # The configured inter-turn delay cap (``--inter-turn-delay-cap-seconds``)
        # bounds orchestrator think-time exactly as it bounds trace-replay delays.
        # Apply it at load time: fold it into the sampled distribution's clamps
        # AND onto each stamped fixed delay below, so neither a fixed nor a
        # sampled think-time can exceed the cap.
        cap_seconds = self._delay_cap_tracker.cap_seconds
        cap_ms = cap_seconds * 1000.0 if cap_seconds is not None else None
        think_time_spec = self._build_think_time_spec(dag_conv, cap_ms)
        if dag_conv.orchestrator and dag_conv.rounds is not None:
            for k, (round_spawns, round_think_ms) in enumerate(
                self._build_round_plans(dag_conv)
            ):
                capped_delay = self._delay_cap_tracker.clamp(round_think_ms)
                turns.append(Turn(no_request=True, delay=capped_delay))
                inline_forks_per_turn.append([])
                inline_spawns_per_turn.append([(round_spawns, k + 1)])
            turns.append(Turn(no_request=True))
            inline_forks_per_turn.append([])
            inline_spawns_per_turn.append([])
        elif dag_conv.orchestrator:
            turns.append(Turn(no_request=True))
            inline_forks_per_turn.append([])
            inline_spawns_per_turn.append(group_spawn_entries(list(dag_conv.spawns)))
        else:
            for turn_idx, t in enumerate(dag_conv.turns):
                turns.append(
                    self._build_turn(
                        t,
                        materializer=materializer,
                        fallback_key=f"{sid}:{turn_idx}",
                    )
                )
                inline_forks_per_turn.append([normalize_fork_entry(e) for e in t.forks])
                inline_spawns_per_turn.append(group_spawn_entries(t.spawns))
        self._conversations[sid] = Conversation(
            session_id=sid,
            turns=turns,
            is_orchestrator=dag_conv.orchestrator,
            think_time=think_time_spec,
            context_mode=dag_conv.context_mode,
        )
        self._inline_forks[sid] = inline_forks_per_turn
        self._inline_spawns[sid] = inline_spawns_per_turn

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

    def _build_turn(
        self,
        t: DagTurn,
        *,
        materializer: DagMultimodalMaterializer | None = None,
        fallback_key: str = "turn",
    ) -> Turn:
        if materializer is not None:
            raw_messages = materializer.materialize_turn_messages(
                t, fallback_key=fallback_key
            )
        else:
            raw_messages = list(t.messages or [])
        return Turn(
            raw_messages=raw_messages,
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
        # Orchestrator conversations have no turns; the pre-session branch lives at
        # the conversation level only. Non-orchestrators still record turn-0 membership
        # so the validator/dispatch turn-0 checks keep working.
        if conv.turns:
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
                is_background=background,
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
        validate_fork_targets_not_payload_isolated(self._conversations)
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
