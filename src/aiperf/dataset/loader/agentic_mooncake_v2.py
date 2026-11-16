# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Dynamo agentic Mooncake v2 request graphs for finite replay."""

from __future__ import annotations

import math
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, ValidationError

from aiperf.common.enums import (
    AgenticReplayLifecycle,
    ConversationBranchMode,
    ReplayDependencyEvent,
)
from aiperf.common.environment import Environment
from aiperf.common.models import (
    Conversation,
    ConversationBranchInfo,
    DatasetMetadata,
    ReplayTurnReference,
    Text,
    Turn,
)
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.dataset.loader.base_trace_loader import (
    _compute_file_hash,
    _has_meaningful_synthesis,
)
from aiperf.dataset.loader.hash_ids_synthesis import (
    HashIdsPromptRequest,
    HashIdsPromptSynthesisMixin,
)
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.generator.prompt import PromptGenerator


class _Source(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = Field(min_length=1, description="Source format.")
    digest: str = Field(min_length=1, description="Source digest.")


class _Header(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["dynamo.agentic_mooncake"] = Field(
        alias="schema", description="Trace schema."
    )
    version: Literal[2] = Field(description="Trace schema version.")
    block_size: int = Field(gt=0, description="Tokens represented by one hash.")
    hash_id_scope: Literal["local"] = Field(description="Hash namespace scope.")
    source: _Source = Field(description="Trace provenance.")


class _Dependency(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, description="Predecessor request ID.")
    trigger: Literal["dispatch", "completion"] = Field(description="Transport event.")
    delay_ms: float = Field(
        ge=0,
        le=(2**63 - 1) / 1_000_000,
        allow_inf_nan=False,
        description="Delay after event.",
    )
    relation: Literal["sequence", "spawn", "join", "replay_barrier"] = Field(
        description="Source edge relation."
    )


class _Request(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(min_length=1, description="Unique request ID.")
    play_id: str = Field(min_length=1, description="Play containing this request.")
    source_play_ordinal: int | None = Field(
        default=None, ge=0, description="Optional source play order."
    )
    session_id: str = Field(min_length=1, description="Source session ID.")
    model: str = Field(min_length=1, description="Recorded model name.")
    input_length: int = Field(gt=0, alias="input_tokens", description="Input tokens.")
    output_length: int = Field(
        ge=0, alias="output_tokens", description="Output tokens."
    )
    output_token_ids: list[int] | None = Field(
        default=None, description="Recorded output token IDs."
    )
    hash_ids: list[Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)]] = Field(
        min_length=1, description="Input prefix block hashes."
    )
    not_before_ms: float = Field(
        ge=0,
        le=(2**63 - 1) / 1_000_000,
        allow_inf_nan=False,
        description="Play-relative request floor.",
    )
    recorded_api_time_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Recorded service duration.",
    )
    priority: int | None = Field(
        default=None, description="Source scheduling priority."
    )
    strict_priority: int | None = Field(
        default=None, ge=0, description="Source strict priority."
    )
    policy_class: str | None = Field(default=None, description="Source policy class.")
    dependencies: list[_Dependency] = Field(
        default_factory=list, description="Required predecessor events."
    )


def _session_parents(
    requests: list[_Request], requests_by_id: dict[str, _Request]
) -> dict[str, str]:
    """Return unambiguous source-session parents without guessing cyclic ancestry."""
    candidates: dict[str, set[str]] = {
        request.session_id: set() for request in requests
    }
    for request in requests:
        for dependency in request.dependencies:
            if dependency.relation == "spawn":
                candidates[request.session_id].add(
                    requests_by_id[dependency.request_id].session_id
                )
    valid: dict[str, bool] = {}
    for session_id in candidates:
        path: list[str] = []
        seen: set[str] = set()
        current = session_id
        while current not in valid and current not in seen:
            seen.add(current)
            path.append(current)
            parents = candidates[current]
            if len(parents) != 1:
                valid[current] = not parents
                break
            current = next(iter(parents))
        resolved = valid.get(current, False)
        for visited in path:
            valid[visited] = resolved
    return {
        session_id: next(iter(parents))
        for session_id, parents in candidates.items()
        if valid[session_id] and parents
    }


class AgenticMooncakeV2Loader(HashIdsPromptSynthesisMixin, BaseFileLoader):
    """Compile each v2 play into a finite replay graph with a virtual root."""

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        inline_records: list[dict[str, Any]]
        | dict[str, list[dict[str, Any]]]
        | None = None,
        prompt_generator: PromptGenerator,
        run: BenchmarkRun | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            filename=filename, inline_records=inline_records, run=run, **kwargs
        )
        self.prompt_generator = prompt_generator
        cfg = self.run.cfg
        if Environment.HTTP.X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID:
            raise ValueError(
                "Agentic Mooncake v2 requires source session IDs; disable "
                "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"
            )
        if run is not None and any(
            phase.agentic_replay_lifecycle != AgenticReplayLifecycle.FINITE
            for phase in cfg.get_profiling_phases()
        ):
            raise ValueError(
                "Agentic Mooncake v2 requires --agentic-replay-lifecycle finite"
            )
        dataset = cfg.get_default_dataset()
        synthesis = getattr(dataset, "synthesis", None)
        if (
            _has_meaningful_synthesis(synthesis)
            or getattr(synthesis, "max_isl", None) is not None
            or getattr(synthesis, "max_osl", None) is not None
            or getattr(dataset, "ignore_trace_delays", False)
            or getattr(dataset, "use_think_time_only", False)
            or getattr(dataset, "inter_turn_delay_cap_seconds", None) is not None
        ):
            raise ValueError(
                "Agentic Mooncake v2 does not support trace transformations"
            )
        tokenizer = cfg.tokenizer
        models = cfg.get_model_names()
        self._configured_models = models
        self._model_map: dict[str, str] = {}
        self._tokenizer_name = (
            getattr(prompt_generator.tokenizer, "resolved_name", None)
            or (tokenizer.name if tokenizer is not None else None)
            or (models[0] if models else "")
        )
        self._trust_remote_code = tokenizer.trust_remote_code if tokenizer else False
        self._tokenizer_revision = tokenizer.revision if tokenizer else None
        self._header: _Header | None = None
        self._block_size = 0

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        return bool(data and data.get("schema") == "dynamo.agentic_mooncake")

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[_Request]]:
        self._model_map.clear()
        records = iter(self._iter_record_dicts())
        try:
            self._header = _Header.model_validate(next(records))
        except StopIteration as exc:
            raise ValueError("Agentic Mooncake v2 trace is empty") from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid agentic Mooncake v2 header: {exc}") from exc

        self._block_size = self._header.block_size
        by_play: dict[str, list[_Request]] = {}
        ids: set[str] = set()
        for line, record in enumerate(records, start=2):
            try:
                request = _Request.model_validate(record)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid agentic Mooncake v2 row {line}: {exc}"
                ) from exc
            if request.request_id in ids:
                raise ValueError(f"Duplicate request_id {request.request_id!r}")
            ids.add(request.request_id)
            if (
                len(request.hash_ids)
                != (request.input_length + self._block_size - 1) // self._block_size
            ):
                raise ValueError(
                    f"Request {request.request_id!r} has an invalid hash count"
                )
            if request.output_length == 0 or request.output_token_ids is not None:
                raise ValueError(
                    f"Request {request.request_id!r}: zero-length or fixed-token outputs "
                    "are not supported by AIPerf replay"
                )
            if (
                request.priority not in (None, 0)
                or request.strict_priority not in (None, 0)
                or request.policy_class is not None
            ):
                raise ValueError(
                    f"Request {request.request_id!r}: priority fields are not supported"
                )
            for dependency in request.dependencies:
                if (
                    dependency.relation != "spawn"
                    and dependency.trigger != "completion"
                ):
                    raise ValueError(
                        f"Request {request.request_id!r}: invalid dependency trigger"
                    )
            if request.model not in self._model_map and self._configured_models:
                self._model_map[request.model] = self._configured_models[
                    len(self._model_map) % len(self._configured_models)
                ]
            by_play.setdefault(request.play_id, []).append(request)
        if not by_play:
            raise ValueError("Agentic Mooncake v2 trace has no requests")
        for requests in by_play.values():
            requests.sort(key=lambda request: request.request_id)
        return by_play

    def convert_to_conversations(
        self, data: dict[str, list[_Request]]
    ) -> list[Conversation]:
        if self._header is None:
            raise ValueError("Load the agentic Mooncake v2 header first")
        requests_by_id = {
            request.request_id: request for rows in data.values() for request in rows
        }
        ordinals = {}
        for play_id, requests in data.items():
            play_ordinals = {request.source_play_ordinal for request in requests}
            if len(play_ordinals) != 1:
                raise ValueError(
                    f"Play {play_id!r} has inconsistent source_play_ordinal"
                )
            ordinals[play_id] = play_ordinals.pop()
            graph = {}
            for request in requests:
                graph[request.request_id] = []
                for dependency in request.dependencies:
                    predecessor = requests_by_id.get(dependency.request_id)
                    if predecessor is None or predecessor.play_id != play_id:
                        raise ValueError(
                            f"Request {request.request_id!r} has missing or cross-play "
                            f"dependency {dependency.request_id!r}"
                        )
                    graph[request.request_id].append(dependency.request_id)
            try:
                tuple(TopologicalSorter(graph).static_order())
            except CycleError as exc:
                raise ValueError(
                    f"Play {play_id!r} contains a dependency cycle"
                ) from exc
        nonempty_ordinals = [
            ordinal for ordinal in ordinals.values() if ordinal is not None
        ]
        if nonempty_ordinals and sorted(nonempty_ordinals) != list(range(len(data))):
            raise ValueError(
                "source_play_ordinal must be present and contiguous for all plays"
            )
        play_ids = sorted(
            data, key=lambda play: ordinals[play] if nonempty_ordinals else play
        )

        self.prompt_generator._cache.clear()
        self.prompt_generator._hash_id_corpus_rng.set_trace_id(
            _compute_file_hash(self.filename)
        )
        prompt_requests = [
            HashIdsPromptRequest(
                key=request.request_id,
                hash_ids=request.hash_ids,
                input_length=request.input_length,
            )
            for play_id in play_ids
            for request in data[play_id]
        ]
        prompts = self.synthesize_prompts_from_hash_ids(
            prompt_requests, complete_hash_blocks=True
        )
        conversations: list[Conversation] = []
        for play_id in play_ids:
            requests = data[play_id]
            session_parents = _session_parents(requests, requests_by_id)
            root_floor = min(r.not_before_ms for r in requests if not r.dependencies)
            root_id = f"v2-play:{play_id}"
            branch_id = f"{root_id}:0"
            children = [f"v2-request:{request.request_id}" for request in requests]
            conversations.append(
                Conversation(
                    session_id=root_id,
                    is_orchestrator=True,
                    turns=[Turn(timestamp=0, no_request=True, branch_ids=[branch_id])],
                    branches=[
                        ConversationBranchInfo(
                            branch_id=branch_id,
                            child_conversation_ids=children,
                            mode=ConversationBranchMode.SPAWN,
                            is_background=True,
                        )
                    ],
                )
            )
            for request, child_id in zip(requests, children, strict=True):
                headers = {"x-dynamo-session-id": request.session_id}
                if parent_session := session_parents.get(request.session_id):
                    headers["x-dynamo-parent-session-id"] = parent_session
                references = [
                    ReplayTurnReference(
                        conversation_id=f"v2-request:{dependency.request_id}",
                        turn_index=0,
                        event=ReplayDependencyEvent(dependency.trigger),
                        delay_ns=math.ceil(dependency.delay_ms * 1_000_000),
                        relation=dependency.relation,
                    )
                    for dependency in request.dependencies
                ]
                conversations.append(
                    Conversation(
                        session_id=child_id,
                        is_root=False,
                        agent_depth=1,
                        parent_conversation_id=root_id,
                        turns=[
                            Turn(
                                timestamp=max(0.0, request.not_before_ms - root_floor),
                                api_time_ms=request.recorded_api_time_ms,
                                source_trace_id=request.session_id,
                                source_kind="agentic_mooncake_v2",
                                model=self._model_map.get(request.model, request.model),
                                max_tokens=request.output_length,
                                extra_headers=headers,
                                texts=[
                                    Text(
                                        name="text",
                                        contents=[prompts[request.request_id]],
                                    )
                                ],
                                replay_predecessors=references,
                            )
                        ],
                    )
                )
        validate_for_orchestrator_v1(
            DatasetMetadata(
                conversations=[
                    conversation.metadata() for conversation in conversations
                ],
                sampling_strategy=self.get_preferred_sampling_strategy(),
            )
        )
        return conversations
