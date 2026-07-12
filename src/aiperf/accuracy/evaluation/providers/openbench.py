# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenBench/Inspect adapter over Inspect's public ``ModelAPI`` seam."""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.machinery
import importlib.metadata
import math
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiperf.accuracy.evaluation.contracts import (
    AggregateMetric,
    AggregationPolicy,
    ArtifactRef,
    ArtifactVisibility,
    AssetRequirement,
    CallContext,
    CaseOccurrenceDescriptor,
    CaseOutcome,
    CaseOutcomeKind,
    CaseTemplateDescriptor,
    EvaluationHostBinding,
    EvaluationIdentityComponent,
    EvaluationPlan,
    EvaluationPlanRequest,
    EvaluationQueueCredits,
    EvaluationWorkerIdentity,
    ExecutionGranularity,
    ExecutionUnitOccurrence,
    ExecutionUnitTemplateDescriptor,
    HostCapabilityRequirement,
    HostOperationRequest,
    LogicalServiceRequirement,
    ProviderScore,
    ResolvedAsset,
    ResponseMode,
    SchedulingMode,
    ScopedProxyBinding,
)
from aiperf.accuracy.evaluation.distributions import (
    OPENBENCH_DISTRIBUTION,
    task_manifest,
)
from aiperf.accuracy.evaluation.host import PipeEvaluationHost, terminal_result_payload
from aiperf.accuracy.evaluation.operation_schemas import OPERATION_SCHEMA_SHA256
from aiperf.accuracy.evaluation.providers.base import ProviderCapabilityError
from aiperf.accuracy.evaluation.providers.gsm8k import (
    ASSET_ID,
    ASSET_MEDIA_TYPE,
    ASSET_REVISION,
    ASSET_SHA256,
    SOURCE_LABEL,
    bind_gsm8k_asset,
    build_identity,
    finish_candidate,
)
from aiperf.accuracy.evaluation.session import (
    BaseEvaluationSession,
    EvaluationSession,
    infrastructure_outcome,
)


@dataclass
class _InspectCallState:
    context: CallContext
    next_ordinal: int = 0


_CURRENT_CONTEXT: contextvars.ContextVar[_InspectCallState | None] = (
    contextvars.ContextVar("aiperf_inspect_model_call_context", default=None)
)


class OpenBenchAdapter:
    """Side-effect-free planner for pinned OpenBench plus Inspect AI."""

    def __init__(self, worker_identity: EvaluationWorkerIdentity) -> None:
        self._worker_identity = worker_identity
        self._request: EvaluationPlanRequest | None = None
        self._plan: EvaluationPlan | None = None

    def plan_session(self, request: EvaluationPlanRequest) -> EvaluationPlan:
        """Resolve only one exact statically manifested OpenBench task."""
        descriptor = OPENBENCH_DISTRIBUTION
        if (
            request.provider_id != descriptor.provider_id
            or request.distribution_id != descriptor.distribution_id
        ):
            raise ValueError("OpenBench provider/distribution identity drift")
        if (
            request.config_schema_version != descriptor.config_schema_version
            or request.config_schema_sha256 != descriptor.config_schema_sha256
        ):
            raise ValueError("OpenBench authored-config schema drift")
        if not isinstance(request.provider_config, dict):
            raise TypeError("OpenBench provider_config must be an object")
        allowed = {"task", "task_args", "epochs", "limit"}
        unknown = set(request.provider_config) - allowed
        if unknown:
            raise ValueError(
                f"OpenBench provider_config has unknown fields: {sorted(unknown)}"
            )
        task = request.provider_config.get("task")
        if not isinstance(task, str):
            raise TypeError("OpenBench task must be a string")
        entry = task_manifest(descriptor).get("tasks", {}).get(task)
        if not isinstance(entry, dict):
            raise ProviderCapabilityError(
                f"OpenBench task {task!r} is not in the frozen manifest"
            )
        if entry.get("executable") is not True:
            raise ProviderCapabilityError("OpenBench manifest entry is not executable")
        task_args = request.provider_config.get("task_args")
        if task_args != {}:
            raise ValueError("OpenBench GSM8K canary accepts no task_args overrides")
        epochs = request.provider_config.get("epochs")
        if (
            not isinstance(epochs, int)
            or isinstance(epochs, bool)
            or not 1 <= epochs <= 8
        ):
            raise ValueError("OpenBench epochs must be an integer from 1 through 8")
        limit = request.provider_config.get("limit", entry["selection_count"])
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= entry["selection_count"]
        ):
            raise ValueError("OpenBench limit is outside frozen canary bounds")
        case_count = limit * epochs
        plan = EvaluationPlan(
            assets=(
                AssetRequirement(
                    asset_id=ASSET_ID,
                    source_kind="task_package",
                    immutable_revision=ASSET_REVISION,
                    content_sha256=ASSET_SHA256,
                    media_type=ASSET_MEDIA_TYPE,
                    visibility=ArtifactVisibility.RESTRICTED,
                ),
            ),
            host_requirements=(
                HostCapabilityRequirement(
                    capability_id="inference.model.generate.v1",
                    schema_sha256=OPERATION_SCHEMA_SHA256["model.generate"],
                ),
            ),
            logical_services=(
                LogicalServiceRequirement(
                    service_id="candidate",
                    purpose="primary",
                    operations=("model.generate",),
                ),
            ),
            aggregation_policy=AggregationPolicy(
                policy_id="openbench_inspect_gsm8k_mean_v1",
                exclude_infrastructure=True,
                exclude_cancelled=True,
                definition={"epoch_reducer": "mean", "epochs": epochs},
            ),
            execution_granularity=ExecutionGranularity.HOST_BATCH,
            scheduling_mode=SchedulingMode.FINITE,
            finite_unit_count=1,
            finite_case_count=case_count,
            queue_credits=EvaluationQueueCredits(
                units=1,
                host_operations=case_count,
                host_operations_per_unit=case_count,
                stream_events=max(64, case_count * 2),
                sandboxes=0,
                processes=0,
                artifacts=8,
                artifact_bytes=64 * 1024 * 1024,
            ),
        )
        self._request = request
        self._plan = plan
        return plan

    async def bind_assets(
        self,
        assets: Sequence[ResolvedAsset],
        proxy: ScopedProxyBinding | None,
        host_binding: EvaluationHostBinding,
        staging_root: Path,
    ) -> EvaluationSession:
        """Construct an asset-bound public Inspect task without registry lookup."""
        if self._plan is None or self._request is None:
            raise RuntimeError("bind_assets requires successful plan_session")
        if proxy is not None:
            raise ValueError("OpenBench GSM8K pipe canary does not request a proxy")
        asset_path, _ = bind_gsm8k_asset(assets)
        config = self._request.provider_config
        limit = config.get("limit", 5)
        epochs = config["epochs"]
        case_templates = tuple(
            CaseTemplateDescriptor(
                template_id=f"openbench-gsm8k-sample-{sample_id}-epoch-{epoch}",
                task="gsm8k",
                source=SOURCE_LABEL,
            )
            for epoch in range(1, epochs + 1)
            for sample_id in range(1, limit + 1)
        )
        unit_template = ExecutionUnitTemplateDescriptor(
            unit_template_id="openbench-gsm8k-host-batch",
            case_template_ids=tuple(item.template_id for item in case_templates),
            granularity=ExecutionGranularity.HOST_BATCH,
            scheduling_class="inspect_host_batch",
        )
        occurrence = ExecutionUnitOccurrence(
            unit_id="openbench-gsm8k-host-batch-occurrence",
            unit_template_id=unit_template.unit_template_id,
            cases=tuple(
                CaseOccurrenceDescriptor(
                    case_id=f"openbench-gsm8k-case-{index}",
                    template_id=template.template_id,
                    issue_ordinal=index,
                    phase_id="evaluation",
                    cycle_index=0,
                )
                for index, template in enumerate(case_templates)
            ),
        )
        identity = build_identity(
            worker=self._worker_identity,
            config_schema_sha256=self._request.config_schema_sha256,
            provider_config=self._request.provider_config,
            case_templates=case_templates,
            unit_templates=(unit_template,),
            components=(
                EvaluationIdentityComponent(
                    name="openbench",
                    version="0.5.3@3f190a835f7fee34ccd96e17242a36a29e0620a6",
                    source_sha256="bdfcc39c2423619696d359970e75611dd0aadee6c87a383961b78ab705acf1d5",
                ),
                EvaluationIdentityComponent(
                    name="inspect_ai",
                    version="0.3.141@bb78d82dde311b68dbfd0b49f3186b9fc13a1465",
                    source_sha256="6bd6016a593ebc0e976285e6416025a0c8a123d8451b0fc180da9a6a17d9794b",
                ),
            ),
            policies=self._plan.aggregation_policy.to_wire(),
            host_binding=host_binding,
        )
        return OpenBenchGsm8kSession(
            session_id=self._request.session_id,
            identity=identity,
            plan=self._plan,
            occurrence=occurrence,
            asset_path=asset_path,
            limit=limit,
            epochs=epochs,
            staging_root=staging_root,
        )


class OpenBenchGsm8kSession(BaseEvaluationSession):
    """One pinned OpenBench task executed as an Inspect host batch."""

    def __init__(
        self,
        *,
        session_id: str,
        identity: Any,
        plan: EvaluationPlan,
        occurrence: ExecutionUnitOccurrence,
        asset_path: Path,
        limit: int,
        epochs: int,
        staging_root: Path,
    ) -> None:
        super().__init__(session_id, identity, plan, (occurrence,))
        self._session_id = session_id
        self._asset_path = asset_path
        self._limit = limit
        self._epochs = epochs
        self._staging_root = staging_root
        self._eval_log: Any | None = None
        self._eval_artifact: Path | None = None

    def _case_for(self, sample_id: int | str, epoch: int) -> str:
        try:
            sample = int(sample_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Inspect sample ID is not the frozen integer identity"
            ) from error
        template_id = f"openbench-gsm8k-sample-{sample}-epoch-{epoch}"
        for case in self.units[0].cases:
            if case.template_id == template_id:
                return case.case_id
        raise ValueError("Inspect sample/epoch escaped the frozen manifest")

    async def run_unit(
        self, unit_id: str, host: PipeEvaluationHost
    ) -> Sequence[CaseOutcome]:
        if unit_id != self.units[0].unit_id:
            raise ValueError("OpenBench received an unknown host batch")
        try:
            from inspect_ai import Task, eval_async
            from inspect_ai.dataset import json_dataset
            from inspect_ai.model import GenerateConfig, Model
            from inspect_ai.solver import generate, solver

            record_to_sample, grade_school_math_scorer = _openbench_gsm8k_symbols()

            dataset = json_dataset(
                str(self._asset_path),
                sample_fields=record_to_sample,
                auto_id=True,
                limit=self._limit,
                name="openbench_gsm8k_canary",
            )
            base_solver = generate()
            base_scorer = grade_school_math_scorer()

            @solver("aiperf_contextual_generate")
            def contextual_generate() -> Any:
                async def solve(state: Any, generate_fn: Any) -> Any:
                    case_id = self._case_for(state.sample_id, state.epoch)
                    context = CallContext(
                        session_id=self._session_id,
                        unit_id=unit_id,
                        case_id=case_id,
                        semantic_attempt_id=f"attempt-{case_id}-0",
                        logical_call_id=f"call-{case_id}",
                    )
                    with inspect_model_call_context(context):
                        return await base_solver(state, generate_fn)

                return solve

            generate_config = GenerateConfig(
                temperature=0.0,
                max_tokens=2048,
                batch=False,
                attempt_timeout=None,
                max_retries=0,
            )
            task = Task(
                dataset=dataset,
                solver=[contextual_generate()],
                scorer=base_scorer,
                config=generate_config,
            )
            api = build_aiperf_pipe_model_api(host)
            model = Model(api, generate_config)
            logs = await eval_async(
                tasks=task,
                model=model,
                log_dir=str(self._staging_root),
                log_format="eval",
                limit=self._limit,
                epochs=self._epochs,
                fail_on_error=False,
                continue_on_fail=True,
                retry_on_error=0,
                max_sandboxes=None,
                log_samples=True,
                log_realtime=False,
                log_images=False,
                score_display=False,
            )
            if len(logs) != 1:
                raise RuntimeError("Inspect returned an unexpected task-log count")
            log = logs[0]
            self._eval_log = log
            candidates = tuple(self._staging_root.glob("*.eval"))
            if len(candidates) != 1:
                raise RuntimeError(
                    "Inspect did not produce exactly one contained .eval artifact"
                )
            self._eval_artifact = candidates[0]
            samples = {
                (int(sample.id), sample.epoch): sample for sample in (log.samples or [])
            }
            outcomes: list[CaseOutcome] = []
            for epoch in range(1, self._epochs + 1):
                for sample_id in range(1, self._limit + 1):
                    case_id = self._case_for(sample_id, epoch)
                    sample = samples.get((sample_id, epoch))
                    if sample is None or sample.error is not None or not sample.scores:
                        outcomes.append(
                            infrastructure_outcome(
                                case_id, "provider_execution", "inspect_sample_error"
                            )
                        )
                        continue
                    scores = {
                        name: ProviderScore(
                            value=score.model_dump(mode="json", exclude_none=True),
                        )
                        for name, score in sample.scores.items()
                    }
                    numeric = {
                        name: float(score.value)
                        for name, score in sample.scores.items()
                        if isinstance(score.value, int | float)
                        and not isinstance(score.value, bool)
                        and math.isfinite(float(score.value))
                    }
                    primary = next(iter(scores))
                    outcomes.append(
                        CaseOutcome(
                            case_id=case_id,
                            kind=CaseOutcomeKind.COMPLETED,
                            scores=scores,
                            numeric_metrics=numeric,
                            primary_score=primary,
                            artifact_refs=(
                                ArtifactRef(
                                    artifact_id="inspect_eval_log",
                                    path=self._eval_artifact.name,
                                    visibility=ArtifactVisibility.RESTRICTED,
                                ),
                            ),
                        )
                    )
            return tuple(outcomes)
        except asyncio.CancelledError:
            raise

    async def finalize(self) -> Any:
        if self._eval_log is None or self._eval_artifact is None:
            raise RuntimeError("OpenBench finalize requires a complete Inspect EvalLog")
        outcomes = self.outcomes
        aggregates: list[AggregateMetric] = []
        results = self._eval_log.results
        if results is not None:
            for score in results.scores:
                for metric_name, metric in score.metrics.items():
                    aggregates.append(
                        AggregateMetric(
                            scorer=score.scorer,
                            reducer=score.reducer or "identity",
                            metric=metric_name,
                            value=metric.value,
                            scored_count=score.scored_samples or 0,
                            unscored_count=score.unscored_samples or 0,
                            definition={
                                "score_name": score.name,
                                "params": score.params,
                                "metric_params": metric.params,
                            },
                        )
                    )
        return finish_candidate(
            identity=self.identity,
            outcomes=outcomes,
            aggregates=tuple(aggregates),
            restricted_records=self._eval_log.model_dump(
                mode="json", exclude_none=True
            ),
            staging_root=self._staging_root,
            filename="openbench_provider_bundle.json",
            additional_artifacts=(
                (
                    "inspect_eval_log",
                    self._eval_artifact,
                    "application/vnd.inspect-ai.eval",
                    ArtifactVisibility.RESTRICTED,
                ),
            ),
        )


def _openbench_gsm8k_symbols() -> tuple[Any, Any]:
    """Load only the attested OpenBench GSM8K leaves.

    OpenBench 0.5.3's top-level ``openbench.__init__`` eagerly imports its CLI
    and every optional coding agent, although the GSM8K task uses none of them.
    The stock AIPerf distribution deliberately contains no ``inspect_ai``
    entry point, so loading the top-level package would both broaden the
    executable surface and make this task depend on unrelated ``inspect_swe``
    packages.  This deterministic namespace bootstrap points Python at the
    already-attested distribution tree, then imports the exact provider leaf
    modules used by the adapter.  It performs no entry-point discovery and no
    provider source mutation.

    Pinned source:
    ``openbench/evals/gsm8k.py:13-30`` and
    ``openbench/scorers/grade_school_math.py:1-41`` at
    ``3f190a835f7fee34ccd96e17242a36a29e0620a6``.
    """
    distribution = importlib.metadata.distribution("openbench")
    root = Path(distribution.locate_file("openbench")).resolve(strict=True)
    if root.name != "openbench" or not root.is_dir():
        raise RuntimeError("attested OpenBench package tree is malformed")
    package_roots = {
        "openbench": root,
        "openbench.evals": root / "evals",
        "openbench.scorers": root / "scorers",
        "openbench.utils": root / "utils",
    }
    for name, package_root in package_roots.items():
        existing = sys.modules.get(name)
        if existing is not None:
            existing_path = tuple(getattr(existing, "__path__", ()))
            if existing_path != (str(package_root),):
                raise RuntimeError(
                    f"OpenBench namespace {name!r} was initialized outside the stock bootstrap"
                )
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(package_root)]
        module.__spec__ = importlib.machinery.ModuleSpec(
            name=name,
            loader=None,
            is_package=True,
        )
        sys.modules[name] = module
    task_module = importlib.import_module("openbench.evals.gsm8k")
    scorer_module = importlib.import_module("openbench.scorers.grade_school_math")
    return task_module.record_to_sample, scorer_module.grade_school_math_scorer


def build_aiperf_pipe_model_api(
    host: PipeEvaluationHost,
    *,
    service_id: str = "candidate",
    purpose: str = "primary",
) -> Any:
    """Construct an Inspect ``ModelAPI`` using only the official public seam.

    Import is intentionally lazy and occurs only after distribution attestation
    and asset binding.  The object never accepts or constructs a provider URL,
    API key, SDK client, cache, batch request, or retry policy.
    """
    from inspect_ai.model import ModelAPI, modelapi

    @modelapi(name="aiperf-pipe")
    class AiperfPipeModelAPI(ModelAPI):
        """Inspect ModelAPI implementation backed by evaluator host pipes."""

        def __init__(self) -> None:
            super().__init__(
                model_name=f"aiperf/{service_id}",
                base_url=None,
                api_key=None,
                api_key_vars=[],
            )

        async def generate(
            self,
            input: list[Any],
            tools: list[Any],
            tool_choice: Any,
            config: Any,
        ) -> Any:
            state = _CURRENT_CONTEXT.get()
            if state is None:
                raise RuntimeError(
                    "Inspect model call has no active sample/epoch context"
                )
            ordinal = state.next_ordinal
            state.next_ordinal += 1
            logical_call_id = f"{state.context.logical_call_id}-{ordinal}"
            call_context = CallContext(
                session_id=state.context.session_id,
                unit_id=state.context.unit_id,
                case_id=state.context.case_id,
                semantic_attempt_id=state.context.semantic_attempt_id,
                logical_call_id=logical_call_id,
            )
            payload = _inspect_generate_payload(input, tools, tool_choice, config)
            request = HostOperationRequest(
                operation_id=f"op-{logical_call_id}",
                context=call_context,
                service_id=service_id,
                purpose=purpose,
                semantic_operation_id="model.generate",
                payload=payload,
                response_mode=ResponseMode.TERMINAL,
                idempotency_key=f"host-{logical_call_id}",
            )
            result = terminal_result_payload(await host.execute(request))
            if not isinstance(result, dict):
                raise ValueError("Inspect host result must be an object")
            from inspect_ai.model import (
                ChatCompletionChoice,
                ChatMessageAssistant,
                ModelCall,
                ModelOutput,
                ModelUsage,
            )

            raw_choices = result.get("choices")
            if not isinstance(raw_choices, list) or not raw_choices:
                raise ValueError("Inspect host result omitted choices")
            choices = []
            for raw_choice in raw_choices:
                if not isinstance(raw_choice, dict) or not isinstance(
                    raw_choice.get("message"), dict
                ):
                    raise ValueError("Inspect host result contained malformed choice")
                choices.append(
                    ChatCompletionChoice(
                        message=ChatMessageAssistant.model_validate(
                            raw_choice["message"]
                        ),
                        stop_reason=raw_choice.get("stop_reason", "unknown"),
                        logprobs=raw_choice.get("logprobs"),
                    )
                )
            raw_usage = result.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raise ValueError("Inspect host result usage must be an object")
            prompt_tokens = raw_usage.get("prompt_tokens") or 0
            completion_tokens = raw_usage.get("completion_tokens") or 0
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                raise ValueError("Inspect host usage token counts must be integers")
            output = ModelOutput(
                model=f"aiperf/{service_id}",
                choices=choices,
                usage=ModelUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    input_tokens_cache_read=raw_usage.get("cached_tokens"),
                    reasoning_tokens=raw_usage.get("reasoning_tokens"),
                ),
            )
            # Inspect's complete EvalLog is a restricted artifact.  Returning a
            # ModelCall here preserves provider-native trace semantics without
            # publishing its prompt/response in AIPerf's public projection.
            call = ModelCall(request=payload, response=result)
            return output, call

        def should_retry(self, ex: Exception) -> bool:
            del ex
            return False

        def allows_cache(self) -> bool:
            """Veto Inspect model cache in measured provider mode."""
            return False

        def max_connections(self) -> int:
            # This bounds only Python producers. Rust SlotPools remain the sole
            # network/inference admission authority.
            return host.producer_capacity

    return AiperfPipeModelAPI()


class inspect_model_call_context:
    """Task-local context wrapper for public Inspect solver/scorer calls."""

    def __init__(self, context: CallContext) -> None:
        self._state = _InspectCallState(context=context)
        self._token: contextvars.Token[_InspectCallState | None] | None = None

    def __enter__(self) -> None:
        self._token = _CURRENT_CONTEXT.set(self._state)

    def __exit__(self, *_: object) -> None:
        assert self._token is not None
        _CURRENT_CONTEXT.reset(self._token)


def _inspect_generate_payload(
    input: list[Any], tools: list[Any], tool_choice: Any, config: Any
) -> dict[str, Any]:
    messages = [_inspect_message(item) for item in input]
    serialized_tools = [_inspect_tool(item) for item in tools]
    serialized_choice = _pydantic_json(tool_choice, "Inspect tool choice")
    raw_config = _pydantic_json(config, "Inspect GenerateConfig")
    forbidden_non_null = {
        "max_retries",
        "timeout",
        "attempt_timeout",
        "extra_body",
        "cache_prompt",
    }
    present = sorted(
        field
        for field in forbidden_non_null
        if raw_config.get(field) is not None
        and not (field == "max_retries" and raw_config.get(field) == 0)
    )
    if present:
        raise ValueError(
            f"Inspect config requests forbidden retry/cache/authority fields: {present}"
        )
    batch = raw_config.get("batch")
    if batch not in (None, False):
        raise ValueError("Inspect provider batching is forbidden")
    max_tokens = raw_config.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("Inspect model.generate requires max_tokens > 0")
    # max_connections is a Python producer bound; it does not enter the model
    # semantic request. system_message has already been incorporated by Inspect.
    for field in (
        "max_retries",
        "timeout",
        "attempt_timeout",
        "max_connections",
        "system_message",
        "extra_body",
        "batch",
        "cache_prompt",
    ):
        raw_config.pop(field, None)
    response_format = raw_config.pop("response_schema", None)
    generation_fields = {"max_tokens", "temperature", "top_p", "stop_seqs"}
    generation = {
        ("stop" if key == "stop_seqs" else key): raw_config.pop(key)
        for key in tuple(raw_config)
        if key in generation_fields and raw_config[key] is not None
    }
    payload: dict[str, Any] = {
        "messages": messages,
        "generation": generation,
    }
    if serialized_tools:
        payload["tools"] = serialized_tools
        payload["tool_choice"] = serialized_choice
    if response_format is not None:
        payload["response_format"] = response_format
    parameters = {key: value for key, value in raw_config.items() if value is not None}
    if parameters:
        payload["parameters"] = parameters
    return payload


def _pydantic_json(value: Any, label: str) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    if isinstance(value, str | int | float | bool | list | dict):
        return value
    raise TypeError(f"{label} cannot be serialized losslessly")


def _inspect_message(value: Any) -> dict[str, Any]:
    raw = _pydantic_json(value, "Inspect message")
    if not isinstance(raw, dict):
        raise TypeError("Inspect message must serialize to an object")
    role = raw.get("role")
    content = raw.get("content")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError("Inspect message has an unsupported role")
    # The audited GSM8K task is text-only. Rich media/tool tasks are not in
    # this task manifest and must add asset-aware normalization before being
    # advertised; silently forwarding a path/data URI would broaden authority.
    if not isinstance(content, str):
        raise ValueError("OpenBench GSM8K manifest permits text content only")
    normalized: dict[str, Any] = {"role": role, "content": content}
    if role == "tool":
        call_id = raw.get("tool_call_id")
        if not isinstance(call_id, str):
            raise ValueError("Inspect tool message omitted tool_call_id")
        normalized["tool_call_id"] = call_id
    if role == "assistant" and raw.get("tool_calls"):
        normalized["tool_calls"] = [
            _inspect_tool_call(item) for item in raw["tool_calls"]
        ]
    return normalized


def _inspect_tool(value: Any) -> dict[str, Any]:
    raw = _pydantic_json(value, "Inspect tool")
    if not isinstance(raw, dict):
        raise TypeError("Inspect tool must serialize to an object")
    if raw.get("options"):
        raise ValueError("Inspect tool provider options are outside the audited schema")
    name = raw.get("name")
    parameters = raw.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, dict):
        raise ValueError("Inspect tool has malformed name/parameters")
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if isinstance(raw.get("description"), str):
        function["description"] = raw["description"]
    return {"type": "function", "function": function}


def _inspect_tool_call(value: Any) -> dict[str, Any]:
    raw = _pydantic_json(value, "Inspect tool call")
    if not isinstance(raw, dict):
        raise TypeError("Inspect tool call must serialize to an object")
    if raw.get("type", "function") != "function" or raw.get("parse_error"):
        raise ValueError("Inspect custom or parse-error tool calls are unsupported")
    return {
        "id": raw["id"],
        "type": "function",
        "function": {
            "name": raw["function"],
            "arguments": raw["arguments"],
        },
    }
