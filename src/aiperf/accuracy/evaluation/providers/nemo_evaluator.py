# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Evaluator adapter and host-backed solver client.

The model client preserves the public solver-facing ``chat``/``vlm_chat``/
``chat_with_tools``/``embed`` behavior while removing endpoint, key, HTTP,
cache, retry, semaphore, and retry-sleep ownership.  Every model effect lowers
to one typed Rust operation through :class:`PipeEvaluationHost`.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aiperf.accuracy.evaluation.contracts import (
    AggregateMetric,
    AggregationPolicy,
    ArtifactVisibility,
    AssetRequirement,
    CallContext,
    CaseOccurrenceDescriptor,
    CaseOutcome,
    CaseOutcomeKind,
    CaseTemplateDescriptor,
    EvaluationHostBinding,
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
    NEMO_EVALUATOR_DISTRIBUTION,
    distribution_identity_components,
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

_CURRENT_CONTEXT: contextvars.ContextVar[CallContext | None] = contextvars.ContextVar(
    "aiperf_nemo_model_call_context", default=None
)
_CURRENT_CALL_ORDINAL: contextvars.ContextVar[int] = contextvars.ContextVar(
    "aiperf_nemo_model_call_ordinal", default=0
)


class NemoEvaluatorAdapter:
    """Side-effect-free planner for the pinned NeMo Evaluator distribution."""

    def __init__(self, worker_identity: EvaluationWorkerIdentity) -> None:
        self._worker_identity = worker_identity
        self._request: EvaluationPlanRequest | None = None
        self._plan: EvaluationPlan | None = None

    def plan_session(self, request: EvaluationPlanRequest) -> EvaluationPlan:
        """Validate an exact manifest entry without importing NeMo Evaluator."""
        descriptor = NEMO_EVALUATOR_DISTRIBUTION
        if (
            request.provider_id != descriptor.provider_id
            or request.distribution_id != descriptor.distribution_id
        ):
            raise ValueError("NeMo Evaluator provider/distribution identity drift")
        if (
            request.config_schema_version != descriptor.config_schema_version
            or request.config_schema_sha256 != descriptor.config_schema_sha256
        ):
            raise ValueError("NeMo Evaluator authored-config schema drift")
        if not isinstance(request.provider_config, dict):
            raise TypeError("NeMo Evaluator provider_config must be an object")
        allowed = {
            "environment",
            "solver",
            "environment_config",
            "solver_config",
            "selection",
        }
        if set(request.provider_config) - allowed:
            raise ValueError("NeMo Evaluator provider_config contains unknown fields")
        environment = request.provider_config.get("environment")
        solver = request.provider_config.get("solver")
        if not isinstance(environment, str) or not isinstance(solver, str):
            raise TypeError("NeMo Evaluator environment/solver must be strings")
        manifest = task_manifest(descriptor)
        entry = manifest.get("environments", {}).get(environment)
        if not isinstance(entry, dict):
            raise ProviderCapabilityError(
                f"NeMo Evaluator environment {environment!r} is not in the frozen manifest"
            )
        if solver not in entry.get("solver_kinds", []):
            raise ProviderCapabilityError(
                f"NeMo Evaluator solver {solver!r} is not declared for {environment!r}"
            )
        if entry.get("executable") is not True:
            raise ProviderCapabilityError(
                "NeMo Evaluator manifest entry is not executable"
            )
        selection = request.provider_config.get("selection")
        if not isinstance(selection, dict) or set(selection) != {"limit", "seed"}:
            raise ValueError("NeMo Evaluator selection requires exactly limit and seed")
        limit = selection["limit"]
        seed = selection["seed"]
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= entry["selection_count"]
        ):
            raise ValueError(
                "NeMo Evaluator selection limit is outside frozen canary bounds"
            )
        if seed != 0:
            raise ValueError("NeMo Evaluator GSM8K canary fixes selection seed to zero")
        environment_config = request.provider_config.get("environment_config", {})
        if environment_config != {}:
            raise ValueError("GSM8K canary accepts no environment_config overrides")
        solver_config = request.provider_config.get("solver_config", {})
        if not isinstance(solver_config, dict):
            raise TypeError("solver_config must be an object")
        allowed_solver = {
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        }
        if set(solver_config) - allowed_solver:
            raise ValueError("GSM8K solver_config contains unsupported fields")
        max_tokens = solver_config.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError("GSM8K solver_config requires max_tokens > 0")
        for field in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = solver_config.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"GSM8K solver_config {field} must be finite numeric")
        solver_seed = solver_config.get("seed")
        if solver_seed is not None and (
            isinstance(solver_seed, bool) or not isinstance(solver_seed, int)
        ):
            raise ValueError("GSM8K solver_config seed must be an integer")
        stop = solver_config.get("stop")
        if stop is not None and (
            not isinstance(stop, list)
            or not all(isinstance(item, str) for item in stop)
        ):
            raise ValueError("GSM8K solver_config stop must be an array of strings")
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
                policy_id="nemo_gsm8k_reward_mean_v1",
                exclude_infrastructure=True,
                exclude_cancelled=True,
                definition={"reducer": "mean", "repeat": 1},
            ),
            execution_granularity=ExecutionGranularity.CASE,
            scheduling_mode=SchedulingMode.FINITE,
            finite_unit_count=limit,
            finite_case_count=limit,
            max_total_host_operations=limit,
            max_total_stream_events=0,
            queue_credits=EvaluationQueueCredits(
                units=min(limit, 8),
                host_operations=max(limit, 1),
                host_operations_per_unit=1,
                stream_events=64,
                sandboxes=0,
                processes=0,
                artifacts=4,
                artifact_bytes=16 * 1024 * 1024,
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
        """Bind the immutable canary and construct provider-owned NEL semantics."""
        if self._plan is None or self._request is None:
            raise RuntimeError("bind_assets requires successful plan_session")
        if proxy is not None:
            raise ValueError(
                "NeMo GSM8K pipe canary does not request a compatibility proxy"
            )
        asset_path, _ = bind_gsm8k_asset(assets)
        limit = self._plan.finite_case_count
        assert limit is not None
        from nemo_evaluator.benchmarks.gsm8k import _PROMPT, _prepare, gsm8k_scorer
        from nemo_evaluator.environments.custom import (
            BenchmarkDefinition,
            ByobEnvironment,
        )

        definition = BenchmarkDefinition(
            name="gsm8k",
            dataset=str(asset_path),
            prompt=_PROMPT,
            target_field="answer",
            prepare_row=_prepare,
            scorer_fn=gsm8k_scorer,
        )
        environment = ByobEnvironment(definition, num_examples=limit)
        case_templates = tuple(
            CaseTemplateDescriptor(
                template_id=f"nemo-gsm8k-{index}", task="gsm8k", source=SOURCE_LABEL
            )
            for index in range(limit)
        )
        unit_templates = tuple(
            ExecutionUnitTemplateDescriptor(
                unit_template_id=f"nemo-gsm8k-unit-{index}",
                case_template_ids=(case.template_id,),
                granularity=ExecutionGranularity.CASE,
                scheduling_class="nemo_case",
            )
            for index, case in enumerate(case_templates)
        )
        units = tuple(
            ExecutionUnitOccurrence(
                unit_id=f"nemo-gsm8k-unit-occurrence-{index}",
                unit_template_id=unit_templates[index].unit_template_id,
                cases=(
                    CaseOccurrenceDescriptor(
                        case_id=f"nemo-gsm8k-case-{index}",
                        template_id=case_templates[index].template_id,
                        issue_ordinal=index,
                        phase_id="evaluation",
                        cycle_index=0,
                    ),
                ),
            )
            for index in range(limit)
        )
        identity = build_identity(
            worker=self._worker_identity,
            config_schema_sha256=self._request.config_schema_sha256,
            provider_config=self._request.provider_config,
            case_templates=case_templates,
            unit_templates=unit_templates,
            components=distribution_identity_components(
                NEMO_EVALUATOR_DISTRIBUTION,
                worker_source_sha256=self._worker_identity.worker_source_sha256,
                dependency_lock_sha256=self._worker_identity.dependency_lock_sha256,
            ),
            policies=self._plan.aggregation_policy.to_wire(),
            host_binding=host_binding,
        )
        solver_config = self._request.provider_config.get("solver_config", {})
        return NemoGsm8kSession(
            session_id=self._request.session_id,
            identity=identity,
            plan=self._plan,
            units=units,
            environment=environment,
            solver_config=solver_config,
            staging_root=staging_root,
        )


class NemoGsm8kSession(BaseEvaluationSession):
    """Pinned NeMo seed -> ChatSolver -> verify lifecycle for GSM8K."""

    def __init__(
        self,
        *,
        session_id: str,
        identity: Any,
        plan: EvaluationPlan,
        units: tuple[ExecutionUnitOccurrence, ...],
        environment: Any,
        solver_config: dict[str, Any],
        staging_root: Path,
    ) -> None:
        super().__init__(session_id, identity, plan, units)
        self._session_id = session_id
        self._environment = environment
        self._solver_config = dict(solver_config)
        self._staging_root = staging_root
        self._solver: Any | None = None
        self._records: dict[str, Any] = {}

    async def run_unit(
        self, unit_id: str, host: PipeEvaluationHost
    ) -> Sequence[CaseOutcome]:
        unit = next(item for item in self.units if item.unit_id == unit_id)
        case = unit.cases[0]
        index = case.issue_ordinal
        try:
            seed = await self._environment.seed(index)
            if self._solver is None:
                from nemo_evaluator.solvers.chat import ChatSolver

                client = NemoPipeModelClient(host, **self._solver_config)
                self._solver = ChatSolver(client)
            context = CallContext(
                session_id=self._session_id,
                unit_id=unit_id,
                case_id=case.case_id,
                semantic_attempt_id=f"attempt-{case.case_id}-0",
                logical_call_id=f"call-{case.case_id}",
            )
            with nemo_model_call_context(context):
                solved = await self._solver.solve(seed)
                verified = await self._environment.verify(
                    solved.response,
                    seed.expected_answer,
                    **seed.metadata,
                )
            if (
                not isinstance(verified.reward, int | float)
                or isinstance(verified.reward, bool)
                or not math.isfinite(float(verified.reward))
                or not 0.0 <= float(verified.reward) <= 1.0
            ):
                raise RuntimeError("NeMo Evaluator GSM8K reward is outside [0, 1]")
            public_reward = float(verified.reward)
            native_score = {
                "reward": verified.reward,
                "extracted_answer": verified.extracted_answer,
                "scoring_details": verified.scoring_details,
                "metadata": verified.metadata,
            }
            self._records[case.case_id] = {
                "prompt": seed.prompt,
                "expected_answer": seed.expected_answer,
                "messages": seed.messages,
                "response": solved.response,
                "score": native_score,
            }
            return (
                CaseOutcome(
                    case_id=case.case_id,
                    kind=CaseOutcomeKind.COMPLETED,
                    scores={
                        "reward": ProviderScore(
                            value=native_score,
                            public_projection={"value": public_reward},
                        )
                    },
                    numeric_metrics={"reward": public_reward},
                    primary_score="reward",
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return (
                infrastructure_outcome(
                    case.case_id, "provider_execution", "nemo_case_error"
                ),
            )

    async def finalize(self) -> Any:
        outcomes = self.outcomes
        completed = [
            item.numeric_metrics["reward"]
            for item in outcomes
            if item.kind is CaseOutcomeKind.COMPLETED
        ]
        aggregates = (
            AggregateMetric(
                scorer="nemo_evaluator.gsm8k_scorer",
                reducer="mean",
                metric="reward",
                value=sum(completed) / len(completed) if completed else 0.0,
                scored_count=len(completed),
                unscored_count=len(outcomes) - len(completed),
                definition={"exclude_infrastructure": True, "exclude_cancelled": True},
            ),
        )
        return finish_candidate(
            identity=self.identity,
            outcomes=outcomes,
            aggregates=aggregates,
            restricted_records=self._records,
            staging_root=self._staging_root,
            filename="nemo_evaluator_bundle.json",
        )

    async def close(self) -> None:
        if self._solver is not None:
            await self._solver.close()
        await self._environment.close()
        await super().close()


class NemoPipeModelClient:
    """NEL solver-facing model client backed only by typed pipe operations."""

    def __init__(
        self,
        host: PipeEvaluationHost,
        *,
        service_id: str = "candidate",
        purpose: str = "primary",
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        media_assets: dict[str, str] | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("NemoPipeModelClient requires max_tokens > 0")
        self._host = host
        self._service_id = service_id
        self._purpose = purpose
        self._generation = {
            key: value
            for key, value in {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": list(stop) if stop is not None else None,
            }.items()
            if value is not None
        }
        self._parameters = {
            key: value
            for key, value in {
                "seed": seed,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }.items()
            if value is not None
        }
        self._media_assets = dict(media_assets or {})

    async def chat(
        self,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Perform one typed ``model.generate`` operation."""
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt or ""})
        payload: dict[str, Any] = {
            "messages": messages,
            "generation": self._generation,
        }
        if self._parameters:
            payload["parameters"] = self._parameters
        result = await self._invoke("model.generate", payload)
        return _nemo_model_response(result, prompt or "", system)

    async def vlm_chat(
        self,
        prompt: str,
        images: list[str],
        system: str | None = None,
        detail: str = "auto",
    ) -> Any:
        """Generate with manifest-bound media references; never dereference URLs."""
        content: list[dict[str, Any]] = []
        for image in images:
            asset_id = self._media_assets.get(image)
            if asset_id is None:
                raise ValueError("VLM image is not a Rust-bound immutable asset")
            content.append({"type": "image", "asset_id": asset_id, "detail": detail})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload = {"messages": messages, "generation": self._generation}
        if self._parameters:
            payload["parameters"] = self._parameters
        result = await self._invoke("model.generate", payload)
        return _nemo_model_response(result, prompt, system)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **overrides: Any,
    ) -> Any:
        """Perform a tool-capable generation with no provider retry/cache."""
        generation, parameters = _merge_generation(
            self._generation, self._parameters, overrides
        )
        payload = {"messages": messages, "generation": generation, "tools": tools}
        if parameters:
            payload["parameters"] = parameters
        result = await self._invoke(
            "model.generate",
            payload,
        )
        from nemo_evaluator.engine.model_client import ToolCallInfo, ToolCallingResponse

        response = _nemo_model_response(result, "", None)
        choice = _first_choice(result)
        message = choice.get("message", {})
        tool_calls = []
        for raw_call in message.get("tool_calls", []):
            function = raw_call.get("function", {})
            arguments = function.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("normalized tool-call arguments must be an object")
            tool_calls.append(
                ToolCallInfo(
                    id=str(raw_call.get("id", "")),
                    name=str(function.get("name", "")),
                    arguments=arguments,
                )
            )
        return ToolCallingResponse(
            content=str(message.get("content") or ""),
            tool_calls=tool_calls,
            finish_reason=str(choice.get("finish_reason") or ""),
            model_response=response,
            reasoning_content=str(message.get("reasoning") or ""),
        )

    async def complete(self, prompt: str, **overrides: Any) -> Any:
        """Perform one public typed completion operation."""
        generation, parameters = _merge_generation(
            self._generation, self._parameters, overrides
        )
        payload = {"prompt": prompt, "generation": generation}
        if parameters:
            payload["parameters"] = parameters
        result = await self._invoke("model.complete", payload)
        return _nemo_completion_response(result, prompt)

    async def embed(self, text: str) -> list[float]:
        """Embed one text through a declared logical service."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a text batch without constructing a Python SDK client."""
        result = await self._invoke("model.embed", {"input": texts})
        if not isinstance(result, dict) or not isinstance(
            result.get("embeddings"), list
        ):
            raise ValueError("model.embed returned malformed normalized result")
        embeddings = result["embeddings"]
        if not all(
            isinstance(row, list)
            and all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in row
            )
            for row in embeddings
        ):
            raise ValueError("model.embed returned non-numeric embedding values")
        return [[float(value) for value in row] for row in embeddings]

    async def close(self) -> None:
        """Close is a no-op because the Rust host owns all transports."""

    async def _invoke(self, operation: str, payload: dict[str, Any]) -> Any:
        context = _CURRENT_CONTEXT.get()
        if context is None:
            raise RuntimeError("NeMo model call has no active case/attempt context")
        ordinal = _CURRENT_CALL_ORDINAL.get()
        _CURRENT_CALL_ORDINAL.set(ordinal + 1)
        logical_call_id = f"{context.logical_call_id}-{ordinal}"
        call_context = CallContext(
            session_id=context.session_id,
            unit_id=context.unit_id,
            case_id=context.case_id,
            semantic_attempt_id=context.semantic_attempt_id,
            logical_call_id=logical_call_id,
        )
        request = HostOperationRequest(
            operation_id=f"op-{logical_call_id}",
            context=call_context,
            service_id=self._service_id,
            purpose=self._purpose,
            semantic_operation_id=operation,
            payload=payload,
            response_mode=ResponseMode.TERMINAL,
            idempotency_key=f"host-{logical_call_id}",
        )
        return terminal_result_payload(await self._host.execute(request))


class nemo_model_call_context:
    """Task-local NEL call context populated around solver/verifier work."""

    def __init__(self, context: CallContext) -> None:
        self._context = context
        self._token: contextvars.Token[CallContext | None] | None = None
        self._ordinal_token: contextvars.Token[int] | None = None

    def __enter__(self) -> None:
        self._token = _CURRENT_CONTEXT.set(self._context)
        self._ordinal_token = _CURRENT_CALL_ORDINAL.set(0)

    def __exit__(self, *_: object) -> None:
        assert self._token is not None
        assert self._ordinal_token is not None
        _CURRENT_CALL_ORDINAL.reset(self._ordinal_token)
        _CURRENT_CONTEXT.reset(self._token)


def _merge_generation(
    defaults: dict[str, Any],
    default_parameters: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    forbidden = {"retry", "retries", "timeout", "cache", "base_url", "api_key", "model"}
    if forbidden.intersection(overrides):
        raise ValueError(
            "NEL generation overrides contain forbidden host authority/policy"
        )
    allowed = {
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported NEL generation override(s): {sorted(unknown)}")
    result = dict(defaults)
    parameters = dict(default_parameters)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"max_tokens", "temperature", "top_p", "stop"}:
            result[key] = value
        else:
            parameters[key] = value
    return result, parameters


def _first_choice(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("model operation result must be an object")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("model operation returned no normalized choices")
    return choices[0]


def _nemo_model_response(result: Any, prompt: str, system: str | None) -> Any:
    from nemo_evaluator.observability.types import ModelResponse

    choice = _first_choice(result)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("model.generate choice omitted normalized message")
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("model.generate usage must be an object")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("NEL chat response requires scalar text content")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return ModelResponse(
        content=content,
        model="logical-service",
        finish_reason=str(
            choice.get("finish_reason") or choice.get("stop_reason") or ""
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(
            prompt_tokens + completion_tokens
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
            else None
        ),
        reasoning_tokens=usage.get("reasoning_tokens"),
        raw_response=result,
        request_prompt=prompt,
        request_system=system,
    )


def _nemo_completion_response(result: Any, prompt: str) -> Any:
    from nemo_evaluator.observability.types import ModelResponse

    choice = _first_choice(result)
    text = choice.get("text")
    if not isinstance(text, str):
        raise ValueError("model.complete choice omitted text")
    usage = result.get("usage") or {}
    return ModelResponse(
        content=text,
        model="logical-service",
        finish_reason=str(choice.get("finish_reason") or ""),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        raw_response=result,
        request_prompt=prompt,
    )
