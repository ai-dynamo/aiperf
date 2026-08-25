# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.create handler for AIPerfSweep CRs.

Validates spec via AIPerfSweepSpec, computes totalVariations/maxTotalRuns,
provisions RBAC for the sweep-controller pod, and creates the JobSet that
schedules it.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import kopf
from pydantic import ValidationError

from aiperf.config.deployment import PodTemplateConfig
from aiperf.config.loader import ConfigurationError
from aiperf.config.sweep import expand_sweep
from aiperf.kubernetes.jobset_helpers import build_security_context
from aiperf.kubernetes.phase import format_timestamp
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.k8s_helpers import (
    ForeignResourceOwnershipError,
    _require_same_controller_owner,
    _resource_name,
)
from aiperf.operator.results_layout import epoch_key_from_body
from aiperf.sweep_controller._naming import (
    MAX_CHILD_JOB_ID_LENGTH,
    MAX_SWEEP_TRIALS,
    MAX_SWEEP_VARIATIONS,
    child_name_suffix_length,
    max_sweep_name_length,
    needs_trial_suffix,
)

logger = logging.getLogger(__name__)

# Child AIPerfJob names are `<sweep>-v<NN>[-t<N>]` (see
# `aiperf.sweep_controller._naming._validate_child_name_indexes`): the
# variation index spans 0..199 (matching `AdaptiveSearchSweep.max_iterations`
# le=200) and the trial index spans 0..9. A sweep whose cardinality exceeds
# these budgets is accepted at create time but crashes the controller mid-run
# when it derives the child name for index 200 / trial 10, after up to 200
# children have already run. Reject it at admission instead.
_MAX_VARIATIONS = MAX_SWEEP_VARIATIONS
_MAX_TRIALS = MAX_SWEEP_TRIALS

# Child AIPerfJob names become the `job_id`, capped at 35 chars by
# `KubernetesDeployment.validate_job_id` (pod names are
# `aiperf-{job_id}-controller-0-0-xxxxx` = 28 + job_id, which must fit the
# 63-char DNS label limit). The child name is `<sweep>-v<NN>[-t<N>]`, so the
# sweep CR name plus its worst-case index suffix must stay within this cap.
_MAX_CHILD_NAME = MAX_CHILD_JOB_ID_LENGTH


def _record_spec_invalid(patch: kopf.Patch, error: Exception) -> None:
    """Stamp a rejected AIPerfSweep spec onto status before raising.

    Mirrors ``handlers/create._validate_spec`` for AIPerfJob so both kinds
    report a bad spec the same way: a ConfigValid=False condition carrying the
    validation error, a Failed phase, and a human-readable ``status.error``.
    Without it the CR is indistinguishable from one the operator never saw.
    """
    now = format_timestamp()
    patch.status["phase"] = "Failed"
    patch.status["error"] = f"Invalid spec: {error}"
    patch.status["completionTime"] = now
    patch.status["conditions"] = [
        {
            "type": "ConfigValid",
            "status": "False",
            "reason": "SpecInvalid",
            "message": str(error)[:32768],
            "lastTransitionTime": now,
        },
        {
            "type": "Failed",
            "status": "True",
            "reason": "SpecInvalid",
            "message": f"Invalid spec: {error}"[:32768],
            "lastTransitionTime": now,
        },
    ]


def _record_permanent_rejection(
    body: dict[str, Any],
    patch: kopf.Patch,
    error: kopf.PermanentError,
) -> None:
    """Expose a non-retryable admission failure through status and an event."""
    now = format_timestamp()
    message = str(error)
    patch.status["phase"] = "Failed"
    patch.status["error"] = message
    patch.status["completionTime"] = now
    patch.status["conditions"] = [
        {
            "type": "ConfigValid",
            "status": "True",
            "reason": "SpecValid",
            "message": "The AIPerfSweep specification is structurally valid.",
            "lastTransitionTime": now,
        },
        {
            "type": "Failed",
            "status": "True",
            "reason": "SweepRejected",
            "message": message[:32768],
            "lastTransitionTime": now,
        },
    ]
    logger.error(f"AIPerfSweep rejected: {message}")
    with contextlib.suppress(LookupError):
        kopf.event(
            body,
            type="Warning",
            reason="SweepRejected",
            message=message,
        )


async def handle(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Validate spec, set status, provision RBAC, create sweep-controller JobSet."""
    # Lazy-imported to avoid pulling kubernetes.crd_models eagerly through the
    # operator package's __init__ when this handler module is collected by kopf.
    from aiperf.common.endpoint_credentials import (
        validate_kubernetes_credential_transport,
        validate_kubernetes_sweep_credential_axes,
    )
    from aiperf.kubernetes.spec_converter import validate_sweep_spec

    try:
        validated = validate_sweep_spec(spec)
        validate_kubernetes_credential_transport(
            validated.benchmark.endpoint, validated.pod_template.env
        )
        validate_kubernetes_sweep_credential_axes(validated.sweep)
    except (ConfigurationError, ValidationError, ValueError) as e:
        # pydantic.ValidationError subclasses ValueError, but a malformed
        # distribution value makes model_validate raise a BARE ValueError that
        # `except ValidationError` alone misses — it would then escape as a
        # generic exception and kopf would retry a permanently-invalid spec
        # forever. Catch both, matching the AIPerfJob handler's `except ValueError`.
        #
        # Record the rejection on the CR before raising. kopf's PermanentError
        # stops the retry loop but writes nothing to status, so an invalid
        # AIPerfSweep used to sit with a completely empty status object --
        # blank phase in `kubectl get`, no conditions, no error -- while the
        # reason existed only in operator logs. AIPerfJob has always surfaced
        # this (ConfigValid=False + Failed phase + error); mirror it here.
        _record_spec_invalid(patch, e)
        raise kopf.PermanentError(f"AIPerfSweep spec invalid: {e}") from e

    # Mirror `sweep_controller.plan_builder.build_plan_from_sweep`:
    # `expand_sweep` expects the envelope shape — body under `benchmark`,
    # cross-variation fields (sweep, variables, random_seed) at envelope
    # level. A flattened dict hides datasets/phases from expand_sweep's
    # scenario-merge logic, spuriously rejecting specs the controller accepts.
    sweep_input: dict[str, Any] = {
        "benchmark": validated.benchmark.model_dump(
            by_alias=True, exclude_none=True, exclude_unset=True
        )
    }
    if validated.sweep is not None:
        sweep_input["sweep"] = validated.sweep.model_dump(by_alias=True)
    if validated.variables:
        sweep_input["variables"] = validated.variables
    if validated.random_seed is not None:
        sweep_input["random_seed"] = validated.random_seed

    has_convergence = bool(
        validated.multi_run is not None and validated.multi_run.convergence is not None
    )
    try:
        n_variations, max_total_runs = _compute_cardinality(validated, sweep_input)
        _reject_overlong_child_names(
            name, n_variations, max_total_runs, has_convergence=has_convergence
        )
    except kopf.PermanentError as e:
        _record_permanent_rejection(body, patch, e)
        raise

    sweep_uid = body["metadata"]["uid"]
    epoch = epoch_key_from_body(body)

    patch.status["phase"] = "Pending"
    patch.status["totalVariations"] = n_variations
    patch.status["maxTotalRuns"] = max_total_runs
    patch.status["completedRuns"] = 0
    patch.status["failedRuns"] = 0
    patch.status["runEpoch"] = int(epoch) if epoch.isdigit() else 0
    patch.status["startedAt"] = format_timestamp()
    base_url = OperatorEnvironment.SERVICE.BASE_URL.rstrip("/")
    patch.status["apiUrl"] = f"{base_url}/api/v1/sweeps/{namespace}/{name}"

    await _provision_rbac(name=name, namespace=namespace, sweep_uid=sweep_uid)
    try:
        await _create_sweep_controller_jobset(
            name=name,
            namespace=namespace,
            sweep_uid=sweep_uid,
            epoch=epoch,
            template_spec=validated.model_dump(
                exclude={
                    "sweep",
                    "multi_run",
                    "failure_policy",
                    "cancel",
                    "ttl_seconds_after_finished",
                    "variables",
                    "random_seed",
                },
                by_alias=True,
                exclude_none=True,
                exclude_unset=True,
            ),
        )
    except kopf.PermanentError as e:
        _record_permanent_rejection(body, patch, e)
        raise

    jobset_name = f"aiperf-{name}"
    patch.status["runtimeRef"] = {
        "jobSetName": jobset_name,
        "sweepControllerHost": (
            f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"
        ),
    }
    generation = body.get("metadata", {}).get("generation")
    if generation is not None:
        patch.status["observedGeneration"] = int(generation)
    logger.info(
        f"AIPerfSweep {namespace}/{name} created: {n_variations} variations, "
        f"max {max_total_runs} total runs"
    )


def _compute_cardinality(
    validated: Any,
    sweep_input: dict[str, Any],
) -> tuple[int, int]:
    """Compute `(totalVariations, maxTotalRuns)` for the create-handler status.

    Adaptive search (Bayesian Optimization) sweeps don't know the final
    variation count up front -- only an upper bound (`max_iterations`).
    Write that bound so dashboards can render a determinate progress bar;
    early convergence routes through the controller pod's terminal-phase
    write, which supersedes any premature rollup-driven Aggregating phase
    via the existing `_conditional_phase_set` test-op guard in
    `child_rollup.py`.

    For non-adaptive sweeps, expand the grid/scenarios input via
    `expand_sweep` and multiply by `multi_run.num_runs` (defaults to 1).
    """
    from aiperf.config.sweep import AdaptiveSearchSweep

    multi_run = validated.multi_run
    max_trials = multi_run.num_runs if multi_run is not None else 1
    if max_trials > _MAX_TRIALS:
        raise kopf.PermanentError(
            f"AIPerfSweep multi_run.num_runs ({max_trials}) exceeds the "
            f"{_MAX_TRIALS}-trial child-name budget (trial index 0..{_MAX_TRIALS - 1}); "
            "reduce num_runs."
        )

    sweep = validated.sweep
    # Reject an over-cap sweep from its cheap O(#dimensions) shape BEFORE
    # expand_sweep materializes the full cartesian product. A grid whose
    # variables multiply to ~1M variations otherwise blocks the kopf event
    # loop ~35s and allocates ~4GB before this cap rejects it — long enough
    # for the liveness probe to kill the pod mid-handler, so kopf re-runs and
    # crashloops. The cheap count equals the expansion length exactly for
    # every concrete sweep type (see `_cheap_variation_count`).
    cheap_count = _cheap_variation_count(sweep)
    if cheap_count is not None and cheap_count > _MAX_VARIATIONS:
        raise kopf.PermanentError(_over_cap_message(cheap_count))

    if isinstance(sweep, AdaptiveSearchSweep):
        n_variations = sweep.max_iterations
    else:
        try:
            n_variations = len(expand_sweep(sweep_input))
        except ValueError as e:
            # expand_sweep validation failures (bad dotted path, mismatched
            # zip lengths, unmergeable scenario overrides) are spec bugs, not
            # transient conditions — retrying the same spec can never succeed.
            raise kopf.PermanentError(
                f"AIPerfSweep sweep expansion rejected the spec: {e}"
            ) from e

    if n_variations > _MAX_VARIATIONS:
        raise kopf.PermanentError(_over_cap_message(n_variations))

    return n_variations, n_variations * max_trials


def _over_cap_message(n_variations: int) -> str:
    """Build the PermanentError message for an over-cap sweep cardinality."""
    return (
        f"AIPerfSweep expands to {n_variations} variations, exceeding the "
        f"{_MAX_VARIATIONS}-variation child-name budget (variation index "
        f"0..{_MAX_VARIATIONS - 1}); reduce the sweep cardinality."
    )


def _cheap_variation_count(sweep: Any) -> int | None:
    """Count the variations a sweep expands to WITHOUT materializing them.

    O(#dimensions) shape inspection of the validated sweep model, used to
    reject an over-cap sweep before ``expand_sweep`` allocates the full
    cartesian product. Cardinality per sweep type:

    * ``grid``            -> product of every parameter list's length
    * ``zip``             -> the shared parameter-list length (validated equal)
    * ``scenarios``       -> number of hand-picked runs
    * ``sobol`` / ``latin_hypercube`` -> ``samples``
    * ``adaptive_search`` -> ``max_iterations`` (upper bound)

    Returns ``None`` when the count can't be derived from the sweep shape
    alone; the caller then falls back to a full ``expand_sweep``.

    Example:
        >>> from aiperf.config.sweep import GridSweep
        >>> _cheap_variation_count(
        ...     GridSweep(parameters={"phases.profiling.concurrency": [1, 2, 4]})
        ... )
        3
    """
    from aiperf.config.sweep import (
        AdaptiveSearchSweep,
        GridSweep,
        LatinHypercubeSweep,
        ScenarioSweep,
        SobolSweep,
        ZipSweep,
    )

    if isinstance(sweep, GridSweep):
        return math.prod(len(values) for values in sweep.parameters.values())
    if isinstance(sweep, ZipSweep):
        # `_check_equal_lengths` guarantees every list shares one length, so
        # any single list's length is the lockstep cardinality.
        return len(next(iter(sweep.parameters.values())))
    if isinstance(sweep, ScenarioSweep):
        return len(sweep.runs)
    if isinstance(sweep, (SobolSweep, LatinHypercubeSweep)):
        return sweep.samples
    if isinstance(sweep, AdaptiveSearchSweep):
        return sweep.max_iterations
    return None


def _reject_overlong_child_names(
    name: str,
    n_variations: int,
    max_total_runs: int,
    has_convergence: bool = False,
) -> None:
    """Reject sweep names whose derived child `job_id` would overflow 35 chars.

    Child AIPerfJob names are `<sweep>-v<NN>[-t<N>]` (see
    `aiperf.sweep_controller._naming.build_child_name`). The child name becomes
    the `job_id`, capped at `_MAX_CHILD_NAME` by `KubernetesDeployment.
    validate_job_id` so that `aiperf-{job_id}-controller-0-0-xxxxx` pod and
    headless-Service names fit the 63-char DNS label limit. A long-but-valid
    sweep CR name (an RFC1123 subdomain may be up to 253 chars) is accepted at
    admission yet crashes child creation mid-sweep once the suffix is appended.
    Reject it up front instead.

    The worst-case suffix is computed from the actual cardinality: the variation
    index is rendered with at least two digits (`-v00`), three once it reaches
    100 (`-v100`), and a `-t<N>` trial suffix is appended only when more than one
    trial runs.
    """
    max_var_idx = max(n_variations - 1, 0)
    trials = max_total_runs // n_variations if n_variations else 1
    # Must match the runtime rule exactly: _naming.needs_trial_suffix returns
    # True whenever convergence is configured, regardless of num_runs, and
    # MultiRunConfig permits num_runs: 1 with convergence. Reserving on
    # `trials > 1` alone let a 31-char sweep pass admission and then be
    # rejected mid-sweep by validate_job_id's 35-char cap -- exactly what this
    # check exists to prevent.
    suffix_len = child_name_suffix_length(
        n_variations,
        max_total_runs,
        has_convergence=has_convergence,
    )
    max_name_len = max_sweep_name_length(
        n_variations,
        max_total_runs,
        has_convergence=has_convergence,
    )
    with_trial_suffix = needs_trial_suffix(trials, has_convergence)
    if len(name) > max_name_len:
        raise kopf.PermanentError(
            f"AIPerfSweep name '{name}' ({len(name)} chars) leaves no room for the "
            f"child job_id suffix (worst case '-v{max_var_idx}"
            f"{'-t0' if with_trial_suffix else ''}', {suffix_len} chars); the derived child "
            f"name would exceed the {_MAX_CHILD_NAME}-char job_id cap. Max sweep name "
            f"is {max_name_len} chars for this cardinality."
        )


async def _provision_rbac(*, name: str, namespace: str, sweep_uid: str) -> None:
    """Create namespace-scoped ServiceAccount + Role + RoleBinding for sweep-controller."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    sa_name = f"aiperf-sweep-controller-{name}"
    role_name = sa_name
    owner_ref = k8s.V1OwnerReference(
        api_version="aiperf.nvidia.com/v1alpha1",
        kind="AIPerfSweep",
        name=name,
        uid=sweep_uid,
        controller=True,
        block_owner_deletion=True,
    )

    async with k8s_client() as api:
        core = k8s.CoreV1Api(api)
        rbac = k8s.RbacAuthorizationV1Api(api)

        sa = k8s.V1ServiceAccount(
            metadata=k8s.V1ObjectMeta(
                name=sa_name,
                namespace=namespace,
                owner_references=[owner_ref],
            )
        )
        await _create_or_skip_409(
            core.create_namespaced_service_account,
            namespace,
            sa,
            read_fn=core.read_namespaced_service_account,
        )

        role = k8s.V1Role(
            metadata=k8s.V1ObjectMeta(
                name=role_name,
                namespace=namespace,
                owner_references=[owner_ref],
            ),
            rules=[
                k8s.V1PolicyRule(
                    api_groups=["aiperf.nvidia.com"],
                    resources=["aiperfjobs", "aiperfjobs/status"],
                    verbs=[
                        "create",
                        "get",
                        "list",
                        "watch",
                        "patch",
                        "update",
                        "delete",
                    ],
                ),
                k8s.V1PolicyRule(
                    api_groups=["aiperf.nvidia.com"],
                    resources=["aiperfsweeps", "aiperfsweeps/status"],
                    verbs=["get", "patch", "update"],
                    resource_names=[name],
                ),
                # Emit kubectl-visible events on the parent CR (progress,
                # cancellation acks, aggregation phase).
                k8s.V1PolicyRule(
                    api_groups=[""],
                    resources=["events"],
                    verbs=["create", "patch"],
                ),
            ],
        )
        await _create_or_skip_409(
            rbac.create_namespaced_role,
            namespace,
            role,
            read_fn=rbac.read_namespaced_role,
        )

        binding = k8s.V1RoleBinding(
            metadata=k8s.V1ObjectMeta(
                name=role_name,
                namespace=namespace,
                owner_references=[owner_ref],
            ),
            subjects=[
                k8s.RbacV1Subject(
                    kind="ServiceAccount",
                    name=sa_name,
                    namespace=namespace,
                )
            ],
            role_ref=k8s.V1RoleRef(
                api_group="rbac.authorization.k8s.io",
                kind="Role",
                name=role_name,
            ),
        )
        await _create_or_skip_409(
            rbac.create_namespaced_role_binding,
            namespace,
            binding,
            read_fn=rbac.read_namespaced_role_binding,
        )


async def _create_or_skip_409(
    create_fn: Any,
    namespace: str,
    body: Any,
    *,
    read_fn: Any | None = None,
) -> None:
    """Create or adopt a resource owned by this sweep CR incarnation.

    Transient apiserver failures (ApiException with non-409 status, connection
    errors, timeouts) raise kopf.TemporaryError so kopf retries with backoff
    rather than hammering the apiserver in an unbounded retry loop.
    """
    import aiohttp
    from kubernetes_asyncio.client import ApiException

    try:
        await create_fn(namespace, body)
    except ApiException as e:
        if e.status == 409:
            if read_fn is not None:
                try:
                    existing = await read_fn(
                        name=_resource_name(body), namespace=namespace
                    )
                    _require_same_controller_owner(existing, body)
                except ApiException as identity_error:
                    raise kopf.TemporaryError(
                        f"waiting for stale resource replacement: {identity_error.reason}",
                        delay=15,
                    ) from identity_error
                except ForeignResourceOwnershipError as identity_error:
                    raise kopf.PermanentError(str(identity_error)) from identity_error
            return
        raise kopf.TemporaryError(
            f"apiserver rejected create ({e.status}): {e.reason}", delay=30
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during create: {e}", delay=30
        ) from e


async def _create_sweep_controller_jobset(
    *,
    name: str,
    namespace: str,
    sweep_uid: str,
    epoch: str,
    template_spec: dict[str, Any],
) -> None:
    """Create a JobSet whose single replica runs `python -m aiperf.sweep_controller.main`.

    The pod runs two containers:

    * ``sweep-controller`` — the orchestrator that drives child AIPerfJobs and
      writes the cross-variation aggregate to ``/results``.
    * ``results-sidecar`` — same image, separate command. Mirrors the AIPerfJob
      controller pod's harvest pattern (``jobset_builder._create_results_sidecar``):
      reads from the shared ``/results`` emptyDir, exposes the file tree over
      HTTP at ``:RESULTS_SIDECAR_PORT/api/results/{list,files/<path>}``, gated
      by the ``.aiperf_results_ready.json`` marker the orchestrator writes after
      aggregate export. The operator harvests the aggregate from this endpoint
      on sweep terminal phase — no PVC mount on the controller pod.
    """
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.constants import Containers
    from aiperf.kubernetes.environment import K8sEnvironment

    image = template_spec.get("image")
    if not image:
        raise kopf.PermanentError("image is required")

    jobset_name = f"aiperf-{name}"
    sa_name = f"aiperf-sweep-controller-{name}"

    container = {
        "name": "sweep-controller",
        "image": image,
        "imagePullPolicy": template_spec.get("imagePullPolicy", "IfNotPresent"),
        "command": ["python", "-m", "aiperf.sweep_controller.main"],
        "env": [
            {"name": "AIPERF_SWEEP_NAME", "value": name},
            {"name": "AIPERF_SWEEP_NAMESPACE", "value": namespace},
            {"name": "AIPERF_SWEEP_UID", "value": sweep_uid},
            {"name": "AIPERF_SWEEP_EPOCH", "value": epoch},
            {"name": "AIPERF_OPERATOR_MANAGED", "value": "1"},
            {"name": "AIPERF_RESULTS_DIR", "value": "/results"},
            # Lets the sweep controller fetch per-child profile_export_aiperf.json
            # from the operator's PVC-backed API when status.summary is empty
            # (CompletedBeforeMonitor race). BASE_URL points at the operator's
            # only FastAPI surface (results-server container, port 8081).
            {
                "name": "AIPERF_OPERATOR_BASE_URL",
                "value": OperatorEnvironment.SERVICE.BASE_URL,
            },
        ],
        "volumeMounts": [
            {"name": "results", "mountPath": "/results"},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
    }
    # Merge user-supplied container env (spec.podTemplate.env) so
    # users can pass HTTP_PROXY, HF_HOME, custom log levels, etc. The
    # controller's reserved AIPERF_SWEEP_* vars take precedence on collision.
    pod_template = template_spec.get("podTemplate") or {}
    controller_template = PodTemplateConfig(
        container_security_context=pod_template.get("containerSecurityContext") or {}
    )
    user_env = pod_template.get("env") or []
    if user_env:
        reserved = {e["name"] for e in container["env"]}
        container["env"].extend(e for e in user_env if e.get("name") not in reserved)
    # Container-level resources/securityContext from podTemplate. Without
    # these, the sweep-controller pod gets no requests/limits (rejected by
    # ResourceQuota on hardened clusters) and no securityContext (rejected
    # by Pod Security Admission baseline/restricted).
    resource_mode = template_spec.get("resourceMode") or "default"
    if pod_template.get("resources") is not None:
        container["resources"] = pod_template["resources"]
    elif resource_mode != "none":
        # Fall back to the sweep-controller defaults. With neither a user value
        # nor a default, a LimitRange or ResourceQuota that requires requests
        # rejects every pod: the JobSet is created fine and the AIPerfSweep sits
        # Pending with no children and no surfaced error.
        #
        # Deliberately NOT SYSTEM_CONTROLLER: under `sweep.type: adaptive_search`
        # this pod imports torch/BoTorch to fit the GP, which SystemController's
        # 192Mi request does not cover (287 MiB after import, 350 MiB after the
        # first GP fit). Undersized, the pod survives the Sobol prefix and then
        # dies the moment the GP first fits -- the Job hits its backoff limit and
        # the JobSet fails, while the AIPerfSweep CR is left stuck reporting
        # `Running` because the controller that would update it is gone.
        container["resources"] = K8sEnvironment.SWEEP_CONTROLLER.to_k8s_resources(
            burstable=resource_mode == "burstable"
        )
    container["securityContext"] = build_security_context(controller_template)

    # Results sidecar: serves /results over HTTP for the operator to harvest.
    # No PVC on this pod by design (sweep-controller is ephemeral); the
    # operator pulls artifacts via the sidecar and persists them on its own
    # results PVC at <ns>/sweeps/<sweep>/<epoch>/.
    sidecar_port = K8sEnvironment.PORTS.RESULTS_SIDECAR
    sidecar = {
        "name": Containers.RESULTS_SIDECAR,
        "image": image,
        "imagePullPolicy": template_spec.get("imagePullPolicy", "IfNotPresent"),
        "command": ["python", "-m", "aiperf.kubernetes.results_sidecar"],
        "env": [
            {"name": "AIPERF_RESULTS_DIR", "value": "/results"},
            {"name": "AIPERF_RESULTS_SIDECAR_PORT", "value": str(sidecar_port)},
        ],
        "volumeMounts": [
            {"name": "results", "mountPath": "/results", "readOnly": True},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
        "ports": [{"containerPort": sidecar_port, "name": "results"}],
    }
    if resource_mode != "none":
        # Mirrors jobset_builder._create_results_sidecar, which has always
        # resolved these; this copy carried no resources at all.
        sidecar["resources"] = K8sEnvironment.RESULTS_SIDECAR.to_k8s_resources(
            burstable=resource_mode == "burstable"
        )
    sidecar["securityContext"] = build_security_context(controller_template)

    pod_spec: dict[str, Any] = {
        "restartPolicy": "OnFailure",
        "serviceAccountName": sa_name,
        "containers": [container, sidecar],
        "volumes": [
            {"name": "results", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
        ],
    }
    # Lift scheduling primitives + pod-level securityContext from the user's
    # spec.podTemplate so the sweep-controller pod can land on the
    # same nodes as its child workers will and meets cluster security
    # baselines.
    for key in (
        "nodeSelector",
        "tolerations",
        "affinity",
        "imagePullSecrets",
        "priorityClassName",
        "runtimeClassName",
        "podSecurityContext",
    ):
        if key in pod_template and pod_template[key] is not None:
            pod_spec["securityContext" if key == "podSecurityContext" else key] = (
                pod_template[key]
            )

    jobset_body = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": jobset_name,
            "namespace": namespace,
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": name,
                    "uid": sweep_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "spec": {
            # enableDNSHostnames provisions the headless service so the
            # operator can harvest the emptyDir-only aggregate from the
            # controller pod's stable DNS name (controller_dns_name(...)).
            "network": {
                "enableDNSHostnames": True,
            },
            "replicatedJobs": [
                {
                    "name": "controller",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "completions": 1,
                            "parallelism": 1,
                            "template": {"spec": pod_spec},
                        },
                    },
                },
            ],
        },
    }

    async with k8s_client() as api:
        custom = k8s.CustomObjectsApi(api)
        await _create_or_skip_409_custom(
            custom,
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=namespace,
            plural="jobsets",
            body=jobset_body,
        )


async def _create_or_skip_409_custom(
    custom: Any,
    *,
    group: str,
    version: str,
    namespace: str,
    plural: str,
    body: Any,
) -> None:
    import aiohttp
    from kubernetes_asyncio.client import ApiException

    try:
        await custom.create_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            body=body,
        )
    except ApiException as e:
        if e.status == 409:
            try:
                existing = await custom.get_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=_resource_name(body),
                )
                _require_same_controller_owner(existing, body)
            except ApiException as identity_error:
                raise kopf.TemporaryError(
                    f"waiting for stale JobSet replacement: {identity_error.reason}",
                    delay=15,
                ) from identity_error
            except ForeignResourceOwnershipError as identity_error:
                raise kopf.PermanentError(str(identity_error)) from identity_error
            return
        raise kopf.TemporaryError(
            f"apiserver rejected JobSet create ({e.status}): {e.reason}", delay=30
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during JobSet create: {e}", delay=30
        ) from e
