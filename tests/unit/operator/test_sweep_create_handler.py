# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest

from aiperf.operator.handlers.sweep import create as sweep_create


def _valid_body() -> dict:
    return {
        "metadata": {
            "name": "s",
            "namespace": "ns",
            "uid": "u",
            "creationTimestamp": "2024-04-25T18:22:03Z",
        },
        "spec": {
            "multiRun": {"numRuns": 3},
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [1]},
            },
            "image": "x:latest",
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "duration": 1,
                        "concurrency": 1,
                    }
                ],
            },
        },
    }


@pytest.mark.asyncio
async def test_handle_validates_spec_and_creates_jobset(monkeypatch):
    body = _valid_body()
    patch = kopf.Patch()
    provision_rbac = AsyncMock()
    create_jobset = AsyncMock()
    monkeypatch.setattr(sweep_create, "_provision_rbac", provision_rbac)
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", create_jobset)

    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )

    provision_rbac.assert_awaited_once()
    create_jobset.assert_awaited_once()
    assert patch.status["phase"] == "Pending"
    assert patch.status["totalVariations"] == 1
    assert patch.status["maxTotalRuns"] == 3
    assert "runtimeRef" in patch.status


@pytest.mark.asyncio
async def test_handle_rejects_invalid_spec(monkeypatch):
    body = {
        "metadata": {"name": "s", "namespace": "ns", "uid": "u"},
        "spec": {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "duration": 1,
                        "concurrency": 1,
                    }
                ],
            },
        },  # no axes (no sweep / multiRun / convergence)
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    with pytest.raises(kopf.PermanentError, match="sweep is required"):
        await sweep_create.handle(
            body=body,
            spec=body["spec"],
            name="s",
            namespace="ns",
            patch=patch,
        )


@pytest.mark.asyncio
async def test_handle_scenarios_sweep_with_singular_dataset_override(monkeypatch):
    """Regression: the handler must feed ``expand_sweep`` the envelope shape.

    ``expand_sweep`` expects ``{"benchmark": <body>, "sweep": ...}`` (what
    ``sweep_controller.plan_builder`` builds). The old flattened input hid
    the base ``datasets:`` list from the scenario-merge logic, so a run
    overriding singular ``benchmark.dataset:`` (valid — the controller
    resolves the name from the single base dataset) was spuriously rejected
    at admission with "base with multiple datasets ([])".
    """
    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "scenarios",
        "runs": [
            {"name": "small", "benchmark": {"dataset": {"type": "synthetic"}}},
            {"name": "large", "benchmark": {"dataset": {"type": "synthetic"}}},
        ],
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    await sweep_create.handle(
        body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
    )

    assert patch.status["totalVariations"] == 2
    assert patch.status["maxTotalRuns"] == 6  # 2 scenarios x numRuns=3


@pytest.mark.asyncio
async def test_handle_expand_sweep_valueerror_becomes_permanent_error(monkeypatch):
    """A ValueError from expand_sweep is a spec bug — kopf must stop retrying.

    Before the fix it escaped as a generic exception, so kopf retried a
    permanently-invalid spec forever.
    """
    body = _valid_body()
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    def _reject(_data):
        raise ValueError("zip sweep parameters must all have equal length")

    monkeypatch.setattr(sweep_create, "expand_sweep", _reject)

    with pytest.raises(
        kopf.PermanentError,
        match=r"sweep expansion rejected the spec: zip sweep parameters",
    ):
        await sweep_create.handle(
            body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
        )


@pytest.mark.asyncio
async def test_handle_bare_valueerror_from_model_validate_becomes_permanent_error(
    monkeypatch,
):
    """A BARE ValueError from spec validation must stop kopf retrying.

    ``pydantic.ValidationError`` subclasses ``ValueError``, but a malformed
    distribution value makes ``AIPerfSweepSpec.model_validate`` raise a plain
    ``builtins.ValueError`` that is *not* a ``ValidationError``. The old
    ``except ValidationError`` missed it, so it escaped as a generic exception
    and kopf retried a permanently-invalid spec forever. The broadened
    ``except (ValidationError, ValueError)`` turns it into a PermanentError.
    """
    body = _valid_body()
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    class _RaisesBareValueError:
        @staticmethod
        def model_validate(_spec):
            raise ValueError("could not coerce distribution value '512' to a float")

    # `handle` lazily does `from aiperf.operator.models import AIPerfSweepSpec`,
    # so patch the attribute on that module (resolved at call time).
    monkeypatch.setattr("aiperf.operator.models.AIPerfSweepSpec", _RaisesBareValueError)

    with pytest.raises(
        kopf.PermanentError,
        match=r"AIPerfSweep spec invalid: could not coerce distribution value",
    ):
        await sweep_create.handle(
            body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
        )


@pytest.mark.asyncio
async def test_handle_rejects_oversized_grid_before_expanding(monkeypatch):
    """A grid whose cardinality exceeds the variation cap is rejected from its
    cheap shape BEFORE ``expand_sweep`` materializes the cartesian product.

    At huge cardinality (1M variations) the full expand blocks the kopf event
    loop ~35s and allocates ~4GB; the liveness probe kills the pod mid-handler
    and kopf crashloops. The cheap ``math.prod`` pre-check must reject first,
    so ``expand_sweep`` is never called.
    """
    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "grid",
        # 201 values -> 201 variations, one over the 200 cap.
        "variables": {"benchmark.phases.profiling.concurrency": list(range(1, 202))},
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    expand_spy = MagicMock(
        side_effect=AssertionError("expand_sweep must not run for an over-cap grid")
    )
    monkeypatch.setattr(sweep_create, "expand_sweep", expand_spy)

    with pytest.raises(
        kopf.PermanentError,
        match=r"expands to 201 variations, exceeding the 200-variation",
    ):
        await sweep_create.handle(
            body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
        )
    expand_spy.assert_not_called()


@pytest.mark.asyncio
async def test_handle_under_cap_grid_still_expands(monkeypatch):
    """An under-cap grid still routes through ``expand_sweep`` for the exact
    count — the cheap pre-check must not short-circuit valid sweeps."""
    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "grid",
        "variables": {"benchmark.phases.profiling.concurrency": [1, 2, 3]},
    }
    body["spec"]["multiRun"]["numRuns"] = 1
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    spy = MagicMock(side_effect=sweep_create.expand_sweep)
    monkeypatch.setattr(sweep_create, "expand_sweep", spy)

    await sweep_create.handle(
        body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
    )

    spy.assert_called_once()
    assert patch.status["totalVariations"] == 3
    assert patch.status["maxTotalRuns"] == 3


@pytest.mark.asyncio
async def test_handle_rejects_multidim_grid_by_product_not_sum(monkeypatch):
    """The cheap cardinality guard counts a MULTI-dimensional grid by the
    PRODUCT of per-dimension lengths, not the sum.

    4 variables x 50 values -> product 6_250_000 (>> the 200 cap), while the
    per-dimension length SUM is only 200 (== the cap, which would NOT trip a
    strict ``> cap`` check). A ``math.prod`` -> ``sum`` regression in
    ``_cheap_variation_count`` would let this 6.25M-variation grid through and
    OOM / liveness-kill the operator pod at ``expand_sweep`` time. Pins the
    product semantic both at the helper and end-to-end through ``handle``.

    The single-dimension over-cap test above (201 values -> 201) cannot catch a
    ``prod`` -> ``sum`` swap: for one dimension the product and the sum are
    equal, so only a multi-dimensional grid distinguishes them.
    """
    from aiperf.config.sweep import GridSweep

    grid = GridSweep(
        variables={
            "phases.profiling.concurrency": list(range(50)),
            "phases.profiling.rate": list(range(50)),
            "phases.profiling.requests": list(range(50)),
            "phases.profiling.duration": list(range(50)),
        }
    )
    # Product 50**4, NOT the per-dimension length sum (50 * 4 == 200).
    assert sweep_create._cheap_variation_count(grid) == 6_250_000
    assert sweep_create._cheap_variation_count(grid) != 200

    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "grid",
        "variables": {
            "benchmark.phases.profiling.concurrency": list(range(50)),
            "benchmark.phases.profiling.rate": list(range(50)),
            "benchmark.phases.profiling.requests": list(range(50)),
            "benchmark.phases.profiling.duration": list(range(50)),
        },
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    expand_spy = MagicMock(
        side_effect=AssertionError("expand_sweep must not run for an over-cap grid")
    )
    monkeypatch.setattr(sweep_create, "expand_sweep", expand_spy)

    with pytest.raises(
        kopf.PermanentError,
        match=r"expands to 6250000 variations, exceeding the 200-variation",
    ):
        await sweep_create.handle(
            body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
        )
    expand_spy.assert_not_called()


@pytest.mark.asyncio
async def test_handle_sweep_input_mirrors_plan_builder_envelope(monkeypatch):
    """The expand_sweep input must be envelope-shaped with variables/random_seed
    parity so admission-time cardinality matches the sweep-controller's plan."""
    body = _valid_body()
    body["spec"]["variables"] = {"region": "us-east"}
    body["spec"]["randomSeed"] = 42
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    captured: dict = {}
    real_expand = sweep_create.expand_sweep

    def _capture(data):
        captured["input"] = data
        return real_expand(data)

    monkeypatch.setattr(sweep_create, "expand_sweep", _capture)

    await sweep_create.handle(
        body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
    )

    sweep_input = captured["input"]
    assert set(sweep_input) == {"benchmark", "sweep", "variables", "random_seed"}
    assert "phases" in sweep_input["benchmark"]
    assert sweep_input["variables"] == {"region": "us-east"}
    assert sweep_input["random_seed"] == 42


@pytest.mark.asyncio
async def test_epoch_from_creation_timestamp():
    """`metadata.creationTimestamp` parses to a decimal epoch in status.runEpoch."""
    from datetime import datetime

    expected = int(datetime(2024, 4, 25, 18, 22, 3, tzinfo=UTC).timestamp())
    assert sweep_create._epoch_from_creation_ts("2024-04-25T18:22:03Z") == str(expected)


@pytest.mark.asyncio
async def test_epoch_from_creation_timestamp_subsecond_precision():
    """RFC3339 timestamps with sub-second precision must parse, not return '0'.

    K8s metadata.creationTimestamp is whole-second by convention, but other
    RFC3339 sources (kopf event payloads, JSON-patched fields from non-
    apiserver writers) may include fractional seconds. `strptime` with the
    bare ``%Y-%m-%dT%H:%M:%SZ`` format string rejects them and falls
    through to ``"0"`` — collapsing every child name onto epoch 0 across
    reruns.
    """
    from datetime import datetime

    # Whole-second baseline.
    whole = int(datetime(2024, 4, 25, 18, 22, 3, tzinfo=UTC).timestamp())
    # Sub-second precision must still parse to the same whole-second epoch.
    assert sweep_create._epoch_from_creation_ts("2024-04-25T18:22:03.123456Z") == str(
        whole
    )
    assert sweep_create._epoch_from_creation_ts("2024-04-25T18:22:03.5Z") == str(whole)
    # Garbage still falls through to "0".
    assert sweep_create._epoch_from_creation_ts("not-a-timestamp") == "0"
    assert sweep_create._epoch_from_creation_ts("") == "0"


@pytest.mark.asyncio
async def test_handle_apiurl_uses_default_base_url(monkeypatch):
    """status.apiUrl uses OperatorEnvironment.SERVICE.BASE_URL — default value."""
    body = _valid_body()
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    await sweep_create.handle(
        body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
    )

    assert (
        patch.status["apiUrl"]
        == "http://aiperf-operator.aiperf-system:8081/api/v1/sweeps/ns/s"
    )


@pytest.mark.asyncio
async def test_handle_apiurl_honors_base_url_override(monkeypatch):
    """AIPERF_OPERATOR_BASE_URL override flows into stamped status.apiUrl.

    Verifies that the deferred-TODO removal — replacing the hardcoded
    ``http://aiperf-operator.aiperf-system:8081`` with
    ``OperatorEnvironment.SERVICE.BASE_URL`` — actually plumbs the env var
    through to the patch. Patches the singleton attribute (not the env var)
    because ``OperatorEnvironment`` is materialized at module import time.
    """
    from aiperf.operator.environment import OperatorEnvironment

    monkeypatch.setattr(
        OperatorEnvironment.SERVICE, "BASE_URL", "https://custom.example:9443/"
    )

    body = _valid_body()
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())

    await sweep_create.handle(
        body=body, spec=body["spec"], name="s", namespace="ns", patch=patch
    )

    # Trailing slash on the override must be stripped, leaving exactly one
    # ``/api/v1/...`` separator.
    assert patch.status["apiUrl"] == "https://custom.example:9443/api/v1/sweeps/ns/s"


def test_service_settings_reads_aiperf_operator_base_url(monkeypatch):
    """``AIPERF_OPERATOR_BASE_URL`` env var binds to ``SERVICE.BASE_URL``.

    Instantiates a fresh ``_OperatorServiceSettings`` so the test is independent of
    the module-level singleton's lifecycle.
    """
    from aiperf.operator.environment import _OperatorServiceSettings

    monkeypatch.setenv("AIPERF_OPERATOR_BASE_URL", "https://override.example:7000")
    settings = _OperatorServiceSettings()
    assert settings.BASE_URL == "https://override.example:7000"


@pytest.mark.asyncio
async def test_handle_computes_max_total_runs_grid_x_trials(monkeypatch):
    body = _valid_body()
    # The path must name a real phase from _valid_body ("profiling"). The
    # pre-fix flattened expand_sweep input hid the benchmark body, so a
    # nonexistent phase name was silently tolerated here; the envelope shape
    # resolves paths against the real body exactly like the sweep-controller.
    body["spec"]["sweep"] = {
        "type": "grid",
        "variables": {"benchmark.phases.profiling.concurrency": [1, 2, 3, 4]},
    }
    body["spec"]["multiRun"]["numRuns"] = 5
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert patch.status["totalVariations"] == 4
    assert patch.status["maxTotalRuns"] == 20


@pytest.mark.asyncio
async def test_handle_writes_max_iterations_for_adaptive_search(monkeypatch):
    """Adaptive sweeps don't know the final variation count up front — the
    kopf create handler writes `max_iterations` (upper bound) to
    `status.totalVariations` and `max_iterations * trials` to
    `status.maxTotalRuns`. The controller pod's terminal-phase write
    supersedes these on early convergence (defended by the existing
    `_conditional_phase_set` test-op guard in `child_rollup.py`).
    """
    body = _valid_body()
    body["spec"]["multiRun"]["numRuns"] = 3
    # Adaptive search is now a sweep variant (`sweep.type=adaptive_search`),
    # not a multi_run sub-block. Replace the default grid sweep on _valid_body.
    body["spec"]["sweep"] = {
        "type": "adaptive_search",
        "search_space": [
            {
                "path": "phases.profiling.concurrency",
                "lo": 1,
                "hi": 1000,
                "kind": "int",
            }
        ],
        "objectives": [
            {
                "metric": "output_token_throughput",
                "stat": "avg",
                "direction": "maximize",
            }
        ],
        "max_iterations": 10,
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert patch.status["totalVariations"] == 10
    assert patch.status["maxTotalRuns"] == 30


@pytest.mark.asyncio
async def test_handle_adaptive_search_without_trials_defaults_to_one(monkeypatch):
    """When `multiRun.adaptiveSearch` is set but `multiRun.trials` is unset,
    `maxTotalRuns` falls back to `max_iterations * 1` (single trial per
    variation) — mirroring the in-process default."""
    body = _valid_body()
    body["spec"]["multiRun"].pop("numRuns", None)
    body["spec"]["sweep"] = {
        "type": "adaptive_search",
        "search_space": [
            {
                "path": "phases.profiling.concurrency",
                "lo": 1,
                "hi": 1000,
                "kind": "int",
            }
        ],
        "objectives": [
            {
                "metric": "output_token_throughput",
                "stat": "avg",
                "direction": "maximize",
            }
        ],
        "max_iterations": 7,
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert patch.status["totalVariations"] == 7
    assert patch.status["maxTotalRuns"] == 7


# ===========================================================================
# Adversarial regression-locks for second-pass fixes (commit 793260d7b):
# `_create_or_skip_409` and `_create_or_skip_409_custom` must wrap non-409
# ApiException + connection errors in `kopf.TemporaryError(delay=30)`.
# ===========================================================================


@pytest.mark.asyncio
async def test_create_or_skip_409_non_409_apiexception_wraps_in_temporary_error():
    """ApiException(status=503) must surface as kopf.TemporaryError so kopf
    backs off rather than retrying unboundedly."""
    from kubernetes_asyncio.client import ApiException

    create_fn = AsyncMock(side_effect=ApiException(status=503, reason="Unavailable"))
    with pytest.raises(kopf.TemporaryError):
        await sweep_create._create_or_skip_409(create_fn, "ns", object())


@pytest.mark.asyncio
async def test_create_or_skip_409_409_apiexception_swallowed():
    """409 (AlreadyExists) is the idempotent-reconcile path — must not raise."""
    from kubernetes_asyncio.client import ApiException

    create_fn = AsyncMock(side_effect=ApiException(status=409, reason="AlreadyExists"))
    # Must not raise.
    await sweep_create._create_or_skip_409(create_fn, "ns", object())


@pytest.mark.asyncio
async def test_create_or_skip_409_aiohttp_connection_error_wraps_temporary_error():
    """aiohttp connection error must surface as kopf.TemporaryError too —
    transient network blips retry with backoff."""
    import aiohttp

    create_fn = AsyncMock(
        side_effect=aiohttp.ClientConnectionError("connection refused")
    )
    with pytest.raises(kopf.TemporaryError):
        await sweep_create._create_or_skip_409(create_fn, "ns", object())


# ===========================================================================
# Adversarial regression-locks for pod-spec lifting (second-pass fix).
# `_create_sweep_controller_jobset` must lift container-level resources,
# containerSecurityContext, pod-level securityContext, and merge user env
# (with reserved AIPERF_SWEEP_* names taking precedence).
# ===========================================================================


def _valid_workload_spec(**overrides):
    """Minimal flat envelope dict (image/podTemplate/benchmark) accepted by the JobSet builder.

    `overrides` are merged into `podTemplate`.
    """
    pod_template = dict(overrides)
    return {
        "image": "x:latest",
        "podTemplate": pod_template,
        "benchmark": {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ],
        },
    }


async def _capture_jobset_body(monkeypatch, workload_spec) -> dict:
    """Drive `_create_sweep_controller_jobset` and capture the JobSet body
    passed to `create_namespaced_custom_object` (plural=jobsets)."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    captured: dict = {}

    create_mock = AsyncMock()

    async def _capture(**kwargs):
        # Only capture the jobsets call.
        if kwargs.get("plural") == "jobsets":
            captured["body"] = kwargs.get("body")
            captured["kwargs"] = kwargs

    create_mock.side_effect = _capture

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kw):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(create_namespaced_custom_object=create_mock),
    )

    await sweep_create._create_sweep_controller_jobset(
        name="s",
        namespace="ns",
        sweep_uid="uid",
        epoch="1714000000",
        template_spec=workload_spec,
    )

    return captured["body"]


def _container_from_jobset(body: dict) -> dict:
    return body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]


def _pod_spec_from_jobset(body: dict) -> dict:
    return body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_container_resources(monkeypatch):
    """`podTemplate.resources` must land on container.resources."""
    workload_spec = _valid_workload_spec(
        resources={"requests": {"cpu": "500m", "memory": "1Gi"}}
    )
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    container = _container_from_jobset(body)
    assert container["resources"] == {"requests": {"cpu": "500m", "memory": "1Gi"}}


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_container_security_context(
    monkeypatch,
):
    """`podTemplate.containerSecurityContext` must land on container.securityContext."""
    workload_spec = _valid_workload_spec(
        containerSecurityContext={
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "allowPrivilegeEscalation": False,
        }
    )
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    container = _container_from_jobset(body)
    assert container["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "allowPrivilegeEscalation": False,
    }


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_lifts_pod_security_context(monkeypatch):
    """Pod-level `podTemplate.securityContext` must land on pod_spec.securityContext."""
    workload_spec = _valid_workload_spec(
        securityContext={"fsGroup": 2000, "runAsNonRoot": True}
    )
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    pod_spec = _pod_spec_from_jobset(body)
    assert pod_spec["securityContext"] == {"fsGroup": 2000, "runAsNonRoot": True}


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_enables_dns_hostnames(monkeypatch):
    """The sweep-controller JobSet must set spec.network.enableDNSHostnames so the
    headless service exists and the operator can harvest the emptyDir-only aggregate
    from the controller pod's stable DNS name (controller_dns_name(...))."""
    workload_spec = _valid_workload_spec()
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    assert body["spec"]["network"]["enableDNSHostnames"] is True


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_merges_user_env_reserved_wins(
    monkeypatch,
):
    """User env (HTTP_PROXY) is merged in; reserved AIPERF_SWEEP_NAME from user
    is overridden by the controller's value, not vice versa."""
    workload_spec = _valid_workload_spec(
        env=[
            {"name": "HTTP_PROXY", "value": "http://proxy"},
            {"name": "AIPERF_SWEEP_NAME", "value": "hijack"},
        ]
    )
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    container = _container_from_jobset(body)
    env_by_name = {e["name"]: e["value"] for e in container["env"]}
    assert env_by_name.get("HTTP_PROXY") == "http://proxy", "user env must merge in"
    assert env_by_name.get("AIPERF_SWEEP_NAME") == "s", (
        "reserved AIPERF_SWEEP_NAME must keep controller's value, not user's 'hijack'"
    )
    # And the reserved var only appears once (no duplicate from user merge).
    sweep_name_entries = [
        e for e in container["env"] if e["name"] == "AIPERF_SWEEP_NAME"
    ]
    assert len(sweep_name_entries) == 1


@pytest.mark.asyncio
async def test_create_sweep_controller_jobset_injects_operator_base_url(monkeypatch):
    """``AIPERF_OPERATOR_BASE_URL`` must be plumbed onto the sweep-controller pod
    so the executor's per-child summary fallback can reach the operator's
    PVC-backed results API instead of the about-to-be-deleted child sidecar.
    BASE_URL points at the operator's only FastAPI surface (results-server
    container, port 8081 in the chart) — the operator container has no
    /api/v1/* routers, only kopf health/metrics.
    """
    from aiperf.operator.environment import OperatorEnvironment

    monkeypatch.setattr(
        OperatorEnvironment.SERVICE, "BASE_URL", "https://op.example:9091"
    )
    workload_spec = _valid_workload_spec()
    body = await _capture_jobset_body(monkeypatch, workload_spec)
    container = _container_from_jobset(body)
    env_by_name = {e["name"]: e["value"] for e in container["env"]}
    assert env_by_name.get("AIPERF_OPERATOR_BASE_URL") == "https://op.example:9091"


# ===========================================================================
# Adversarial regression-lock: Role grants events create/patch.
# ===========================================================================


@pytest.mark.asyncio
async def test_provision_rbac_role_grants_events_create_patch(monkeypatch):
    """`_provision_rbac` Role must include a PolicyRule for events.create/patch
    (so the sweep-controller can emit kubectl-visible events)."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    captured: dict = {}

    async def _capture_role(_namespace, body):
        # The Role body has a `rules` attribute (V1Role).
        captured["role_rules"] = body.rules

    async def _noop_sa(_ns, _body):
        return None

    async def _noop_binding(_ns, _body):
        return None

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kw):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )

    core = MagicMock()
    core.create_namespaced_service_account = AsyncMock(side_effect=_noop_sa)
    rbac = MagicMock()
    rbac.create_namespaced_role = AsyncMock(side_effect=_capture_role)
    rbac.create_namespaced_role_binding = AsyncMock(side_effect=_noop_binding)

    import kubernetes_asyncio.client as k8s_client_mod

    monkeypatch.setattr(k8s_client_mod, "CoreV1Api", lambda _api: core)
    monkeypatch.setattr(k8s_client_mod, "RbacAuthorizationV1Api", lambda _api: rbac)

    await sweep_create._provision_rbac(name="s", namespace="ns", sweep_uid="uid")

    rules = captured["role_rules"]
    # Find the events rule.
    events_rules = [
        r for r in rules if r.api_groups == [""] and r.resources == ["events"]
    ]
    assert len(events_rules) == 1, "events PolicyRule missing"
    verbs = set(events_rules[0].verbs)
    assert "create" in verbs and "patch" in verbs
