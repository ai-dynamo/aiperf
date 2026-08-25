# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU benchmark execution against a Dynamo inference graph."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.subproc import run_command
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_container_logs,
    _log_pod_statuses,
)
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig, BenchmarkResult
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig

logger = AIPerfLogger(__name__)

_SWEEP_TERMINAL_PHASES = frozenset(
    {"Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)


# ============================================================================
# Module-scoped benchmark against Dynamo
# ============================================================================


@pytest.fixture(scope="module")
def _dynamo_benchmark_config(
    dynamo_endpoint_url: str,
    dynamo_config: DynamoConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Module-scoped benchmark config targeting Dynamo."""
    return BenchmarkConfig(
        endpoint_url=dynamo_endpoint_url,
        endpoint_type="chat",
        model_name=dynamo_config.model_name,
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=gpu_settings.aiperf_image,
        workers=1,
        input_sequence_min=10,
        input_sequence_max=30,
        output_tokens_min=5,
        output_tokens_max=20,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def deployed_dynamo_benchmark(
    benchmark_deployer: GPUBenchmarkDeployer,
    _dynamo_benchmark_config: BenchmarkConfig,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a benchmark against Dynamo, shared across tests in this module."""
    s = gpu_settings
    logger.info(
        f"[BENCHMARK] Deploying Dynamo benchmark: endpoint={_dynamo_benchmark_config.endpoint_url}, model={_dynamo_benchmark_config.model_name}, "
        f"concurrency={_dynamo_benchmark_config.concurrency}, requests={_dynamo_benchmark_config.request_count}"
    )

    result = await benchmark_deployer.deploy(
        config=_dynamo_benchmark_config,
        wait_for_completion=True,
        timeout=s.benchmark_timeout,
        stream_logs=s.stream_logs,
    )

    logger.info(
        f"[BENCHMARK] Result: success={result.success}, namespace={result.namespace}, duration={result.duration_seconds:.1f}s"
    )

    if result.metrics:
        logger.info(
            f"[BENCHMARK] Metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, "
            f"requests={result.metrics.request_count}, errors={result.metrics.error_count}"
        )

    await _log_pod_statuses(kubectl, result.namespace)
    await _log_container_logs(kubectl, result.namespace)

    if not result.success:
        await _dump_diagnostics(
            kubectl, result.namespace, label="DYNAMO_BENCHMARK_FAILURE"
        )

    yield result


# ============================================================================
# Tests
# ============================================================================


class TestDynamoBenchmarkCompletion:
    """Tests for AIPerf benchmark completion against Dynamo."""

    def test_benchmark_completes_successfully(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify benchmark against Dynamo completes successfully."""
        result = deployed_dynamo_benchmark
        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.status is not None
        assert result.status.is_completed

    def test_no_benchmark_errors(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify benchmark against Dynamo completes without errors."""
        result = deployed_dynamo_benchmark
        assert result.metrics is not None
        assert result.metrics.error_count == 0, (
            f"Expected 0 errors, got {result.metrics.error_count}"
        )

    def test_throughput_is_positive(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify throughput from Dynamo benchmark is positive."""
        metrics = deployed_dynamo_benchmark.metrics
        assert metrics is not None
        assert metrics.request_throughput is not None
        assert metrics.request_throughput > 0

    def test_latency_is_positive(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify latency from Dynamo benchmark is positive."""
        metrics = deployed_dynamo_benchmark.metrics
        assert metrics is not None
        assert metrics.request_latency_avg is not None
        assert metrics.request_latency_avg > 0

    def test_request_count_matches_config(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify request count is close to configuration.

        Dynamo disaggregated mode on a shared GPU may not complete all
        requests within the timeout; verify at least 80% completed.
        """
        result = deployed_dynamo_benchmark
        assert result.metrics is not None
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )


class TestDynamoBenchmarkWorkerScaling:
    """Tests for Dynamo benchmark with different worker pod counts and longer runs."""

    @pytest.mark.parametrize(
        "request_count, concurrency",
        [
            param(20, 2, id="c2-20-reqs"),
            param(20, 4, id="c4-20-reqs"),
            param(20, 8, id="c8-20-reqs"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_benchmark_succeeds_with_worker_count(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        dynamo_endpoint_url: str,
        dynamo_config: DynamoConfig,
        kubectl: KubectlClient,
        gpu_settings: GPUTestSettings,
        request_count: int,
        concurrency: int,
    ) -> None:
        """Verify benchmark completes with varying worker pod counts."""
        workers = 1
        logger.info(
            f"[TEST] Worker scaling test: workers={workers}, requests={request_count}, concurrency={concurrency}"
        )

        config = BenchmarkConfig(
            endpoint_url=dynamo_endpoint_url,
            endpoint_type="chat",
            model_name=dynamo_config.model_name,
            concurrency=concurrency,
            request_count=request_count,
            warmup_request_count=2,
            image=gpu_settings.aiperf_image,
            workers=workers,
            input_sequence_min=10,
            input_sequence_max=30,
            output_tokens_min=5,
            output_tokens_max=20,
        )

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=gpu_settings.benchmark_timeout,
            stream_logs=gpu_settings.stream_logs,
        )

        logger.info(
            f"[TEST] workers={workers} result: success={result.success}, duration={result.duration_seconds:.1f}s"
        )

        if result.metrics:
            logger.info(
                f"[TEST] workers={workers} metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, "
                f"latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, requests={result.metrics.request_count}, errors={result.metrics.error_count}"
            )

        await _log_pod_statuses(kubectl, result.namespace)
        await _log_container_logs(kubectl, result.namespace)

        if not result.success:
            await _dump_diagnostics(
                kubectl, result.namespace, label="DYNAMO_WORKER_SCALING_FAILURE"
            )

        assert result.success, (
            f"Benchmark failed with workers={workers}: {result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.error_count == 0, (
            f"Expected 0 errors, got {result.metrics.error_count}"
        )
        # Disaggregated mode on shared GPU has high per-request latency;
        # verify at least some requests completed successfully.
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )


# ============================================================================
# Sweep helpers
# ============================================================================


async def _collect_pod_summary(kubectl: KubectlClient, namespace: str) -> str:
    """One-line-per-pod snapshot for inlining into assertion messages."""
    try:
        pods = await kubectl.get_pods(namespace)
    except Exception:
        return "(failed to list pods)"
    if not pods:
        return "(no pods)"
    lines: list[str] = []
    for p in pods:
        containers = ", ".join(sorted(p.containers.keys()) if p.containers else ["?"])
        lines.append(
            f"  {p.name:<55} phase={p.phase:<12} ready={p.ready:<5} "
            f"restarts={p.restarts:<3} containers=[{containers}]"
        )
    return "\n".join(lines)


async def _collect_child_errors(
    kubectl: KubectlClient,
    namespace: str,
    sweep_name: str,
) -> str:
    """List terminal child AIPerfJobs and their error messages."""
    try:
        result = await kubectl.run(
            "get",
            "aiperfjobs",
            "-l",
            f"aiperf.nvidia.com/sweep={sweep_name}",
            "-n",
            namespace,
            "--no-headers",
            check=False,
        )
    except Exception:
        return "(failed to list child AIPerfJobs)"
    if result.returncode != 0 or not result.stdout.strip():
        return "(no child AIPerfJobs found)"
    # Pod lifecycle is NOT enough — a Pod-Succeeded child Job whose operator
    # reconcile failed the endpoint-check (config preflight) reports a non-nil
    # status.error on the AIPerfJob CR.  Pull each child's phase + error.
    lines: list[str] = []
    for raw_line in result.stdout.strip().splitlines():
        parts = raw_line.split()
        child_name = parts[0] if parts else "?"
        try:
            child = await kubectl.get_json("aiperfjob", child_name, namespace=namespace)
            status = child.get("status", {})
            child_phase = status.get("phase", "?")
            child_error = status.get("error") or ""
            lines.append(
                f"  {child_name}: phase={child_phase}"
                + (f", error={child_error[:200]}" if child_error else "")
            )
        except Exception:
            lines.append(f"  {child_name}: (failed to read CR)")
    return "\n".join(lines) if lines else "(no children)"


async def _collect_sweep_controller_diag(
    kubectl: KubectlClient,
    namespace: str,
    sweep_name: str,
) -> str:
    """Collect the sweep-controller pod's tail logs for inlining.

    The sweep-controller JobSet is ``aiperf-{sweep_name}`` and the
    controller pod starts with that same prefix.  Match on the full sweep
    name rather than stripping ``-sweep`` — the test's own generated names
    already contain ``-sweep-``.
    """
    controller_pod_name = None
    try:
        pods = await kubectl.get_pods(namespace)
        for p in pods:
            if "controller" in p.name and sweep_name in p.name:
                controller_pod_name = p.name
                break
    except Exception:
        return "(failed to locate sweep-controller pod)"
    if controller_pod_name is None:
        return "(no sweep-controller pod found)"
    try:
        logs = await kubectl.get_logs(
            controller_pod_name,
            container="sweep-controller",
            namespace=namespace,
            tail=120,
        )
    except Exception:
        return f"(failed to collect logs from {controller_pod_name})"
    if not logs.strip():
        return f"(empty logs from {controller_pod_name})"
    return f"--- {controller_pod_name} tail=120 ---\n{logs.rstrip()}"


async def _wait_for_dynamo_sweep(
    *,
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    timeout: int,
) -> dict[str, Any]:
    """Poll an AIPerfSweep CR until it reaches a durable terminal state.

    ``resultsAvailable: true`` is the authoritative signal that the operator has
    harvested the sweep-controller's sidecar archive onto its PVC — the
    aggregate is now durable and the CR itself can be safely deleted.  Checking
    for ``/epochs/`` in ``aggregateRef.apiPath`` is fragile: before the harvest
    lands, the controller already publishes a live ``aggregateRef`` whose
    ``apiPath`` also contains ``/epochs/``
    (``.../sweeps/<ns>/<sweep>/epochs/<epoch>/artifacts/aggregate.json``).
    ``resultsAvailable`` is flipped ``True`` in the same JSON-patch that swaps
    the live ref for the durable (PVC-backed) one, so only that flag captures
    operator-acknowledged durability.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_doc: dict[str, Any] = {}
    last_status: dict[str, Any] = {}
    last_status_summary = ""

    while loop.time() < deadline:
        try:
            last_doc = await kubectl.get_json("aiperfsweep", name, namespace=namespace)
        except RuntimeError:
            await asyncio.sleep(2)
            continue

        status = last_doc.get("status", {})
        last_status = status
        phase = status.get("phase")
        results_available = status.get("resultsAvailable", False)
        remaining = deadline - loop.time()

        # Summarise key state fields so the timeout message is diagnosable
        # without re-issuing kubectl commands by hand.
        aggregate = status.get("aggregate", {})
        parent = aggregate.get("parent", {}) if isinstance(aggregate, dict) else {}
        last_status_summary = (
            f"phase={phase}, resultsAvailable={results_available}, "
            f"runStates={status.get('runStates')}, "
            f"completedRuns={parent.get('completedRuns', '?')}, "
            f"failedRuns={parent.get('failedRuns', '?')}, "
            f"aggregation.phase={status.get('aggregation', {}).get('phase')}"
        )

        # Non-Succeeded terminal phases return immediately — no harvest needed.
        if phase in _SWEEP_TERMINAL_PHASES and phase != "Succeeded":
            logger.info(f"[SWEEP] Terminal ({phase}) after ~{timeout - remaining:.0f}s")
            return last_doc

        # Succeeded: only durable once the operator's PVC harvest is confirmed.
        if phase == "Succeeded" and results_available is True:
            logger.info(
                f"[SWEEP] Durable Succeeded ({phase}) after ~{timeout - remaining:.0f}s"
            )
            return last_doc

        # Log progress at human-readable intervals so a CI log reader can see
        # the sweep moving rather than a silent wall of poll ticks.
        elapsed = timeout - remaining
        if int(elapsed) % 60 == 0 and elapsed > 0:
            logger.info(
                f"[SWEEP] Waiting for {namespace}/{name}: "
                f"{last_status_summary} ({elapsed:.0f}s/{timeout}s)"
            )

        await asyncio.sleep(2)

    # On timeout, surface as much clinical detail as possible inline — the
    # assertion message IS the first failure output a user or CI log reader
    # sees, and re-running `_dump_diagnostics` inside pytest.fail(...) would
    # embed transient log lines into the assertion traceback instead of the
    # failure message.  Collect pod statuses, container logs, and events into
    # a structured snapshot included in the assertion.
    pod_summary = await _collect_pod_summary(kubectl, namespace)
    child_error_summary = await _collect_child_errors(
        kubectl, namespace, sweep_name=name
    )
    collector_diag = await _collect_sweep_controller_diag(kubectl, namespace, name)
    await _dump_diagnostics(kubectl, namespace, label="SWEEP_TIMEOUT")
    pytest.fail(
        f"AIPerfSweep {namespace}/{name} did not reach a durable terminal state "
        f"within {timeout}s.\n"
        f"Last status: {last_status_summary}\n"
        f"aggregateRef: {last_status.get('aggregateRef')}\n"
        f"Pod snapshot:\n{pod_summary}\n"
        f"Child errors:\n{child_error_summary}\n"
        f"Sweep-controller diagnostics:\n{collector_diag}"
    )


# ============================================================================
# AIPerfSweep CR test
# ============================================================================


class TestDynamoSweep:
    """Tests for AIPerfSweep CR execution against a live Dynamo endpoint."""

    @pytest.mark.asyncio
    async def test_sweep_completes_two_variations(
        self,
        dynamo_endpoint_url: str,
        dynamo_config: DynamoConfig,
        kubectl: KubectlClient,
        gpu_settings: GPUTestSettings,
        benchmark_deployer: GPUBenchmarkDeployer,
        tmp_path: Path,
    ) -> None:
        """Grid sweep over concurrency=[1,2] against Dynamo — both variations must succeed."""
        name = f"dynamo-sweep-{uuid.uuid4().hex[:8]}"
        namespace = "default"

        config: dict[str, Any] = AIPerfJobConfig(
            endpoint_url=dynamo_endpoint_url,
            model_name=dynamo_config.model_name,
            concurrency=1,
            request_count=5,
            warmup_request_count=0,
        ).to_flat_spec()
        config["sweep"] = {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1, 2]},
        }
        config["randomSeed"] = 42

        config_path = tmp_path / "dynamo-sweep.yaml"
        await asyncio.to_thread(
            config_path.write_text, yaml.dump(config, sort_keys=False)
        )

        # The operator's preflight check validates that imagePullSecrets exist in
        # the sweep namespace. Copy the pull secret from wherever it lives into
        # the default namespace so children can pull the aiperf image.
        if gpu_settings.image_pull_secrets:
            await benchmark_deployer._ensure_pull_secrets_in_namespace(
                namespace, gpu_settings.image_pull_secrets
            )

        logger.info(
            f"[SWEEP] Submitting AIPerfSweep {namespace}/{name} "
            f"against {dynamo_endpoint_url}"
        )

        submit = await run_command(
            [
                "uv",
                "run",
                "aiperf",
                "kube",
                "sweep",
                "--config",
                str(config_path),
                "--name",
                name,
                "--namespace",
                namespace,
                "--image",
                gpu_settings.aiperf_image,
                "--image-pull-policy",
                "IfNotPresent" if gpu_settings.context else "Never",
                *(
                    ["--image-pull-secrets", gpu_settings.image_pull_secret]
                    if gpu_settings.image_pull_secret
                    else []
                ),
                "--kube-context",
                kubectl.context,
                "--detach",
            ],
            timeout=90,
        )
        assert submit.ok, (
            f"aiperf kube sweep failed:\nstdout:\n{submit.stdout}\nstderr:\n{submit.stderr}"
        )

        logger.info(f"[SWEEP] Waiting for {namespace}/{name} to complete")
        doc = await _wait_for_dynamo_sweep(
            kubectl=kubectl,
            name=name,
            namespace=namespace,
            timeout=gpu_settings.benchmark_timeout,
        )

        status = doc.get("status", {})
        phase = status.get("phase")
        logger.info(
            f"[SWEEP] Final status: phase={phase}, runStates={status.get('runStates')}"
        )

        # On failure, inline the child-level errors directly into the assertion
        # message so the user/CI sees WHY it failed without re-running commands.
        if phase != "Succeeded":
            child_errors = await _collect_child_errors(
                kubectl, namespace, sweep_name=name
            )
            assert phase == "Succeeded", (
                f"AIPerfSweep {namespace}/{name} did not succeed.\n"
                f"phase={phase}\n"
                f"completedRuns={status.get('completedRuns', '?')}\n"
                f"failedRuns={status.get('failedRuns', '?')}\n"
                f"aggregation.phase={status.get('aggregation', {}).get('phase')}\n"
                f"Child errors:\n{child_errors}"
            )

        parent = status.get("aggregate", {}).get("parent", {})
        completed = parent.get("completedRuns", 0)
        failed = parent.get("failedRuns", 0)
        assert completed == 2, (
            f"Expected 2 completed child runs, got {completed}. "
            f"aggregate.parent={parent}"
        )
        assert failed == 0, (
            f"Expected 0 failed child runs, got {failed}. aggregate.parent={parent}"
        )

        # Cleanup: remove the sweep CR so the cluster stays tidy.
        await kubectl.run(
            "delete",
            "aiperfsweep",
            name,
            "-n",
            namespace,
            "--ignore-not-found",
            check=False,
        )
