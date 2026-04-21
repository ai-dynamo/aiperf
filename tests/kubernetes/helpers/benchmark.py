# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark deployment and result collection for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import (
    JobSetStatus,
    KubectlClient,
    PodStatus,
    background_status,
)
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.watchdog import BenchmarkWatchdog, make_watchdog_source

logger = AIPerfLogger(__name__)


@asynccontextmanager
async def timed_operation(operation: str):
    """Context manager that logs timing information for an operation."""
    logger.info(f"[START] {operation}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[DONE] {operation} ({elapsed:.2f}s)")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    """Inference server endpoint URL."""

    endpoint_type: str = "chat"
    """Endpoint type (chat, completions, embeddings)."""

    model_name: str = "mock-model"
    """Model name to benchmark."""

    concurrency: int = 5
    """Number of concurrent requests."""

    request_count: int = 50
    """Total number of requests to send."""

    warmup_request_count: int = 5
    """Number of warmup requests before measurement."""

    concurrency_ramp_duration: float | None = None
    """Duration in seconds to ramp up concurrency, or None for instant."""

    tokenizer_name: str = "gpt2"
    """Tokenizer name for token counting."""

    input_sequence_min: int = 50
    """Minimum input sequence length in tokens."""

    input_sequence_max: int = 100
    """Maximum input sequence length in tokens."""

    output_tokens_min: int = 10
    """Minimum output token count."""

    output_tokens_max: int = 50
    """Maximum output token count."""

    image: str = "aiperf:local"
    """Container image for benchmark pods."""

    workers: int = 1
    """Number of worker pods."""

    queue_name: str | None = None
    """Kueue LocalQueue name for gang-scheduling, or None."""

    priority_class: str | None = None
    """Kubernetes PriorityClass name, or None."""

    def to_temp_file(self) -> Path:
        """Write a placeholder config file (not used by generate, kept for API compat).

        Returns:
            Path to the temporary config file.
        """
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="benchmark-config-")
        os.close(fd)
        return Path(path)


@dataclass
class BenchmarkMetrics:
    """Extracted benchmark metrics."""

    request_throughput: float | None = None
    """Requests per second."""

    output_token_throughput: float | None = None
    """Output tokens per second."""

    request_count: int | None = None
    """Total completed requests."""

    request_latency_avg: float | None = None
    """Average request latency in milliseconds."""

    request_latency_min: float | None = None
    """Minimum request latency in milliseconds."""

    request_latency_max: float | None = None
    """Maximum request latency in milliseconds."""

    request_latency_p50: float | None = None
    """Median request latency in milliseconds."""

    request_latency_p90: float | None = None
    """90th percentile request latency in milliseconds."""

    request_latency_p99: float | None = None
    """99th percentile request latency in milliseconds."""

    input_sequence_length: float | None = None
    """Average input sequence length in tokens."""

    output_sequence_length_avg: float | None = None
    """Average output sequence length in tokens."""

    error_count: int = 0
    """Number of failed requests."""

    raw_logs: str = ""
    """Raw controller log output for fallback parsing."""

    @classmethod
    def from_api_results(cls, api_results: dict[str, Any]) -> BenchmarkMetrics:
        """Parse metrics from API response or operator CR results.

        Supports two formats:
        1. API records format: ``{"results": {"records": [{"tag": "...", "avg": N}]}}``
        2. File/CR format: ``{"results": {"request_throughput": {"avg": N}, ...}}``

        Args:
            api_results: JSON response from /api/results or operator CR.

        Returns:
            Extracted metrics.
        """
        metrics = cls()

        inner = api_results.get("results", {})
        if isinstance(inner, dict):
            inner = inner.get("results", inner)
        if not isinstance(inner, dict):
            return metrics

        # Map tag names to metric attributes
        tag_map: dict[str, str] = {
            "request_throughput": "request_throughput",
            "output_token_throughput": "output_token_throughput",
            "request_count": "request_count",
            "request_latency": "request_latency_avg",
            "error_request_count": "error_count",
        }

        # Strategy 1: records list (API format)
        records = inner.get("records", [])
        if isinstance(records, list) and records:
            for rec in records:
                tag = rec.get("tag", "")
                attr = tag_map.get(tag)
                if attr is None:
                    continue
                avg = rec.get("avg")
                if avg is None:
                    continue
                if attr in ("request_count", "error_count"):
                    setattr(metrics, attr, int(avg))
                else:
                    setattr(metrics, attr, float(avg))

                if tag == "request_latency":
                    for key in ("min", "max", "p50", "p90", "p99"):
                        val = rec.get(key)
                        if val is not None:
                            setattr(metrics, f"request_latency_{key}", float(val))
            return metrics

        # Strategy 2: flat dict (file/CR format)
        # API format nests metrics under {"metrics": {...}}, file format is flat
        metrics_dict = inner.get("metrics", {})
        if isinstance(metrics_dict, dict) and metrics_dict:
            source = metrics_dict
        else:
            source = inner
        for tag, attr in tag_map.items():
            val = source.get(tag)
            if isinstance(val, dict):
                avg = val.get("avg")
            elif isinstance(val, int | float):
                avg = val
            else:
                continue
            if avg is None:
                continue
            if attr in ("request_count", "error_count"):
                setattr(metrics, attr, int(avg))
            else:
                setattr(metrics, attr, float(avg))

            if tag == "request_latency" and isinstance(val, dict):
                for key in ("min", "max", "p50", "p90", "p99"):
                    pval = val.get(key)
                    if pval is not None:
                        setattr(metrics, f"request_latency_{key}", float(pval))

        return metrics

    @classmethod
    def from_logs(cls, logs: str) -> BenchmarkMetrics:
        """Parse metrics from system-controller logs (fallback).

        Args:
            logs: Raw log content.

        Returns:
            Extracted metrics.
        """
        metrics = cls(raw_logs=logs)

        # Strip ANSI codes
        clean_logs = re.sub(r"\x1b\[[0-9;]*m", "", logs)

        # Parse metrics table using regex
        patterns = {
            "request_throughput": r"Request Throughput.*?│\s*([\d,]+\.?\d*)",
            "output_token_throughput": r"Output Token Throughput.*?│\s*([\d,]+\.?\d*)",
            "request_count": r"Request Count.*?│\s*([\d,]+\.?\d*)",
            "request_latency_avg": r"Request Latency.*?│\s*([\d,]+\.?\d*)",
        }

        for attr, pattern in patterns.items():
            match = re.search(pattern, clean_logs)
            if match:
                value_str = match.group(1).replace(",", "")
                try:
                    if attr == "request_count":
                        setattr(metrics, attr, int(float(value_str)))
                    else:
                        setattr(metrics, attr, float(value_str))
                except ValueError:
                    pass

        # Parse error count from "Errors: X / Y" pattern
        error_match = re.search(r"Errors:\s*(\d+)\s*/\s*\d+", clean_logs)
        if error_match:
            metrics.error_count = int(error_match.group(1))

        return metrics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    namespace: str
    """Kubernetes namespace for this benchmark."""

    jobset_name: str
    """Name of the JobSet resource."""

    job_id: str
    """Unique benchmark job identifier."""

    config: BenchmarkConfig
    """Configuration used for this benchmark."""

    status: JobSetStatus | None = None
    """Final JobSet status, or None if not collected."""

    metrics: BenchmarkMetrics | None = None
    """Parsed benchmark metrics, or None if unavailable."""

    api_results: dict[str, Any] | None = None
    """Raw API results from the controller, or None."""

    pods: list[PodStatus] = field(default_factory=list)
    """Pod statuses at collection time."""

    duration_seconds: float = 0.0
    """Total wall-clock duration in seconds."""

    success: bool = False
    """Whether the benchmark completed successfully."""

    error_message: str | None = None
    """Error message if the benchmark failed."""

    @property
    def controller_pod(self) -> PodStatus | None:
        """Get the controller pod for this benchmark's job_id."""
        for pod in self.pods:
            if "controller" in pod.name and self.job_id in pod.name:
                return pod
        # Fallback: any controller pod
        for pod in self.pods:
            if "controller" in pod.name:
                return pod
        return None

    @property
    def worker_pods(self) -> list[PodStatus]:
        """Get worker pods."""
        return [
            p for p in self.pods if "worker" in p.name and "controller" not in p.name
        ]

    def print_results(self, header: str = "BENCHMARK RESULTS") -> None:
        """Print a formatted summary of the benchmark results.

        Args:
            header: Header text for the output block.
        """
        sep = "=" * 70
        thin = "-" * 70
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"  Job ID:    {self.job_id}")
        print(f"  Namespace: {self.namespace}")
        print(f"  Success:   {self.success}")
        print(f"  Duration:  {self.duration_seconds:.2f}s")
        if self.error_message:
            print(f"  Error:     {self.error_message}")
        if self.status:
            print(f"  JobSet:    {self.status.terminal_state}")

        if self.api_results:
            self._print_api_results(thin)

        if self.metrics:
            print(f"\n  {thin}")
            print("  PARSED METRICS (from controller logs)")
            print(f"  {thin}")
            self._print_metric("Request Count", self.metrics.request_count, "")
            self._print_metric(
                "Request Throughput", self.metrics.request_throughput, "req/s"
            )
            self._print_metric(
                "Output Token Throughput",
                self.metrics.output_token_throughput,
                "tok/s",
            )
            self._print_metric(
                "Request Latency Avg", self.metrics.request_latency_avg, "ms"
            )
            self._print_metric("Error Count", self.metrics.error_count, "")

        print(sep + "\n")

    def _print_api_results(self, thin: str) -> None:
        """Print parsed API results in a readable table format."""
        results = self.api_results
        if not results:
            return

        inner = results.get("results", {})
        if isinstance(inner, dict):
            inner = inner.get("results", inner)

        print(f"\n  {thin}")
        print(f"  API RESULTS (status={results.get('status', 'unknown')})")
        print(f"  {thin}")

        # Print metric records as a table
        records = inner.get("records", []) if isinstance(inner, dict) else []
        if records:
            hdr = f"  {'Metric':<30} {'Avg':>12} {'P50':>12} {'P99':>12} {'Min':>12} {'Max':>12} {'Unit':<8}"
            print(hdr)
            print(f"  {'-' * (len(hdr) - 2)}")
            for rec in records:
                tag = rec.get("header", rec.get("tag", "?"))
                unit = rec.get("unit", "")
                avg = rec.get("avg")
                p50 = rec.get("p50")
                p99 = rec.get("p99")
                mn = rec.get("min")
                mx = rec.get("max")
                print(
                    f"  {tag:<30} "
                    f"{self._fmt(avg):>12} "
                    f"{self._fmt(p50):>12} "
                    f"{self._fmt(p99):>12} "
                    f"{self._fmt(mn):>12} "
                    f"{self._fmt(mx):>12} "
                    f"{unit:<8}"
                )

        # Print summary fields
        # NOTE: "completed" is the number of metric record types, not requests.
        # "total_expected" is the configured request count.
        summary_fields = [
            ("completed", "Metric Records"),
            ("total_expected", "Total Expected"),
            ("was_cancelled", "Was Cancelled"),
        ]
        printed_summary = False
        for key, label in summary_fields:
            val = inner.get(key) if isinstance(inner, dict) else None
            if val is not None:
                if not printed_summary:
                    print()
                    printed_summary = True
                print(f"  {label:<30} {val}")

        # Print errors
        errors = results.get("errors", [])
        if not errors and isinstance(inner, dict):
            errors = inner.get("error_summary", [])
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for err in errors[:10]:
                print(f"    - {err}")

    @staticmethod
    def _fmt(val: float | int | None) -> str:
        """Format a numeric value for display."""
        if val is None:
            return "N/A"
        if isinstance(val, int):
            return f"{val:,}"
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        if abs(val) >= 1:
            return f"{val:.4f}"
        return f"{val:.6f}"

    @staticmethod
    def _print_metric(label: str, value: float | int | None, unit: str) -> None:
        """Print a single metric line."""
        if value is None:
            print(f"    {label + ':':<30} N/A")
        elif isinstance(value, int):
            print(f"    {label + ':':<30} {value:,} {unit}")
        else:
            print(f"    {label + ':':<30} {value:,.2f} {unit}")


class BenchmarkDeployer:
    """Deploys and manages AIPerf benchmarks on Kubernetes."""

    def __init__(
        self,
        kubectl: KubectlClient,
        project_root: Path,
        default_image: str = "aiperf:local",
        default_namespace: str | None = None,
    ) -> None:
        """Initialize benchmark deployer.

        Args:
            kubectl: Kubectl client.
            project_root: Path to project root.
            default_image: Default image to use for benchmarks.
            default_namespace: If set, overrides the ``--namespace`` passed to
                ``aiperf kube generate`` so each xdist worker gets an isolated
                namespace and tests can run in parallel.
        """
        self.kubectl = kubectl
        self.project_root = project_root
        self.default_image = default_image
        self.default_namespace = default_namespace
        self._deployments: list[BenchmarkResult] = []

    async def deploy(
        self,
        config: BenchmarkConfig,
        wait_for_completion: bool = True,
        timeout: int = 300,
        stream_logs: bool = False,
        pre_wait_hook: Any | None = None,
    ) -> BenchmarkResult:
        """Deploy a benchmark.

        Args:
            config: Benchmark configuration.
            wait_for_completion: Wait for benchmark to complete.
            timeout: Timeout in seconds.
            stream_logs: If True, stream pod logs in the background.
            pre_wait_hook: Optional async callable(namespace) invoked after
                the manifest is applied but before waiting for completion.
                Useful for Kueue tests that need to create a LocalQueue in the
                dynamically-created benchmark namespace.

        Returns:
            Benchmark result.
        """
        start_time = time.time()
        logger.info(
            f"[DEPLOY] Starting benchmark: concurrency={config.concurrency}, "
            f"requests={config.request_count}, image={config.image}"
        )

        # Write config to temp file
        config_path = config.to_temp_file()

        try:
            # Generate manifest using aiperf kube deploy --dry-run
            async with timed_operation("Generating Kubernetes manifest"):
                manifest = await self._generate_manifest(config, config_path)
                logger.debug(
                    lambda manifest=manifest: f"Generated manifest ({len(manifest)} bytes)"
                )

            # Patch imagePullPolicy for kind clusters (locally loaded images)
            manifest = self._patch_image_pull_policy(manifest, config.image)

            # Extract namespace from CR metadata and ensure a clean slate
            namespace = self._extract_namespace("", manifest)
            if namespace:
                await self._ensure_clean_namespace(namespace)

            # Apply the AIPerfJob CR
            async with timed_operation("Applying AIPerfJob CR to cluster"):
                output = await self.kubectl.apply(manifest)
                if not namespace:
                    namespace = self._extract_namespace(output, manifest)

            if not namespace:
                raise RuntimeError("Failed to extract namespace from deployment output")

            logger.info(f"[DEPLOY] Created namespace: {namespace}")

            # Wait for operator to create the JobSet from the AIPerfJob CR
            jobset_name = ""
            for _attempt in range(30):
                jobsets = await self.kubectl.get_jobsets(namespace)
                if jobsets:
                    jobset_name = jobsets[0].name
                    break
                await asyncio.sleep(2)

            if not jobset_name:
                logger.warning(
                    f"No JobSet found in {namespace} after 60s - "
                    "operator may not have reconciled yet"
                )

            # Extract job_id from JobSet name (format: aiperf-{job_id})
            job_id = jobset_name.removeprefix("aiperf-") if jobset_name else namespace

            result = BenchmarkResult(
                namespace=namespace,
                jobset_name=jobset_name,
                job_id=job_id,
                config=config,
            )

            logger.info(
                f"[DEPLOY] Benchmark deployed: namespace={namespace}, jobset={jobset_name}"
            )

            if pre_wait_hook is not None:
                await pre_wait_hook(namespace)

            if wait_for_completion:
                async with PodLogStreamer(
                    self.kubectl, namespace, prefix="BENCH"
                ) as streamer:
                    if stream_logs:
                        streamer.watch()
                    async with timed_operation(
                        f"Waiting for benchmark completion (timeout={timeout}s)"
                    ):
                        result = await self._wait_and_collect(result, timeout)

            result.duration_seconds = time.time() - start_time
            logger.info(
                f"[DEPLOY] Total deployment time: {result.duration_seconds:.2f}s"
            )
            if wait_for_completion:
                result.print_results()
            self._deployments.append(result)
            return result

        finally:
            # Clean up temp config file
            config_path.unlink(missing_ok=True)

    async def _ensure_clean_namespace(self, namespace: str) -> None:
        """Ensure the namespace exists and is clean for a new benchmark.

        Strips finalizers from and deletes stale AIPerfJobs/JobSets,
        then waits briefly for pods to terminate.
        """
        # Wait for terminating namespace
        for _ in range(60):
            result = await self.kubectl.run(
                "get",
                "namespace",
                namespace,
                "-o",
                "jsonpath={.status.phase}",
                check=False,
            )
            if result.returncode != 0:
                break
            if result.stdout.strip() == "Terminating":
                logger.info(
                    f"[DEPLOY] Waiting for namespace {namespace} to terminate..."
                )
                await asyncio.sleep(2)
            else:
                break

        await self.kubectl.create_namespace(namespace)

        # Strip finalizers and delete stale AIPerfJobs/JobSets
        await self._strip_aiperfjob_finalizers(namespace)
        await self.kubectl.run(
            "delete",
            "aiperfjobs,jobsets",
            "--all",
            "-n",
            namespace,
            check=False,
        )
        # Wait for stale pods to terminate
        await asyncio.sleep(3)

    async def _generate_manifest(
        self, config: BenchmarkConfig, config_path: Path
    ) -> str:
        """Generate Kubernetes manifest using aiperf CLI.

        Args:
            config: Benchmark configuration.
            config_path: Path to config file (not used, kept for signature compat).

        Returns:
            YAML manifest string.
        """
        import uuid

        unique_suffix = uuid.uuid4().hex[:8]
        cmd = [
            "uv",
            "run",
            "aiperf",
            "kube",
            "generate",
            "--name",
            f"bench-{unique_suffix}",
            "--model",
            config.model_name,
            "--url",
            config.endpoint_url,
            "--endpoint-type",
            config.endpoint_type,
            "--image",
            config.image,
            "--concurrency",
            str(config.concurrency),
            "--request-count",
            str(config.request_count),
            "--warmup-request-count",
            str(config.warmup_request_count),
            "--tokenizer",
            config.tokenizer_name,
            "--workers-max",
            str(config.workers),
            "--ui",
            "none",
            "--isl",
            str((config.input_sequence_min + config.input_sequence_max) // 2),
            "--osl",
            str((config.output_tokens_min + config.output_tokens_max) // 2),
            "--operator",
        ]

        if self.default_namespace is not None:
            cmd.extend(["--namespace", self.default_namespace])

        if config.concurrency_ramp_duration is not None:
            cmd.extend(
                ["--concurrency-ramp-duration", str(config.concurrency_ramp_duration)]
            )

        if config.queue_name is not None:
            cmd.extend(["--queue-name", config.queue_name])

        if config.priority_class is not None:
            cmd.extend(["--priority-class", config.priority_class])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.project_root),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            msg = f"Failed to generate manifest (exit {proc.returncode}): {stderr.decode()}"
            logger.error(msg)
            raise RuntimeError(msg)

        output = stdout.decode()
        # Strip any non-YAML prefix lines (e.g. warnings printed to stdout).
        # YAML manifests always start with "apiVersion:" or "---".
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("apiVersion:") or line.startswith("---"):
                return "\n".join(lines[i:])
        return output

    def _patch_image_pull_policy(self, manifest: str, image: str) -> str:
        """Patch AIPerfJob CR manifest to set imagePullPolicy: Never.

        Parses the YAML, sets spec.imagePullPolicy, and re-serializes.
        This is the native CR field - no string hacks needed.

        Args:
            manifest: YAML manifest string (AIPerfJob CR).
            image: Image name (unused, kept for API compat).

        Returns:
            Patched manifest.
        """
        cr = yaml.safe_load(manifest)
        if cr and cr.get("kind") == "AIPerfJob":
            cr.setdefault("spec", {})["imagePullPolicy"] = "Never"
        return yaml.dump(cr, default_flow_style=False, sort_keys=False)

    def _extract_namespace(
        self, apply_output: str, manifest: str | None = None
    ) -> str | None:
        """Extract namespace from kubectl apply output or CR manifest.

        For AIPerfJob CRs, the operator creates the namespace from the CR's
        metadata.namespace field. We extract it from the manifest directly.

        Args:
            apply_output: Output from kubectl apply.
            manifest: Original YAML manifest (for extracting CR namespace).

        Returns:
            Namespace name or None.
        """
        # Try extracting from manifest metadata
        if manifest:
            try:
                for doc in yaml.safe_load_all(manifest):
                    if not doc:
                        continue
                    kind = doc.get("kind", "")
                    ns = doc.get("metadata", {}).get("namespace")
                    if ns and kind in ("AIPerfJob", "JobSet"):
                        return ns
            except yaml.YAMLError:
                pass

        # Fallback: parse kubectl apply output for namespace/ lines
        for line in apply_output.splitlines():
            if line.startswith("namespace/"):
                parts = line.split()
                if parts:
                    return parts[0].replace("namespace/", "")
        return None

    async def _wait_and_collect(
        self,
        result: BenchmarkResult,
        timeout: int,
    ) -> BenchmarkResult:
        """Wait for benchmark completion and collect results.

        Two-strategy approach:
        1. Primary: port-forward to controller pod API (/api/results) for
           real-time progress and full results (same as ``aiperf kube watch``)
        2. Fallback: poll AIPerfJob CR status (operator stores results there
           before deleting the JobSet, so this always works)

        Args:
            result: Partial benchmark result.
            timeout: Timeout in seconds.

        Returns:
            Updated benchmark result with metrics.
        """
        from tests.kubernetes.helpers.operator import AIPerfJobStatus

        start_time = time.time()

        async with (
            BenchmarkWatchdog(
                await make_watchdog_source(self.kubectl),
                result.namespace,
                timeout=timeout,
                poll_interval=5.0,
                pending_threshold=30.0,
            ) as _watchdog,
            background_status(
                self.kubectl, result.namespace, label="BENCH", interval=15
            ),
        ):
            # Strategy 1: port-forward to controller pod API
            try:
                controller_pod = await self._wait_for_controller_pod(
                    result.namespace, timeout=min(120, timeout)
                )
                if controller_pod:
                    remaining = max(30, timeout - int(time.time() - start_time))
                    logger.info(
                        f"[COLLECT] Port-forwarding to {controller_pod.name} "
                        f"(remaining={remaining}s)"
                    )
                    api_results = await self.kubectl.wait_for_benchmark_api(
                        pod=controller_pod.name,
                        namespace=result.namespace,
                        timeout=remaining,
                    )
                    result.api_results = api_results
                    result.success = api_results.get("status") == "complete"
                    logger.info(
                        f"[COLLECT] API results: status={api_results.get('status')}"
                    )
            except (RuntimeError, TimeoutError, Exception) as e:
                logger.info(f"[COLLECT] Port-forward path failed: {e}")

            # Strategy 2: poll AIPerfJob CR (fallback, always reliable)
            if not result.api_results:
                cr_name = result.job_id
                logger.info(f"[COLLECT] Falling back to CR polling for {cr_name}")
                try:
                    while time.time() - start_time < timeout:
                        try:
                            data = await self.kubectl.get_json(
                                "aiperfjob", cr_name, namespace=result.namespace
                            )
                            cr_status = AIPerfJobStatus.from_json(data)
                        except RuntimeError:
                            await asyncio.sleep(5)
                            continue

                        if cr_status.is_completed or cr_status.is_failed:
                            cr_results = cr_status.results or cr_status.live_metrics
                            # When Completed, wait up to 30s for status.results
                            # to be populated by the operator; it's written
                            # after phase=Completed in the operator's handler.
                            if (
                                cr_status.is_completed
                                and not cr_results
                                and time.time() - start_time < timeout - 30
                            ):
                                results_deadline = time.time() + 30
                                while time.time() < results_deadline:
                                    await asyncio.sleep(3)
                                    try:
                                        data = await self.kubectl.get_json(
                                            "aiperfjob", cr_name, namespace=result.namespace
                                        )
                                        cr_status = AIPerfJobStatus.from_json(data)
                                    except RuntimeError:
                                        continue
                                    cr_results = cr_status.results or cr_status.live_metrics
                                    if cr_results:
                                        break
                            result.success = cr_status.is_completed
                            if cr_status.is_failed:
                                result.error_message = cr_status.error
                            result.api_results = {
                                "status": "complete"
                                if cr_status.is_completed
                                else "failed",
                                **({"results": cr_results} if cr_results else {}),
                            }
                            logger.info(
                                f"[COLLECT] CR: phase={cr_status.phase}, "
                                f"has_results={cr_results is not None}"
                            )
                            break

                        elapsed = int(time.time() - start_time)
                        workers = cr_status.workers or {}
                        logger.info(
                            f"[COLLECT] CR: phase={cr_status.phase}, "
                            f"workers={workers.get('ready', 0)}/{workers.get('total', 0)}, "
                            f"elapsed={elapsed}s"
                        )
                        await asyncio.sleep(5)
                    else:
                        result.success = False
                        result.error_message = f"Timeout after {timeout}s"
                except Exception as e:
                    result.success = False
                    result.error_message = str(e)
                    logger.error(f"[COLLECT] CR polling failed: {e}")

        # Try to get JobSet status (may be gone if operator deleted it)
        if result.jobset_name:
            try:
                result.status = await self.kubectl.get_jobset(
                    result.jobset_name, result.namespace
                )
            except RuntimeError:
                # JobSet deleted by operator - synthesize status from result
                if result.success:
                    result.status = JobSetStatus(
                        name=result.jobset_name,
                        namespace=result.namespace,
                        terminal_state="Completed",
                        completed=True,
                        restarts=0,
                    )

        # Collect final pods (may be empty if operator deleted JobSet)
        result.pods = await self.kubectl.get_pods(result.namespace)

        # Parse metrics
        if result.api_results:
            result.metrics = BenchmarkMetrics.from_api_results(result.api_results)
            if result.metrics and result.metrics.request_count is None:
                logger.warning(
                    f"[COLLECT] Metrics parsed but request_count=None. "
                    f"api_results keys: {list(result.api_results.keys())}, "
                    f"results type: {type(result.api_results.get('results'))}, "
                    f"inner sample: {str(result.api_results.get('results', {}))[:200]}"
                )

        # Try controller logs for metrics if not available
        controller_pod_status = result.controller_pod
        if controller_pod_status:
            logs = await self.kubectl.get_logs(
                controller_pod_status.name,
                container="control-plane",
                namespace=result.namespace,
            )
            if result.metrics is None or result.metrics.request_count is None:
                result.metrics = BenchmarkMetrics.from_logs(logs)
            elif result.metrics is not None:
                result.metrics.raw_logs = logs

        logger.info(
            f"[COLLECT] Final: success={result.success}, "
            f"has_results={result.api_results is not None}, "
            f"has_metrics={result.metrics is not None}"
        )
        return result

    async def _check_aiperfjob_cr_status(
        self,
        namespace: str,
        job_id: str,
    ) -> dict[str, Any] | None:
        """Check AIPerfJob CR status for completion and results.

        The operator updates the CR status before deleting the JobSet,
        so this is the most reliable source of truth.

        Returns:
            Dict with results if CR shows completion, None otherwise.
        """
        try:
            result = await self.kubectl.run(
                "get",
                "aiperfjob",
                "-n",
                namespace,
                "-o",
                "json",
                check=False,
            )
            if result.returncode != 0:
                return None
            import orjson

            data = orjson.loads(result.stdout)
            items = data.get("items", [data]) if "items" in data else [data]
            for item in items:
                status = item.get("status", {})
                phase = status.get("phase", "")
                if phase in ("Completed", "Failed"):
                    cr_results = status.get("results")
                    if cr_results:
                        return {"status": "complete", "results": cr_results}
                    return {"status": "complete"}
        except Exception as e:
            logger.debug(f"Failed to check AIPerfJob CR status: {e}")
        return None

    async def _wait_for_controller_pod(
        self,
        namespace: str,
        timeout: int = 120,
    ) -> PodStatus | None:
        """Wait for the controller pod to be running.

        Args:
            namespace: Kubernetes namespace.
            timeout: Timeout in seconds.

        Returns:
            Controller PodStatus or None if not found.
        """
        start_time = time.time()
        delay = 0.5
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            pods = await self.kubectl.get_pods(namespace)
            for pod in pods:
                if "controller" in pod.name and pod.phase == "Running":
                    logger.info(
                        f"Controller pod ready: {pod.name} (waited {elapsed:.0f}s)"
                    )
                    return pod

            logger.info(
                f"Waiting for controller pod ({int(elapsed)}s, {len(pods)} pods exist)..."
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 3.0)
        logger.error(f"Controller pod not found in {namespace} within {timeout}s")
        return None

    async def cleanup(
        self, result: BenchmarkResult, *, delete_namespace: bool = False
    ) -> None:
        """Clean up a benchmark deployment.

        Strips finalizers from AIPerfJobs and force-deletes pods. By default
        the namespace is left in place because xdist workers share their
        per-worker benchmark namespace across sequential tests; tearing it
        down forces the next test to wait for finalizers to clear. Pass
        ``delete_namespace=True`` for tests that explicitly verify namespace
        removal.

        Args:
            result: Benchmark result to clean up.
            delete_namespace: Also delete the benchmark namespace.
        """
        logger.info(f"Cleaning up benchmark in namespace: {result.namespace}")
        await self._strip_aiperfjob_finalizers(result.namespace)
        await self.kubectl.run(
            "delete",
            "aiperfjobs,jobsets",
            "--all",
            "-n",
            result.namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
        await self.kubectl.force_cleanup_namespace_pods(result.namespace)
        if delete_namespace:
            await self.kubectl.delete_namespace(result.namespace, wait=True)

    async def _strip_aiperfjob_finalizers(self, namespace: str) -> None:
        """Remove finalizers from all AIPerfJob CRs in a namespace.

        This prevents namespace deletion from blocking when the operator
        cannot process the delete event (e.g. during test teardown).
        """
        result = await self.kubectl.run(
            "get",
            "aiperfjobs",
            "-n",
            namespace,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        for name in result.stdout.strip().split():
            await self.kubectl.run(
                "patch",
                "aiperfjob",
                name,
                "-n",
                namespace,
                "--type=json",
                '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                check=False,
            )

    async def cleanup_all(self, timeout: int = 120) -> None:
        """Clean up all deployed benchmarks, deduplicating by namespace."""

        async def _safe_cleanup(result: BenchmarkResult) -> None:
            try:
                await self.cleanup(result)
            except Exception as e:
                logger.warning(f"Failed to cleanup {result.namespace}: {e}")

        if self._deployments:
            seen: set[str] = set()
            unique = []
            for r in self._deployments:
                if r.namespace not in seen:
                    seen.add(r.namespace)
                    unique.append(r)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[_safe_cleanup(r) for r in unique]),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup timed out after {timeout}s, continuing")

        self._deployments.clear()

    def get_deployment_count(self) -> int:
        """Get number of active deployments.

        Returns:
            Number of deployments.
        """
        return len(self._deployments)
