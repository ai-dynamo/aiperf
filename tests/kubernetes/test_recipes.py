# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests that deploy every recipe YAML via the operator against the mock server.

Each recipe is loaded, adapted for the test environment (mock server URL, reduced
concurrency, gpt2 tokenizer, no PVC volumes), and submitted as an AIPerfJob CR.
The test waits for the job to reach Completed phase.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from aiperf.kubernetes.validate import validate_file
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.operator import (
    AIPerfJobStatus,
    OperatorDeployer,
    OperatorJobResult,
)
from tests.kubernetes.helpers.watchdog import BenchmarkWatchdog, make_watchdog_source

RECIPES_DIR = Path(__file__).parents[2] / "recipes"
MOCK_SERVER_URL = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
MOCK_MODEL = "mock-model"
MOCK_TOKENIZER = "gpt2"

# Small load for test environment
TEST_CONCURRENCY = 2
TEST_REQUEST_COUNT = 10
TEST_WARMUP = 2


def _discover_recipes() -> list[Path]:
    """Find all perf.yaml recipe files in the recipes directory.

    Recipes ship as ``recipes/**/perf.yaml`` (each an ``AIPerfJob`` CRD doc).
    """
    if not RECIPES_DIR.exists():
        return []
    return sorted(RECIPES_DIR.rglob("perf.yaml"))


def _adapt_recipe_for_mock(doc: dict[str, Any], image: str) -> dict[str, Any]:
    """Adapt a recipe for the mock-server test environment.

    Rewrites endpoint, models, tokenizer, datasets, load, and strips PVC/secret
    references so the recipe can run against the in-cluster mock server.

    Expects the flat CRD spec format (no userConfig wrapper).
    """
    doc = copy.deepcopy(doc)
    spec = doc["spec"]
    bench = spec.setdefault("benchmark", {})

    # -- models --
    bench["models"] = [MOCK_MODEL]

    # -- endpoint: replace entirely with mock-compatible config --
    bench["endpoint"] = {"urls": [MOCK_SERVER_URL]}

    # -- tokenizer --
    bench["tokenizer"] = {"name": MOCK_TOKENIZER}

    # -- datasets: replace with small synthetic --
    bench["datasets"] = [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 100,
            "prompts": {
                "isl": {"mean": 32, "stddev": 0},
                "osl": {"mean": 16, "stddev": 0},
            },
        },
    ]

    # -- phases (small counts) - replace any existing warmup/profiling --
    bench.pop("warmup", None)
    bench.pop("profiling", None)
    bench["phases"] = [
        {
            "name": "warmup",
            "kind": "warmup",
            "type": "concurrency",
            "concurrency": TEST_CONCURRENCY,
            "requests": TEST_WARMUP,
        },
        {
            "name": "profiling",
            "kind": "profiling",
            "type": "concurrency",
            "concurrency": TEST_CONCURRENCY,
            "requests": TEST_REQUEST_COUNT,
        },
    ]

    # -- runtime: fit the single-node Kind test cluster --
    runtime = bench.setdefault("runtime", {})
    runtime["ui"] = "none"
    runtime["recordProcessors"] = 1

    # -- remove sections that won't work in test env --
    bench.pop("server_metrics", None)
    bench.pop("artifacts", None)
    bench.pop("slos", None)

    # -- image --
    spec["image"] = image
    spec["imagePullPolicy"] = "Never"

    # -- strip podTemplate of production-only fields --
    spec.pop("podTemplate", None)

    # -- short unique name (job_id max 35 chars) --
    original_name = doc["metadata"]["name"]
    short_hash = hashlib.sha1(original_name.encode()).hexdigest()[:6]
    doc["metadata"]["name"] = f"r-{short_hash}-{original_name}"[:35].rstrip("-")

    return doc


RECIPE_FILES = _discover_recipes()


def test_recipe_discovery_is_non_empty() -> None:
    """Tripwire: fail loudly if recipe discovery finds nothing.

    Requires no cluster fixtures so it runs during plain collection. If the
    recipe glob ever drifts from the shipped filename again (the suite once
    globbed ``aiperfjob.yaml`` after recipes were renamed to ``perf.yaml``),
    the parametrized deployment cases silently collapse to zero and nothing
    fails. This assertion turns that silent zero into a hard failure.
    """
    recipes = _discover_recipes()
    assert len(recipes) > 0, (
        f"No recipe files discovered under {RECIPES_DIR}; the parametrized "
        "recipe suite would silently collect zero cases. Check the glob in "
        "_discover_recipes() against the shipped recipe filename."
    )


def test_adapt_recipe_for_mock_scales_record_processors() -> None:
    """Production processor counts must not overcommit the Kind test node."""
    recipe = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "large-production-recipe"},
        "spec": {"benchmark": {"runtime": {"recordProcessors": 32}}},
    }

    adapted = _adapt_recipe_for_mock(recipe, image="aiperf:local")

    assert adapted["spec"]["benchmark"]["runtime"]["recordProcessors"] == 1


@pytest.mark.parametrize(
    "recipe_path",
    RECIPE_FILES,
    ids=[str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES],
)
def test_recipe_validates_strictly(recipe_path: Path) -> None:
    """Every checked-in recipe must satisfy the current kind-specific schema."""
    result = validate_file(recipe_path, strict=True)
    assert result.passed, f"{recipe_path}: {result.errors}"


@pytest.mark.parametrize(
    "recipe_path",
    RECIPE_FILES,
    ids=[str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES],
)
class TestRecipeDeployment:
    """Deploy each recipe via the operator and verify it completes."""

    @pytest.mark.timeout(600)
    @pytest.mark.asyncio
    async def test_recipe_completes(
        self,
        recipe_path: Path,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
        k8s_settings: Any,
        operator_job_namespace: str,
    ) -> None:
        """Load recipe, adapt for mock server, deploy, and wait for Completed."""
        doc = yaml.safe_load(recipe_path.read_text())
        adapted = _adapt_recipe_for_mock(doc, image=k8s_settings.aiperf_image)

        name = adapted["metadata"]["name"]
        namespace = operator_job_namespace
        adapted.setdefault("metadata", {})["namespace"] = namespace
        manifest = yaml.dump(adapted, default_flow_style=False)

        await kubectl.apply(manifest)
        operator_ready._deployed_jobs.append(
            OperatorJobResult(
                namespace=namespace,
                job_name=name,
                config=None,  # type: ignore[arg-type]
            )
        )

        timeout = k8s_settings.benchmark_timeout
        recipe_rel = recipe_path.relative_to(RECIPES_DIR)

        async with (
            make_watchdog_source(kubectl) as watchdog_source,
            BenchmarkWatchdog(
                watchdog_source,
                namespace,
                timeout=timeout,
                poll_interval=5.0,
                pending_threshold=30.0,
            ) as _watchdog,
            PodLogStreamer(kubectl, namespace, prefix=str(recipe_rel)) as streamer,
            background_status(kubectl, namespace, label="RECIPE", interval=15),
        ):
            if k8s_settings.stream_logs:
                streamer.watch()

            try:
                status: AIPerfJobStatus = await operator_ready.wait_for_job_completion(
                    name,
                    namespace,
                    timeout=timeout,
                )
            except TimeoutError:
                status = await operator_ready.get_job_status(name, namespace)
                logs = await operator_ready.get_operator_logs(tail=50)
                pytest.fail(
                    f"Recipe {recipe_rel} timed out.\n"
                    f"Phase: {status.phase}, CurrentPhase: {status.current_phase}\n"
                    f"Error: {status.error}\n"
                    f"Operator logs (last 50):\n{logs}"
                )

        assert status.is_completed, (
            f"Recipe {recipe_rel} ended with phase={status.phase}, error={status.error}"
        )
        assert status.is_condition_true("Complete"), (
            f"Recipe {recipe_rel} completed without Complete=True: {status.conditions}"
        )
        assert status.is_condition_true("ResultsAvailable"), (
            f"Recipe {recipe_rel} completed without durable results: {status.conditions}"
        )
        assert status.results is not None, (
            f"Recipe {recipe_rel} completed without status.results"
        )

        metrics = status.results.get("metrics", status.results)
        request_count = metrics.get("request_count", {})
        request_count_avg = (
            request_count.get("avg")
            if isinstance(request_count, dict)
            else request_count
        )
        assert request_count_avg == TEST_REQUEST_COUNT, (
            f"Recipe {recipe_rel} reported request_count={request_count_avg}; "
            f"expected {TEST_REQUEST_COUNT}"
        )

        throughput = metrics.get("request_throughput", {})
        throughput_avg = (
            throughput.get("avg") if isinstance(throughput, dict) else throughput
        )
        assert throughput_avg is not None and throughput_avg > 0, (
            f"Recipe {recipe_rel} reported request_throughput={throughput_avg}"
        )
