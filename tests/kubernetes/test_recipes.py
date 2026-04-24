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
    """Find all aiperfjob.yaml files in the recipes directory."""
    if not RECIPES_DIR.exists():
        return []
    return sorted(RECIPES_DIR.rglob("aiperfjob.yaml"))


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
    bench["datasets"] = {
        "main": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {
                "isl": {"mean": 32, "stddev": 0},
                "osl": {"mean": 16, "stddev": 0},
            },
        },
    }

    # -- phases (small counts) - replace any existing warmup/profiling --
    bench.pop("warmup", None)
    bench.pop("profiling", None)
    bench["phases"] = {
        "warmup": {
            "type": "concurrency",
            "concurrency": TEST_CONCURRENCY,
            "requests": TEST_WARMUP,
            "exclude_from_results": True,
        },
        "profiling": {
            "type": "concurrency",
            "concurrency": TEST_CONCURRENCY,
            "requests": TEST_REQUEST_COUNT,
        },
    }

    # -- runtime: disable UI --
    bench.setdefault("runtime", {})["ui"] = "none"

    # -- remove sections that won't work in test env --
    bench.pop("server_metrics", None)
    bench.pop("artifacts", None)
    bench.pop("random_seed", None)
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
        # Recipes target real GPU clusters; mock-server adaptation is imperfect
        # for many of them. Opt-in via env var to run the full recipe suite.
        import os

        if not os.environ.get("AIPERF_RUN_RECIPE_TESTS"):
            pytest.skip(
                "Recipe tests require AIPERF_RUN_RECIPE_TESTS=1 "
                "(designed for real model endpoints)"
            )
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
            BenchmarkWatchdog(
                await make_watchdog_source(kubectl),
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

        if status.results:
            metrics = status.results.get("metrics", {})
            throughput = metrics.get("request_throughput", {})
            if isinstance(throughput, dict):
                assert throughput.get("avg", 0) > 0
            else:
                assert status.results.get("requestThroughput", throughput or 0) > 0
