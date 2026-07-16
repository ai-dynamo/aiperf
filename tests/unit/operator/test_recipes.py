# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate all AIPerfJob recipe YAML files against the nested CRD schema and AIPerfConfig model.

Recipes live as ``recipes/**/perf.yaml`` and may use Jinja2 ``{{ ... }}`` templates
and ``${ENV_VAR}`` substitutions; ``AIPerfJobSpecConverter.to_aiperf_config`` renders
those before validating against ``AIPerfConfig``.
"""

from pathlib import Path

import pytest
import yaml

from aiperf.config import AIPerfConfig
from aiperf.kubernetes.validate import KNOWN_SPEC_FIELDS
from aiperf.operator.spec_converter import AIPerfJobSpecConverter

RECIPES_DIR = Path(__file__).parents[3] / "recipes"


def _discover_recipes() -> list[Path]:
    """Find all perf.yaml files in the recipes directory."""
    if not RECIPES_DIR.exists():
        return []
    return sorted(RECIPES_DIR.rglob("perf.yaml"))


RECIPE_FILES = _discover_recipes()


@pytest.mark.parametrize(
    "recipe_path",
    RECIPE_FILES,
    ids=[str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES],
)
class TestRecipeValidation:
    """Validate each recipe YAML against the nested AIPerfJob CRD schema."""

    def test_yaml_structure(self, recipe_path: Path) -> None:
        """Verify required YAML structure: apiVersion, kind, metadata.name, spec.benchmark."""
        doc = yaml.safe_load(recipe_path.read_text())
        assert doc["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert doc["kind"] == "AIPerfJob"
        assert "name" in doc["metadata"]
        spec = doc["spec"]
        # Nested format: models/endpoint must be under spec.benchmark
        benchmark = spec.get("benchmark", {})
        assert "models" in benchmark or "endpoint" in benchmark

    def test_config_validates(self, recipe_path: Path) -> None:
        """Verify nested spec produces a valid AIPerfConfig via to_aiperf_config()."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        config = converter.to_aiperf_config()

        assert isinstance(config, AIPerfConfig)
        assert config.benchmark.get_model_names()
        assert config.benchmark.endpoint.urls
        for url in config.benchmark.endpoint.urls:
            assert url.startswith("http://") or url.startswith("https://")

    def test_deployment_config(self, recipe_path: Path) -> None:
        """Verify deployment fields convert to valid DeploymentConfig."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        deploy = converter.to_deployment_config()
        assert deploy is not None

    def test_worker_calculation(self, recipe_path: Path) -> None:
        """Verify worker count calculation produces >= 1."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        workers = converter.calculate_workers()
        assert workers >= 1

    def test_metadata_name_is_valid_k8s_name(self, recipe_path: Path) -> None:
        """Verify metadata.name is a valid Kubernetes resource name."""
        import re

        doc = yaml.safe_load(recipe_path.read_text())
        name = doc["metadata"]["name"]

        assert len(name) <= 253
        assert re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$", name), (
            f"Invalid K8s name: {name}"
        )

    def test_no_unknown_top_level_spec_fields(self, recipe_path: Path) -> None:
        """Verify spec only contains known CRD fields."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        unknown = set(spec.keys()) - KNOWN_SPEC_FIELDS
        assert not unknown, f"Unknown spec fields: {unknown}"


class TestRecipeCompleteness:
    """Verify all expected recipes exist."""

    def test_recipe_count(self) -> None:
        assert len(RECIPE_FILES) == 27, (
            f"Expected 27 recipes, found {len(RECIPE_FILES)}: "
            + ", ".join(str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES)
        )

    def test_expected_recipes_exist(self) -> None:
        expected = [
            "deepseek-r1/trtllm/disagg/wide_ep/gb200/perf.yaml",
            "deepseek-v32-fp4/trtllm/agg-round-robin/perf.yaml",
            "deepseek-v32-fp4/trtllm/disagg-kv-router/perf.yaml",
            "deepseek-v4-flash/sglang/perf.yaml",
            "deepseek-v4-flash/vllm/agg/perf.yaml",
            "deepseek-v4-pro/sglang/perf.yaml",
            "deepseek-v4-pro/vllm/agg/perf.yaml",
            "glm-5-nvfp4/sglang/disagg/perf.yaml",
            "gpt-oss-120b/trtllm/agg/perf.yaml",
            "gpt-oss-120b/trtllm/disagg/perf.yaml",
            "kimi-k2.5/trtllm/agg/baseten/perf.yaml",
            "kimi-k2.5/trtllm/agg/nvidia/perf.yaml",
            "llama-3-70b/vllm/agg/perf.yaml",
            "llama-3-70b/vllm/disagg-multi-node/perf.yaml",
            "llama-3-70b/vllm/disagg-single-node/perf.yaml",
            "nemotron-3-super-fp8/sglang/agg/perf.yaml",
            "nemotron-3-super-fp8/sglang/disagg/perf.yaml",
            "nemotron-3-super-fp8/trtllm/disagg/perf.yaml",
            "nemotron-3-super-fp8/vllm/agg/perf.yaml",
            "qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml",
            "qwen3-235b-a22b-fp8/trtllm/disagg/perf.yaml",
            "qwen3-32b-fp8/trtllm/agg/perf.yaml",
            "qwen3-32b-fp8/trtllm/disagg/perf.yaml",
            "qwen3-32b-fp8/vllm/disagg/perf.yaml",
            "qwen3-32b/vllm/agg-round-robin/perf.yaml",
            "qwen3-32b/vllm/disagg-kv-router/perf.yaml",
            "qwen3-vl-30b/vllm/agg-embedding-cache/perf.yaml",
        ]
        for path_str in expected:
            full_path = RECIPES_DIR / path_str
            assert full_path.exists(), f"Missing recipe: {path_str}"
