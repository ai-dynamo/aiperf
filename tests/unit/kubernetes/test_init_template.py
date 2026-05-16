# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.init_template.wrap_as_aiperf_job.

Focuses on:
- Filename + job_name substitution in the AIPerfJob shell
- AIPerfJob CR structure: apiVersion/kind/metadata/spec.benchmark
- Body indentation under spec.benchmark
- Stripping of yaml-language-server and `# @template` metadata headers
- Deployment-options / pod / scheduling commented blocks preserved
"""

from __future__ import annotations

import pytest
import ruamel.yaml
from pytest import param

from aiperf.kubernetes.init_template import wrap_as_aiperf_job

MINIMAL_BODY = """\
model: meta-llama/Llama-3.1-8B-Instruct

endpoint:
  url: http://localhost:8000

dataset:
  type: synthetic
  entries: 100
"""


class TestWrapAsAIPerfJob:
    """Verify the AIPerfJob wrapper structure and substitutions."""

    def test_wrap_substitutes_filename_into_usage_comments(self) -> None:
        result = wrap_as_aiperf_job(MINIMAL_BODY, filename="my-config.yaml")
        assert "kubectl apply -f my-config.yaml" in result
        assert "aiperf kube profile --config my-config.yaml" in result

    def test_wrap_substitutes_job_name_into_metadata(self) -> None:
        result = wrap_as_aiperf_job(MINIMAL_BODY, job_name="my-run-42")
        assert "name: my-run-42" in result

    @pytest.mark.parametrize(
        "filename",
        [
            param("config with spaces.yaml", id="spaces-in-name"),
            param("path/to/config.yaml", id="path-separators"),
            param("config_v2.0.yaml", id="dots-in-name"),
            param("a", id="single-char"),
        ],
    )  # fmt: skip
    def test_wrap_preserves_special_filenames(self, filename: str) -> None:
        result = wrap_as_aiperf_job(MINIMAL_BODY, filename=filename)
        assert filename in result


class TestWrappedStructure:
    """Verify the rendered CR has the required AIPerfJob sections."""

    @pytest.fixture
    def rendered(self) -> str:
        return wrap_as_aiperf_job(MINIMAL_BODY, filename="config.yaml")

    def test_contains_api_version(self, rendered: str) -> None:
        assert "apiVersion: aiperf.nvidia.com/v1alpha1" in rendered

    def test_contains_kind(self, rendered: str) -> None:
        assert "kind: AIPerfJob" in rendered

    def test_contains_spec_benchmark(self, rendered: str) -> None:
        assert "spec:" in rendered
        assert "  benchmark:" in rendered

    def test_body_indented_under_benchmark(self, rendered: str) -> None:
        assert "    model: meta-llama/Llama-3.1-8B-Instruct" in rendered
        assert "    endpoint:" in rendered
        assert "      url: http://localhost:8000" in rendered

    def test_contains_pod_template_commented(self, rendered: str) -> None:
        assert "# podTemplate:" in rendered

    def test_contains_scheduling_commented(self, rendered: str) -> None:
        assert "# scheduling:" in rendered

    def test_parses_as_valid_yaml(self, rendered: str) -> None:
        """Uncommented lines should parse as a valid AIPerfJob CR."""
        yaml_lines = [
            line
            for line in rendered.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        parsed = ruamel.yaml.YAML().load("\n".join(yaml_lines))
        assert parsed["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert parsed["kind"] == "AIPerfJob"
        assert "benchmark" in parsed["spec"]
        assert parsed["spec"]["benchmark"]["endpoint"]["url"] == "http://localhost:8000"


class TestHeaderStripping:
    """`# yaml-language-server` and `# @template` blocks must not leak into output."""

    def test_strips_yaml_language_server_header(self) -> None:
        body = (
            "# yaml-language-server: $schema=../schema/aiperf-config.schema.json\n"
            "model: x\n"
        )
        result = wrap_as_aiperf_job(body)
        assert "yaml-language-server" not in result
        assert "    model: x" in result

    def test_strips_template_metadata_block(self) -> None:
        body = (
            "# @template\n"
            "# title: Example\n"
            "# description: desc\n"
            "# category: Getting Started\n"
            "\n"
            "model: x\n"
        )
        result = wrap_as_aiperf_job(body)
        assert "@template" not in result
        assert "# title: Example" not in result
        assert "    model: x" in result

    def test_keeps_body_comments_after_metadata(self) -> None:
        """Non-metadata comments inside the body must survive the wrap."""
        body = "model: x\n# this is a real comment\nphases:\n  type: concurrency\n"
        result = wrap_as_aiperf_job(body)
        assert "    # this is a real comment" in result
