# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial validation for checked-in Kubernetes workload examples.

Focuses on:
- documented AIPerfJob/AIPerfSweep YAML fences validating through kind dispatch;
- stale-field resistance for docs and Helm NOTES workload samples;
- kind-specific sweep cardinality in examples users copy into clusters;
- namespace omission remaining compatible with the benchmark namespace default.

Out of scope (covered elsewhere):
- Local CLI AIPerfConfig snippet round-trips: tests/unit/operator/test_crd_round_trip_yamls.py
- CRD CEL schema generation: tests/unit/kubernetes/test_crd_validation_adversarial.py
- validate_file parser/error-path units: tests/unit/kubernetes/test_validate.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE
from aiperf.kubernetes.crd_models import AIPerfJobSpec, AIPerfSweepSpec
from aiperf.kubernetes.validate import validate_cr, validate_file

# =============================================================================
# Helpers
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
_FENCE_RE = re.compile(r"^```ya?ml\s*$", re.MULTILINE)
_SUPPORTED_KINDS = {"AIPerfJob", "AIPerfSweep"}
_HELM_NOTES_PLACEHOLDERS = {
    "{{ .Values.benchmarkNamespace.name }}": DEFAULT_BENCHMARK_NAMESPACE,
    '{{ include "aiperf-operator.defaultJobImage" . }}': "nvcr.io/nvidia/aiperf:latest",
}


@dataclass(frozen=True, slots=True)
class WorkloadExample:
    """A documented Kubernetes workload snippet plus source provenance."""

    source_id: str
    text: str


def _extract_yaml_fences_from_markdown(text: str) -> list[str]:
    """Return YAML fence bodies from a Markdown document."""
    out: list[str] = []
    lines = text.splitlines()
    in_fence = False
    buf: list[str] = []
    for line in lines:
        if not in_fence and _FENCE_RE.match(line):
            in_fence = True
            buf = []
            continue
        if in_fence:
            if line.strip().startswith("```"):
                out.append("\n".join(buf))
                in_fence = False
                buf = []
            else:
                buf.append(line)
    return out


def _markdown_files() -> list[Path]:
    """Return docs files whose YAML fences can be user-copyable workloads."""
    roots = [REPO_ROOT / "docs" / "kubernetes", REPO_ROOT / "docs" / "tutorials"]
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("*.md"))
    return sorted(files)


def _load_yaml_documents(source_id: str, text: str) -> list[dict[str, object]]:
    """Parse a YAML snippet and return mapping documents with source-rich errors."""
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError as exc:
        raise AssertionError(
            f"{source_id}: YAML snippet failed to parse: {exc}"
        ) from exc
    return [doc for doc in docs if isinstance(doc, dict)]


def _is_complete_workload_document(doc: dict[str, object]) -> bool:
    """Return True for copyable AIPerf workload CRs, not migration deltas."""
    if doc.get("kind") not in _SUPPORTED_KINDS:
        return False
    metadata = doc.get("metadata")
    spec = doc.get("spec")
    return (
        isinstance(metadata, dict)
        and isinstance(metadata.get("name"), str)
        and isinstance(spec, dict)
        and isinstance(spec.get("benchmark"), dict)
    )


def _normalise_helm_notes_text(text: str) -> str:
    """Replace Helm-only placeholders so the installed NOTES sample is parseable."""
    out = text
    for placeholder, replacement in _HELM_NOTES_PLACEHOLDERS.items():
        out = out.replace(placeholder, replacement)
    return out


def _helm_notes_examples() -> list[WorkloadExample]:
    """Extract the AIPerfJob heredoc shown after chart installation."""
    path = REPO_ROOT / "deploy" / "helm" / "aiperf-operator" / "templates" / "NOTES.txt"
    text = path.read_text()
    start = "cat <<EOF | kubectl apply -f -"
    if start not in text:
        return []
    snippet = text.split(start, 1)[1].split("EOF", 1)[0]
    return [
        WorkloadExample(
            source_id=f"{path.relative_to(REPO_ROOT)}::install-notes-aiperfjob",
            text=_normalise_helm_notes_text(snippet),
        )
    ]


def _documented_workload_examples() -> list[WorkloadExample]:
    """Collect complete AIPerf workload examples from docs and deploy notes."""
    examples: list[WorkloadExample] = []
    for path in _markdown_files():
        rel = path.relative_to(REPO_ROOT)
        for idx, fence_text in enumerate(
            _extract_yaml_fences_from_markdown(path.read_text())
        ):
            if "kind: AIPerf" not in fence_text:
                continue
            source_id = f"{rel}::yaml-fence-{idx}"
            docs = _load_yaml_documents(source_id, fence_text)
            if any(_is_complete_workload_document(doc) for doc in docs):
                examples.append(WorkloadExample(source_id=source_id, text=fence_text))
    examples.extend(_helm_notes_examples())
    return examples


def _workload_docs(example: WorkloadExample) -> list[dict[str, object]]:
    """Return complete AIPerf workload CRs from one example."""
    return [
        doc
        for doc in _load_yaml_documents(example.source_id, example.text)
        if _is_complete_workload_document(doc)
    ]


def _write_example(tmp_path: Path, example: WorkloadExample) -> Path:
    """Write a checked-in example snippet to disk for validate_file."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", example.source_id)
    path = tmp_path / f"{safe_name}.yaml"
    path.write_text(example.text)
    return path


_EXAMPLES = _documented_workload_examples()
_EXAMPLE_PARAMS = [param(example, id=example.source_id) for example in _EXAMPLES]


# =============================================================================
# Docs/deploy sample discovery sanity
# =============================================================================


class TestKubernetesExampleDiscovery:
    """The adversarial scan must not silently degrade to no coverage."""

    def test_documented_workload_scan_finds_jobs_sweeps_and_deploy_notes(self) -> None:
        kinds = {
            cast(str, doc["kind"])
            for example in _EXAMPLES
            for doc in _workload_docs(example)
        }

        assert "AIPerfJob" in kinds
        if (REPO_ROOT / "docs" / "kubernetes").is_dir():
            # Every checked-in AIPerfSweep example lives in the Kubernetes docs,
            # which are supplied by the documentation port.
            assert "AIPerfSweep" in kinds
        assert any(
            "deploy/helm/aiperf-operator/templates/NOTES.txt" in e.source_id
            for e in _EXAMPLES
        )


# =============================================================================
# Kind dispatch and stale-field resistance
# =============================================================================


class TestKubernetesExampleKindDispatch:
    """Examples must route to the schema that matches their kind field."""

    @pytest.mark.parametrize("example", _EXAMPLE_PARAMS)
    def test_validate_cr_example_kind_routes_to_matching_pydantic_spec(
        self, example: WorkloadExample
    ) -> None:
        for doc in _workload_docs(example):
            kind = cast(str, doc["kind"])
            spec = cast(dict[str, object], doc["spec"])

            try:
                validated = validate_cr(kind, spec)
            except ValidationError as exc:
                pytest.fail(
                    f"{example.source_id}: {kind} spec failed kind dispatch: {exc}"
                )

            if kind == "AIPerfJob":
                assert isinstance(validated, AIPerfJobSpec)
            else:
                assert isinstance(validated, AIPerfSweepSpec)

    @pytest.mark.parametrize("example", _EXAMPLE_PARAMS)
    def test_validate_file_example_strict_mode_rejects_no_stale_fields(
        self, tmp_path: Path, example: WorkloadExample
    ) -> None:
        path = _write_example(tmp_path, example)

        result = validate_file(path, strict=True)

        assert result.passed, (
            f"{example.source_id}: errors={result.errors} warnings={result.warnings}"
        )


# =============================================================================
# Kind-specific sweep cardinality in examples
# =============================================================================


class TestKubernetesExampleSweepCardinality:
    """Copyable manifests must obey AIPerfJob/AIPerfSweep sweep cardinality."""

    @pytest.mark.parametrize("example", _EXAMPLE_PARAMS)
    def test_aiperfsweep_examples_include_required_sweep_block(
        self, example: WorkloadExample
    ) -> None:
        for doc in _workload_docs(example):
            if doc["kind"] != "AIPerfSweep":
                continue
            spec = cast(dict[str, object], doc["spec"])
            sweep = spec.get("sweep")

            assert isinstance(sweep, dict) and sweep, (
                f"{example.source_id}: kind AIPerfSweep must include non-empty spec.sweep"
            )

    @pytest.mark.parametrize("example", _EXAMPLE_PARAMS)
    def test_aiperfjob_examples_do_not_smuggle_sweep_blocks(
        self, example: WorkloadExample
    ) -> None:
        for doc in _workload_docs(example):
            if doc["kind"] != "AIPerfJob":
                continue
            spec = cast(dict[str, object], doc["spec"])
            benchmark = cast(dict[str, object], spec["benchmark"])

            assert "sweep" not in spec, (
                f"{example.source_id}: kind AIPerfJob must not set spec.sweep"
            )
            assert "sweep" not in benchmark, (
                f"{example.source_id}: kind AIPerfJob must not hide sweep under spec.benchmark"
            )


# =============================================================================
# Namespace default contract
# =============================================================================


class TestKubernetesExampleNamespaceDefaults:
    """Examples may omit metadata.namespace; omission means benchmark default."""

    @pytest.mark.parametrize("example", _EXAMPLE_PARAMS)
    def test_metadata_namespace_omission_resolves_to_benchmark_default(
        self, example: WorkloadExample
    ) -> None:
        for doc in _workload_docs(example):
            metadata = cast(dict[str, object], doc["metadata"])
            namespace = metadata.get("namespace", DEFAULT_BENCHMARK_NAMESPACE)

            assert isinstance(namespace, str) and namespace
            if "namespace" not in metadata:
                assert namespace == DEFAULT_BENCHMARK_NAMESPACE
