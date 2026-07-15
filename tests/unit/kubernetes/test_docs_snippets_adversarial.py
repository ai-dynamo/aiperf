# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial validation for Kubernetes documentation AIPerf YAML snippets.

Focuses on:
- fenced ``yaml`` snippets under ``docs/kubernetes/*.md`` that describe
  ``AIPerfJob`` or ``AIPerfSweep`` resources rather than generated deploy
  manifests;
- local ``aiperf kube validate`` parity, including current kind dispatch to
  ``AIPerfJobSpec`` versus ``AIPerfSweepSpec``;
- docs drift where snippets keep stale field shapes such as
  ``models.items`` or singular adaptive-search ``objective``;
- examples remaining runnable by declaring at least one dataset and phase.

Out of scope: Helm values examples and non-AIPerf Kubernetes resources; those
belong with the Helm/render tests in this directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import yaml
from pytest import param

from aiperf.kubernetes.validate import KIND_AIPERFJOB, KIND_AIPERFSWEEP, validate_file
from aiperf.operator.models import AIPerfJobSpec, AIPerfSweepSpec

# ============================================================================
# Helpers
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_KUBERNETES = _REPO_ROOT / "docs" / "kubernetes"
_YAML_FENCE_RE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_AIPERF_KINDS = {KIND_AIPERFJOB, KIND_AIPERFSWEEP}


@dataclass(frozen=True, slots=True)
class DocsYamlSnippet:
    """Parsed YAML fence with source location for documentation failures."""

    path: Path
    line: int
    fence_index: int
    body: str
    doc: dict[str, object]

    @property
    def source_id(self) -> str:
        """Return a stable pytest id that points at the docs snippet."""
        rel = self.path.relative_to(_REPO_ROOT)
        return f"{rel}:{self.line}"


_KIND_TO_SPEC = {
    KIND_AIPERFJOB: AIPerfJobSpec,
    KIND_AIPERFSWEEP: AIPerfSweepSpec,
}


def _mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def _sequence(value: object) -> list[object] | None:
    if isinstance(value, list):
        return cast("list[object]", value)
    return None


def _load_yaml_mappings(body: str, *, source_id: str) -> list[dict[str, object]]:
    try:
        loaded_docs = list(yaml.safe_load_all(body))
    except yaml.YAMLError as exc:  # pragma: no cover - failure message path
        pytest.fail(f"{source_id}: YAML snippet is not parseable: {exc}")
    return [
        mapping for loaded in loaded_docs if (mapping := _mapping(loaded)) is not None
    ]


def _yaml_snippets() -> list[DocsYamlSnippet]:
    snippets: list[DocsYamlSnippet] = []
    for path in sorted(_DOCS_KUBERNETES.glob("*.md")):
        text = path.read_text()
        for fence_index, match in enumerate(_YAML_FENCE_RE.finditer(text), start=1):
            info = match.group("info").strip().lower()
            if "yaml" not in info and "yml" not in info:
                continue
            line = text[: match.start()].count("\n") + 1
            body = match.group("body")
            source_id = f"{path.relative_to(_REPO_ROOT)}:{line}"
            for doc_index, doc in enumerate(
                _load_yaml_mappings(body, source_id=source_id), start=1
            ):
                snippets.append(
                    DocsYamlSnippet(
                        path=path,
                        line=line,
                        fence_index=fence_index * 1000 + doc_index,
                        body=body,
                        doc=doc,
                    )
                )
    return snippets


def _aiperf_cr_snippets() -> list[DocsYamlSnippet]:
    return [
        snippet
        for snippet in _yaml_snippets()
        if snippet.doc.get("kind") in _AIPERF_KINDS
    ]


def _spec_mapping(snippet: DocsYamlSnippet) -> dict[str, object] | None:
    return _mapping(snippet.doc.get("spec"))


def _benchmark_mapping(snippet: DocsYamlSnippet) -> dict[str, object] | None:
    spec = _spec_mapping(snippet)
    if spec is None:
        return None
    return _mapping(spec.get("benchmark"))


def _aiperf_benchmark_snippets() -> list[DocsYamlSnippet]:
    return [
        snippet
        for snippet in _yaml_snippets()
        if _benchmark_mapping(snippet) is not None
    ]


def _snippet_file(tmp_path: Path, snippet: DocsYamlSnippet) -> Path:
    safe_stem = re.sub(r"[^a-z0-9-]", "-", snippet.path.stem.lower())
    path = tmp_path / f"{safe_stem}-{snippet.line}.yaml"
    path.write_text(snippet.body)
    return path


_AIPERF_CR_SNIPPETS = _aiperf_cr_snippets()
_AIPERF_BENCHMARK_SNIPPETS = _aiperf_benchmark_snippets()


# ============================================================================
# Discovery sanity
# ============================================================================


def test_docs_kubernetes_aiperf_cr_snippet_discovery_finds_examples() -> None:
    assert _AIPERF_CR_SNIPPETS, (
        "expected at least one AIPerfJob/AIPerfSweep docs YAML snippet"
    )


# ============================================================================
# Current kind dispatch and full validation
# ============================================================================


@pytest.mark.parametrize(
    "snippet",
    [param(snippet, id=snippet.source_id) for snippet in _AIPERF_CR_SNIPPETS],
)  # fmt: skip
def test_docs_kubernetes_aiperf_cr_snippet_validate_file_dispatches_by_kind(
    snippet: DocsYamlSnippet, tmp_path: Path
) -> None:
    path = _snippet_file(tmp_path, snippet)

    result = validate_file(path, strict=True)

    assert result.passed, (
        f"{snippet.source_id}: errors={result.errors}; warnings={result.warnings}"
    )


@pytest.mark.parametrize(
    "snippet",
    [param(snippet, id=snippet.source_id) for snippet in _AIPERF_CR_SNIPPETS],
)  # fmt: skip
def test_docs_kubernetes_aiperf_cr_snippet_kind_selects_expected_pydantic_spec(
    snippet: DocsYamlSnippet,
) -> None:
    kind = cast("str", snippet.doc["kind"])
    spec = _spec_mapping(snippet)
    assert spec is not None, f"{snippet.source_id}: spec must be a mapping"

    validated = _KIND_TO_SPEC[kind].model_validate(spec)

    assert isinstance(validated, _KIND_TO_SPEC[kind])


# ============================================================================
# Required runnable workload shape
# ============================================================================


@pytest.mark.parametrize(
    "snippet",
    [param(snippet, id=snippet.source_id) for snippet in _AIPERF_CR_SNIPPETS],
)  # fmt: skip
def test_docs_kubernetes_aiperf_cr_snippet_declares_required_datasets_and_phases(
    snippet: DocsYamlSnippet,
) -> None:
    benchmark = _benchmark_mapping(snippet)
    assert benchmark is not None, (
        f"{snippet.source_id}: spec.benchmark must be a mapping"
    )

    datasets = _sequence(benchmark.get("datasets"))
    phases = _sequence(benchmark.get("phases"))

    # First-class shorthands (config/loader/normalizers.py) count as declared:
    # a singular `dataset:` mapping expands to datasets=[{name: "default", ...}]
    # and a flat `phases:` mapping expands to phases=[{name: "profiling", ...}].
    # Asserted on the raw YAML because validation no longer mutates the
    # snippet's dict in place (normalizers operate on copies).
    if not datasets and _mapping(benchmark.get("dataset")):
        datasets = [{"name": "default", **_mapping(benchmark.get("dataset"))}]
    if not phases and _mapping(benchmark.get("phases")):
        phases = [{"name": "profiling", **_mapping(benchmark.get("phases"))}]

    assert datasets, (
        f"{snippet.source_id}: spec.benchmark.datasets must be a non-empty list"
    )
    assert phases, (
        f"{snippet.source_id}: spec.benchmark.phases must be a non-empty list"
    )
    assert all(
        _mapping(dataset).get("name") for dataset in datasets if _mapping(dataset)
    ), f"{snippet.source_id}: every dataset entry needs a name"
    assert all(_mapping(phase).get("name") for phase in phases if _mapping(phase)), (
        f"{snippet.source_id}: every phase entry needs a name"
    )


# ============================================================================
# Stale field-shape regression locks
# ============================================================================


def test_docs_kubernetes_aiperf_benchmark_snippets_use_current_field_shapes() -> None:
    failures: list[str] = []
    for snippet in _AIPERF_BENCHMARK_SNIPPETS:
        benchmark = _benchmark_mapping(snippet)
        if benchmark is None:
            continue

        models = benchmark.get("models")
        if isinstance(models, dict) and "items" in models:
            failures.append(
                f"{snippet.source_id}: replace stale spec.benchmark.models.items with "
                "the current list form, e.g. models: [meta-llama/Llama-3.1-8B-Instruct]"
            )

        spec = _spec_mapping(snippet) or {}
        sweep = _mapping(spec.get("sweep"))
        if sweep is not None and sweep.get("type") == "adaptive_search":
            if "objective" in sweep:
                failures.append(
                    f"{snippet.source_id}: replace stale sweep.objective with "
                    "sweep.objectives: [{metric, stat, direction}]"
                )
            objectives = _sequence(sweep.get("objectives"))
            if objectives is None:
                failures.append(
                    f"{snippet.source_id}: adaptive_search sweep must document "
                    "objectives as a non-empty list"
                )
            elif not objectives:
                failures.append(
                    f"{snippet.source_id}: sweep.objectives must not be empty"
                )

    assert not failures, "\n".join(failures)
