# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-omission migration ledger for evaluator benchmark/provider surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.accuracy import worker as legacy_worker
from aiperf.accuracy.evaluation.distributions import (
    STOCK_DISTRIBUTIONS,
    executable_tasks,
    task_manifest,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

_ROOT = Path(__file__).resolve().parents[3]
_COVERAGE_MANIFEST = (
    _ROOT / "src/aiperf/accuracy/evaluation/manifests/provider_coverage.json"
)


def _coverage() -> dict[str, Any]:
    value = orjson.loads(_COVERAGE_MANIFEST.read_bytes())
    assert isinstance(value, dict)
    assert set(value) == {
        "schema_version",
        "claim_policy",
        "migrated_canary_proofs",
        "static_benchmarks",
        "agentic_providers",
    }
    assert value["schema_version"] == "aiperf-evaluator-provider-coverage-v1"
    assert "Only the finite cases" in value["claim_policy"]
    return value


def test_coverage_manifest_enumerates_every_static_registry_entry() -> None:
    coverage = _coverage()
    entries = coverage["static_benchmarks"]
    assert all(
        set(entry)
        == {
            "canonical_registry_id",
            "plugin_registry_id",
            "legacy_authority",
            "legacy_execution_path",
            "legacy_grading_path",
            "status",
            "retention_reason",
            "deletion_gate",
            "migrated_canary_coverage_ids",
        }
        for entry in entries
    )
    by_id = {entry["canonical_registry_id"]: entry for entry in entries}
    assert len(by_id) == len(entries)

    registrations = legacy_worker._REGISTRATIONS
    assert set(by_id) == set(registrations)
    plugin_entries = tuple(
        entry
        for entry in plugins.list_entries(PluginType.ACCURACY_BENCHMARK)
        if entry.metadata.get("is_implemented", True)
    )
    plugin_by_canonical = {
        legacy_worker._canonical_benchmark(entry.name): entry.name
        for entry in plugin_entries
    }
    assert set(plugin_by_canonical) == set(registrations)

    for registry_id, registration in registrations.items():
        entry = by_id[registry_id]
        assert entry["plugin_registry_id"] == plugin_by_canonical[registry_id]
        assert entry["legacy_authority"] == ("aiperf.accuracy.worker:_REGISTRATIONS")
        assert entry["legacy_execution_path"] == registration.benchmark_class
        assert entry["legacy_grading_path"] == registration.grader_class

    # MMLU-Pro is now the custom TIGER-Lab benchmark registered in
    # _REGISTRATIONS (MMLUProBenchmark + MMLUProGrader), covered by the loop
    # above; it is no longer the special-cased lighteval _load_mmlu_pro path.
    _assert_explicit_retention(entries)


def test_coverage_manifest_enumerates_every_agentic_provider() -> None:
    coverage = _coverage()
    entries = coverage["agentic_providers"]
    assert all(
        set(entry)
        == {
            "capability",
            "namespace_prefix",
            "legacy_authority",
            "legacy_execution_path",
            "required_versions",
            "status",
            "retention_reason",
            "deletion_gate",
        }
        for entry in entries
    )
    by_capability = {entry["capability"]: entry for entry in entries}
    providers = {
        provider.capability: provider
        for provider in legacy_worker._AGENTIC_HARNESS_PROVIDERS
    }
    assert len(by_capability) == len(entries)
    assert set(by_capability) == set(providers)
    for capability, provider in providers.items():
        entry = by_capability[capability]
        assert entry["namespace_prefix"] == provider._namespace_prefix
        assert entry["legacy_authority"] == (
            "aiperf.accuracy.worker:_AGENTIC_HARNESS_PROVIDERS"
        )
        assert entry["legacy_execution_path"] == provider._factory
        assert entry["required_versions"] == provider._required_versions
    _assert_explicit_retention(entries)


def test_only_stock_gsm8k_canaries_claim_migration() -> None:
    coverage = _coverage()
    proofs = coverage["migrated_canary_proofs"]
    assert all(
        set(proof)
        == {
            "coverage_id",
            "status",
            "benchmark_registry_id",
            "provider_id",
            "distribution_id",
            "task_id",
            "asset_id",
            "asset_revision",
            "asset_content_sha256",
            "asset_record_count",
            "executed_case_count",
            "operation_id",
            "parity_semantics",
            "proof_test",
        }
        for proof in proofs
    )
    by_distribution = {proof["distribution_id"]: proof for proof in proofs}
    assert len(by_distribution) == len(proofs)
    assert set(by_distribution) == set(STOCK_DISTRIBUTIONS)
    assert {proof["benchmark_registry_id"] for proof in proofs} == {"gsm8k"}
    assert {proof["status"] for proof in proofs} == {"migrated_canary"}
    assert {proof["executed_case_count"] for proof in proofs} == {1}
    assert {proof["asset_record_count"] for proof in proofs} == {5}

    for distribution_id, descriptor in STOCK_DISTRIBUTIONS.items():
        proof = by_distribution[distribution_id]
        assert proof["provider_id"] == descriptor.provider_id
        assert proof["task_id"] == "gsm8k"
        assert executable_tasks(descriptor) == ("gsm8k",)
        manifest = task_manifest(descriptor)
        task = (
            manifest["environments"]["gsm8k"]
            if descriptor.provider_id == "nemo_evaluator"
            else manifest["tasks"]["gsm8k"]
        )
        asset = task["assets"][0]
        assert proof["asset_id"] == asset["asset_id"]
        assert proof["asset_revision"] == asset["immutable_revision"]
        assert proof["asset_content_sha256"] == asset["content_sha256"]
        operation_id = (
            task["logical_services"][0]["operations"][0]
            if descriptor.provider_id == "nemo_evaluator"
            else task["operations"][0]
        )
        assert proof["operation_id"] == operation_id
        test_path, node = proof["proof_test"].split("::", 1)
        test_source = (_ROOT / test_path).read_text(encoding="utf-8")
        assert node.startswith("test_stock_provider_over_dedicated_fds[")
        assert (
            node.removeprefix("test_stock_provider_over_dedicated_fds[").removesuffix(
                "]"
            )
            in test_source
        )

    static = {
        entry["canonical_registry_id"]: entry for entry in coverage["static_benchmarks"]
    }
    migrated_ids = {proof["coverage_id"] for proof in proofs}
    assert set(static["gsm8k"]["migrated_canary_coverage_ids"]) == migrated_ids
    assert all(
        not entry["migrated_canary_coverage_ids"]
        for registry_id, entry in static.items()
        if registry_id != "gsm8k"
    )


def _assert_explicit_retention(entries: list[dict[str, Any]]) -> None:
    assert entries
    for entry in entries:
        assert entry["status"] == "legacy_retained"
        assert isinstance(entry["retention_reason"], str)
        assert entry["retention_reason"].strip()
        assert isinstance(entry["deletion_gate"], str)
        assert entry["deletion_gate"].strip()
