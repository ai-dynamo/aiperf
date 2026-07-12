# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral evaluation stays authored, strict, and v2-only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from pydantic import ValidationError

from aiperf.config import (
    AIPerfConfig,
    BenchmarkRun,
    EvaluationWorkloadConfig,
    load_config_from_string,
)
from aiperf.orchestrator.runner_installation import (
    RunnerInstallation,
    _validate_v2_capabilities,
)
from aiperf.orchestrator.rust_wire import (
    RustWireError,
    build_authored_run_request,
    build_run_request,
)

_DISTRIBUTION_ID = "blake3:" + "a" * 64


def _config(tmp_path: Path, workload_config: dict) -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["candidate", "judge"],
                "endpoint": {
                    "urls": ["http://candidate.invalid"],
                    "type": "chat",
                    "streaming": True,
                    "api_key": "outer-candidate-secret",
                },
                "endpoint_profiles": {
                    "judge_anthropic": {
                        "urls": ["https://judge.invalid"],
                        "type": "messages",
                        "streaming": True,
                        "api_key": "outer-judge-secret",
                    }
                },
                # Config v2 retains its ordinary required body. The evaluation
                # projection proves these compatibility fields do not cross
                # into the evaluation factory-owned object.
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 2},
                },
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
                "tokenizer": {"name": "builtin"},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "artifacts": {"dir": str(tmp_path)},
                "workload": {"type": "evaluation", "config": workload_config},
            }
        }
    )


def _run(tmp_path: Path, workload_config: dict) -> BenchmarkRun:
    config = _config(tmp_path, workload_config)
    return BenchmarkRun(
        benchmark_id="evaluation-v2",
        cfg=config.benchmark,
        artifact_dir=tmp_path,
        label="evaluation",
        trial=0,
    )


def _workload_config() -> dict:
    return {
        "provider": {
            "type": "nemo_evaluator",
            "distribution": "nvidia_nemo_evaluator_0_4_locked",
        },
        "evaluation": {
            "benchmark": "fixture/exact@locked",
            "provider_options": {"opaque": [1, 2, 3]},
        },
        "routes": {
            "primary": {"model": "candidate", "endpoint_profile": "default"},
            "judge": {
                "model": "judge",
                "endpoint_profile": "judge_anthropic",
            },
        },
        "resources": {
            "workspace": {
                "type": "contained_directory",
                "config": {"quota_bytes": 4096},
            }
        },
        "unit_concurrency": 2,
    }


def test_evaluation_projection_is_first_class_multi_route_and_provider_opaque(
    tmp_path: Path,
) -> None:
    request = build_authored_run_request(
        _run(tmp_path, _workload_config()),
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_ID,
    )

    authored = request["run"]
    assert authored["workload"] == {
        "type": "evaluation",
        "config": _workload_config(),
    }
    assert (
        not {
            "worker_count",
            "dataset",
            "tokenizer",
            "phases",
            "python_executable",
            "worker_module",
            "environment",
        }
        & authored["workload"]["config"].keys()
    )
    assert [profile["id"] for profile in authored["endpoints"]["profiles"]] == [
        "default",
        "judge_anthropic",
    ]
    assert authored["workload"]["config"]["routes"]["judge"] == {
        "model": "judge",
        "endpoint_profile": "judge_anthropic",
    }
    provider_wire = orjson.dumps(authored["workload"])
    assert b"outer-candidate-secret" not in provider_wire
    assert b"outer-judge-secret" not in provider_wire
    assert b"candidate.invalid" not in provider_wire
    assert b"judge.invalid" not in provider_wire


@pytest.mark.parametrize("coordinate", ["executable", "module", "environment"])
def test_provider_selector_cannot_author_launch_coordinates(
    tmp_path: Path, coordinate: str
) -> None:
    config = _workload_config()
    config["provider"][coordinate] = "attacker-controlled"

    with pytest.raises(ValidationError, match=coordinate):
        _config(tmp_path, config)


def test_evaluator_config_is_not_imported_or_interpreted_by_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dynamic_import(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Config v2 imported an evaluator provider")

    monkeypatch.setattr("importlib.import_module", fail_dynamic_import)
    # The provider-neutral structural model has no package lookup or factory
    # discovery; those operations belong to the exact runner distribution.
    value = EvaluationWorkloadConfig.model_validate(_workload_config())

    assert value.evaluation["provider_options"]["opaque"] == [1, 2, 3]


def test_route_references_fail_before_runner_without_resolving_endpoints(
    tmp_path: Path,
) -> None:
    missing_profile = _workload_config()
    missing_profile["routes"]["judge"]["endpoint_profile"] = "missing"
    with pytest.raises(ValidationError, match="not defined"):
        _config(tmp_path, missing_profile)

    missing_model = _workload_config()
    missing_model["routes"]["judge"]["model"] = "undeclared-model"
    with pytest.raises(ValidationError, match="not present in benchmark.models"):
        _config(tmp_path, missing_model)


def test_evaluation_and_named_profiles_never_fall_back_to_protocol_v1(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, _workload_config())

    with pytest.raises(RustWireError, match="explicit workload selection"):
        build_run_request(run)


def test_config_schema_documents_the_first_class_evaluation_shape() -> None:
    schema = AIPerfConfig.model_json_schema(mode="serialization")
    evaluation = schema["$defs"]["EvaluationWorkloadConfig"]

    assert evaluation["additionalProperties"] is False
    assert set(evaluation["required"]) >= {"provider", "routes"}
    provider = schema["$defs"]["EvaluationProviderConfig"]
    assert provider["additionalProperties"] is False
    assert set(provider["required"]) == {"type", "distribution"}


def test_plan_preflight_checks_every_named_route_endpoint_profile(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, _workload_config()).benchmark
    installation = RunnerInstallation(
        binary=Path("/unused/aiperf-runner"),
        capabilities={"endpoint_types": ["chat"]},
    )

    with pytest.raises(RuntimeError, match="messages"):
        installation.preflight_plan(SimpleNamespace(configs=[config]))


def test_yaml_aliases_normalize_to_the_strict_runner_shape(tmp_path: Path) -> None:
    config = load_config_from_string(
        f"""
schemaVersion: "2.0"
benchmark:
  models: [candidate, judge]
  endpoint:
    urls: [http://candidate.invalid]
    type: chat
    streaming: true
  endpointProfiles:
    judge_anthropic:
      urls: [https://judge.invalid]
      type: messages
      streaming: true
  dataset:
    type: synthetic
    entries: 1
    prompts: {{isl: 8, osl: 2}}
  phases:
    - name: profiling
      type: concurrency
      requests: 1
      concurrency: 1
  tokenizer: {{name: builtin}}
  gpuTelemetry: {{enabled: false}}
  serverMetrics: {{enabled: false}}
  artifacts: {{dir: {tmp_path!s}}}
  workload:
    type: evaluation
    config:
      provider:
        type: openbench
        distribution: groq_openbench_0_5_3_inspect_0_3_141_locked
      evaluation:
        task: simpleqa
      routes:
        candidate:
          model: candidate
          endpointProfile: default
        judge:
          model: judge
          endpointProfile: judge_anthropic
      resources: {{}}
      unitConcurrency: 3
"""
    )
    run = BenchmarkRun(
        benchmark_id="yaml-evaluation",
        cfg=config.benchmark,
        artifact_dir=tmp_path,
        label="yaml",
        trial=0,
    )

    workload = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_ID,
    )["run"]["workload"]
    assert workload["config"]["unit_concurrency"] == 3
    assert workload["config"]["routes"]["judge"]["endpoint_profile"] == (
        "judge_anthropic"
    )


def _evaluation_capabilities() -> dict:
    digest = "a" * 64
    return {
        "capabilities_schema_version": 2,
        "protocol_versions": [1, 2],
        "supported_pairs": [["online_http", "evaluation"]],
        "statically_compatible_pairs": [["online_http", "evaluation"]],
        "backends": [{"id": "online_http"}],
        "workloads": [{"id": "evaluation"}],
        "endpoints": [{"id": "chat"}],
        "endpoint_types": ["chat"],
        "extensions": [],
        "evaluation_providers": [
            {
                "id": "nemo_evaluator",
                "display_name": "NeMo Evaluator",
                "worker_protocol_versions": [2],
                "execution_granularities": ["case"],
                "scheduling_modes": ["finite"],
                "config_schema_version": 1,
                "config_schema_sha256": digest,
                "isolation_profile_id": "strict_process_tree_v1",
                "declared_operations": ["model.generate"],
                "distributions": [
                    {
                        "id": "nvidia_nemo_evaluator_0_4_locked",
                        "package": "nemo-evaluator",
                        "package_version": "0.4.0",
                        "provider_source_sha256": digest,
                        "worker_source_sha256": digest,
                        "dependency_lock_sha256": digest,
                        "launch_closure_sha256": digest,
                    }
                ],
            }
        ],
        "evaluation_host_operations": [
            {
                "id": "model.generate",
                "family": "inference",
                "request_schema_sha256": digest,
                "response_schema_sha256": digest,
                "true_streaming": True,
                "endpoint_capabilities": ["model.generate"],
            }
        ],
        "supported_evaluation_combinations": [
            {
                "backend": "online_http",
                "workload": "evaluation",
                "provider": "nemo_evaluator",
                "distribution": "nvidia_nemo_evaluator_0_4_locked",
                "operations": ["model.generate"],
                "resources": [],
                "isolation_profile_id": "strict_process_tree_v1",
            }
        ],
    }


def _evaluation_request(distribution: str) -> dict:
    return {
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "backend": {"type": "online_http", "config": {}},
            "workload": {
                "type": "evaluation",
                "config": {
                    "provider": {
                        "type": "nemo_evaluator",
                        "distribution": distribution,
                    },
                    "evaluation": {"benchmark": "fixture/exact@locked"},
                    "routes": {
                        "primary": {
                            "model": "candidate",
                            "endpoint_profile": "default",
                        }
                    },
                    "resources": {},
                    "unit_concurrency": 1,
                },
            },
            "endpoints": {"profiles": [{"id": "default", "type": "chat"}]},
        },
    }


def test_capability_preflight_requires_exact_provider_distribution() -> None:
    capabilities = _evaluation_capabilities()
    _validate_v2_capabilities(capabilities)
    installation = RunnerInstallation(
        binary=Path("/unused/aiperf-runner"), capabilities=capabilities
    )

    installation.preflight_request(
        _evaluation_request("nvidia_nemo_evaluator_0_4_locked")
    )
    with pytest.raises(RuntimeError, match="no benchmark-name or provider fallback"):
        installation.preflight_request(
            _evaluation_request("nvidia_nemo_evaluator_mutable")
        )


def test_capability_preflight_rejects_unlinked_resource_adapter() -> None:
    capabilities = _evaluation_capabilities()
    installation = RunnerInstallation(
        binary=Path("/unused/aiperf-runner"), capabilities=capabilities
    )
    request = _evaluation_request("nvidia_nemo_evaluator_0_4_locked")
    request["run"]["workload"]["config"]["resources"] = {
        "workspace": {"type": "contained_directory", "config": {}}
    }

    with pytest.raises(RuntimeError, match="not executable"):
        installation.preflight_request(request)
