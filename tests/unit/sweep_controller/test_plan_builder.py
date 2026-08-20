# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.config import (
    BenchmarkPlan,
    build_benchmark_plan,
    load_config_from_mapping,
)
from aiperf.kubernetes.spec_converter import build_config_envelope
from aiperf.sweep_controller.plan_builder import build_plan_from_sweep


def _sweep_cr(spec: dict) -> dict:
    return {
        "metadata": {"name": "test-sweep", "namespace": "default", "uid": "abc"},
        "spec": spec,
    }


def _benchmark() -> dict:
    return {
        "models": "mock",
        "endpoint": {"urls": ["http://x:8000/v1/chat/completions"]},
        "datasets": [{"name": "main", "type": "synthetic"}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            }
        ],
    }


def _build_local_plan(spec: dict) -> BenchmarkPlan:
    config = load_config_from_mapping(build_config_envelope(spec))
    return build_benchmark_plan(config)


def _assert_canonical_plan_parity(
    kube_plan: BenchmarkPlan, local_plan: BenchmarkPlan
) -> None:
    assert kube_plan.configs == local_plan.configs
    assert kube_plan.variations == local_plan.variations
    assert kube_plan.variation_seeds == local_plan.variation_seeds
    assert kube_plan.trials == local_plan.trials
    assert kube_plan.cooldown_seconds == local_plan.cooldown_seconds
    assert kube_plan.confidence_level == local_plan.confidence_level
    assert kube_plan.random_seed == local_plan.random_seed
    assert kube_plan.set_consistent_seed == local_plan.set_consistent_seed
    assert kube_plan.disable_warmup_after_first == local_plan.disable_warmup_after_first
    assert kube_plan.no_sweep_table == local_plan.no_sweep_table
    assert kube_plan.multi_run == local_plan.multi_run
    assert kube_plan.sweep == local_plan.sweep
    assert kube_plan.variables == local_plan.variables
    assert kube_plan.plot == local_plan.plot


def test_build_plan_grid_sweep():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [8, 32]},
            },
            "multiRun": {"numRuns": 2},
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 2
    assert len(plan.variations) == 2
    assert plan.trials == 2
    assert plan.variations[0].values == {"phases.profiling.concurrency": 8}
    assert plan.variations[1].values == {"phases.profiling.concurrency": 32}


def test_build_plan_rejects_credential_axis_for_legacy_accepted_cr() -> None:
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "parameters": {"variables.api_token": ["token-secret"]},
            },
            "variables": {"api_token": ""},
            "benchmark": _benchmark(),
        }
    )

    with pytest.raises(ValueError, match="credential-bearing values") as exc_info:
        build_plan_from_sweep(cr)

    assert "token-secret" not in str(exc_info.value)


def test_build_plan_no_sweep_just_multirun():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [16]},
            },
            "multiRun": {"numRuns": 5, "cooldownSeconds": 10},
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 1
    assert plan.trials == 5


def test_build_plan_convergence_uses_num_runs_for_trials():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [16]},
                "iteration_order": "independent",
            },
            "multiRun": {
                "numRuns": 7,
                "cooldownSeconds": 30,
                "convergence": {"metric": "ttft_p99", "threshold": 0.05},
            },
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    # Convergence early-stops within numRuns; numRuns is the worst-case cap.
    assert plan.trials == 7
    assert plan.multi_run.convergence is not None
    assert plan.multi_run.convergence.metric == "ttft_p99"
    assert plan.multi_run.convergence.threshold == 0.05


def test_build_plan_matches_local_config_v2_for_templated_grid_sweep():
    spec = {
        "randomSeed": 41,
        "noSweepTable": True,
        "variables": {"concurrency": 1, "request_count": 7},
        "sweep": {
            "type": "grid",
            "parameters": {"variables.concurrency": [8, 32]},
        },
        "multiRun": {
            "numRuns": 3,
            "cooldownSeconds": 2,
            "confidenceLevel": 0.9,
            "setConsistentSeed": True,
            "disableWarmupAfterFirst": True,
        },
        "plot": {
            "visualization": {
                "multiRunDefaults": ["throughput"],
                "multiRunPlots": {
                    "throughput": {
                        "type": "line",
                        "x": {
                            "metric": "request_latency",
                            "stat": "avg",
                        },
                        "y": {
                            "metric": "output_token_throughput",
                            "stat": "avg",
                        },
                    }
                },
            }
        },
        "benchmark": {
            **_benchmark(),
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": "{{ request_count }}",
                    "concurrency": "{{ concurrency }}",
                }
            ],
        },
    }
    cr = _sweep_cr(spec)

    kube_plan = build_plan_from_sweep(cr)
    local_plan = _build_local_plan(spec)

    _assert_canonical_plan_parity(kube_plan, local_plan)
    assert [cfg.phases[0].concurrency for cfg in kube_plan.configs] == [8, 32]
    assert kube_plan.variation_seeds == [41, 42]
    assert kube_plan.failure_policy is not None


def test_build_plan_matches_local_config_v2_for_adaptive_search():
    spec = {
        "variables": {"base_concurrency": 2},
        "sweep": {
            "type": "adaptive_search",
            "searchSpace": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 64,
                    "kind": "int",
                }
            ],
            "objectives": [
                {
                    "metric": "output_token_throughput",
                    "stat": "avg",
                    "direction": "maximize",
                }
            ],
            "maxIterations": 8,
            "randomSeed": 42,
        },
        "multiRun": {"numRuns": 2},
        "benchmark": {
            **_benchmark(),
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": "{{ base_concurrency }}",
                }
            ],
        },
    }
    cr = _sweep_cr(spec)

    kube_plan = build_plan_from_sweep(cr)
    local_plan = _build_local_plan(spec)

    _assert_canonical_plan_parity(kube_plan, local_plan)
    assert kube_plan.is_adaptive_search is True
    assert kube_plan.configs[0].phases[0].concurrency == 2
    assert kube_plan.failure_policy is not None


def _adaptive_spec() -> dict:
    return {
        "sweep": {
            "type": "adaptive_search",
            "searchSpace": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 64,
                    "kind": "int",
                }
            ],
            "objectives": [
                {
                    "metric": "output_token_throughput",
                    "stat": "avg",
                    "direction": "maximize",
                }
            ],
            "maxIterations": 8,
        },
        "benchmark": _benchmark(),
    }


def _qmc_spec(sweep_type: str, *, seed: int | None = None) -> dict:
    sweep: dict = {
        "type": sweep_type,
        "samples": 4,
        "dimensions": [
            {
                "path": "phases.profiling.concurrency",
                "lo": 1,
                "hi": 64,
                "kind": "int",
            }
        ],
    }
    if seed is not None:
        sweep["seed"] = seed
    return {"sweep": sweep, "benchmark": _benchmark()}


@pytest.mark.parametrize(
    "spec,path",
    [
        param(_adaptive_spec(), "variables.my_auth_token", id="adaptive"),
        param(_qmc_spec("sobol"), "endpoint.headers.X-Custom-Auth", id="qmc"),
    ],
)  # fmt: skip
def test_build_plan_rejects_credential_paths_for_non_grid_sweeps(
    spec: dict,
    path: str,
) -> None:
    axis_key = (
        "searchSpace" if spec["sweep"]["type"] == "adaptive_search" else "dimensions"
    )
    spec["sweep"][axis_key][0]["path"] = path

    with pytest.raises(ValueError, match="credential-bearing values"):
        build_plan_from_sweep(_sweep_cr(spec))


@pytest.mark.parametrize(
    "spec",
    [
        param(_adaptive_spec(), id="adaptive"),
        param(_qmc_spec("sobol"), id="qmc"),
    ],
)  # fmt: skip
def test_build_plan_allows_benign_max_tokens_path_for_non_grid_sweeps(
    spec: dict,
) -> None:
    axis_key = (
        "searchSpace" if spec["sweep"]["type"] == "adaptive_search" else "dimensions"
    )
    spec["sweep"][axis_key][0]["path"] = "variables.max_tokens"

    plan = build_plan_from_sweep(_sweep_cr(spec))

    assert plan.sweep is not None


def test_build_plan_derives_restart_stable_adaptive_seed_from_sweep_uid():
    first = build_plan_from_sweep(_sweep_cr(_adaptive_spec()))
    second = build_plan_from_sweep(_sweep_cr(_adaptive_spec()))

    assert first.sweep is not None
    assert second.sweep is not None
    assert first.sweep.random_seed is not None
    assert second.sweep.random_seed == first.sweep.random_seed


def test_build_plan_uses_distinct_adaptive_seed_for_distinct_sweep_uid():
    first_cr = _sweep_cr(_adaptive_spec())
    second_cr = _sweep_cr(_adaptive_spec())
    second_cr["metadata"]["uid"] = "different-uid"

    first = build_plan_from_sweep(first_cr)
    second = build_plan_from_sweep(second_cr)

    assert first.sweep is not None
    assert second.sweep is not None
    assert first.sweep.random_seed != second.sweep.random_seed


def test_build_plan_preserves_explicit_adaptive_random_seed():
    spec = _adaptive_spec()
    spec["sweep"]["randomSeed"] = 73

    plan = build_plan_from_sweep(_sweep_cr(spec))

    assert plan.sweep is not None
    assert plan.sweep.random_seed == 73


def test_build_plan_rejects_adaptive_restart_without_apiserver_uid():
    cr = _sweep_cr(_adaptive_spec())
    del cr["metadata"]["uid"]

    with pytest.raises(ValueError, match="metadata.uid"):
        build_plan_from_sweep(cr)


@pytest.mark.parametrize(
    "sweep_type",
    [
        param("sobol", id="sobol"),
        param("latin_hypercube", id="latin-hypercube"),
    ],
)
def test_build_plan_derives_restart_stable_qmc_seed_from_sweep_uid(
    sweep_type: str,
) -> None:
    """Unseeded Kubernetes QMC plans must retain child identity after restart."""
    cr = _sweep_cr(_qmc_spec(sweep_type))

    first = build_plan_from_sweep(cr)
    second = build_plan_from_sweep(cr)

    assert first.sweep is not None
    assert second.sweep is not None
    assert first.sweep.seed is not None
    assert second.sweep.seed == first.sweep.seed
    assert second.variations == first.variations
    assert second.configs == first.configs


@pytest.mark.parametrize(
    "sweep_type",
    [
        param("sobol", id="sobol"),
        param("latin_hypercube", id="latin-hypercube"),
    ],
)
def test_build_plan_preserves_explicit_qmc_seed(sweep_type: str) -> None:
    """Kubernetes restart hardening must not replace a user-authored seed."""
    plan = build_plan_from_sweep(_sweep_cr(_qmc_spec(sweep_type, seed=73)))

    assert plan.sweep is not None
    assert plan.sweep.seed == 73


@pytest.mark.parametrize(
    "sweep_type",
    [
        param("sobol", id="sobol"),
        param("latin_hypercube", id="latin-hypercube"),
    ],
)
def test_local_unseeded_qmc_plan_remains_unseeded(sweep_type: str) -> None:
    """Kubernetes restart seeding must not change local Config-v2 semantics."""
    plan = _build_local_plan(_qmc_spec(sweep_type))

    assert plan.sweep is not None
    assert plan.sweep.seed is None


@pytest.mark.parametrize(
    "sweep_type",
    [
        param("sobol", id="sobol"),
        param("latin_hypercube", id="latin-hypercube"),
    ],
)
def test_build_plan_rejects_unseeded_qmc_restart_without_apiserver_uid(
    sweep_type: str,
) -> None:
    """A missing UID cannot safely seed an otherwise-random Kubernetes plan."""
    cr = _sweep_cr(_qmc_spec(sweep_type))
    del cr["metadata"]["uid"]

    with pytest.raises(ValueError, match="metadata.uid"):
        build_plan_from_sweep(cr)
