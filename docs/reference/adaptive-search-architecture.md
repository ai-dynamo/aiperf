---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Adaptive-Search Architecture
---

# Adaptive-Search Architecture

How AIPerf's Bayesian-Optimization outer loop is wired together: the protocols, the runtime sequence, and the config-to-execution flow. This is a developer reference; for the user-facing CLI guide see [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md).

> **Schema-2.0 note (May 2026).** Class diagrams and field paths below still
> reference the pre-redesign shape: `MultiRunConfig.adaptive_search`,
> `BenchmarkPlan.adaptive_search`, and the flat
> `objective_metric`/`objective_stat`/`objective_direction` triple on
> `AdaptiveSearchConfig`. In schema 2.0 the production type is
> `AdaptiveSearchSweep` (one of the `AIPerfConfig.sweep` discriminator
> variants, alongside `GridSweep` and `ScenarioSweep`) and the objective
> triple is nested under a single `objective:` block. `AdaptiveSearchConfig`
> survives only as the recipe-author shim. Sequence diagrams and dispatch
> mechanics (`MultiRunOrchestrator.execute` / `execute_adaptive_search`,
> `BayesianSearchPlanner`, `search_history.json`) are unchanged.

## Class structure

The planner and the orchestrator talk through narrow protocols. `MultiRunOrchestrator` doesn't know about Bayesian Optimization — it only knows `SearchPlanner` and `RunExecutor`. The skopt dependency is hidden inside `BayesianSearchPlanner`; `MonotonicSLASearchPlanner` and `SmoothIsotonicSLAPlanner` are 1D-feasibility-search planners that plug in at the same `SearchPlanner` ABC. Future planners (Optuna TPE, random-search baseline, etc.) plug in identically.

### Registered planners

| Plugin name | Class | Module | Purpose |
|---|---|---|---|
| `bayes` | `BayesianSearchPlanner` | `bayesian.py` | Multi-dim BO with skopt; objective-driven; supports SLA filters via penalty / EIC mode |
| `monotonic_sla` | `MonotonicSLASearchPlanner` | `monotonic.py` | 1D exponential probe + bisection mirroring perf_analyzer's `--binary-search`. Margin-magnitude-blind. |
| `smooth_isotonic` | `SmoothIsotonicSLAPlanner` | `smooth_isotonic.py` (+ helpers `_smooth_isotonic_fit.py`, `_replicate_budget.py`, `_cliff_detect.py`, `_margin_normalize.py`) | 1D PAVA + PCHIP smooth-isotonic fit; opt-in replicates and bootstrap CI; cliff-curve guard. Default for `max-concurrency-under-sla`. |

All three are registered in `src/aiperf/plugin/plugins.yaml` under the `search_planner:` category and resolved via `plugins.get_class(PluginType.SEARCH_PLANNER, name)`.

```mermaid
classDiagram
    %% --- Config layer (aiperf.config) -----------------------------------
    class AdaptiveSearchConfig {
        <<Pydantic>>
        +algorithm: Literal["bayes"]
        +search_space: list[SearchSpaceDimension]
        +objective_metric: str
        +objective_stat: Literal[avg/p50/p90/p95/p99]
        +objective_direction: OptimizationDirection
        +max_iterations: int
        +n_initial_points: int
        +plateau_window: int
        +plateau_threshold: float
        +improvement_patience: int
        +random_seed: int | None
    }
    class SearchSpaceDimension {
        <<Pydantic>>
        +path: str
        +lo: float
        +hi: float
        +kind: Literal["int","real"]
    }
    class MultiRunConfig {
        <<Pydantic; in-process AND K8s-side>>
        +num_runs: int  // in-process (le=10)
        +trials: int    // K8s-side (le=20)
        +adaptive_search: AdaptiveSearchConfig | None
    }
    class BenchmarkPlan {
        <<Pydantic>>
        +configs: list[BenchmarkConfig]
        +variations: list[SweepVariation]
        +trials: int
        +adaptive_search: AdaptiveSearchConfig | None
        +is_sweep: bool
        +is_adaptive_search: bool
    }

    %% --- Planner layer (aiperf.orchestrator.search_planner) -------------
    class SearchPlanner {
        <<abstract>>
        +ask() tuple[BenchmarkConfig, SweepVariation] | None
        +tell(variation, results) None
        +is_converged() bool
        +history() list[SearchIteration]
        +convergence_reason() str | None
    }
    class BayesianSearchPlanner {
        -_opt: skopt.Optimizer
        -_iter: int
        -_history: list[SearchIteration]
        -_pending_x: list | None
        -_best_loss: float | None
        -_iters_since_improvement: int
        -_convergence_reason: str | None
        -_extract_trial_objectives(results)
        -_objective_to_loss(objective)
        -_failed_iteration_loss()
    }
    class MonotonicSLASearchPlanner {
        <<1D feasibility>>
        -_lo: float
        -_hi: float
        -_phase: Literal["bracket","bisect"]
        +boundary_summary() dict
    }
    class SmoothIsotonicSLAPlanner {
        <<1D feasibility, denoised>>
        -_phase: Literal["bracket","fit","replicate","done"]
        -_candidate_x: float | None
        -boundary_type: "smooth" | "cliff" | None
        -binding_constraint: str | None
        -boundary_ci_low/high: float | None
        +boundary_summary() dict
    }
    class SearchIteration {
        <<dataclass>>
        +iteration_idx: int
        +variation_values: dict
        +objective_value: float | None
        +results: list[RunResult]
    }

    %% --- Orchestration layer --------------------------------------------
    class MultiRunOrchestrator {
        +base_dir: Path
        +execute(plan, executor, *, search_planner, cancel_check)
        +execute_adaptive_search(plan, executor, planner, *, cancel_check)
        -_run_independent_cell(plan, executor, ...)
    }
    class RunExecutor {
        <<abstract>>
        +execute(run: BenchmarkRun) RunResult
        +derive_id(plan, var_idx, trial) str
    }
    class LocalSubprocessExecutor
    class K8sChildJobExecutor

    %% --- Inheritance & composition --------------------------------------
    SearchPlanner <|-- BayesianSearchPlanner
    SearchPlanner <|-- MonotonicSLASearchPlanner
    SearchPlanner <|-- SmoothIsotonicSLAPlanner
    RunExecutor <|-- LocalSubprocessExecutor
    RunExecutor <|-- K8sChildJobExecutor

    AdaptiveSearchConfig "1" *-- "1..*" SearchSpaceDimension : search_space
    MultiRunConfig "1" o-- "0..1" AdaptiveSearchConfig : adaptive_search
    BenchmarkPlan "1" o-- "0..1" AdaptiveSearchConfig : adaptive_search

    BayesianSearchPlanner ..> AdaptiveSearchConfig : reads config
    BayesianSearchPlanner "1" *-- "*" SearchIteration : history

    MultiRunOrchestrator ..> SearchPlanner : drives via ask/tell
    MultiRunOrchestrator ..> RunExecutor : delegates execute()
    MultiRunOrchestrator ..> BenchmarkPlan : reads adaptive_search
```

The CLI grammar lives in `aiperf.orchestrator.search_planner.parsing.parse_search_space(values)`, which converts `--search-space "path:lo,hi[:kind]"` strings into `SearchSpaceDimension` instances. After the v1→v2 converter (`build_multi_run` in `aiperf.config.v1._converter_optionals`) packages everything into a typed `AdaptiveSearchConfig`, the CLI half of the path drops away — the cluster path arrives at the same typed config via `AIPerfSweepSpec.multi_run.adaptive_search`.

## Runtime sequence — one BO iteration

`MultiRunOrchestrator.execute_adaptive_search` is a thin loop. Every iteration: ask the planner for a `(BenchmarkConfig, SweepVariation)`; run all configured trials at that point via the same `_run_independent_cell` grid sweeps use; tell the planner what happened; write `search_history.json` incrementally. When `ask()` returns `None`, surface the planner's `convergence_reason()` and exit.

```mermaid
sequenceDiagram
    participant Runner as cli_runner._run_multi_benchmark<br/>(or sweep_controller.main)
    participant Orch as MultiRunOrchestrator<br/>.execute_adaptive_search
    participant Planner as BayesianSearchPlanner
    participant Skopt as skopt.Optimizer
    participant Cell as _run_independent_cell
    participant Exec as RunExecutor<br/>(Local or K8s)
    participant Hist as write_search_history

    Runner->>Orch: execute(plan, executor, search_planner=planner)
    Orch->>Orch: dispatch on plan.is_adaptive_search

    loop until ask() returns None
        Orch->>Planner: ask()
        Planner->>Skopt: opt.ask() → x
        Planner->>Planner: deep-copy base_config<br/>+ _set_nested_value(x)
        Planner-->>Orch: (BenchmarkConfig, SweepVariation)<br/>or None on convergence

        Note over Orch: if proposal is None,<br/>read planner.convergence_reason(),<br/>log & write history, exit

        Orch->>Cell: _run_independent_cell(plan, executor, cfg, variation, ...)
        loop trials per BO point
            Cell->>Exec: execute(BenchmarkRun)
            Exec-->>Cell: RunResult(summary_metrics, ...)
        end
        Cell-->>Orch: list[RunResult] (length = plan.trials)

        Orch->>Planner: tell(variation, results)
        Planner->>Planner: _extract_trial_objectives(results)<br/>→ list[float] (one per successful trial)
        alt all trials failed
            Planner->>Planner: _failed_iteration_loss()<br/>(worst-seen + max(10%, 1.0 abs)<br/>or 1e6 sentinel if no prior)
            Planner->>Skopt: opt.tell(x, fallback_loss)
        else exactly one successful trial
            Planner->>Planner: _objective_to_loss()
            Planner->>Skopt: opt.tell(x, loss)<br/>(scalar form)
        else N≥2 successful trials
            Planner->>Planner: _objective_to_loss() per trial
            Planner->>Skopt: opt.tell([x]*N, [loss_1..loss_N])<br/>(Letham 2017: per-trial obs)
        end
        Planner->>Planner: update _best_loss<br/>+ _iters_since_improvement<br/>+ append SearchIteration

        Orch->>Hist: write_search_history(base_dir, history, cfg)
        Note over Hist: iterations[], best{}, convergence_reason
    end

    Orch-->>Runner: list[RunResult]
```

A few things this view makes explicit:

- **Per-trial observations to the GP.** When N≥2 trials succeed, the planner calls `opt.tell([x]*N, [y_1..y_N])` instead of pre-averaging — skopt's GP estimates the noise term σ²ₙ from the within-point spread. Pre-averaging discards information the GP could use. (Letham et al. 2017, [arXiv:1706.07094](https://arxiv.org/abs/1706.07094).)
- **Failed-iteration loss.** When zero trials succeed, the planner synthesizes a `worst-seen-loss + max(10%, 1.0 absolute)` margin (or a `1e6` sentinel if nothing has yet succeeded) so skopt's ask/tell pairing stays consistent and the GP kernel matrix doesn't see inf/nan.
- **Three convergence signals.** `is_converged()` checks max_iterations, then improvement-over-best patience, then coefficient-of-variation plateau. The first to fire wins; the reason is recorded in `search_history.json`.

## Config flow — CLI / YAML → execution

Two entry points feed the same typed config; from `MultiRunConfig.adaptive_search` onward, the in-process and cluster paths are identical except for which `RunExecutor` impl is plugged in.

```mermaid
flowchart TB
    subgraph CLI["Two entry points"]
        direction LR
        CLI_FLAGS["aiperf profile --search-space ...<br/>--search-metric ... --search-direction ...<br/>--search-max-iterations N"]
        CRD["AIPerfSweep CR with<br/>spec.multiRun.adaptive_search"]
    end

    subgraph V1["v1 layer (CLI input)"]
        LG["LoadGeneratorConfig<br/>(search_space, search_metric,<br/>search_stat, search_direction,<br/>search_max_iterations, ...)"]
    end

    subgraph V2["v2 typed layer"]
        BMR["build_multi_run<br/>(_converter_optionals.py)"]
        AS["AdaptiveSearchConfig<br/>+ SearchSpaceDimension[]"]
        MR["MultiRunConfig.adaptive_search"]
    end

    subgraph PLAN["Plan build"]
        BBP["build_benchmark_plan<br/>(in-process)"]
        BPS["build_plan_from_sweep<br/>(K8s controller pod)"]
        BP["BenchmarkPlan<br/>.adaptive_search<br/>.is_adaptive_search"]
    end

    subgraph EXEC["Execution"]
        DISPATCH["MultiRunOrchestrator.execute<br/>dispatch on is_adaptive_search"]
        BSP["BayesianSearchPlanner<br/>(skopt-backed)"]
        EAS["execute_adaptive_search<br/>(ask/tell loop)"]
        LSE["LocalSubprocessExecutor<br/>(in-process)"]
        KCE["K8sChildJobExecutor<br/>(operator path)"]
    end

    subgraph OUT["Outputs"]
        HIST["search_history.json<br/>{config, iterations, best,<br/>convergence_reason}"]
        AGG["sweep_aggregate/<br/>profile_export_aiperf_sweep.{json,csv}<br/>(unchanged)"]
        ITER["search_iter_NNNN/<br/>profile_runs/run_NNNN/<br/>(per-trial artifacts)"]
    end

    CLI_FLAGS --> LG
    LG -->|build_multi_run<br/>parses, validates,<br/>builds typed config| BMR
    BMR --> AS
    AS -->|model_dump| MR
    CRD -->|already typed| MR

    MR -->|via AIPerfConfig.multi_run| BBP
    MR -->|via AIPerfSweepSpec.multiRun| BPS
    BBP --> BP
    BPS --> BP

    BP --> DISPATCH
    DISPATCH -->|is_adaptive_search=true| EAS
    DISPATCH -.instantiate.-> BSP
    EAS <-->|ask/tell| BSP
    EAS -->|in-process| LSE
    EAS -->|cluster| KCE

    EAS -->|every iter| HIST
    EAS -->|all results| AGG
    EAS -->|per cell| ITER
```

## Notes on extension points

- **Adding a new planner backend** (Optuna TPE, random-search baseline, etc.): subclass `SearchPlanner`, implement the four abstract methods (`ask`/`tell`/`is_converged`/`history`); optionally override `convergence_reason` (default returns `None`) and `boundary_summary` (1D feasibility planners only). No orchestrator changes required — `MonotonicSLASearchPlanner` and `SmoothIsotonicSLAPlanner` are existing examples of non-BO planners reusing the same ABC. The `algorithm` field on `AdaptiveSearchConfig` is `Literal["bayes"]` today; promoting it to a `CaseInsensitiveStrEnum` and dispatching in `cli_runner._run_multi_benchmark` (or the cluster equivalent in `sweep_controller.main`) is the only wiring.
- **Adding a new executor backend**: subclass `RunExecutor`. Both `LocalSubprocessExecutor` and `K8sChildJobExecutor` already iterate one (variation, trial) at a time via `execute(BenchmarkRun)` — the seam is adaptive-shaped by construction.
- **Replacing skopt.** The `BayesianSearchPlanner` boundary is the only skopt-aware code in the project (plus the `[bo]` extra in `pyproject.toml`). Swapping to BoTorch / Optuna is a single-file change with no API break, which is what unlocks the deferred upgrades documented under "What this implementation isn't" in the [user-facing BO doc](../sweeping/bayesian-optimization.md).

## Related

- [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md) — user-facing CLI guide.
- [Adaptive search on Kubernetes](../kubernetes/sweeps.md#adaptive-search-bayesian-optimization) — cluster-side wiring.
- [Parameter Sweeping](../tutorials/parameter-sweeping.md) — grid-sweep alternative when adaptive search isn't the right tool.
