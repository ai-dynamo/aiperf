---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Adaptive-Search Architecture
---

# Adaptive-Search Architecture

How AIPerf's adaptive-search outer loop is wired together: the protocols, the runtime sequence, and the config-to-execution flow. This is a developer reference; for the user-facing CLI guide see [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md).

The optimization backend is [Optuna](https://optuna.org/) (a default dependency), with the modern Gaussian-process path supplied by the optional [BoTorch](https://botorch.org/) stack (`pip install aiperf[botorch]`). There is no skopt anywhere in the code — the default planner is a curated Optuna preset.

## Class structure

The planner and the orchestrator talk through narrow protocols. `MultiRunOrchestrator` doesn't know about Bayesian Optimization — it only knows `SearchPlanner` and `RunExecutor`. The Optuna dependency is hidden inside `OptunaSearchPlanner`; `BayesianSearchPlanner` is a thin curated subclass of it, and `MonotonicSLASearchPlanner` / `SmoothIsotonicSLAPlanner` are 1D-feasibility-search planners that plug in at the same `SearchPlanner` ABC.

### Registered planners

Four planners are registered in `src/aiperf/plugin/plugins.yaml` under the `search_planner:` category and resolved via `plugins.get_class(PluginType.SEARCH_PLANNER, str(cfg.planner))`.

| Plugin name | Class | Module | Purpose |
|---|---|---|---|
| `bayesian` | `BayesianSearchPlanner` | `bayesian.py` | Curated Optuna preset. Tries `optuna_sampler="botorch"` with `qlognei` (single-objective) / `qlognehvi` (multi-objective); falls back to Optuna's core TPE with a warning when the optional BoTorch stack is unavailable. Default planner. |
| `optuna` | `OptunaSearchPlanner` | `optuna_planner.py` | Expert mode. Explicit `--optuna-sampler` (`tpe` / `gp` / `botorch`) and `--optuna-acquisition` selection; optional posterior-regret terminator. `bayesian`'s superclass. |
| `monotonic_sla` | `MonotonicSLASearchPlanner` | `monotonic.py` | 1D exponential probe + bisection mirroring perf_analyzer's `--binary-search`. Requires exactly one `int` dimension and >=1 SLA filter. Margin-magnitude-blind. |
| `smooth_isotonic` | `SmoothIsotonicSLAPlanner` | `smooth_isotonic.py` (+ helpers `_smooth_isotonic_fit.py`, `_smooth_isotonic_phases.py`, `_replicate_budget.py`, `_cliff_detect.py`, `_margin_normalize.py`) | 1D PAVA + PCHIP smooth-isotonic fit; opt-in replicates and bootstrap CI; cliff-curve guard. Default for `max-concurrency-under-sla`. |

When two or more SLO tiers are configured (via `--search-sla-tier`), `_build_search_planner` overrides the selection with `MultiTierPlanner` (`multi_tier_planner.py`) regardless of `planner`.

```mermaid
classDiagram
    %% --- Config layer (aiperf.config.sweep) -----------------------------
    class AdaptiveSearchSweep {
        <<Pydantic; SweepConfig union variant>>
        +type: Literal["adaptive_search"]
        +planner: SearchPlannerType
        +search_space: list[SearchSpaceDimension]
        +objectives: list[Objective]
        +outcome_constraints: list[OutcomeConstraint]
        +sla_filters: list[SLAFilter]
        +max_iterations: int  // 2..200
        +n_initial_points: int  // >=1, default 5
        +plateau_window: int  // default 8
        +plateau_threshold: float  // default 0.01
        +improvement_patience: int  // default 10
        +optuna_sampler: Literal["gp","tpe","botorch"]
        +optuna_acquisition: str | None
        +optuna_terminator: Literal["regret","emmr","none"]
        +objective_pooling: Literal["mean","pooled"]
        +random_seed: int | None
    }
    class Objective {
        <<Pydantic>>
        +metric: str
        +stat: Literal["avg","p50","p90","p95","p99"]
        +direction: OptimizationDirection
        +threshold: float | None  // Pareto reference point
    }
    class SearchSpaceDimension {
        <<Pydantic>>
        +path: str
        +lo: float
        +hi: float
        +kind: Literal["int","real"]
        +prior: Literal["uniform","log-uniform"]
    }
    class BenchmarkPlan {
        <<Pydantic>>
        +configs: list[BenchmarkConfig]
        +variations: list[SweepVariation]
        +trials: int
        +sweep: SweepConfig | None
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
        +iter_count: int
        +convergence_reason() str | None
        +boundary_summary() dict | None
    }
    class OptunaSearchPlanner {
        -_study: optuna.Study
        -_iter: int
        -_history: list[SearchIteration]
        -_pending_trial: Any | None
        -_best_loss: float | None
        -_best_hypervolume: float | None
        -_iters_since_improvement: int
        -_sla_filters / _outcome_constraints
        -_terminator / _terminator_reason
        -_extract_objective_vector(results)
        -_failure_sentinel_vector()
        -_track_improvement(...)
        -_maybe_install_qnehvi_candidates_func()
    }
    class BayesianSearchPlanner {
        <<curated Optuna preset>>
        // locks optuna_sampler="botorch"
        // + qlognei / qlognehvi acquisition
        // + TPE fallback
    }
    class MonotonicSLASearchPlanner {
        <<1D feasibility>>
        -_lo: int
        -_hi: int
        -_phase: Literal["probe","bisect"]
        +boundary_summary() dict
    }
    class SmoothIsotonicSLAPlanner {
        <<1D feasibility, denoised>>
        -_phase: Literal["bracket","fit","replicate","cliff_bisect"]
        -_candidate_x: float | None
        -boundary_type: Literal["smooth","cliff"] | None
        -binding_constraint: str | None
        -boundary_ci_low/high: float | None
        +boundary_summary() dict
    }
    class SearchIteration {
        <<dataclass>>
        +iteration_idx: int
        +variation_values: dict
        +objective_value: float | None
        +objective_values: list[float] | None
        +results: list[RunResult]
        +feasible: bool
        +non_monotonic_warning: bool
    }

    %% --- Orchestration layer --------------------------------------------
    class MultiRunOrchestrator {
        +base_dir: Path
        +execute(plan, executor, *, cancel_check, search_planner)
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
    SearchPlanner <|-- OptunaSearchPlanner
    OptunaSearchPlanner <|-- BayesianSearchPlanner
    SearchPlanner <|-- MonotonicSLASearchPlanner
    SearchPlanner <|-- SmoothIsotonicSLAPlanner
    RunExecutor <|-- LocalSubprocessExecutor
    RunExecutor <|-- K8sChildJobExecutor

    AdaptiveSearchSweep "1" *-- "1..*" SearchSpaceDimension : search_space
    AdaptiveSearchSweep "1" *-- "1..*" Objective : objectives
    BenchmarkPlan "1" o-- "0..1" AdaptiveSearchSweep : sweep

    OptunaSearchPlanner ..> AdaptiveSearchSweep : reads config
    OptunaSearchPlanner "1" *-- "*" SearchIteration : history

    MultiRunOrchestrator ..> SearchPlanner : drives via ask/tell
    MultiRunOrchestrator ..> RunExecutor : delegates execute()
    MultiRunOrchestrator ..> BenchmarkPlan : reads is_adaptive_search
```

`AdaptiveSearchSweep` is one of the `SweepConfig` discriminated-union variants (`Discriminator("type")`, alongside `GridSweep`, `ZipSweep`, `ScenarioSweep`, `SobolSweep`, `LatinHypercubeSweep`). `BenchmarkPlan.is_adaptive_search` is simply `isinstance(self.sweep, AdaptiveSearchSweep)`.

The CLI grammar lives in `aiperf.orchestrator.search_planner.parsing`: `parse_search_space(values)` converts `--search-space "path:lo,hi[:kind]"` into `SearchSpaceDimension` instances, `parse_sla_filter` / `parse_sla_tier` handle `--search-sla` / `--search-sla-tier`. The objective (`--search-metric` / `--search-stat` / `--search-direction`) is three plain Pydantic-validated fields and needs no parser.

## Runtime sequence — one search iteration

`MultiRunOrchestrator.execute_adaptive_search` is a thin loop. Every iteration: ask the planner for a `(BenchmarkConfig, SweepVariation)`; run all configured trials at that point via the same `_run_independent_cell` grid sweeps use; tell the planner what happened; write `search_history.json` incrementally. When `ask()` returns `None`, surface the planner's `convergence_reason()` and exit.

```mermaid
sequenceDiagram
    participant Runner as cli_runner._run_multi_benchmark<br/>(or sweep_controller.main)
    participant Orch as MultiRunOrchestrator<br/>.execute_adaptive_search
    participant Planner as OptunaSearchPlanner<br/>(or BayesianSearchPlanner)
    participant Study as optuna.Study
    participant Cell as _run_independent_cell
    participant Exec as RunExecutor<br/>(Local or K8s)
    participant Hist as write_search_history

    Runner->>Orch: execute(plan, executor, search_planner=planner)
    Orch->>Orch: dispatch on plan.is_adaptive_search

    loop until ask() returns None
        Orch->>Planner: ask()
        Planner->>Planner: is_converged()?  (returns None if so)
        Planner->>Study: trial = study.ask()
        Planner->>Study: trial.suggest_int / suggest_float(log=prior)
        Planner->>Planner: base_config.model_dump(python, include_secrets)<br/>+ _set_nested_value(path, val) per dim
        Planner-->>Orch: (BenchmarkConfig, SweepVariation)<br/>label="search_iter_NNNN"

        Note over Orch: if proposal is None,<br/>read planner.convergence_reason(),<br/>log & flush history, exit

        Orch->>Cell: _run_independent_cell(plan, executor, cfg, variation, ...)
        loop trials per search point
            Cell->>Exec: execute(BenchmarkRun)
            Exec-->>Cell: RunResult(summary_metrics, ...)
        end
        Cell-->>Orch: list[RunResult] (length = plan.trials)

        Orch->>Planner: tell(variation, results)
        Planner->>Planner: _populate_user_attrs(trial, results)<br/>(per-SLA averaged obs for constraints_func)
        Planner->>Planner: _extract_objective_vector(results)<br/>-> list[float] (mean or pooled percentile per objective)
        alt objective vector is None (no usable data)
            Planner->>Planner: _failure_sentinel_vector()<br/>(per-objective worst-of-prior + max(10%,1.0),<br/>or +/- 1e6 when no prior)
            Planner->>Study: study.tell(trial, sentinel_vector)
        else objective vector usable
            Planner->>Study: study.tell(trial, objective_vector)
        end
        Planner->>Planner: _track_improvement (scalar best-loss<br/>or feasible-set hypervolume)<br/>+ append SearchIteration + _iter++
        Planner->>Planner: _maybe_install_qnehvi_candidates_func()<br/>(multi-objective, once initial points land)

        Orch->>Hist: write_search_history(base_dir, history, sweep, planner=planner)
        Note over Hist: iterations[], best{}, best_trials[],<br/>boundary_summary, convergence_reason
    end

    Orch-->>Runner: list[RunResult]
```

A few things this view makes explicit:

- **Pre-averaged objective per trial group.** `_extract_objective_vector` reduces the per-point trial set to one value per objective — the arithmetic mean of finite per-trial values (`objective_pooling="mean"`), or, when `objective_pooling="pooled"` and the stat is a percentile, `np.percentile` over the pooled raw-sample bag walked from each trial's `profile_export.jsonl` (requires `--export-level records`). Optuna receives exactly one `study.tell(trial, vector)` per iteration.
- **Multi-objective is first-class.** `objectives` is a list. Length 1 is single-objective BO; length N is Pareto BO (BoTorch `qLogNEHVI` / `qNEHVI` / `qEHVI`). The `study` is created with `directions=[...]` and, for the multi-objective path, `_maybe_install_qnehvi_candidates_func` binds the qNEHVI `candidates_func` once `n_initial_points` feasible probes have accumulated so a reference point can be derived.
- **Constraints via Optuna, not penalty reweighting.** SLA filters and `outcome_constraints` are written to `trial.user_attrs` during `tell()` and read back by Optuna's first-class `constraints_func` at `study.tell()` time. `iteration_feasibility` also stamps `SearchIteration.feasible` so `write_search_history` can do feasibility-first best-result selection.
- **Failed-iteration sentinel.** When zero trials produce a usable objective, `_failure_sentinel_vector` synthesizes a per-objective value strictly worse than any seen so far (worst-of-prior plus a `max(10%, 1.0)` margin in that objective's worse direction), or `±1e6` (`NO_DATA_SENTINEL_LOSS`) when nothing has yet succeeded — so the ask/tell pairing stays consistent and the GP kernel never sees NaN/inf.
- **Convergence signals.** `is_converged()` runs the shared three-signal check first (`max_iterations`, then `improvement_patience`, then `plateau_cv`), then — for `optuna` / `bayesian` only — an optional Optuna terminator (`posterior_regret_bound` for Makarova 2022, `emmr` for Ishibashi 2023). The first to fire wins; the reason is recorded in `search_history.json`. The 1D SLA planners add their own reasons (`monotonic_precision_reached` / `monotonic_no_pass_in_range` / `monotonic_no_failure_in_range`, and `smooth_isotonic_*`).

## Config flow — CLI / YAML -> execution

Two entry points feed the same typed `AdaptiveSearchSweep`; from `AIPerfConfig.sweep` onward, the in-process and cluster paths are identical except for which `RunExecutor` impl is plugged in.

```mermaid
flowchart TB
    subgraph CLI["Two entry points"]
        direction LR
        CLI_FLAGS["aiperf profile --search-space ...<br/>--search-metric ... --search-direction ...<br/>--search-max-iterations N [--search-planner ...]"]
        CRD["AIPerfSweep CR with<br/>spec.sweep (type: adaptive_search)"]
    end

    subgraph FLAGS["Flag layer (CLI input)"]
        CC["CLIConfig<br/>(search_space, search_metric,<br/>search_stat, search_direction,<br/>search_max_iterations, search_planner,<br/>optuna_sampler, optuna_acquisition, ...)"]
    end

    subgraph TYPED["Typed config layer"]
        BS["build_sweep / build_multi_run<br/>(config.flags._converter_optionals)"]
        AS["AdaptiveSearchSweep<br/>+ SearchSpaceDimension[] + Objective[]"]
        AC["AIPerfConfig.sweep"]
    end

    subgraph PLAN["Plan build"]
        BBP["build_benchmark_plan<br/>(config.loader.plan; in-process)"]
        BPS["build_plan_from_sweep<br/>(sweep_controller.plan_builder; K8s pod)"]
        BP["BenchmarkPlan<br/>.sweep / .is_adaptive_search"]
    end

    subgraph EXEC["Execution"]
        BSP["_build_search_planner(plan)<br/>plugins.get_class(SEARCH_PLANNER, planner)"]
        DISPATCH["MultiRunOrchestrator.execute<br/>dispatch on is_adaptive_search"]
        EAS["execute_adaptive_search<br/>(ask/tell loop)"]
        LSE["LocalSubprocessExecutor<br/>(in-process)"]
        KCE["K8sChildJobExecutor<br/>(sweep-controller pod)"]
    end

    subgraph OUT["Outputs"]
        HIST["search_history.json<br/>{config, iterations, best,<br/>best_trials, boundary_summary,<br/>convergence_reason}"]
        AGG["sweep_aggregate/<br/>profile_export_aiperf_sweep.{json,csv}"]
        ITER["search_iter_NNNN/<br/>profile_runs/run_NNNN/<br/>(per-trial artifacts)"]
    end

    CLI_FLAGS --> CC
    CC -->|build_sweep parses,<br/>validates, builds typed config| BS
    BS --> AS
    AS --> AC
    CRD -->|already typed| AC

    AC -->|in-process| BBP
    AC -->|via AIPerfSweep spec| BPS
    BBP --> BP
    BPS --> BP

    BP --> BSP
    BSP -.instantiate planner.-> EAS
    BP --> DISPATCH
    DISPATCH -->|is_adaptive_search=true| EAS
    EAS -->|in-process| LSE
    EAS -->|cluster| KCE

    EAS -->|every iter| HIST
    EAS -->|all results| AGG
    EAS -->|per cell| ITER
```

## Notes on extension points

- **Adding a new planner backend**: subclass `SearchPlanner`, implement the four abstract methods (`ask` / `tell` / `is_converged` / `history`); optionally override `iter_count` (default reads `self._iter`), `convergence_reason` (default returns `None`), and `boundary_summary` (1D feasibility planners only). Register the class in `plugins.yaml` under `search_planner:` — `_build_search_planner` resolves it via `plugins.get_class(PluginType.SEARCH_PLANNER, str(cfg.planner))`, so no orchestrator changes are required. `BayesianSearchPlanner` (subclass of `OptunaSearchPlanner`) and the 1D SLA planners are existing examples spanning both the "curated preset" and "non-BO" shapes.
- **Adding a new executor backend**: subclass `RunExecutor` and implement `execute(BenchmarkRun) -> RunResult` + `derive_id`. Both `LocalSubprocessExecutor` and `K8sChildJobExecutor` already iterate one (variation, trial) at a time, so the seam is adaptive-shaped by construction.
- **Swapping the optimizer.** The Optuna boundary is confined to `OptunaSearchPlanner` plus `_optuna_helpers` (sampler construction, hypervolume, qNEHVI candidates_func) and the `botorch` extra in `pyproject.toml`. `optuna` core is a default dependency; only the GP/BoTorch path is optional, and `BayesianSearchPlanner` degrades to TPE when it is absent.

## Related

- [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md) — user-facing CLI guide.
- [Adaptive search on Kubernetes](../kubernetes/sweeps.md) — cluster-side wiring.
- [Parameter Sweeping](../tutorials/parameter-sweeping.md) — grid-sweep alternative when adaptive search isn't the right tool.
