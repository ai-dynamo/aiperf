<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweep architecture — overview

A one-page mental model for how AIPerf turns a config into a pile of benchmark results, whether you run it locally or on a Kubernetes cluster. For deep dives (per-handler kopf wiring, status field shapes, per-method signatures) see [sweep-orchestrator-flow.md](sweep-orchestrator-flow.md).

## The big idea

Every AIPerf run — single benchmark, parameter grid, or Bayesian search — is the same pipeline with different cardinalities:

```
config → expand into N variations → run each variation M trials → aggregate
```

The **same** `BenchmarkPlan` / `MultiRunOrchestrator` / `RunExecutor` machinery powers `aiperf profile` (local), `AIPerfJob` (one cluster benchmark), and `AIPerfSweep` (cluster sweep). The only thing that swaps is the executor: subprocess vs. child Kubernetes job.

## Three execution paths

| Path | Trigger | Where the orchestrator runs | Per-cell executor |
|---|---|---|---|
| **Local** | `aiperf profile -c config.yaml` | Same Python process | `LocalSubprocessExecutor` (forks `aiperf.orchestrator.subprocess_runner`) |
| **AIPerfJob** | `kubectl apply` an `AIPerfJob` CR | N/A — single-config plan, no orchestrator needed | (executor not used; controller pod runs the benchmark directly) |
| **AIPerfSweep** | `kubectl apply` an `AIPerfSweep` CR | Sweep-controller pod (always spawned) | `K8sChildJobExecutor` (stamps one child `AIPerfJob` per cell) |

`AIPerfJob` is "one benchmark, in a cluster." `AIPerfSweep` is "N benchmarks, parameterized, in a cluster" — and under the hood it just creates N (or N×M, with trials) `AIPerfJob` children.

## Key types

The whole flow uses about a dozen types. If you know these, you can read any sweep code.

- **`AIPerfConfig`** — top-level envelope. Holds a `BenchmarkConfig` body plus envelope-level knobs: `sweep`, `multi_run`, `variables`, `random_seed`.
- **`BenchmarkConfig`** — the actual benchmark settings (models, endpoint, datasets, phases, artifacts, …). The unit of "what to benchmark."
- **`SweepConfig`** — discriminated union (six variants, discriminated on `type`): `GridSweep` (cartesian over `variables`), `ZipSweep` (variables zipped in lockstep), `ScenarioSweep` (deep-merge `runs[i]`), `AdaptiveSearchSweep` (BO / monotonic), `SobolSweep`, and `LatinHypercubeSweep` (quasi-Monte-Carlo samplers over `dimensions`).
- **`MultiRunConfig`** — trial mechanics: `num_runs` (= trials per variation), cooldown, optional `convergence: ConvergenceConfig`.
- **`SweepVariation`** — `{index, label, values}`. One per variation; carries the parameter values that differ from base.
- **`BenchmarkPlan`** — the "expanded" form: `configs[N]`, `variations[N]`, `trials=M`, plus the originating `sweep` + `multi_run`. Output of `build_benchmark_plan`.
- **`BenchmarkRun`** — one cell: `(cfg, variation, trial, artifact_dir)`. The smallest unit of work.
- **`RunResult`** — `{success, summary_metrics, artifacts_path, variation_label, variation_values, trial_index, error}`. One per `BenchmarkRun`.
- **`MultiRunOrchestrator`** — drives the N×M loop. Picks REPEATED (trials outer) or INDEPENDENT (variations outer) based on `sweep.iteration_order`; dispatches to `execute_adaptive_search` if the sweep is adaptive.
- **`RunExecutor`** — ABC with `execute(run) -> RunResult`. Two implementations: `LocalSubprocessExecutor`, `K8sChildJobExecutor`.
- **`SweepAnalyzer`** — post-hoc aggregator. Groups `list[RunResult]` by `variation_values`; produces `best_configurations`, `pareto_optimal`, `per_combination_metrics`. Written to `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}`.

## End-to-end flow

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 70, 'padding': 16, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '16px'}}}%%
flowchart TD
    subgraph INPUT["1. user input"]
        yaml["YAML / CLI flags"]
        cr["AIPerfJob CR or<br/>AIPerfSweep CR"]
    end

    subgraph CFG["2. config (one source of truth)"]
        ac["AIPerfConfig<br/>benchmark + sweep + multi_run +<br/>variables + random_seed"]
    end

    subgraph PLAN["3. expand"]
        es["expand_sweep<br/>+ per-variation Jinja"]
        bp["BenchmarkPlan<br/>configs[N], variations[N],<br/>trials=M"]
    end

    subgraph ORCH["4. iterate (N × M cells)"]
        mo["MultiRunOrchestrator.execute<br/>(adaptive? → execute_adaptive_search)<br/>(else iteration_order →<br/>REPEATED / INDEPENDENT)"]
        br["BenchmarkRun<br/>(cfg, variation, trial)"]
    end

    subgraph EXEC["5. execute one cell"]
        re["RunExecutor"]
        loc["LocalSubprocessExecutor<br/>(forks subprocess_runner)"]
        k8s["K8sChildJobExecutor<br/>(stamps child AIPerfJob,<br/>waits for terminal phase)"]
        run["aiperf workload runs:<br/>SystemController +<br/>Worker / TimingManager /<br/>CreditIssuer / DatasetManager /<br/>RecordProcessor / RecordsManager"]
        rr["RunResult"]
    end

    subgraph AGG["6. aggregate"]
        sa["SweepAnalyzer.compute<br/>group by variation_values"]
        out["sweep_aggregate/<br/>profile_export_aiperf_sweep.{json,csv}<br/>(best_configurations, pareto_optimal,<br/>per_combination_metrics)"]
    end

    yaml --> ac
    cr --> ac
    ac --> es --> bp --> mo --> br --> re
    re --> loc
    re --> k8s
    loc --> run
    k8s --> run
    run --> rr
    rr --> sa --> out

    style INPUT fill:transparent,stroke:#1976d2,stroke-width:2px
    style CFG fill:transparent,stroke:#2e7d32,stroke-width:2px
    style PLAN fill:transparent,stroke:#ef6c00,stroke-width:2px
    style ORCH fill:transparent,stroke:#6a1b9a,stroke-width:2px
    style EXEC fill:transparent,stroke:#00838f,stroke-width:2px
    style AGG fill:transparent,stroke:#f9a825,stroke-width:2px
```

The flow is identical regardless of where it runs. Local: the subprocess at step 5 forks on the same machine. Cluster sweep: the orchestrator at step 4 lives inside a sweep-controller pod, and step 5 stamps a child `AIPerfJob` (which is itself a `JobSet` with a controller pod + worker pods + LLM endpoint).

## What happens between runs (per-cell loop)

A "cell" is one `(variation, trial)` slot. Inside a cell, an `ExecutionStrategy` decides whether to keep going. `FixedTrialsStrategy` stops after M trials. `AdaptiveStrategy` keeps going until its `ConvergenceCriterion` is met (or a hard cap). Around each `executor.execute(run)`, the orchestrator threads cancel-checking, sweep-wide failure thresholds, and inter-run cooldowns.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 60, 'padding': 14, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart TD
    enter(["enter cell<br/>(variation v, fresh strategy)"])
    cont{"strategy.<br/>should_continue<br/>(cell_results)?"}
    cancel{"cancel_check()?"}
    nextcfg["strategy.get_next_config(cfg, prior)<br/>(handles disable_warmup_after_first)"]
    build["build BenchmarkRun<br/>(label, artifact_dir,<br/>random_seed via variation_seeds)"]
    exec["await executor.execute(run)<br/>← BIG WAIT: subprocess or<br/>child AIPerfJob to terminal"]
    stamp["stamp variation metadata<br/>on RunResult; append to<br/>cell_results + all_results"]
    fail{"failure threshold<br/>exceeded?<br/>(plan.failure_policy)"}
    cooldown["if more trials coming:<br/>asyncio.sleep(<br/>strategy.get_cooldown_seconds())"]
    exit_done(["cell complete<br/>(strategy stopped)"])
    exit_cancel(["abort sweep<br/>(cancelled or<br/>failure-threshold tripped)"])

    enter --> cont
    cont -->|no| exit_done
    cont -->|yes| cancel
    cancel -->|yes| exit_cancel
    cancel -->|no| nextcfg
    nextcfg --> build --> exec --> stamp --> fail
    fail -->|yes| exit_cancel
    fail -->|no| cooldown
    cooldown --> cont

    style exit_done fill:#e8f5e9,stroke:#2e7d32
    style exit_cancel fill:#ffebee,stroke:#c62828
    style exec fill:#fff3e0,stroke:#ef6c00
```

The strategy is fresh per cell in INDEPENDENT mode, so adaptive trial-convergence resets between variations. In REPEATED mode there's only one trial per cell — the "outer trial loop" replays the whole grid.

## REPEATED vs INDEPENDENT — loop nesting

Two ways to interleave variations and trials. `sweep.iteration_order` picks; default is REPEATED.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 60, 'padding': 14, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart LR
    subgraph IND["INDEPENDENT — variations outer, trials inner"]
        i_outer["for variation v in 0..N:"]
        i_cool1["if v &gt; 0: inter-variation<br/>cooldown (sweep.cooldown_seconds)"]
        i_strat["fresh strategy per cell<br/>(adaptive convergence is<br/>cell-local)"]
        i_inner["per-cell trial loop<br/>(see above)"]
        i_outer --> i_cool1 --> i_strat --> i_inner --> i_outer
    end

    subgraph REP["REPEATED — trials outer, variations inner (default)"]
        r_outer["for trial t in 0..M:"]
        r_inner["for variation v in 0..N:"]
        r_strat["per-variation strategy<br/>(reused across trials,<br/>growing prior_results)"]
        r_one["exactly one run per cell<br/>(strategy not in trial-loop)"]
        r_cool_v["if v+1 &lt; N: inter-variation<br/>cooldown within trial"]
        r_cool_t["if t+1 &lt; M:<br/>inter-trial cooldown"]
        r_outer --> r_inner --> r_strat --> r_one --> r_cool_v --> r_inner
        r_inner -.trial done.-> r_cool_t --> r_outer
    end

    style IND fill:transparent,stroke:#1976d2,stroke-width:2px
    style REP fill:transparent,stroke:#ef6c00,stroke-width:2px
```

Why two modes? REPEATED interleaves trials across variations so transient cluster effects (warm caches, thermal drift) hit every variation similarly — better for cross-variation comparison. INDEPENDENT runs one variation to completion before moving on — required for convergence-based adaptive trials, since a strategy needs to observe all of one cell's results in sequence.

## Adaptive outer loop (ask / tell)

Adaptive search is the same pipeline with one swap: instead of "expand a fixed grid into N configs up front," the planner *generates* configs one at a time, learning from each result.

- The sweep block is `AdaptiveSearchSweep` (`type: adaptive_search`) instead of `GridSweep` / `ScenarioSweep`.
- `BenchmarkPlan.configs` starts with one seed config; the planner extends it as it asks.
- `MultiRunOrchestrator` dispatches to `execute_adaptive_search`, which runs `planner.ask() → execute trials → planner.tell(results)` until `planner.ask()` returns `None` (or cancellation / abort).
- Four planner plugins ship: `BayesianSearchPlanner` (curated Optuna+BoTorch preset, subclass of `OptunaSearchPlanner`), `OptunaSearchPlanner` (Optuna TPE / GP / BoTorch backends), `MonotonicSLASearchPlanner` (1D probe + bisection), and `SmoothIsotonicSLAPlanner` (1D PAVA + PCHIP isotonic regression).
- Optional `search_recipe` plugins build the whole `AdaptiveSearchSweep` from a higher-level recipe (e.g. `max-concurrency-under-sla`, `prefill-ttft-curve`).
- An optional `post_process` handler (`degradation_knee_detect`, `ttft_curve_fit`, `itl_surface_fit`, `sla_breach_knee`) runs after the final iteration.
- Cluster sweeps and adaptive search compose naturally: the sweep-controller pod's orchestrator drives the planner, and each `ask()` becomes a stamped child `AIPerfJob`.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 60, 'padding': 14, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart TD
    start(["execute_adaptive_search<br/>(planner pre-built upstream)"])
    cancel{"cancel_check()?"}
    ask["proposal = planner.ask()"]
    none{"proposal is None?"}
    converged(["write final<br/>search_history.json,<br/>return all_results"])
    run["_run_independent_cell<br/>(M trials inner — same per-cell<br/>loop as INDEPENDENT)"]
    tell["planner.tell(variation,<br/>cell_results)<br/>(filter by SLA, compute<br/>objective scalar, plateau /<br/>patience / max-iter check)"]
    hist["write_search_history<br/>(incremental, includes<br/>boundary_summary if<br/>planner has it)"]
    abort{"cell aborted?<br/>(cancelled / failure threshold)"}
    halt(["halt search,<br/>return partial results"])

    start --> cancel
    cancel -->|yes| halt
    cancel -->|no| ask --> none
    none -->|yes| converged
    none -->|no| run --> tell --> hist --> abort
    abort -->|yes| halt
    abort -->|no| cancel

    style start fill:#fff3e0,stroke:#ef6c00
    style converged fill:#e8f5e9,stroke:#2e7d32
    style halt fill:#ffebee,stroke:#c62828
```

Each iteration adds one `SearchIteration` to `planner.history()`. Convergence terminates the loop via `planner.ask()` returning `None`; the reason (plateau / improvement-patience / max-iterations) comes from `planner.convergence_reason()`. `search_history.json` is rewritten after every iteration so a crashed sweep still has a usable trail.

## Fan-out math

The cardinality of any sweep is `N variations × M trials = N×M cells`. Where N and M come from depends on the path.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 70, 'padding': 14, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart LR
    cfg["1 × AIPerfConfig"]

    subgraph FANOUT["expansion (1 → N)"]
        grid["GridSweep:<br/>cartesian product of<br/>sweep.variables<br/>N = ∏ |dim|"]
        scen["ScenarioSweep:<br/>N = len(runs[])"]
        adapt["AdaptiveSearchSweep:<br/>N = 1 seed (grows<br/>by 1 per planner.ask)"]
    end

    vars["N × SweepVariation<br/>+ N × BenchmarkConfig"]

    subgraph TRIALS["trials per variation"]
        m_fixed["M = multi_run.num_runs<br/>(FixedTrialsStrategy)"]
        m_conv["M = until convergence<br/>(AdaptiveStrategy + ConvergenceCriterion,<br/>capped by num_runs)"]
    end

    cells["N × M × BenchmarkRun<br/>= N × M × RunResult"]
    agg["1 × sweep_aggregate/<br/>profile_export_aiperf_sweep<br/>.{json,csv}<br/>(group by variation_values)"]

    cfg --> grid
    cfg --> scen
    cfg --> adapt
    grid --> vars
    scen --> vars
    adapt --> vars
    vars --> m_fixed
    vars --> m_conv
    m_fixed --> cells
    m_conv --> cells
    cells --> agg

    style FANOUT fill:transparent,stroke:#1976d2,stroke-width:2px
    style TRIALS fill:transparent,stroke:#ef6c00,stroke-width:2px
```

For adaptive search, `N` is the iteration count: bounded above by `max_iterations`, possibly less if the planner converges early. `M` (trials per iteration) still applies — adaptive runs M trials per planner-proposed point, then `tell()`s the planner the aggregate.

## K8s fan-out — what gets stamped where

`AIPerfSweep` is the fan-out apex. The operator never stamps benchmark workloads directly — it spawns one sweep-controller pod, and *that* pod's orchestrator stamps child `AIPerfJob` CRs. Each child `AIPerfJob` is itself a fan-out: the operator sees it and produces a `JobSet` with a controller pod + N worker pods.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 65, 'padding': 14, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart TB
    sweep["1 × AIPerfSweep CR"]

    subgraph OP["operator (kopf)"]
        op_create["sweep_create.handle<br/>validates spec,<br/>computes cardinality,<br/>provisions RBAC"]
        op_js["1 × sweep-controller JobSet<br/>(replica=1)"]
    end

    subgraph SCPOD["sweep-controller pod (1)"]
        sc["container: sweep-controller<br/>MultiRunOrchestrator +<br/>K8sChildJobExecutor"]
        rs["container: results-sidecar<br/>(serves /results over HTTP)"]
    end

    subgraph CHILDREN["children (N × M)"]
        kids["N × M child AIPerfJob CRs<br/>(or 1-at-a-time for adaptive)<br/>name: &lt;sweep&gt;-v&lt;NN&gt;[-t&lt;N&gt;]<br/>spec.benchmark = variation.cfg<br/>spec.sweep = null"]
    end

    subgraph PERCHILD["per child AIPerfJob (×N×M)"]
        op2["operator handler<br/>create.on_create"]
        cjs["1 × JobSet"]
        ctrl["1 × controller pod<br/>+ results-sidecar<br/>+ event-bus sidecar"]
        wks["K × worker pods<br/>(K from spec.deployment.workers)"]
        llm[("LLM endpoint")]
    end

    fetch["operator pulls aggregate<br/>via results-sidecar HTTP<br/>(/results is emptyDir{} —<br/>lost when pod cleans up)"]
    final["operator PVC:<br/>&lt;base&gt;/&lt;ns&gt;/sweeps/&lt;sweep&gt;/&lt;epoch&gt;/<br/>sweep_aggregate/"]

    sweep --> op_create --> op_js --> sc
    op_js --> rs
    sc -->|stamps| kids
    kids --> op2 --> cjs --> ctrl
    cjs --> wks
    wks -->|HTTP/SSE| llm
    ctrl --> wks
    sc -->|aggregate written<br/>to /results emptyDir| fetch
    fetch --> final

    style OP fill:transparent,stroke:#00838f,stroke-width:2px
    style SCPOD fill:transparent,stroke:#ef6c00,stroke-width:2px
    style CHILDREN fill:transparent,stroke:#1976d2,stroke-width:2px
    style PERCHILD fill:transparent,stroke:#6a1b9a,stroke-width:2px
```

Two things that surprise people:

1. The sweep-controller pod's `/results` is `emptyDir{}`, not a PVC. The operator must harvest the aggregate via the results-sidecar's HTTP API before the JobSet's TTL reaper deletes the pod, or the data is gone. Fetched on `status.aggregation.phase` flipping to `Complete`.
2. Each child `AIPerfJob` runs on its own JobSet with its own controller + workers + LLM. So an N×M sweep with K workers per benchmark uses N×M×(1+K) pods plus the one sweep-controller pod. Plan capacity accordingly.

## Where it lives

| Concept | File |
|---|---|
| Envelope + body (`AIPerfConfig`, `BenchmarkConfig`) | `src/aiperf/config/config.py` |
| Multi-run / trials (`MultiRunConfig`) | `src/aiperf/config/sweep/multi_run.py` |
| Sweep variants + expansion | `src/aiperf/config/sweep/{config,expand}.py` |
| Plan loader (CLI/YAML → plan) | `src/aiperf/config/loader/plan.py` |
| Orchestrator | `src/aiperf/orchestrator/orchestrator.py` |
| Executors | `src/aiperf/orchestrator/{executor,local_executor}.py`, `src/aiperf/sweep_controller/k8s_executor.py` |
| Aggregation | `src/aiperf/orchestrator/aggregation/sweep.py` |
| Search planners + recipes | `src/aiperf/orchestrator/search_planner/`, `src/aiperf/search_recipes/` |
| Operator (kopf handlers) | `src/aiperf/operator/main.py`, `src/aiperf/operator/handlers/` |
| Sweep-controller pod entrypoint | `src/aiperf/sweep_controller/main.py` |

For the full diagrams (kopf decorators, status field shapes, child-name patterns, plugin category list), see [sweep-orchestrator-flow.md](sweep-orchestrator-flow.md).
