<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweep / Job / Orchestrator flow

End-to-end view of how a YAML config or a CRD becomes an `AIPerfConfig` envelope, expands into a `BenchmarkPlan`, and is executed by `MultiRunOrchestrator` against a backend `RunExecutor`. Multiple zoom levels — pick whichever matches what you're trying to understand.

> **Schema-2.0 note (May 2026).** Some inset diagrams below still label
> fields with their pre-redesign names: `parameter_sweep_mode` /
> `parameter_sweep_*` (now `sweep.iteration_order` / `sweep.cooldown_seconds`
> / `sweep.same_seed`), `MultiRunConfig.adaptive_search` (now
> `AIPerfConfig.sweep` with `type: adaptive_search`), and
> `BenchmarkPlan.adaptive_search` (now `BenchmarkPlan.sweep`, accessed via
> `BenchmarkPlan.is_adaptive_search`). The high-level flow — `AIPerfConfig`
> envelope -> `BenchmarkPlan` -> `MultiRunOrchestrator` dispatch -> backend
> executor — is unchanged.

The same `BenchmarkPlan` / `MultiRunOrchestrator` / `RunExecutor` machinery powers both local (`aiperf profile`) and Kubernetes (`AIPerfJob`, `AIPerfSweep`) execution. The only swap is which `RunExecutor` implementation the controller picks.

## 1. 30,000 ft — what happens, period

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart LR
    A["user input<br/>YAML / CLI / CRD"] --> B["AIPerfConfig<br/>(envelope)"]
    B --> C["BenchmarkPlan<br/>configs[N] × trials"]
    C --> D["execute each cell<br/>(MultiRunOrchestrator)"]
    D --> E["results +<br/>sweep aggregate"]
```

## 2. 10,000 ft — local vs K8s, same core

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    user["user"]

    subgraph INPUTS["inputs"]
        cli["aiperf profile -c config.yaml"]
        kjob["kubectl apply<br/>kind: AIPerfJob"]
        ksweep["kubectl apply<br/>kind: AIPerfSweep"]
    end

    subgraph CORE["plan / orchestration"]
        plan["BenchmarkPlan"]
        orch["MultiRunOrchestrator"]
    end

    subgraph EXEC["executors"]
        local["LocalSubprocessExecutor"]
        k8sx["K8sChildJobExecutor<br/>(in sweep_controller pod)"]
    end

    subgraph RUNTIME["runtime"]
        sysctrl["SystemController + services"]
    end

    subgraph RESULTS["results"]
        out["RunResult → SweepAnalyzer →<br/>profile_export_aiperf_sweep.{json,csv}"]
    end

    user --> cli --> plan
    user --> kjob --> plan
    user --> ksweep --> plan

    plan --> orch
    orch --> local
    orch --> k8sx
    local --> sysctrl
    k8sx -->|"stamps child AIPerfJob<br/>(JobSet → controller pod)"| sysctrl
    sysctrl --> out

    style INPUTS fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
    style CORE fill:transparent,stroke:#2e7d32,stroke-width:2px,stroke-dasharray:0
    style EXEC fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style RUNTIME fill:transparent,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray:0
    style RESULTS fill:transparent,stroke:#f9a825,stroke-width:2px,stroke-dasharray:0
```

## 3. 5,000 ft — three lanes (local, single AIPerfJob, AIPerfSweep)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    subgraph LOCAL["local"]
        L1["YAML / CLI"] --> L2["AIPerfConfig"] --> L3["BenchmarkPlan"]
        L3 --> L4["MultiRunOrchestrator<br/>+ LocalSubprocessExecutor"]
        L4 --> L5["controller subprocess"]
        L5 --> L6["RunResults → aggregate"]
    end

    subgraph SINGLE["AIPerfJob"]
        S1["AIPerfJob CR"] --> S2["handlers/create.py"]
        S2 --> S3["JobSet"] --> S4["controller pod<br/>aiperf profile<br/>(single-config plan)"]
        S4 --> S5["AIPerfJob.status"]
    end

    subgraph SWEEP["AIPerfSweep"]
        W1["AIPerfSweep CR"] --> W2["handlers/sweep/create.py<br/>(always spawns sweep-controller)"]
        W2 --> W3["sweep_controller pod<br/>MultiRunOrchestrator<br/>+ K8sChildJobExecutor"]
        W3 -->|stamps N| W4["child AIPerfJob CRs"]
        W4 --> W5["each: handlers/create.py<br/>→ JobSet → controller pod"]
        W5 --> W6["AIPerfSweep.status<br/>(rollup)"]
    end

    style LOCAL fill:transparent,stroke:#2e7d32,stroke-width:2px,stroke-dasharray:0
    style SINGLE fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
    style SWEEP fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
```

## 4. Sub-flow — config layer (YAML/CLI → BenchmarkPlan)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    subgraph INPUTS["inputs"]
        yaml["YAML file"]
        cli["CLI flags (cyclopts)"]
    end

    subgraph PARSE["parse"]
        loader["load_config_from_string<br/>(reject flat shape, env-var sub)"]
        v1["v1 UserConfig<br/>(no validators)"]
        conv["CLIConfig → AIPerfConfig converter (config/flags/converter.py)<br/>• _assemble_optional<br/>• _apply_recipe_sweep_parameters<br/>• _promote_magic_lists_to_sweep_block<br/>• _wrap_under_envelope<br/>(envelope keys: sweep, multi_run,<br/>variables, random_seed, benchmark)"]
    end

    subgraph VALIDATE["validate"]
        envelope["AIPerfConfig envelope<br/>schema_version / benchmark / sweep /<br/>multi_run / variables / random_seed"]
    end

    subgraph BUILD["expand / render"]
        expand["expand_sweep (config/sweep/expand.py)<br/>grid: cartesian over sweep.variables<br/>scenarios: deep-merge runs[i].benchmark<br/>adaptive_search: 1-element placeholder"]
        jinja["per-variation Jinja render<br/>(variables overlay)"]
        bp["BenchmarkPlan<br/>configs[N], variations[N],<br/>variation_seeds[N], trials,<br/>multi_run, sweep, failure_policy"]
    end

    yaml --> loader
    cli --> v1
    v1 --> conv
    loader --> envelope
    conv --> envelope

    envelope --> jinja
    envelope --> expand
    expand --> jinja
    jinja --> bp

    style INPUTS fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
    style PARSE fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style VALIDATE fill:transparent,stroke:#2e7d32,stroke-width:2px,stroke-dasharray:0
    style BUILD fill:transparent,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray:0
```

## 5. Sub-flow — orchestrator iteration

`cli_runner.run_benchmark` peels off single-run plans (`plan.is_single_run`) before the orchestrator is constructed; only multi-run plans reach `MultiRunOrchestrator.execute`. Inside `execute()`, dispatch is two-way: adaptive-search vs. grid/scenarios. Grid/scenarios further branch on `_plan_iteration_order(plan)` which reads `plan.sweep.iteration_order` (REPEATED default, or INDEPENDENT) — there is no `parameter_sweep_mode` field on `BenchmarkPlan`.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    cli["cli_runner.run_benchmark"] --> single{"plan.is_single_run?"}
    single -->|yes| sgl["_run_single_benchmark<br/>(LocalSubprocessExecutor.execute)"]
    single -->|no| orch["MultiRunOrchestrator.execute<br/>(plan, executor,<br/>*, cancel_check, search_planner)"]

    subgraph DECIDE["dispatch (inside execute)"]
        ad{"plan.is_adaptive_search?"}
        order{"_plan_iteration_order(plan)<br/>(reads sweep.iteration_order)"}
    end

    orch --> ad
    ad -->|yes| bo["execute_adaptive_search(plan, executor, planner)<br/>BO outer loop: ask → execute trials → tell"]
    ad -->|no| order

    subgraph MODES["modes (variations × trials cells)"]
        rep["_execute_repeated<br/>trials OUTER × variations INNER<br/>&lt;base&gt;/profile_runs/trial_NNNN/<br/>&lt;variation&gt;/profile_runs/run_NNNN/"]
        ind["_execute_independent<br/>variations OUTER × trials INNER<br/>&lt;base&gt;/&lt;variation&gt;/<br/>profile_runs/run_NNNN/"]
    end

    order -->|REPEATED| rep
    order -->|INDEPENDENT| ind

    subgraph CELL["per cell"]
        cell["build BenchmarkRun(<br/>benchmark_id, cfg, variation, trial,<br/>artifact_dir, label, random_seed, resolved)<br/>→ executor.execute(run) → RunResult"]
    end

    bo --> cell
    rep --> cell
    ind --> cell
    sgl --> cell

    style DECIDE fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style MODES fill:transparent,stroke:#2e7d32,stroke-width:2px,stroke-dasharray:0
    style CELL fill:transparent,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray:0
```

## 6. Sub-flow — RunExecutor backends

`RunExecutor` is a 2-method ABC: `execute(run) -> RunResult` and `derive_id(plan, var_idx, trial) -> str`. Local picks a fresh hex id; K8s computes the child-job name and reuses it as the id.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    run["BenchmarkRun"] --> abc["RunExecutor (ABC)<br/>execute(run) → RunResult<br/>derive_id(plan, var_idx, trial) → str"]

    subgraph LOCAL_BACKEND["local backend"]
        loc["LocalSubprocessExecutor"]
        sub["aiperf.orchestrator.subprocess_runner<br/>(SystemController + services)<br/>argv: [python -u -m … config_file]"]
        jr["artifacts on disk:<br/>profile_export_&lt;prefix&gt;.json{,.zst}<br/>+ records.parquet"]
        rrl["RunResult(label, success,<br/>summary_metrics, error,<br/>artifacts_path, variation_label,<br/>variation_values, trial_index)"]
    end

    subgraph K8S_BACKEND["k8s backend (inside sweep_controller pod)"]
        k8s["K8sChildJobExecutor"]
        child["child AIPerfJob CR<br/>name: &lt;sweep&gt;-v&lt;NN&gt; (zero-pad-2)<br/>±&lt;-t&lt;N&gt;&gt; iff trials&gt;1 OR convergence<br/>ownerRef: AIPerfSweep (controller=true)<br/>labels: sweep, sweep-uid,<br/>sweep-run-epoch, variation-index,<br/>variation-label, trial-index<br/>spec.benchmark: run.cfg.model_dump<br/>spec.sweep: None (single-variation)"]
        wait["_wait_until_terminal<br/>poll every 5s for status.phase ∈<br/>{Completed, Succeeded, Failed,<br/>Cancelled, PartiallyFailed}"]
        sum["_pull_summary_metrics<br/>1) child.status.summary<br/>2) fallback: HTTP GET<br/>profile_export_aiperf.json<br/>via results-sidecar (ProgressClient)"]
        rrk["RunResult"]
    end

    abc --> loc
    abc --> k8s

    loc -->|spawn subprocess| sub
    sub --> jr
    jr --> rrl

    k8s -->|create child CR| child
    child --> wait --> sum --> rrk

    style LOCAL_BACKEND fill:transparent,stroke:#2e7d32,stroke-width:2px,stroke-dasharray:0
    style K8S_BACKEND fill:transparent,stroke:#00838f,stroke-width:2px,stroke-dasharray:0
```

`needs_trial_suffix(trials, has_convergence)` (`sweep_controller/_naming.py`) is the source of truth on `-t<N>`: it fires whenever `trials > 1` **or** `multi_run.convergence` is set. INDEPENDENT mode does not, by itself, change the naming rule.

## 7. Sub-flow — inside one BenchmarkRun (service mesh)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart LR
    subgraph CONTROL["control"]
        sys["SystemController<br/>(BaseService)"]
        bus(("ZMQ bus<br/>inproc/IPC"))
    end

    subgraph LOAD["load gen"]
        ci["CreditIssuer"]
        tm["TimingManager"]
        dm["DatasetManager"]
        w["Worker(s)"]
    end

    subgraph DATA["data"]
        rm["RecordsManager"]
        res["ResultsManager"]
        art["artifacts:<br/>profile_export_aiperf.{json,csv}<br/>profile_export_aiperf_records.parquet"]
    end

    llm[("LLM endpoint")]

    sys --> bus
    bus --> ci
    bus --> tm
    bus --> dm
    bus --> w
    bus --> rm
    bus --> res

    ci -->|credits| tm
    tm -->|gated turns| w
    dm -->|conversation payload| w
    w -->|HTTP / SSE| llm
    llm -->|chunks| w
    w -->|RequestRecord| rm
    rm -->|aggregated| res
    res --> art

    style CONTROL fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
    style LOAD fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style DATA fill:transparent,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray:0
```

## 8. Sub-flow — K8s AIPerfSweep stamping

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    cr["AIPerfSweep CR"]

    subgraph OPERATOR["operator (kopf)"]
        h["handlers/sweep/create.py<br/>@kopf.on.create(aiperfsweeps)"]
        v["AIPerfSweepSpec.model_validate"]
        card["_compute_cardinality<br/>adaptive: max_iterations × num_runs<br/>grid/scenarios: expand_sweep × num_runs"]
        rbac["_provision_rbac<br/>SA + Role + RoleBinding<br/>(aiperfjobs CRUD, this sweep<br/>status patch, events)"]
        sj["_create_sweep_controller_jobset<br/>JobSet aiperf-&lt;name&gt;,<br/>1 replica, restartPolicy=OnFailure,<br/>2 containers + 2 emptyDir volumes"]
    end

    subgraph SWPOD["sweep-controller pod (always spawned)"]
        ctr["container: sweep-controller<br/>python -m aiperf.sweep_controller.main"]
        rs["container: results-sidecar<br/>python -m aiperf.kubernetes.results_sidecar<br/>(serves /results over HTTP)"]
        bp["build_plan_from_sweep(cr)<br/>→ BenchmarkPlan (configs[N]<br/>or 1-element seed for BO)"]
        sp["_build_search_planner(plan)<br/>None for grid; BayesianSearchPlanner /<br/>MonotonicSLASearchPlanner for adaptive"]
        mo["MultiRunOrchestrator(/results).execute(<br/>plan, K8sChildJobExecutor,<br/>cancel_check, search_planner)"]
        stamp1["grid/scenarios:<br/>K8sChildJobExecutor.execute<br/>called per (variation, trial)"]
        stamp2["adaptive:<br/>execute_adaptive_search drives<br/>planner.ask → execute trials → tell;<br/>one variation at a time"]
    end

    subgraph CHILDREN["children"]
        children["child AIPerfJob CRs<br/>&lt;sweep&gt;-v00, -v01, … ±&lt;-tN&gt;<br/>spec.sweep=null, spec.benchmark=variation"]
    end

    cr --> h
    h --> v --> card --> rbac --> sj
    sj --> ctr
    sj --> rs
    ctr --> bp --> sp --> mo

    mo -->|grid / scenarios| stamp1
    mo -->|adaptive_search| stamp2

    stamp1 --> children
    stamp2 --> children

    style OPERATOR fill:transparent,stroke:#00838f,stroke-width:2px,stroke-dasharray:0
    style SWPOD fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style CHILDREN fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
```

## 9. Sub-flow — K8s AIPerfJob → controller pod

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    job["AIPerfJob CR<br/>(user-authored or child of sweep)"]

    subgraph OPERATOR["operator (kopf wiring in operator/main.py)"]
        c["@kopf.on.create(aiperfjobs)<br/>→ handlers/create.py:on_create"]
        sp["AIPerfJobSpecConverter.to_aiperf_config<br/>→ to_deployment_config<br/>→ KubernetesDeployment.get_jobset_spec"]
        rbac2["_create_rbac + _create_configmap<br/>(propagation delay)"]
        mon["@kopf.timer(aiperfjobs,<br/>interval=10s, initial_delay=5s)<br/>→ handlers/monitor.py:monitor_progress"]
        jt["@kopf.on.field(jobsets,<br/>field=status.conditions)<br/>→ jobset_terminal.py<br/>(annotates AIPerfJob on<br/>Completed/True for fast path)"]
        pr["@kopf.on.event(v1/pods,<br/>labels={jobset-name PRESENT})<br/>→ pod_restarts.py<br/>(event-based; field-based<br/>would need pods:patch RBAC)"]
        bc["@kopf.on.update(field=annotations,<br/>aiperf.nvidia.com/benchmark-complete=true)<br/>→ on_benchmark_complete"]
    end

    subgraph PODS["JobSet → pods (jobset.sigs.k8s.io)"]
        js["JobSet aiperf-&lt;name&gt;<br/>replicatedJobs: controller + workers"]
        ctrl["controller Pod (replica 1)<br/>aiperf profile<br/>AIPERF_OPERATOR_MANAGED=1<br/>(in-process grid sweep rejected;<br/>BO/adaptive permitted)"]
        wpod["worker Pods (replica N)"]
        sc["controller sidecars:<br/>results-sidecar, event-bus"]
    end

    subgraph PERSIST["persist + read-back"]
        art["artifacts on PVC<br/>+ .aiperf_results_ready.json gate"]
        sapi["results-sidecar HTTP<br/>/api/progress, /api/workers,<br/>/api/metrics, /api/server-metrics,<br/>/api/results/list, /api/results/files/&lt;p&gt;,<br/>/api/shutdown"]
    end

    llm[("LLM endpoint")]

    job --> c --> sp --> rbac2 --> js
    js --> ctrl
    js --> wpod
    ctrl --> sc

    ctrl -->|ZMQ message bus| wpod
    wpod -->|HTTP / SSE| llm

    ctrl --> art --> sapi
    sapi -.HTTP poll.-> mon
    js -.condition flips.-> jt
    js -.pod restart events.-> pr
    jt --> bc

    style OPERATOR fill:transparent,stroke:#00838f,stroke-width:2px,stroke-dasharray:0
    style PODS fill:transparent,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:0
    style PERSIST fill:transparent,stroke:#f9a825,stroke-width:2px,stroke-dasharray:0
```

## 10. Sub-flow — status & rollup

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TD
    pod["controller pod"]

    subgraph SOURCES["status sources (kopf wiring)"]
        mon["handlers/monitor.py:monitor_progress<br/>@kopf.timer(aiperfjobs,<br/>interval=10s, initial_delay=5s)"]
        jt["handlers/jobset_terminal.py<br/>@kopf.on.field(jobsets,<br/>field=status.conditions)"]
        pr["handlers/pod_restarts.py<br/>@kopf.on.event(v1/pods,<br/>labels={jobset-name PRESENT})"]
        cph["handlers/sweep/child_rollup.py<br/>@kopf.on.field(aiperfjobs,<br/>field=status.phase)"]
    end

    subgraph PERJOB["AIPerfJob.status"]
        js["phase, subPhase, currentPhase<br/>phases[name].isRequestsComplete /<br/>isRecordsComplete / sendingComplete<br/>liveMetrics, serverMetrics<br/>conditions: Complete, Failed<br/>(batchv1.Job-shaped:<br/>Complete=True iff phase=Completed<br/>AND ResultsAvailable=True)"]
    end

    subgraph AGGREGATE["AIPerfSweep rollup"]
        rollup["child_rollup.on_child_phase_transition<br/>filters by sweep-run-epoch label;<br/>terminal phase →<br/>_child_runs.append_run_entry<br/>(threshold 1500 → runsTruncated)"]
        ss["AIPerfSweep.status<br/>completedRuns, failedRuns,<br/>runStates: {pending, running,<br/>completed, failed, cancelled}<br/>currentChildRef: {name, index, label}<br/>lastChildEvent: {name, phase}<br/>runs[] slim summary,<br/>runsTruncated: {total, included, fetchURL}"]
        adv["phase advance to Aggregating<br/>(JSON-patch test/replace, gated by<br/>completed+failed+cancelled ≥ maxTotalRuns)"]
        agf["@kopf.on.field(aiperfsweeps,<br/>field=status.aggregation.phase,<br/>new=Complete) →<br/>_aggregate_fetch.fetch_sweep_aggregate_to_disk<br/>(ProgressClient.download_all_results<br/>from results-sidecar, since /results<br/>is emptyDir{} on the sweep-controller pod)"]
        exp["operator PVC:<br/>&lt;base&gt;/&lt;ns&gt;/sweeps/&lt;sweep&gt;/&lt;epoch&gt;/<br/>sweep_aggregate/<br/>profile_export_aiperf_sweep.{json,csv}"]
    end

    pod -->|HTTP poll| mon
    pod -.JobSet conditions.-> jt
    pod -.pod restart events.-> pr
    pod -.phase transitions.-> cph

    mon --> js
    jt --> js
    pr --> js

    cph --> rollup --> ss
    ss --> adv --> agf --> exp

    style SOURCES fill:transparent,stroke:#00838f,stroke-width:2px,stroke-dasharray:0
    style PERJOB fill:transparent,stroke:#1976d2,stroke-width:2px,stroke-dasharray:0
    style AGGREGATE fill:transparent,stroke:#f9a825,stroke-width:2px,stroke-dasharray:0
```

## 11. Class / module map

```mermaid
%%{init: {'class': {'titleTopMargin': 25}, 'themeVariables': {'fontSize': '15px'}}}%%
classDiagram
    class AIPerfConfig {
        schema_version: Literal["2.0"]
        benchmark: BenchmarkConfig
        sweep: SweepConfig | None
        multi_run: MultiRunConfig
        variables: dict[str, Any]
        random_seed: int | None
    }
    class BenchmarkConfig {
        models, endpoint, datasets, phases
        artifacts, slos, tokenizer
        gpu_telemetry, server_metrics
        runtime, logging, metrics, accuracy
    }
    class SweepConfig {
        <<discriminated union>>
        type: grid | scenarios | adaptive_search
    }
    class GridSweep {
        type: "grid"
        variables: dict[str, list]
        iteration_order: REPEATED | INDEPENDENT
        same_seed: bool
        cooldown_seconds, sla_filters,
        post_process
    }
    class ScenarioSweep {
        type: "scenarios"
        runs: list[dict]
        iteration_order, same_seed,
        cooldown_seconds, sla_filters,
        post_process
    }
    class AdaptiveSearchSweep {
        type: "adaptive_search"
        algorithm: "bayes"
        planner: SearchPlannerType
        search_space: list[SearchSpaceDimension]
        objective: AdaptiveObjective
        max_iterations, n_initial_points
        plateau_window, plateau_threshold
        improvement_patience, random_seed
        recipe_name, optuna_sampler
        monotonic_stability_trials
        constraint_mode
        cooldown_seconds, sla_filters,
        post_process
    }
    class SweepVariation {
        index: int
        label: str
        values: dict[str, Any]
    }
    class MultiRunConfig {
        num_runs: int
        cooldown_seconds: float
        confidence_level: float
        set_consistent_seed: bool
        disable_warmup_after_first: bool
        convergence: ConvergenceConfig | None
    }
    class ConvergenceConfig {
        metric, stat, mode,
        threshold, min_runs
    }

    class BenchmarkPlan {
        configs: list[BenchmarkConfig]
        variations: list[SweepVariation]
        variation_seeds: list[int | None]
        trials: int
        cooldown_seconds, confidence_level
        set_consistent_seed
        disable_warmup_after_first
        random_seed, variables
        export_level, export_jsonl_file
        multi_run: MultiRunConfig
        sweep: SweepConfig | None
        failure_policy: FailurePolicy | None
        use_adaptive (prop)
        is_single_run / is_sweep /
        is_adaptive_search (props)
    }
    class BenchmarkRun {
        benchmark_id: str
        cfg: BenchmarkConfig
        variation: SweepVariation | None
        trial: int
        artifact_dir: Path
        label: str
        random_seed: int | None
        resolved: ResolvedConfig
    }
    class RunResult {
        label: str
        success: bool
        summary_metrics: dict[str, JsonMetricResult]
        error: str | None
        artifacts_path: Path | None
        variation_label: str
        variation_values: dict[str, Any]
        trial_index: int
    }

    class MultiRunOrchestrator {
        base_dir: Path
        execute(plan, executor, *,
        cancel_check, search_planner)
        execute_adaptive_search(plan,
        executor, planner, *, cancel_check)
    }
    class RunExecutor {
        <<abstract>>
        execute(run) RunResult
        derive_id(plan, var_idx, trial) str
    }
    class LocalSubprocessExecutor
    class K8sChildJobExecutor
    class SweepAnalyzer {
        compute(per_combination_stats,
        sweep_parameters, sla_filters?)
    }

    AIPerfConfig --> BenchmarkConfig : benchmark
    AIPerfConfig --> SweepConfig : sweep
    AIPerfConfig --> MultiRunConfig : multi_run
    MultiRunConfig --> ConvergenceConfig : convergence
    SweepConfig <|-- GridSweep
    SweepConfig <|-- ScenarioSweep
    SweepConfig <|-- AdaptiveSearchSweep

    BenchmarkPlan --> BenchmarkConfig : configs[]
    BenchmarkPlan --> SweepVariation : variations[]
    BenchmarkPlan --> MultiRunConfig : multi_run
    BenchmarkPlan --> SweepConfig : sweep
    BenchmarkRun --> BenchmarkConfig : cfg
    BenchmarkRun --> SweepVariation : variation

    MultiRunOrchestrator ..> BenchmarkPlan : reads
    MultiRunOrchestrator ..> BenchmarkRun : builds
    MultiRunOrchestrator ..> RunExecutor : delegates
    RunExecutor <|-- LocalSubprocessExecutor
    RunExecutor <|-- K8sChildJobExecutor
    RunExecutor ..> RunResult : returns

    SweepAnalyzer ..> RunResult : aggregates
```

## 12. Sequence — a sweep run end to end

```mermaid
%%{init: {'sequence': {'actorMargin': 60, 'boxMargin': 18, 'noteMargin': 12, 'messageMargin': 40}, 'themeVariables': {'fontSize': '16px'}}}%%
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as aiperf profile / kubectl
    participant Conv as v1→v2 converter
    participant Cfg as AIPerfConfig
    participant Plan as build_benchmark_plan
    participant Orch as MultiRunOrchestrator
    participant Exec as RunExecutor
    participant Sub as SystemController + services
    participant Agg as SweepAnalyzer

    U->>CLI: launch with config.yaml
    CLI->>Conv: parsed v1 UserConfig
    Conv->>Cfg: envelope-shaped dict<br/>(benchmark/sweep/...)
    Cfg->>Plan: validated AIPerfConfig
    Plan->>Plan: expand_sweep + per-variation Jinja
    Plan-->>CLI: BenchmarkPlan(configs[N], variations[N])
    CLI->>Orch: execute(plan, executor[, search_planner])

    loop for each (variation, trial)
        Orch->>Orch: build BenchmarkRun(cfg, variation, trial)
        Orch->>Exec: execute(run)
        Exec->>Sub: launch (subprocess or child AIPerfJob)
        Sub->>Sub: SystemController boots services on ZMQ bus<br/>CreditIssuer → TimingManager → Worker → LLM
        Sub-->>Exec: artifacts on disk + summary
        Exec-->>Orch: RunResult
    end

    Orch-->>CLI: list[RunResult]
    CLI->>Agg: aggregate_sweep_and_export
    Agg-->>U: sweep_aggregate/profile_export_aiperf_sweep.{json,csv}
```

## 13. Where to look in the code

| Concept | File |
|---|---|
| `AIPerfConfig` envelope, `BenchmarkConfig` body | `src/aiperf/config/config.py` |
| `BenchmarkPlan`, `BenchmarkRun`, `ResolvedConfig` | `src/aiperf/config/resolution/plan.py` |
| `MultiRunConfig`, `ConvergenceConfig` | `src/aiperf/config/sweep/multi_run.py` |
| `SweepConfig` union / `GridSweep` / `ScenarioSweep` / `AdaptiveSearchSweep` / `AdaptiveObjective` / `SweepVariation` | `src/aiperf/config/sweep/config.py` |
| `expand_sweep` (definition) | `src/aiperf/config/sweep/expand.py` (re-exported from the `sweep` package) |
| `SearchSpaceDimension`, `SLAFilter` | `src/aiperf/config/sweep/adaptive.py` |
| `PostProcessSpec`, `SearchRecipe`, `SearchRecipeContext`, `SearchRecipeOutput` | `src/aiperf/search_recipes/_base.py` |
| `PostProcessHandler` Protocol + built-ins | `src/aiperf/search_recipes/post_process.py` |
| `build_benchmark_plan` (load → plan) | `src/aiperf/config/loader/plan.py` |
| CLIConfig→AIPerfConfig converter | `src/aiperf/config/flags/converter.py` |
| `MultiRunOrchestrator` | `src/aiperf/orchestrator/orchestrator.py` |
| `RunExecutor` ABC + `RunResult` | `src/aiperf/orchestrator/executor.py`, `src/aiperf/orchestrator/models.py` |
| `LocalSubprocessExecutor` | `src/aiperf/orchestrator/local_executor.py` |
| Subprocess runner entry (`python -m`) | `src/aiperf/orchestrator/subprocess_runner.py` |
| `K8sChildJobExecutor`, `derive_child_name`, `needs_trial_suffix` | `src/aiperf/sweep_controller/{k8s_executor,_naming}.py` |
| `build_plan_from_sweep` (CR → plan, in-pod) | `src/aiperf/sweep_controller/plan_builder.py` |
| `sweep_controller` entrypoint | `src/aiperf/sweep_controller/main.py` |
| `SearchPlanner` ABC + `SearchIteration` | `src/aiperf/orchestrator/search_planner/base.py` |
| `BayesianSearchPlanner` / `MonotonicSLASearchPlanner` / `OptunaSearchPlanner` | `src/aiperf/orchestrator/search_planner/{bayesian,monotonic,optuna_planner}.py` |
| `parse_sla_filter`, `parse_search_space` | `src/aiperf/orchestrator/search_planner/parsing.py` |
| `SweepAnalyzer` + exporters | `src/aiperf/orchestrator/aggregation/sweep.py` |
| `aggregate_sweep_and_export` (file writer) | `src/aiperf/cli_runner/_sweep_aggregate.py` |
| `write_search_history` | `src/aiperf/exporters/search_history.py` |
| `run_benchmark` (single vs multi dispatch) + `_reject_in_process_sweep_under_operator` | `src/aiperf/cli_runner/` (`__init__.py`, `_multi_run.py`) |
| Operator kopf wiring | `src/aiperf/operator/main.py` |
| AIPerfJob create / monitor / completion handlers | `src/aiperf/operator/handlers/{create,monitor,completion}.py` |
| JobSet terminal-condition + pod-restart watchers | `src/aiperf/operator/handlers/{jobset_terminal,pod_restarts}.py` |
| AIPerfSweep handlers (stamp children, rollup, aggregate fetch, lifecycle) | `src/aiperf/operator/handlers/sweep/{create,child_rollup,_aggregate_fetch,lifecycle,_child_runs}.py` |
| Plugin registry + categories | `src/aiperf/plugin/{_registry,categories.yaml,metadata.py,plugins.py}` |

## 14. Plugin system — registration & lookup

Every extensible feature in AIPerf is a registered plugin: services, datasets, exporters, telemetry, convergence criteria, search planners, UIs. Built-ins live in `plugins.yaml`; third parties register via setuptools entry points and can override built-ins by priority.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart LR
    subgraph SOURCES["sources"]
        yaml["plugins.yaml<br/>(built-in)"]
        ep["setuptools<br/>entry points<br/>(third-party)"]
    end

    subgraph CORE["registry"]
        loader["plugin/_loader.py"]
        cats["categories.yaml<br/>(category schemas)"]
        meta["metadata.py<br/>(typed accessors)"]
        reg[("PluginRegistry<br/>(singleton)")]
    end

    subgraph LOOKUP["resolve at runtime"]
        gc["plugins.get_class(<br/>PluginType.X, name)"]
        gtm["get_typed_metadata(<br/>category, name)"]
        ent["PluginEntry<br/>cls + metadata"]
    end

    yaml --> loader
    ep --> loader
    cats --> loader
    loader --> reg
    reg --> gc
    reg --> gtm
    meta --> gtm
    gc --> ent
    gtm --> ent

    style SOURCES fill:transparent,stroke:#1976d2,stroke-width:2px
    style CORE fill:transparent,stroke:#2e7d32,stroke-width:2px
    style LOOKUP fill:transparent,stroke:#6a1b9a,stroke-width:2px
```

## 15. Plugin categories — grouped by domain

Each category has a Pydantic `metadata_class` and a typical interface (Protocol or ABC). New built-ins land as a `plugins.yaml` entry plus an implementation class; third-party wheels register an entry point and override by priority.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 70, 'padding': 16, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '16px'}}}%%
flowchart TB
    subgraph LOADGEN["load gen"]
        timing["timing_strategy"]
        arrival["arrival_pattern"]
        ramp["ramp"]
        urlsel["url_selection_strategy"]
        sched["search_recipe"]
    end

    subgraph DATASET["dataset"]
        bs["dataset_backing_store"]
        cs["dataset_client_store"]
        samp["dataset_sampler"]
        comp["dataset_composer"]
        cdl["custom_dataset_loader"]
        pdl["public_dataset_loader"]
    end

    subgraph ENDPOINTS["endpoints"]
        ept["endpoint"]
        tp["transport"]
    end

    subgraph PIPELINE["records pipeline"]
        rp["record_processor"]
        resp["results_processor"]
        acc["accumulator"]
        an["analyzer"]
    end

    subgraph EXPORTERS["exporters"]
        de["data_exporter"]
        ce["console_exporter"]
        se["stream_exporter"]
        ap["artifact_publisher"]
    end

    subgraph RECIPES["recipes / post-process"]
        rec["search_recipe"]
        rpp["search_recipe_post_process"]
    end

    subgraph ACCURACY["accuracy"]
        ag["accuracy_grader"]
        ab["accuracy_benchmark"]
    end

    subgraph TELEMETRY["telemetry"]
        gc["gpu_telemetry_collector"]
        sm["server_metrics_processor"]
    end

    subgraph ORCH["orchestrator"]
        cc["convergence_criterion"]
        sp["search_planner"]
    end

    subgraph PLATFORM["platform"]
        svc["service"]
        smgr["service_manager"]
        cm["communication"]
        cmc["communication_client"]
        zp["zmq_proxy"]
        ar["api_router"]
        ui["ui"]
        plot["plot"]
    end

    style LOADGEN fill:transparent,stroke:#1976d2,stroke-width:2px
    style DATASET fill:transparent,stroke:#2e7d32,stroke-width:2px
    style ENDPOINTS fill:transparent,stroke:#ef6c00,stroke-width:2px
    style PIPELINE fill:transparent,stroke:#6a1b9a,stroke-width:2px
    style EXPORTERS fill:transparent,stroke:#00838f,stroke-width:2px
    style RECIPES fill:transparent,stroke:#9c27b0,stroke-width:2px
    style ACCURACY fill:transparent,stroke:#c2185b,stroke-width:2px
    style TELEMETRY fill:transparent,stroke:#5d4037,stroke-width:2px
    style ORCH fill:transparent,stroke:#1976d2,stroke-width:2px
    style PLATFORM fill:transparent,stroke:#2e7d32,stroke-width:2px
```

The full registry has 37 categories (see `src/aiperf/plugin/categories.yaml`). All are reachable via `plugins.get_class(PluginType.X, name)`; per-category typed metadata via `get_typed_metadata(category, name)` is available for `endpoint`, `transport`, `plot`, `service`, `custom_dataset_loader`, `convergence_criterion`, `search_planner` (others fall back to the raw metadata dict).

## 16. ABC hierarchy — orchestrator-side

The orchestrator layer's extension points are abstract base classes; implementations are registered as plugins or instantiated directly by category-aware factories.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TB
    subgraph EXEC["execute a BenchmarkRun"]
        re["RunExecutor (ABC)<br/>execute(run) → RunResult"]
        loc["LocalSubprocessExecutor"]
        k8s["K8sChildJobExecutor"]
        re --> loc
        re --> k8s
    end

    subgraph STRAT["per-cell control"]
        es["ExecutionStrategy (ABC)<br/>validate_config / should_continue / record_result"]
        ft["FixedTrialsStrategy"]
        cv["ConvergenceStrategy"]
        es --> ft
        es --> cv
    end

    subgraph CONV["convergence"]
        cc["ConvergenceCriterion (ABC)<br/>from_plan / update / converged"]
        ci["CIWidthConvergence"]
        cvc["CVConvergence"]
        dc["DistributionConvergence"]
        cc --> ci
        cc --> cvc
        cc --> dc
    end

    subgraph SEARCH["adaptive search"]
        sp["SearchPlanner (ABC)<br/>propose_next / record / converged"]
        bo["BayesianSearchPlanner<br/>(skopt)"]
        sp --> bo
    end

    subgraph AGG["aggregation"]
        as["AggregationStrategy (ABC)<br/>aggregate / metric_keys"]
        sa["SweepAnalyzer<br/>(post-hoc analyzer)"]
        as --> sa
    end

    subgraph DATAS["dataset"]
        bds["BaseDatasetSampler (ABC)<br/>sample"]
        bds --> rng["RandomSampler / SeqSampler / …"]
    end

    style EXEC fill:transparent,stroke:#1976d2,stroke-width:2px
    style STRAT fill:transparent,stroke:#ef6c00,stroke-width:2px
    style CONV fill:transparent,stroke:#2e7d32,stroke-width:2px
    style SEARCH fill:transparent,stroke:#6a1b9a,stroke-width:2px
    style AGG fill:transparent,stroke:#00838f,stroke-width:2px
    style DATAS fill:transparent,stroke:#c2185b,stroke-width:2px
```

## 17. Service / lifecycle base classes

Every long-running component descends from `BaseService` (orchestrator only) or `BaseComponentService` (workers, managers). `AIPerfLifecycleMixin` powers the standalone components that don't run as services.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TB
    subgraph MIX["mixins"]
        lc["AIPerfLifecycleMixin<br/>CREATED → INITIALIZING → INITIALIZED →<br/>STARTING → RUNNING → STOPPING → STOPPED"]
        tm["TaskManagerMixin"]
        hh["HooksMixin"]
        hsm["HealthServerMixin"]
        chm["CommandHandlerMixin"]
        phm["ProcessHealthMixin"]
        com["CommunicationMixin"]
    end

    subgraph BASE["service ABCs"]
        bs["BaseService (ABC)<br/>= HealthServer + CommandHandler + ProcessHealth"]
        bcs["BaseComponentService<br/>= BaseService + bus subscription glue"]
        bs --> bcs
    end

    subgraph CTRL["orchestrator role"]
        sc["SystemController<br/>(BaseService directly)"]
        bs --> sc
    end

    subgraph IMPL["implementations"]
        wk["Worker"]
        ti["TimingManager"]
        ci["CreditIssuer<br/>(plain class, not a service)"]
        dm["DatasetManager"]
        rm["RecordsManager"]
        rs["ResultsManager"]
        bcs --> wk
        bcs --> ti
        bcs --> dm
        bcs --> rm
        bcs --> rs
    end

    lc --> bs
    tm --> lc
    hh --> lc
    hsm --> bs
    chm --> bs
    phm --> bs
    com --> bcs

    style MIX fill:transparent,stroke:#1976d2,stroke-width:2px
    style BASE fill:transparent,stroke:#2e7d32,stroke-width:2px
    style CTRL fill:transparent,stroke:#ef6c00,stroke-width:2px
    style IMPL fill:transparent,stroke:#6a1b9a,stroke-width:2px
```

## 18. Protocol map — what each Protocol abstracts

Protocols are AIPerf's "structural typing" lever: any class shaped like the Protocol satisfies it without inheritance. Reach for them when you want plugin polymorphism without forcing a base class.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 70, 'padding': 16, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '16px'}}}%%
flowchart TB
    subgraph DATA["dataset"]
        cdl["CustomDatasetLoaderProtocol"]
        pdl["PublicDatasetLoaderProtocol"]
        dss["DatasetSamplingStrategyProtocol"]
    end

    subgraph END["endpoints / transport"]
        ept["EndpointProtocol"]
        isr["InferenceServerResponse"]
        cred["CreditRouterProtocol"]
    end

    subgraph TIM["timing"]
        ig["IntervalGeneratorProtocol"]
        uss["URLSelectionStrategyProtocol"]
    end

    subgraph PIPE["records pipeline"]
        ar["AccumulatorResult"]
        ap["AccumulatorProtocol"]
        an["AnalyzerProtocol"]
        sep["StreamExporterProtocol"]
    end

    subgraph EXP["exporters"]
        cep["ConsoleExporterProtocol"]
        dep["DataExporterProtocol"]
        app["ArtifactPublisherProtocol"]
    end

    subgraph ACC["accuracy"]
        agp["AccuracyGraderProtocol"]
        abp["AccuracyBenchmarkProtocol"]
    end

    subgraph TEL["telemetry"]
        gtcp["GPUTelemetryCollectorProtocol"]
        gtpp["GPUTelemetryProcessorProtocol"]
        smpp["ServerMetricsProcessorProtocol"]
    end

    subgraph PLAT["platform"]
        mcp["MessageCodecProtocol"]
        alp["AIPerfLoggerProtocol"]
        wr["WatchRenderer"]
        wds["WatchdogDataSource"]
        pth["PlotTypeHandlerProtocol"]
    end

    style DATA fill:transparent,stroke:#2e7d32,stroke-width:2px
    style END fill:transparent,stroke:#1976d2,stroke-width:2px
    style TIM fill:transparent,stroke:#ef6c00,stroke-width:2px
    style PIPE fill:transparent,stroke:#6a1b9a,stroke-width:2px
    style EXP fill:transparent,stroke:#00838f,stroke-width:2px
    style ACC fill:transparent,stroke:#c2185b,stroke-width:2px
    style TEL fill:transparent,stroke:#5d4037,stroke-width:2px
    style PLAT fill:transparent,stroke:#455a64,stroke-width:2px
```

## 19. Plugin lookup — runtime sequence

How a registered plugin gets instantiated at run time. Same flow regardless of whether it's a built-in (`plugins.yaml`) or third-party (entry point).

```mermaid
%%{init: {'sequence': {'actorMargin': 60, 'boxMargin': 18, 'noteMargin': 12, 'messageMargin': 40}, 'themeVariables': {'fontSize': '16px'}}}%%
sequenceDiagram
    autonumber
    participant Caller as Caller<br/>(e.g. cli_runner)
    participant Reg as PluginRegistry
    participant Cat as categories.yaml
    participant Ent as PluginEntry
    participant Cls as Plugin class

    Caller->>Reg: plugins.get_class(<br/>PluginType.X, name)
    Reg->>Cat: lookup category schema
    Cat-->>Reg: metadata_class +<br/>interface contract
    Reg->>Ent: select highest-priority entry
    Ent-->>Reg: cls + validated metadata
    Reg-->>Caller: cls
    Caller->>Cls: cls.from_plan(plan) /<br/>cls(...)
    Cls-->>Caller: instance
    Caller->>Cls: invoke (e.g. converged?,<br/>propose_next, execute)
```

## 20. Plugin → ABC/Protocol cross-reference

Each plugin category points at the interface its implementations must satisfy. Categories under "ABC" are class-inheritance based (factory dispatches via `plugins.get_class`); categories under "Protocol" are structural (any matching shape is acceptable).

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart LR
    subgraph CAT["plugin category"]
        c1["convergence_criterion"]
        c2["search_planner"]
        c3["dataset_sampler"]
        c4["custom_dataset_loader"]
        c5["public_dataset_loader"]
        c6["data_exporter"]
        c7["console_exporter"]
        c8["accuracy_grader"]
        c9["accuracy_benchmark"]
        c10["gpu_telemetry_collector"]
        c11["server_metrics_processor"]
        c12["service"]
        c13["api_router"]
        c14["timing_strategy"]
    end

    subgraph IF["interface (ABC or Protocol)"]
        a1["ConvergenceCriterion (ABC)"]
        a2["SearchPlanner (ABC)"]
        a3["BaseDatasetSampler (ABC)"]
        p1["CustomDatasetLoaderProtocol"]
        p2["PublicDatasetLoaderProtocol"]
        p3["DataExporterProtocol"]
        p4["ConsoleExporterProtocol"]
        p5["AccuracyGraderProtocol"]
        p6["AccuracyBenchmarkProtocol"]
        p7["GPUTelemetryCollectorProtocol"]
        p8["ServerMetricsProcessorProtocol"]
        b1["BaseComponentService (ABC)"]
        a4["IntervalGeneratorProtocol"]
        r1["FastAPI APIRouter (factory fn)"]
    end

    c1 --> a1
    c2 --> a2
    c3 --> a3
    c4 --> p1
    c5 --> p2
    c6 --> p3
    c7 --> p4
    c8 --> p5
    c9 --> p6
    c10 --> p7
    c11 --> p8
    c12 --> b1
    c13 --> r1
    c14 --> a4

    style CAT fill:transparent,stroke:#1976d2,stroke-width:2px
    style IF fill:transparent,stroke:#2e7d32,stroke-width:2px
```

## 21. Sweep execution flow — class module map in motion

How the types from the §11 class diagram actually flow through a sweep run. Read it as: each box is an instance of a class from §11; arrows show what produces what; cardinality annotations make the fan-out explicit (1 plan → N variations × M trials → N×M results → 1 aggregate).

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TB
    subgraph CFG["envelope (1)"]
        a["AIPerfConfig<br/>schema_version / benchmark / sweep /<br/>multi_run / variables / random_seed"]
        bc["BenchmarkConfig<br/>(envelope.benchmark, body type)"]
        sw["SweepConfig (Annotated union)<br/>GridSweep | ScenarioSweep |<br/>AdaptiveSearchSweep"]
        mr["MultiRunConfig<br/>num_runs (→ plan.trials)<br/>cooldown_seconds, confidence_level,<br/>set_consistent_seed,<br/>disable_warmup_after_first,<br/>convergence: ConvergenceConfig | None"]
        a --> bc
        a --> sw
        a --> mr
    end

    subgraph EXP["expand → plan (1 → N)"]
        es["expand_sweep(envelope dict)<br/>→ list[(dict, SweepVariation)]<br/>(grid: cartesian; scenarios: deep-merge;<br/>adaptive_search: 1-element placeholder)"]
        cfgs["BenchmarkConfig × N<br/>(one per variation,<br/>built post per-variation Jinja render)"]
        vars["SweepVariation × N<br/>index, label, values"]
        bp["BenchmarkPlan<br/>configs[N], variations[N],<br/>variation_seeds[N], trials=M,<br/>multi_run, sweep,<br/>failure_policy, …"]
        es --> cfgs
        es --> vars
        cfgs --> bp
        vars --> bp
        mr -.num_runs.-> bp
        sw -.iteration_order/.-> bp
    end

    subgraph ITER["orchestrator iterates (N × M)"]
        orch["MultiRunOrchestrator.execute(plan, executor, *,<br/>cancel_check, search_planner)<br/>dispatch: is_adaptive_search →<br/>execute_adaptive_search; else<br/>_plan_iteration_order(plan) →<br/>_execute_repeated / _execute_independent"]
        loop{"per (variation v, trial t)"}
        run["BenchmarkRun<br/>benchmark_id, cfg=configs[v],<br/>variation=variations[v], trial=t,<br/>artifact_dir, label, random_seed,<br/>resolved"]
        orch --> loop
        loop --> run
    end

    subgraph EXEC["executor (N × M)"]
        re["RunExecutor.execute(run)<br/>(LocalSubprocessExecutor or<br/>K8sChildJobExecutor)"]
        rr["RunResult<br/>success, summary_metrics,<br/>artifact paths"]
        re --> rr
    end

    subgraph AGG["aggregate (N×M → 1)"]
        results["list[RunResult] (N × M)"]
        sa["SweepAnalyzer.compute<br/>group by variation_values"]
        out["sweep_aggregate/<br/>profile_export_aiperf_sweep.{json,csv}<br/>+ best_configurations + pareto_optimal"]
        results --> sa --> out
    end

    bp --> orch
    run --> re
    rr --> results

    style CFG fill:transparent,stroke:#1976d2,stroke-width:2px
    style EXP fill:transparent,stroke:#2e7d32,stroke-width:2px
    style ITER fill:transparent,stroke:#ef6c00,stroke-width:2px
    style EXEC fill:transparent,stroke:#6a1b9a,stroke-width:2px
    style AGG fill:transparent,stroke:#00838f,stroke-width:2px
```

```mermaid
%%{init: {'sequence': {'actorMargin': 60, 'boxMargin': 18, 'noteMargin': 12, 'messageMargin': 40}, 'themeVariables': {'fontSize': '16px'}}}%%
sequenceDiagram
    autonumber
    participant Cfg as AIPerfConfig
    participant Loader as build_benchmark_plan
    participant Sweep as expand_sweep
    participant Plan as BenchmarkPlan
    participant Orch as MultiRunOrchestrator
    participant Run as BenchmarkRun
    participant Exec as RunExecutor
    participant Res as RunResult
    participant Ana as SweepAnalyzer

    Cfg->>Loader: validated envelope<br/>(benchmark, sweep, multi_run, …)
    Loader->>Sweep: envelope dict (with `sweep` block)
    Sweep-->>Loader: [(BenchmarkConfig_v, SweepVariation_v) × N]
    Loader->>Plan: BenchmarkPlan(<br/>configs[N], variations[N],<br/>variation_seeds[N], trials=M,<br/>multi_run, sweep, failure_policy, …)
    Loader-->>Orch: plan
    Orch->>Orch: dispatch:<br/>plan.is_adaptive_search → execute_adaptive_search<br/>else _plan_iteration_order(plan) →<br/>REPEATED (_execute_repeated) /<br/>INDEPENDENT (_execute_independent)

    loop for each (variation v in 0..N, trial t in 0..M)
        Orch->>Run: BenchmarkRun(cfg=configs[v],<br/>variation=variations[v], trial=t,<br/>artifact_dir=…)
        Orch->>Exec: execute(run)
        Exec-->>Res: RunResult(success,<br/>summary_metrics, paths)
        Res-->>Orch: append to results
    end

    Orch-->>Ana: list[RunResult] (N × M)
    Ana->>Ana: group by variation.values<br/>compute best_configurations,<br/>pareto_optimal, per_combination_metrics
    Ana-->>Cfg: profile_export_aiperf_sweep.{json,csv}
```

The two views together: the **flowchart** shows cardinality and which class produces which (the data shape of a sweep); the **sequence** shows the temporal call pattern between the same classes. Both use only the types from the §11 class diagram — no module-internal helpers.

## 22. Adaptive search — class types

The adaptive search path layers atop the same `BenchmarkPlan` / `MultiRunOrchestrator` / `RunExecutor` core. Adaptive config is **not** a separate field — it's the `AdaptiveSearchSweep` variant of the `SweepConfig` discriminated union (`type: adaptive_search`). Two plugin categories cooperate: a `search_planner` (drives the outer loop) and an optional `search_recipe` (curates the search space / objective / post-process from a higher-level recipe template). The optional terminal `post_process` is a single `PostProcessSpec` resolved via `search_recipe_post_process` plugins.

```mermaid
%%{init: {'class': {'titleTopMargin': 25}, 'themeVariables': {'fontSize': '15px'}}}%%
classDiagram
    class AdaptiveSearchSweep {
        type: "adaptive_search"
        algorithm: "bayes"
        planner: SearchPlannerType
        search_space: list[SearchSpaceDimension]
        objective: AdaptiveObjective
        max_iterations: int
        n_initial_points: int
        plateau_window / plateau_threshold
        improvement_patience
        random_seed
        recipe_name
        optuna_sampler
        monotonic_stability_trials
        constraint_mode: penalty | eic
        cooldown_seconds (from _SweepBase)
        sla_filters: list[SLAFilter]
        post_process: PostProcessSpec | None
    }
    class AdaptiveObjective {
        metric: str
        stat: avg | p50 | p90 | p95 | p99
        direction: OptimizationDirection
    }
    class SearchSpaceDimension {
        path: str  (envelope-rooted)
        low / high
        type: int | float
        log: bool
    }
    class SLAFilter {
        metric: str
        stat: avg | p50 | p90 | p95 | p99
        op: lt | gt | le | ge
        threshold: float
    }
    class PostProcessSpec {
        handler: str  (plugin name)
        params: dict
    }

    class BenchmarkPlan {
        sweep: SweepConfig | None
        is_adaptive_search: bool
        (true iff isinstance(sweep,
        AdaptiveSearchSweep))
    }

    class SearchPlanner {
        <<abstract>>
        ask() (BenchmarkConfig, SweepVariation) | None
        tell(variation, results)
        is_converged() bool
        history() list[SearchIteration]
        convergence_reason() str | None  (default None, not abstract)
    }
    class BayesianSearchPlanner {
        skopt Optimizer (lazy import)
        constraint_mode: penalty / eic
        plateau / patience checks
    }
    class MonotonicSLASearchPlanner {
        1D exponential probe + bisection
        boundary_summary() dict
        (feasible_max / infeasible_min /
        first_breach)
    }
    class OptunaSearchPlanner {
        sampler: gp | tpe | botorch
    }
    class SearchIteration {
        iteration_idx: int
        variation_values: dict[str, Any]
        objective_value: float
        results: list[RunResult]
        feasible: bool
        non_monotonic_warning: str | None
    }

    class SearchRecipe {
        <<protocol>>
        name: ClassVar[str]
        description: ClassVar[str]
        expand(ctx) SearchRecipeOutput
    }
    class SearchRecipeContext {
        benchmark_config: BenchmarkConfig
        sla_targets: dict
        sweep_overrides: dict
    }
    class SearchRecipeOutput {
        adaptive_search: AdaptiveSearchSweep
        post_process: PostProcessSpec | None
    }

    class PostProcessHandler {
        <<protocol>>
        name / description: ClassVar[str]
        process(sweep_aggregate, params) dict
    }
    class DegradationKneeDetect
    class TTFTCurveFit
    class ItlSurfaceFit
    class SLABreachKnee

    class MultiRunOrchestrator {
        execute(plan, executor, *,
        cancel_check, search_planner)
        execute_adaptive_search(plan,
        executor, planner, *, cancel_check)
    }

    AdaptiveSearchSweep --> AdaptiveObjective : objective
    AdaptiveSearchSweep --> SearchSpaceDimension : search_space[]
    AdaptiveSearchSweep --> SLAFilter : sla_filters[]
    AdaptiveSearchSweep --> PostProcessSpec : post_process
    BenchmarkPlan --> AdaptiveSearchSweep : sweep
    SearchPlanner <|-- BayesianSearchPlanner
    SearchPlanner <|-- MonotonicSLASearchPlanner
    SearchPlanner <|-- OptunaSearchPlanner
    SearchPlanner ..> SearchIteration : history
    SearchPlanner ..> AdaptiveSearchSweep : configured by
    SearchRecipe ..> SearchRecipeContext : reads
    SearchRecipe ..> SearchRecipeOutput : returns
    SearchRecipeContext --> BenchmarkConfig : benchmark_config
    SearchRecipeOutput --> AdaptiveSearchSweep : produces
    SearchRecipeOutput --> PostProcessSpec : post_process
    PostProcessHandler <|.. DegradationKneeDetect
    PostProcessHandler <|.. TTFTCurveFit
    PostProcessHandler <|.. ItlSurfaceFit
    PostProcessHandler <|.. SLABreachKnee
    PostProcessSpec ..> PostProcessHandler : resolved via plugin

    MultiRunOrchestrator ..> BenchmarkPlan : reads
    MultiRunOrchestrator ..> SearchPlanner : drives ask/tell
```

Built-in `search_recipe` plugins (`src/aiperf/search_recipes/`):

- `max-throughput-ttft-sla`, `max-throughput-itl-sla`
- `concurrency-ramp`
- `prefill-ttft-curve`, `decode-itl-curve`
- `max-goodput-under-slo`, `max-concurrency-under-sla`

Built-in `search_recipe_post_process` plugins: `degradation_knee`, `ttft_curve_fit`, `itl_surface_fit`, `sla_breach_knee`.

## 23. Adaptive search — execution flow

The BO outer loop is a `propose → execute → record` cycle inside `MultiRunOrchestrator.execute_adaptive_search`. `BenchmarkRun` and `RunExecutor` are the same as in the grid path; the difference is that `BenchmarkPlan.configs` starts with one seed config and grows by one per iteration as the planner asks for the next point.

```mermaid
%%{init: {'sequence': {'actorMargin': 60, 'boxMargin': 18, 'noteMargin': 12, 'messageMargin': 40}, 'themeVariables': {'fontSize': '16px'}}}%%
sequenceDiagram
    autonumber
    participant Plan as BenchmarkPlan<br/>(sweep is AdaptiveSearchSweep)
    participant Orch as MultiRunOrchestrator
    participant Pl as SearchPlanner<br/>(Bayesian / Monotonic / Optuna)
    participant Run as BenchmarkRun
    participant Exec as RunExecutor
    participant Res as RunResult
    participant PP as PostProcessHandler
    participant Out as search_history.json /<br/>sweep_aggregate

    Orch->>Pl: planner instantiated upstream<br/>(_build_search_planner(plan))<br/>and passed into execute()

    loop until is_converged() or max_iterations
        Orch->>Pl: ask()
        Pl-->>Orch: (BenchmarkConfig_k, SweepVariation_k)<br/>or None (converged → convergence_reason())
        alt got proposal
            Orch->>Orch: _run_independent_cell<br/>(fresh ExecutionStrategy per cell)
            loop trials inner (until strategy says stop)
                Orch->>Run: BenchmarkRun(cfg_k, variation_k, t, …)
                Orch->>Exec: execute(run)
                Exec-->>Res: RunResult
            end
            Orch->>Pl: tell(variation_k, cell_results)
            Pl->>Pl: filter by SLAFilter,<br/>compute objective scalar,<br/>plateau / patience / max-iter check
            Orch-->>Out: write_search_history(...)<br/>(incremental, includes<br/>boundary_summary if planner has it)
        end
    end

    Orch->>PP: process(sweep_aggregate, params)<br/>(per PostProcessSpec on sweep)
    PP-->>Out: knees, curve fits, …
    Orch-->>Out: profile_export_aiperf_sweep.{json,csv}
```

## 24. Adaptive search — recipe → AdaptiveSearchSweep

A user can either author an `AdaptiveSearchSweep` directly under `sweep:` (low level) or pick a `search_recipe` plugin (high level) that builds one from a recipe + the user's existing benchmark config. There is no intermediate `MultiRunConfig.adaptive_search` field — the adaptive block lives entirely on `sweep`.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 18, 'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '17px'}}}%%
flowchart TB
    subgraph IN["user inputs"]
        cli["--search-recipe NAME --param k=v<br/>or<br/>sweep: { type: adaptive_search, … } in YAML<br/>or<br/>--search-space PATH:LO,HI[:KIND]<br/>--search-objective metric:stat:direction<br/>--search-sla metric:stat:op:threshold (×N)"]
        uc["AIPerfConfig.benchmark<br/>(models, endpoint, phases, …)"]
    end

    subgraph RECIPE["recipe layer (optional)"]
        ctx["SearchRecipeContext"]
        rc["SearchRecipe plugin (Protocol)<br/>built-ins:<br/>max-throughput-ttft-sla<br/>max-throughput-itl-sla<br/>concurrency-ramp<br/>prefill-ttft-curve / decode-itl-curve<br/>max-goodput-under-slo<br/>max-concurrency-under-sla"]
        out["SearchRecipeOutput<br/>(adaptive_search, post_process)"]
    end

    subgraph CFG["adaptive sweep variant"]
        asc["AdaptiveSearchSweep<br/>(SweepConfig variant,<br/>type=adaptive_search)<br/>search_space, objective,<br/>max_iterations, sla_filters,<br/>post_process, planner, …"]
    end

    subgraph DRIVE["runtime drivers"]
        plan["AIPerfConfig.sweep<br/>= AdaptiveSearchSweep"]
        plan2["BenchmarkPlan.sweep<br/>(is_adaptive_search=True)"]
        sp["SearchPlanner plugin<br/>(BayesianSearchPlanner |<br/>MonotonicSLASearchPlanner |<br/>OptunaSearchPlanner)"]
        pph["search_recipe_post_process plugin<br/>(degradation_knee, ttft_curve_fit,<br/>itl_surface_fit, sla_breach_knee)"]
    end

    cli --> rc
    uc --> ctx --> rc --> out --> asc
    cli -.direct path.-> asc

    asc --> plan
    plan --> plan2
    plan2 --> sp
    plan2 --> pph

    style IN fill:transparent,stroke:#1976d2,stroke-width:2px
    style RECIPE fill:transparent,stroke:#2e7d32,stroke-width:2px
    style CFG fill:transparent,stroke:#ef6c00,stroke-width:2px
    style DRIVE fill:transparent,stroke:#6a1b9a,stroke-width:2px
```
