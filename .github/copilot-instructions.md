<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

Python 3.10+ async AI benchmarking tool for measuring LLM inference server performance. Services communicate via ZMQ message bus; optionally deployable on Kubernetes via a kopf-based operator.

**Reference documentation:**
- [`llms.txt`](llms.txt) - Agent session-bootstrap index: single-page topology of every doc in the repo with a one-line purpose for each. Start here when unsure which doc to read.
- [`docs/architecture.md`](docs/architecture.md) - Three-plane architecture, core components, credit system, data flow, communication patterns
- [`docs/dev/patterns.md`](docs/dev/patterns.md) - Code examples for CLI commands, services, models, messages, plugins, error handling, logging, testing
- [`docs/cli-options.md`](docs/cli-options.md) - Complete CLI command and option reference
- [`docs/environment-variables.md`](docs/environment-variables.md) - All `AIPERF_*` environment variables by subsystem
- [`docs/metrics-reference.md`](docs/metrics-reference.md) - Metric definitions, formulas, and requirements
- [`docs/plugins/plugin-system.md`](docs/plugins/plugin-system.md) - Plugin architecture, categories, creation guide
- [`docs/dev/kubernetes-flow.md`](docs/dev/kubernetes-flow.md) and [`docs/kubernetes/`](docs/kubernetes/) - Kubernetes operator, CRD lifecycle, cluster deployment
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - Development setup, available commands, pre-commit hooks, DCO

## Coding Standards

- async/await for ALL I/O - no `time.sleep`, no blocking calls.
- `Field(description="...")` on EVERY Pydantic field. Docstrings on dataclass fields.
- Type hints on ALL functions (params and return).
- KISS + DRY: minimal code, optimize for reader.
- `AIPerfBaseModel` for data, `BaseConfig` for configuration. `@dataclass(slots=True)` for hot-path inner models created at high volume (e.g. SSE chunks, parsed responses) where Pydantic overhead matters. Use `__pydantic_config__ = ConfigDict(extra="forbid")` on dataclasses that participate in Pydantic union discrimination.
- `BaseComponentService` for services, `BaseService` for SystemController only.
- Message bus for inter-service communication - no shared mutable state.
- CLI commands: one file per command in `cli_commands/`, lazily loaded via import strings in `cli.py`. See `docs/dev/patterns.md`.
- YAML plugin registry for extensible features (`plugins.yaml`).
- Lambda for expensive logs: `self.debug(lambda: f"{self._x()}")`. Direct string for cheap ones.
- Always `orjson.loads(s)`, `orjson.dumps(d)` for JSON.
- No `Optional[X]` or `Union[X, Y]` - use `X | Y`.
- Comments only for "why?" not "what".
- Enums are string-based - use `MessageType.X` directly, never `.value`.
- Dependencies: always use `uv` (never pip) - `uv add package`, `uv run pytest`.
- Use mermaid diagrams instead of ASCII art in markdown files.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding comments to explain why.
- No emojis in code or comments.

## Build and Test Commands

```bash
make first-time-setup                                      # Initial environment setup
make install                                               # Install project + mock server
uv run pytest tests/unit/ -n auto                          # Unit tests (fast, isolated)
uv run pytest -m integration -n auto                       # Integration tests (real services, multiprocess)
uv run pytest -m component_integration -n auto             # Component integration tests (single process)
ruff format . && ruff check --fix .                        # Format and lint
make validate-plugin-schemas                               # Validate plugin registry
pre-commit run                                             # Pre-commit on staged files
pre-commit run --all-files                                 # Pre-commit on all files
make generate-all-docs                                     # Regenerate CLI + env var docs
make generate-all-plugin-files                             # Regenerate plugin enums, overloads, schemas
```

## Pre-Commit Hooks

Run pre-commit after every code change, even before creating commits:

```bash
pre-commit run              # Staged files only
pre-commit run --all-files  # All files (recommended after significant changes)
```

Hooks: `check-ast`, `debug-statements`, `detect-private-key`, `check-added-large-files`, `check-case-conflict`, `check-executables-have-shebangs`, `check-merge-conflict`, `check-json`, `check-toml`, `check-yaml`, `check-shebang-scripts-are-executable`, `end-of-file-fixer`, `mixed-line-ending`, `no-commit-to-branch`, `requirements-txt-fixer`, `trailing-whitespace`, `codespell`, `add-license`, `generate-cli-docs`, `generate-env-vars-docs`, `generate-plugin-artifacts`, `validate-plugin-schemas`, `test-imports`, `check-agent-files-sync`, `check-ergonomics`, `check-ruff-baselined`, `ruff`, `ruff-format`.

## Adding a New Service

1. Create class extending `BaseComponentService` with `@on_message` handlers
2. Register in `plugins.yaml` under `service` category with `class`, `description`, `metadata`
3. Add message type to `common/enums/enums.py` if new messages needed
4. Create message class in `messages/` with `message_type` field
5. Validate with `aiperf plugins --validate`

## Adding a New Message

1. Add enum value to `MessageType` in `common/enums/enums.py`
2. Create message class in `messages/` inheriting from `Message` with `message_type` field set
3. Add `@on_message(MessageType.X)` handler in the receiving service
4. Auto-subscription happens during `@on_init` phase

## Adding a New Plugin

1. Create plugin class implementing the appropriate base
2. Add entry to `plugins.yaml` with `class`, `description`, `metadata`
3. Validate with `make validate-plugin-schemas`
4. Use via `plugins.get_class(PluginType.X, 'name')`

## Adding a New Config Field

`AIPerfConfig` is an envelope; `BenchmarkConfig` is the swept body.

- **Does the field vary per sweep variation?** -> add it to `BenchmarkConfig` (`src/aiperf/config/config.py`).
- **Is it cross-variation machinery?** (Jinja context, sweep config, seed, multi-run trial settings.) -> add it to `AIPerfConfig` envelope.

YAML configs follow the same shape: body keys nest under `benchmark:`; envelope keys (`sweep`, `multi_run`, `variables`, `random_seed`) stay at top level.

Reading body fields:

```python
bench = config.benchmark
if bench.endpoint.streaming:
    ...
```

## Kubernetes

The Kubernetes operator and CLI layer live in `src/aiperf/operator/`, `src/aiperf/kubernetes/`, and `src/aiperf/cli_commands/kube/`. Key patterns:

- **kopf handlers** — The operator entry point is `src/aiperf/operator/main.py`. All `@kopf.on.*` decorators live there; handler bodies are decorator-free functions in `src/aiperf/operator/handlers/{create,cleanup,completion,lifecycle,monitor}.py`. Raise `kopf.PermanentError` to stop retrying, `kopf.TemporaryError(..., delay=N)` to retry after delay — generic exceptions retry forever. kopf calls handlers with a fixed kwarg set (`body, spec, name, namespace, patch, uid, **_: Any`); these signatures are baselined against `keyword-only-args` because kopf owns the calling convention.
- **kubernetes_asyncio access** — Always use `async with k8s_client() as api:` from `aiperf.kubernetes.client`; never instantiate `ApiClient()` directly. The helper handles in-cluster-or-kubeconfig fallback and closure.
- **`aiperf kube` CLI** — Subcommands live in `src/aiperf/cli_commands/kube/` and are registered in `_app.py`. Composite flags (`namespace`, `kubeconfig`, `kube-context`) pass via `KubeManageOptions` from `aiperf.config.kube`.
- **FastAPI routers** — Two patterns: module-level `router = APIRouter(...)` in `src/aiperf/api/routers/*.py`, and factory `create_xxx_router(deps...) -> APIRouter` in `src/aiperf/operator/routers/jobs.py` when the router closes over live state.
- **Shellouts** — Always `aiperf.kubernetes.subproc.run_command(...)` / `check_command` / `start_streaming_process` + `terminate_process`; never `asyncio.create_subprocess_exec` directly. 60 s default timeout.
- **Chaos testing** — `tests/kubernetes/chaos/` holds fault-injection tests (`k8s_slow`). The suite uses `podTemplate.shareProcessNamespace` (chart value / `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` env) to enable cross-container `kill` via `kubectl exec`; production default is false. Toxiproxy fixture in `fixtures/toxiproxy.yaml` supplies API-disruption faults. See `docs/superpowers/specs/2026-04-23-chaos-expansion-design.md`.
- **CLI user output** — All kube-CLI output goes through `from aiperf.kubernetes import console as kube_console`; never `print` or `rich.print`. Last-benchmark persistence (`save_last_benchmark`) lives there too — do not roll your own `last_X.json`.
- **`--output text|json`** — Read-only CLI checks (preflight, validate) expose `Literal["text", "json"]`; in JSON mode, downshift the `aiperf.kube` logger to WARNING in a `try/finally` and print via `orjson.dumps(..., option=OPT_INDENT_2)`. Result dataclasses own the `to_dict()` schema.
- **Watch orchestration** — `aiperf kube watch` is a three-layer split: `*Poller` classes, a `WatchOrchestrator` owning one `k8s_client()` + signal handlers, and renderers implementing the `WatchRenderer` Protocol (`start`/`render`/`stop`). New renderer = new Protocol implementor + one line in the renderer factory.
- **CRD generator** — Both CRDs in `deploy/helm/aiperf-operator/templates/crd*.yaml` are auto-generated from `AIPerfConfig` and `AIPerfSweepSpec` Pydantic models by `tools/generate_crd.py`. Do NOT hand-edit the YAML; run `uv run python tools/generate_crd.py` and verify with `--check`. Cross-field invariants are enforced as CEL `x-kubernetes-validations` rules attached by shape-detector decorators (`_decorate_aiperf_config_node`, `_decorate_endpoint_node`, etc.) so the same rule fires on both `AIPerfJob.spec.benchmark` and `AIPerfSweep.spec.template.spec.benchmark`. CEL gotchas: `has(self.X)` requires X to be a declared property (anything inside `x-kubernetes-preserve-unknown-fields` is invisible to CEL); opaque preserve-unknown array items can't be dereferenced (so phase/dataset name uniqueness, phase→dataset reference integrity, and "seamless not on first" stay in the operator's `@model_validator` decorators on `AIPerfConfig`); `oldSelf` only triggers on update (use `!has(oldSelf.X) || oldSelf.X == self.X` for first-set-freezes semantics). User-facing rule catalog in `docs/kubernetes/crd-validation.md`.
- **Durable completion claim** — Exactly-once completion work is gated by `await try_claim_completion(...)` in `operator/client_cache.py` — a JSON-patch with a `test` op, so concurrent ticks race atomically on the apiserver. The in-process `_shutdown_sent` set is only a fast path; the CR annotation is authoritative.
- **Cooperative cancellation** — `on_delete` calls `request_cancellation(job_key(ns, name))`; handlers poll `is_cancellation_requested(key)` at every `await` boundary and exit early. Inject the check as a `Callable[[], bool]` into helpers rather than importing the flag deep.
- **Results-ready marker** — The controller writes `.aiperf_results_ready.json` via `write_ready_marker(base_dir)` only after all artifacts are on disk; the results sidecar refuses to serve top-level files until the marker is present (checkpoints under `checkpoints/` bypass the gate).
- **K8s-vs-local audit suite** — `tests/kubernetes/audit/` runs each workflow case twice (operator + `aiperf kube results` download path; bare `batch/v1.Job` running `aiperf profile` directly) and diffs the artifact trees through three buckets (exact / tolerance / structural). Opt-in: `pytest -m k8s_audit tests/kubernetes/audit/ -n auto`. Spec: `docs/superpowers/specs/2026-04-26-k8s-vs-local-audit-design.md`.
- **Runs/sweep index** — `<RESULTS.DIR>/.aiperf_index.sqlite`, owned by `src/aiperf/operator/runs_index.py`. Single writer (the operator's kopf-owning process); readers open `mode=ro&cache=shared`. Two tables: `runs` (one row per `(ns, job, epoch)`) and `sweep_variations` (one row per `(ns, sweep, epoch, variation_idx)`). Both carry the six `DEFAULT_COMPARE_METRICS` as flat columns plus a zstd-compressed `metrics_json` blob. All read sites in `results_layout.list_runs_async`, `results_db.ResultsDB`, and `routers/results_files.py` go index-first with disk fallback + lazy backfill, so a stale or missing index degrades to slower never wrong. Bootstrap runs as an asyncio task at operator startup; manual rebuild via `aiperf kube index rebuild` (calls `POST /admin/index/rebuild`).
- **Status convention** — `StatusBuilder.set_observed_generation(body["metadata"]["generation"])` is called from every successful kopf reconcile path so `kubectl wait` and GitOps tooling can detect spec acknowledgment. Never stamp on error/early-return paths. Sites: `monitor.monitor_progress`, `create._finalize_success`, `lifecycle.on_cancel` + `on_benchmark_complete`, `sweep/create.handle`.
- **Lifecycle surface (`status.subPhase`)** — operator stamps `status.subPhase` from the controller's `SystemState` (`src/aiperf/common/enums/lifecycle_enums.py:42`) on every monitor tick. Values: `initializing | configuring | ready | profiling | processing | stopping | shutdown`. Distinct from `status.phase` (operator's view) and `status.currentPhase` (per-benchmark stage). Cleared on terminal `set_phase` transitions, mirroring the existing `currentPhase` clear. The controller publishes `SystemStateChangedMessage` (`src/aiperf/common/messages/progress_messages.py`) on the bus at six transitions wired in `SystemController._set_system_state`; `ProgressRouter` mirrors to `/api/progress.system_state` for the operator to consume. Also pushed as `aiperf.nvidia.com/system-state` annotation on the AIPerfJob (and JobSet) every 10s by `_patch_jobset_progress`.
- **Per-phase completion booleans** — `status.phases.<name>` carries `isRequestsComplete` (all responses received, `requests_end_ns` set) and `isRecordsComplete` (records aggregated, `records_end_ns` set) alongside the existing counters. `sendingComplete` semantically means "all requests dispatched" (`sent_end_ns is not None`); previously this field was wrongly bound to `requests_end_ns` — fixed in the same commit that added the new booleans. The wire format already carries the source `*_end_ns` timestamps as fields on `CombinedPhaseStats`; the booleans are derived in `_build_phase_progress` from the `@property` accessors.
- **`Complete` / `Failed` conditions (`batchv1.Job` convention)** — `kubectl wait --for=condition=Complete aiperfjob/<name>` works identically to a `batchv1.Job`. Derived in `StatusBuilder._derive_terminal_conditions` from `(phase, ResultsAvailable)`: `Complete=True` only when `phase=Completed ∧ ResultsAvailable=True` (protects against premature latching during the result-fetch window). `Failed=True` when `phase=Failed`. `Cancelled` clears both — neither Complete nor Failed, matching `batchv1.Job` cancellation semantics. Mutually exclusive: writing one to True writes the other to False in the same `finalize()`.
- **Operator metrics** — Prometheus `/metrics` lives at `aiperf.operator.metrics`, served by an in-process daemon thread on `OperatorEnvironment.METRICS_PORT` (default 9090; 0 disables). Wrap **only** kopf reconcile handlers with `@track_handler("name")` (between the `@kopf.*` decorator and the function); never instrument helper functions. The kopf operator and the FastAPI results-server are separate sidecar containers — operator metrics cannot be served by the FastAPI app.
- **Watch-driven shortcuts** — pod-restart events come from `handlers/pod_restarts.py` wired via `@kopf.on.event` on `Pod` filtered by `labels={"jobset.sigs.k8s.io/jobset-name": kopf.PRESENT}`. **Why event, not field:** `@kopf.on.field` requires kopf to write a `kopf.zalando.org/last-handled-configuration` diff-base annotation on every observed Pod, which needs `pods: patch` RBAC. The operator only has `pods: [get, list, watch]`, so field-watching produced 9-retry 403 storms per Pod restart event in production (discovered on `dynamo-aws-dev-02` 2026-05-03; fixed in `a5910b08a`). The same trap applies to any `@kopf.on.field/create/update/delete` on a non-CRD resource — if the operator lacks `patch` on that kind, prefer `@kopf.on.event` (finalizer-free, stores no state) and dedup in-process. The previous timer-poll `_check_pod_restarts` is deleted. Decorator-free handler bodies live in their own modules; kopf wiring lives in `operator/main.py` per the existing convention.
- **JobSet completion fast path** — `handlers/jobset_terminal.py` watches `JobSet.status.conditions`, and on `Completed/True` patches the parent AIPerfJob's `metadata.annotations[aiperf.nvidia.com/benchmark-complete] = "true"` to trigger the existing `on_benchmark_complete` handler. Drops completion latency below the monitor-tick interval. Failure branch (`Failed/True`) intentionally stays on the timer because `_handle_jobset_failed_condition` has nontrivial recovery logic. Idempotent against the controller-pod race that sets the same annotation when results have been written.

## Parameter Sweeping

Two mutually-exclusive sweep paths share the codebase; both are reachable from a CLI flag list.

- **In-process sweep** — `MultiRunOrchestrator` + `aggregate_sweep_and_export`, triggered by `--concurrency 10,20,30` (or any other magic-list flag) on a single `aiperf profile` invocation. The v1->v2 converter (`_promote_magic_lists_to_sweep_block` in `src/aiperf/config/v1/converter.py`) lifts list-shaped values into a `sweep.variables` block before `AIPerfConfig` validation, keyed by `phases.<name>.<field>`; `expand_sweep` consumes that during `build_benchmark_plan` to materialize one `BenchmarkConfig` per variation. `BenchmarkPlan.is_sweep` flips true when there are >1 expanded configs. Sequential single-machine execution; best for local iteration and CI.
- **Cluster sweep** — `AIPerfSweep` CRD + `operator/handlers/sweep/`. The k8s operator owns the cluster-wide cardinality contract: one `AIPerfJob` (and one controller pod) per variation; each child pod sees a single-config plan. Best for parallelism across nodes and restart durability.
- **Adaptive outer loop (BO)** — `aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" --search-metric output_token_throughput --search-direction maximize --search-max-iterations 30`. `BenchmarkPlan.adaptive_search` carries an `AdaptiveSearchConfig` (typed `Any` in `config/benchmark.py` to break a circular import; coerced via a `field_validator(mode="before")` in `_models_benchmark.MultiRunConfig`). `MultiRunOrchestrator.execute` dispatches to `execute_adaptive_search` which drives a `BayesianSearchPlanner` (skopt soft dep behind the `[bo]` extra). Runs both in-process and cluster-side: under the operator, the same planner is instantiated by `sweep_controller/main.py` when an `AIPerfSweep` CR includes a `multi_run.adaptive_search` block, and the K8s executor creates one `AIPerfJob` per iteration; kopf-side handlers stay BO-agnostic. Mutually exclusive with magic-list/grid sweeps (`build_benchmark_plan` rejects sweep+adaptive_search at the v1→v2 boundary) and with `--convergence-metric` (rejected in `_converter_optionals.build_multi_run`); BO is **not** rejected under `AIPERF_OPERATOR_MANAGED=1` — `_reject_in_process_sweep_under_operator` rejects only `plan.is_sweep`, and `sweep_controller.plan_builder` propagates BO through to the controller pod. `status.totalVariations`/`maxTotalRuns` become upper bounds when BO converges early, mirroring `convergence.maxRuns`. `search_history.json` is written incrementally next to `sweep_aggregate/`. See `docs/sweeping/bayesian-optimization.md` and `docs/kubernetes/sweeps.md#adaptive-search-bayesian-optimization`.
- **Mutual-exclusion gate** — when `AIPERF_OPERATOR_MANAGED=1` is set in a controller pod, `cli_runner._reject_in_process_sweep_under_operator` hard-fails any `plan.is_sweep` to keep both layers from sweeping at once. The error message names the variation count, swept parameter names, and points at `docs/kubernetes/sweeps.md` + the AIPerfSweep CR alternative.
- **Mode dispatch** — `MultiRunOrchestrator.execute` dispatches on `plan.parameter_sweep_mode` (`SweepMode.REPEATED` default, or `INDEPENDENT`). Repeated: trials outer / variations inner, paths under `<base>/profile_runs/trial_NNNN/<variation>/`. Independent: variations outer / trials inner, paths under `<base>/<variation>/`. Both produce the same `sweep_aggregate/` output (aggregation is mode-agnostic — `aggregate_sweep_and_export` groups by `variation_values` post-hoc). Adaptive convergence is incompatible with repeated. The k8s sweep_controller's children-manifest walk in `sweep_controller/main.py` mirrors the same idx → (var,trial) derivation.

Polymorphic CLI flags use `Annotated[Any, ..., BeforeValidator(parse_int_or_int_list), CLIParameter(...)]` (see `src/aiperf/config/parsing.py`). The `Any` is intentional per `gotcha_cyclopts_polymorphic_cli_any.md` — cyclopts cannot dispatch off `int | list[int] | None`. Never tighten the type, or argument resolution breaks. `parse_int_or_int_list` raises `TypeError` (not stdlib `ValueError`) on malformed string/list inputs and names the offending CLI flag in the message.

Sweep aggregation output lives at `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` under the artifact dir. The schema (produced by `SweepAnalyzer.compute()` in `src/aiperf/orchestrator/aggregation/sweep.py`) carries `metadata`, `best_configurations`, `pareto_optimal`, and `per_combination_metrics`. Pareto objectives use `OptimizationDirection` (a `CaseInsensitiveStrEnum` matching `SweepMode`); `_dominates` compares `Objective.metric_key` per direction. The CSV exporter (`AggregateSweepCsvExporter`) writes a four-section layout (per-combination / best / pareto / metadata) and the JSON exporter writes the dict as-is — both byte-compatible with the upstream PR #699 schema.

## Testing Conventions

- `@pytest.mark.asyncio` for async tests, `@pytest.mark.parametrize` for data-driven
- `from tests.harness import mock_plugin` for plugin mocking
- Name: `test_<function>_<scenario>_<expected>` e.g. `test_parse_config_missing_field_raises_error`
- Imports at file top, fixtures for setup, one focus per test
- Use `from pytest import param` and put `# fmt: skip` on the `)` line:
  ```python
  @pytest.mark.parametrize(
      "arg",
      [
          param(..., id="case1"),
          param(..., id="case2"),
      ],
  )  # fmt: skip
  ```
- Auto-fixtures (always active): asyncio.sleep runs instantly, RNG=42, singletons reset between tests

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`. One PR = one concern.

## Tips

- SystemController uses `BaseService` (not `BaseComponentService`) - it's the orchestrator.
- Worker/TimingManager disable GC for latency - see `service_metadata.disable_gc`.
- macOS child processes close terminal FDs to prevent Textual UI corruption.
- Plugin priority resolves conflicts: higher wins, external beats built-in at equal priority.
- Decorators: `@on_init`, `@on_start`, `@on_stop`, `@on_message`, `@on_command`, `@background_task`, `@on_pull_message`, `@on_request`.
- Communication: `publish()` for broadcast, `@on_message` to subscribe, `send_command_and_wait_for_response()` for sync.
- `AIPerfLifecycleMixin` for standalone components: `CREATED` -> `INITIALIZING` -> `INITIALIZED` -> `STARTING` -> `RUNNING` -> `STOPPING` -> `STOPPED`; `FAILED` terminal.

## LLM-Ergonomics

AIPerf treats agent-readability as a first-class quality axis. The code is expected to be mimicked by LLMs — so conventions are explicit (not tacit), exceptions self-describe, types carry domain meaning, and reference files are kept exemplary. See `docs/dev/patterns.md` for concrete good/bad examples.

**Mechanical floor (enforced in CI, zero new violations allowed):**

```bash
make check-ergonomics        # 10 custom AST checks: file-size, function-size, nesting-depth, keyword-only-args, module-state, duplicate-classes, pydantic-fields, stdlib-json, exception-message, isinstance-tuple
make check-ruff-baselined    # 9 ruff rules: PLR0915, PLR0912, C901, TID251, BLE001, S110, S112, ANN201, D103
```

Baselines (`tools/ergonomics_baseline.json`, `tools/ruff_baseline.json`) grandfather pre-existing violations. New code must pass clean; do not add entries to the baselines.

**Semantic ceiling (reviewed, not mechanically enforced):**

- **Error messages** name the operation, the specific input, and a likely cause or next step. `raise DatasetLoadError(f"dataset '{name}' missing column 'answer_key' at row {row_idx}; add it to the config or set skip_validation=true")` — not `raise ValueError("bad input")`.
- **Type hints** carry domain meaning: `Literal[...]` for enum-like strings, `Protocol` / `TypedDict` for structural contracts, parameterized containers (`list[ResultBundle]`, not bare `list`). Avoid `Any` on public return types. Use `X | None`, never `Optional[X]`.
- **Docstrings** include a runnable example with realistic identifiers (`job_id="aiperf-bench-7f2a"`), not `foo`/`bar` placeholders. Side-effects (publishes, file writes, state mutations) are named in the docstring. Project-specific exceptions are listed under `Raises:`.
- **Naming** disambiguates synonyms: if `Credit`, `Request`, and `Session` overlap in domain meaning, the docstring of the authoritative class mentions the synonyms so grep finds it.
- **Comments** document WHY (non-local constraint, past bug, subtle invariant) — never WHAT. Rename instead of commenting when possible.
- **New patterns** must be documented in CLAUDE.md (+ the two sync files) or `docs/dev/patterns.md` before the branch ships — agents do not reliably absorb tacit conventions from surrounding code.
- **Reference files** (the ones cited in `docs/dev/patterns.md` via leading-comment paths like `# aiperf/kubernetes/client.py`) must stay exemplary: no `# noqa` without an explanatory comment, and no entries in the ergonomics/ruff baselines for files that teach the rule they'd be violating.

Run `/aiperf-llm-ergonomics-review` before shipping a PR that touches public API, exceptions, or a reference file.

## Config v1 (CLI input layer)

`UserConfig` / `ServiceConfig` (`src/aiperf/config/v1/`) are the cyclopts-facing
CLI input DTOs. They carry CLI flag annotations and Pydantic field metadata,
but **NO validators** — `AIPerfConfig` is the single validation gate.

The converter (`src/aiperf/config/v1/converter.py`) is the only allowed v1→v2
boundary. Downstream of `cli_commands/`, only `AIPerfConfig` / `BenchmarkPlan`
/ `BenchmarkRun` flow. Enforced by ruff TID251 in `pyproject.toml` and a
redundant `v1-import-leak` AST check in `tools/check_ergonomics.py`.

Hard rules for adding new config fields:
1. **Every new config field MUST be added to `AIPerfConfig` (`src/aiperf/config/`)** — the YAML/v2 layer is the source of truth and the single validation gate. CLI exposure is optional.
2. **NEVER add a CLI flag without first adding the field to `AIPerfConfig`.** v1 (`UserConfig`) is a CLI-input shim; a flag with no v2 destination has nowhere to flow.
3. CLI-flag placement on v1: fits an existing nested class (Endpoint/Input/LoadGen/Output/Tokenizer/Accuracy)? Add it there. Doesn't fit? Add as a top-level field on `UserConfig`. NEVER add new nested classes to v1.
4. NO validators on v1 classes — ever. New validation goes on `AIPerfConfig` (cross-field validators) or in the v1→v2 converter (input-shape coercion).

## Orchestrator Plugin Categories

Convergence criteria and search planners are plugin-registered:

- **`convergence_criterion`** — `CIWidthConvergence` / `CVConvergence` / `DistributionConvergence` are registered in `plugins.yaml` under the `convergence_criterion` category. `_cli_runner_helpers._build_convergence_criterion(plan)` is a one-liner over `plugins.get_class(PluginType.CONVERGENCE_CRITERION, plan.convergence_mode).from_plan(plan)`. Each criterion subclass owns its `from_plan(plan)` factory because constructor signatures differ. Third-party criteria ship as wheels with setuptools entry points and override built-ins via priority.
- **`search_planner`** — `BayesianSearchPlanner` is registered under the `search_planner` category. `_cli_runner_helpers._build_search_planner(plan)` dispatches via `plugins.get_class(PluginType.SEARCH_PLANNER, plan.adaptive_search.planner)(plan.configs[0], plan.adaptive_search)`. Both the in-process (`cli_runner.py`) and cluster-side (`sweep_controller/main.py`) paths use the same helper. The `[bo]` extra remains for shipping the built-in skopt-backed planner without forcing skopt on every install — third-party wheels can ship their own planners with their own dep trees.
- **Typed metadata** — Every plugin entry's metadata is validated against the category's `metadata_class` Pydantic model and accessed via the per-category helper in `src/aiperf/plugin/metadata.py` (e.g. `get_convergence_criterion_metadata("ci_width") -> ConvergenceCriterionMetadata`) or the generic `get_typed_metadata(category, name)`. **Do not access `entry.metadata[...]` as a dict** — always use the typed accessor.
- **Never instantiate** `BayesianSearchPlanner(...)` / `CIWidthConvergence(...)` / `CVConvergence(...)` / `DistributionConvergence(...)` directly outside the registered plugin entries themselves and the convergence module's internal tests. Production code dispatches via the registry.

The convergence/orchestrator metadata schemas live in `src/aiperf/plugin/schema/_orchestrator_schemas.py` (re-exported from `schemas.py`) to keep `schemas.py` under the 500-line ergonomics ceiling.

## Pre-Commit Checklist

1. Review diff: all lines required?
2. `ruff format . && ruff check --fix .`
3. `uv run pytest tests/unit/ -n auto`
4. Type hints on all functions
5. `Field(description=...)` on all Pydantic fields
6. `git commit -s`

## Four-File Sync Rule

`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` must contain identical content (only headers/frontmatter differ; the cursor file's `alwaysApply: true` frontmatter is required and must be preserved). When updating one, update all four. Run `make check-agent-files-sync` after editing to confirm sync — pre-commit enforces this on every commit that touches one of these files.

## Documentation Updates

> **DOCUMENTATION IS REQUIRED, NOT OPTIONAL.** Any PR that adds or changes a feature, CLI option, env var, plugin, message type, or service without updating the relevant docs is incomplete and will not be merged.

When making changes, update the appropriate documentation files using the table below. When adding a new tutorial, also add it to `README.md`'s tutorial index. **Any new file under `docs/` must also be added to `docs/index.yml`** (the Fern site index) — `tools/check_docs_index.py` enforces this in CI. If the change is internal-only and not user-facing (e.g. developer reference, internal mechanics, debugging notes), put the doc under `docs/reference/` rather than skipping documentation.

| Change type | Files to update |
|---|---|
| Adding/removing a doc file, or changing its purpose | `llms.txt` |
| Architecture, components, data flow, communication | `docs/architecture.md` |
| Coding standards, build commands, new patterns | `AGENTS.md` + `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` |
| Code patterns, examples, base classes | `docs/dev/patterns.md` |
| CLI arguments or commands | `docs/cli-options.md` (auto-generated via `make generate-cli-docs`) |
| Environment variables | `docs/environment-variables.md` (auto-generated via `make generate-env-vars-docs`) |
| Metrics definitions or formulas | `docs/metrics-reference.md` |
| Plugin system, categories, creation | `docs/plugins/plugin-system.md` |
| Accuracy benchmarks, graders | `docs/accuracy/` |
| Server metrics, schemas | `docs/server-metrics/` |
| Benchmark modes, timing, traces | `docs/benchmark-modes/` |
| Tokenizer, reference docs | `docs/reference/` |
| Dataset synthesis API | `docs/api/synthesis.md` |
| Dev setup, make targets, pre-commit | `CONTRIBUTING.md` |
| Contribution process, DCO | `CONTRIBUTING.md` |
| New services, message types, plugin types | `docs/architecture.md` + `docs/dev/patterns.md` |
| Kubernetes operator, handlers, CR lifecycle | `docs/dev/kubernetes-flow.md` |
| Kubernetes deployment, Helm chart, cluster setup | `docs/kubernetes/getting-started.md` + `docs/kubernetes/configuration.md` + `docs/kubernetes/production.md` + `docs/kubernetes/workflow.md` |
| User-defined templated output files (`artifacts.user_files`) | `docs/kubernetes/user-files.md` |
| Kubernetes RBAC, securityContext, NetworkPolicy | `docs/kubernetes/rbac-security.md` |
| Kube preflight checks / validate / debug | `docs/kubernetes/preflight.md` + `docs/kubernetes/validate.md` + `docs/kubernetes/debug-command.md` |
| Kube attach / logs command deep-dives | `docs/kubernetes/attach.md` + `docs/kubernetes/logs.md` |
| Watch diagnosis issues / thresholds | `docs/kubernetes/diagnosis-issues.md` |
| Operator HTTP API / web dashboard | `docs/kubernetes/results-api.md` + `docs/kubernetes/dashboard-ui.md` |
| Controller-pod sidecars (event-bus, results) | `docs/kubernetes/sidecars.md` |
| Memory estimator model / tuning | `docs/kubernetes/memory-estimator.md` |
| Kueue integration | `docs/kubernetes/kueue.md` |
| Sweeps on Kubernetes | `docs/kubernetes/sweeps.md` |
| GPU telemetry on Kubernetes | `docs/kubernetes/gpu-telemetry.md` |
| Direct mode (`--no-operator`) | `docs/kubernetes/direct-mode.md` |
| `aiperf kube init` / `generate` | `docs/kubernetes/init-and-generate.md` |
| Kube CLI commands | `docs/cli-options.md` (auto-generated via `make generate-cli-docs`) |
| Tutorials and feature guides | `docs/tutorials/` + `README.md` tutorial index |

**A feature is incomplete until documentation is updated.**
