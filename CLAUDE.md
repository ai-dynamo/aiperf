<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf

Python 3.11+ async AI benchmarking tool for measuring LLM inference server performance. 10 services communicate via ZMQ message bus.

**Reference documentation:**
- [`docs/architecture.md`](docs/architecture.md) - Three-plane architecture, core components, credit system, data flow, communication patterns
- [`docs/dev/patterns.md`](docs/dev/patterns.md) - Code examples for CLI commands, services, models, messages, plugins, error handling, logging, testing
- [`docs/cli-options.md`](docs/cli-options.md) - Complete CLI command and option reference
- [`docs/environment-variables.md`](docs/environment-variables.md) - All `AIPERF_*` environment variables by subsystem
- [`docs/metrics-reference.md`](docs/metrics-reference.md) - Metric definitions, formulas, and requirements
- [`docs/plugins/plugin-system.md`](docs/plugins/plugin-system.md) - Plugin architecture, categories, creation guide
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
- YAML plugin registry for extensible features (`src/aiperf/plugin/plugins.yaml`).
- Lambda for expensive logs: `self.debug(lambda: f"{self._x()}")`. Direct string for cheap ones.
- Always `orjson.loads(s)`, `orjson.dumps(d)` for JSON.
- Reading filesystem paths: use `aiperf.common.path_safety.safe_read_template_path` instead of inline `Path().read_text()` / `open().read()`. Returns `None` on safety-check failure; caller picks the fallback semantic. See `docs/dev/patterns.md` § "Safe Filesystem Reads Pattern".
- No `Optional[X]` or `Union[X, Y]` - use `X | Y`.
- Comments only for "why?" not "what".
- Enums are string-based - use `MessageType.X` directly, never `.value`.
- Dependencies: always use `uv` (never pip) - `uv add package`, `uv run pytest`.
- Use mermaid diagrams instead of ASCII art in markdown files.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding comments to explain why.
- Multi-line docstrings are permitted. Use them when the behavior is non-obvious.
- No emojis in code or comments.
- Hide a metric from the console table with `console_group = MetricConsoleGroup.NONE`; group it into a separate section with `MetricConsoleGroup.{USAGE,CACHE,PREDICTION,AUDIO,REASONING,SPEC_DECODE,GPU_POWER_EFFICIENCY_NVIDIA,GPU_POWER_EFFICIENCY_AMD}`. Default is `DEFAULT`. See `docs/metrics-reference.md` "Metric Console Group Reference".
- Platform-conditional code MUST branch on `IS_WINDOWS` / `IS_MACOS` / `IS_LINUX` from `aiperf.common.constants`, never on `platform.system()` directly. The constants are evaluated once at import time, are uniformly greppable, and produce smaller diffs. See `src/aiperf/common/bootstrap.py` and `src/aiperf/config/comm/ipc.py` for canonical examples.

## NaN/Inf Discipline

Numeric metric values crossing a serialization boundary or feeding a numerical algorithm must be finite or explicitly `None`. Use `aiperf.common.finite` (`FiniteFloat`, `scrub_non_finite`, `nan_safe_mean`/`std`, `is_finite_value`). Mechanical CI invariants in `tests/unit/property/test_finite_invariants.py` reject new violations and ratchet existing debt to zero via baseline files. See [`docs/dev/patterns.md`](docs/dev/patterns.md) § "NaN/Inf Discipline Pattern" and [`docs/dev/global-invariants.md`](docs/dev/global-invariants.md) for the full contract.

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
2. Register in `src/aiperf/plugin/plugins.yaml` under `service` category with `class`, `description`, `metadata`
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
2. Add entry to `src/aiperf/plugin/plugins.yaml` with `class`, `description`, `metadata`
3. Validate with `make validate-plugin-schemas`
4. Use via `plugins.get_class(PluginType.X, 'name')`

## Adding a New CLI Flag

See `docs/dev/patterns.md` § "Adding a New CLI Flag". CLIConfig is flat; never add a nested config class.

## Kubernetes

The Kubernetes operator and CLI layer live in `src/aiperf/operator/`, `src/aiperf/kubernetes/`, and `src/aiperf/cli_commands/kube/`. Key patterns:

- **kopf handlers** — The operator entry point is `src/aiperf/operator/main.py`. All `@kopf.on.*` decorators live there; handler bodies are decorator-free functions in `src/aiperf/operator/handlers/{create,cleanup,completion,lifecycle,monitor}.py`. Raise `kopf.PermanentError` to stop retrying, `kopf.TemporaryError(..., delay=N)` to retry after delay — generic exceptions retry forever. kopf calls handlers with a fixed kwarg set (`body, spec, name, namespace, patch, uid, **_: Any`); these signatures are baselined against `keyword-only-args` because kopf owns the calling convention.
- **kubernetes_asyncio access** — Always use `async with k8s_client() as api:` from `aiperf.kubernetes.client`; never instantiate `ApiClient()` directly. The helper handles in-cluster-or-kubeconfig fallback and closure.
- **`aiperf kube` CLI** — Subcommands live in `src/aiperf/cli_commands/kube/` and are registered in `_app.py`. Composite flags (`namespace`, `kubeconfig`, `kube-context`) pass via `KubeManageOptions` from `aiperf.config.kube`.
- **FastAPI routers** — Two patterns: module-level `router = APIRouter(...)` in `src/aiperf/api/routers/*.py`, and factory `create_xxx_router(deps...) -> APIRouter` in `src/aiperf/operator/routers/jobs.py` when the router closes over live state.
- **Shellouts** — Always `aiperf.kubernetes.subproc.run_command(...)` / `check_command` / `start_streaming_process` + `terminate_process`; never `asyncio.create_subprocess_exec` directly. 60 s default timeout.
- **CLI user output** — All kube-CLI output goes through `from aiperf.kubernetes import console as kube_console`; never `print` or `rich.print`. Last-benchmark persistence (`save_last_benchmark`) lives there too — do not roll your own `last_X.json`.
- **`--output text|json`** — Read-only CLI checks (preflight, validate) expose `Literal["text", "json"]`; in JSON mode, downshift the `aiperf.kube` logger to WARNING in a `try/finally` and print via `orjson.dumps(..., option=OPT_INDENT_2)`. Result dataclasses own the `to_dict()` schema.
- **Benchmark diagnosis** — `aiperf kube watch` and its `WatchOrchestrator`/`WatchRenderer` split were removed; the salvaged detectors live in `src/aiperf/kubernetes/benchmark_diagnosis.py` and are consumed by `aiperf kube debug`. Tunables are `AIPERF_K8S_DIAGNOSIS_*`.
- **CRD generator** — Both CRDs (`deploy/helm/aiperf-operator/templates/crd-aiperfjob.yaml` and `crd-aiperfsweep.yaml`) are auto-generated from `AIPerfJobSpec` / `AIPerfSweepSpec` Pydantic models by `tools/generate_crd.py`. Do NOT hand-edit the YAML; run `uv run python tools/generate_crd.py` and verify with `--check`. User-facing validation rule catalog in `docs/kubernetes/crd-validation.md`.
- **Durable completion claim** — Exactly-once completion work is gated by `await try_claim_completion(...)` in `operator/client_cache.py` — a JSON-patch with a `test` op, so concurrent ticks race atomically on the apiserver. The in-process `_shutdown_sent` set is only a fast path; the CR annotation is authoritative.
- **Serialized Kubernetes run config** — Every service container reads the same controller-rendered `BenchmarkRun` through `aiperf.kubernetes.serialized_run.read_serialized_run_json`; never independently resolve `CLIConfig` inside `aiperf service`, because seeds, synthesized defaults, and artifact identity must remain identical across pods.
- **Endpoint credential transport** — Serialized `BenchmarkRun` and ConfigMap data remain secret-free. Redact endpoint credentials before persistence, inject them only through Secret-backed `AIPERF_INJECTED_API_KEY`, `AIPERF_INJECTED_HEADERS`, or `AIPERF_INJECTED_ENDPOINT_URLS`, and use `aiperf.common.endpoint_credentials` for validation and rehydration.
- **Operator metrics** — Wrap **only** kopf reconcile handlers with `@track_handler("name")` (between the `@kopf.*` decorator and the function); never instrument helper functions.
- **`ServiceRunType.KUBERNETES`** is a generated plugin enum member: it is declared in `src/aiperf/plugin/plugins.yaml` and emitted into `src/aiperf/plugin/enums.py` by `make generate-all-plugin-files`. Never hand-edit `src/aiperf/plugin/enums.py`.
- **Operator-namespace fallback** — code that needs the chart-default operator namespace MUST import `DEFAULT_OPERATOR_NAMESPACE` from `aiperf.kubernetes.constants`, never hardcode `"aiperf-system"`. Callers with cluster API access should still prefer `find_operator_namespace` (cluster-wide pod-label search).

For complete implementation details see `docs/dev/kubernetes-flow.md`.

## Parameter Sweeping (Kubernetes)

In-process sweeps and adaptive search are documented in main's `docs/sweeping/` tutorials. The Kubernetes-side path is:

- **Cluster sweep** — `AIPerfSweep` CRD + `operator/handlers/sweep/`. The k8s operator owns the cluster-wide cardinality contract: one `AIPerfJob` (and one controller pod) per variation; each child pod sees a single-config plan. Best for parallelism across nodes and restart durability.
- **Adaptive outer loop (BO) under the operator** — the sweep-controller pod in `sweep_controller/main.py` instantiates the same `BayesianSearchPlanner` plugin the in-process path uses; the K8s executor creates one `AIPerfJob` per iteration. kopf-side handlers stay BO-agnostic.
- **Mutual-exclusion gate** — when `AIPERF_OPERATOR_MANAGED=1` is set in a controller pod, `cli_runner._reject_in_process_sweep_under_operator` hard-fails any `plan.is_sweep` to keep both layers from sweeping at once.
- **Mode dispatch** — `MultiRunOrchestrator.execute` dispatches on `plan.is_adaptive_search`, then on the grid sweep's iteration order (`_plan_iteration_order(plan)` → `SweepMode.REPEATED`/`INDEPENDENT`). The k8s sweep_controller's children-manifest walk in `sweep_controller/main.py` mirrors the same idx → (var,trial) derivation.

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
- `dag_jsonl` input type: conversation DAG benchmarks (fork + spawn modes). See `docs/benchmark-modes/dag.md` for abstractions and authoring.
- Validator gate convention: unsupported constructs raise `NotImplementedError` with a leading `"<loc>: <reason>"` prefix where `<loc>` identifies the conversation/turn (e.g. `"conversation 'foo' turn 3: ..."`). New validators must follow this shape.
- Per-turn payload contract: `extra_body` / `max_tokens` / `model` are dispatch-turn only; `raw_tools` is the lone field that walks history (system-prompt-like). Dataset rows author `extra`, not `extra_body`. See `docs/dev/patterns.md` "Per-turn dataset `extra`".

## Plot Envelope Section

`AIPerfConfig.plot: PlotEnvelopeConfig | None` lets a single AIPerf YAML own its
visualization. Two forms: bare-string path (`plot: ./plots/baseline.yaml`,
resolved relative to the AIPerf YAML's dir) or inline dict mirroring
`src/aiperf/plot/default_plot_config.yaml`. When set, `~/.aiperf/plot_config.yaml`
is ignored and `artifacts.auto_plot` flips to True (unless explicitly False).
The auto-plot callback materializes the resolved envelope to
`<artifact_dir>/.aiperf-plot-config.yaml` so `aiperf plot <dir>` reproduces.
See `src/aiperf/config/plot.py` for the Pydantic models.

## Pre-Commit Checklist

1. Review diff: all lines required?
2. `ruff format . && ruff check --fix .`
3. `uv run pytest tests/unit/ -n auto`
4. `uv run pytest tests/unit/property/ -n auto` (mechanical NaN/inf and field-validator invariants)
5. Type hints on all functions
6. `Field(description=...)` on all Pydantic fields
7. `git commit -s`

## Four-File Sync Rule

`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` must contain identical content (only headers/frontmatter differ). When updating one, update all four. Run `make check-agent-files-sync` after editing to confirm sync — pre-commit enforces this on every commit that touches one of these files.

## Documentation Updates

> **DOCUMENTATION IS REQUIRED, NOT OPTIONAL.** Any PR that adds or changes a feature, CLI option, env var, plugin, message type, or service without updating the relevant docs is incomplete and will not be merged.

When making changes, update the appropriate documentation files using the table below. When adding a new tutorial, also add it to `README.md`'s tutorial index. **Any new file under `docs/` must also be added to `docs/index.yml`** (the Fern site index) — `tools/check_docs_index.py` enforces this in CI. If the change is internal-only and not user-facing (e.g. developer reference, internal mechanics, debugging notes), put the doc under `docs/reference/` rather than skipping documentation.

| Change type | Files to update |
|---|---|
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
| Kube preflight checks / validate / debug | `docs/kubernetes/preflight.md` + `docs/kubernetes/validate.md` + `docs/kubernetes/debug-command.md` |
| Operator HTTP API / web dashboard | `docs/kubernetes/results-api.md` + `docs/kubernetes/dashboard-ui.md` |
| Sweeps on Kubernetes | `docs/tutorials/sweeps.md#running-sweeps-on-kubernetes` |
| Tutorials and feature guides | `docs/tutorials/` + `README.md` tutorial index |

**A feature is incomplete until documentation is updated.**
