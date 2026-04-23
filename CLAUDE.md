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

Hooks: `check-ast`, `debug-statements`, `detect-private-key`, `check-added-large-files`, `check-case-conflict`, `check-merge-conflict`, `check-json`, `check-toml`, `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `codespell`, `add-license`, `generate-cli-docs`, `generate-env-vars-docs`, `generate-plugin-artifacts`, `validate-plugin-schemas`, `test-imports`, `ruff` (lint + format).

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

## Kubernetes

The Kubernetes operator and CLI layer live in `src/aiperf/operator/`, `src/aiperf/kubernetes/`, and `src/aiperf/cli_commands/kube/`. Key patterns:

- **kopf handlers** — The operator entry point is `src/aiperf/operator/main.py`. All `@kopf.on.*` decorators live there; handler bodies are decorator-free functions in `src/aiperf/operator/handlers/{create,cleanup,completion,lifecycle,monitor}.py`. Raise `kopf.PermanentError` to stop retrying, `kopf.TemporaryError(..., delay=N)` to retry after delay — generic exceptions retry forever. kopf calls handlers with a fixed kwarg set (`body, spec, name, namespace, patch, uid, **_: Any`); these signatures are baselined against `keyword-only-args` because kopf owns the calling convention.
- **kubernetes_asyncio access** — Always use `async with k8s_client() as api:` from `aiperf.kubernetes.client`; never instantiate `ApiClient()` directly. The helper handles in-cluster-or-kubeconfig fallback and closure.
- **`aiperf kube` CLI** — Subcommands live in `src/aiperf/cli_commands/kube/` and are registered in `_app.py`. Composite flags (`namespace`, `kubeconfig`, `kube-context`) pass via `KubeManageOptions` from `aiperf.config.kube`.
- **FastAPI routers** — Two patterns: module-level `router = APIRouter(...)` in `src/aiperf/api/routers/*.py`, and factory `create_xxx_router(deps...) -> APIRouter` in `src/aiperf/operator/routers/jobs.py` when the router closes over live state.
- **Shellouts** — Always `aiperf.kubernetes.subproc.run_command(...)` / `check_command` / `start_streaming_process` + `terminate_process`; never `asyncio.create_subprocess_exec` directly. 60 s default timeout.
- **CLI user output** — All kube-CLI output goes through `from aiperf.kubernetes import console as kube_console`; never `print` or `rich.print`. Last-benchmark persistence (`save_last_benchmark`) lives there too — do not roll your own `last_X.json`.
- **`--output text|json`** — Read-only CLI checks (preflight, validate) expose `Literal["text", "json"]`; in JSON mode, downshift the `aiperf.kube` logger to WARNING in a `try/finally` and print via `orjson.dumps(..., option=OPT_INDENT_2)`. Result dataclasses own the `to_dict()` schema.
- **Watch orchestration** — `aiperf kube watch` is a three-layer split: `*Poller` classes, a `WatchOrchestrator` owning one `k8s_client()` + signal handlers, and renderers implementing the `WatchRenderer` Protocol (`start`/`render`/`stop`). New renderer = new Protocol implementor + one line in the renderer factory.
- **Durable completion claim** — Exactly-once completion work is gated by `await try_claim_completion(...)` in `operator/client_cache.py` — a JSON-patch with a `test` op, so concurrent ticks race atomically on the apiserver. The in-process `_shutdown_sent` set is only a fast path; the CR annotation is authoritative.
- **Cooperative cancellation** — `on_delete` calls `request_cancellation(job_key(ns, name))`; handlers poll `is_cancellation_requested(key)` at every `await` boundary and exit early. Inject the check as a `Callable[[], bool]` into helpers rather than importing the flag deep.
- **Results-ready marker** — The controller writes `.aiperf_results_ready.json` via `write_ready_marker(base_dir)` only after all artifacts are on disk; the results sidecar refuses to serve top-level files until the marker is present (checkpoints under `checkpoints/` bypass the gate).

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

## Pre-Commit Checklist

1. Review diff: all lines required?
2. `ruff format . && ruff check --fix .`
3. `uv run pytest tests/unit/ -n auto`
4. Type hints on all functions
5. `Field(description=...)` on all Pydantic fields
6. `git commit -s`

## Three-File Sync Rule

`CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` must contain identical content (only file-specific headers/frontmatter differ; the cursor file's `alwaysApply: true` frontmatter is required and must be preserved). When updating one, update all three. Always diff them after editing to confirm sync.

## Documentation Updates

When making changes, update the appropriate documentation files. When adding a new tutorial, also add it to `README.md`'s tutorial index.

| Change type | Files to update |
|---|---|
| Adding/removing a doc file, or changing its purpose | `llms.txt` |
| Architecture, components, data flow, communication | `docs/architecture.md` |
| Coding standards, build commands, new patterns | `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` |
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
