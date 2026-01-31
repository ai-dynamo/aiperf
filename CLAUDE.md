<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Dev Guide

Python 3.10+ async AI Benchmarking Tool for measuring LLM inference server performance. 9 services communicate via ZMQ message bus. Update this guide only for major architectural shifts.

**Principles**: KISS + DRY. Extensibility + Usability + Accuracy + Scalability. One concern per PR. Review own diff first.

## Rules (Read First)
**CRITICAL (YOU MUST):**
- async/await for ALL I/O (no `time.sleep`, no blocking calls)
- `Field(description="...")` on EVERY Pydantic field
- Type hints on ALL functions (params and return)
- KISS + DRY: minimal code, optimize for reader

**Architecture:**
- BaseComponentService for services (BaseService for SystemController only)
- AIPerfBaseModel for data; BaseConfig for configuration
- Message bus for inter-service communication (no shared mutable state)
- YAML plugin registry for extensible features (plugins.yaml)

**Code Quality:**
- Lambda for expensive logs: `self.debug(lambda: f"{self._x()}")`
- Tests: fixtures + helpers + `@pytest.mark.parametrize`

## Patterns

See [docs/dev/patterns.md](docs/dev/patterns.md) for detailed code examples.

| Pattern | Key Points |
|---------|------------|
| **Service** | `BaseComponentService` + `@on_message` + `plugins.yaml` registration |
| **Models** | `AIPerfBaseModel` for data, `BaseConfig` for config, `Field(description=...)` always |
| **Messages** | Set `message_type`, use `@on_message(MessageType.X)`, auto-subscribes at `@on_init` |
| **Plugin** | YAML registry, `plugins.get_class(PluginType.X, 'name')`, lazy-loaded |
| **Errors** | `self.error(f"msg: {e!r}")` + `ErrorDetails.from_exception(e)` in response |
| **Logging** | Lambda for expensive: `self.debug(lambda: f"{len(x)}")`, direct for cheap |
| **JSON** | Always `orjson.loads(s)`, `orjson.dumps(d)` |

## Base Classes & Mixins

- **BaseComponentService**: For 8 services (heartbeat + registration with SystemController)
- **BaseService**: For SystemController only (no heartbeat/registration)
- **AIPerfLifecycleMixin**: For standalone components (`CREATED`→`INITIALIZING`→`INITIALIZED`→`STARTING`→`RUNNING`→`STOPPING`→`STOPPED`; `FAILED` terminal)

**Decorators:** `@on_init`, `@on_start`, `@on_stop`, `@on_message`, `@on_command`, `@background_task`, `@on_pull_message`, `@on_request`

**Communication:** `publish()` for broadcast, `@on_message` to subscribe, `send_command_and_wait_for_response()` for sync

## Anti-Patterns
- One PR = one goal (no scope creep)
- Comments only for "why?" not "what"
- No shared mutable state between services
- No blocking I/O (use async alternatives)
- No `Optional[X]` or `Union[X, Y]` (use `X | Y`)

## Key Directories
```
src/aiperf/
├── cli_commands/      # CLI command handlers
├── common/            # Base classes, mixins, models, messages
├── controller/        # SystemController service
├── credit/            # Credit system for flow control
├── dataset/           # DatasetManager service
├── endpoints/         # API endpoint implementations
├── exporters/         # Results export (CSV, JSON, console)
├── gpu_telemetry/     # GPUTelemetryManager service
├── metrics/           # Metric definitions and calculations
├── plot/              # Plotting and visualization
├── plugin/            # Plugin system (plugins.py, plugins.yaml)
├── post_processors/   # Record and results processors
├── records/           # RecordProcessor, RecordsManager services
├── server_metrics/    # ServerMetricsManager service
├── timing/            # TimingManager service
├── transports/        # HTTP transport implementations
├── ui/                # Textual UI components
├── workers/           # Worker, WorkerManager services
└── zmq/               # ZMQ communication layer
tests/
├── harness/               # Test harness for mocking plugins and services
├── aiperf_mock_server/    # Mock server for integration tests
├── unit/                  # Fast isolated tests
├── component_integration/ # Integration tests with mocked communication and single process
└── integration/           # End-to-end tests with real communication and multiple processes
```

## Services
**SystemController**: orchestration, lifecycle management
**DatasetManager**: prompt/token generation
**TimingManager**: request scheduling, credit issuance
**WorkerManager**: worker lifecycle, health monitoring
**Worker** (N): LLM API calls, conversation state
**RecordProcessor** (N): metric computation, scales with load
**RecordsManager**: record aggregation
**GPUTelemetryManager**: GPU telemetry from DCGM
**ServerMetricsManager**: server metrics from Prometheus

Communication: ZMQ message bus via `await self.publish(msg)`. Services auto-subscribe based on `@on_message` decorators during `@on_init`.

## Enums
All enums use `ExtensibleStrEnum` or `CaseInsensitiveStrEnum`:
- **DO**: `MessageType.MY_MSG` (use directly)
- **DON'T**: `MessageType.MY_MSG.value` (no `.value` needed)

## Testing

**Auto-fixtures** (always active): asyncio.sleep runs instantly, RNG=42, singletons reset between tests.

**Commands:**
- `uv run pytest tests/unit/ -n auto` - Fast, isolated, mock dependencies
- `uv run pytest -m integration -n auto` - Full system, real services in multiple processes
- `uv run pytest -m component_integration -n auto` - Component integration and cli tests in single process

**Conventions:**
- `@pytest.mark.asyncio` for async tests, `@pytest.mark.parametrize` for data-driven
- `from tests.harness import mock_plugin` for plugin mocking in tests
- Name: `test_<function>_<scenario>_<expected>` e.g. `test_parse_config_missing_field_raises_error`
- Imports at file top, fixtures for setup, one focus per test

See [docs/dev/patterns.md](docs/dev/patterns.md) for code examples.

## Package Management
Always use `uv` (never pip): `uv add package`, `uv run pytest`
- `make first-time-setup` - Initial environment setup
- `make install` - When dependencies are missing

## Verification Commands
```bash
ruff format . && ruff check --fix .   # Format and lint
uv run pytest tests/unit/ -n auto    # Unit tests in parallel
uv run pytest -m integration -n auto   # Integration tests in multiple processes
uv run pytest -m component_integration -n auto # Component integration and cli tests in single process
make validate-plugin-schemas           # Validate plugin registry
pre-commit run                         # Pre-commit on staged files
```

## Pre-Commit Checklist
1. Review diff: all lines required?
2. `ruff format . && ruff check --fix .`
3. `uv run pytest tests/unit/ -n auto`
4. Type hints on all functions
5. `Field(description=...)` on all Pydantic fields
6. `git commit -s`

## Gotchas
- **SystemController uses BaseService** (not BaseComponentService) - it's the orchestrator
- **Worker/TimingManager disable GC** for latency - see `service_metadata.disable_gc`
- **macOS child processes** close terminal FDs to prevent Textual UI corruption
- **Plugin priority** resolves conflicts: higher wins, external beats built-in at equal priority
- **Enums are string-based** - use `MessageType.X` directly, never `.value`

## Common Tasks
**Service**: BaseComponentService → add to `plugins.yaml` under `service` category
**Message**: Enum `common/enums/enums.py` → class `messages/` → `@on_message()`
**Plugin**: Create class → add to `plugins.yaml` with `class`, `description`, `metadata` → validate with `aiperf plugins --validate`

**Build systems that scale. Write code that lasts.**
