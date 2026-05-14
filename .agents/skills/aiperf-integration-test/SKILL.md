---
name: aiperf-integration-test
description: Use BEFORE writing or modifying an integration test in aiperf — "add an integration test for X", "write a multi-service test", "test the full pipeline end-to-end", "verify the worker → records-manager flow", any new test file under tests/integration/ or tests/component_integration/. Picks the right tier, reuses existing fixtures, sets up the mock server correctly, and avoids the xdist gotchas that bite multi-process integration tests.
---

# AIPerf Integration Test Authoring

aiperf has three test tiers, each with distinct fixture sets, conftest behaviors, and execution semantics. Putting a test in the wrong tier (or hand-rolling fixtures the conftest already provides) produces flaky tests and slow runs.

## Test tier decision

| Tier | Path | Marker | Lives in | When |
|---|---|---|---|---|
| Unit | `tests/unit/` | (none — default) | Single process, mocked deps | Pure function tests. No services, no message bus, no I/O. |
| Component integration | `tests/component_integration/` | `@pytest.mark.component_integration` | Single process, real services, real message bus | Multi-service interactions WITHOUT subprocess isolation. Fast relative to integration. |
| Integration | `tests/integration/` | `@pytest.mark.integration` | Multi-process, real services, real mock server | True end-to-end through the CLI. Slow — subprocess spawn dominates. |
| Stress | (marker `@pytest.mark.stress`) | `@pytest.mark.stress` | Multi-process, heavy load | Throughput / soak / RSS-leak hunting. There is no dedicated `tests/stress/` directory — apply the marker inside existing files. Not run in CI by default. |

**Rule:** if a unit test fits, write it as a unit test. Component integration only when crossing service boundaries within a single process. Full integration only when you NEED a subprocess (e.g., testing the `aiperf` CLI itself, the bootstrap path, FD-close behavior on macOS).

## Existing fixtures (do not re-roll)

### `tests/integration/conftest.py`

- `aiperf_mock_server` — boots the in-repo mock on a free port; tears down on test exit.
- `cli` — `AIPerfCLI` wrapper that runs `aiperf profile ...` as a subprocess and returns parsed `AIPerfResults`. (Earlier versions of this skill called this `aiperf_cli`; the fixture is `cli`.)
- `aiperf_runner` — lower-level subprocess runner that `cli` wraps.
- `mock_server_factory`, `temp_output_dir`, `signal_cli` (returns an `AIPerfSignalCLI` instance) — additional fixtures available for less common scenarios.
- Subprocess lifecycle (timeout → SIGINT → terminate → kill) is handled inside `aiperf_runner`.
- `MALLOC_ARENA_MAX=2` is **NOT** set automatically — export it before invoking pytest manually (it's in the canonical `aiperf-pytest` command for integration runs).

### `tests/component_integration/conftest.py`

- Single-process service harness (no subprocess overhead).
- Singleton resets between tests (no factory-warning spam) via `reset_singleton_factories` autouse fixture.
- RNG re-seed (= 42) on every test via `reset_random_generator` autouse fixture.
- **`asyncio.sleep` is real here** — the instant-sleep patch only applies in `tests/unit/conftest.py`. If your component-integration test sleeps for real time, it really waits.

### `tests/harness/`

- `mock_plugin` — for plugin mocking in unit / component-integration tests. See `tests/harness/`.

**Always reuse these.** Re-rolling the mock server launch, the SpawnProcess management, or the singleton reset gets you flakes the conftest already solved.

## Steps for a new integration test

### 1. Pick the tier (see decision matrix above)

### 2. Name it

```
test_<feature>_<scenario>_<expected>.py
```

E.g. `test_credit_pipeline_with_burst_arrival_completes_under_30s.py`. The test function inside uses the same `test_<x>_<y>_<z>` pattern.

### 3. Set the marker

```python
import pytest

@pytest.mark.integration   # or @pytest.mark.component_integration
@pytest.mark.asyncio       # for async tests
def test_my_scenario(aiperf_mock_server, cli):
    ...
```

### 4. Use parametrize via the canonical pattern

```python
from pytest import param

@pytest.mark.parametrize(
    "endpoint,concurrency",
    [
        param("chat", 1, id="chat-c1"),
        param("chat", 8, id="chat-c8"),
        param("embeddings", 4, id="emb-c4"),
    ],
)  # fmt: skip
def test_endpoint_scales(endpoint, concurrency, cli):
    ...
```

`# fmt: skip` on the closing `)` line keeps ruff-format from collapsing the parametrize block (project convention).

### 5. Run

Use `aiperf-pytest` for the canonical invocation:

```bash
uv run pytest -n auto tests/component_integration/test_your_thing.py
# or for true integration:
MALLOC_ARENA_MAX=2 uv run pytest -n auto tests/integration/test_your_thing.py
```

## Common patterns

### Run aiperf CLI and assert on jsonl

```python
def test_chat_streaming_produces_ttft(cli):
    result = cli(
        "--endpoint-type", "chat", "--streaming",
        "--model", "gpt-4o-mini",
        "--request-count", "20", "--concurrency", "4",
        "--random-seed", "42", "--tokenizer", "builtin",
    )
    assert result.returncode == 0
    # `result` exposes parsed AIPerfResults; check the per-request export and aggregate
    # depending on what the test asserts. The per-request data is `profile_export.jsonl`.
    assert result.profile_export_jsonl.exists()
    records = [json.loads(line) for line in result.profile_export_jsonl.read_text().splitlines()]
    assert len(records) == 20
```

### Assert on a service's `@on_message` handler firing

```python
@pytest.mark.component_integration
async def test_handler_receives_message(service_under_test):
    received = asyncio.Event()
    service_under_test.handler_callback = lambda msg: received.set()
    await message_bus.publish(YourMessage(...))
    await asyncio.wait_for(received.wait(), timeout=2.0)
```

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll write a unit test and patch the whole service" | If you patch 5 things, you're testing patches not behavior. Use component_integration with the real services. |
| "I'll roll my own mock server launch in the test" | `aiperf_mock_server` fixture exists. Use it; it handles port-pick, /health, teardown. |
| "I'll skip MALLOC_ARENA_MAX, the conftest sets it" | The conftest does NOT set it on this branch. You must export it before invoking pytest. |
| "I'll use MagicMock for the Pydantic config" | Builds real config in at least one test. MagicMock auto-creates whatever attribute path you ask for, hiding "validator reads wrong path" bugs. |
| "Long-running integration test, I'll skip `-n auto`" | `-n auto` works fine for slow tests. The xdist worker-per-test isolation actually helps catch fixture leaks. |
| "I'll use `time.sleep` to wait for the message" | Use `asyncio.Event` / `asyncio.wait_for`. Sleeping is flaky; events are deterministic. Note: in `component_integration/`, `asyncio.sleep` is real (NOT instant-patched). The instant-patch is only in `tests/unit/conftest.py`. |
| "I'll skip the jsonl check, the CLI exit code is enough" | Exit 0 with 0 records in `profile_export.jsonl` is silent data loss. Always assert `len(records) > 0` for any test that should produce output. |

## Common mistakes

- **Picking the wrong tier.** Component integration tests in `tests/integration/` lose the single-process speed; unit tests in `tests/component_integration/` lose the speed-of-unit and the isolation isn't gained.
- **Hand-rolling the mock server launch** — `aiperf_mock_server` fixture exists. Use it.
- **Forgetting `@pytest.mark.asyncio`** on `async def test_...` functions — they silently skip.
- **`time.sleep` in async tests** — auto-fixture makes `asyncio.sleep` instant; `time.sleep` is real wall clock and flakes.
- **Cross-fixture contamination from missing singleton reset.** The component_integration conftest already resets singletons between tests; only relevant for fixtures you add.
- **Asserting on log content instead of parquet/events.** Logs are formatted; structure changes break tests; logs may not capture everything. Use the parquet / event types.

## Composition

- `aiperf-pytest` for the invocation rules.
- `aiperf-mock-server` is what the fixture wraps — read that skill to understand what flags it accepts.
- `aiperf-correctness-testing` if your test is one of a matrix of endpoint scenarios; the testing skill already encodes the matrix.
