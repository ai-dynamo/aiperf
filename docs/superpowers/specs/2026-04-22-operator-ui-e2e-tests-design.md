# Design: Operator Web UI End-to-End Test Suite

**Status:** Draft
**Date:** 2026-04-22
**Author:** acasagrande@nvidia.com
**Scope:** Add a live-browser integration test suite for the operator web UI served by `aiperf.operator.results_server`.

---

## 1. Overview

Add `tests/e2e/operator_ui/` — a Playwright-driven suite that exercises the operator's Preact SPA in a real headless Chromium browser against a live, in-process `create_app()` FastAPI instance.

Coverage target: **full behavior** across all six pages (dashboard, jobs, job-detail, leaderboard, compare, history) — page loads, data rendering, interactions (sort, filter, command palette, navigation, compare flow, error banner).

Opt-in via `@pytest.mark.e2e`; never runs in the default unit or integration suite. Installs via a new `make install-e2e-browsers` target.

### Goals

1. Catch UI regressions that API-level tests cannot detect — client-side routing, component state, chart rendering, keyboard shortcuts, CSS layout breakage that masks interactive elements.
2. Document observable UI behavior through readable, role-based selectors (`get_by_role`, `get_by_text`) — the test file doubles as user-facing behavior spec.
3. Run fast and deterministic — zero network egress, zero cluster, sub-60s full suite.

### Non-goals

- Performance / load testing of the UI.
- Visual-regression snapshot testing (deferrable to a later spec).
- Testing the Textual terminal UI (already covered by `tests/unit/ui/` and `tests/integration/test_dashboard_ui.py`).
- Testing the production result server's file-download behavior beyond what the UI itself exercises (covered by `tests/unit/operator/test_results_server.py`).

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────┐
│ pytest session                                             │
│                                                            │
│  session fixture: live_operator_app                        │
│    • create_app(results_dir=<tmp>)                         │
│    • uvicorn.Server on 127.0.0.1:<random>                  │
│    • background asyncio task                               │
│    • yields base_url                                       │
│                                                            │
│  function fixture: seeded_results_dir                      │
│    • copies tests/fixtures/operator_ui/results/ to tmp     │
│    • builder overrides (empty, single-job, all-failed)     │
│                                                            │
│  function fixture: fake_k8s_client                         │
│    • monkeypatch aiperf.kubernetes.client.{list_aiperf_    │
│      jobs, find_aiperf_job, get_pods, cluster_version,     │
│      get_raw_aiperfjob_status, cancel_aiperf_job}          │
│    • default canned response; per-test overrides           │
│                                                            │
│  function fixture: page (pytest-playwright)                │
│    • route("**/esm.sh/**")  → serves ui/vendor/*           │
│    • route("**/cdn.jsdelivr.net/**") → serves vendor/*     │
│    • console errors collected; fails test on any           │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
                  Chromium (headless)
                          │
                          ▼
              http://127.0.0.1:<port>/
              (real UI + real FastAPI + real routers)
```

### Key seams

- **App boot via `create_app(results_dir=...)`** — already takes `results_dir`; the test fixture constructs a `tmp_path` tree and passes it in directly.
- **In-process uvicorn** — `uvicorn.Server(Config(app, host="127.0.0.1", port=0))` in a background asyncio task. `server.started` event is awaited before yielding. A random free port is bound by passing `port=0` and reading `server.servers[0].sockets[0].getsockname()[1]` after startup.
- **K8s client mocking** — the jobs router calls a small number of helpers from `aiperf.kubernetes.client`: `list_aiperf_jobs`, `find_aiperf_job`, `get_raw_aiperfjob_status`, `get_pods`, `cluster_version`, `cancel_aiperf_job`. The `fake_k8s_client` fixture monkeypatches these to return canned `V1Pod`/`V1Node`/dict structures from `tests/fixtures/operator_ui/k8s/`. The router's own error handling, serialization, and auth paths run unmocked.
- **CDN interception** — Playwright's `page.route("**/esm.sh/**", ...)` intercepts every CDN request the UI issues for `preact`, `preact/hooks`, `htm/preact`, `@preact/signals`, plus the chart.js CDN. The handler serves bytes from `src/aiperf/operator/ui/vendor/` with the correct MIME type. No UI code change.

### Why not other approaches

- **Subprocess uvicorn** — rejected: 2–3s startup cost × 30 tests; harder to inject the mocked ApiClient into the running process.
- **Mock at Playwright layer only** — rejected: bypasses the real router code, defeats the purpose of an end-to-end suite.
- **Switch UI to vendored-first** — deferred: that's a production UI change, out of scope for "add a test suite." The route-interception approach leaves the UI untouched.
- **Golden fixtures only** — rejected: edge-case scenarios (empty state, all-failed) are more ergonomic to express as builder fixtures than to hand-craft as directory trees.
- **Builder fixtures only** — rejected: realistic assertions ("row for `aiperf-llama3-c128` shows TTFT p99 = 42.1ms") are more valuable as committed golden data; builders would produce synthetic numbers that drift from the real output format.

---

## 3. Components

### 3.1 Test fixtures — `tests/e2e/operator_ui/conftest.py`

**Session-scoped:**
- `live_operator_app` → `LiveApp` dataclass with `base_url: str`, `app: FastAPI`, `results_dir: Path`. Creates a tmp `results/` dir, calls `create_app(results_dir=...)`, starts uvicorn once per session. Per-test fixtures mutate the contents of this dir (never its path) and monkeypatch the k8s helpers — no respawn.

**Function-scoped:**
- `seeded_results_dir` → `Path` — base fixture: clears `live_operator_app.results_dir` and copies `tests/fixtures/operator_ui/results/` into it. Returns the path.
- `empty_results_dir` — yields an empty tmp dir; tests for the "no data" dashboard state.
- `single_job_results_dir` — builder: one synthetic job with minimal valid outputs.
- `all_failed_results_dir` — builder: jobs with `status: Failed` conditions and no metrics.
- `fake_k8s_client(monkeypatch)` — patches the six `aiperf.kubernetes.client` helpers; default canned response from `tests/fixtures/operator_ui/k8s/`; per-test overrides via `fake_k8s_client.set_jobs([...])`, `.set_pods([...])`.
- `page` override — wraps `pytest-playwright`'s `page` with CDN route interception and a console-error collector that fails the test on any `page.on("pageerror")` or `console.error(...)` event.

### 3.2 Golden fixtures — `tests/fixtures/operator_ui/`

- `results/aiperf-bench/aiperf-llama3-c128/` — complete job output: `profile_export_aiperf.json`, `profile_export_aiperf.csv`, `profile_export_aiperf.parquet`, conditions JSON, `.aiperf_results_ready.json` marker.
- `results/aiperf-bench/aiperf-llama3-c256/` — second job, same model, different concurrency; enables leaderboard sort + compare.
- `results/ml-lab/mistral-7b-run1/` — third job, different namespace + model; enables namespace-filter tests.
- `results/ml-lab/failed-run/` — job with no metrics and a `Failed` condition; enables error-state tests.
- `k8s/jobs.json` — canned `list_aiperf_jobs` response matching the four jobs above plus one `Running` job with no results yet (for the live-status path).
- `k8s/pods.json` — canned pod list for `get_pods`.
- `k8s/version.json` — canned `cluster_version` response.

Total size target: **under 500 KB committed**. If parquet pushes over, generate it at test-session start from a small Python script in `conftest.py` (builder path).

### 3.3 Builder — `tests/e2e/operator_ui/_builders.py`

Small helpers to synthesize result directories and k8s objects for edge cases:

```python
def build_empty_results(base: Path) -> None: ...
def build_single_job(base: Path, *, job_id: str, metrics: dict[str, float]) -> None: ...
def build_all_failed(base: Path, *, n: int) -> None: ...
def build_running_job_cr(name: str, namespace: str) -> dict: ...
```

### 3.4 Page-object helpers — `tests/e2e/operator_ui/_pages.py`

Thin page-object wrappers so test bodies read like behavior specs. Not a full POM — just a handful of methods per page that wrap the most common selector chains:

```python
class JobsPage:
    def __init__(self, page: Page, base_url: str): ...
    async def goto(self) -> None: ...
    async def rows(self) -> list[Locator]: ...
    async def sort_by(self, column: str) -> None: ...
    async def filter_namespace(self, ns: str) -> None: ...
    async def click_row(self, job_id: str) -> "JobDetailPage": ...

class JobDetailPage: ...
class LeaderboardPage: ...
class ComparePage: ...
class HistoryPage: ...
class DashboardPage: ...
```

### 3.5 Test files — one per page, ~5 tests each

- `test_dashboard.py` — loads, summary cards populate, empty-state renders, error banner on backend 500.
- `test_jobs.py` — table loads, namespace filter, column sort, row click navigates to detail, live-status column reflects `fake_k8s_client` state.
- `test_job_detail.py` — metrics render, conditions render, chart draws, pods list renders, cancel button (fake k8s returns 200).
- `test_leaderboard.py` — ranked rows, metric selector changes ranking, click-through to job detail.
- `test_compare.py` — select two jobs, compare page renders side-by-side, metric selector changes charts.
- `test_history.py` — chart renders with multi-run series, metric selector changes series, date-range filter.
- `test_navigation.py` — top-nav links, breadcrumb, command palette (Ctrl+K), deep-link routing, 404 route.
- `test_robustness.py` — no console errors on any page, all images/fonts load, all fetches return <500, survives a route change mid-load.

---

## 4. Data flow

### 4.1 Page-load happy path

1. Test calls `page.goto(base_url + "/jobs")`.
2. Chromium requests `/jobs` → FastAPI's `StaticFiles` returns `index.html`.
3. `index.html` imports `./app.js` and resolves bare specifiers via the import map → requests to `https://esm.sh/preact@10` etc.
4. Playwright's route handler intercepts each esm.sh request and serves the matching file from `src/aiperf/operator/ui/vendor/`.
5. `app.js` runs, mounts Preact app, navigates to `/jobs` route, and fetches `/api/v1/jobs`.
6. FastAPI's jobs router calls `list_aiperf_jobs(api, ...)` — monkeypatched by `fake_k8s_client` to return canned job list.
7. Preact renders the jobs table. Test asserts row count, column values, and interaction behavior.

### 4.2 Seeded-results happy path

Per-test data swaps without respawning uvicorn:

1. At session start, `live_operator_app` creates a single session-scoped `tmp_path / "results"` and passes it to `create_app(results_dir=...)`. The FastAPI app, DuckDB connection, and routers all bind to this single path for the entire session.
2. Per-test `seeded_results_dir` fixture fully clears the session results dir (`shutil.rmtree(results_dir); results_dir.mkdir()`), then copies the golden tree into it.
3. Because DuckDB reads parquet files lazily per query (no cached catalog of file paths — see `ResultsDB.query()` in `src/aiperf/operator/results_db.py`), new files are picked up on the next query. No DB reconnect needed. If profiling shows this assumption wrong, add `app.state.db = ResultsDB(...)` reconnection to the fixture's teardown.
4. Test navigates to `/leaderboard`; UI fetches `/api/v1/analytics/leaderboard?metric=ttft_p99`.
5. Router queries parquet files in the freshly seeded dir, returns ranked JSON.
6. UI renders ranked table; test asserts row order and displayed values.

### 4.3 Console-error collection

The overridden `page` fixture attaches:

```python
errors: list[str] = []
page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
page.on("console", lambda msg: errors.append(f"console.{msg.type}: {msg.text}") if msg.type == "error" else None)
request.addfinalizer(lambda: _assert_no_console_errors(errors))
```

Every test fails if any unexpected console error or unhandled exception fires. Whitelist is empty by default; tests opt-in to specific expected errors via a `allow_console_errors` marker.

---

## 5. Error handling

- **Uvicorn fails to bind** — fixture raises `RuntimeError("failed to start test server")`; session aborts. No retry.
- **Playwright browser not installed** — `pytest-playwright` produces a clear skip/error; document the fix (`make install-e2e-browsers`) in the suite's README.
- **CDN route handler miss** — if a request arrives for a CDN URL we haven't mapped to a vendored file, the handler logs a warning and falls through to the real network. A test assertion at session-teardown fails if any fall-through occurred. Rationale: a new CDN dep landing in the UI should surface as a test failure, not a silent flake.
- **Fake k8s helper called with unexpected args** — `fake_k8s_client` is strict: raises `AssertionError` if a monkeypatched function is called with an argument combination the fixture wasn't primed for. Forces tests to declare their expectations.
- **Flake control** — `pytest.ini` sets `--max-worker-restart=0` for the `e2e` suite so a hanging test aborts the run instead of retrying. Playwright's default 30s action timeout kept; page loads use `await page.wait_for_url(...)` instead of arbitrary sleeps.

---

## 6. Testing conventions

- All tests `@pytest.mark.asyncio` + `@pytest.mark.e2e`; no unmarked tests in the e2e tree.
- Selectors: prefer `page.get_by_role("button", name="Compare")` and `page.get_by_test_id(...)` over CSS selectors. Add `data-testid="..."` attributes to components as needed during implementation — one tactical change to UI components is in scope.
- One focused assertion per test; don't stack independent scenarios.
- Async fixtures, no `asyncio.sleep` — always `expect(locator).to_be_visible()` or `wait_for_url`.
- No test depends on another test's state.
- Run command: `uv run pytest tests/e2e/ -m e2e -n auto`.

### Running locally

```bash
# One-time setup
make install-e2e-browsers      # runs `uv run playwright install chromium`

# Run the suite
uv run pytest tests/e2e/ -m e2e -n auto

# Run a single page's tests with the browser visible
uv run pytest tests/e2e/operator_ui/test_jobs.py -m e2e --headed

# Debug a failing test
uv run pytest tests/e2e/operator_ui/test_jobs.py::test_sort_by_ttft -m e2e --headed --pdb
```

### CI

- New job `e2e-operator-ui` in the existing CI workflow. Installs Playwright chromium, runs `uv run pytest tests/e2e/ -m e2e -n auto`.
- Uploads Playwright traces (`on-first-retry`) as CI artifacts on failure for post-mortem debugging.
- Does not block merge initially; promoted to required after the suite is stable for two weeks.

---

## 7. Dependencies

New dev dependencies (added via `uv add --dev`):

- `playwright` (Python bindings)
- `pytest-playwright` (pytest plugin — `page`, `browser_context_args`, `--headed`, etc.)

Existing deps reused:
- `pytest`, `pytest-asyncio`, `pytest-xdist` (already in dev deps)
- `uvicorn`, `fastapi`, `httpx` (production deps)
- `orjson` (production dep, for fixture JSON)

Makefile additions:
- `install-e2e-browsers:` → `uv run playwright install chromium`
- `test-e2e:` → `uv run pytest tests/e2e/ -m e2e -n auto`

Pyproject marker addition:
```toml
"e2e: marks tests as browser-based end-to-end UI tests (requires playwright chromium, deselected by default)",
```

---

## 8. Out-of-scope decisions deliberately deferred

These are NOT part of this spec — each should be its own follow-up if/when prioritized:

- **Visual regression** — Playwright supports screenshot comparison (`expect(page).to_have_screenshot()`). Adds maintenance burden and is noisy on font/rendering diffs. Defer.
- **Vendored-first UI** — swapping `index.html`'s import map from CDN to vendored would eliminate the need for route interception entirely and make the UI airgap-ready. Worth doing, separate from this spec.
- **Cross-browser coverage** — Firefox/WebKit adds ~5min CI. Chromium-only until a specific bug surfaces.
- **Authenticated scenarios** — the operator UI has no auth today; when auth lands, this suite should grow a fixture for logged-in sessions.
- **Textual terminal UI e2e** — separate concern, already has coverage.

---

## 9. Open questions

None currently. All scope questions resolved during brainstorming:

- Coverage level: **C — full behavior** (30+ tests).
- CDN handling: **D — Playwright route interception of esm.sh → vendored files**.
- Data seeding: **C — hybrid golden fixtures + builder for edge cases**.
- K8s API: **A — mock `ApiClient` via existing seam (monkeypatch six helpers)**.
- App runner: **A — in-process uvicorn on random port, session-scoped**.
- Marker / CI: **A — `@pytest.mark.e2e` + opt-in CI job**.
