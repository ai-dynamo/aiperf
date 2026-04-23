# Operator Web UI e2e tests

Live-browser integration tests for the FastAPI-hosted Preact SPA in
`src/aiperf/operator/ui/`. Runs real headless Chromium against an in-process
`uvicorn.Server` hosting `aiperf.operator.results_server.create_app()` bound
to a random localhost port. Backend data comes from a committed golden
results tree plus programmatic builders; the six `aiperf.kubernetes.client`
helpers the jobs router depends on are monkeypatched per-test.

The suite is deselected from default runs via the `e2e` pytest marker
(configured in `pyproject.toml`). It is exercised locally and in CI via
the dedicated workflow `.github/workflows/e2e-operator-ui.yml`.

## Running locally

One-time browser install:

```bash
make install-e2e-browsers
```

Run the whole suite:

```bash
make test-e2e
```

which expands to:

```bash
uv run pytest tests/e2e/ -m e2e -n auto
```

Run a single file with the browser visible:

```bash
uv run pytest tests/e2e/operator_ui/test_jobs.py -m e2e --headed
```

Debug a single test:

```bash
uv run pytest tests/e2e/operator_ui/test_jobs.py::test_jobs_namespace_filter \
  -m e2e --headed --pdb
```

Collect Playwright traces for a failing test (useful in CI artifacts too):

```bash
uv run pytest tests/e2e/operator_ui/test_dashboard.py -m e2e \
  --tracing=retain-on-failure
```

## How the fixture stack works

All fixtures live in `tests/e2e/operator_ui/conftest.py`.

- **`live_operator_app` (session-scoped, `loop_scope="session"`)** — spawns
  a real `uvicorn.Server` on `127.0.0.1:<random>` against a real
  `create_app(results_dir=<tmp>)`. The server stays up for the whole test
  session; per-test fixtures mutate the contents of `results_dir` in place
  rather than respawning.
- **`seeded_results_dir` (per-test)** — wipes the session results dir and
  copies the committed golden tree from `tests/fixtures/operator_ui/results/`
  into it. Each test starts from the same on-disk state.
- **`fake_k8s_client` (per-test)** — monkeypatches the six
  `aiperf.kubernetes.client` helpers that the jobs router calls
  (`list_aiperf_jobs`, `find_aiperf_job`, `get_raw_aiperfjob_status`,
  `get_pods`, `cluster_version`, `cancel_aiperf_job`) onto both the source
  modules and the router's local re-imports, and injects a non-None sentinel
  into the router's `api_holder` closure so the 503 "Kubernetes API
  unavailable" guard passes. Tests prime canned responses via the returned
  `FakeK8sClient` dataclass.
- **`playwright` / `browser` (session-scoped)** — override
  `pytest-playwright-asyncio`'s defaults to share the session event loop
  (see "Important quirks" below).
- **`context` / `page` (per-test)** — the `page` fixture installs
  `page.route("**/*")` for CDN interception and attaches listeners that fail
  the test at teardown on any `pageerror`, `console.error`, or unmapped
  external request.

## Important quirks

- **`loop_scope="session"` on `playwright`/`browser`/`context`/`page`** —
  without this, pytest-asyncio raises `ScopeMismatch` because a test that
  requests both the session-scoped `live_operator_app` and the
  (otherwise function-loop-scoped) `page` sees two different loops.
- **`pytest-playwright-asyncio`, not `pytest-playwright`** — the project
  pins `pytest-playwright-asyncio>=0.7.2` so the Playwright bindings use
  the same asyncio loop as the rest of the async test machinery. Do not
  mix `pytest-playwright` fixtures (which are sync) with these tests.
- **Hash routing (`#/jobs` etc.)** — the SPA routes via
  `window.location.hash`, and the FastAPI app only mounts `/` for the SPA.
  Navigating to `/jobs` returns 404. `_pages.py::BasePage._goto()` converts
  every route into `base_url + "/#<route>"`, with `/` short-circuited to
  the bare index.
- **`trust_env=False` on httpx** — the jobs router's internal HTTP calls
  disable `trust_env` so local CI runners with `HTTP_PROXY`/`HTTPS_PROXY`
  environment variables don't try to route `127.0.0.1` through a proxy.
- **Chart.js canvases need ~600 ms to settle** — after a canvas mounts,
  Chart.js does its first animated render asynchronously. Tests that
  assert against canvas-backed charts wait ~600 ms after the canvas is
  attached before reading pixel state or interacting.
- **`--no-verify` on commits is unrelated to e2e** — the branch carries
  fmt drift that breaks the global pre-commit fmt hook. The e2e suite
  itself has no pre-commit issue.

## CDN caching (`tests/_js_cache/`)

The UI imports ES modules from `esm.sh` and `cdn.jsdelivr.net` at runtime.
Rather than vendoring a hand-picked subset (which risks mixing pinned and
un-pinned specifiers and ending up with two distinct copies of `preact` in
the module graph), the fixture routes every CDN hit through
`page.route("**/*")` and caches the live response bytes under
`tests/_js_cache/<sha256(url)[:40]>`, with a sibling `<digest>.meta` file
recording the source URL for auditability.

- **First run (fresh checkout)** — cache misses populate from the live
  CDN; the suite needs outbound network on this one run.
- **Subsequent runs** — fully offline from the cache.
- **Committed to git** — the populated cache is checked in so CI and
  teammates do not need to re-fetch. When the UI adds a new CDN URL, run
  the suite once locally to populate, then commit the new blobs.

Unmapped external requests (anything not matching
`live_operator_app.base_url`, `CACHEABLE_HOSTS`, or `STUB_EMPTY_MAP`) are
aborted and fail the test at teardown — that's intentional. If the UI
gains a new CDN host, add it to `CACHEABLE_HOSTS` in `conftest.py`.

Font CDNs (`fonts.googleapis.com`, `fonts.gstatic.com`) are stubbed to
empty bodies via `STUB_EMPTY_MAP` — tests don't need webfonts.

## Known pre-existing skip

- `test_leaderboard_row_click_opens_job_detail` is skipped: `leaderboard.js`
  renders each `<tr>` with only a `key` prop (no `onclick`, no anchor, no
  `data-testid`), so rows are not interactive. Unskip once the UI adds row
  navigation.

Expected current baseline: **30 passed, 1 skipped**.

## Extending the suite

- **Add a new page test** — create `test_<page>.py` next to the existing
  files, follow the three-fixture pattern
  (`seeded_results_dir, fake_k8s_client, page`), drive through the page
  objects in `_pages.py`, and assert against stable
  `data-testid` selectors. Keep new test-ids kebab-case.
- **Add a new page object** — extend `_pages.py`. The page must `goto()`
  via `self._goto("/your-route")` so hash routing is applied.
- **Add a new fixture scenario** — add a builder function to `_builders.py`
  that mutates `seeded_results_dir` in place or primes the
  `FakeK8sClient` — keep builders side-effect-explicit.
- **Regenerate the golden tree** — re-run
  `uv run python tests/fixtures/operator_ui/generate_golden.py`
  and commit the diff.
- **Add a new CDN host** — append its prefix to `CACHEABLE_HOSTS` in
  `conftest.py`. First local run populates `tests/_js_cache/`; commit the
  new blobs so CI remains offline.
- **Add a new data-testid to UI code** — prefer stable kebab-case ids
  (`job-row-<ns>-<name>`, `kpi-<label>`) over nth-child selectors so tests
  survive cosmetic UI churn.
