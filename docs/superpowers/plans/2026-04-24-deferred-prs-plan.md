# Deferred PRs implementation plan

> **For agentic workers:** Five independent tasks in one parallel wave. Each in an isolated worktree branched from `ajc/k8s`. Merge back via cherry-pick.

**Spec:** `docs/superpowers/specs/2026-04-24-deferred-prs-design.md`

**Per-task contract:**
- DCO sign-off: `git commit -s` with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- ONE `uv run pytest -n auto tests/unit/` per task.
- `make check-ergonomics && make check-ruff-baselined` — must be green, 0 new violations.
- `ruff format` + `ruff check --fix` on touched files before commit.
- **No `git stash` or `git restore`** — user standing rule, shell-blocked. Use `git checkout HEAD -- <path>` if you need to discard working-tree changes.

---

## Task 1: Run-diff UI (`views/compare.js`)

**Files:**
- Create: `src/aiperf/operator/ui/views/compare.js`
- Modify: `src/aiperf/operator/ui/app.js` — route `#/compare/:ns/:name/:a/:b` → `Compare` view.
- Modify: `src/aiperf/operator/ui/views/run.js` — add "Compare with…" dropdown next to the existing run-history dropdown.
- Modify: `src/aiperf/operator/ui/lib/api.js` — add `fetchRunSummary(ns, name, epoch)` that GETs `/api/v1/results/<ns>/<name>/runs/<epoch>/profile_export_aiperf.json` and returns the parsed top-level metrics.
- Create: `tests/e2e/operator_ui/test_compare.py` — one Playwright assertion that seeding two runs + navigating to `#/compare/...` renders a 2-column table.

**Steps:**

1. Inspect existing view patterns (`run.js`, `archive.js`) and the signals-based state module in `lib/state.js`. Understand how existing views fetch + render.
2. Implement `fetchRunSummary`:
   ```javascript
   export async function fetchRunSummary(namespace, jobId, epoch) {
     const url = `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(jobId)}/runs/${encodeURIComponent(epoch)}/profile_export_aiperf.json`;
     const resp = await fetch(url);
     if (!resp.ok) throw new Error(`fetchRunSummary ${namespace}/${jobId}/${epoch}: ${resp.status}`);
     return await resp.json();
   }
   ```
3. Implement `views/compare.js`. Extract the following top-level metric keys from each summary:
   - `request_throughput.avg` (req/s)
   - `request_latency.avg`, `.p50`, `.p99` (ms)
   - `time_to_first_token.avg`, `.p50`, `.p99` (ms)
   - `inter_token_latency.avg` (ms)
   - `output_token_throughput.avg` (tok/s)
4. Render a table with columns `Metric | Run A (<ts-a>) | Run B (<ts-b>) | Δ`. For the Δ column, compute `(b-a)/a * 100`, round to 1 decimal. Color cue: green if the delta direction is "better" for that metric (higher throughput = green; lower latency = green), red if worse, gray if |Δ| < 1%.
5. Add the routes to `app.js` — the regex literal in `/compare` was already anticipated in a comment (`app.js:20`). Follow the existing `router.js` pattern.
6. Add the "Compare with…" dropdown in `run.js`. Populate from the same `listRuns` helper used by the history dropdown. On change, navigate to the compare URL.
7. Add the E2E test — seed two runs with different metric values; navigate; assert two columns and one delta cell.
8. Run: `uv run pytest -n auto tests/unit/`, `make check-ergonomics`, `make check-ruff-baselined`. Commit:
   ```
   feat(ui): add run-diff view comparing two epochs side-by-side
   ```

**Scope guardrails:**
- Single-metric subset; no multi-select, no >2-run support.
- If a summary file is missing (e.g. legacy run with no profile_export), show a graceful message in the affected column.
- Delta-sign convention per metric must be correct — latency lower-is-better, throughput higher-is-better. Encode this in a small map.

---

## Task 2: Per-run timeline chart in `views/run.js`

**Files:**
- Modify: `src/aiperf/operator/ui/views/run.js` — add `LatencyTimelineChart` component below the existing summary metrics.
- Modify: `src/aiperf/operator/ui/lib/api.js` — add `fetchRunRequests(ns, name, epoch)` (or extend existing export-fetcher) that returns the raw per-request array.

**Steps:**

1. Inspect `profile_export_aiperf.json` structure:
   ```bash
   grep -rn "request_latency\|records\|profile_export_aiperf" src/aiperf/ | head -20
   ```
   Identify how per-request records are serialized. The exporter lives under `src/aiperf/exporters/`. Look for the records array key.

2. Implement `fetchRunRequests(namespace, jobId, epoch="latest")`. When `epoch="latest"`, use `/api/v1/results/<ns>/<jobId>/profile_export_aiperf.json`; else `/runs/<epoch>/...`.

3. Implement the chart:
   ```javascript
   function LatencyTimelineChart({ data }) {
     // data: array of per-request records; each has an end-to-end latency (ms)
     const canvasRef = useRef(null);
     useEffect(() => {
       if (!canvasRef.current || !data?.length) return;
       const ctx = canvasRef.current.getContext("2d");
       const sampled = stride(data, 10000);  // defensive downsample
       const chart = new window.Chart(ctx, {
         type: "line",
         data: {
           labels: sampled.map((_, i) => i),
           datasets: [{
             label: "End-to-end latency (ms)",
             data: sampled.map(r => r.end_to_end_latency_ms),
             borderWidth: 1,
             pointRadius: 0,
           }],
         },
         options: {
           parsing: false,
           animation: false,
           scales: { x: { title: { display: true, text: "Request index" } }, y: { title: { display: true, text: "Latency (ms)" } } },
         },
       });
       return () => chart.destroy();
     }, [data]);
     return html`<canvas ref=${canvasRef} style="max-height: 300px;" />`;
   }
   ```
   Exact field name for latency: verify against the actual export. If it's `end_to_end_latency.ms` or similar, adjust.

4. Mount below the existing run view's summary. Wrap in a `try/catch` that renders an error placeholder on parse failure.

5. Chart.js is vendored at `src/aiperf/operator/ui/vendor/chart.umd.min.js` and loaded via CDN in `index.html` (double-loaded for CDN-fallback behavior). Use `window.Chart`. Pull dark-theme defaults from `src/aiperf/operator/ui/lib/chart-theme.js`.

6. Stride-sample helper:
   ```javascript
   function stride(arr, maxPoints) {
     if (arr.length <= maxPoints) return arr;
     const step = Math.ceil(arr.length / maxPoints);
     return arr.filter((_, i) => i % step === 0);
   }
   ```

7. No new automated test needed (UI chart). Manual smoke check via dev server is sufficient; if an E2E harness exists, add one "chart canvas is present" assertion.

8. Run: `uv run pytest -n auto tests/unit/`, `make check-ergonomics`, `make check-ruff-baselined`. Commit:
   ```
   feat(ui): add request-latency timeline chart to run detail view
   ```

**Scope guardrails:**
- One chart (latency). TTFT and throughput charts are out of scope.
- Stride-sample above 10k points; fail gracefully if the file is > 200 MB (don't try to parse huge files client-side).

---

## Task 3: `aiperf kube results --run <epoch>`

**Files:**
- Modify: `src/aiperf/cli_commands/kube/results.py` — add `--run` param to the default `@app.default` download command.
- Modify: `src/aiperf/cli_commands/kube/results.py` — update the `_run_results` internal to route through `/runs/<epoch>/...` when `--run` is set.
- Modify: `tests/unit/operator/test_cli_kube_results_list.py` (or wherever the existing `results` tests live) — add 2 tests: URL routing and artifact-dir naming.

**Steps:**

1. Inspect the current `@app.default` signature and `_run_results` implementation.
2. Add param:
   ```python
   run: Annotated[str | None, Parameter(name="--run", help="Pin to a specific historical run (epoch seconds from `aiperf kube results list-runs`). Default: latest.")] = None,
   ```
3. Validate in the top of `_run_results`:
   ```python
   from aiperf.operator.results_layout import EPOCH_RE
   if run is not None and not EPOCH_RE.match(run):
       raise ValueError(f"Invalid --run value '{run}'. Expected decimal epoch-seconds or 'legacy'.")
   ```
4. Route HTTP paths based on `run`:
   - `None` → existing `/api/v1/results/<ns>/<job>/...` paths.
   - set → `/api/v1/results/<ns>/<job>/runs/<run>/...` for list, zip bundle, and file downloads.
5. Artifact-dir default: if `run` is set and `output` is None, set output to `Path(f"./artifacts/{namespace}__{job_id}__{run}")`.
6. Update `list-runs` text output footer to hint: `\nPass --run <epoch> to \`aiperf kube results\` to pin a historical download.`
7. Tests:
   - `test_results_routes_through_runs_prefix_when_run_set` — mock HTTP; assert the URL passed to the client contains `/runs/<epoch>/`.
   - `test_results_artifact_dir_includes_epoch_when_run_set` — assert the default output path has the epoch.
8. Run verification + commit.

**Guardrails:** Do not change existing default behavior (run=None is unchanged).

---

## Task 4: Retention dry-run / preview

**Files:**
- Modify: `src/aiperf/operator/results_layout.py` — `enforce_retention(..., dry_run: bool = False)`.
- Modify: `tests/unit/operator/test_results_layout.py` — 2 dry-run tests.
- Create: `src/aiperf/operator/routers/config.py` (or extend existing) — `GET /api/v1/config/retention` returning `{retain_runs, retain_days}`.
- Modify: `src/aiperf/operator/results_server.py` — register the new router.
- Modify: `src/aiperf/cli_commands/kube/results.py` — add `--preview` to `list-runs`.
- Modify: `tests/unit/operator/test_cli_kube_results_list.py` — 2 tests for `--preview`.

**Steps:**

1. Extend `enforce_retention`:
   ```python
   def enforce_retention(
       base, namespace, name, *,
       keep, protect_epoch, retain_days=0, dry_run=False,
   ) -> list[str]:
       # ... existing logic ...
       for r in runs:
           if r.name in count_keepers:
               continue
           if cutoff is not None and r.stat().st_mtime >= cutoff:
               continue
           if dry_run:
               deleted.append(r.name)
               continue
           try:
               shutil.rmtree(r)
               deleted.append(r.name)
           except OSError as exc:
               logger.warning(...)
       return deleted
   ```
   Update docstring.

2. Add 2 tests to `test_results_layout.py`:
   - `test_enforce_retention_dry_run_returns_candidates_without_deleting` — assert the dirs still exist after dry_run.
   - `test_enforce_retention_dry_run_matches_live_candidates` — run twice (once dry, once live) with same fixture; assert same deleted list.

3. Add the config endpoint. Create `src/aiperf/operator/routers/config.py` with a `create_config_router()`:
   ```python
   from fastapi import APIRouter
   from pydantic import BaseModel, Field

   from aiperf.operator.environment import OperatorEnvironment


   class RetentionConfigResponse(BaseModel):
       retain_runs: int = Field(description="Current RETAIN_RUNS setting.")
       retain_days: int = Field(description="Current RETAIN_DAYS setting (0 = disabled).")


   def create_config_router() -> APIRouter:
       router = APIRouter(prefix="/api/v1/config", tags=["config"])

       @router.get("/retention", response_model=RetentionConfigResponse)
       async def get_retention_config() -> RetentionConfigResponse:
           return RetentionConfigResponse(
               retain_runs=OperatorEnvironment.RESULTS.RETAIN_RUNS,
               retain_days=OperatorEnvironment.RESULTS.RETAIN_DAYS,
           )
       return router
   ```
   Register in `results_server.py::create_app` alongside existing routers.

4. Add `--preview` to `list-runs`:
   ```python
   preview: Annotated[bool, Parameter(name="--preview", help="Show which runs would be reaped under current retention settings.")] = False,
   ```
   When set:
   - Also fetch `GET /api/v1/config/retention` for current retention settings.
   - Compute `would_delete` per run. The latest run (is_latest=True) is always protected. Sort runs by mtime desc → keep `retain_runs` newest → mark as "would delete" everything else that also fails the age policy (mtime < now - retain_days*86400, when retain_days > 0). Mirror the `enforce_retention` dry-run logic.
   - Text output: add a `WOULD DELETE` column.
   - JSON output: add `would_delete: bool` to each entry + a top-level `retention: {retain_runs, retain_days}` block.

5. Tests:
   - `test_list_runs_preview_marks_old_for_deletion` — mock `/runs` and `/config/retention`; assert the `would_delete` column/field matches the expected candidates.
   - `test_list_runs_preview_protects_latest` — even with `retain_runs=0`, latest must not be marked.

6. Run verification + commit (single commit for retention mechanics + CLI flag + endpoint).

**Guardrails:** CLI never actually deletes. Only the operator success-gate invokes live retention. Preview is read-only.

---

## Task 5: CLI docs generator recurses into sub-apps

**Files:**
- Modify: `tools/generate_cli_docs.py`.
- Regenerate: `docs/cli-options.md`.

**Steps:**

1. Inspect `tools/generate_cli_docs.py` to find where it walks commands (`app.commands`, `app._commands`, or similar — cyclopts-specific).
   ```bash
   grep -n "for .* in app\|\.commands\|\.subapps\|\.sub_apps\|App" tools/generate_cli_docs.py | head -20
   ```

2. Identify the cyclopts sub-app API. `App` instances have a `_commands` dict or similar. Walk it recursively:
   ```python
   def walk_commands(app, prefix: list[str] | None = None, depth: int = 0):
       if depth > 5:
           return  # guard against cycles
       prefix = prefix or []
       for name, cmd in app._commands.items():  # adjust to actual attribute
           full_name = [*prefix, name]
           if isinstance(cmd, App):  # nested sub-app
               yield from walk_commands(cmd, full_name, depth + 1)
           else:
               yield full_name, cmd
   ```

3. Replace the existing flat iteration with the recursive walk. Ensure the generator renders nested commands under a parent header (e.g. `## aiperf kube results list-runs`, not a flat list).

4. Regenerate:
   ```bash
   make generate-cli-docs
   grep "aiperf kube results list-runs" docs/cli-options.md
   ```
   Expected: `list-runs` appears with its params.

5. If any `--check` mode exists (e.g. `make check-cli-docs`), run it:
   ```bash
   ./tools/generate_cli_docs.py --check
   ```
   Must pass.

6. Run `uv run pytest -n auto tests/unit/` (full suite), `make check-ergonomics`, `make check-ruff-baselined`.

7. Commit:
   ```
   fix(docs): CLI docs generator recurses into sub-app commands
   ```

**Guardrails:** No new validation logic; no refactor beyond the recursion. Depth bound of 5 is a belt-and-braces measure against accidental cycles.

---

## Execution

Single wave, 5 parallel worktreed agents. After all five return, cherry-pick their branches onto `ajc/k8s` in any order (all disjoint except Tasks 3 and 4 both touch `src/aiperf/cli_commands/kube/results.py` and `tests/unit/operator/test_cli_kube_results_list.py` — those merges will conflict and need resolution by taking both additions).

Final gate:
```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
git log --oneline origin/main..HEAD | head -25
```

Expect 6 new feature commits (5 tasks + this plan) on `ajc/k8s`, no new ergonomics/ruff-baseline violations.
