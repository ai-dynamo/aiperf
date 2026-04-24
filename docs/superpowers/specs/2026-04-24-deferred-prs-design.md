# Deferred follow-up PRs from epoch-layout work

**Date:** 2026-04-24
**Branch:** `ajc/k8s`
**Prior specs:**
- `docs/superpowers/specs/2026-04-24-uid-keyed-results-layout.md`
- `docs/superpowers/specs/2026-04-24-epoch-layout-followups-design.md`

## Problem

Five items were deferred from the epoch-keyed results layout + follow-ups:
1. Dashboard run-diff (compare two runs side-by-side).
2. Per-run timeline charts (visualize metrics over the course of a single run).
3. `aiperf kube results --run <epoch>` flag on the existing download command.
4. Retention dry-run / preview mode.
5. CLI docs generator recursing into nested `@app.command` sub-apps.

All five are independent → safe to dispatch in a single parallel wave.

## Items

### 1. Run-diff UI (`#/compare/<ns>/<name>/<epoch-a>/<epoch-b>`)

Side-by-side comparison of two runs' summary metrics. No new backend endpoint — reuses the existing `/runs/<epoch>/<filename>` download routes to pull `profile_export_aiperf.json` from each run.

**Scope (minimal):**
- New view `src/aiperf/operator/ui/views/compare.js`.
- URL: `#/compare/<ns>/<name>/<epoch-a>/<epoch-b>`. Additive — no existing route changes.
- Entry points from the existing run view: a "Compare with…" dropdown that picks a second epoch; clicking routes to the compare view.
- Rendered: a two-column table of top-level summary metrics (throughput, latency p50/p99, TTFT, OTPS). Highlight differences with a color cue (green for improvement, red for regression) using a deterministic threshold (>1% delta).
- URL must be shareable — load directly into the view without requiring the run view to be traversed first.

**Out of scope:** diff of raw request-level data, multi-run comparison (>2 runs), statistical significance testing.

### 2. Per-run timeline charts

A request-latency-over-time line chart inside the existing run view, rendered from `profile_export_aiperf.json`'s per-request records.

**Scope (minimal):**
- One chart: request-level end-to-end latency, x-axis = request index (ordinal, not wall clock — avoids edge cases with concurrent dispatches).
- Chart.js already vendored (`index.html` includes `chart.umd.min.js`); `chart-theme.js` supplies dark-theme defaults.
- Rendered below the existing summary metrics in `views/run.js`.
- Defensive: if the export file exceeds 100k points, stride-sample down to 10k points before plotting. Don't break the view if the file is missing.

**Out of scope:** TTFT / inter-token latency / tokens-per-second secondary charts (each doubles the JS surface area). Overlay of multiple runs (covered by item 1 separately). Wall-clock x-axis (harder; depends on how `profile_export_aiperf.json` encodes request start times).

### 3. `aiperf kube results --run <epoch>` flag

Add a `--run <epoch>` parameter to the default `aiperf kube results` download command that pins downloads to a specific historical run instead of latest.

**Scope:**
- Extend the `@app.default` function in `src/aiperf/cli_commands/kube/results.py`.
- New param: `run: Annotated[str | None, Parameter(name="--run", help="...")]` — validated against `EPOCH_RE` before making HTTP calls (otherwise 422 from the server anyway, but an early validation gives a cleaner error).
- When set: route fetches from `/api/v1/results/<ns>/<job>` → `/api/v1/results/<ns>/<job>/runs/<epoch>` with analogous rewrites for `.zip` and file-download subpaths.
- CLI-level default artifact dir: `./artifacts/<ns>__<job>__<epoch>/` when `--run` is set, existing scheme otherwise.
- Update the `list-runs` command's text output to include a footer hint: `pass --run <epoch> to aiperf kube results to pin a historical download`.

**Tests:** 2 tests — `--run` routes through `/runs/<epoch>/` URLs and the artifact dir changes.

### 4. Retention dry-run / preview mode

Show what `enforce_retention` *would* delete without touching disk. Useful for validating `RETAIN_RUNS` / `RETAIN_DAYS` settings before they bite.

**Scope:**
- Extend `enforce_retention(..., dry_run: bool = False)` in `src/aiperf/operator/results_layout.py`. When true, return the list of epochs that *would* be deleted without calling `shutil.rmtree`.
- Add `--preview` to `aiperf kube results list-runs`. When set, the CLI:
  1. Fetches `/runs` payload (existing behavior).
  2. Computes which runs would be reaped under current `RETAIN_RUNS` / `RETAIN_DAYS` — needs a small new operator endpoint OR the CLI re-implements the logic from `/runs` metadata. Go with the re-implementation — simpler, no new endpoint, and the retention math is simple enough.
  3. Adds a `WOULD DELETE` column to the text table; in JSON output, adds `would_delete: bool` to each run entry.
- To reflect operator-side settings, expose `GET /api/v1/config/retention` returning `{retain_runs, retain_days}`. The CLI fetches this to compute accurately. (Alternative: let user pass `--retain-runs N --retain-days N` on the CLI — worse ergonomics because they'd have to know the current values.)

**Out of scope:** Deletion from the CLI. The CLI only previews; actual deletion stays server-side after successful completions.

### 5. CLI docs generator recursion into sub-apps

`tools/generate_cli_docs.py` currently walks the top-level cyclopts `App` but doesn't recurse into `@app.command(name="…")` sub-apps, so `aiperf kube results list-runs` (and any future nested command) is invisible to `docs/cli-options.md`.

**Scope:**
- Inspect `tools/generate_cli_docs.py` to find where it walks commands.
- Cyclopts `App` objects have `.subapps` / `.commands` attributes (verify the exact API — the generator likely already uses one of them).
- Add recursion: for each command, if its value is another `App`, recurse into its commands. Bound depth to ~5 to prevent runaway.
- Regenerate `docs/cli-options.md` and verify `aiperf kube results list-runs` appears.

## Parallelization

All five items are independent. Single wave, 5 parallel subagents, isolated worktrees.

## Constraints reminder

- Do NOT use `git stash` or `git restore` — both are now blocked at the shell level (user standing rules).
- If a subagent needs to discard working-tree changes, use `git checkout HEAD -- <path>` explicitly.
- Commit on each worktree's own branch; controller cherry-picks back to `ajc/k8s`.
- `uv run pytest -n auto tests/unit/` — one invocation per task.
- `make check-ergonomics && make check-ruff-baselined` at end.
- DCO sign-off with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Model opus on every subagent dispatch.

## Estimated scope

- Item 1: ~250 lines (new view + 1 route) + minimal tests.
- Item 2: ~120 lines (chart addition) + defensive rendering.
- Item 3: ~80 lines (flag + URL rewrites) + 2 tests.
- Item 4: ~150 lines (dry_run flag, config endpoint, CLI column, preview logic) + 4 tests.
- Item 5: ~50 lines (recursion) + regenerated doc.

Total: ~650 lines. One PR would be fine but these are split for parallelism.
