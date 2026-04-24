# Follow-up PRs from epoch-keyed results layout

**Date:** 2026-04-24
**Branch:** `ajc/k8s`
**Depends on:** `docs/superpowers/specs/2026-04-24-uid-keyed-results-layout.md` (merged)

## Problem

The epoch-keyed results layout shipped without:
1. A way to enumerate historical runs from the HTTP API, CLI, or dashboard.
2. Age-based retention as an alternative to count-based.
3. Env-var generator coverage of `src/aiperf/operator/environment.py`.

Each is a small, independent follow-up suitable for parallel dispatch.

## Items

### A. `GET /api/v1/results/<ns>/<name>/runs` — list runs endpoint

Foundation for items B and C. Returns all uid-shaped run dirs under `<ns>/<name>/` with per-run metadata, newest first.

```json
{
  "namespace": "default",
  "job_id": "foo",
  "latest_epoch": "1714150923",
  "runs": [
    {"epoch": "1714150923", "mtime_epoch": 1714150925, "file_count": 7, "total_size_bytes": 4823912, "is_latest": true},
    {"epoch": "1714064523", "mtime_epoch": 1714064526, "file_count": 7, "total_size_bytes": 4810044, "is_latest": false}
  ]
}
```

**Implementation:**
- New `ResultsHistoryEntry` / `ResultsHistoryListResponse` schemas in `src/aiperf/operator/routers/results_schemas.py` (reuse existing naming if present).
- New function `list_runs(base, ns, name) -> list[RunEntry]` in `src/aiperf/operator/results_layout.py` (sorts by mtime desc, flags latest via `resolve_latest`).
- New route `GET /api/v1/results/{ns}/{job_id}/runs` in `routers/results_files.py`. 404 when the job has no pointer / no uid dirs.
- Unit tests in `test_results_server.py` — seed two epoch dirs, assert shape, ordering, `is_latest`.

### B. `aiperf kube results list-runs` CLI subcommand

Thin wrapper over endpoint A, with port-forward to the operator's results API (same pattern as existing `aiperf kube results`).

**UX:**
```
$ aiperf kube results list-runs foo
EPOCH        TIMESTAMP              FILES  SIZE    LATEST
1714150923   2024-04-25 18:05:23Z   7      4.6 MB  ✓
1714064523   2024-04-25 18:00:23Z   7      4.6 MB

$ aiperf kube results list-runs foo --output json
{...payload from endpoint A...}
```

**Implementation:**
- Add `list_runs` cyclopts subcommand to `src/aiperf/cli_commands/kube/results.py` (or a new `_list_runs.py` imported from `results.py`).
- Reuse `KubeManageOptions` + the existing port-forward helper.
- Text + JSON output modes (`--output text|json`, default text).
- Tests in `tests/unit/operator/test_cli_kube_results_list.py` — mock the endpoint, verify formatting, verify JSON mode downshifts logger to WARNING.

### C. Dashboard run-history dropdown

Per-job detail view in the operator UI gets a dropdown of historical runs. Selecting one navigates to a URL that pins that run.

**Scope (minimal):**
- New view state: `selectedEpoch: string | "latest"` on the job-detail page.
- Dropdown populated from endpoint A, labeled by human-readable timestamp.
- URL shape: `#/job/<ns>/<name>/runs/<epoch>` pins historical; `#/job/<ns>/<name>` still shows latest.
- File-list + download links bind to the selected epoch (already supported by `/runs/<epoch>/` routes from the main layout PR).

**Files:**
- `src/aiperf/operator/ui/views/job.js` (or equivalent) — add dropdown component + epoch-aware file-list.
- `src/aiperf/operator/ui/lib/api.js` — new `listRuns(ns, name)` client helper.
- Visual: minimal — a `<select>` next to the job title. No new CSS beyond a top-margin nudge.

**Out of scope:** diffing between runs, per-run timeline chart. Defer until users ask.

### D. `AIPERF_RESULTS_RETAIN_DAYS` — age-based retention

Complement to `RETAIN_RUNS`. Applied in the same success-gate path; the union of both policies' "keep" sets wins.

**Behavior:**
- New `_ResultsSettings.RETAIN_DAYS: int = Field(default=0, ge=0)`. 0 disables age-based retention.
- `enforce_retention` gains a `retain_days: int = 0` kwarg. When non-zero, any run dir whose mtime is older than `retain_days * 86400` seconds is eligible for deletion (intersected with the count-based reap).
- Intersection semantics: a run is deleted only if BOTH policies agree (count policy says "outside keep window" AND age policy says "older than cutoff"). Conservative — prevents accidental loss on misconfig.
- `protect_epoch` still wins over both policies.

**Files:**
- `src/aiperf/operator/environment.py` — `RETAIN_DAYS` field.
- `src/aiperf/operator/results_layout.py` — `enforce_retention` signature + logic.
- `src/aiperf/operator/handlers/completion.py` — pass `retain_days=OperatorEnvironment.RESULTS.RETAIN_DAYS` at the success-gate call site.
- `tests/unit/operator/test_results_layout.py` — 3 new tests: both-pass deletion, age-only keeps (count violated but age OK), count-only keeps (age violated but count OK).

### E. env-vars doc generator covers `operator/environment.py`

One-line extension to `tools/generate_env_vars_docs.py` so `make generate-env-vars-docs` picks up operator env vars (`AIPERF_RESULTS_*`, `AIPERF_OPERATOR_MONITOR_*`, `AIPERF_DEFAULT_IMAGE`, etc.).

**Files:**
- `tools/generate_env_vars_docs.py` — append `Path("src/aiperf/operator/environment.py")` to `ENV_FILES`.
- Category/section routing: the generator parses section headers from docstrings/class names. If that's too coarse, add a small mapping to group operator settings under a new "Operator" section.
- `docs/environment-variables.md` — regenerated; contains `AIPERF_RESULTS_DIR`, `AIPERF_RESULTS_MAX_RETRIES`, `AIPERF_RESULTS_RETRY_DELAY`, `AIPERF_RESULTS_TTL_DAYS`, `AIPERF_RESULTS_COMPRESS_ON_DISK`, `AIPERF_RESULTS_RETAIN_RUNS`, `AIPERF_RESULTS_RETAIN_DAYS` (after D), `AIPERF_OPERATOR_MONITOR_INTERVAL`, `AIPERF_OPERATOR_MONITOR_INITIAL_DELAY`, plus top-level `AIPERF_DEFAULT_IMAGE`, `AIPERF_JOB_TIMEOUT_SECONDS`, `AIPERF_POD_RESTART_THRESHOLD`, `AIPERF_ENDPOINT_CHECK_TIMEOUT`, `AIPERF_PREFLIGHT_TIMEOUT`, `AIPERF_CONFIGMAP_PROPAGATION_DELAY_SECONDS`.

## Parallelization

- **Wave 1** (3 in parallel): Items A, D, E — all independent.
- **Wave 2** (2 in parallel): Items B, C — both depend on A's endpoint.

Each item is < 250 lines of change. Total follow-up scope: ~800 lines + tests.

## Out of scope (explicit)

- Dashboard run-diff UI.
- Per-run timeline charts.
- `aiperf kube results` getting a `--run <epoch>` flag on the existing download command. Trivial add but out of scope for this follow-up batch.
- Retention dry-run / preview mode.
