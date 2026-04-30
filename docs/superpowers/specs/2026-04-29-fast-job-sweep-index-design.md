# Fast Job + Sweep Index Design

**Date:** 2026-04-29
**Status:** Draft, awaiting user review
**Owner:** ajc

## 1. Problem

Several operator endpoints answer "what's on the PVC and what does it look like?" by scanning the filesystem on every call:

- `results_layout.list_runs` / `list_sweep_epochs` — `iterdir()` + `stat()` per run dir, every call.
- `results_files.py` global "all jobs latest" — double `iterdir()` over `<base>/<ns>/<job>` plus a third over the latest epoch's contents.
- `results_db.ResultsDB` — globs `<base>/*/*/*/profile_export_aiperf.json[.zst]` and runs DuckDB `read_json` (with `union_by_name=true`) for `leaderboard`, `history`, `compare`, `summary`. Each call re-parses every summary file on the PVC.

A partial cache exists today (`operator/job_index.py` writes `jobs_index.json` at create + completion), but it covers only top-level jobs, only carries two metrics (throughput_rps, latency_p99_ms), and is not consulted by the DuckDB analytics path. Sweep variations are not indexed at all.

At dashboard load time the operator can issue 5–10 of these scans concurrently. Latency grows linearly with the number of run dirs on the PVC; the user has reported this is the dominant cost on production-scale PVCs.

## 2. Goals and non-goals

**Goals**

- Replace per-query filesystem scans with a single read from a structured index.
- Cover both runs (`<ns>/<job>/<epoch>/`) and sweep variations (`<ns>/sweeps/<name>/<epoch>/`).
- Carry enough metric detail in flat columns that leaderboard / history / compare run as one indexed `SELECT`.
- Carry the full per-run summary as a compressed blob so the `summary` endpoint never reads disk.
- Tolerate index loss: corruption, deletion, and out-of-band PVC edits all degrade to "slower" never "wrong".
- Replace `jobs_index.json` outright; no dual-write window.

**Non-goals**

- Multi-replica operator support. The design pins single-writer; if the operator is ever scaled up, only the kopf-owning process writes.
- Cross-PVC federation. One index per PVC; no aggregation across clusters.
- Time-series of per-request data. The index stores per-run aggregates; raw request records remain in their existing parquet files.
- Replacing the on-disk `profile_export_aiperf.json` and friends. They remain canonical; the index is a cache derived from them.

## 3. Storage

### 3.1 Location and engine

Single SQLite database at `<RESULTS.DIR>/.aiperf_index.sqlite`, opened in WAL mode (`journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`). One writer (the operator's kopf-owning process), many readers (operator FastAPI workers if any, results-server sidecar) opening with `mode=ro&cache=shared`.

WAL mode is the canonical fit for "one writer, many cross-process readers"; readers never block on the writer, the writer takes a single short `BEGIN IMMEDIATE` transaction per upsert, and `busy_timeout` absorbs incidental contention without us writing retry loops.

Pure SQLite for both write and read paths. DuckDB is removed from `results_db.py` entirely. Analytics queries (leaderboard / history / compare / summary) are flat-column `SELECT`s against indexed columns; none of DuckDB's JSON-shape features are needed once the metrics live in columns.

### 3.2 Schema

Two tables, one view, one meta table.

```sql
CREATE TABLE runs (
    namespace             TEXT    NOT NULL,
    job_id                TEXT    NOT NULL,
    epoch                 TEXT    NOT NULL,         -- decimal epoch-seconds string

    -- lifecycle
    phase                 TEXT    NOT NULL,         -- Pending|Running|Succeeded|Failed|...
    is_latest             INTEGER NOT NULL DEFAULT 0,
    start_time            TEXT,                     -- ISO-8601
    end_time              TEXT,
    created_unix          INTEGER NOT NULL,         -- creation row insert time
    mtime_epoch           INTEGER,                  -- run dir mtime, populated at completion + bootfill
    error                 TEXT,

    -- identity / spec digest
    model                 TEXT,
    endpoint              TEXT,
    gpu_count             INTEGER NOT NULL DEFAULT 0,
    gpu_name              TEXT,
    file_count            INTEGER NOT NULL DEFAULT 0,
    total_size_bytes      INTEGER NOT NULL DEFAULT 0,
    spec_json             BLOB,                     -- zstd-compressed CR spec at create time

    -- six narrow metrics, four stats each (24 columns)
    request_throughput_avg                       REAL,
    request_throughput_p50                       REAL,
    request_throughput_p99                       REAL,
    request_throughput_unit                      TEXT,
    request_latency_avg                          REAL,
    request_latency_p50                          REAL,
    request_latency_p99                          REAL,
    request_latency_unit                         TEXT,
    time_to_first_token_avg                      REAL,
    time_to_first_token_p50                      REAL,
    time_to_first_token_p99                      REAL,
    time_to_first_token_unit                     TEXT,
    output_token_throughput_avg                  REAL,
    output_token_throughput_p50                  REAL,
    output_token_throughput_p99                  REAL,
    output_token_throughput_unit                 TEXT,
    output_token_throughput_per_user_avg         REAL,
    output_token_throughput_per_user_p50         REAL,
    output_token_throughput_per_user_p99         REAL,
    output_token_throughput_per_user_unit        TEXT,
    inter_token_latency_avg                      REAL,
    inter_token_latency_p50                      REAL,
    inter_token_latency_p99                      REAL,
    inter_token_latency_unit                     TEXT,

    -- full summary blob
    metrics_json          BLOB,                     -- zstd-compressed profile_export_aiperf.json summary section

    -- sweep linkage (nullable for non-sweep runs)
    sweep_namespace       TEXT,
    sweep_name            TEXT,
    sweep_epoch           TEXT,
    sweep_variation_idx   INTEGER,

    PRIMARY KEY (namespace, job_id, epoch)
);

-- exactly one is_latest=1 row per job, enforced at the engine level
CREATE UNIQUE INDEX runs_one_latest
    ON runs(namespace, job_id) WHERE is_latest = 1;

CREATE INDEX runs_model        ON runs(model);
CREATE INDEX runs_start_time   ON runs(start_time);
CREATE INDEX runs_sweep_link   ON runs(sweep_namespace, sweep_name, sweep_epoch);

CREATE VIEW runs_latest AS
    SELECT * FROM runs WHERE is_latest = 1;

CREATE TABLE sweep_variations (
    namespace             TEXT    NOT NULL,
    sweep_name            TEXT    NOT NULL,
    sweep_epoch           TEXT    NOT NULL,
    variation_idx         INTEGER NOT NULL,

    variation_values_json BLOB    NOT NULL,         -- zstd-compressed dict of swept-param -> value
    mode                  TEXT    NOT NULL,         -- REPEATED|INDEPENDENT
    phase                 TEXT,
    pareto_rank           INTEGER,                  -- NULL when not computed
    is_best               INTEGER NOT NULL DEFAULT 0,

    -- back-pointer to the runs row produced by this variation, when one exists.
    -- Three columns instead of a packed key so joins stay indexable.
    child_namespace       TEXT,
    child_job_id          TEXT,
    child_epoch           TEXT,

    -- same six narrow metrics as runs (24 columns) — duplicated, not joined,
    -- so leaderboard-by-variation is one SELECT
    request_throughput_avg                       REAL,
    request_throughput_p50                       REAL,
    request_throughput_p99                       REAL,
    request_throughput_unit                      TEXT,
    request_latency_avg                          REAL,
    request_latency_p50                          REAL,
    request_latency_p99                          REAL,
    request_latency_unit                         TEXT,
    time_to_first_token_avg                      REAL,
    time_to_first_token_p50                      REAL,
    time_to_first_token_p99                      REAL,
    time_to_first_token_unit                     TEXT,
    output_token_throughput_avg                  REAL,
    output_token_throughput_p50                  REAL,
    output_token_throughput_p99                  REAL,
    output_token_throughput_unit                 TEXT,
    output_token_throughput_per_user_avg         REAL,
    output_token_throughput_per_user_p50         REAL,
    output_token_throughput_per_user_p99         REAL,
    output_token_throughput_per_user_unit        TEXT,
    inter_token_latency_avg                      REAL,
    inter_token_latency_p50                      REAL,
    inter_token_latency_p99                      REAL,
    inter_token_latency_unit                     TEXT,

    metrics_json          BLOB,

    PRIMARY KEY (namespace, sweep_name, sweep_epoch, variation_idx)
);

CREATE INDEX sweep_variations_best   ON sweep_variations(sweep_name, is_best);
CREATE INDEX sweep_variations_pareto ON sweep_variations(pareto_rank);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- meta keys: schema_version, last_bootstrap_unix
```

Notes on the schema:

- `metrics_json` and `spec_json` and `variation_values_json` are `BLOB` (zstd-compressed bytes), not `TEXT`. The narrow metric columns sit alongside as the indexable / sortable / filterable surface.
- Telemetry-derived `gpu_count` and `gpu_name` are flat columns populated at write time. `_summarize_telemetry` (today in `results_db.py`) moves to a write-side helper; the analytics path no longer parses telemetry JSON per request.
- Schema migrations are forward-only and additive: `meta.schema_version` is checked at startup; per-version `ALTER TABLE ADD COLUMN` runs in a transaction; columns are never dropped or renamed. The narrow metric column set is fixed for `schema_version=1`; new metrics arrive as additional columns in a v2 migration, never as ad-hoc text overlays on the v1 columns.

## 4. Module layout

New module `src/aiperf/operator/runs_index.py` owns the connection, schema bootstrap, write API, and lazy-backfill helper. The module's docstring pins the "single writer = kopf-owning process" assumption explicitly.

Public API:

```python
async def open(path: Path) -> None: ...                             # idempotent: creates/migrates schema
async def close() -> None: ...

async def upsert_run_created(ns, job_id, epoch, *, spec) -> None: ...
async def upsert_run_phase(ns, job_id, epoch, *, phase) -> None: ...
async def upsert_run_completed(
    ns, job_id, epoch, *,
    summary_blob: bytes,                                            # zstd-compressed profile_export_aiperf.json
    metrics: dict,                                                  # parsed summary, used to populate flat cols
    files: list[str],
    mtime_epoch: int,
) -> None: ...
async def upsert_run_failed(ns, job_id, epoch, *, error, phase) -> None: ...
async def set_latest(ns, job_id, epoch) -> None: ...                # flips one row to is_latest=1, others to 0
async def delete_run(ns, job_id, epoch) -> None: ...

async def upsert_sweep_variation(
    ns, sweep_name, sweep_epoch, idx, *,
    variation_values, mode, phase, metrics, child_ref,
) -> None: ...
async def mark_sweep_pareto(ns, sweep_name, sweep_epoch, *, rankings) -> None: ...

# read API used by results_layout, results_files, results_db
async def list_runs_for_job(ns, job_id) -> list[RunIndexRow]: ...
async def list_all_latest() -> list[RunIndexRow]: ...
async def list_sweep_epochs_for_sweep(ns, sweep_name) -> list[SweepEpochRow]: ...
async def get_run(ns, job_id, epoch) -> RunIndexRow | None: ...
async def get_summary_blob(ns, job_id, epoch) -> bytes | None: ...

async def leaderboard(metric, stat, order, limit, *, epoch=None) -> list[dict]: ...
async def history(*, model, endpoint, metric, stat, limit, epoch=None) -> list[dict]: ...
async def compare(job_ids, metrics, *, epoch=None) -> list[dict]: ...

# bootstrap / recovery
async def bootstrap(*, force: bool = False) -> BootstrapStats: ...
async def integrity_check() -> bool: ...
async def stats() -> dict: ...                                      # for /admin/index/stats
```

Each upsert is one `INSERT ... ON CONFLICT(...) DO UPDATE` so partial state is always consistent — no read-modify-write races. The today-needed `_index_lock` on `jobs_index.json` is removed; SQLite's `BEGIN IMMEDIATE` handles serialization, and `busy_timeout=5000` absorbs contention.

## 5. Write hooks

Six trigger points in the operator. All errors on the index path are logged but never raised — the index is a cache, not a source of truth.

| Trigger | Site (today) | New call |
|---|---|---|
| `kopf.on.create(AIPerfJob)` | `operator/handlers/create.py` | `upsert_run_created` (replaces today's `index_job_created`) |
| Phase transition | `operator/client_cache.py` and lifecycle handlers | `upsert_run_phase` |
| Run completion (results downloaded, ready marker written) | `operator/handlers/completion.py` | `upsert_run_completed` + `set_latest` |
| `write_latest()` callsite | `operator/results_layout.write_latest` | `set_latest` (folded in as a sibling write under the same `BEGIN IMMEDIATE`) |
| Sweep aggregate emitted | the operator-side handler that observes sweep aggregate-export completion | `upsert_sweep_variation` per variation, then `mark_sweep_pareto` once |
| `kopf.on.delete` and `enforce_retention` | `operator/handlers/cleanup.py`, `results_layout.enforce_retention` | `delete_run` per removed epoch |

`jobs_index.json` and `operator/job_index.py` are deleted in the same change. The CR spec moves into `runs.spec_json`. The standalone `job_spec.json` belt-and-suspenders file written by `save_job_spec_file` is unrelated and stays — it serves PVC self-describing for `kubectl cp` recovery, not index reads.

The implementation plan must grep for any other reader of `jobs_index.json` before deleting `operator/job_index.py`.

## 6. Read swap

Three consumers replace their walk/glob code with index queries. Each keeps the legacy scan path as a fallback used only when the index miss-coincides with non-empty disk state.

### 6.1 results_layout

Today's `list_runs` and `list_sweep_epochs` `iterdir()`+`stat()` per call. New shape:

```python
async def list_runs(base, ns, job) -> list[RunEntry]:
    rows = await runs_index.list_runs_for_job(ns, job)
    if not rows and job_dir(base, ns, job).is_dir():
        rows = _list_runs_from_disk(base, ns, job)               # legacy implementation, renamed
        asyncio.create_task(_lazy_backfill_runs(base, ns, job))  # fire-and-forget
    return rows
```

The legacy disk-walk functions are renamed `_*_from_disk` and serve as the fallback. The public functions (`list_runs`, `list_sweep_epochs`, `resolve_latest` if it benefits) become thin index-first wrappers. Routers (`routers/jobs.py`, `routers/results_files.py`, `routers/sweeps.py`) need no changes — the swap is invisible at their boundary.

### 6.2 results_db (analytics, biggest change)

`results_db.py` is rewritten end-to-end against SQLite via `aiosqlite`. The DuckDB import, `_find_summary_files`, `_extract_job_path_parts`, `_latest_epoch_filter`, `_epoch_clause`, `_summarize_telemetry`, and the `read_json` glob path are all removed. Helpers `_validate_identifier` and `_escape_like` stay (column names are still interpolated for the polymorphic metric / stat parameters).

```python
# leaderboard, after rewrite
sql = f"""
    SELECT namespace, job_id, epoch,
           {metric}_{stat} AS value, {metric}_unit AS unit,
           start_time, end_time, model, endpoint
    FROM runs
    WHERE {metric}_{stat} IS NOT NULL
      AND {"is_latest = 1" if epoch is None else "epoch = ?"}
    ORDER BY value {order_dir}
    LIMIT ?
"""
```

`history` is the same shape with `WHERE model LIKE ? AND start_time BETWEEN ? AND ?`. `compare` becomes one SELECT with all metric columns at once — the two-pass JSON-of-JSON pattern in `_compare_base_sql` / `_compare_metric_sql` collapses. `summary` becomes `SELECT metrics_json FROM runs WHERE ...` followed by `zstd.decompress(...)` + `orjson.loads(...)`; falls back to a direct file read only when `metrics_json` is null (mid-completion race).

### 6.3 routers/results_files

The double `iterdir()` over `<base>/<ns>/<job>` (today around lines 254 / 257 / 263) becomes `await runs_index.list_all_latest()`. Same fallback rule: empty result + non-empty `<base>` → walk + backfill.

### 6.4 The fallback contract

In all three sites, the rule is identical: **read from index, fall through to disk on miss, lazy-backfill in background**. So a partially-stale index never returns wrong data, only slow data. The fallback is what makes the design safe to ship.

## 7. Bootstrap, fallback, recovery

### 7.1 Startup scan

`runs_index.bootstrap()` runs once during operator startup, scheduled as an asyncio task so it does not block the kopf event loop or the readiness probe.

```
1. Walk <base>/<ns>/<job>/ excluding name == "sweeps" (collision: <ns>/sweeps/ would
   otherwise look like a job dir).
2. For each (ns, job), list_run_epochs(); for each unknown epoch (PK miss),
   call _index_run_from_disk(ns, job, epoch).
3. _index_run_from_disk only indexes runs whose .aiperf_results_ready.json marker
   is present — otherwise the run is still being written and a partial row would
   be captured.
4. Backfilled rows set is_latest from latest.txt, not from "newest mtime in the
   table" — keeps the disk pointer authoritative.
5. Walk <base>/<ns>/sweeps/<name>/<epoch>/ for variations, mirror the same logic
   against sweep_variations.
6. Update meta.last_bootstrap_unix.
```

Bootstrap is bounded by the actual disk contents — one-shot, expected sub-minute at production PVC sizes; bench numbers go in the PR description, not this spec.

### 7.2 Lazy fallback

Already shown in section 6 — every read site that misses falls through to the legacy `_*_from_disk` function and fires `asyncio.create_task(_lazy_backfill_*(...))`. The backfill helper reuses `_index_run_from_disk`, so the ready-marker and `latest.txt` rules from 7.1 apply identically.

### 7.3 Corruption recovery

`PRAGMA integrity_check` runs at startup. On failure: log `index corrupt: rebuilding`, rename the file to `.aiperf_index.sqlite.broken-<unix>` for forensics, create a fresh DB, run `bootstrap()`. Same path triggers if the schema-version row in `meta` is older than the compiled-in version *and* a migration step fails.

### 7.4 Re-imported runs

A run dir restored via `kubectl cp` of an archive after retention deleted it hits the upsert by PK and overwrites — index stays consistent. Same for any out-of-band `cp` that introduces an epoch dir the operator never saw.

### 7.5 Manual rebuild CLI

New CLI subcommand `aiperf kube index rebuild`:

- File `src/aiperf/cli_commands/kube/index.py`, registered in `cli_commands/kube/_app.py` alongside the existing kube subcommands.
- Calls a new HTTP endpoint on the operator: `POST /admin/index/rebuild`. Endpoint inherits the operator API's existing auth posture (cluster-internal today; this design does not invent new auth).
- Endpoint backgrounds a `bootstrap(force=True)` call that drops both tables, recreates them, and re-walks disk.
- CLI output goes through `aiperf.kubernetes.console`, supports `--output text|json`, downshifts the `aiperf.kube` logger to WARNING in JSON mode (per CLAUDE.md).
- Reports `{runs_indexed: N, sweep_variations_indexed: M, duration_seconds: T}`.

A companion `GET /admin/index/stats` returns `{runs_count, sweep_variations_count, db_bytes, last_bootstrap_unix, schema_version}` for confirmation and future debugging.

## 8. Testing

Three tiers, mirroring existing project conventions.

### 8.1 Unit (`tests/unit/operator/test_runs_index.py`)

- Schema-creation idempotency (call `bootstrap()` twice; no error, no duplicate rows).
- `upsert_run_*` semantics: `created → phase → completed → failed` sequence ends with the right `(phase, end_time, error)` regardless of order.
- Concurrent kopf-style `asyncio.gather([upsert_run_phase(...) for _ in N])` produces one row, no clobbering — the bug `_index_lock` papered over for `jobs_index.json`.
- `set_latest` flips exactly one row's `is_latest=1` per `(ns, job)`. The `runs_one_latest` partial index makes a second concurrent `set_latest` for a different epoch fail loudly, not silently corrupt.
- `_summarize_telemetry` parity: pre-compute at write time matches what the old DuckDB-side query produced for fixture telemetry payloads.
- `metrics_json` zstd round-trip; `summary()` falls back to file read on null blob.
- Corrupt-DB recovery: write garbage bytes, restart, assert `.broken-<unix>` rename + fresh schema.
- Schema migration: load v1 fixture DB into v2 code, assert additive columns appear with NULL, no row loss.
- Re-imported run upsert by PK overwrites cleanly.

### 8.2 Component-integration (`tests/component_integration/operator/test_runs_index_handlers.py`)

- Drive create → completion → delete handlers against a tmp PVC dir + tmp SQLite, assert post-condition row state.
- Lazy-fallback: pre-populate `<base>/<ns>/<job>/<epoch>/profile_export_aiperf.json` + ready marker, leave the index empty, hit `list_runs` and `leaderboard()`, assert correct results AND that a backfill row appears within ~100ms.
- Lazy-fallback respect-marker: pre-populate WITHOUT the ready marker, assert no backfill row written.
- Retention: call `enforce_retention(keep=2)` on a 5-epoch fixture, assert deleted epochs are gone from both disk AND `runs` table.
- Sweep collision: ensure `<ns>/sweeps/` is not treated as a job during bootstrap.

### 8.3 Audit suite (`tests/kubernetes/audit/`, `pytest -m k8s_audit`)

The existing operator-vs-bare-Job audit already runs each workflow case twice and diffs artifact trees. Add an `index_consistency` bucket: after each operator run, assert the `runs` row matches the on-disk `profile_export_aiperf.json` for the six narrow metrics (within tolerance for floats) and that `metrics_json` decompresses to a superset of those same values. This catches "index drifts from disk" silently.

### 8.4 Property tests

Hypothesis-driven `upsert_run_*` reorderings: for any permutation of `[created, phase=Running, completed]`, the final row state is invariant.

## 9. Rollout

Two PRs:

**PR 1 — Index core.** `runs_index.py`, schema, write hooks at all six trigger points, bootstrap, lazy fallback wrappers in `results_layout.py`, delete `operator/job_index.py` and all `jobs_index.json` callsites (after the implementation plan's grep confirms there are no others). The doc-sync edits (section 11) ride here. The CLI rebuild and `/admin/*` endpoints can ride here or as a micro-PR.

**PR 2 — Analytics swap.** Rewrite `results_db.py` against SQLite; delete the DuckDB JSON-glob path. Independently revertable: until PR 2 lands, the old DuckDB glob path keeps working alongside the new write hooks (the index is just unused by analytics).

Bench numbers (before / after p95 for `/leaderboard`, `/history`, `/runs`, `/files` at N=200 and N=2000 synthetic run dirs) appear in each PR description as evidence the change is worth the surface area.

## 10. Observability

- Logger `aiperf.operator.runs_index` at INFO for bootstrap counts and corruption events; DEBUG for per-row upserts.
- Two Prometheus counters on the existing operator `/metrics` endpoint:
  - `aiperf_index_writes_total{op}` — labels `op` ∈ {created, phase, completed, failed, set_latest, sweep_variation, sweep_pareto, delete}.
  - `aiperf_index_read_fallbacks_total{kind}` — labels `kind` ∈ {list_runs, list_sweep_epochs, list_all_latest, summary}. A non-trivial rate here is the early-warning that backfill is unhealthy.
- `GET /admin/index/stats` already covered in 7.5.

## 11. Documentation

Per the doc-update table in CLAUDE.md and the four-file sync rule:

- `CLAUDE.md` + `AGENTS.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` — new "Run/sweep index" subsection under the Kubernetes section, naming the file path, the lazy-fallback rule, the single-writer assumption, and the rebuild CLI.
- `docs/dev/kubernetes-flow.md` — single paragraph + sequence diagram showing index writes interleaved with handler events.
- `docs/kubernetes/results-api.md` — note that analytics are now index-backed and the cold-start cost moved to bootstrap.
- `docs/cli-options.md` regenerates automatically via `make generate-cli-docs` once `aiperf kube index rebuild` is registered.
- `llms.txt` — single line under the Kubernetes section.
- `docs/index.yml` — no entry needed (no new top-level doc, only updates to existing files).

## 12. Open questions

None blocking. The implementation plan must answer two during planning:

1. Exact file path for the sweep-aggregate hook (section 5, row 5). The spec describes the trigger; the plan pins the file.
2. Whether anything outside `operator/job_index.py` and `routers/jobs.py` reads `jobs_index.json` today (section 5 closing note). Greppable.
