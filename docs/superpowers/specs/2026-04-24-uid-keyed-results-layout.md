# UID-keyed results layout for AIPerfJob

**Date:** 2026-04-24
**Status:** Spec, pending plan
**Branch:** ajc/k8s

## Problem

Legacy dynamo `perf.yaml` used `EPOCH=$(date +%s)` baked into the artifact path:

```
/model-cache/perf/${EPOCH}_${JOB_NAME}/concurrency_${n}/
```

so `backoffLimit: 1` retries AND repeat submissions of the same-named Job landed in distinct, non-colliding directories. History was preserved by construction.

The AIPerfJob operator stores results flat:

```
<RESULTS.DIR>/<namespace>/<CR-name>/
```

— eight call sites compute this, and all eight key only by CR name. Consequence: deleting and recreating a CR with the same name (the natural `kubectl apply` workflow) **overwrites** the previous run's artifacts. Users lose history without warning.

We want the legacy "each submission is a new dir" semantics, while preserving the stable CR-name handle that `kubectl get aiperfjob`, `aiperf kube watch <name>`, and `kubectl apply -f perf.yaml` all rely on.

## Approach

Store results under `<RESULTS.DIR>/<namespace>/<name>/<uid>/` where `uid` is the Kubernetes `metadata.uid` of the CR. A single-line pointer file `<name>/latest.txt` records the uid of the most recent run.

- The CR name stays stable and user-visible.
- Every submission (including re-creates of the same name) gets a fresh uid and a fresh dir.
- The results HTTP API stays backward-compatible: existing two-arg routes resolve via `latest.txt`; a new additive route pins a specific historical run by uid.
- Retention bounds disk growth to 10 runs per `<ns>/<name>/` by default; env var overrides.

## File inventory

### New file

**`src/aiperf/operator/results_layout.py`** — ~110 lines including docstrings.

Single owner of the on-disk layout. Consumers import `run_dir`, `job_dir`, `write_latest`, `resolve_run_dir`, `enforce_retention`, `migrate_legacy_layout`.

Public API:

```python
LATEST_POINTER = "latest.txt"
UID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^legacy$")

def job_dir(base: Path, namespace: str, name: str) -> Path: ...
def run_dir(base: Path, namespace: str, name: str, uid: str) -> Path: ...
def write_latest(base: Path, namespace: str, name: str, uid: str) -> None: ...
def resolve_latest(base: Path, namespace: str, name: str) -> str | None: ...
def resolve_run_dir(
    base: Path, namespace: str, name: str, uid: str | None = None
) -> Path | None: ...
def enforce_retention(
    base: Path, namespace: str, name: str, keep: int, protect_uid: str
) -> list[str]: ...
def migrate_legacy_layout(base: Path) -> list[tuple[str, str]]: ...
def list_run_uids(base: Path, namespace: str, name: str) -> list[str]: ...
```

### Modified files — write path

| File | Line | Current | Change |
|---|---|---|---|
| `src/aiperf/operator/handlers/monitor.py` | 1053 | `dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id` | `dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)` |
| `src/aiperf/operator/handlers/completion.py` | 242 | same pattern | same replacement |
| `src/aiperf/operator/handlers/completion.py` | 373 | same | same |
| `src/aiperf/operator/handlers/_completion_fetch.py` | 401 | same | same |
| `src/aiperf/operator/job_index.py` | 198 | same | same |

All five sites receive `uid` from kopf's handler kwargs (already in scope — see `handlers/create.py:413`). For any function that doesn't yet take `uid`, thread it through as a positional argument.

After `sb.set_results_path(str(dest_dir))` at `handlers/completion.py:243`, add:

```python
write_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)
sb.set_run_uid(uid)
enforce_retention(
    OperatorEnvironment.RESULTS.DIR, namespace, job_id,
    keep=OperatorEnvironment.RESULTS.RETAIN_RUNS,
    protect_uid=uid,
)
```

This is the single atomic success gate — pointer flip + status write + retention trim all happen together, and only once results are on disk.

### Modified files — read path

| File | Line | Intent |
|---|---|---|
| `src/aiperf/operator/routers/results_files.py` | 224–228 (`_resolve_job_dir`) | Replace body: call `resolve_run_dir(base_dir, namespace, job_id)` (uid=None → latest). 404 when None. |
| `src/aiperf/operator/routers/results_files.py` | (new) | Add route `GET /results/{namespace}/{job_id}/runs/{uid}/{filename:path}`; resolve via `resolve_run_dir(..., uid=uid)`. Also `GET .../runs/{uid}` list route and `.../runs/{uid}.zip` bundle for parity. |
| `src/aiperf/operator/routers/results_files.py` | 186 (`_scan_job_dirs`) | Walk `<ns>/<name>/<uid>/` instead of `<ns>/<name>/`; collapse to one `JobEntry` per `<ns>/<name>` using latest. |
| `src/aiperf/operator/routers/results_analytics.py` | 208 | `spec_file = resolve_run_dir(base_dir, namespace, job_id) / "job_spec.json"`. 404 on None. |
| `src/aiperf/operator/routers/jobs.py` | 160 | `job_dir = resolve_run_dir(results_dir, namespace, name) or <skip>`. |
| `src/aiperf/operator/results_db.py` | 282 | `job_dir = resolve_run_dir(self._results_dir, namespace, job_id)`; 404/None → "no data". |
| `src/aiperf/operator/job_union.py` | 115–118, 191–194, 320 | Directory walk currently iterates `<name>/` children as job dirs. Change to iterate `<name>/` children as run dirs via `list_run_uids`, and treat `<name>/` itself as the job. `summary_path` becomes `resolve_run_dir(...) / _SUMMARY_FILE`. |

### Modified files — CRD / config

| File | Change |
|---|---|
| `src/aiperf/operator/status.py` | Add `set_run_uid(uid: str) -> StatusBuilder` helper that writes `status.runUid`. |
| `src/aiperf/common/environment.py` | Add `_ResultsSettings.retain_runs: int = Field(default=10, ge=1, description="Max runs kept per <ns>/<name>/ before retention trimming.")` surfaced as `AIPERF_RESULTS_RETAIN_RUNS`. |
| `deploy/helm/aiperf-operator/templates/crd.yaml` | Under `status.properties`, add `runUid: {type: string, description: "metadata.uid of the most recent successful run. Used to pin historical artifacts via /api/v1/results/<ns>/<name>/runs/<uid>/."}`. |

### Modified files — lifespan

| File | Change |
|---|---|
| `src/aiperf/operator/results_server.py` | Inside `lifespan` (line 76 area), before DB init, call `migrate_legacy_layout(base_dir)`. Log count of migrated jobs. |

## Data model

### `latest.txt`

Plain UTF-8 text file, one line, trailing newline. Contents: a single uid string matching `UID_RE` (either a canonical k8s UUID or the literal `legacy`).

```
5f8b2a3c-7d4e-4f1a-9b2c-1e3f4a5b6c7d
```

Reads: `path.read_text().strip()` — fault-tolerant to trailing whitespace.
Writes: staged write + `os.replace` for atomicity on POSIX (atomic within a filesystem; both files live under `<name>/` so this is guaranteed on ext4/xfs/NFSv4).

```python
def write_latest(base, ns, name, uid):
    parent = job_dir(base, ns, name)
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{LATEST_POINTER}.tmp"
    tmp.write_text(f"{uid}\n")
    os.replace(tmp, parent / LATEST_POINTER)
```

### `run_dir` vs `job_dir` signatures

```python
def job_dir(base: Path, namespace: str, name: str) -> Path:
    """<base>/<namespace>/<name>/ — the parent dir of all runs. Never contains artifact files directly."""
    return base / namespace / name

def run_dir(base: Path, namespace: str, name: str, uid: str) -> Path:
    """<base>/<namespace>/<name>/<uid>/ — one run's artifacts."""
    return base / namespace / name / uid
```

Invariant after migration: every file lives under a uid-shaped subdir. The only non-uid entries directly under `<name>/` are `latest.txt` and (transiently) `.latest.txt.tmp`.

## Write-path changes

Five sites today compute `dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id`. Each becomes `run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)`.

`handlers/completion.py:243` is the **single success gate**: when `set_results_path` is called, we know results are on disk. That is the site that also calls `write_latest`, writes `status.runUid`, and triggers retention. Pointer flip and retention trim run last; if the retention code raises, the pointer is already flipped (latest is always valid) and `enforce_retention` protects `uid` so it can't delete itself.

`_completion_fetch.py` at line 250/259/299 references `RESULTS.DIR` for path-safety checks only (not dir computation). Those checks use `resolve().relative_to(RESULTS.DIR.resolve())` which still works under the new layout — no change needed.

## Read-path changes

`_resolve_job_dir` is the chokepoint for five HTTP routes. New body:

```python
def _resolve_job_dir(base_dir: Path, namespace: str, job_id: str) -> Path:
    resolved = resolve_run_dir(base_dir, namespace, job_id)
    if resolved is None:
        raise HTTPException(404, f"No results for {namespace}/{job_id}")
    return resolved
```

All existing two-arg routes now silently serve the latest run.

New additive routes for historical pinning:

```python
@router.get("/results/{namespace}/{job_id}/runs/{uid}", response_model=FileListResponse)
@router.get("/results/{namespace}/{job_id}/runs/{uid}.zip")
@router.get("/results/{namespace}/{job_id}/runs/{uid}/{filename:path}")
```

All three validate `uid` against `UID_RE` before touching disk (cheap guardrail against `..` injection), then use `resolve_run_dir(..., uid=uid)`.

`_scan_job_dirs` currently yields one `JobEntry` per `<ns>/<name>/` directory. New behavior:

```python
for ns_dir in sorted(base_dir.iterdir()):
    if not ns_dir.is_dir(): continue
    for name_dir in sorted(ns_dir.iterdir()):
        if not name_dir.is_dir(): continue
        latest = resolve_run_dir(base_dir, ns_dir.name, name_dir.name)
        if latest is None: continue
        files = [f for f in latest.iterdir() if f.is_file()]
        yield JobEntry(...)
```

## Retention

Algorithm in `enforce_retention(base, ns, name, keep, protect_uid)`:

```python
def enforce_retention(base, ns, name, keep, protect_uid):
    parent = job_dir(base, ns, name)
    if not parent.is_dir(): return []
    runs = [p for p in parent.iterdir() if p.is_dir() and UID_RE.match(p.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keepers = set(r.name for r in runs[:keep])
    keepers.add(protect_uid)  # hard guarantee: never delete the just-written run
    deleted = []
    for r in runs:
        if r.name in keepers: continue
        shutil.rmtree(r, ignore_errors=False)
        deleted.append(r.name)
    return deleted
```

**Trigger:** once per successful completion, immediately after `write_latest` at `handlers/completion.py:243`. A single point of truth — no periodic sweeper, no startup sweeper. If the operator is down when a run finishes, retention gets caught up on the next successful run of the same `<ns>/<name>`.

**Edge cases:**

- `latest.txt` points at a dir we'd reap (mtime-sorted, retention boundary hits exactly on latest): `protect_uid` always wins — we add the just-written uid to `keepers` unconditionally.
- `latest.txt` points at a uid that no longer exists on disk (corruption, manual delete): `resolve_run_dir` returns None → reads 404, next successful run rewrites the pointer. Acceptable.
- Retention fails midway (I/O error, permissions): exception propagates, completion handler logs but does not fail the CR (results are already delivered — retention is bookkeeping).
- Concurrent writes to the same `<ns>/<name>/` (unusual, but possible if a user deletes+recreates fast): `os.replace` is atomic; the latter wins. The earlier retention pass might see both uids and keep both. Not a correctness issue.

## Migration shim

Runs once per operator process, inside `results_server.py::lifespan`, before DB init. Idempotent.

```python
def migrate_legacy_layout(base: Path) -> list[tuple[str, str]]:
    """Detect pre-migration <ns>/<name>/ layouts (files directly under <name>/) and relocate under <name>/legacy/.

    Idempotent: if <name>/ contains only uid-shaped children or already has latest.txt, no-op.
    Returns list of (namespace, name) pairs migrated.
    """
    migrated = []
    if not base.is_dir(): return migrated
    for ns_dir in base.iterdir():
        if not ns_dir.is_dir(): continue
        for name_dir in ns_dir.iterdir():
            if not name_dir.is_dir(): continue
            children = list(name_dir.iterdir())
            uid_children = [c for c in children if c.is_dir() and UID_RE.match(c.name)]
            has_latest = any(c.name == LATEST_POINTER for c in children)
            if uid_children or has_latest:
                continue  # already migrated, skip
            files = [c for c in children if c.is_file()]
            if not files:
                continue  # empty dir, no-op
            legacy_dir = name_dir / "legacy"
            legacy_dir.mkdir(exist_ok=True)
            for f in files:
                shutil.move(str(f), str(legacy_dir / f.name))
            # Also move any non-uid subdirs (e.g. legacy checkpoints/) under legacy/
            for sub in list(name_dir.iterdir()):
                if sub.name == "legacy": continue
                if sub.name == LATEST_POINTER: continue
                if sub.is_dir() and UID_RE.match(sub.name): continue
                shutil.move(str(sub), str(legacy_dir / sub.name))
            write_latest(base, ns_dir.name, name_dir.name, "legacy")
            migrated.append((ns_dir.name, name_dir.name))
    return migrated
```

Safety: uses `shutil.move` (rename within filesystem), not copy+delete, so it's atomic per file and crash-safe. A crash mid-migration leaves a partial `legacy/` dir; re-running the shim completes the move because `legacy/` existing doesn't block (`mkdir(exist_ok=True)`).

## CRD / schema additions

### `status.runUid`

New string field on `AIPerfJobStatus`. Written by `sb.set_run_uid(uid)` at the same point `sb.set_results_path(...)` is called.

CRD YAML:

```yaml
status:
  properties:
    runUid:
      type: string
      description: metadata.uid of the most recent successful run. Use as {uid} in /api/v1/results/<ns>/<name>/runs/<uid>/ to pin historical artifacts.
```

Pydantic model (`src/aiperf/kubernetes/models.py::CRJobStatus`):

```python
run_uid: str | None = Field(default=None, description="UID of the most recent successful run.")
```

### `AIPERF_RESULTS_RETAIN_RUNS`

New env var, surfaced via `_ResultsSettings` in `src/aiperf/common/environment.py`. Default 10, min 1. Documented in `docs/environment-variables.md` (regenerated via `make generate-env-vars-docs`).

## Tests

### Unit — `tests/unit/operator/test_results_layout.py` (new)

- `test_write_latest_atomic` — write twice, verify only the second value is readable (no partial read).
- `test_resolve_latest_missing_returns_none`.
- `test_resolve_run_dir_uid_none_uses_latest`.
- `test_resolve_run_dir_explicit_uid`.
- `test_resolve_run_dir_uid_not_on_disk_returns_none`.
- `test_enforce_retention_keeps_n_newest` — create 15 run dirs with descending mtimes, keep=10, verify 5 oldest deleted.
- `test_enforce_retention_protects_uid_even_if_oldest` — pin the oldest via `protect_uid`, verify it survives.
- `test_enforce_retention_empty_dir_noop`.
- `test_migrate_legacy_layout_relocates_files` — seed `<ns>/<name>/foo.json`, run, verify `<name>/legacy/foo.json` and `latest.txt=legacy`.
- `test_migrate_legacy_layout_idempotent` — run twice, assert second run is no-op.
- `test_migrate_legacy_layout_skips_already_migrated` — seed `<name>/<uid>/foo.json` + `latest.txt`, run, assert nothing moves.
- `test_migrate_legacy_layout_mixed_subdirs` — seed uid-shaped subdir + non-uid subdir side-by-side, assert only the non-uid one moves under legacy/.

### Unit — `tests/unit/operator/test_results_files_router.py` (modify)

- `test_list_job_files_resolves_latest` — set up two runs, verify API returns the latest's files.
- `test_historical_route_pins_uid` — verify `/runs/<uid>/` reads the older run.
- `test_historical_route_invalid_uid_rejected` — verify `/runs/../evil` returns 404/422 without disk access.

### Integration — `tests/integration/operator/test_same_name_resubmit.py` (new, marker `component_integration`)

- Submit AIPerfJob `foo`, wait for completion, record `status.runUid` and fetch results.
- Delete `foo`, resubmit identical YAML. Wait for completion. Record new `status.runUid`.
- Assert old uid's results still exist on disk at `<ns>/foo/<old-uid>/`.
- Assert new uid's results are what `/api/v1/results/<ns>/foo/` returns.
- Assert `/api/v1/results/<ns>/foo/runs/<old-uid>/profile_export_aiperf.json` still serves the old run.

## Risks and open questions

**Risks**

1. **Disk exhaustion if retention env is too high.** Mitigation: default=10 with 10GB avg run is 100GB — fits typical ops-volume sizing. Operator deployment chart should document this.
2. **NFS pointer atomicity.** `os.replace` is POSIX atomic on NFSv4 but was not on NFSv3. All supported k8s storage classes are NFSv4 or native ext4/xfs. Call out in deployment docs but don't block.
3. **HTTP API URL surface.** Adds three new routes. Verified non-conflicting with existing `/results/{namespace}/{job_id}.zip` (the `runs/` path segment disambiguates). OpenAPI schema regen needed.
4. **Migration shim at lifespan start.** If the base dir has many jobs with dense file trees, migration might add seconds to startup. Measured mitigation: `shutil.move` of a subtree is a single rename when within one filesystem, so even large job dirs migrate in milliseconds.
5. **Kopf state + kube uid mismatch on CR rename.** If someone patches `metadata.name` (rare but possible via raw API), existing state becomes stranded. Acceptable — the `status.runUid` field names the actual directory either way.

**Decisions needed before plan — self-answered defaults**

- **Retention fires from every successful run, not periodically.** Simpler; no sweeper thread.
- **Retention fails softly.** Logged warning, no CR failure.
- **Migration treats any non-uid, non-`latest.txt` subdir as pre-migration cruft** and folds it into `legacy/`. Alternative (leave non-uid subdirs in place) rejected because downstream `_scan_job_dirs` expects post-migration invariants.
- **`status.runUid` is optional.** If the operator crashes between `set_results_path` and `set_run_uid`, the CR will have a stale or missing `runUid`. Acceptable — `latest.txt` on disk is authoritative; `status.runUid` is a convenience mirror.
- **No new Helm chart values.** `AIPERF_RESULTS_RETAIN_RUNS` is plumbed via the existing env-var passthrough on the operator deployment template. Users who want to override set it in their values override file as a normal env var.
- **No dashboard changes in this PR.** The dashboard's job listing will collapse to "one entry per `<ns>/<name>`" showing latest, which is a strict improvement over today's implicit-latest behavior. Per-run history UI is a follow-up.

**Out of scope (explicit)**

- Dashboard run-history UI.
- `aiperf kube results` CLI command for listing historical runs.
- Retention by age (vs count).
- Operator-side garbage collection of orphaned pointer targets.
- Rename-CR-in-place workflow.

## Estimated scope

- 1 new file (~110 lines)
- 1 new unit test file (~200 lines, 12 tests)
- 1 new integration test (~80 lines)
- Modifications across 10 existing files (~60 lines touched total)
- 1 CRD schema addition
- 1 env var + regenerated docs

Target: single PR, one `pytest -n auto tests/unit/` run, one integration validation, under ~450 total lines of change.
