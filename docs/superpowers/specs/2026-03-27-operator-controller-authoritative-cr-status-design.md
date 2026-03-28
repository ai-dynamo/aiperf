# Operator controller-authoritative CR status design

Date: 2026-03-27
Status: Proposed
Scope: Kubernetes operator status reporting for `AIPerfJob`
Compatibility: Update status semantics to match the current controller truth; preserve existing top-level status fields where practical.

## Summary

The operator currently leaves the CR in `Initializing` even after the `SystemController` has already started profiling. This happens because the operator still derives the `Running` transition from worker readiness counts instead of using the controller as the authoritative status source.

The fix is to make the controller the source of truth for operator status throughout. The operator should consume controller progress and controller-exposed aggregate worker-pod state, and use Kubernetes JobSet state only for bootstrap, controller-unavailable fallback, and terminal infrastructure failure handling.

## Problem statement

Today, `src/aiperf/operator/handlers/monitor.py` derives top-level phase from a mix of sources:
- JobSet replicated job readiness
- app-level worker startup counts fetched via `/api/workers`
- controller progress fetched via `/api/progress`

That split ownership creates an inconsistency:
- the controller may already have started profiling
- but the operator keeps `status.phase=Initializing`
- because the operator still waits for an all-workers-ready condition before setting `Running`

This is stale and incorrect. The controller has already made the start decision using pod-local aggregate logic, so the operator should reflect that decision directly.

## Design goals

1. `SystemController` is the source of truth for benchmark status everywhere.
2. `status.phase` reflects controller benchmark state, not operator-inferred worker readiness.
3. `status.currentPhase` and live progress remain aligned with controller progress.
4. `status.workers` reflects controller-authoritative aggregate pod-local readiness/dispatchability, not raw JobSet pod readiness.
5. JobSet status remains limited to bootstrap and infrastructure fallback cases.
6. The design removes duplicated start logic from the operator.

## Current code paths

### Operator monitor

`src/aiperf/operator/handlers/monitor.py` currently:
- reads JobSet replicated job status
- computes `workers_ready` from JobSet worker readiness
- fetches app worker startup state via `ProgressClient.get_worker_startup_states()`
- promotes to `Running` only when `app_workers_ready == total_workers`
- fetches `/api/progress` later for phase progress and metrics

This is the core mismatch. The operator uses controller progress for progress details, but not as the authoritative lifecycle source.

### Controller truth

`src/aiperf/controller/system_controller.py` now starts profiling from aggregate pod-local readiness logic. The controller already tracks `WorkerPodStateMessage` snapshots and decides when there is sufficient dispatchable pod capacity to start.

That means the controller, not the operator, owns the real answer to:
- whether initialization is still in progress
- whether profiling has started
- what worker capacity is actually usable

### Progress client limitation

`src/aiperf/operator/progress_client.py` currently exposes:
- `/api/progress` parsed into `JobProgress`
- `/api/workers` parsed into per-worker startup states

That worker endpoint is still worker-centric. To make the controller authoritative throughout, the operator needs controller-exposed aggregate worker-pod status instead of reconstructing readiness from per-worker startup states.

## Proposed design

### 1. Controller-authoritative phase ownership

The operator should derive `status.phase` from controller-reported benchmark state whenever the controller API is reachable.

Required semantics:
- `Pending` / `Queued` may still come from operator/bootstrap state before controller progress is available.
- `Initializing` means the controller exists but has not yet started profiling.
- `Running` means the controller has started profiling.
- `Completed`, `Failed`, and `Cancelled` continue to reflect terminal truth, with completion/failure recovery logic unchanged where needed.

The operator must stop gating `Running` on `app_workers_ready == total_workers`.

### 2. Controller-authoritative worker aggregate status

The CR should stop treating `status.workers.ready/total` as raw JobSet or per-worker startup-derived counts.

Instead, the operator should publish a controller-authoritative aggregate view derived from pod-local state already owned by the controller.

The status schema should expand `status.workers` to support aggregate fields such as:
- `ready`
- `total`
- `dispatchable`
- `routerConnected`
- `readyRecordProcessors`
- `declaredRecordProcessors`
- `readyPods`
- `totalPods`
- `degradedPods`

Exact field names can follow existing project naming conventions, but the key requirement is that `status.workers` becomes an aggregate status block driven by controller truth rather than inferred operator logic.

The existing `ready` and `total` fields should remain if practical so existing consumers still have a simple summary.

### 3. Progress API contract grows to include aggregate worker-pod status

The cleanest implementation is to extend the controller API contract so the operator can fetch one controller-authored snapshot containing:
- progress phases
- current benchmark phase
- aggregate worker-pod status
- optional pod-state details if needed later

This can be done either by:
- extending `/api/progress` to include aggregate worker status, or
- adding a dedicated controller endpoint for aggregate worker-pod status and having the operator fetch both endpoints.

Recommendation: extend the progress response so the operator consumes one authoritative status snapshot.

That keeps lifecycle state and aggregate readiness coupled to the same controller decision point.

### 4. Operator monitor becomes consumer, not decider

In `src/aiperf/operator/handlers/monitor.py`:
- fetch controller progress as early as possible once the controller is reachable
- use controller progress/current phase to drive top-level CR phase
- use controller aggregate worker snapshot to populate `status.workers`
- keep JobSet replicated-job counts only as fallback/bootstrap data before the controller responds

JobSet state remains responsible for:
- queued/suspended bootstrap status
- initial worker pod existence hints before the controller API responds
- infrastructure failure detection if the controller becomes unavailable
- sidecar salvage / terminal recovery paths already present

The operator should no longer duplicate controller readiness/start rules.

### 5. Status builder supports aggregate worker status

`src/aiperf/operator/status.py` should gain a worker-aggregate setter that writes the expanded `status.workers` payload.

The old `set_workers(ready, total)` helper is too narrow for the new semantics. It should either:
- be replaced with a richer aggregate setter, or
- delegate to the richer setter for backward-compatible simple cases.

The builder should remain the single place where operator status shape is written.

## Data-flow changes

### Before

1. Operator reads JobSet ready counts.
2. Operator fetches worker startup states.
3. Operator infers whether all workers are ready.
4. Operator decides whether benchmark is `Running`.
5. Operator fetches controller progress for secondary details.

### After

1. Operator reads bootstrap JobSet state only until controller responds.
2. Operator fetches controller progress snapshot.
3. Controller progress snapshot supplies benchmark lifecycle truth.
4. Controller progress snapshot supplies aggregate worker-pod truth.
5. Operator writes CR status from controller truth.
6. Operator falls back to JobSet/infrastructure handling only when controller truth is unavailable.

## Testing requirements

### Operator monitor tests

Add or update tests to prove:
- when controller progress reports profiling started, the CR becomes `Running` even if aggregate worker readiness is still partial
- operator no longer requires all workers ready before setting `Running`
- bootstrap still uses JobSet state before controller progress is available
- fallback behavior remains correct when controller API is unavailable

### Progress client / API contract tests

Add or update tests to prove:
- aggregate worker-pod status is parsed from the controller API correctly
- malformed or partial aggregate payloads degrade safely
- controller progress and aggregate worker status stay internally consistent in the parsed client model

### CRD schema tests

Add or update tests to prove:
- the expanded `status.workers` schema accepts the new aggregate shape
- existing simple fields remain present if compatibility is preserved

## Documentation updates

If the CR status shape or semantics change, update:
- `docs/architecture.md` for controller-authoritative Kubernetes status behavior
- any operator-facing docs that describe CR status fields

## Final recommendation

Make the controller authoritative for operator CR status throughout. The operator should stop inferring benchmark lifecycle from worker readiness and instead project controller progress plus controller aggregate worker-pod status into the CR. This fixes the current stale `Initializing` state and keeps operator status aligned with the real runtime decisions already made by the controller.
