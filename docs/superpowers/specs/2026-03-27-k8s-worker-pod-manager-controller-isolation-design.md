# Kubernetes worker-pod-manager controller isolation design

Date: 2026-03-27
Status: Proposed
Scope: Kubernetes mode only
Compatibility: No backwards-compatibility layer required; replace the current Kubernetes startup and readiness contract outright.

## Summary

In Kubernetes mode, only `WorkerPodManager` should communicate with `SystemController` over the control channel. Workers and record processors should no longer register with, receive commands from, or publish controller-facing readiness directly to the controller.

`WorkerPodManager` becomes the sole controller-facing authority for a worker pod. It owns:
- pod registration
- pod aggregate readiness and degraded-capacity reporting
- pod-local lifecycle coordination for sibling workers and record processors
- pod-local dataset authority for workers
- churn-safe recovery for both workers and whole worker pods

The credit router should separately track worker transport connectivity and worker dispatch eligibility. A worker may connect to the router early, before dataset availability, but must not receive credits until it explicitly becomes dispatchable.

The design must support:
- workers coming and going during an active benchmark
- whole worker pods coming and going during an active benchmark
- the controller deciding when to start the benchmark once enough worker pods are ready, including after a waiting period if the full desired topology has not arrived
- partial pod capacity remaining usable

## Current code patterns to replace

### Controller currently expands Kubernetes topology to child services

`SystemController` currently treats Kubernetes worker and record-processor counts as controller-visible required services in `src/aiperf/controller/system_controller.py:167`.

That causes the controller to wait for:
- `WORKER_POD_MANAGER`
- every `WORKER`
- every `RECORD_PROCESSOR`

This is the main coupling that should be removed.

### Controller currently gates benchmark start on per-worker READY

After configure, `SystemController` currently:
1. checks pod health
2. waits for all expected workers to reach app-level `READY`
3. checks endpoint readiness
4. sends `PROFILE_START`

That logic lives in `src/aiperf/controller/system_controller.py:484` and `src/aiperf/controller/system_controller.py:779`.

In Kubernetes mode, worker startup truth currently reaches the controller through `WorkerStatusSummaryMessage` published by `WorkerPodManager` in `src/aiperf/workers/worker_pod_manager.py:663` and consumed by the controller in `src/aiperf/controller/system_controller.py:846`.

### Router currently overloads `WorkerReady`

The credit router currently treats `WorkerReady` as both presence and dispatch eligibility in:
- `src/aiperf/credit/messages.py:69`
- `src/aiperf/credit/sticky_router.py:494`
- `src/aiperf/credit/sticky_router.py:525`

This must be replaced. The router should know the difference between:
- worker connected to the router
- worker eligible to receive credits

### Dataset handoff is currently rebroadcast-oriented

Dataset configuration is currently pushed as `DatasetConfiguredNotification` and rebroadcast every second by `DatasetManager` in `src/aiperf/dataset/dataset_manager.py:397` and `src/aiperf/dataset/dataset_manager.py:407`.

In Kubernetes mode:
- `WorkerPodManager` downloads the dataset after receiving `DatasetConfiguredNotification` in `src/aiperf/workers/worker_pod_manager.py:216`
- workers wait for `PodDatasetReady` in `src/aiperf/workers/worker.py:323`
- `WorkerPodManager` re-sends dataset-ready for late workers in `src/aiperf/workers/worker_pod_manager.py:225` and `src/aiperf/workers/worker_pod_manager.py:312`

This is good enough for initial startup but not the cleanest churn-safe model.

### Dataset HTTP API currently exposes file blobs, not a queryable benchmark snapshot

`DatasetRouter` currently exposes:
- `GET /api/dataset/data`
- `GET /api/dataset/index`

in `src/aiperf/api/routers/dataset.py:133` and `src/aiperf/api/routers/dataset.py:148`.

That API is sufficient for file transfer, but it does not expose authoritative benchmark generation, join policy, or current dataset-state metadata needed for pod churn.

## Design goals

1. Only `WorkerPodManager` talks to `SystemController` in Kubernetes mode.
2. Workers and record processors use only pod-local lifecycle/state coordination plus their existing data-plane connections.
3. Router connectivity and dispatch eligibility are separate first-class states.
4. Dataset state is queryable current truth, not dependent on catching a rebroadcast.
5. Whole worker pods can join during active credit flow if they complete full startup convergence before becoming dispatchable.
6. The controller can start when enough worker pods are ready, even if the full target topology has not arrived, based on explicit policy.
7. No compatibility layer is required; simplify the Kubernetes path directly.

## Proposed architecture

### 1. Controller-visible topology becomes pod-only

In Kubernetes mode, `SystemController` should only require registration of:
- control-plane services it already manages directly
- `WORKER_POD_MANAGER` instances

It should stop expecting controller-visible registration of:
- `WORKER`
- `RECORD_PROCESSOR`

This means removing the Kubernetes-mode child expansion in `src/aiperf/controller/system_controller.py:167` and treating child service counts as pod-manager-reported capacity, not controller registration requirements.

### 2. `WorkerPodManager` becomes the sole controller-facing authority

`WorkerPodManager` should own the controller contract for a pod:
- register once per pod
- receive pod-targeted control commands
- report aggregate pod capacity
- report aggregate pod readiness
- report degraded pod state
- report pod drain completion
- report benchmark generation alignment

The existing registration payload in `src/aiperf/common/control_structs.py:35` already carries `pod_name`, `pod_index`, `num_workers`, and `num_record_processors`. The new design should keep pod-level registration but reinterpret child counts as declared pod capacity rather than controller-visible child-service expectations.

### 3. Workers and record processors become pod-local lifecycle participants only

In Kubernetes mode:
- workers do not register with the controller
- record processors do not register with the controller
- controller does not send them direct commands
- their lifecycle state is reported only to `WorkerPodManager`

The existing pod-local lifecycle channel in `src/aiperf/common/pod_lifecycle_structs.py:14` is the right foundation, but it should evolve from mostly push-style notifications into a state/query contract.

## Worker/router state model

### Replace `WorkerReady` with explicit routing-state messages

Replace the current overloaded router protocol with explicit worker-to-router state transitions:
- `WorkerConnected`
- `WorkerDispatchable`
- `WorkerUndispatchable`
- `WorkerShutdown`

`TimePing` and `TimePong` remain transport/RTT measurement only. They should not imply routing eligibility.

### Router-owned truth

`StickyCreditRouter` should maintain two separate state collections:
- connected workers
- dispatchable workers

Only dispatchable workers belong in the routing pool structures that currently back load balancing in `src/aiperf/credit/sticky_router.py:277`, `src/aiperf/credit/sticky_router.py:282`, and `src/aiperf/credit/sticky_router.py:1001`.

Recommended structure:
- `connected_workers: set[str]`
- `dispatchable_workers: dict[str, WorkerLoad]`
- load-balancing caches/indexes built only from dispatchable workers

### Router semantics

- `WorkerConnected`
  - worker is known alive on the routing channel
  - do not route credits yet
- `WorkerDispatchable`
  - create or reactivate `WorkerLoad`
  - add to dispatchable pool
- `WorkerUndispatchable`
  - remove from dispatchable pool immediately
  - preserve connected state
  - if in-flight credits remain, keep detached-drain behavior until they reconcile
- `WorkerShutdown`
  - remove from connected and dispatchable state
  - preserve existing stranded-credit reclaim behavior

### Why this is the cleanest approach

This removes ambiguous meaning from `WorkerReady` and keeps authority in the right place:
- worker owns its local convergence
- router owns dispatch eligibility
- pod manager owns aggregate pod truth
- controller sees only pod aggregates

No compatibility layer means the old `WorkerReady` contract should be deleted, not adapted.

## Pod-local state/query model

### Replace rebroadcast-style dataset handoff with queryable pod-local truth

`WorkerPodManager` should become the authoritative pod-local dataset service for sibling workers.

Instead of workers depending on rebroadcast timing, workers should query current state from `WorkerPodManager`:
- current benchmark generation
- pod admission state
- dataset generation/version
- whether dataset is available locally
- mmap metadata if available
- whether the pod is draining

### Pod-local protocol

Extend the pod lifecycle channel to support request/response style operations, not only fire-and-forget structs.

The pod-local contract should include something conceptually equivalent to:
- `PodStateQuery`
- `PodStateSnapshot`
- `PodDatasetStateQuery`
- `PodDatasetStateSnapshot`
- `PodWorkerRoutingStateUpdate`
- `PodRecordProcessorStateUpdate`

Exact names can vary, but the contract should be state-based rather than event-lucky.

### Worker startup convergence loop

In Kubernetes mode a worker should converge like this:
1. start process
2. connect to router
3. complete ping/probe and send `WorkerConnected`
4. register/query `WorkerPodManager`
5. fetch current pod/benchmark/dataset state
6. initialize dataset client if local dataset is ready
7. report pod-local startup/routing state to `WorkerPodManager`
8. send `WorkerDispatchable` to router only when all prerequisites hold

A restarted worker should repeat the same sequence with no dependency on missed events.

## Controller/pod-manager query model

### Whole pods must also recover by querying current truth

`WorkerPodManager` should not depend on catching a one-time `DatasetConfiguredNotification` or other startup-only controller event.

Instead, `WorkerPodManager` should be able to register or restart during an active benchmark and ask for the current benchmark snapshot.

### Authoritative state split

- `SystemController` owns benchmark lifecycle, start policy, and admission/drain decisions.
- dataset authority owns dataset descriptor/source metadata and transfer endpoints.
- `WorkerPodManager` pulls current truth, converges locally, and reports pod aggregate state back upward.

### Controller-facing snapshot contract

Add a queryable controller/API snapshot for pod managers that contains at least:
- `benchmark_id`
- `benchmark_generation`
- benchmark phase/state (`configuring`, `admitting`, `profiling`, `draining`, `completed`, `aborted`)
- current pod-admission policy
- dataset generation/version
- dataset metadata needed to validate local files
- dataset fetch URLs / API base URLs
- controller decision parameters relevant to startup/admission

This should be available through a query path that is safe for late-joining pod managers.

### API direction

Because `WorkerPodManager` already downloads files over HTTP using `dataset_api_base_url` in `src/aiperf/workers/worker_pod_manager.py:421`, the cleanest design is to add HTTP snapshot endpoints under the controller API rather than inventing another controller control-channel RPC for pod managers.

Recommended direction:
- keep data/index file endpoints
- add a benchmark-state snapshot endpoint
- add a dataset-state snapshot endpoint

This keeps the churn-safe query path aligned with the existing HTTP-based pod-manager download behavior.

## Benchmark generation and dataset generation

Both worker-level and pod-level churn require versioned current truth.

Introduce explicit generation/version identifiers for:
- benchmark generation
- dataset generation
- optionally pod admission generation if controller policy changes independently

These generation IDs must be carried through:
- controller snapshot responses
- pod-manager aggregate reports
- pod-local worker snapshots
- router/pod-manager local state where needed for validation

A worker or pod manager that sees mismatched generations must re-converge before becoming dispatchable.

## Controller start policy

### Current behavior to replace

The controller currently waits for all expected workers to become READY in `src/aiperf/controller/system_controller.py:779`. That is too rigid for the new design and still tied to child-level readiness.

### New behavior

The controller should make a pod-level start decision using aggregate pod-manager reports.

Each `WorkerPodManager` should report at least:
- declared worker capacity
- declared record-processor capacity
- router-connected worker count
- credit-eligible worker count
- ready worker count
- ready record-processor count
- degraded worker count
- degraded record-processor count
- benchmark generation currently loaded
- pod state (`starting`, `ready`, `degraded`, `draining`, `failed`)
- admission state (`admitting`, `dispatchable`, `closed`, `draining`)

### Start decision rule

The controller should support starting the benchmark once enough worker pods are ready, even if not all desired pods have arrived, provided a configurable waiting condition has been satisfied.

Recommended policy model:
- `target_worker_pods`: desired total pods
- `minimum_ready_worker_pods`: minimum pod count required to start
- `minimum_credit_eligible_workers`: optional absolute worker threshold
- `pod_start_grace_period`: how long the controller waits for more pods before deciding whether to start with partial capacity

Controller decision sequence:
1. enter pod-admission phase after configuration
2. query/collect aggregate pod-manager reports
3. if full target capacity arrives before grace period, start immediately
4. otherwise, once grace period expires, start if minimum readiness thresholds are met
5. if thresholds are not met, keep waiting until a hard timeout or fail according to policy

This allows the controller to make targeted decisions about when the best time to start is, rather than treating startup as a binary all-or-nothing worker registration gate.

### During active credit flow

Late-arriving pods may join while credits are already flowing if:
- they load the current benchmark generation
- they load the current dataset generation
- they complete the full pod-local convergence sequence
- their workers become router-dispatchable only after full local readiness

The controller should treat these pods as additional admitted capacity once their aggregate report reaches dispatchable state.

## Aggregate pod-state reporting

The current `WorkerStatusSummaryMessage` in `src/aiperf/common/messages/worker_messages.py:34` is worker-centric. For the new design, Kubernetes mode should move to an explicit pod-aggregate message rather than continuing to push child-worker detail into the controller.

Recommended new controller-facing message family:
- `PodCapacityReport`
- `PodReadinessReport`
- `PodAdmissionReport`
- `PodDrainReport`
- or a single `WorkerPodStateMessage` carrying a full snapshot

The single-snapshot approach is cleaner because it is idempotent and easy to reconcile.

Recommended contents of a single pod snapshot:
- `pod_index`
- `pod_name`
- `benchmark_generation`
- `dataset_generation`
- `declared_workers`
- `declared_record_processors`
- `router_connected_workers`
- `dispatchable_workers`
- `ready_workers`
- `ready_record_processors`
- `degraded_workers`
- `degraded_record_processors`
- `pod_state`
- `admission_state`
- `draining`
- `failed`
- timestamps/versioning metadata

The controller should store the latest snapshot per pod and make start/admission decisions from those snapshots.

## Partial pod capacity

Partial pod capacity is explicitly allowed.

A pod is usable if it still has enough local capacity to make forward progress. Recommended default rule:
- usable if `dispatchable_workers >= 1` and `ready_record_processors >= 1`
- degraded if some expected local children are missing or failed but the pod remains usable
- unavailable if no dispatchable worker capacity or no viable record-processing path remains

The controller should not treat any single child failure as automatic whole-pod failure.

## Record processor behavior

Record processors also stop talking directly to the controller in Kubernetes mode.

They should:
- register with `WorkerPodManager`
- expose local readiness/state to `WorkerPodManager`
- participate in pod-local queryable state
- continue their existing data-plane/results flow

Their local availability should be part of the pod snapshot. This is important because pod admission and degraded-state decisions depend on both worker and record-processor capacity.

## Shutdown and drain

Controller drain/shutdown should be pod-targeted only.

Flow:
1. controller sends drain/stop command to `WorkerPodManager`
2. pod manager marks pod non-admitting and coordinates local shutdown
3. workers become `WorkerUndispatchable` before losing eligibility
4. in-flight credits drain or reconcile through the router’s existing reclaim paths
5. record processors flush local outputs
6. pod manager waits for local completion and performs final uploads if needed
7. pod manager reports drain completion upward

The current raw-record upload path in `src/aiperf/workers/worker_pod_manager.py:770` should remain pod-manager-owned.

## What to delete or simplify

Because compatibility is not required, the Kubernetes path should be simplified aggressively:
- remove controller expectation that workers and record processors register directly in Kubernetes mode
- remove overloaded use of `WorkerReady`
- remove Kubernetes worker dependence on rebroadcasted `DatasetConfiguredNotification`
- remove pod-manager re-broadcast logic for late worker subscribers
- stop using worker-centric controller readiness as the Kubernetes benchmark-start gate
- replace worker-centric `WorkerStatusSummaryMessage` usage with pod-centric aggregate state for Kubernetes mode

## Testing plan for the eventual implementation

### Router state tests
- connected worker is not dispatchable by default
- `WorkerConnected` does not add worker to routing pool
- `WorkerDispatchable` adds worker to routing pool
- `WorkerUndispatchable` removes worker from routing pool without losing connected identity
- dispatchable pool alone drives `send_credit()` selection

### Worker churn tests
- worker restarts before dataset availability and later becomes dispatchable
- worker restarts during active profiling and is not sent credits until fully converged
- stale worker generation prevents dispatchability until re-sync

### Pod churn tests
- `WorkerPodManager` starts after benchmark admission has already begun and converges from snapshot query
- pod manager restarts during active profiling and re-forms local authority without requiring replayed events
- late pod joins during credit flow and adds capacity only after dispatchable aggregate report

### Controller start-policy tests
- controller starts immediately when full target pod readiness arrives before grace period
- controller starts with partial ready pods once grace period expires and minimum thresholds are met
- controller does not start if minimum thresholds are not met
- late pods can join after benchmark start without destabilizing controller state

### Dataset-state tests
- worker can query dataset state before local dataset is ready and retries cleanly
- pod manager can query current dataset/benchmark snapshot without having seen earlier events
- dataset generation rollover forces re-convergence and prevents stale dispatchability

### Degraded-capacity tests
- pod remains usable with partial worker loss
- pod remains usable with partial record-processor loss when minimum viable path exists
- pod becomes unavailable only when usable thresholds are no longer met

## Recommended implementation direction for the later plan

1. Redefine Kubernetes controller topology as pod-manager-only.
2. Introduce new router worker-state messages and remove `WorkerReady` semantics.
3. Refactor `StickyCreditRouter` to track connected vs dispatchable workers separately.
4. Introduce pod-local query/snapshot protocol between workers/record processors and `WorkerPodManager`.
5. Introduce controller/API query snapshot for pod-manager churn recovery.
6. Replace worker-centric Kubernetes startup gate with pod-centric aggregate admission/start policy.
7. Delete rebroadcast-based Kubernetes dataset handoff code paths.
8. Add churn-focused tests before implementation is considered complete.

## Final recommendation

Implement the Kubernetes runtime around queryable current truth and explicit dispatchability, with `WorkerPodManager` as the only controller-facing authority and the controller making benchmark-start decisions from aggregate pod readiness rather than child-service registration.
