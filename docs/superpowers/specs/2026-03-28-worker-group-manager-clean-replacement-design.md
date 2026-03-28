# WorkerGroupManager clean replacement design

## Goal

Replace the current `WorkerPodManager` direction with a transport-agnostic `WorkerGroupManager` that becomes the universal child-service authority across Kubernetes, local, and future SLURM run modes.

This is a clean replacement with zero backwards compatibility. `WorkerPodManager` is not treated as a stable public architecture and should not be preserved.

## Summary

`WorkerGroupManager` becomes the single controller-facing authority for a group of workers and record processors.

The controller sees only:
- control-plane services it already manages directly
- `WorkerGroupManager` instances

Workers and record processors become group-local children only. They do not:
- register with the controller
- depend on controller-facing startup probes
- consume controller/event-bus broadcasts directly during startup

The same group-local contract must work in all run modes:
- Kubernetes
- local multiprocess
- future SLURM

Run-mode differences should be isolated behind injected protocols/adapters rather than spread through core lifecycle logic with conditionals.

## Why this replacement is worth doing

The current direction leaves too much split-brain architecture in place:
- Kubernetes has a pod-manager concept while local mode does not
- workers still depend on controller-hosted PUB/SUB for startup even when pod-local coordination exists
- dataset readiness is moving toward a pod-local authority, but only in one mode
- a future SLURM mode would otherwise need to solve similar node-local orchestration again

A universal `WorkerGroupManager` boundary simplifies the system around one model:
- controller manages groups
- router manages workers
- workers talk to group manager
- run-mode-specific details are injected behind protocols

## Non-goals

- Preserving the `WorkerPodManager` name or compatibility behavior
- Supporting dual old/new architectures simultaneously
- Adding a compatibility layer for direct worker/controller lifecycle traffic
- Defining the full SLURM launch mechanism in this work

## Design principles

1. Clean replacement over migration scaffolding
2. One startup/readiness contract everywhere
3. Controller sees groups, not child services
4. Workers use group-local lifecycle/state only
5. Run-mode differences live behind protocols, not `if/else` branches in core logic
6. Dataset authority is group-local in every mode

## Current behavior to replace

Today, the intended Kubernetes direction already says only the pod manager should be controller-facing, but the implementation still leaves workers dependent on controller PUB/SUB startup connectivity.

In current code:
- workers still use `MessageBusClientMixin` startup probes against controller-hosted event-bus addresses
- workers directly consume `DATASET_CONFIGURED_NOTIFICATION`
- workers publish health/startup state onto the global event bus
- `WorkerPodManager` already owns pod-local dataset readiness and sibling coordination in Kubernetes mode

This split is exactly what should be removed.

## Proposed architecture

### 1. Rename and generalize the abstraction

Replace `WorkerPodManager` with `WorkerGroupManager`.

The new abstraction is not Kubernetes-specific. A “group” is the unit of local orchestration for child services.

Examples by run mode:
- Kubernetes: one worker pod = one group
- Local multiprocess: one local host/process family = one group
- SLURM future: one node or allocation-local worker bundle = one group

`WorkerGroupManager` owns:
- group registration with controller
- group aggregate readiness/capacity reporting
- child lifecycle coordination
- dataset authority for child workers
- command fanout to child workers/record processors
- child health/startup aggregation
- drain/shutdown coordination

### 2. Controller contract becomes group-only

The controller should only manage:
- existing control-plane services
- `WorkerGroupManager` instances

The controller must stop expecting direct registration from:
- workers
- record processors

This means child counts are group-manager-reported capacity, not controller-visible service cardinality.

## Child-service model

Workers and record processors become group-local children only.

They should not:
- register with the controller
- consume controller lifecycle directly
- require controller PUB/SUB startup probes
- depend on controller-hosted event-bus connectivity to complete startup

Instead, child startup should be:
1. start child process/container
2. connect to group-local lifecycle transport
3. query current group state snapshot
4. wait for dataset readiness and startup prerequisites
5. initialize local runtime state
6. report child-local health/startup state to `WorkerGroupManager`
7. become dispatchable only when the group manager and router contracts allow it

## Broadcast and messaging changes

### What workers currently use

Current direct worker event-bus usage is small:
- `DATASET_CONFIGURED_NOTIFICATION`
- worker health/startup updates on the shared bus

Record processors are already much closer to the desired model because their primary path is data-plane/pull-driven rather than event-bus-broadcast-driven.

### Clean replacement

Move those paths behind `WorkerGroupManager`:
- `DATASET_CONFIGURED_NOTIFICATION` is consumed by `WorkerGroupManager`, not workers
- worker health/startup updates are reported group-locally, not via global broadcast
- record processors also report group-locally
- any startup-relevant broadcast becomes queryable or forwarded by the group manager

This allows the generic worker startup dependence on `MessageBusClientMixin` PUB/SUB probing to be removed.

## Dataset authority model

`WorkerGroupManager` becomes the universal dataset authority for child workers.

The worker-facing contract is identical in every run mode:
- workers query group-local dataset state
- workers wait for a `dataset_downloaded` / dataset-ready concept in the current group snapshot
- workers initialize from group-provided mmap/client metadata
- workers do not care how the dataset became available

Run-mode differences only affect acquisition strategy:
- Kubernetes: group manager downloads/stages dataset and marks ready
- Local: group manager attaches to local mmap/dataset state and exposes the same ready snapshot
- SLURM future: group manager uses node/allocation-local staging, but exposes the same worker-facing contract

This unifies child behavior and removes mode-specific worker startup code.

## Router and dispatchability model

This design assumes the already-desired separation remains:
- controller manages groups
- router manages workers
- `WorkerGroupManager` manages child convergence

Workers may still connect to the router directly for data-plane/routing semantics, but they do not become dispatchable until group-local prerequisites are satisfied.

That keeps authority aligned:
- group manager owns local readiness truth
- router owns dispatchability
- controller owns benchmark orchestration

## Run-mode protocol/adaptor design

To avoid widespread run-mode branching, `WorkerGroupManager` should use injected protocols/adapters for the swappable pieces.

Recommended split:

### Shared core in `WorkerGroupManager`

Core logic should be shared across all run modes:
- child lifecycle state machine
- readiness aggregation
- health aggregation
- dataset gating decisions
- command fanout semantics
- controller-facing reporting
- drain/shutdown coordination

### Swappable run-mode protocols

Use injected interfaces such as:
- `GroupDatasetAuthority`
- `GroupLifecycleTransport`
- `GroupMemberProvider`
- `GroupRuntimeAdapter`

Responsibilities:
- `GroupDatasetAuthority`: acquire dataset state and expose the canonical ready snapshot
- `GroupLifecycleTransport`: group-local request/response and command transport for child services
- `GroupMemberProvider`: enumerate/manage child services for a run mode
- `GroupRuntimeAdapter`: compose run-mode-specific behavior that does not belong in core orchestration

Examples:
- Kubernetes adapter: pod-local router + dataset download + sibling containers
- Local adapter: local IPC + local mmap attach + subprocess children
- SLURM adapter: allocation/node-local lifecycle transport and child discovery

The core lifecycle code should branch once when selecting the adapter, not repeatedly during runtime behavior.

## Testing strategy

Testing should mirror the architecture boundary.

### Shared contract tests

Add shared tests for `WorkerGroupManager` core behavior:
- readiness aggregation
- health aggregation
- dataset gating
- command fanout
- dispatchability transitions
- drain/shutdown completion

### Adapter contract tests

Each run-mode adapter must pass the same protocol-oriented tests:
- Kubernetes adapter
- local adapter
- future SLURM adapter

### Mode-specific integration tests

Add focused integration coverage for:
- Kubernetes group startup
- local group startup
- child churn/restart behavior
- dataset-ready behavior across modes

This means future SLURM support should mostly require:
- implementing the adapter(s)
- passing shared contract tests
- adding a limited set of mode-specific integrations

not forking the architecture.

## Naming and cleanup requirements

This is a clean replacement.

Requirements:
- rename `WorkerPodManager` concepts to `WorkerGroupManager`
- rename “pod-local” concepts to “group-local” where the concept is no longer Kubernetes-specific
- delete obsolete controller-visible child-registration assumptions
- delete worker startup dependence on controller PUB/SUB probes
- avoid compatibility shims and alias layers unless strictly required by transient implementation mechanics

## Expected simplifications

This replacement should simplify:
- controller topology expectations
- worker startup logic
- dataset handoff behavior
- K8s vs local divergence
- future SLURM design work
- routing/readiness mental model

The resulting system boundary becomes:
- controller ↔ group managers
- group managers ↔ child workers/record processors
- router ↔ workers

That is cleaner than the current partial split.

## Open decisions resolved during brainstorming

- Use a full clean replacement, not an incremental compatibility path
- Rename and evolve `WorkerPodManager` into `WorkerGroupManager`
- Apply the new architecture to Kubernetes, local, and future SLURM modes
- Keep workers/record processors as group-local children only
- Make dataset readiness a universal group-local concept across all run modes
- Use protocols/adapters for run-mode differences instead of pervasive conditional branching
