<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Task 5 erased-executor rejection

## Decision

The proposed per-request erasure of `ExecutionSinkBuilder::Sink` and
`ThreadPerCoreExecutor<B>` is rejected. No production executor, HTTP, gRPC, or
WebSocket code was changed, and the ABI baseline was not regenerated.

The measured trait-object paths were slower than their concrete controls with
disjoint Criterion intervals. That violates the design's absolute no-new-
dispatch and zero-loss requirements. Repeating the benchmark until a favorable
sample appears would invalidate the gate.

## Plan correction

The implementation plan named `chat_dispatch_bench` as a Criterion target, but
the integrated source explicitly defines it as a default libtest harness. It
rejects `--save-baseline`, and its measured paths never traverse
`ExecutionSinkBuilder`, `WorkerSink`, or `RequestExecutor`. The failed command
was retained rather than relabeled as valid evidence.

The corrected task-local `executor_dispatch_bench` target measures the exact
async-trait boxed-future construction/drop call surfaces through concrete and
erased paths in the same optimized binary:

- `WorkerSink::dispatch_measured`: the worker-loop call that the proposed
  `Box<dyn WorkerSinkExec>` would make dynamic;
- `RequestExecutor::execute_measured`: a control at the adjacent placement
  boundary.

Input construction occurs outside the timed section. Both variants retain the
existing async-trait future allocation and perform no added request wrapper,
allocation, lock, thread/channel hop, serialization, or callback. The measured
structural difference is static versus vtable dispatch.

## Authoritative paper-rig controls

| Field | Value |
|---|---|
| Source base | `f84b9dbe9da6ada33b9ee1e997cacf0ec35df465` |
| Benchmark commit | `7e3cb6e1f91383b276d95a1f5e8b7e6e647706e7` |
| Cluster | `dynamo-gcp-dev-02` |
| Namespace / pod / container | `acasagrande-paper-rig` / `paper-rig` / `scratch` |
| CPU affinity | `Cpus_allowed_list: 0-143`; cgroup effective cpuset `0-143` |
| Logical CPUs | `144` |
| Load before valid run | `1.14 2.16 2.49` |
| Toolchain | rustc `1.98.0 (88d9e12ae 2026-08-18)`; Cargo `1.98.0 (797e8a9bc 2026-08-05)` |
| Build/cache | `CARGO_BUILD_JOBS=144`, `CARGO_INCREMENTAL=1`, `CARGO_TARGET_DIR=/nvme/cargo-target` |
| Criterion | 200 samples; 5 s warmup; 15 s measurement; saved baseline `pre-erase` |

The existing persistent pod was used in place. It was never deployed, deleted,
or replaced.

## Results

| Call surface | Concrete/static 95% interval | Erased/dyn 95% interval | Point delta | Decision |
|---|---:|---:|---:|---|
| `WorkerSink::dispatch_measured` | 89.129-89.577 ns | 113.56-114.17 ns | +27.45% | reject |
| `RequestExecutor::execute_measured` | 104.08-104.76 ns | 116.92-117.57 ns | +12.27% | reject |

The conservative interval-ratio ranges are +26.77% to +28.10% for the worker-
sink surface and +11.61% to +12.96% for the request-executor control. Both are
strictly worse than the zero-loss requirement.

## Retained evidence

Remote evidence remains under
`/work-pvc/paper-rig/worktrees/task-5-erased-executor` and the saved Criterion
baseline remains under `/nvme/cargo-target/criterion`:

| Evidence | SHA-256 |
|---|---|
| `pre-erase-build-failure.log` (legacy sync omitted embedded assets) | `fde3f5391a698da0433c8bea72c01d87658e418d2b272984b2fe63b3517a59f9` |
| `pre-erase-criterion-cli-failure.log` (libtest rejected Criterion flag) | `aa7655333eb3d7f4d15efd1c21caf3ef25dcfdc9d8edcbbe52853d8331b096ed` |
| `pre-erase-handrolled.log` (diagnostic only; not acceptance evidence) | `1c2de0df27d0834eaec98d81258de5a03fd09b53a9ae04f8a05e7d6477b9bd21` |
| `pre-erase-executor-criterion.log` (authoritative) | `b08fef50f08262ad2ea206410b4f8a6f4d8e0cd3d647103c9ccfae52ee101575` |
| Worker static `estimates.json` | `e0e023a997826b7d5f63ab92f73c4fa68b081153eaa7e81c4f7df9239988f0e2` |
| Worker erased `estimates.json` | `b454654857db0182b937c6b0daaa93fbb5fc2d370d34f8cd22a3fd21ca70c941` |
| Executor static `estimates.json` | `c10c40af0f4efb8a9f36a1f0ac969473b1ab93c7cf04a1f8a1d48f512966ba4c` |
| Executor erased `estimates.json` | `2d4c056235d38a40b037ab2b0897a50c0a8581e3a3425e37bbf7a277a044bff5` |

## Required replacement direction

A replacement design must move the dynamic boundary above the request/token
loop. Frozen startup may dispatch once to the selected transport plugin, after
which that plugin owns the complete monomorphized worker loop. It must preserve
host scheduling, admission, measurement, cancellation, drain, and credit
semantics without introducing a per-request trait-object call or any new
wrapper/allocation/lock/hop/callback. That redesign is intentionally not
implemented in Task 5.
