# Native Rust Runtime Plugins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by task.
> Steps use checkbox (`- [ ]`) syntax for tracking. Every AIPerf feature also
> requires an independent `graham-code-review` approval before integration.

**Goal:** Implement the complete generation-1 native Rust runtime plugin system
for AIPerf, including exact-build compatibility, immutable discovery and
loading, frozen endpoint/transport/exporter composition, first-party migration,
multi-process reproduction, four-platform packaging, and zero-loss performance
gates.

**Architecture:** The host loads exact-build Rust `cdylib` packages through one
immutable catalog and one process-lifetime resident library set before creating
runtime effects. Plugins use published `aiperf-plugin-api`, `aiperf-core`, and
category SDK crates; compatibility records, a single shared mimalloc provider,
strict manifests, content-addressed acquisition, transactional registration,
and a canonical lock make each process reproduce the same frozen universe.
First-party endpoints, transports, and exporters migrate through the same path
as third parties and lose their static fallback only after conformance and
performance parity.

**Tech Stack:** Rust 2024, Cargo resolver 3, BLAKE3, serde/YAML/JSON Schema,
libloading plus platform loader APIs, mimalloc shared provider, Tokio
current-thread runtimes and `LocalSet`, existing AIPerf protocol v2/cellular
seams, GitHub Actions on Linux x86_64/macOS ARM64/Windows x86_64/Windows ARM64.

**Spec:**
`docs/superpowers/specs/2026-08-26-native-rust-runtime-plugins-design.md`

**Execution tracker:**
`docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-execution-tracker.md`

## Global Constraints

- All sixteen invariants in the spec apply together; a task cannot waive one.
- Plugin entry/category/data boundaries use native Rust ABI, traits, and values;
  no `extern "C"` entry, ABI facade, serialization bridge, generated function
  table, Python runtime, IPC, or RPC is permitted there.
- `aiperf_plugin_entry_v1` has exact type
  `unsafe fn() -> PluginDeclarationV1` and is handle-scoped.
- Binary loading is exact-build only. `host_abi_universe_id` and
  `plugin_artifact_build_id` are distinct BLAKE3 identities and neither is
  described as proof of a stable Rust ABI.
- Every production host/plugin boundary artifact uses `panic=abort`; no
  `catch_unwind` crosses or classifies boundary panics.
- `aiperf_alloc_v1` is the pinned shared mimalloc provider itself. Host and
  plugin `GlobalAlloc` shims import `mi_*` directly with eager binding and no
  AIPerf wrapper, function table, selector, metadata, lock, or lazy dispatch.
- No owned `Global` storage crosses the boundary until allocator conformance is
  green on all four targets. Thereafter only ownership-table container families
  may cross.
- AIPerf acquires, hashes, stages, rehashes, and loads the complete
  distribution-controlled non-system executable closure through no-follow
  handles. Hash-path-then-load is forbidden.
- Discovery, static validation, fixed priority, activation, transactional
  registration, freeze, and lock derivation finish before host clocks, Tokio or
  Velo runtimes, workers, clients, artifact directories, dashboards, control
  hooks, datasets, cells, or benchmark effects.
- All successfully returned loader handles become process-resident before
  symbol resolution or pointer escape and remain resident until OS process
  teardown. First loader entry establishes the poison boundary.
- Registration exists only before consuming freeze. There is no reload, unload,
  enable, disable, reprioritization, fallback promotion, or thaw.
- Priority is signed `i32`, resolved per normalized `(category, name)`, unique
  maximum wins, equal maximum is ambiguous, canonical IDs resolve before
  aliases, and filesystem/load order never breaks ties.
- Only endpoint, transport, and exporter are dynamic generation-1 categories.
- Plugins never depend on the orchestration runtime and never receive
  `RunContext`, CLI config, closed transport enums, concrete HTTP internals, or
  complete mutable registries.
- Request/token paths gain no wrapper call, serialization, conversion,
  allocation, lock, thread hop, plugin callback, or IPC. Existing native trait
  dispatch remains the only category dispatch.
- Worker hot paths remain current-thread Tokio plus `LocalSet`, worker-local
  `Rc` state, bounded channels, host clocks, observation, reduction, and
  measurement.
- Exporter capture is host-planned and host-owned. Plugins receive finalized
  projections after report commit; no exporter-specific per-record callback is
  added.
- Config v2 accepts legacy or open forms exactly as specified, rejects mixed
  forms, normalizes before protocol projection, and serializes only open forms.
- Parent, re-exec child, controller, same-host cell, and remote cell reproduce
  the complete `LockedCatalogBundle`, plugin lock, and validated-plan receipts
  before effects.
- Native plugins are trusted code. Discovery authority, ownership/mode/ACL
  policy, privileged execution, distribution inventory authentication,
  revocation, and no-sandbox diagnostics fail closed.
- First-party and third-party packages use the identical manifest, entry,
  registrar, priority, freeze, lock, and loader path. Production IDs never have
  simultaneous hidden static and dynamic registration.
- Every source file receives the two NVIDIA SPDX lines, module docs where
  appropriate, and public-item docs. Production code has no `unwrap()` or
  `expect()` unless an adjacent comment proves infallibility.
- Every behavior change follows RED → verified expected failure → minimal GREEN
  → verified pass → refactor while green. A passing-first test is rewritten.
- Every task uses a dedicated local worktree and paired paper-rig worktree from
  its recorded base. Shared-file or producer/consumer dependencies follow the
  wave schedule below.
- The user explicitly requires parallel implementation. The controller fans out
  every ready, file-disjoint task in the current wave to separate implementer
  agents/worktrees at the same time; it does not serialize ready work merely
  because the generic SDD workflow defaults to one implementer. Review/fix loops
  also remain isolated per task. Integration stays in the dependency order
  recorded below.
- Purely mechanical implementation tasks use `gpt-5.6-terra` with low or
  medium reasoning effort as appropriate. Architecture, unsafe loader/allocator,
  concurrency, hot-path, statistical, and final whole-branch judgment use the
  strongest available model. Every dispatch records model and effort in the
  execution tracker.
- Paper-rig commands set `CARGO_BUILD_JOBS=144`, `CARGO_INCREMENTAL=1`, and
  `CARGO_TARGET_DIR=/cargo-target`; Cargo may use fewer compiler processes only
  when its dependency DAG exposes fewer ready units.
- Paper-rig is the default executor for large builds and test suites to remove
  contention from the local workstation. Record local versus paper-rig wall
  time, queue time, CPU utilization, and storage wait for representative gates.
  If paper-rig is slower for ordinary non-benchmark gates, move those gates
  fully local rather than splitting one target directory across machines.
- Every authoritative static-versus-dynamic A/B benchmark runs on an otherwise-
  idle paper-rig, never on the shared local workstation. Pin CPU topology and
  affinity, isolate the mock server on disjoint cores, freeze frequency/governor
  conditions, record node/kernel/microcode/background-load facts, and invalidate
  attempts that violate the spec’s noise protocol.
- Every feature runs focused tests, its complete task suite, formatting, the
  relevant Clippy command, and a full Graham review with mandatory second pass.
  Zero unresolved Graham findings is required before merge.
- Every distinct feature commit is exported as its own Git bundle, verified by
  `git bundle verify`, fetched into this local repository, and checked for exact
  object-ID equality before integration.
- Static fallback removal additionally requires all ten removal conditions in
  the spec, including four-target package lifecycle, production searches, and
  measured non-inferiority.
- All task commands execute from the nested `rust/` Cargo workspace in the
  repository worktree. File paths in **Files** remain repository-root-relative;
  `--manifest-path` values in commands are workspace-relative and must not start
  with `rust/`. Every paper-rig Rust subprocess suite runs beneath `tini -s`
  (the scratch PID 1 is `sleep` and otherwise adopts killed grandchildren as
  zombies) and requires Python 3. Never unset or replace `RUSTC_WRAPPER`.

## Fixed execution protocol

For Task `N`, the controller records `BASE=$(git rev-parse HEAD)`, creates branch
`ajc/native-plugin-tNN-<slug>` and local worktree
`/home/anthony/nvidia/projects/aiperf/ajc/native-plugin-worktrees/tNN-<slug>`.
It creates the same branch from the same object in paper-rig at
`/work-pvc/paper-rig/aiperf-native-plugin-worktrees/tNN-<slug>`. The implementer
edits only its worktree. The controller mirrors commits by bundle, runs large
gates on paper-rig, dispatches a fresh Graham reviewer against `BASE..HEAD`,
loops findings back to the same implementer, reruns gates, creates one verified
bundle per commit under `/work-pvc/paper-rig/plugin-bundles/tNN/`, fetches each
bundle locally, verifies object IDs, and then integrates in dependency order.

For every named implementation unit `U` in the Implementation-unit Gate Matrix,
the branch is exactly `ajc/native-plugin-unit-U`, the local worktree is exactly
`/home/anthony/nvidia/projects/aiperf/ajc/native-plugin-worktrees/unit-U`, and
the paper-rig worktree is exactly
`/work-pvc/paper-rig/aiperf-native-plugin-worktrees/unit-U`, with literal `U`
replaced by the table’s lowercase hyphenated unit name. Each row is written to
the tracker before either worktree is created. A path or branch already present
for another tracker row is a hard error; reuse is forbidden.

Each task report records:

```text
base_commit=
implementer_model_effort=
red_command=
red_failure=
green_commit=
focused_command=
focused_output_digest=
refactor_commit_or_green_commit_if_empty=
refactor_diff_or_empty_decision=
refactor_focused_command=
refactor_focused_output_digest=
full_gate_command=
full_gate_output_digest=
fmt_command_output_digest=
clippy_command_output_digest=
reviewer_model_effort=
graham_first_review_range=
graham_first_verdict_and_report_digest=
graham_fix_commits=
graham_second_review_range=
graham_second_verdict_and_report_digest=
post_review_gate_command=
post_review_gate_output_digest=
bundle_paths=
local_object_ids=
commit_bundle_import_object_map=
integration_commit=
```

Every Task 1–40 expands its written Step 4 and Step 5 with these mandatory
executable substeps; task prose cannot waive them:

1. After minimal GREEN, record `green_commit` and its focused-output digest.
2. Inspect `BASE..green_commit` for duplicated logic, misleading names,
   unnecessary ownership/allocation/clones, comment noise, and avoidable scope.
   Apply the refactor in a separate commit when any change is warranted. If the
   minimal implementation is already final-form, record the exact empty diff and
   a reason tied to that checklist, set the refactor object to `green_commit`,
   and do not fabricate a commit. Rerun the exact focused command in either case.
3. Run `./scripts/run-plugin-task-gates.sh N` for task N on paper-rig beneath the
   fixed `tini -s -- sh -c` protocol. The script executes the task’s complete
   suite, `cargo fmt --check`, and the exact Clippy package/feature scope in the
   Task Gate Matrix; all three output digests are separately recorded.
4. Dispatch a fresh Graham reviewer for `BASE..HEAD`, record every finding, send
   fixes through RED/GREEN when behavior changes, and rerun the task gate script.
5. Dispatch a second fresh Graham reviewer for the exact final `BASE..HEAD` even
   when the first pass had zero findings. `PASS` requires both report digests and
   zero unresolved blocker/important/minor/style findings after the second pass.
6. Rerun the task gate script against the second-pass object, then create and
   verify one bundle per distinct commit. Record an exact
   `commit -> bundle -> fetched local object` map before integration.

Task `37a-tooling` integrates the candidate-mode assembler/tests/CI before 39a;
Task `38a-harness` integrates the parity harness before 39a. The private 39a
chain is based on that integration tip. `37b-package` and `38b-benchmark` run the
already-integrated tools against the private chain without creating or
integrating source commits. Tasks 39a-* are never integrated. Task 39b alone
fast-forwards the integration branch to the gated 39a tip and publishes the
retained Task-37 bytes.

The controller invokes the recorded command as `cd rust && <command>` (and, on
paper-rig, through `tini -s -- sh -c 'cd rust && <command>'`). Commands that
need files above the workspace use paths explicitly relative to the workspace.

## Task Gate Matrix

`rust/scripts/run-plugin-task-gates.sh` executes the exact complete-suite and
Clippy commands below for its task ID, followed by `cargo fmt --check`; `&&`
short-circuits on the first failure. Focused RED/GREEN/refactor commands remain
the exact commands in each task. A task report records separate output digests
for focused, complete-suite, Clippy, and formatting commands.

| Task | Complete task suite command | Exact Clippy command |
|---|---|---|
| 1 | `cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine && cargo test && cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory --test plugin_task_gate_inventory` | `cargo clippy --all-targets` |
| 2 | `cargo test -p aiperf-plugin-api && cargo check --workspace` | `cargo clippy -p aiperf-plugin-api --all-targets` |
| 3 | `cargo test -p aiperf-bench-tools --all-targets` | `cargo clippy -p aiperf-bench-tools --all-targets` |
| 4 | `cargo test -p aiperf-core --all-targets && cargo test -p aiperf-runtime --lib` | `cargo clippy -p aiperf-core -p aiperf-runtime --all-targets` |
| 5 | `cargo test -p aiperf-plugin-api --all-targets` | `cargo clippy -p aiperf-plugin-api --all-targets` |
| 6 | `cargo test -p aiperf-core -p aiperf-plugin-api -p aiperf-endpoint-sdk -p aiperf-transport-sdk -p aiperf-export-sdk -p aiperf-plugin-test-support --all-targets && cargo test -p aiperf-runtime --features engine` | `cargo clippy -p aiperf-core -p aiperf-plugin-api -p aiperf-endpoint-sdk -p aiperf-transport-sdk -p aiperf-export-sdk -p aiperf-plugin-test-support -p aiperf-runtime --all-targets --features aiperf-runtime/engine` |
| 7 | `cargo test -p aiperf-allocator-provider -p aiperf-allocator-shim -p aiperf-plugin-conformance --test allocator && cargo test -p aiperf-cli && cargo build -p aiperf-cli --release` | `cargo clippy -p aiperf-allocator-provider -p aiperf-allocator-shim -p aiperf-plugin-conformance -p aiperf-cli --all-targets` |
| 8 | `cargo test -p aiperf-plugin-sdk --all-targets` | `cargo clippy -p aiperf-plugin-sdk --all-targets` |
| 9 | `cargo test -p aiperf-plugin-sdk -p aiperf-plugin-sdk-macros --all-targets` | `cargo clippy -p aiperf-plugin-sdk -p aiperf-plugin-sdk-macros --all-targets` |
| 10 | `cargo test -p aiperf-plugin-host --test manifest` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 11 | `cargo test -p aiperf-plugin-host --test acquisition --test acquisition_races` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 12 | `cargo test -p aiperf-plugin-host --test static_inspection` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 13 | `cargo test -p aiperf-plugin-host --test discovery --test priority --test authority` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 14 | `cargo test -p aiperf-plugin-host --test loader --test residency --test poison` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 15 | `cargo test -p aiperf-plugin-host --test registration && cargo test -p aiperf-runtime --features engine` | `cargo clippy -p aiperf-plugin-host -p aiperf-runtime --all-targets --features aiperf-runtime/engine` |
| 16 | `cargo test -p aiperf-plugin-host --test lock --test bundle --test lock_mismatch --test lock_input` | `cargo clippy -p aiperf-plugin-host --all-targets` |
| 17 | `cargo test -p aiperf-cli --test plugin_effect_order --test plugin_commands --test plugin_lock_input --test plugin_abort_contract --test plugin_route_census && cargo test -p aiperf-runtime --features engine` | `cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-runtime/engine` |
| 18 | `cargo test -p aiperf-cli --test plugin_config_open_selection && cargo test -p aiperf-runtime --test plugin_protocol_projection --features engine` | `cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-runtime/engine` |
| 19 | `cargo test -p aiperf-runtime --test plugin_capture_plan --test factory_validation_receipt --features engine` | `cargo clippy -p aiperf-runtime --all-targets --features engine` |
| 20 | `cargo test -p aiperf-cli --test plugin_reexec --test plugin_reexec_plan` | `cargo clippy -p aiperf-cli --all-targets` |
| 21 | `cargo test -p aiperf-cli --test plugin_cellular --test plugin_kube_slurm_projection --features cellular && cargo test -p aiperf-runtime --features engine,cellular` | `cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-cli/cellular,aiperf-runtime/engine,aiperf-runtime/cellular` |
| 22 | `cargo test -p aiperf-e2e-tests --test plugin_report_provenance && cargo test -p aiperf-runtime --features engine,cellular` | `cargo clippy -p aiperf-runtime -p aiperf-e2e-tests --all-targets --features aiperf-runtime/engine,aiperf-runtime/cellular` |
| 23 | `cargo test -p aiperf-e2e-tests --test plugin_cellular_capture --test plugin_exporter_outcomes && cargo test -p aiperf-runtime --features engine,cellular` | `cargo clippy -p aiperf-runtime -p aiperf-e2e-tests --all-targets --features aiperf-runtime/engine,aiperf-runtime/cellular` |
| 24 | `cargo test -p aiperf-plugin-export-basic --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_basic` | `cargo clippy -p aiperf-plugin-export-basic -p aiperf-e2e-tests --all-targets` |
| 25 | `cargo test -p aiperf-plugin-export-parquet --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_parquet` | `cargo clippy -p aiperf-plugin-export-parquet -p aiperf-e2e-tests --all-targets` |
| 26 | `cargo test -p aiperf-plugin-export-mlflow --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_mlflow` | `cargo clippy -p aiperf-plugin-export-mlflow -p aiperf-e2e-tests --all-targets` |
| 27 | `cargo test -p aiperf-plugin-export-wandb --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_wandb` | `cargo clippy -p aiperf-plugin-export-wandb -p aiperf-e2e-tests --all-targets` |
| 28 | `cargo test -p aiperf-plugin-export-otel --all-targets && cargo test -p aiperf-e2e-tests --test plugin_telemetry_capture` | `cargo clippy -p aiperf-plugin-export-otel -p aiperf-e2e-tests --all-targets` |
| 29 | `cargo test -p aiperf-plugin-endpoints --all-targets && cargo test -p aiperf-e2e-tests --test plugin_endpoints` | `cargo clippy -p aiperf-plugin-endpoints -p aiperf-e2e-tests --all-targets` |
| 30 | `cargo test -p aiperf-plugin-endpoints --test grpc_binding && cargo test -p aiperf-e2e-tests --test plugin_endpoint_grpc_override --features grpc` | `cargo clippy -p aiperf-plugin-endpoints -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/grpc` |
| 31 | `cargo test -p aiperf-plugin-transport-http --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_http` | `cargo clippy -p aiperf-plugin-transport-http -p aiperf-e2e-tests --all-targets` |
| 32 | `cargo test -p aiperf-plugin-transport-grpc --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_grpc --features grpc` | `cargo clippy -p aiperf-plugin-transport-grpc -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/grpc` |
| 33 | `cargo test -p aiperf-plugin-transport-websocket -p aiperf-plugin-transport-dry-run --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_websocket --features websocket && cargo test -p aiperf-dry-run-tests` | `cargo clippy -p aiperf-plugin-transport-websocket -p aiperf-plugin-transport-dry-run -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/websocket` |
| 34 | `cargo test -p aiperf-plugin-transport-dynosim --all-targets --features dynosim && cargo test -p aiperf-e2e-tests --test plugin_transport_dynosim --features dynosim` | `cargo clippy -p aiperf-plugin-transport-dynosim -p aiperf-e2e-tests --all-targets --features aiperf-plugin-transport-dynosim/dynosim,aiperf-e2e-tests/dynosim` |
| 35 | `cargo test -p aiperf-plugin-host --test discovery_authority --test atomic_generations && cargo test -p aiperf-plugin-packaging-tests --test distribution_lifecycle` | `cargo clippy -p aiperf-plugin-host -p aiperf-plugin-packaging-tests --all-targets` |
| 36 | `cargo test -p aiperf-plugin-conformance --all-targets` | `cargo clippy -p aiperf-plugin-conformance --all-targets` |
| 37 | `cargo test -p aiperf-plugin-packaging-tests --test distribution_census && cargo test -p aiperf-plugin-static-comparator --all-targets && make -C .. native-cli-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml && make -C .. bundle-cli-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml CLI_FEATURES='--features full' && make -C .. wheel-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml && make -C .. native-cli-static-comparator AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml AIPERF_STATIC_COMPARATOR_OUTPUT=/cargo-target/plugin-static-fixture` | `cargo clippy -p aiperf-plugin-packaging-tests -p aiperf-plugin-static-comparator --all-targets` |
| 38 | `cargo test -p aiperf-plugin-perf --all-targets` | `cargo clippy -p aiperf-plugin-perf --all-targets` |
| 39 | `cargo test -p aiperf-plugin-conformance --test no_static_paths && cargo test && cargo test -p aiperf-runtime --features engine && cargo build -p aiperf-cli --features full` | `cargo clippy --all-targets` |
| 40 | `cargo test -p aiperf-plugin-sdk --test docs_examples && cargo test -p aiperf-cli --test plugin_commands && cargo test -p aiperf-plugin-conformance --test final_package_and_removal && cargo test && cargo test -p aiperf-runtime --features engine && cargo build -p aiperf-cli --features full` | `cargo clippy --all-targets` |

Task 12 has independently tracked implementation units `12-core`, `12-elf`,
`12-macho`, and `12-pe`; after `12-core` freezes the inspector trait, the three
backend worktrees fan out and each executes Task-12 gates plus its native
platform job. Task 33 has `33-websocket` and `33-dry-run` rows/worktrees and
separate Graham/bundle evidence. Task 34 has serial `34-dynosim-offline` and
`34-dynosim-online` rows/worktrees because they share one package manifest.
Task 37 has `37a-tooling` (source commit/integration) and `37b-package`
(evidence-only private execution). Task 38 has `38a-harness` (source
commit/integration) and `38b-benchmark` (evidence-only private execution).

Task 39a has separate serial worktree/feature/review/bundle rows named
`39a-basic`, `39a-parquet`, `39a-mlflow`, `39a-wandb`,
`39a-otel`, `39a-endpoints-grpc-bindings`, `39a-http`, `39a-grpc`,
`39a-websocket`, `39a-dry-run`, `39a-dynosim-offline`, and
`39a-dynosim-online`. Each starts from the preceding reviewed object, runs its
affected task/component gates and the component-scoped no-static-path manifest,
receives two Graham passes, and is bundled/imported for object verification but
not integrated. `39b` receives its own ledger row and Task-39 full gates.

### Implementation-unit Gate Matrix

The gate script accepts these exact additional IDs. It always appends
`cargo fmt --check`; each called numbered task gate already runs its exact
complete suite and Clippy command.

| Unit | Exact command before formatting |
|---|---|
| `12-core` | `./scripts/run-plugin-task-gates.sh 12` |
| `12-elf` | `AIPERF_INSPECTOR_BACKEND=elf cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12` |
| `12-macho` | `AIPERF_INSPECTOR_BACKEND=macho cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12` |
| `12-pe` | `AIPERF_INSPECTOR_BACKEND=pe cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12` |
| `33-websocket` | `./scripts/run-plugin-task-gates.sh 33` |
| `33-dry-run` | `./scripts/run-plugin-task-gates.sh 33` |
| `34-dynosim-offline` | `./scripts/run-plugin-task-gates.sh 34` |
| `34-dynosim-online` | `./scripts/run-plugin-task-gates.sh 34` |
| `37a-tooling` | `./scripts/run-plugin-task-gates.sh 37` |
| `37b-package` | `PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_WORKTREE=/work-pvc/paper-rig/aiperf-native-plugin-worktrees/unit-39a-dynosim-online AIPERF_CANDIDATE_OUTPUT=/cargo-target/plugin-release-candidate AIPERF_STATIC_COMPARATOR_OUTPUT=/cargo-target/plugin-static-baseline make -C .. native-cli-candidate bundle-cli-candidate wheel-candidate native-cli-static-comparator` |
| `38a-harness` | `./scripts/run-plugin-task-gates.sh 38` |
| `38b-benchmark` | `cargo run -p aiperf-plugin-perf --release --bin parity -- --inventory benchmarks/plugin-parity.yaml --candidate-root /cargo-target/plugin-release-candidate --baseline-root /cargo-target/plugin-static-baseline --pairs 30 --warmups 5 --bootstrap-resamples 100000 --output ../artifacts/native-plugin-parity` |
| `39a-basic` | `AIPERF_STATIC_PATH_COMPONENT=basic cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 24` |
| `39a-parquet` | `AIPERF_STATIC_PATH_COMPONENT=parquet cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 25` |
| `39a-mlflow` | `AIPERF_STATIC_PATH_COMPONENT=mlflow cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 26` |
| `39a-wandb` | `AIPERF_STATIC_PATH_COMPONENT=wandb cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 27` |
| `39a-otel` | `AIPERF_STATIC_PATH_COMPONENT=otel cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 28` |
| `39a-endpoints-grpc-bindings` | `AIPERF_STATIC_PATH_COMPONENT=endpoints-grpc-bindings cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 29 && ./scripts/run-plugin-task-gates.sh 30` |
| `39a-http` | `AIPERF_STATIC_PATH_COMPONENT=http cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 31` |
| `39a-grpc` | `AIPERF_STATIC_PATH_COMPONENT=grpc cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 32` |
| `39a-websocket` | `AIPERF_STATIC_PATH_COMPONENT=websocket cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 33` |
| `39a-dry-run` | `AIPERF_STATIC_PATH_COMPONENT=dry-run cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 33` |
| `39a-dynosim-offline` | `AIPERF_STATIC_PATH_COMPONENT=dynosim-offline cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 34` |
| `39a-dynosim-online` | `AIPERF_STATIC_PATH_COMPONENT=dynosim-online cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 34` |
| `39b` | `./scripts/run-plugin-task-gates.sh 39` |

## Dependency and parallelism waves

Tasks inside a wave run in parallel only when their file sets are disjoint. A
consumer starts after every listed producer is integrated. Cross-wave
cherry-picks are prohibited; each worktree starts from the current integration
HEAD.

| Wave | Tasks | Shared-interface ruling |
|---|---|---|
| A | 1 | Baseline evidence only; production implementation does not start until it passes |
| B | 2 → 3 | Serial: Task 2 owns dependency-neutral provisional workspace/member shells and initial lock; Task 3 consumes Task 1 evidence and is the single owner of the measured, reviewed package-topology matrix and sanctioned follow-up manifest amendment |
| C1 | 4 | Core value extraction is the single owner of moved boundary types |
| C2 | 5 | Starts after Task 4; freezes public API/ownership table without a concurrent interface editor |
| C3 | 6 | Starts after Task 5; freezes category contracts before downstream fan-out |
| D1 | 7 | Gate and integrate the authoritative provider before any universe record or native activation can claim allocator identity |
| D2 | 8 | Starts from the exact Task-7 object and binds its provider/import/topology facts into canonical universe/build records |
| D3 | 9 | Starts after Task 8; consumes the authoritative allocator and exact identity contracts for SDK builds |
| E | 10 → 11 → 12 | Decode DTOs, then acquisition, then static inspection; Task 12 fans out ELF/Mach-O/PE backend subworktrees behind one frozen inspector trait |
| F | 13 → 14 → 15 | Intended catalog, loader state, then manifest-bound freeze share central state types and integrate serially |
| G | 16 → (17, 18) | After lock format integrates, fan out CLI composition and pure Config-v2 compatibility decoding; neither task creates a run plan or removes static production paths |
| H1 | 19 | Consume Task 6 receipt/capture API and implement the runtime validated-plan/capture owner |
| H2 | 20 → 21 → 23 → 22 | Serial: re-exec owns `execute_mode.rs`, cellular bootstrap owns controller protocol, capture/outcomes follows cellular, and report provenance follows all three producers |
| J | 24, 25, 26, 27 | After Task 36 feasibility PASS, fan out package-only comparator implementations and manifest fragments; no task edits central production registration/inventory or removes a static ID. Integrate basic → Parquet → MLflow → W&B |
| K | 28 | OTel starts after Task 23 and exporter integration; it is the final exporter migration |
| L1 | 29 | Freeze endpoint package/binding interface |
| L2 | 30, 31 | After Task 29, fan out endpoint-owned gRPC binding comparator and HTTP transport comparator; production cutover remains Task 39 |
| M | 32 | gRPC transport starts after Task 30; HTTP remains the first integrated transport |
| N | 33 → 34 | Serial package work: the shared central registry/inventory is untouched until Task 39; integrate WebSocket, dry-run, then Dynosim comparator artifacts in specified order |
| O | 35 → 36 | Trust/install tooling first; then complete four-target real-artifact feasibility, allocator, conformance, and representative Task-3 performance gate. Task 36 PASS is a predecessor of Task 24 |
| P1 | 37a-tooling → 38a-harness → 39a → 37b-package → 38b-benchmark → 39b | Packaging/performance tools integrate before the private chain; 37b/38b create evidence only; 39b can therefore fast-forward the unchanged gated 39a ancestry |
| Q | 40 | Final static-path/package/conformance/docs audit after publication |
| R | 40 | Documentation, final whole-branch Graham review, and requirement audit |

---

### Task 1: Freeze and prove the pre-plugin baseline

**Files:**
- Create: `rust/benchmarks/plugin-parity.yaml`
- Create: `artifacts/native-plugin-baseline/README.md`
- Create: `rust/e2e-tests/tests/plugin_baseline_inventory.rs`
- Create: `rust/e2e-tests/tests/plugin_task_gate_inventory.rs`
- Create: `rust/scripts/run-plugin-task-gates.sh`
- Modify: `rust/runtime/src/metrics_core/report.rs`
- Modify: `docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-execution-tracker.md`
- Create: `artifacts/native-plugin-baseline/package-topology.json`

**Interfaces:**
- Consumes: commit `d4159dc91a` plus the tracker-recorded current integration
  changes.
- Produces: immutable benchmark/build inventory `plugin-parity.yaml` with host
  commit, toolchain, target, features, profiles, commands, artifact digests,
  clean/incremental build timings, binary size, runtime CPU, latency, throughput,
allocations, and raw-sample locations used by Tasks 37–39.
  Produces measured package coupling evidence in `package-topology.json`; Task 2
  consumes it before final member manifests/feature ownership are authored.

- [ ] **Step 1: Write the baseline schema test**

Create `rust/e2e-tests/tests/plugin_baseline_inventory.rs` with a test that
loads the YAML and requires these exact top-level keys:

```rust
const REQUIRED: &[&str] = &[
    "schema_version", "host_commit", "rustc", "target", "cargo_profile",
    "feature_sets", "build_commands", "runtime_scenarios", "artifacts",
    "allocation_probe", "raw_samples",
];
```

The schema fixture also requires, for every scenario, request budget, minimum
duration, core assignment, mock placement and artifact digest, response shape,
warmups, estimator, bootstrap seed, one legal primary metric exactly equal to
`successful_requests_per_second`, `output_tokens_per_second`,
`cpu_nanoseconds_per_successful_request`, or
`exporter_nanoseconds_per_record`, its exact ratio direction, measured metric
set, invalidation classifier, harness/mock digest, firmware, memory topology,
and canonical inventory digest. TTFT and ITL p50/p90/p99 are mandatory measured
secondary metrics and are rejected as `primary_metric`. It rejects an omitted
identity field.
For every exporter scenario it additionally requires exactly these fields and
values: `corpus_records: 100000`, `sample_repetitions: 16`,
`processed_records: 1600000`, and `retained_artifact_records: 100000`.
It requires 16 sequential per-repetition receipts, each proving exactly 100,000
emitted records and the same output digest, one retained 100,000-record output
artifact, active exporter duration equal to the receipt-duration sum, no
sleep/padding, and active duration at least 30 seconds. Exporter
nanoseconds-per-record must divide by `processed_records`, not the retained
artifact count or wall-clock duration.
`plugin_task_gate_inventory.rs` parses the script, Task Gate Matrix, and
Implementation-unit Gate Matrix; it requires exactly one case for every integer
1 through 40 plus every named implementation unit, and rejects a case missing a
complete suite, `cargo fmt --check`, or exact Clippy package/feature command.
Before any change, run the existing v2 report golden unmodified and retain its
version-drift failure as RED. Step 3 then applies
`report.aiperf_version = "0.0.0".to_owned()` only in the test fixture as GREEN,
leaving production `env!("CARGO_PKG_VERSION")` and checked-in JSON unchanged.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory --test plugin_task_gate_inventory`

Expected: FAIL because
  `rust/benchmarks/plugin-parity.yaml` does not exist.
The focused report test initially fails only on the known version drift; it must
not change production version behavior to make the golden pass.

- [ ] **Step 3: Capture the immutable baseline**

On paper-rig run clean and second-build invocations for the default, `engine`,
`grpc`, `parquet`, `dynosim`, and `full` matrices without changing compiler
cache/profile/features. Record canonical commands and BLAKE3 artifact digests.
Run the spec’s paired runtime scenarios against the in-repo mock server and
store raw samples under `artifacts/native-plugin-baseline/raw/`. Author
`rust/benchmarks/plugin-parity.yaml` using schema version `1` and relative
raw-sample paths. For the exporter sample, retain one deterministic 100,000-
record artifact and 16 sequential receipts for identical 100,000-record passes;
record `processed_records: 1600000` and only the summed active pass duration.
Implement `run-plugin-task-gates.sh` as an exhaustive integer `case` over the
Task Gate Matrix below. It rejects missing/unknown task IDs, runs with
`set -eu`, preserves all compiler/cache variables, and propagates the first
nonzero command status.

- [ ] **Step 4: Verify GREEN and baseline health**

Run on paper-rig:

```bash
cargo test -p aiperf-runtime
cargo test -p aiperf-runtime --features engine
cargo test
cargo clippy --all-targets
cargo fmt --check
cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory
cargo test -p aiperf-e2e-tests --test plugin_task_gate_inventory
```

Expected: every command exits 0; tracker records exact commit and output log
digests. Run every Rust subprocess suite beneath `tini -s` with Python 3
available; retain wrapper/cache variables unchanged. The report golden passes
only through the test fixture’s explicit version assignment above.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review `BASE..HEAD`, resolve all findings, rerun Step 4, commit as
`test(plugins): freeze static parity baseline`, bundle the commit, verify/import
it locally, and mark Task 1 `PASS` in the tracker.

### Task 2: Add the plugin-facing workspace crate shells and policy guard

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Modify: `rust/e2e-tests/Cargo.toml`
- Create: `rust/plugin-api/Cargo.toml`
- Create: `rust/plugin-api/src/lib.rs`
- Create: `rust/core/Cargo.toml`
- Create: `rust/core/src/lib.rs`
- Create: `rust/plugin-sdk/Cargo.toml`
- Create: `rust/plugin-sdk/src/lib.rs`
- Create: `rust/endpoint-sdk/Cargo.toml`
- Create: `rust/endpoint-sdk/src/lib.rs`
- Create: `rust/transport-sdk/Cargo.toml`
- Create: `rust/transport-sdk/src/lib.rs`
- Create: `rust/export-sdk/Cargo.toml`
- Create: `rust/export-sdk/src/lib.rs`
- Create: `rust/plugin-sdk-macros/Cargo.toml`
- Create: `rust/plugin-sdk-macros/src/lib.rs`
- Create: `rust/plugin-host/Cargo.toml`
- Create: `rust/plugin-host/src/lib.rs`
- Create: `rust/plugin-conformance/Cargo.toml`
- Create: `rust/plugin-conformance/src/lib.rs`
- Create: `rust/plugin-test-support/Cargo.toml`
- Create: `rust/plugin-test-support/src/lib.rs`
- Create: `rust/plugin-packaging-tests/Cargo.toml`
- Create: `rust/plugin-packaging-tests/src/lib.rs`
- Create: `rust/plugin-perf/Cargo.toml`
- Create: `rust/plugin-perf/src/lib.rs`
- Create: `rust/plugin-static-comparator/Cargo.toml`
- Create: `rust/plugin-static-comparator/src/lib.rs`
- Create: `rust/allocator-provider/Cargo.toml`
- Create: `rust/allocator-provider/src/lib.rs`
- Create: `rust/allocator-provider/build.rs`
- Create: `rust/allocator-shim/Cargo.toml`
- Create: `rust/allocator-shim/src/lib.rs`
- Create: final empty package shells and manifests under
  `rust/plugins/{export-basic,export-parquet,export-mlflow,export-wandb,export-otel,endpoints,transport-http,transport-grpc,transport-websocket,transport-dry-run,transport-dynosim}/`
- The exact Cargo package names for those shells, in the same order, are
  `aiperf-plugin-export-basic`, `aiperf-plugin-export-parquet`,
  `aiperf-plugin-export-mlflow`, `aiperf-plugin-export-wandb`,
  `aiperf-plugin-export-otel`, `aiperf-plugin-endpoints`,
  `aiperf-plugin-transport-http`, `aiperf-plugin-transport-grpc`,
  `aiperf-plugin-transport-websocket`, `aiperf-plugin-transport-dry-run`, and
  `aiperf-plugin-transport-dynosim`.
- The remaining exact package names are `aiperf-core`, `aiperf-plugin-api`,
  `aiperf-plugin-sdk`, `aiperf-endpoint-sdk`, `aiperf-transport-sdk`,
  `aiperf-export-sdk`, `aiperf-plugin-sdk-macros`, `aiperf-plugin-host`,
  `aiperf-plugin-conformance`, `aiperf-plugin-test-support`,
  `aiperf-plugin-packaging-tests`,
  `aiperf-plugin-perf`, `aiperf-plugin-static-comparator`,
  `aiperf-allocator-provider`, and
  `aiperf-allocator-shim`.
- Create: provisional `Cargo.toml`, `src/lib.rs`, and `plugins.yaml.in` in each
  package shell above so later `Modify` paths exist from their claimed base.
- Create: `rust/plugin-api/api-allowlist.toml`
- Create: `rust/plugin-api/feature-ownership.toml`
- Create: `rust/plugin-conformance/candidate-source-inventory.toml`
- Create: `rust/plugin-api/tests/dependency_policy.rs`
- Create: `rust/tests/plugin-third-party/Cargo.toml`
- Create: `rust/tests/plugin-third-party/src/lib.rs`

**Interfaces:**
- Produces every final workspace member and plugin package shell in one
  foundation commit so parallel feature worktrees never edit `rust/Cargo.toml`
  or add package identities to `rust/Cargo.lock` concurrently. Task 2 adds
  `tempfile = "3"` to `[workspace.dependencies]`. Manifests declare a
  dependency-neutral provisional matrix except for the exact foundational
  downward/test-only edges named in this task; later feature tasks
  modify only their own crate’s sources and tests unless the tracker records an
  explicit single-owner manifest amendment. `allocator-provider` and
  `allocator-shim` are initial members; `tests/plugin-third-party` is an explicit
  standalone `[workspace]` exemplar excluded from the parent workspace. Task 7
  has the sole allocator-topology amendment to CLI dependencies and the lock;
  Task 37 is the sole later distribution-membership/lock amendment. Other later
  tasks may alter only their precreated package manifests.
- Task 2 consumes Task 1’s measured `package-topology.json` only to prove every
  required shell exists without prematurely assigning implementation
  dependencies. Task 3 owns the measured, reviewed topology matrix and one
  explicitly scoped workspace/member-manifest and lock amendment. Task 7 then
  owns only the CLI allocator dependency/lock delta; Task 37 alone may later
  amend final distribution membership/lock.
- Adds exact e2e pass-through features `grpc = ["aiperf-runtime/grpc",
  "aiperf-cli/grpc"]`, `websocket = ["aiperf-runtime/websocket",
  "aiperf-cli/websocket"]`, and `dynosim = ["aiperf-runtime/dynosim",
  "aiperf-cli/dynosim"]` so later `aiperf-e2e-tests --features ...` commands are
  real Cargo features; later feature tasks do not race on this manifest.
- `aiperf-plugin-api` may depend only on `aiperf-core`, `blake3`, `serde`,
  `serde_json`, `thiserror`, and standard library entries explicitly present in
  `api-allowlist.toml`; it may not depend on Tokio, Hyper, Tonic, Clap, exporter
  backends, or `aiperf-runtime`.

- [ ] **Step 1: Write the failing dependency-policy test**

The test runs `cargo metadata --format-version 1 --no-deps`, requires every
final workspace/package shell named in this task, verifies the checked
host/core/plugin/distribution feature ownership matrix, and asserts the API
crate dependency-name set is a subset of the allowlist and excludes:

```rust
const FORBIDDEN: &[&str] = &[
    "aiperf-runtime", "tokio", "hyper", "tonic", "clap", "parquet",
    "opentelemetry", "reqwest", "wandb",
];
```

The test separately inspects normal/build versus test dependencies, rejects both
`aiperf-plugin-host -> aiperf-runtime` and `aiperf-runtime ->
aiperf-plugin-host`, permits host only on API/core/SDK, and requires runtime to
consume the plugin-API-owned `FrozenPluginUniverse`/`FrozenAIPerfRegistry` view
(API depends on core). CLI is the sole
composition layer that depends on both host and runtime.
The test also requires `aiperf-plugin-test-support` as a non-published,
distribution-excluded workspace member whose normal dependencies are limited to
`aiperf-core`, `tempfile`, and the standard library. Only dev-dependency edges
from SDK/candidate/e2e test targets may point to it; no production normal/build
dependency, plugin API allowlist entry, native/wheel/container/Kubernetes
inventory, or host-universe boundary may include it. Task 2 predeclares the
`aiperf-export-sdk` independent-leaf test’s dev-dependency edge so Task 6 does
not amend workspace membership or shared manifests.
The test also parses `candidate-source-inventory.toml`, requires every exact
source, test, golden, and proto path in the plan’s Candidate Source Inventory
appendix, rejects duplicate/unknown owners, and requires a candidate destination
plus `implementation_leaf` or explicitly reviewed `facade` classification for
every present entry. Exact post-Task-6 split paths may be absent only as
`planned` rows with `producer_task = 6`; every other missing source fails.
Task 6 must replace every planned row with a present BLAKE3-bound
`implementation_leaf` before downstream candidate staging.

- [ ] **Step 2: Verify RED**

Run: `cargo test --manifest-path plugin-api/Cargo.toml`

Expected: FAIL because the crate manifest does not exist.

- [ ] **Step 3: Add minimal documented crate shells**

Add all final workspace members, edition `2024`, resolver-inherited workspace
dependencies, two-line SPDX headers, module docs, and no public behavior beyond
an exact source API constant:

```rust
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
```

- [ ] **Step 4: Verify GREEN**

Run: `cargo test --manifest-path plugin-api/Cargo.toml && cargo check --workspace`

Expected: policy test passes and `cargo check --workspace` resolves every
declared member while preserving the standalone exemplar boundary.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review only crate manifests/shells/policy test, resolve all findings, run
`cargo fmt --check && cargo clippy -p aiperf-plugin-api --all-targets`, commit
as `build(plugins): add public plugin crate boundaries`, bundle/import by exact
ID, and integrate.

### Task 3: Add reproducible build and performance measurement tooling

**Files:**
- Modify: `rust/bench-tools/Cargo.toml`
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Modify: `rust/plugin-api/feature-ownership.toml`
- Modify: only the precreated plugin/package `Cargo.toml` files whose measured
  dependency islands differ from the Task-2 dependency-neutral shells
- Create: `rust/bench-tools/src/lib.rs`
- Create: `rust/bench-tools/src/bin/plugin_build_bench.rs`
- Create: `rust/bench-tools/src/bin/plugin_runtime_bench.rs`
- Create: `rust/bench-tools/src/plugin_stats.rs`
- Create: `rust/bench-tools/tests/plugin_stats.rs`
- Create: `rust/bench-tools/tests/plugin_topology.rs`
- Create: `rust/scripts/run-plugin-parity.sh`

**Interfaces:**
- Produces canonical JSONL samples with `{scenario, pair_id, variant, metric,
  value, unit, commit, artifact_digest}` and deterministic bootstrap summaries
  consumed by Task 38.
- The runner never changes Cargo profile, features, compiler cache, LTO, or
  baseline commands between static/dynamic variants. It exposes a library target
  so integration tests and Task 38 share one
  implementation rather than duplicating statistics.
- Consumes Task 1 `package-topology.json`, writes the sole measured and reviewed
  final dependency/feature-ownership matrix, and performs the foundational
  root-workspace/lock amendment. Task 7’s explicitly recorded CLI allocator
  dependency/lock delta is the only pre-Task-37 exception. The matrix test
  rejects every dependency or feature edge not justified by measured source
  coupling.

- [ ] **Step 1: Write failing statistical tests**

Test one-sided paired differences, simultaneous bound evaluation over the full
non-allocation metric/case matrix, the separate exact allocation gate, finite-value
rejection, coefficient-of-variation retry limits, and deterministic bootstrap
seeding using a fixed sample vector. The production change that makes each test
pass is the corresponding `plugin_stats` function.
The topology test reads Task-1 `package-topology.json`, requires one reviewed
owner for every measured dependency/feature edge, compares it to
`plugin-api/feature-ownership.toml` and Cargo metadata, and rejects an
unmeasured edge or a dependency-neutral shell left unresolved.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-bench-tools --test plugin_stats --test plugin_topology`

Expected: FAIL with unresolved `plugin_stats` imports.

- [ ] **Step 3: Implement the canonical sample and statistics API**

Expose:

```rust
pub struct PairedSample { /* exact serialized fields listed above */ }
pub struct NonInferiorityGate {
    pub metric: String,
    pub max_relative_regression: f64,
    pub confidence: f64,
}
pub enum MetricGateResult {
    SimultaneousNonInferiority {
        lower_confidence_bound: f64,
        threshold: f64,
    },
    ExactNoAllocationIncrease {
        minimum_paired_ratio: f64,
    },
}
pub fn evaluate_paired_gate(
    samples: &[PairedSample],
    gate: &NonInferiorityGate,
    bootstrap_seed: u64,
) -> Result<GateReport, PluginStatsError>;
pub fn evaluate_simultaneous_gate(
    cases: &[PairedCase],
    policy: &SimultaneousGatePolicy,
    bootstrap_seed: u64,
) -> Result<SimultaneousGateReport, PluginStatsError>;
```

`PairedCase` carries these exact legal primary metric/direction pairs:
`successful_requests_per_second` and `output_tokens_per_second` use
`dynamic/static`; `cpu_nanoseconds_per_successful_request` and
`exporter_nanoseconds_per_record` use `static/dynamic`. TTFT and ITL
p50/p90/p99 are secondary `static/dynamic` measurements and are rejected as
primary names. Only these throughput, CPU, exporter-duration, and permitted
secondary latency metrics enter the simultaneous max-degradation bootstrap.
Allocation count and allocated bytes are excluded from that distribution and
from its critical-degradation penalty. Each allocation metric is instead an
exact gate over every retained pair using `static/dynamic`: zero/zero is `1.0`,
positive dynamic allocation against zero static allocation fails, and any
ratio below `1.0` fails. Reports use `MetricGateResult` so the two gate kinds
cannot be conflated, and the full report passes only when both gate kinds pass.
A regression test combines varying non-allocation ratios with allocation ratios
fixed at exactly `1.0` and proves the simultaneous penalty is never applied to
the allocation result.
The simultaneous report retains all three required CV vectors
(30 static summaries, 30 dynamic summaries, and 30 positive paired ratios), the
maximum-degradation bootstrap distribution, and invalidation attempts. Tests
pin AB/BA pairing, same-member-order replacement of only invalid pairs, retained
members/reason, refusal to replace valid failures, and the five-replacement/
three-attempt limits.
Exporter sample construction is fixed: one deterministic 100,000-record corpus
and exact pass, 16 sequential repetitions per retained member, identical output
digest and exactly 100,000 records per repetition, one retained 100,000-record
artifact, and `processed_records = 1600000`. Its duration is the sum of active
repetition durations; startup and inter-repetition gaps are excluded, no sleep
or padding is permitted, and valid active duration is at least 30 seconds.
`exporter_nanoseconds_per_record` divides by `processed_records`. Fixed-vector
tests reject changes to `corpus_records`, `sample_repetitions`,
`processed_records`, or `retained_artifact_records` as performance-contract
changes.

The shell runner sets `CARGO_INCREMENTAL=1`, accepts an explicit target path,
and records rather than mutates existing wrapper/cache variables.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p aiperf-bench-tools --test plugin_stats --test plugin_topology`

Expected: all fixed-vector tests pass and two identical invocations emit
byte-identical summaries.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review the measurement code for statistical determinism, allocation noise, and
scope; run focused tests plus Clippy; commit as
`bench(plugins): add reproducible parity harness`; bundle/import/integrate.

### Task 4: Extract boundary-owned core values and host service traits

> **Prerequisite:** run
> [`2026-08-27-plugin-abi-boundary-gap-closure.md`](2026-08-27-plugin-abi-boundary-gap-closure.md)
> first. Its six tasks evict ~75 types from the measured ABI closure
> (193 → ~118) by cutting leak edges this plan does not address:
> `WorkerMaterializer`, `NativeReport`, closed `EndpointType`/`MetricTag`, the
> `ExecutionSinkBuilder::Sink` associated type, and implementation co-resident
> with boundary types. Every type it evicts is a type this task then never has
> to move; running it afterward moves the same code twice.

**Files:**
- Modify: `rust/core/src/lib.rs`
- Create: `rust/core/src/clock.rs`
- Create: `rust/core/src/dispatch.rs`
- Create: `rust/core/src/endpoint.rs`
- Create: `rust/core/src/measure.rs`
- Create: `rust/core/src/report.rs`
- Create: `rust/core/src/artifact.rs`
- Modify: `rust/runtime/src/clock/mod.rs`
- Modify: `rust/runtime/src/dispatch/mod.rs`
- Modify: `rust/runtime/src/clock/runtime_clock.rs`
- Modify: `rust/runtime/src/dispatch/sink.rs`
- Modify: `rust/runtime/src/transport/core/mod.rs`
- Modify: `rust/runtime/src/body_plan.rs`
- Modify: `rust/runtime/src/dataset/materialize.rs`
- Modify: `rust/runtime/src/dataset/segment.rs`
- Modify: `rust/runtime/src/report.rs`
- Create: `rust/core/tests/public_contract.rs`

**Interfaces:**
- Produces transport-neutral request/response, `Clock`, `Dispatchable`,
  `RequestObserver`, observations, measurement, finalized report projection,
  endpoint body-plan/WebSocket-operation values, the narrow segment-reader and
  authored-override views required by endpoint formatters, and capability-
  limited artifact traits used by all category SDKs. Runtime retains adapters
  and compatibility re-exports; no SDK crate depends on `aiperf-runtime`.
- `ArtifactAccess` exposes scoped list/read and approved relative create/write;
  it never exposes a raw directory path or unchecked join.

- [ ] **Step 1: Write the failing public-contract test**

Compile a fixture that imports the new types only from `aiperf_core`, implements
a fake `Clock` and `ArtifactAccess`, and proves no runtime crate dependency is
needed. Add a compile-fail fixture that tries to call `raw_path()` and must fail
because no such method exists.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-core --test public_contract`

Expected: FAIL because the modules/types are absent.

- [ ] **Step 3: Move definitions without changing behavior**

Move the existing definitions, re-export them from their former runtime paths
for internal migration, replace internal imports, and implement runtime-owned
adapters. Do not copy behavior. Preserve `RequestObserver` without `Send` or
`Sync`, all clock calls, SSE byte semantics, and exact report serialization.

- [ ] **Step 4: Verify GREEN**

Run:
`cargo test -p aiperf-core --test public_contract && cargo test -p aiperf-runtime --lib`

Expected: fixture and affected existing tests pass with byte-identical report
snapshots.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review the moved hot-path contracts and adapters, resolve all findings, run
runtime default/engine tests on paper-rig, commit as
`refactor(plugins): extract boundary core contracts`, then bundle/import.

### Task 5: Define the source API, descriptors, ownership table, and rustdoc guard

**Files:**
- Modify: `rust/plugin-api/src/lib.rs`
- Create: `rust/plugin-api/src/id.rs`
- Create: `rust/plugin-api/src/descriptor.rs`
- Create: `rust/plugin-api/src/error.rs`
- Create: `rust/plugin-api/src/extension.rs`
- Create: `docs/specs/plugin-api-ownership.md`
- Create: `rust/plugin-api/tests/ownership_table.rs`
- Create: `rust/plugin-api/src/bin/check-plugin-api-ownership.rs`
- Modify: `.github/workflows/rust-docs-guard.yml`

**Interfaces:**
- Produces `PluginSourceApiVersion`, normalized `RegistryId`, package/category
  descriptors, `PluginDeclarationV1`, `AIPerfExtension`, `PluginRegistrar`, and
  explicit typed errors.
- Fixes the entry shape:

```rust
pub type PluginEntryV1 = unsafe fn() -> PluginDeclarationV1;
pub struct PluginDeclarationV1 {
    pub package: &'static PluginPackageDescriptor,
    pub extension: &'static dyn AIPerfExtension,
}
pub trait AIPerfExtension {
    fn register(
        &self,
        registrar: &mut PluginRegistrar<'_>,
    ) -> Result<(), ExtensionError>;
}
```

- [ ] **Step 1: Write failing normalization and rustdoc-table tests**

Test normalization version 1 exactly: reject non-ASCII; trim ASCII space/tab;
lowercase ASCII; replace each `-` with `_`; require
`^[a-z0-9][a-z0-9_]{0,127}$`; reject empty, other bytes, consecutive authored
separators, redundant aliases, and unsupported versions. The rustdoc guard must
fail when a boundary method/type is absent from the ownership table.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-api`

Expected: FAIL because the API modules and ownership table do not exist.

- [ ] **Step 3: Implement the exact public API surface**

Define only generation-1 endpoint/transport/exporter registration methods.
Each ownership-table row records method, owning crate, argument/return type,
allocation owner, drop owner, `panic=abort`, and startup/hot-path classification.
The rustdoc checker compares exported JSON signatures and fails on additions,
removals, or type drift.

- [ ] **Step 4: Verify GREEN**

Run:
`cargo test -p aiperf-plugin-api && cargo run --manifest-path plugin-api/Cargo.toml --bin check-plugin-api-ownership`

Expected: normalization and ownership-table checks pass.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review the complete public surface for unnecessary ownership, heap allocation,
misleading names, and orchestration leakage; commit as
`feat(plugins): define generation one source API`; bundle/import/integrate.

### Task 6: Define category contracts and split reusable implementation leaves

**Files:**
- Modify: `rust/endpoint-sdk/src/lib.rs`
- Create: `rust/endpoint-sdk/src/config_helpers.rs`
- Create: `rust/endpoint-sdk/src/request_helpers.rs`
- Create: `rust/endpoint-sdk/src/response_helpers.rs`
- Create: `rust/endpoint-sdk/src/grpc.rs`
- Create: `rust/plugin-api/src/category.rs`
- Create: `rust/plugin-api/src/factory.rs`
- Create: `rust/plugin-api/src/prepared.rs`
- Create: `rust/plugin-api/src/transport.rs`
- Modify: `rust/core/src/endpoint.rs`
- Create: `rust/core/src/capture.rs`
- Create: `rust/core/src/services.rs`
- Modify: `rust/runtime/Cargo.toml`
- Modify: `rust/runtime/src/endpoints/mod.rs`
- Modify: `rust/runtime/src/endpoints/config.rs`
- Modify: `rust/runtime/src/endpoints/metadata.rs`
- Modify: `rust/runtime/src/endpoints/models.rs`
- Modify: `rust/runtime/src/endpoints/implementation.rs`
- Modify: `rust/runtime/src/endpoints/registry.rs`
- Modify: `rust/runtime/src/transport/grpc/binding.rs`
- Create: `rust/runtime/src/transport/grpc/kserve_binding.rs`
- Modify: `rust/transport-sdk/src/lib.rs`
- Create: `rust/transport-sdk/src/direct.rs`
- Create: `rust/transport-sdk/src/execution.rs`
- Create: `rust/transport-sdk/src/measure.rs`
- Create: `rust/transport-sdk/src/reduce.rs`
- Create: `rust/transport-sdk/src/retry.rs`
- Create: `rust/transport-sdk/src/service_helpers.rs`
- Create: `rust/transport-sdk/leaf-ownership.toml`
- Create: `rust/transport-sdk/tests/independent_leaves.rs`
- Modify: `rust/runtime/src/transport/core/mod.rs`
- Modify: `rust/runtime/src/transport/measure.rs`
- Modify: `rust/runtime/src/transport/reduce.rs`
- Modify: `rust/runtime/src/transport/retry.rs`
- Modify: `rust/runtime/src/transport/http/sink.rs`
- Modify: `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs`
- Modify: `rust/runtime/src/transport/grpc/sink.rs`
- Create: `rust/runtime/src/transport/ws/sink.rs`
- Modify: `rust/runtime/src/transport/ws.rs`
- Modify: `rust/runtime/src/transport/ws/connector.rs`
- Modify: `rust/runtime/src/transport/ws/dialect.rs`
- Modify: `rust/runtime/src/transport/ws/driver.rs`
- Create: `rust/runtime/src/transport/dry_run.rs`
- Create: `rust/runtime/src/dynosim/direct.rs`
- Modify: `rust/runtime/src/dynosim.rs`
- Modify: `rust/runtime/src/endpoints/dynosim.rs`
- Modify: `rust/runtime/src/engine/execution_factories.rs`
- Modify: `rust/runtime/src/engine/online_execution.rs`
- Modify: `rust/runtime/src/engine/ws_execution.rs`
- Modify: `rust/runtime/src/engine/dry_run.rs`
- Modify: `rust/runtime/src/engine/offline_execution.rs`
- Modify: `rust/plugin-conformance/candidate-source-inventory.toml`
- Modify: `rust/export-sdk/src/lib.rs`
- Create: `rust/export-sdk/src/capture_helpers.rs`
- Create: `rust/export-sdk/src/artifact_helpers.rs`
- Create: `rust/export-sdk/src/helpers.rs`
- Create: `rust/export-sdk/src/prepared_helpers.rs`
- Create: `rust/export-sdk/leaf-ownership.toml`
- Create: `rust/export-sdk/tests/independent_leaves.rs`
- Modify: `rust/plugin-test-support/src/lib.rs`
- Create: `rust/plugin-test-support/src/export.rs`
- Modify: `rust/runtime/src/export/mod.rs`
- Create: `rust/runtime/src/export/adapters.rs`
- Modify: `rust/runtime/src/export/accuracy_csv.rs`
- Modify: `rust/runtime/src/export/analysis_html.rs`
- Modify: `rust/runtime/src/export/analysis_txt.rs`
- Modify: `rust/runtime/src/export/console_txt.rs`
- Modify: `rust/runtime/src/export/dataset_analysis.rs`
- Modify: `rust/runtime/src/export/genai_perf.rs`
- Modify: `rust/runtime/src/export/server_metrics/mod.rs`
- Modify: `rust/runtime/src/export/timeslice.rs`
- Modify: `rust/runtime/src/export/parquet.rs`
- Modify: `rust/runtime/src/export/parquet/units.rs`
- Modify: `rust/runtime/src/export/per_record_parquet.rs`
- Modify: `rust/runtime/src/export/parquet_util.rs`
- Modify: `rust/runtime/src/export/mlflow.rs`
- Modify: `rust/runtime/src/export/wandb/mod.rs`
- Modify: `rust/runtime/src/export/wandb/datastore.rs`
- Modify: `rust/runtime/src/export/wandb/proto.rs`
- Modify: `rust/runtime/src/export/otel.rs`
- Create: `rust/plugin-api/src/validation.rs`
- Create: `rust/plugin-api/src/capture.rs`
- Create: `rust/plugin-api/tests/category_contracts.rs`
- Modify: `rust/plugin-api/src/lib.rs`
- Create: `rust/core/src/histogram.rs`
- Create: `rust/core/tests/histogram_contract.rs`

**Interfaces:**
- `aiperf-plugin-api` solely owns the generation-1 endpoint, transport, and
  exporter boundary traits and vocabulary: `EndpointFactory`,
  `TransportFactory`, `ExporterFactory`, the native `Endpoint` compatibility
  trait, `PreparedEndpoint`, opaque validated/prepared handles, exactly-one
  transport execution-shape values, readiness/WebSocket capabilities, optional
  `GrpcEndpointBindingFactory`, exporter capture requirements,
  `FactoryValidationReceiptV1`, category descriptors, category errors/outcomes,
  and other category-specific boundary vocabulary. This ownership explicitly
  excludes the transport-neutral product/service values assigned to core below,
  even when an API-owned trait method references them. In particular,
  `aiperf-plugin-api::transport` owns `ExecutionSinkBuilder`, `WorkerSink`, and
  the boundary request/terminal contexts. `aiperf-core` solely owns the
  transport-neutral product and service values used by those contracts,
  including
  `RawEndpointConfig`, `EffectiveEndpointConfig`, reset/profiler/content-type
  policy values, request/response/turn/media DTOs, `PreparedRequest`, finalized
  report/capture projections, histogram vocabulary, and the narrow
  clock/graph/metrics/artifact/cancellation service traits. The runtime endpoint
  modules become temporary compatibility re-export/adapter homes; runtime-only
  `EndpointRegistryBuilder`, frozen lookup/table state, legacy descriptor
  mapping, and concrete dialects remain in runtime until their owning tasks.
- The same task replaces every runtime definition of those API/core-owned types
  with API/core imports/re-exports, so there is exactly one Rust type identity.
  It does not copy or independently redefine config/model/descriptor/trait
  vocabulary in a category SDK or candidate package.
- `aiperf-endpoint-sdk`, `aiperf-transport-sdk`, and `aiperf-export-sdk` own
  only shared pure plugin-private helpers with isolated dependency surfaces. A
  category-SDK-defined concrete type MUST NOT occur in
  an exported plugin-API/core boundary signature, trait-object vtable,
  allocation/drop contract, or host-owned stored value. The category SDKs may
  supply helper implementations for endpoint formatting/binding,
  reduction/measurement/retry, exporter formatting, and capability-limited
  service adapters, but those helpers consume and produce API/core-owned
  boundary values. Their compiled artifacts remain plugin-private inputs and
  are selectively rebuilt only for actual consumers.
- The transport SDK owns reusable implementation helpers for the API-owned
  transport traits and core-owned services, not their boundary type identities.
  Runtime retains scheduling, admission, phase orchestration, capture, and
  adapters, but no plugin implementation leaf names `RunContext`,
  `aiperf-runtime`, or a private engine/metrics/scheduled/multiturn type.
- This task performs the behavior-preserving production source split required
  for later equality copies. HTTP and gRPC sink leaves are rewritten against
  SDK/core contexts while host adapters map existing scheduling state. WebSocket
  gains plugin-owned `transport/ws/sink.rs`; `engine/ws_execution.rs` remains
  a host adapter. Dry-run gains plugin-owned `transport/dry_run.rs`;
  `engine/dry_run.rs` remains a host adapter. Dynosim gains plugin-owned
  `dynosim/direct.rs`; `dynosim.rs` and `engine/offline_execution.rs` remain
  host orchestration/adapters. Later candidates hash/copy only those
  plugin-owned leaves, never the engine adapters.
- Those runtime leaf paths are temporary source-staging locations while static
  parity remains active. Their final compiled owners are the named plugin
  packages in `candidate-source-inventory.toml`, not core or a category SDK.
  Tasks 24–34 copy the pinned sources into those packages; Task 39 removes the
  runtime-compiled static copies. After Task 39, runtime compiles only host
  adapters and each backend-specific leaf is compiled only by its plugin
  package.
- Task 2 records not-yet-existing split leaves as `planned` rows with exact
  `producer_task = 6` and destination. This task atomically changes those rows
  to `implementation_leaf`, records their post-split BLAKE3 digests, and proves
  every former mixed/private source is either a named host adapter or absent
  from candidate ownership. Tasks 24–34 never edit this inventory.
- The generic gRPC binding factory/DTO boundary contract moves to API/core;
  reusable binding helpers move to endpoint SDK.
  `runtime/transport/grpc/binding.rs` retains only host registry/builder state;
  concrete KServe implementation moves to
  `runtime/transport/grpc/kserve_binding.rs`, which Task 30 may equality-copy.
- `aiperf-plugin-api` solely owns the post-report execution boundary
  (`PreparedExporterV1`) and exporter error/outcome values. `aiperf-export-sdk`
  owns the shared pure compatibility helpers `SummarySeries`, `CanonicalStats`, `summary_series`,
  `flatten_stats`, `finite_passthrough`, `finite_guarded`, `crlf_csv_writer`,
  `normalize_endpoint_display`, and `default_run_name`. Their report/stat input
  types are the exact `aiperf-core` finalized-report types moved by Task 4;
  artifact I/O uses only `ArtifactAccess`. No helper accepts a raw artifact
  directory or a runtime-private report/config type.
- This task also performs the behavior-preserving exporter source split needed
  before Tasks 24–28 freeze hashes. Every exporter implementation leaf listed
  in `candidate-source-inventory.toml` imports only API/core/export-SDK values
  plus its own package-internal leaves; each exporter entrypoint accepts only
  its implementation-owned validated config. No listed leaf contains
  `crate::export`, `aiperf-runtime`, `ExporterRegistry`, monolithic
  `ExportConfig`, a raw artifact path, or a sibling-backend dependency. The legacy
  closed `ExportConfig`, static `Exporter` compatibility trait,
  `ExporterRegistry`, built-in ordering/composition, and adapters remain
  host-owned in `runtime/src/export/{mod,adapters}.rs` until Tasks 18/39. Those
  adapters call the same split leaves, so Task 6 changes ownership/import
  topology without changing production authority or behavior.
- `export-sdk/leaf-ownership.toml` classifies every exporter source as
  `host_adapter`, `shared_sdk_helper`, `package_internal_leaf`, or
  `plugin_leaf`, pins the final candidate owner and post-split BLAKE3, and is
  reconciled atomically with the central candidate inventory. Shared report,
  artifact, summary/stat, CSV, endpoint-display, and default-name behavior has
  exactly one SDK/core implementation; it is never copied into a candidate.
  `plugin-test-support::export` owns the common deterministic report fixture and
  capability-scoped temporary artifact harness used by exporter leaf tests; it
  is test-only and is not a plugin boundary dependency.
- API-owned `ExporterCaptureRequirementsV1` is a sorted set over exactly
  `FinalReport`, `ExactRecordsV1`, and
  `FoldedProjectionV1(GenAiClientHistogramsV1)`.
- `aiperf-plugin-api` solely defines and exports
  `FactoryValidationReceiptV1`, including selected **category**, canonical
  factory ID, descriptor digest, authored-config digest, semantic-config digest,
  sorted host resources, and canonical capture
  requirements. `aiperf-core::capture` solely defines the complete public
  `ExactRecordV1`, `ExactRecordsV1`, `GenAiClientHistogramsV1`,
  `ExplicitHistogramV1`, projection schema/version, metric/dimension/bounds,
  ordering, and capture error DTO vocabulary. Plugin API references or
  re-exports those exact core types and never redefines them; Task 19 implements
  behavior but never redefines the types.
- Direct transport services are narrow host traits for clock, graph, metrics,
  artifacts, and cancellation; `RunContext` is not representable.

- [ ] **Step 1: Write failing contract/compile-fail fixtures**

Compile one minimal factory per category from a separate fixture crate. Add
compile-fail fixtures for a transport declaring both execution shapes, an
exporter inventing a projection ID, and a plugin importing `RunContext`.
Add compile-fail/source guards rejecting cross-artifact `Any`/`TypeId` and host
downcasts of plugin-defined values. Positive fixtures prove `FinalReport` is
always available, exporter `{id, config}` defaults `config` to `{}`, and a
third-party exporter band/tie-break cannot place an uploader before a local
writer or use authored list order to override descriptor ordering.
Golden-test the complete receipt key set, including `category`, and mutate each
field independently. Golden-test every core-owned metric-source alias, unit
conversion, explicit bounds array, inclusion rule, and attribute-normalization
constant, including first-upper-bound selection at `value <= bound`.
Compile a separate-workspace endpoint plugin that imports every common endpoint
boundary type only from `aiperf-plugin-api` or `aiperf-core` and imports only
pure helpers from `aiperf-endpoint-sdk`; source guards reject duplicate
definitions in runtime/category SDKs and any `aiperf-runtime` dependency from
API/core/category SDKs.
Compile the actual post-split HTTP, gRPC, WebSocket, dry-run, Dynosim-direct,
KServe-binding, and Riva-binding implementation leaves in a standalone fixture
using only plugin API/core/category SDK dependencies. `leaf-ownership.toml` classifies
every source as `host_adapter` or `plugin_leaf`, pins its candidate owner and
BLAKE3, and rejects an unclassified leaf, private runtime import, `RunContext`,
or duplicate shared reduction/measurement/retry implementation.
Compile every post-split exporter implementation leaf from the central
candidate inventory in standalone package-shaped fixtures using only its own
backend dependencies plus API/core/export SDK. The export ownership test rejects
an unclassified source, `crate::export`/`aiperf-runtime`, `ExporterRegistry`,
monolithic `ExportConfig`, a raw artifact path, a copied shared helper, or one
telemetry backend importing another. It also proves each runtime legacy adapter
and its candidate facade invoke the same BLAKE3-pinned leaf.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-api --test category_contracts`

Expected: FAIL because factory and receipt types are absent.

- [ ] **Step 3: Implement minimal native contracts**

Use sealed host-owned enums only for execution shape and capture vocabulary,
not factory IDs. Opaque validated configuration remains owned by the exact
factory and is returned with a receipt containing selected category, canonical
factory ID, descriptor digest, authored-config digest, semantic-config digest,
sorted host resources, and canonical capture requirements.
`aiperf-core::histogram` owns the exact metric-source aliases, unit conversions,
explicit bounds arrays, inclusion rules, attribute-normalization constants, and
the first-upper-bound rule `value <= bound`; no exporter owns a private copy.
Move the common boundary definitions listed in Interfaces into
`aiperf-plugin-api` or `aiperf-core` according to their stated ownership,
preserve existing source/serde behavior exactly, and leave runtime
compatibility re-exports so downstream product migration can be staged without
a second type family. Category SDK files MUST NOT define the boundary factory
traits.
Move shared transport boundary contracts once into API/core and reusable
reduction/measurement/retry logic into transport SDK, split the concrete leaves
and host adapters at the exact paths listed above, and preserve runtime
compatibility re-exports. The runtime continues to execute the same static
factories until Task 39.
Likewise, move the shared exporter helpers once into export SDK, rewrite each
listed exporter leaf around its implementation-owned config plus finalized
core report/capture and `ArtifactAccess`, and leave legacy static registry,
closed config, ordering, and compatibility adapters in runtime. Freeze both
leaf-ownership manifests and the central candidate inventory only after the
standalone package-shaped fixtures compile. This task changes ownership/import
topology, not production authority.

- [ ] **Step 4: Verify GREEN**

Run:
`cargo test -p aiperf-plugin-api --test category_contracts && cargo test -p aiperf-core --test histogram_contract && cargo test -p aiperf-endpoint-sdk -p aiperf-transport-sdk -p aiperf-export-sdk -p aiperf-plugin-test-support && cargo test -p aiperf-transport-sdk --test independent_leaves && cargo test -p aiperf-export-sdk --test independent_leaves && cargo test -p aiperf-runtime --features engine`

Expected: positive fixtures compile/run and negative fixtures fail for the
expected type-system reason.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review allocation and async shape at every method, resolve findings, commit as
`feat(plugins): define native category factory contracts`, bundle/import.

### Task 7: Gate and install the authoritative shared allocator provider topology

**Files:**
- Modify: `rust/allocator-provider/Cargo.toml`
- Modify: `rust/allocator-provider/src/lib.rs`
- Modify: `rust/allocator-provider/build.rs`
- Modify: `rust/allocator-shim/Cargo.toml`
- Modify: `rust/allocator-shim/src/lib.rs`
- Modify: `rust/cli/Cargo.toml`
- Modify: `rust/cli/build.rs`
- Modify: `rust/cli/src/main.rs`
- Modify: `rust/cli/src/execute_mode.rs`
- Delete: `rust/cli/src/mimalloc_options.c`
- Modify: `rust/Cargo.lock`
- Create: `.github/workflows/rust-plugin-allocator.yml`
- Create: `rust/plugin-conformance/fixtures/allocator-candidate-host/Cargo.toml`
- Create: `rust/plugin-conformance/fixtures/allocator-candidate-host/src/main.rs`
- Create: `rust/plugin-conformance/fixtures/allocator-candidate-host/candidate.yaml`
- Create: `rust/plugin-conformance/fixtures/allocator-candidate-plugin/Cargo.toml`
- Create: `rust/plugin-conformance/fixtures/allocator-candidate-plugin/src/lib.rs`
- Create: `rust/plugin-conformance/tests/allocator.rs`
- Create: `rust/scripts/inspect-plugin-allocator.rs`

**Interfaces:**
- Produces authoritative distribution-baseline module identity `aiperf_alloc_v1`, direct
  imported `mi_malloc`, `mi_zalloc`, aligned allocation/reallocation, and
  `mi_free` symbols, plus one `GlobalAlloc` shim shared as exact prebuilt input.
- Zero-size sentinels/null are never freed; alignment routing and realloc
  failure obey the exact semantics in the spec. After and only after this task’s
  four-target conformance, constructor/memory checks, and authoritative
  paper-rig static-mimalloc-versus-provider non-inferiority gate pass, the same
  reviewed commit makes the provider mandatory for the ordinary `aiperf`
  executable. It replaces the executable-owned mimalloc instance/preinit hook
  with the direct provider-importing shim, adds the mandatory/non-delay loader
  dependency, verifies the mapped provider and host relocation origins at the
  first AIPerf instruction, and supplies exact universe/build-record inputs.
  Task 17 and every native entry invocation depend on this authoritative state.
  No owned `Global` boundary value crosses before the Task-7 commit is
  integrated. Task 36 later re-runs the full matrix against the exact integrated
  topology; it is not the authority cutover.

- [ ] **Step 1: Write failing allocation/import tests**

Add real dynamic-loader subprocess fixtures covering ordinary, zeroed, aligned,
reallocated, and freed `String`, `Vec`, `Box`, `Arc`, `Rc`, error, trait object,
and boxed future storage in both directions. Add static import-map assertions
that fail if any shim lacks direct `mi_*` imports or contains an AIPerf wrapper.
Enumerate and require `mi_malloc`, `mi_zalloc`, `mi_malloc_aligned`,
`mi_zalloc_aligned`, `mi_realloc`, `mi_realloc_aligned`, `mi_free`, `mi_subproc_main`, and
`mi_version`; check `mi_version == 30500`, identical process-global
`mi_subproc_main()` pointer, eager relocation origin, and checked `calloc`
multiplication overflow when a calloc path is used.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-conformance --test allocator`

Expected: FAIL because the provider/shim and imported artifacts do not exist.

- [ ] **Step 3: Implement provider and shim**

Package the pinned shared mimalloc binary under platform loader names, first
build the production-shaped candidate host with a mandatory non-delay dependency,
make allocator option initialization provider-owned, inject the same shim into
the candidate plugin build, enable eager binding, enumerate/verify the mapped
provider at first AIPerf instruction, and validate host relocation origins
before discovery. After Step 4 passes on the exact candidate, apply that same
topology to ordinary `aiperf`: remove the executable-owned preinit C bridge and
embedded allocator, link the reviewed provider/shim, and rerun the identical
artifact-origin and performance gates against the final commit.

- [ ] **Step 4: Verify GREEN on all four targets**

Run the allocator conformance suite and import/runtime-origin inspectors on
Linux x86_64, macOS ARM64, Windows x86_64, and Windows ARM64, including
preload/interposition and constructor-order fixtures. On an otherwise-idle
paper-rig, compare the old static-mimalloc executable and the exact proposed
provider executable for startup, steady-state, teardown, allocation
count/bytes, CPU/request, throughput, latency, and RSS under the Task-3 paired
protocol. Expected: all ownership, origin, eager-binding, option, memory, abort,
and simultaneous non-inferiority gates pass before the production edit is
committed.
The checked workflow builds the exact commit on all four targets, uploads final
provider/host/plugin import maps and runtime receipts, and makes the four job
IDs/artifact digests mandatory tracker evidence; a result from a different lab
revision cannot authorize the product cutover.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review every unsafe block, layout branch, symbol binding, initialization order,
and hot-path instruction; commit as
`feat(plugins): install shared allocator provider`; bundle/import every
distinct RED, candidate, and production-cutover commit after four-target and
paper-rig approval. Integrate the exact reviewed production object; Task 17
cannot start from an earlier base.

### Task 8: Generate exact host-universe and plugin-build records

**Files:**
- Create: `rust/plugin-sdk/src/identity.rs`
- Create: `rust/plugin-sdk/src/canonical.rs`
- Create: `rust/plugin-sdk/src/artifact_section.rs`
- Create: `rust/plugin-sdk/src/inspect.rs`
- Create: `rust/plugin-sdk/tests/identity.rs`
- Create: `rust/plugin-sdk/tests/artifact_section.rs`
- Create: `rust/plugin-sdk/tests/fixtures/identity/`
- Modify: `rust/plugin-sdk/src/lib.rs`
- Create: `rust/plugin-sdk/src/abi_closure.rs`
- Create: `rust/plugin-sdk/tests/abi_closure.rs`

**Interfaces:**
- Consumes the exact integrated Task-7 provider/shim/import-map object and binds
  that authoritative allocator identity; it cannot run from the earlier static-
  mimalloc baseline or a candidate-only provider.
- Produces canonical `HostAbiUniverseRecordV1`,
  `PluginArtifactBuildRecordV1`, `HostAbiUniverseId`, and
  `PluginArtifactBuildId` with the exact field sets in the spec.
- Canonical bytes are sorted, length-delimited, path-remapped, reject unknown
  fields, and never contain host-specific absolute paths.
- Produces the reviewed, checked-in common-ABI artifact classifier and its
revocation record. The classifier includes private fields, generated code, and
every representation/validity/drop dependency crossing the boundary; it moves a
previously private artifact to common on first crossing and rejects/revokes a
record built from an incomplete set. It records observed system-library
version/build identities where available and target policy version.

- [ ] **Step 1: Write failing canonicalization/mismatch tests**

Use golden records to assert the exact serialized key census from the spec and
test an independent one-field mutation for every field: rustc executable digest,
commit/full version, sysroot and proc-macro binaries, target triple/CPU/features,
pointer width/endian, codegen backend, normalized cfg/codegen flags, ABI crates,
allocator/panic, target/linker policies, complete rustc/linker invocation
digests, common/private sources and features, hermetic environment, build
scripts, generated sources/outputs, native/system dependency identities,
pre-embedding link payload, loader identity, observed system libraries, and
final embedded artifact. Reject every unknown field.
Assert common changes alter every host-universe ID while private changes alter
only that package-build ID. Test that no build ID hashes bytes containing itself.
Test generated/private-field crossing, closure reclassification/revocation,
allowlist-file mutation changing the host universe, observed system-library
identity capture/diagnostics, and a checked-in artifact-set review golden.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-sdk --test identity`

Expected: FAIL because identity modules are absent.

- [ ] **Step 3: Implement exact records and platform sections**

Emit/parse non-executable ELF, Mach-O, and PE record sections. Bind the link
payload before record embedding and separately expose final artifact digest.
Implement first-field-difference diagnostics without claiming ABI proof.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p aiperf-plugin-sdk --test identity && cargo test -p aiperf-plugin-sdk --test artifact_section && cargo test -p aiperf-plugin-sdk --test abi_closure`

Expected: golden bytes/digests and every mismatch classification pass.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review canonicalization, self-reference avoidance, path remapping, and parsing
bounds; commit as `feat(plugins): generate exact build identities`; bundle/import.

### Task 9: Generate the native entry symbol and hermetic author build command

**Files:**
- Modify: `rust/plugin-sdk-macros/Cargo.toml`
- Modify: `rust/plugin-sdk-macros/src/lib.rs`
- Create: `rust/plugin-sdk/src/declaration.rs`
- Create: `rust/plugin-sdk/src/build.rs`
- Create: `rust/plugin-sdk/src/sandbox.rs`
- Create: `rust/plugin-sdk/src/bin/cargo-aiperf-plugin.rs`
- Create: `rust/plugin-sdk/src/bin/aiperf-plugin-inspect.rs`
- Create: `rust/plugin-sdk/src/bin/aiperf-plugin-conformance.rs`
- Create: `rust/plugin-sdk/src/manifest.rs`
- Create: `rust/plugin-sdk/src/conformance.rs`
- Create: `rust/plugin-sdk/tests/entry_symbol.rs`
- Create: `rust/plugin-sdk/tests/hermetic_build.rs`
- Create: `rust/plugin-sdk/tests/author_tools.rs`
- Modify: `rust/plugin-sdk/src/lib.rs`
- Create: `rust/plugin-conformance/fixtures/minimal-plugin/Cargo.toml`
- Create: `rust/plugin-conformance/fixtures/minimal-plugin/src/lib.rs`
- Create: `rust/plugin-sdk/examples/minimal-plugin/Cargo.toml`
- Create: `rust/plugin-sdk/examples/minimal-plugin/src/lib.rs`

**Interfaces:**
- Produces `#[aiperf_plugin]` glue exporting only
  `aiperf_plugin_entry_v1` with native Rust ABI and returning a borrowed-static
  `PluginDeclarationV1`.
- Produces `cargo aiperf-plugin build --release [--sdk <directory>]`, which
  verifies toolchain, exact prebuilt ABI artifacts, `cdylib`, `panic=abort`,
  linker policy, allocator imports, export map, executable closure, records,
  manifest, and final hashes in a network-disabled declared-input sandbox.
- Ships SDK-local strict manifest generation/validation, build/universe/artifact
  closure inspection, a minimal third-party example, and a local conformance
  runner. These author tools depend only on published SDK/API/core artifacts and
  never on orchestration-private crates.

- [ ] **Step 1: Write failing real-artifact tests**

Build the minimal external fixture with ordinary `cargo build` and assert the
SDK validator rejects it. Invoke the absent SDK command and assert failure.
Define expected export-map output containing exactly
`aiperf_plugin_entry_v1` and expected sandbox rejections for undeclared file,
environment, network, and writes outside private output. Positively test fixed
and recorded clock, randomness, locale, cwd, and path remapping plus declared
private output writes.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-sdk --test entry_symbol && cargo test -p aiperf-plugin-sdk --test hermetic_build && cargo test -p aiperf-plugin-sdk --test author_tools`

Expected: FAIL because the macro and command are absent.

- [ ] **Step 3: Implement macro and controlled build**

The macro emits:

```rust
#[unsafe(export_name = "aiperf_plugin_entry_v1")]
pub unsafe fn __aiperf_plugin_entry_v1() -> PluginDeclarationV1 {
    PluginDeclarationV1 { package: &PACKAGE, extension: &EXTENSION }
}
```

The command consumes the SDK bundle’s toolchain/allowlist/prebuilt artifacts,
normalizes admitted inputs and environment, uses private output, embeds Task 8
records, validates the Task 7 allocator/panic/import policy, and emits strict
schema-2.0 artifacts. It rejects author-selected allocator/symbol/crate type.

- [ ] **Step 4: Verify GREEN**

Run:
`cargo test -p aiperf-plugin-sdk --test entry_symbol && cargo test -p aiperf-plugin-sdk --test hermetic_build && cargo test -p aiperf-plugin-sdk --test author_tools && cargo run -p aiperf-plugin-sdk --bin cargo-aiperf-plugin -- aiperf-plugin build --release --manifest-path plugin-conformance/fixtures/minimal-plugin/Cargo.toml --artifact-path-file target/aiperf-plugin/minimal-plugin/artifact-path.txt && cargo run -p aiperf-plugin-sdk --bin aiperf-plugin-inspect -- --artifact-path-file target/aiperf-plugin/minimal-plugin/artifact-path.txt && cargo run -p aiperf-plugin-sdk --bin aiperf-plugin-conformance -- --manifest plugin-conformance/fixtures/minimal-plugin/plugins.yaml`

Expected: the conforming fixture builds; undeclared-input fixtures fail with
stable codes; export/import/record inspection passes.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review proc-macro expansion, command execution, sandbox authority, path
remapping, subprocess cleanup, and diagnostics; commit as
`feat(plugins): add controlled plugin author build`; bundle/import.

### Task 10: Implement strict schema-2.0 manifest decoding and normalization

**Files:**
- Modify: `rust/plugin-host/Cargo.toml`
- Modify: `rust/plugin-host/src/lib.rs`
- Create: `rust/plugin-host/src/manifest.rs`
- Create: `rust/plugin-host/src/normalize.rs`
- Create: `rust/plugin-host/src/error.rs`
- Create: `rust/plugin-host/schema/plugins-2.0.schema.json`
- Create: `rust/plugin-host/tests/manifest.rs`
- Create: `rust/plugin-host/tests/fixtures/manifests/`

**Interfaces:**
- Produces strict DTOs `PluginManifestV2`, `PluginPackageManifestV2`,
  `BaselineRequirementV2`, `ArtifactRecordV2`, category entries, dependency
  edges, and `NormalizedIdV1`.
- Accepts only exact native root shape and `schema_version: "2.0"`; Python
  schema/root returns stable code `python-plugin-manifest-not-native`.

- [ ] **Step 1: Write failing manifest fixture matrix**

Cover the complete example from the spec plus unknown fields at every level,
missing required fields, noncanonical SemVer, absolute/parent/ADS paths,
baseline artifact duplication, unsupported category, no category, invalid
priority, aliases, artifact edges, Python schema 1.0, and every normalization
rule. Assert canonical serialization and stable error codes.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-host --test manifest`

Expected: FAIL because the host crate/decoder is absent.

- [ ] **Step 3: Implement strict decoding and generated schema**

Use `deny_unknown_fields`, canonical SemVer, signed `i32` default-zero priority,
empty description/metadata defaults, sorted unique aliases/edges, and relative
normalized artifact paths. Generate JSON Schema from the same DTO contract and
check it into `schema/plugins-2.0.schema.json` byte-for-byte.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p aiperf-plugin-host --test manifest`

Expected: all fixture classifications and schema golden comparison pass.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review parser bounds, canonicalization, error content, and schema drift guard;
commit as `feat(plugins): decode native plugin manifests`; bundle/import.

### Task 11: Acquire immutable manifests and artifact closures

**Files:**
- Create: `rust/plugin-host/src/acquire.rs`
- Create: `rust/plugin-host/src/closure.rs`
- Create: `rust/plugin-host/src/stage.rs`
- Create: `rust/plugin-host/src/platform/fs.rs`
- Create: `rust/plugin-host/src/platform/mod.rs`
- Modify: `rust/plugin-host/src/lib.rs`
- Create: `rust/plugin-host/tests/acquisition.rs`
- Create: `rust/plugin-host/tests/acquisition_races.rs`
- Create: `rust/plugin-host/tests/fixtures/closures/`

**Interfaces:**
- Produces `AcquiredManifest`, `AcquiredArtifact`, `AcquiredClosure`,
  content-addressed `StagedObject`, and process-wide `CanonicalObjectMap` keyed
  by `(loader_identity, digest)` with origin
  `Executable|Baseline|CanonicalStage`.
- Every open is no-follow and directory-handle relative. The loader receives
  only a private staged absolute path after reopen and rehash.

- [ ] **Step 1: Write failing path/race/closure tests**

Use real files and race barriers to replace manifest, main library, generation
directory, and private dependency at acquire/stage/load boundaries. Cover
symlinks, junctions/reparse points, traversal, hardlink metadata changes,
unresolved ambient dependencies, identical-claim coalescing, baseline typed
reuse, conflicting loader identity/digest, and staged-byte tamper.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-host --test acquisition --test acquisition_races`

Expected: FAIL because acquisition/staging APIs are absent.

- [ ] **Step 3: Implement handle-relative acquisition and staging**

Acquire each discovery input once, retain presence-tagged raw/canonical bytes,
open the selected immutable generation, hash exact handles, validate complete
non-system dependency edges, copy into a private host-owned mode-immutable
content-addressed generation, reopen/rehash, and coalesce identical objects.
Never mutate loader search environment.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p aiperf-plugin-host --test acquisition --test acquisition_races`

Expected: every replacement is either pinned to original bytes or rejected;
no staged load path observes attacker replacement.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review all filesystem authority, handles, unsafe/platform calls, cleanup RAII,
and race tests; commit as `feat(plugins): acquire immutable artifact closures`;
bundle/import.

### Task 12: Statically inspect platform artifacts before loading

**Files:**
- Create: `rust/plugin-host/src/inspect.rs`
- Create: `rust/plugin-host/src/platform/elf.rs`
- Create: `rust/plugin-host/src/platform/macho.rs`
- Create: `rust/plugin-host/src/platform/pe.rs`
- Create: `rust/plugin-host/src/platform/mod.rs`
- Create: `rust/plugin-host/tests/static_inspection.rs`
- Create: `rust/plugin-conformance/fixtures/artifact-mismatch/`
- Modify: `rust/plugin-host/src/lib.rs`

**Interfaces:**
- Consumes Task 8 build records and Task 11 acquired objects.
- Produces `StaticallyValidatedCatalog` plus typed receipts for artifact kind,
  digest, record, compiler/sysroot/target, dependency/search policy, allocator
  import/eager binding, panic strategy, exported symbols, constructor sections,
  and loader identity.

- [ ] **Step 1: Write failing real-binary mismatch fixtures**

Build ELF/Mach-O/PE fixtures for wrong kind, stale universe, tampered package
record, wrong main identity, missing/extra dependency, ambient search path,
allocator wrapper/boundary-escaping other allocator/lazy binding, unwind profile, extra export,
constructor section, identity collision, and final-signing digest drift.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-host --test static_inspection`

Expected: FAIL because the inspectors are absent.

- [ ] **Step 3: Implement bounded platform inspection**

Parse each target format without executing code; validate exact Task 8 embedded
records and Task 7 imports; enforce `$ORIGIN`, `@loader_path`, or approved PE
search policy; calculate sorted dependency graph/identities; emit stable
receipts with no free-form text in lock inputs.
Normalize Mach-O leading-underscore symbol spellings and preserve a regression
where a native inspector fails through a shell pipeline. Permit a private native
allocator only when allocation and final free remain wholly inside that library.

- [ ] **Step 4: Verify GREEN on target matrix**

Run the same test on each target runner. Expected: every bad artifact is
quarantined before entry; every good artifact yields canonical identical
receipt fields where platform-independent.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review parser bounds and all trust decisions; commit as
`feat(plugins): inspect native artifacts before activation`; bundle/import.

### Task 13: Implement discovery sources, authority, aliases, and fixed priority

**Files:**
- Create: `rust/plugin-host/src/discovery.rs`
- Create: `rust/plugin-host/src/authority.rs`
- Create: `rust/plugin-host/src/priority.rs`
- Create: `rust/plugin-host/src/catalog.rs`
- Create: `rust/plugin-host/tests/discovery.rs`
- Create: `rust/plugin-host/tests/priority.rs`
- Create: `rust/plugin-host/tests/authority.rs`
- Modify: `rust/plugin-host/src/lib.rs`

**Interfaces:**
- Consumes decoded/acquired/static receipts and produces `IntendedCatalog` plus
  immutable winner/shadow/ambiguous/quarantined maps and exact load set.
- Discovery source IDs are canonical
  `(source_kind_ordinal, authored_index)` over distribution, platform-system,
  platform-user, environment, explicit-directory, explicit-manifest, and
  hermetic-bundle sources.

- [ ] **Step 1: Write failing discovery/priority matrix**

Cover every OS default root/order, missing versus unreadable roots, empty/invalid
`AIPERF_PLUGIN_PATH`, `--no-auto-plugins`, explicit ordering/authority,
nonrecursive exact basenames, directory sorting, identical dedupe, multiple
versions, identity conflicts, canonical-first aliases, alias conflicts,
redundant aliases, normalization collisions, signed priorities, unique maxima,
equal maxima, required packages, required component keys, fully shadowed
optional/required packages, and no-promotion load set.
Cover invalid/unreadable/empty/wrong-kind/missing explicit paths as fatal. Add
non-executing `--plugins-manifest-only`, the sole flag allowed to omit the
distribution source; prove every run command rejects it and `--no-auto-plugins`
still retains distribution. Golden receipts enumerate all sort tuple/ordinal,
basename regex, relative-ID, presence, logical-object, and evidence fields.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-host --test discovery --test priority --test authority`

Expected: FAIL because resolution modules are absent.

- [ ] **Step 3: Implement steps 1–6 of the eight-phase resolution algorithm**

Implement spec algorithm steps 1–6 only: acquire, dedupe/authority, static
inspect/quarantine, canonical/alias grouping, unique-max resolution, and fixed
load-set establishment. Sort every failure receipt by the normative tuple and
retain all reasons. No loader call occurs in this task.

- [ ] **Step 4: Verify GREEN and determinism**

Run the three tests repeatedly with randomized filesystem enumeration and input
insertion order. Expected: canonical catalog bytes/digest are identical and
winners never depend on enumeration/load order.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review authority semantics, tie handling, sorting, and no-promotion guarantees;
commit as `feat(plugins): resolve immutable intended catalog`; bundle/import.

### Task 14: Activate libraries with process-lifetime residency and poison

**Files:**
- Create: `rust/plugin-host/src/loader.rs`
- Create: `rust/plugin-host/src/residency.rs`
- Create: `rust/plugin-host/src/platform/loaded_modules.rs`
- Create: `rust/plugin-host/tests/loader.rs`
- Create: `rust/plugin-host/tests/residency.rs`
- Create: `rust/plugin-host/tests/poison.rs`
- Create: `rust/plugin-conformance/fixtures/loader/`
- Modify: `rust/plugin-host/src/lib.rs`

**Interfaces:**
- Produces process-global `ActivatingLibrarySet`, the storage-capable but not yet
  constructible `LoadedLibrarySet` type, and `PoisonedLibrarySet`. Task 14 never
  seals success because actual registrations and the canonical lock do not yet
  exist. Any loader entry establishes poison; every returned handle is retained
  before symbol lookup; no close/drop/unload path exists. Task 16 alone seals an
  activating set with the derived lock digest and enables identical-lock reuse.
- Linux opens `RTLD_NOW|RTLD_LOCAL` with retained handle and optional
  `RTLD_NODELETE`; macOS uses eager local loading; Windows uses fully qualified
  `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` plus approved flags and pins each module.

- [ ] **Step 1: Write failing loader subprocess fixtures**

Use real `cdylib`s for success, dependency failure, initializer failure before
handle, entry-symbol absence, entry panic/abort, descriptor failure, plugin
object `Drop`, attempted unload, same-lock reuse, different-lock reuse, preload,
interposition, loaded-module collision, and winning-package failure with a
lower-priority candidate present.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-host --test loader --test residency --test poison`

Expected: FAIL because loader/residency APIs are absent.

- [ ] **Step 3: Implement activation and runtime origin checks**

Enumerate and authenticate executable/baseline modules before discovery; map
canonical staged objects; retain each returned handle immediately; verify
actual loaded module identities/paths/digests and Task 7 relocation targets;
resolve the entry from the exact handle; preserve original poison error forever.

- [ ] **Step 4: Verify GREEN**

Run all loader tests in isolated subprocesses. Expected: success objects drop
before process exit while code remains resident; poison never retries/promotes;
different-lock reuse fails; panic fixtures terminate abnormally rather than
returning caught errors.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review every unsafe loader operation, global state transition, RAII boundary,
platform flag, and error path; commit as
`feat(plugins): retain and poison native libraries`; bundle/import.

### Task 15: Make package registration transactional and freeze the universe

**Files:**
- Modify: `rust/runtime/src/extensions/mod.rs`
- Modify: `rust/runtime/src/extensions/transactional.rs`
- Modify: `rust/runtime/src/extensions/registry_id.rs`
- Modify: `rust/plugin-api/src/extension.rs`
- Create: `rust/plugin-api/src/frozen.rs`
- Modify: `rust/plugin-api/src/lib.rs`
- Create: `rust/plugin-host/src/register.rs`
- Create: `rust/plugin-host/src/freeze.rs`
- Create: `rust/plugin-host/tests/registration.rs`
- Modify: `rust/runtime/tests/extensions_compile_time_extension.rs`
- Modify: `rust/plugin-host/src/lib.rs`

**Interfaces:**
- Produces plugin-API-owned `RegistryBuilder`, manifest-bound package-scoped
  `PluginRegistrar<'_>`, and `FrozenPluginUniverse` containing
  `FrozenAIPerfRegistry`, endpoint bindings, exporter factories/capture
  vocabulary, direct bindings, descriptors, provenance, and resident handle-set
  interface. `aiperf-plugin-host` depends only on API/core/SDK, retains concrete
  process-global handles, and returns the plugin-API-owned
  `FrozenPluginUniverse`; runtime depends downward on API/core and consumes that
  view while CLI composes host plus runtime.
- Freeze carries the still-unsealed `ActivatingLibrarySet`; it cannot claim a
  reusable `LoadedLibrarySet` until Task 16 derives the canonical lock over the
  actual frozen registrations.
- Freeze consumes the builder. Frozen types expose lookup/catalog only; no
  registration, mutable category accessor, thaw, or fresh built-in factory.

- [ ] **Step 1: Write failing registration/freeze tests**

Test a package registering multiple categories, failure after an earlier staged
entry, undeclared/missing/category-mismatched registration, descriptor mismatch,
plugin priority spoof attempt, canonical winner plus partial alias loss, compile-
fail registration-after-freeze, and no prefix visibility on error.
Mutate every repeated descriptor field (package name/version/source API version,
host universe ID, artifact build ID) and prove comparison necessarily follows
native entry invocation, retains every returned handle, poisons the process on
mismatch, and precedes transaction commit, freeze, lock sealing, and effects.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-host --test registration && cargo test -p aiperf-runtime --test extensions_compile_time_extension`

Expected: new tests fail because manifest-bound freeze is absent.

- [ ] **Step 3: Adapt the existing clone-and-commit transaction**

Retain useful transactional staging, but source package/priority/provenance from
the active intended catalog. Observe actual registrations, compare exactly to
manifest and precomputed status, commit all declared entries together, freeze
once, and carry `ActivatingLibrarySet`. Do not re-resolve priority during entry;
Task 16 is the only success-sealing transition.

- [ ] **Step 4: Verify GREEN**

Run the same tests plus `cargo test -p aiperf-runtime --features engine`.
Expected: rollback and compile-fail behavior pass; every existing static test
uses an explicitly inventory-declared temporary static path rather than a hidden
production built-in path.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review registry ownership, clone cost, duplicate semantics, and frozen API;
commit as `feat(plugins): freeze transactional plugin universe`; bundle/import.

### Task 16: Derive and reproduce the complete canonical plugin lock

**Files:**
- Create: `rust/plugin-host/src/lock.rs`
- Create: `rust/plugin-host/src/bundle.rs`
- Create: `rust/plugin-host/src/diff.rs`
- Create: `rust/plugin-host/tests/lock.rs`
- Create: `rust/plugin-host/tests/bundle.rs`
- Create: `rust/plugin-host/tests/lock_mismatch.rs`
- Create: `rust/plugin-host/tests/lock_input.rs`
- Modify: `rust/plugin-host/src/lib.rs`

**Interfaces:**
- Produces `PluginLockV1`, `PluginLockDigest`, `LockedCatalogBundle`, canonical
  length-delimited encoding, atomic bundle publication, full recomputation, and
  first structured difference.
- Lock includes every raw/canonical presence tag/digest, decoded identity,
  closure/build IDs, baseline, authority/status, all failure receipts, actual
  descriptor digests, canonical/alias maps, required packages/keys,
  normalization/schema/system-policy versions; it excludes absolute paths and
  diagnostic prose.
- Consumes the Task-15 frozen universe and its activating resident handles,
  derives the canonical lock from actual descriptors/status, and atomically
  seals them as `LoadedLibrarySet { lock_digest, ... }`. Only this transition
  permits same-process reuse, and only for an identical requested digest.
- Defines `--plugin-lock <bundle>/plugin.lock` consumption: no-follow open of
lock and sibling `store/`, full rehash, continued baseline/preloaded-module
validation, conflict rejection for ordinary discovery inputs, and exact
status/absence reproduction inherited by re-exec and cells.

- [ ] **Step 1: Write failing lock-difference fixtures**

Construct pairs differing only in private dependency, shadow, quarantine,
absence marker, actual descriptor, normalization version, host record, baseline,
required authority, alias map, or system allowlist. Assert every pair has a
different digest and deterministic first difference. Cover complete bundle
round-trip and missing/extra/tampered object rejection.
Cover `--plugin-lock` sibling-store no-follow access, baseline verification,
conflicts with every ordinary discovery source, and exact statuses/absence
reproduction without manifest rediscovery.
Separate pre-activation input mismatches from post-entry actual-descriptor
mismatches: malformed/tampered bundle, baseline, closure, host, and intended-
catalog differences fail before loader entry; a plugin whose actual declaration
differs from its locked/manifest descriptor can only be detected after entry,
so it retains handles, poisons the process, and fails before registration
commit/seal/effects.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-host --test lock --test bundle --test lock_mismatch --test lock_input`

Expected: FAIL because lock/bundle modules are absent.

- [ ] **Step 3: Implement canonical lock and atomic bundle**

Encode all normative fields and sorted receipts, derive BLAKE3 digest, publish
`plugin.lock` plus complete store through a private sibling directory and atomic
rename to an absent same-filesystem output, and reproduce from only the bundle.
Malformed/unreadable inputs retain truthful presence tags without fabricated
digests.

- [ ] **Step 4: Verify GREEN**

Run all four test targets and compare bundle bytes from two independent recomputations.
Expected: byte-identical success; every pre-activation input mutation fails
before loader entry with the exact first difference, while every actual-
descriptor mutation follows the explicitly tested poison path and never seals.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review canonical encoding, publication durability/cleanup, absence semantics,
and error determinism; commit as `feat(plugins): lock complete plugin catalog`;
bundle/import.

### Task 17: Compose the frozen universe before effects and add plugin commands

**Files:**
- Modify: `rust/cli/Cargo.toml`
- Modify: `rust/runtime/Cargo.toml`
- Modify: `rust/runtime/src/engine/application.rs`
- Modify: `rust/runtime/src/engine/coordinator.rs`
- Modify: `rust/runtime/src/engine/execution_factories.rs`
- Modify: `rust/cli/src/main.rs`
- Modify: `rust/cli/src/dispatch.rs`
- Modify: `rust/cli/src/execute_mode.rs`
- Create: `rust/cli/src/plugins.rs`
- Create: `rust/cli/tests/plugin_effect_order.rs`
- Create: `rust/cli/tests/plugin_commands.rs`
- Create: `rust/cli/tests/plugin_lock_input.rs`
- Create: `rust/cli/tests/plugin_abort_contract.rs`
- Create: `rust/cli/tests/plugin_route_census.rs`
- Create: `rust/cli/tests/fixtures/plugin-route-census.json`
- Create: `docs/specs/plugin-abort-contract-approval.md`

**Interfaces:**
- Hard prerequisite: the exact Task-7 authoritative allocator-provider object
  and its four-target/paper-rig evidence are integrated. The composer refuses
  to build or run without the mandatory provider dependency and first-
  instruction baseline/origin verification; no candidate-only allocator path
  exists.
- `Application::new` consumes an already composed plugin-API-owned
  `FrozenPluginUniverse` and cannot call a registry factory. It retains each
  unmigrated production ID only as an inventory-declared static implementation
  inside this sole frozen-universe path. Static factories/bindings are removed
  only by Task 39; no hidden `Application::stock`, fresh registry, or fresh gRPC
  binding construction survives Task 17.
- Adds `aiperf plugins list`, `validate`, `inspect-build`, and
  `lock --output <new-directory>` and consuming `--plugin-lock
  <bundle>/plugin.lock` with the exact execution boundaries in the spec.
- Adds diagnostic-only `--plugins-manifest-only`, which may decode and inspect
  manifests but cannot activate a library, construct a runnable application, or
  execute a benchmark. Supplying it to any execution path is a pre-effect error.
- Owns an exhaustive parsed CLI-route census covering every public and internal
  route. Each route is classified exactly once as `no_discovery`,
  `manifest_only`, or `full_composition`; a new or renamed command without a
  census row fails the test. Unrelated commands are `no_discovery` unless they
  explicitly request the capability catalog.

- [ ] **Step 1: Write failing subprocess effect-order and command tests**

Instrument every host effect with an ordered ledger. Assert root help and
completion do not discover; `plugins list` decodes manifests but never calls an
entry; `validate` activates only winners/required packages; explicit shadowed
manifest becomes required; `lock` warns before trusted activation and publishes
only an absent outside-root same-filesystem directory. Assert config/profile/
eval compose and freeze before all named effects.
Assert `--plugins-manifest-only` works only for diagnostic list/validation
authority and is refused by every run-producing command before discovery or
runtime construction.
Use an abort-on-entry fixture to prove `inspect-build` parses records without
mapping/calling code. Parse the actual Clap command graph plus pre-Clap internal
routes, compare it to `plugin-route-census.json`, and run an effect-ledger
subprocess assertion for every row; every unrelated native command, not only
representative `analyze-trace`, graph-inspection, and synthesis routes, must
remain no-discovery unless it requests the capability catalog. Test
`--plugin-lock` no-follow bundle/store access, baseline checks, ordinary-input
conflicts, and missing/extra/tampered objects. Before adopting panic semantics,
the approval artifact/test must identify the public CLI/protocol contract and
approve abort or select an outer supervisor.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-cli --test plugin_effect_order --test plugin_commands --test plugin_lock_input --test plugin_abort_contract --test plugin_route_census`

Expected: FAIL because commands and pre-effect composition are absent.

- [ ] **Step 3: Replace fresh static composition**

At the first AIPerf instruction verify the allocator/baseline, then route
component-using commands through one host composition function implementing the
spec’s eleven lifecycle steps. Keep help/list-only paths separate. Pass the
frozen universe into `Coordinator`; retain static factories/bindings for
unmigrated inventory IDs. Preserve typed recoverable errors and adopt
process-abort semantics only after the approved public-contract test; no
`catch_unwind` may author a plugin panic envelope.

- [ ] **Step 4: Verify GREEN**

Run all five focused test targets, existing CLI help/completion/config/profile/
eval tests, and `cargo test -p aiperf-runtime --features engine` on paper-rig.
Expected: ledgers prove exact ordering/lock consumption and no unapproved static
removal; the approval artifact’s exact decision is asserted; only Task 39 makes
a dynamic package authoritative.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review startup authority, effect ordering, panic behavior, and every composition
call site; commit as `feat(plugins): compose frozen universe before effects`;
bundle/import.

### Task 18: Normalize open transport and exporter Config v2 forms

**Files:**
- Modify: `rust/runtime/src/config/model/transport.rs`
- Modify: `rust/runtime/src/config/model/export.rs`
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: `rust/runtime/src/engine/protocol_v2.rs`
- Modify: `rust/cli/src/yaml.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/flags.rs`
- Create: `rust/runtime/tests/plugin_protocol_projection.rs`
- Create: `rust/cli/tests/plugin_config_open_selection.rs`

**Interfaces:**
- Canonical transport is exactly `{id, config}` with required normalized ID and
  empty-object config default. Absent transport normalizes to HTTP. Legacy
  `{type, ...flat}` and open form are mutually exclusive. The compatibility
  decoder remains through the next major Config schema; a removal-record test
  refuses cleanup without an explicit approved migration record.
- Canonical exporters are an ordered list of `{id, config}`; Config v2 accepts
  legacy fixed export, open list, or neither, never both. Serialization emits
  open form/omits none, and exporter `config` defaults to `{}`. Duplicate
  resolved winner identity is rejected.

- [ ] **Step 1: Write failing schema/projection golden matrix**

Cover absent transport, all legacy transports, open registered third-party ID,
mixed `type`/`id`/`config`, unknown ID/field, legacy/open exporter equivalence,
neither exporters, both forms, deterministic legacy order and exact canonical
  IDs/order keys, aliases resolving same exporter, rejection of both `json` and
  `otlp` while naming `otel`, open serialization, CLI flags,
cell projection, and absence of `transport_typed`.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-cli --test plugin_config_open_selection && cargo test -p aiperf-runtime --test plugin_protocol_projection --features engine`

Expected: FAIL because current transport/export configuration remains closed.

- [ ] **Step 3: Implement one normalization/projection path**

Decode the strict compatibility unions, separate host-owned endpoint connection
fields from plugin raw config, and serialize protocol/open config without a
second typed copy or hard-coded name switch. Freeze canonical exporter mapping
and ordering exactly as the spec table (`genai_perf_v1` through `wandb`).
Task 19 alone resolves factories and constructs validation receipts/plans.

- [ ] **Step 4: Verify GREEN**

Run focused tests plus existing YAML/config/protocol/online/grpc/offline stdio
tests. Expected: all legacy public behavior remains except the explicitly
approved open serialization; mixed/unknown cases fail before preparation.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review wire compatibility, strict decoding, duplicate/alias logic, and removal
of closed switches; commit as `feat(plugins): open runtime component selection`;
bundle/import.

### Task 19: Validate run plans and host-owned exporter capture requirements

**Files:**
- Create: `rust/runtime/src/engine/validated_run_plan.rs`
- Create: `rust/runtime/src/export/capture.rs`
- Modify: `rust/runtime/src/engine/coordinator.rs`
- Modify: `rust/runtime/src/engine/records.rs`
- Modify: `rust/runtime/src/engine/execute/capture.rs`
- Modify: `rust/runtime/src/engine/execute/compose_sidecars.rs`
- Modify: `rust/runtime/src/engine/execute/entrypoints.rs`
- Modify: `rust/runtime/src/engine/execute/plan.rs`
- Modify: `rust/runtime/src/engine/online_execution.rs`
- Modify: `rust/runtime/src/metrics_core/mod.rs`
- Modify: `rust/runtime/src/metrics_core/report.rs`
- Modify: `rust/runtime/src/export/otel.rs`
- Delete: `rust/runtime/src/export/otel/accumulator.rs`
- Create: `rust/runtime/tests/plugin_capture_plan.rs`
- Create: `rust/runtime/tests/factory_validation_receipt.rs`

**Interfaces:**
- Produces immutable `ValidatedRunPlan` consuming Task 6’s
  `FactoryValidationReceiptV1`,
  canonical plan digest, opaque factory values, combined capture plan, and
  retention reason `RequiredByExporter(<canonical-id>)`.
- Implements Task 6’s `ExactRecordV1`, `GenAiClientHistogramsV1`, and
  `ExplicitHistogramV1` vocabulary and checked merge/ordering semantics without
  redefining their public types.
- Replaces all production OTel-specific capture state before Task 28:
  `OtelRecordAccumulator`, `NativeReport::otel_per_record`,
  `native_otel_enabled`, and `observe_otel_record` disappear from engine,
  worker, report, and exporter seams. OTel’s remaining static exporter consumes
  the same generic finalized folded projection later exposed to plugins.

- [ ] **Step 1: Write failing validation/receipt/capture tests**

Test deterministic semantic receipts, mutation/nonrepeatability, opaque-state
nontransfer, exact-record requirement forcing reason-tagged retention, default
exact-fold replacement, sketch/incompatible rejection, sorted union, canonical
scheduled/user/graph/cellular ordering, missing/duplicate keys, histogram
finite/bounds/bucket/count/overflow/key mismatch, and two OTLP configurations
sharing one host fold.
Pin each ordering tuple (including absent/duplicate `request_index` behavior),
warmup/profile scope, full projection field census, histogram names/dimensions/
bounds/count/sum/min/max/merge order, and diagnostics containing exporter ID,
requirement, conflict, and remediation.
Add source/type guards proving the engine, execution capture, online execution,
metrics report, and exporter tree contain no OTel-specific per-record callback,
retention boolean, report field, or accumulator module. The only permitted
production `otel` references after this task are static exporter
configuration/decoration/upload and registry identity awaiting Task 28/39.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-runtime --test plugin_capture_plan --test factory_validation_receipt --features engine`

Expected: FAIL because plan/capture types are absent.

- [ ] **Step 3: Implement validation before preparation**

Factory validation returns opaque native state plus host receipt. Hash canonical
run DTO plus sorted receipts; never serialize/downcast opaque state. Compute
capture/retention compatibility before runtime construction. Install generic
worker-local folds only when required; no plugin callback or plugin-specific
accumulator enters record/token paths.
Repoint the current static OTel exporter to that generic projection and delete
its private accumulator/report side channel in this task, while production
static exporter authority remains unchanged until Task 39.

- [ ] **Step 4: Verify GREEN and hot-path shape**

Run focused tests plus metrics exact/sketch/sharded suites and structural
instrumentation. Expected: plan digests reproduce, capture failures precede
effects, and disabled capture adds no calls/allocations/locks.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review hot-path allocation/locking, deterministic floating merge, overflow,
opaque ownership, and receipt trust; commit as
`feat(plugins): validate host capture plans`; bundle/import.

### Task 20: Reproduce lock and validated plans across ordinary re-exec

**Files:**
- Modify: `rust/cli/src/execute.rs`
- Modify: `rust/cli/src/execute_mode.rs`
- Create: `rust/cli/src/plugin_bootstrap.rs`
- Create: `rust/cli/tests/plugin_reexec.rs`
- Create: `rust/cli/tests/plugin_reexec_plan.rs`
- Create: `rust/cli/tests/fixtures/reexec_plugins/`

**Interfaces:**
- Produces a dedicated private bootstrap authority containing canonical lock DTO,
  expected plan digest, and handles to the complete acquired
  `LockedCatalogBundle`; benchmark stdin remains unchanged.
- The parent accepts inherited `--plugin-lock` only as the already no-follow
validated Task-16 bundle, passes no rediscovery authority, and rejects a child
lock/baseline/preloaded-module conflict before request stdin or effects.
- Unix uses explicit inherited no-follow descriptors; Windows uses only handles
  named in the child process attribute list with ambient inheritance disabled.

- [ ] **Step 1: Write failing child-authority tests**

Test same-lock success; mutable installed generation replacement; env/CLI
rediscovery attempt; missing/extra bundle object; every one-field lock mismatch;
semantic default mutation; nonrepeatable receipt; descriptor/handle leakage;
stdin byte identity; and child effect ledger. Count discovery/open calls.
Cover explicit lock-mode inheritance, conflicting CLI/environment discovery
inputs, no-follow baseline revalidation, and a child attempting a different
bundle/status receipt.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-cli --test plugin_reexec --test plugin_reexec_plan`

Expected: FAIL because private plugin bootstrap is absent.

- [ ] **Step 3: Implement inherited immutable authority**

Parent composes and validates before child launch, creates the platform-private
authority, restricts inheritance, and supplies lock/plan expectations. Child
rehashes only reachable bundle objects, reconstructs/fixes the same universe,
compares full lock, then reads request, reruns factory validation, compares plan,
and only then prepares effects.

- [ ] **Step 4: Verify GREEN**

Run focused tests on all four targets and existing execute/protocol stdio tests.
Expected: no mutable rediscovery/reopen; first structured mismatch precedes
child effects; request protocol remains byte-identical.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review descriptor/handle authority, inheritance, cleanup, ordering, and Windows
attribute lists; commit as `feat(plugins): reproduce universe across reexec`;
bundle/import.

### Task 21: Bind plugin universe and plans into cellular launch and attestation

**Files:**
- Modify: `rust/runtime/src/engine/cell_launcher.rs`
- Modify: `rust/runtime/src/engine/cellular_bootstrap.rs`
- Modify: `rust/runtime/src/engine/cellular_cell.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs`
- Modify: `rust/runtime/src/engine/cellular_registration.rs`
- Modify: `rust/cli/src/execute_mode.rs`
- Create: `rust/cli/tests/plugin_cellular.rs`
- Create: `rust/cli/tests/plugin_kube_slurm_projection.rs`

**Interfaces:**
- Extends `CellLaunchContext` with expected canonical lock, normalized full run
  DTO, deterministic partition, expected cell plan digest, and complete locked
  catalog inventory.
- `CellRegister` and its signed transcript bind lock and plan digests. Reply
  repeats prebootstrapped slice/digests for agreement, never first authority.
- A lock-mode cell receives only Task-16 validated bundle handles/paths and
rehashes them no-follow; it cannot substitute an installed generation or revive
ordinary discovery.

- [ ] **Step 1: Write failing same-/cross-host bootstrap tests**

Cover same-host pipe, fixed-0600 cross-host file, old schema refusal, K8s native
envelope/image-capabilities/JobSet/result provenance, SLURM run/generate, no argv/
environment secrets or plugin code transfer, every lock/plan/partition mismatch,
registration transaction rollback, Velo startup order, and cell effect ledger.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-cli --test plugin_cellular --test plugin_kube_slurm_projection --features cellular`

Expected: FAIL because cellular DTOs/transcripts do not bind plugin state.

- [ ] **Step 3: Implement pre-Velo composition and signed agreement**

Controller derives every slice/plan before launch. Cell reads bootstrap,
reproduces full lock, validates its local slice/receipts, and compares digests
before Tokio/Velo/dataset/artifact/barrier effects. Registration transactionally
verifies both digests before installing routes, artifact authority, or barriers.
Remote images preinstall exact inventory; no automatic library transfer.

- [ ] **Step 4: Verify GREEN**

Run focused local projections on paper-rig and real K8s/SLURM cross-host tests.
Expected: all roles report the same complete lock; every mismatch fails before
effects/routes; old plugin-incapable schemas fail closed.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review authentication transcript, pre-runtime ordering, no secret mixing,
bounded state, and transaction boundaries; commit as
`feat(plugins): attest cellular plugin universe`; bundle/import.

### Task 22: Add plugin lock and catalog provenance to all reports

**Files:**
- Modify: `rust/runtime/src/report.rs`
- Modify: `rust/runtime/src/engine/distribution_identity.rs`
- Modify: `rust/runtime/src/engine/coordinator.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs`
- Create: `rust/e2e-tests/tests/plugin_report_provenance.rs`

**Interfaces:**
- Preserves existing executable `distribution_id` semantics and adds separate
  required `plugin_lock_digest` plus detailed catalog artifact without absolute
  paths. Merged cellular report provenance is controller-authored and exact.

- [ ] **Step 1: Write failing report/artifact tests**

Execute single-process, re-exec, same-host cellular, and cross-host cellular
runs. Assert distribution ID is unchanged, plugin lock equals the actual frozen
universe, detailed catalog has all statuses/descriptors/receipts/provenance, no
absolute path/config secret appears, and cells cannot omit/change identity.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_report_provenance`

Expected: FAIL because reports lack plugin lock/catalog provenance.

- [ ] **Step 3: Add orthogonal report fields and catalog artifact**

Thread immutable lock identity from `Application`/controller into report commit.
Render canonical redacted catalog details and include Task 23 exporter outcome
metadata. Do not overload distribution identity or expose local discovery paths.

- [ ] **Step 4: Verify GREEN**

Run focused test plus existing report/JSON/cellular snapshot suites. Expected:
all process modes expose exact same lock and retain previous distribution ID.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review provenance ownership, redaction, serialization finiteness, and cellular
merge; commit as `feat(plugins): report frozen catalog provenance`; bundle/import.

### Task 23: Implement generic cellular capture transfer and exporter outcomes

**Files:**
- Create: `rust/runtime/src/cellular/capture.rs`
- Modify: `rust/runtime/src/cellular/protocol.rs`
- Modify: `rust/runtime/src/engine/cellular_cell.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs`
- Modify: `rust/runtime/src/export/mod.rs`
- Modify: `rust/runtime/src/engine/coordinator.rs`
- Create: `rust/e2e-tests/tests/plugin_cellular_capture.rs`
- Create: `rust/e2e-tests/tests/plugin_exporter_outcomes.rs`

**Interfaces:**
- Adds bounded `ExactRecordsPartitionV1` chunks and mandatory
  `CellCaptureBundleV1` presence-tagged folded projections bound to cell/plan.
- Exporter runner returns ordered structured outcomes; failures are persisted,
  remaining exporters continue deterministically, and committed report remains.

- [ ] **Step 1: Write failing protocol/outcome fixtures**

Cover exact-record chunk gaps, duplicates, digest/count/length mismatch,
unexpected cells, byte/record bounds, backpressure, canonical reordering,
compact-path zero transfer, capture bundle missing/empty/duplicate/injected/
schema/plan/payload mismatch in Records and Store modes, and a failing exporter
followed by successful exporter with terminal metadata.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_cellular_capture --test plugin_exporter_outcomes`

Expected: FAIL because generic transfer/outcomes are absent.

- [ ] **Step 3: Implement bounded post-run transfer and deterministic runner**

Cells emit exact chunks only when requested and always emit the complete folded
projection set, including expected empty values. Controller authenticates,
validates, reassembles, orders, and checked-merges before exporter preparation.
Run exporters in descriptor order with capability-limited artifacts, record each
outcome, and continue after typed exporter error.

- [ ] **Step 4: Verify GREEN and compact-path parity**

Run focused tests plus cellular Records/Store and exporter tests. Expected:
requested capture succeeds identically same/cross-host; unrequested exact
records transfer zero bytes; malformed payload fails before export.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review authenticated protocol bounds/backpressure, memory accounting,
deterministic merge, artifact authority, and error continuation; commit as
`feat(plugins): transfer generic cellular capture`; bundle/import.

### Task 24: Stage/test the basic exporter candidate package

**Precondition for Tasks 24–34:** Task 36 is `PASS`. These tasks build/test real
SDK artifacts using their final canonical IDs/descriptors/manifests only in an
explicit non-shipping candidate binary/catalog that excludes the corresponding
static registration. The separate monolithic baseline comparator retains the
static canonical ID. They publish self-contained manifest/inventory fragments,
do not centrally register, remove, install, or make authoritative a production
static ID, and Task 39 alone performs production cutover after all ten gates.

**Files:**
- Modify: `rust/plugins/export-basic/Cargo.toml`
- Modify: `rust/plugins/export-basic/src/lib.rs`
- Modify: `rust/plugins/export-basic/plugins.yaml.in`
- Create: candidate-only copied exporter sources under `rust/plugins/export-basic/src/`
- Create: `rust/plugins/export-basic/tests/factory.rs`
- Create: `rust/e2e-tests/tests/plugin_export_basic.rs`

**Interfaces:**
- Registers canonical IDs `genai_perf_v1`, `server_metrics`, `timeslice`,
  `accuracy_csv`, and `console_txt` with exact legacy configs/order keys and
  host capture requirements.
- One small basic-exporter artifact uses Task 9 SDK and Task 15 registrar; it
  has no telemetry backend dependency.

- [ ] **Step 1: Write failing separate-artifact parity tests**

Run the same normalized configs through current static baseline and an SDK-built
basic plugin. Compare complete artifacts/records/order, JSON/CSV toggles, phase
filtering, no-export omission, artifact path denial, override provenance, and a
late exporter error continuing to the next exporter. Assert dynamic plugin edit
does not rebuild/relink host.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_export_basic`

Expected: FAIL because the precreated shell has no SDK-built comparator artifact,
fragment, or dynamic provenance; static production behavior remains authoritative.

- [ ] **Step 3: Copy and adapt the private candidate behind `ExporterFactory`**

Reuse implementation logic without adapter callbacks; strictly validate raw
config at plan time, return canonical receipt/capture set, prepare post-report,
and write through `ArtifactAccess`. Emit a self-contained comparator manifest/
inventory fragment; do not alter production inventory or static registration.
Copy every Task-24 implementation leaf/golden named in
`candidate-source-inventory.toml`, preserve its BLAKE3 digest, and allow only the
candidate facade/registration files to differ. The test fails on a missing,
extra, or changed implementation leaf.

- [ ] **Step 4: Verify GREEN and diagnostic candidate parity**

Run focused/e2e/export suites, four-target real loading, structural hot-path/
build isolation, and the already-integrated Task 3 applicable paired gate.
Expected: behavior/artifacts equal, zero added request-path work, performance
and allocation gates pass.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review copied logic, path authority, clone/allocation use, structured errors,
and proof that production sources/registration are unchanged; commit as
`test(plugins): stage basic exporter candidate`; bundle/import only after the
candidate parity gate. Task 38 owns final exact-candidate D6 performance.

### Task 25: Stage/test the Parquet exporter candidate package

**Files:**
- Modify: `rust/plugins/export-parquet/Cargo.toml`
- Modify: `rust/plugins/export-parquet/src/lib.rs`
- Modify: `rust/plugins/export-parquet/plugins.yaml.in`
- Create: candidate-only copied Parquet sources under `rust/plugins/export-parquet/src/`
- Create: `rust/e2e-tests/tests/plugin_export_parquet.rs`

**Interfaces:**
- Registers only `server_metrics_parquet` with exact normalized
  `ParquetExport` config and `file+4` order.
- Parquet exporter dependency is distinct from independently gated host dataset
  reader capability; selecting one cannot link the other.

- [ ] **Step 1: Write failing dependency/artifact parity tests**

Assert minimal host and basic exporter graphs exclude Arrow/Parquet; load the
SDK-built Parquet plugin, compare schema/row/order/artifact bytes semantically,
exercise disabled/missing feature diagnostics, cellular capture, install/
uninstall, and rebuild isolation.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_export_parquet`

Expected: FAIL because the precreated shell lacks a conforming candidate artifact
and static production provenance remains authoritative.

- [ ] **Step 3: Copy the private Parquet candidate and isolate feature ownership**

The factory declares exact-record/report capture actually required, validates
before effects, and performs post-report writing. The already-owned Task-3
feature matrix keeps host dataset-reader and exporter plugin features
independent; this task changes only the precreated Parquet package manifest.
Copy every Task-25 implementation leaf/test named in the candidate inventory and
prove BLAKE3 equality. Emit the self-contained candidate fragment. Task 39 alone
removes the production static exporter path during its gated cutover.

- [ ] **Step 4: Verify GREEN and diagnostic candidate parity**

Run focused tests, existing Parquet/dataset suites, four-platform packaging,
structural/build isolation, and paired exporter gate. Expected: artifacts and
behavior match baseline; unrelated host/plugin does not link Parquet backend.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review memory/row batching, capture ownership, error propagation, and Cargo
graph; commit as `test(plugins): stage parquet exporter candidate`; bundle/import.

### Task 26: Stage/test MLflow exporter candidate package

**Files:**
- Modify: `rust/plugins/export-mlflow/Cargo.toml`
- Modify: `rust/plugins/export-mlflow/src/lib.rs`
- Modify: `rust/plugins/export-mlflow/plugins.yaml.in`
- Create: candidate-only copied MLflow sources under `rust/plugins/export-mlflow/src/`
- Create: `rust/e2e-tests/tests/plugin_export_mlflow.rs`

**Interfaces:**
- Registers only `mlflow`, exact normalized `MlflowExport`, order
  `uploader+1`, strict config/credential redaction, and declared capture.

- [ ] **Step 1: Write failing mock-receiver parity tests**

Against the in-repo MLflow receiver compare requests/artifacts/order/error
outcomes for static versus SDK-built dynamic implementation; cover credentials
redaction, server error continuation, scoped artifact reads, override, package
install/uninstall, and no MLflow dependency in host/basic/other telemetry crates.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_export_mlflow`

Expected: FAIL because the precreated shell lacks a conforming candidate artifact
and static production provenance remains authoritative.

- [ ] **Step 3: Copy the private candidate behind the post-report factory**

Strictly validate/normalize once, keep network work after report commit, use
capability-limited artifacts, and return typed outcomes. Copy the exact MLflow
implementation leaf named by the candidate inventory and prove BLAKE3 equality.
Do not edit production inventory, source, manifests, or static removal paths.

- [ ] **Step 4: Verify GREEN and diagnostic candidate parity**

Run focused and existing MLflow/mock-server suites, four-target loader/package,
dependency/build isolation, structural and paired exporter gates.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review HTTP lifecycle, credential logging, deterministic ordering, clones, and
error continuation; commit as `test(plugins): stage mlflow exporter candidate`;
bundle/import.

### Task 27: Stage/test W&B exporter candidate package

**Files:**
- Modify: `rust/plugins/export-wandb/Cargo.toml`
- Modify: `rust/plugins/export-wandb/src/lib.rs`
- Modify: `rust/plugins/export-wandb/plugins.yaml.in`
- Create: candidate-only copied W&B sources under `rust/plugins/export-wandb/src/`
- Create: `rust/e2e-tests/tests/plugin_export_wandb.rs`

**Interfaces:**
- Registers only `wandb`, exact normalized `WandbExport`, order
  `uploader+2`, canonical offline `.wandb` ownership, strict config, and declared
  capture.

- [ ] **Step 1: Write failing datastore/offline parity tests**

Use the mock W&B datastore and offline file. Compare canonical requests/file/
outcomes, cover credential redaction, sync failure continuation, artifact scope,
override, install/uninstall, and prove no W&B dependency in other artifacts.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_export_wandb`

Expected: FAIL because the precreated shell lacks a conforming candidate artifact
and static production provenance remains authoritative.

- [ ] **Step 3: Copy the private candidate behind the exporter factory**

Perform strict plan-time validation, preserve offline canonical artifact,
execute upload after local writers in descriptor order, and emit only its
candidate fragment. Copy every W&B implementation leaf/test named by the
candidate inventory and prove BLAKE3 equality; Task 39 alone removes production
static `wandb`.

- [ ] **Step 4: Verify GREEN and diagnostic candidate parity**

Run focused/existing W&B suites, loader/package matrix, dependency/build
isolation, structural and paired exporter gates.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review file/network ownership, credential safety, outcomes, clones, and
telemetry dependency isolation; commit as `test(plugins): stage wandb exporter candidate`;
bundle/import.

### Task 28: Stage/test OTel exporter candidate package

**Files:**
- Modify: `rust/plugins/export-otel/Cargo.toml`
- Modify: `rust/plugins/export-otel/src/lib.rs`
- Modify: `rust/plugins/export-otel/plugins.yaml.in`
- Create: candidate-only copied OTel decoration sources under `rust/plugins/export-otel/src/`
- Create: `rust/e2e-tests/tests/plugin_telemetry_capture.rs`

**Interfaces:**
- Registers canonical ID `otel` only, exact `OtelExport`, `uploader+0`, and
  requires `FoldedProjectionV1(GenAiClientHistogramsV1)`.
- Plugin decorates host-folded histograms with operation/provider/model config;
  host report contains no OTel-specific type and no per-record plugin callback.

- [ ] **Step 1: Write failing topology/parity tests**

Compare retain, exact-fold, sketch, sharded, same-/cross-host cellular output;
two OTel configs at fixture level must share one host fold and decorate
independently (generation-1 run duplicate rule is tested separately). Cover all
metrics/bounds/attributes/error/token dimensions, error continuation, canonical
ID rejection of `otlp`, source scan for accumulator/report leakage, and
telemetry-enabled hot-path instrumentation.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_telemetry_capture`

Expected: FAIL because no SDK-built dynamic OTel candidate/provenance exists;
Task 19 has already removed the implementation-specific accumulator and report
side channel while static production behavior remains authoritative.

- [ ] **Step 3: Copy only decoration/upload into the private candidate**

Require Task 19’s source guards to prove the production OTel-specific
accumulator/report side channel is already absent. Copy only the post-Task-19
OTel configuration/decoration/upload implementation leaves and tests named by
the candidate inventory, prove BLAKE3 equality, strictly validate config and
receipt, consume the generic immutable projection, decorate/export post-report,
emit a candidate fragment, and leave static `otel` authority unchanged until
Task 39.

- [ ] **Step 4: Verify GREEN and diagnostic candidate parity**

Run focused/existing OTel tests across all modes, real receiver, four targets,
dependency/build isolation, structural no-callback/no-allocation gate, and full
paired telemetry scenarios. Expected: exact capture parity and simultaneous
performance/allocation pass.

- [ ] **Step 5: Graham review, bundle, and integrate**

Apply strict hot-path/metrics/cellular review to generic capture plus plugin
diff, resolve every finding, commit as `test(plugins): stage otel exporter candidate`;
bundle/import.

### Task 29: Stage/test endpoint factory candidate packages

**Files:**
- Modify: `rust/plugins/endpoints/Cargo.toml`
- Modify: `rust/plugins/endpoints/src/lib.rs`
- Modify: `rust/plugins/endpoints/plugins.yaml.in`
- Create: candidate-only copied endpoint sources under `rust/plugins/endpoints/src/`
- Create: `rust/e2e-tests/tests/plugin_endpoints.rs`

**Interfaces:**
- Registers atomic endpoint capabilities with canonical IDs/aliases,
  `EndpointFactory`, optional companion bindings, and effective aliases.
- Preserves authored `endpoint`, mapped `endpoint_profiles`, and protocol
  injected profile IDs as three distinct shapes. Host separates connection/
  generic policy fields from strict plugin configuration.

- [ ] **Step 1: Write failing endpoint parity/override tests**

Exercise every endpoint family over HTTP unary/streaming, strict raw config,
profiles/default ID injection, unknown registered third-party ID, alias partial
ownership, hard-coded capability removal, worker-local preparation, split UTF-8
SSE bytes, observation/usage semantics, override provenance, package isolation,
and static production path searches.
Pin transparent `EndpointType` source spelling and serialization compatibility;
only the closed typed endpoint field schema is removed, never this public spelling.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_endpoints`

Expected: FAIL because endpoint implementations are statically registered.

- [ ] **Step 3: Copy measured endpoint package islands into the private candidate**

Use the checked feature/dependency matrix rather than one library per dialect.
Factories own dialect config/format/parse and prepare existing worker-local
`PreparedEndpoint`; host retains generic policies. Emit candidate inventory
fragments only. Copy every endpoint implementation leaf/test named by the
candidate inventory and prove BLAKE3 equality, excluding only the reviewed
candidate facade. Task 39 makes an endpoint ID exclusively dynamic.

- [ ] **Step 4: Verify GREEN and diagnostic endpoint-candidate parity**

Run focused/existing endpoint/HTTP/mock-server/config suites, four-platform
loader/package lifecycle, build isolation, structural call/allocation gate, and
paired formatting/streaming/end-to-end performance cases.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review streaming byte correctness, usage/observation, worker locality,
allocation, alias binding, and proof production remains unchanged; commit as
`test(plugins): stage endpoint factory candidate`; bundle/import.

### Task 30: Stage/test endpoint-owned gRPC binding factories with their winners

**Files:**
- Modify: `rust/plugins/endpoints/src/lib.rs`
- Create: `rust/plugins/endpoints/src/grpc/factory.rs`
- Create: `rust/plugins/endpoints/src/grpc/kserve.rs`
- Create: `rust/plugins/endpoints/src/grpc/riva.rs`
- Create: equality-hashed copies under `rust/plugins/endpoints/src/transport/grpc/`
  of `kserve_binding.rs`, `proto.rs`, `riva_binding.rs`, `riva_codec.rs`, and
  `riva_proto.rs`
- Create: `rust/plugins/endpoints/tests/grpc_binding.rs`
- Create: `rust/e2e-tests/tests/plugin_endpoint_grpc_override.rs`

**Interfaces:**
- Winning endpoint capability owns optional `GrpcEndpointBindingFactory` for
  unary/server-streaming/bidirectional codecs. The private candidate freezes one
  binding table alongside its endpoint winners; production gRPC execution and
  `GrpcBindingRegistry::builtin()` remain unchanged until Task 39a.

- [ ] **Step 1: Write failing third-party gRPC override tests**

Build a separate endpoint plugin overriding a first-party ID and returning a
distinguishable codec over gRPC unary, server streaming, and bidirectional
paths. Assert winning descriptor/binding provenance inside the private candidate
and assert production fresh/builtin static binding constructors remain present
and unchanged.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_endpoint_grpc_override --features grpc`

Expected: FAIL because the candidate endpoint package does not yet expose its
companion binding factories.

- [ ] **Step 3: Bind candidate codecs atomically to endpoint winners**

Use Task 15’s freeze contract in the private candidate harness so companion
factories freeze with the canonical endpoint capability. Alias winners cannot
independently select a codec. Do not edit any production runtime/CLI source,
registry, config, manifest, or direct-execution path.
Copy and BLAKE3-prove the five Task-30 endpoint-owned gRPC leaves from the
candidate inventory. Candidate-only `grpc/factory.rs`, `kserve.rs`, and
`riva.rs` adapt the Task-6 public contracts; the copied implementation leaves
remain unchanged.

- [ ] **Step 4: Verify GREEN and diagnostic binding-candidate parity**

Run focused plus existing KServe/Riva/grpc transport/mock-server suites and
four-target override fixtures. Expected: the candidate winner’s codec is chosen
in the candidate harness while ordinary production keeps its unchanged static
codec.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review codec ownership/lifetime, streaming cancellation, static search, and
endpoint/transport separation; commit as
`test(plugins): stage endpoint grpc binding candidate`; bundle/import.

### Task 31: Stage/test HTTP transport candidate package

**Files:**
- Modify: `rust/plugins/transport-http/Cargo.toml`
- Modify: `rust/plugins/transport-http/src/lib.rs`
- Modify: `rust/plugins/transport-http/plugins.yaml.in`
- Create: candidate-only copied HTTP sources under `rust/plugins/transport-http/src/`
- Create: `rust/e2e-tests/tests/plugin_transport_http.rs`

**Interfaces:**
- Registers canonical `http` with `RequestTransportExecution`; factory owns
  pools/dispatchers/request executors, host owns scheduling/admission/clock/
  observer/reduction/measurement/cancellation/phase orchestration.

- [ ] **Step 1: Write failing full HTTP parity tests**

Exercise HTTP/1, h2c, TLS verify on/off, UDS, SSE split bytes, SageMaker event
stream, CONNECT proxy/loopback exclusion, retries/clock backoff, scheduled/
graph/worker modes, cancellation/drain, strict legacy/open config, override,
runtime error no fallback, and structural hot-path instrumentation.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-e2e-tests --test plugin_transport_http`

Expected: FAIL because HTTP is still a static factory.

- [ ] **Step 3: Copy transport-specific implementation into the private candidate**

Prepare worker-local sink builders through the native factory without a request
adapter. Reuse existing reduction/measurement/observer seams directly. Copy
every HTTP implementation leaf/test/proxy/TLS helper named by the candidate
inventory and prove BLAKE3 equality, excluding only the reviewed candidate
facade. Emit the candidate HTTP manifest/fragment; Task 39 owns production
registration/closed-switch removal and Task 37 assembles the candidate
distribution.

- [ ] **Step 4: Verify GREEN and diagnostic HTTP-candidate parity**

Run all HTTP/runtime/CLI/mock-server suites, four-target loader/package,
build-isolation, IR/call-graph/allocation/lock instrumentation, and every
normative paired HTTP case. Expected: simultaneous 0.99 bounds and zero
allocation-count/bytes increase.

- [ ] **Step 5: Graham review, bundle, and integrate**

Apply maximum strictness to request/token/cancellation paths; resolve all
findings, commit as `test(plugins): stage http transport candidate`; bundle/import only
after the performance gate.

### Task 32: Stage/test gRPC transport independently of endpoint codecs

**Files:**
- Modify: `rust/plugins/transport-grpc/Cargo.toml`
- Modify: `rust/plugins/transport-grpc/src/lib.rs`
- Modify: `rust/plugins/transport-grpc/plugins.yaml.in`
- Create: candidate-only copied gRPC sources under `rust/plugins/transport-grpc/src/`
- Create: `rust/e2e-tests/tests/plugin_transport_grpc.rs`

**Interfaces:**
- Registers canonical `grpc` request transport; Tonic/channel/request execution
  remains transport-owned while selected endpoint codecs come only from Task 30
  frozen endpoint binding table.

- [ ] **Step 1: Write failing gRPC parity tests**

Exercise KServe OIP/Riva ASR/TTS/NLP unary/streaming/bidirectional, TLS,
connection retry/clock, cancellation/drain, workers/graph, endpoint override,
strict config, no fallback, observer/reduction/measurement parity, and hot-path
instrumentation.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_transport_grpc --features grpc`

Expected: FAIL because gRPC transport is static.

- [ ] **Step 3: Copy Tonic execution behind the private candidate factory**

Keep worker-local prepared endpoint/channel state, use injected endpoint binding
table, emit a candidate fragment, and leave production registration/closed enum
to Task 39. Copy every gRPC implementation leaf/test/proto asset named by the
candidate inventory and prove BLAKE3 equality, excluding only the reviewed
facade. Do not copy endpoint descriptor factories into the transport package;
generic Tonic framing/channel/raw-codec execution remains
transport-candidate-owned, while concrete KServe/Riva binding factories and
their generated endpoint proto/codec leaves remain Task-30 endpoint-owned.

- [ ] **Step 4: Verify GREEN and diagnostic gRPC-candidate parity**

Run all grpc/runtime/CLI/mock-server suites, four targets, build isolation,
structural hot-path and paired unary/streaming c=1/64 performance/allocation.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review Tonic streams, cancellation/backpressure, clocks, clones, worker locality,
binding separation, and proof production remains unchanged; commit as
`test(plugins): stage grpc transport candidate`; bundle/import.

### Task 33: Stage/test WebSocket and dry-run transport candidates

**Files:**
- Modify: `rust/plugins/transport-websocket/Cargo.toml`
- Modify: `rust/plugins/transport-websocket/src/lib.rs`
- Modify: `rust/plugins/transport-websocket/plugins.yaml.in`
- Modify: `rust/plugins/transport-dry-run/Cargo.toml`
- Modify: `rust/plugins/transport-dry-run/src/lib.rs`
- Modify: `rust/plugins/transport-dry-run/plugins.yaml.in`
- Create: candidate-only copied WebSocket sources under `rust/plugins/transport-websocket/src/`
- Create: candidate-only copied dry-run sources under `rust/plugins/transport-dry-run/src/`
- Create: `rust/e2e-tests/tests/plugin_transport_websocket.rs`
- Modify: `rust/dry-run-tests/tests/dry_run.rs`

**Interfaces:**
- Registers `websocket` and `dry_run` as independent request-transport
  capabilities; default distribution requires dry-run, full adds WebSocket.

- [ ] **Step 1: Write failing real-loader parity tests**

Cover WebSocket frame/chunk/cancellation/error/worker behavior and dry-run
socket-free deterministic behavior/config/provenance, third-party override,
missing package diagnostics, no fallback, packaging/uninstall, and build
isolation between the two packages.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_transport_websocket --features websocket && cargo test -p aiperf-dry-run-tests`

Expected: FAIL dynamic provenance/package assertions because both are static.

- [ ] **Step 3: Copy implementations into separate private candidates**

Use request execution contract and existing host scheduling/observation paths,
emit default/full candidate fragments, and leave each production static
registration/closed-switch commit to Task 39. Copy every WebSocket and dry-run
implementation leaf/test named by the post-Task-6 candidate inventory and prove
BLAKE3 equality, excluding only reviewed facades. WebSocket copies only
`transport/ws.rs` plus `transport/ws/**`, including the extracted SDK-facing
`sink.rs`; dry-run copies only extracted `transport/dry_run.rs`.
`engine/ws_execution.rs` and `engine/dry_run.rs` are host adapters and are
never candidate sources. Do not edit those adapters, `online_execution.rs`, or
any production registry/config in this candidate task.

- [ ] **Step 4: Verify GREEN and both diagnostic candidate parity gates**

Run all WS/dry-run/runtime/CLI suites, target package matrices, build isolation,
structural and applicable paired performance/allocation cases separately.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review each commit/diff independently; commit messages
`test(plugins): stage websocket transport candidate` and
`test(plugins): stage dry-run transport candidate`; create/import one bundle per commit.

### Task 34: Stage/test Dynosim offline and online direct-execution candidates

**Files:**
- Modify: `rust/plugins/transport-dynosim/Cargo.toml`
- Modify: `rust/plugins/transport-dynosim/src/lib.rs`
- Modify: `rust/plugins/transport-dynosim/plugins.yaml.in`
- Create: candidate-only copied Dynosim sources under `rust/plugins/transport-dynosim/src/`
- Create: `rust/e2e-tests/tests/plugin_transport_dynosim.rs`

**Interfaces:**
- Registers `dynosim_offline` and `dynosim_online` through
  `DirectTransportExecution`, owns Dynamo dependencies, and receives only narrow
  clock/graph/metrics/artifact/cancellation service traits.

- [ ] **Step 1: Write failing direct-binding tests**

Exercise offline SimClock socket-free and online wall-clock replay, graph/
metrics/artifacts/cancellation, strict config, override, typed error no fallback,
compile-fail `RunContext` reach-through, and production search for
`dynosim_or_unsupported!`/ID switches/static direct paths.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-e2e-tests --test plugin_transport_dynosim --features dynosim`

Expected: FAIL because direct execution remains statically selected.

- [ ] **Step 3: Copy Dynamo implementation behind private narrow services**

Package both IDs together by dependency coupling, keep host orchestration and
clocks authoritative, and adapt only the candidate facade. Copy the post-Task-6
SDK-facing `dynosim/direct.rs` and `endpoints/dynosim.rs` leaves/tests named by
the candidate inventory and prove BLAKE3 equality. `dynosim.rs` and
`engine/offline_execution.rs` are host orchestration/adapters and are never
candidate sources. Emit the candidate
full-distribution fragment and leave every production macro/switch/static path
byte-for-byte unchanged until Task 39a.

- [ ] **Step 4: Verify GREEN and two diagnostic candidate parity gates**

Run dynosim runtime/CLI/graph suites, four-target applicable build/package
matrix, build isolation from minimal/default host, structural and paired
offline/online performance/allocation gates.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review direct execution capabilities, clock fidelity, cancellation, Dynamo
dependency isolation, and proof production remains unchanged; commit as
`test(plugins): stage dynosim transport candidates`; bundle/import.

### Task 35: Implement discovery trust, authenticated inventory, and atomic installation

**Files:**
- Create: `rust/plugin-host/src/install.rs`
- Create: `rust/plugin-host/src/inventory.rs`
- Create: `rust/plugin-host/src/platform/acl_unix.rs`
- Create: `rust/plugin-host/src/platform/acl_windows.rs`
- Modify: `rust/plugin-host/src/platform/mod.rs`
- Modify: `rust/plugin-host/src/lib.rs`
- Create: `rust/plugin-host/tests/discovery_authority.rs`
- Create: `rust/plugin-host/tests/atomic_generations.rs`
- Modify: `rust/plugin-packaging-tests/Cargo.toml`
- Modify: `rust/plugin-packaging-tests/src/lib.rs`
- Create: `rust/plugin-packaging-tests/tests/distribution_lifecycle.rs`

**Interfaces:**
- Produces authenticated first-party distribution inventory containing canonical
  manifests, complete closures, universe/build IDs, required packages/keys, and
  authentication root without absolute paths.
- Produces atomic install/update/rollback/uninstall of immutable generations and
  handle-relative system/user/environment/explicit authority decisions.

- [ ] **Step 1: Write failing trust/lifecycle fixtures**

Cover POSIX owner/mode/NFS ACL and Windows owner/DACL/inherited write/delete/
replace grants; links/reparse points; owner/mode changes during acquisition;
uninterpretable ACL; privileged/elevated/service execution disabling user/env/
ordinary explicit roots and rejecting unsafe override; inventory signature/
generation tamper/rollback/mix; atomic install crash points, concurrent readers,
upgrade/rollback/uninstall/GC, and Windows deferred DLL deletion.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-host --test discovery_authority --test atomic_generations && cargo test -p aiperf-plugin-packaging-tests --test distribution_lifecycle`

Expected: FAIL because trust/install/inventory support is absent.

- [ ] **Step 3: Implement handle-relative authority and atomic publication**

Apply exact per-root ownership/ACL rules, fail closed on unknown ACL semantics,
bind inventory authentication into executable build record, stage/verify complete
generations off-path, publish manifests atomically, publish absence before GC,
and never download/resolve repositories/modify installs from the loader.

- [ ] **Step 4: Verify GREEN on all four platforms**

Run ordinary and privileged fixture matrices plus concurrent lifecycle loops.
Expected: authorized roots reproduce one lock; unauthorized mutation never
loads; readers observe old or new complete generation only.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review ACL authority, TOCTOU, atomicity/durability, rollback/revocation, and
cleanup; commit as `feat(plugins): secure immutable plugin installation`;
bundle/import.

### Task 36: Gate complete four-target native-boundary and allocator feasibility

**Files:**
- Modify: `rust/plugin-conformance/Cargo.toml`
- Modify: `rust/plugin-conformance/src/lib.rs`
- Create: `rust/plugin-conformance/tests/native_boundary.rs`
- Create: `rust/plugin-conformance/tests/compatibility_closure.rs`
- Create: `rust/plugin-conformance/tests/residency_and_poison.rs`
- Create: `rust/plugin-conformance/tests/package_transaction.rs`
- Create: `rust/plugin-conformance/tests/priority_alias_matrix.rs`
- Create: `rust/plugin-conformance/tests/allocator_ownership.rs`
- Create: `rust/plugin-conformance/tests/abort_boundary.rs`
- Create: `rust/plugin-conformance/tests/no_promotion.rs`
- Create: `rust/plugin-conformance/fixtures/`
- Modify: `rust/tests/plugin-third-party/Cargo.toml`
- Modify: `rust/tests/plugin-third-party/src/lib.rs`
- Create: `rust/plugin-conformance/tests/diagnostics.rs`
- Create: `rust/plugin-conformance/tests/initialization_shutdown.rs`
- Create: `rust/plugin-conformance/tests/platform_loader_policy.rs`
- Create: `artifacts/native-plugin-feasibility/README.md`

**Interfaces:**
- Produces separately SDK-built artifacts and a real system-loader harness. A
  direct in-process extension construction can supplement registry tests but
  never satisfies loader/linkage/ABI/allocator/residency evidence.
- This is the complete four-target production-shaped native-boundary and full
allocator normative architecture gate: exact `cdylib` entry/import/export,
closure, panic-abort, every owned-container crossing in both directions,
residency, provider initialization/origin/interposition, and allocator
non-inferiority re-verify the exact Task-7 authoritative host/provider artifacts.
Task 7 already gates and installs the provider before any Task-17 native
activation; Task 36 detects drift and proves the full category boundary over
that integrated topology. It includes only
representative early category hot-path rejection, not Task 38’s complete final
per-component/full-distribution performance proof. PASS is required before any
non-shipping first-party package staging in Tasks 24–34. Task-7 conformance
permits only the ownership-table types used by infrastructure/third-party
fixtures before then; no first-party migration begins before Task 36 PASS.

- [ ] **Step 1: Create fixture-inventory validation test first**

The test requires every fixture to declare manifest/final digests, closure and
module identities, entry-call counter, descriptors, target allowlist, and
expected process outcome. It enumerates these exact groups:

```text
endpoint-only; transport-plus-endpoint; multiple-exporters;
winning-and-shadowed; late-registration-failure; descriptor-disagreement;
stale-universe; tampered-build-record; equal-priority; alias-matrix;
fully-shadowed-optional; fully-shadowed-required; load-failing-winner;
runtime-error-winner; acquisition-races; dependency-collisions;
generation-lifecycle; lock-sensitivity; allocator-containers;
panic-and-residency; config-projection; endpoint-grpc-override;
dynosim-direct; generic-otel; exact-cellular-records;
cellular-folded-bundle; capture-planning; exporter-post-report;
effects-and-provenance; discovery-trust
```

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-conformance`

Expected: FAIL listing every missing fixture group.

- [ ] **Step 3: Implement all required spec fixtures**

Use Task 9 build command for each independent artifact. Implement every listed
fixture and failure-policy row needed before staging, with exact named cases:
common-closure private/generated/reclassification/revocation; full descriptor
field mutation; mapped-exactly-once initializer counter; post-load metadata is
diagnostic only; system-library identities; required diagnostic category/name/
package/version/source-API/first-difference/remediation fields plus normal/debug
redaction; initializer/entry/registration benchmark/network/thread prohibition
and shutdown join/drop; platform negative scans for `LD_LIBRARY_PATH`,
`DYLD_LIBRARY_PATH`, `AddDllDirectory`, and `LOAD_LIBRARY_SEARCH_USER_DIRS`; and
all exact allocator symbol/version/pointer/overflow checks. Entry counters and
abort-on-call make pre-load rejection observable. Packaging and static-removal
fixtures are deliberately final Task-40 evidence, not this pre-staging gate.

- [ ] **Step 4: Verify GREEN on four-target matrix**

Run the complete conformance crate on Linux x86_64, macOS ARM64, Windows x86_64,
and Windows ARM64. Expected: all fixture groups execute their real artifacts,
every negative case returns the exact stable code or abnormal process result,
and fixture inventory has no missing proof point.
On the otherwise-idle paper-rig, run the Task-3 paired runner against the exact
Task-7 authoritative host/provider plus conformance plugins for the four mandatory early cases
`allocator_owned_values`, `endpoint_factory_dispatch`,
`transport_factory_dispatch`, and `exporter_capture_projection`, with five
warmups, exactly 30 retained AB/BA pairs, static median at least 30 seconds,
simultaneous one-sided 95% lower bounds `>= 0.99`, primary CV `<= 2%`, and zero
allocation-count/byte increase. The allocator case crosses and drops every
ownership-table family both directions and includes startup, steady-state,
ordinary/aligned reallocation, and process teardown. Retain raw samples,
structural import/origin maps, experiment identity, and report digests under
`artifacts/native-plugin-feasibility/`. This is the complete allocator
architecture/non-inferiority gate and an early representative category
hot-path rejection; Task 38 remains the complete exact-final-candidate gate for
every first-party component and the joint distribution.

Run:

```bash
cargo run -p aiperf-bench-tools --release --bin plugin_runtime_bench -- \
  --inventory benchmarks/plugin-parity.yaml \
  --candidate plugin-conformance/fixtures/allocator-candidate-host/candidate.yaml \
  --cases allocator_owned_values,endpoint_factory_dispatch,transport_factory_dispatch,exporter_capture_projection \
  --pairs 30 --warmups 5 --bootstrap-resamples 100000 \
  --output ../artifacts/native-plugin-feasibility
```

Expected: the command exits zero only when every structural, CV, allocation,
and simultaneous non-inferiority condition above passes; it writes the retained
identity and raw/evaluated evidence digests consumed by the tracker.

- [ ] **Step 5: Graham review, bundle, and integrate**

Review harness truthfulness, subprocess/loader evidence, race synchronization,
test economy, and negative-control strength; commit as
`test(plugins): enforce complete native conformance`; bundle/import.

### Task 37: Assemble default/full distributions and four-platform CI

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Modify: `Makefile`
- Modify: `Dockerfile`
- Modify: `tools/wheel_repack.py`
- Modify: `pyproject.toml`
- Create: `rust/scripts/assemble-plugin-distribution.rs`
- Modify: `rust/plugin-static-comparator/Cargo.toml`
- Modify: `rust/plugin-static-comparator/src/lib.rs`
- Create: `rust/plugin-static-comparator/src/main.rs`
- Create: `rust/plugin-static-comparator/src/static_inventory.rs`
- Create: `rust/plugin-static-comparator/tests/comparator_census.rs`
- Create: `rust/plugin-packaging-tests/tests/distribution_census.rs`
- Create: `rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml`
- Create: `.github/workflows/rust-native-plugins.yml`
- Modify: `.github/workflows/nightly.yml`
- Modify: `.github/workflows/rust-e2e-tests.yml`

**Interfaces:**
- Phase `37a-tooling` produces and integrates one candidate-only assembler plus
  tests/CI using the checked synthetic candidate fixture; it is an ancestor of
  every 39a commit. Phase `37b-package`, after the final 39a object, uses that
  unchanged assembler to produce one **private non-shipping release-candidate**
  from the exact Task-39a host/plugin worktree: host, allocator provider, SDK,
  immutable first-party generations/manifests, authenticated inventory, lock
  assets, license/SBOM, and RECORD digests. It never replaces installed/default
production inventory; candidate mode requires explicit test authority and emits
the exact bytes Task 39b may atomically publish without rebuild/relink.
- The same 37a ancestor produces a test-only
  `aiperf-plugin-static-comparator` binary target. It links the exact candidate
  endpoint/transport/exporter implementation crates directly, constructs the
  same frozen descriptors/configuration under a comparator-only static registrar,
  retains fat LTO and the pre-provider static-mimalloc baseline topology, and is
  excluded from every wheel/container/native/Kubernetes product inventory.
  Its checked census must equal the dynamic default/full component census and
  reject any separate implementation source, feature, profile, config default,
  or component omission.
- Default requires HTTP/gRPC/dry-run transports, shipped endpoints, and basic
  exporters. Full adds WebSocket, both Dynosim IDs, Parquet, OTel, MLflow, W&B.

- [ ] **Step 1: Write failing package census/install tests**

For editable/native, wheel, container, and Kubernetes-image default/full private
candidate installs on all four
targets assert exact package census, native wheel tags from every artifact,
modes/RECORD/license/SBOM/inventory digests, discovery/load, upgrade/rollback/
uninstall, missing package diagnostics, and no cross-linked telemetry backend.
Also build ordinary installed/default packages without candidate authority and
assert their inventory, artifact, RECORD, and entrypoint digests remain equal to
the pre-Task-37 production baseline.
Build the static comparator from the synthetic fixture and prove it contains
only statically linked candidate implementation crates, static registration,
fat LTO, and static mimalloc; it must not discover/load a plugin library.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-packaging-tests --test distribution_census`

Expected: FAIL because the new candidate census is absent and current packaging
has no complete candidate closure.

- [ ] **Step 3: Implement distribution-aware assembly and CI jobs**

Add explicit `PLUGIN_RELEASE_CANDIDATE=1` assembly targets for wheel, container,
native, and Kubernetes-image consumers. The exact Make targets are
`native-cli-candidate`, `bundle-cli-candidate`, and `wheel-candidate`; container
and Kubernetes-image jobs invoke the same candidate assembler directly.
Add `native-cli-static-comparator`, which requires
`AIPERF_STATIC_COMPARATOR_OUTPUT`, consumes the same source worktree and
Cargo.lock as the candidate targets, preserves the optimized/fat-LTO profile,
and writes no product inventory.
Ordinary product targets must refuse a
candidate root and remain byte-for-byte unchanged. Generate the authenticated
candidate inventory without regenerating Task-3 feature ownership. Add four jobs that build
real candidate artifacts, inspect arch/import/export/closure, run full conformance/product
integration, install/update/uninstall default/full packages, and upload complete
evidence keyed to commit.

- [ ] **Step 4: Verify GREEN for 37a tooling against the synthetic fixture**

Run ordinary `make native-cli`, `make bundle-cli CLI_FEATURES="--features full"`,
and `make wheel` for the unchanged-production negative proof; then run their
explicit `PLUGIN_RELEASE_CANDIDATE=1` counterparts against
`plugin-packaging-tests/fixtures/candidate-generation/fixture.toml`, package
tests, and the GitHub Actions fixture matrix. Expected: all four platform jobs
and clean-environment synthetic candidate installs pass while ordinary
inventory stays unchanged.
Also run `native-cli-static-comparator` against the synthetic fixture and the
comparator census/behavior suite; its source-tree/Cargo.lock/implementation-leaf
digests must equal the candidate assembly inputs.

- [ ] **Step 5: Two-pass Graham review and bundle candidate-mode tooling only**

Review assembler failure propagation, feature graph, artifact permissions/
digests/tags, CI truthfulness, and shell pipelines; commit as
`build(plugins): package native plugin distributions`; bundle/import the tooling
commit and integrate only its explicit candidate-mode assembler/tests/CI. Keep
all 39a source objects and assembled native artifacts unintegrated and
unpublished until 39b.

- [ ] **Step 6: Run 37b exact private packaging after the final 39a commit**

Without changing assembler, manifests, packaging code, or the private 39a source
worktree, run:

```bash
PLUGIN_RELEASE_CANDIDATE=1 \
AIPERF_CANDIDATE_WORKTREE=/work-pvc/paper-rig/aiperf-native-plugin-worktrees/unit-39a-dynosim-online \
AIPERF_CANDIDATE_OUTPUT=/cargo-target/plugin-release-candidate \
AIPERF_STATIC_COMPARATOR_OUTPUT=/cargo-target/plugin-static-baseline \
make -C .. native-cli-candidate bundle-cli-candidate wheel-candidate native-cli-static-comparator
```

Run the native/wheel/container/Kubernetes default/full lifecycle matrix on all
four targets and retain platform artifact/inventory/RECORD/signature digests.
Retain the comparator binary, symbol/import map, static-registration census,
source/Cargo.lock/profile/feature/implementation digests, and behavior parity
for the same final 39a worktree. Task 38 refuses either root unless these
identities prove same-revision/same-implementation inputs.
`37b-package` creates no Git commit and integrates nothing; any tool/source
change returns to 37a review, rebuilds the 39a chain from the new ancestor, and
reruns 37b.

### Task 38: Prove zero-loss hot-path and measured equivalence on paper-rig

**Files:**
- Modify: `rust/plugin-perf/Cargo.toml`
- Modify: `rust/plugin-perf/src/lib.rs`
- Create: `rust/plugin-perf/src/bin/parity.rs`
- Create: `rust/plugin-perf/tests/hot_path_shape.rs`
- Create: `rust/plugin-perf/tests/statistics.rs`
- Create: `rust/plugin-perf/tests/comparator_identity.rs`
- Create: `artifacts/native-plugin-parity/README.md`

**Interfaces:**
- Phase `38a-harness` implements, fixture-tests, reviews, bundles, and integrates
  the performance harness before 39a without changing Task-1’s immutable
  `plugin-parity.yaml`. Phase `38b-benchmark`, after 37b, consumes the test-only
  monolithic static comparator and exact Task-39a private release candidate from
  identical source/compiler/target/optimized profile/
affinity/dependencies; the candidate already has final canonical plugin IDs and
the prepared static removals. No pre-removal approximation can satisfy this gate.
  comparator retains static registration, fat LTO, and static mimalloc and never
  ships.
- The comparator is the Task-37 `aiperf-plugin-static-comparator` target, built
  by 37b into `/cargo-target/plugin-static-baseline` from the exact final 39a
  worktree and Cargo.lock. It statically links the same candidate implementation
  crates and checked component census used by the dynamic assembly; Task 38
  rejects historical Task-1 binaries, a separate source tree, a different
  feature/profile/LTO setting, or an unproved implementation digest.
- Produces immutable experiment identity, append-only attempts, raw paired
  samples, structural artifacts, simultaneous bootstrap report, CV/allocation/
  build-isolation results.

- [ ] **Step 1: Write failing statistical/state/structural tests**

Golden-test experiment identity mutation for every required fact, balanced AB/
BA schedule, five warmups, exactly 30 retained pairs, static median >=30s,
product-error immediate failure, fixed blinded invalidation/max-five/max-three
attempt rules, Hyndman-Fan type 7, deterministic >=100,000-resample paired
maximum-degradation bootstrap, simultaneous one-sided 95% bounds, primary-
metric CV <=2%, allocation no-increase, and forbidden hot-path edge detection.
Pin all inventory/identity fields: canonical inventory/harness/mock digests,
firmware, memory topology, every admitted environment value, each legal metric
and ratio direction, TTFT/ITL p50/p90/p99, and three separate CV vectors. Pin
same-member-order replacement only for invalid AB/BA pairs, both retained raw
members plus reason, and refusal to rerun valid failures.
Reject a comparator whose source-tree, Cargo.lock, implementation-leaf census,
config-default digest, feature set, optimized profile/fat-LTO mode, static
registration proof, or static-mimalloc import map differs from the dynamic
candidate’s bound Task-37 evidence.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-perf`

Expected: FAIL because the performance harness is absent.

- [ ] **Step 3: Implement exact protocol without weakening baseline**

Record complete source/lock/toolchain/profile/artifact/provider, candidate commit
object IDs, manifest descriptors, package IDs, feature/link/signing topology, and paper-rig
hardware/kernel/governor/affinity/isolation/mock placement/environment identity.
Instrument allocation count/bytes, IR/call graph, locks/spawns/channels/copies/
serialization/dispatch. The frozen case matrix contains, and the runner refuses
to start unless it contains, every applicable case for every exact final
candidate component:

- every Task-29 canonical endpoint family runs HTTP nonstreaming and streaming
  at concurrency 1 and 64 plus four-worker formatting/reduction; KServe/Riva
  families additionally run gRPC unary/streaming/bidirectional cases;
- `http` runs HTTP/1, h2c, TLS, UDS, SSE with 32 chunks, SageMaker event stream,
  proxy, retries, graph, concurrency 1/64, and four-worker cases;
- `grpc` runs KServe and every Riva unary/streaming/bidirectional family, TLS,
  retry, graph, concurrency 1/64, and four-worker cases;
- `websocket`, `dry_run`, `dynosim_offline`, and `dynosim_online` each run their
  Task-33/34 deterministic single-worker and multi-worker applicable cases;
- `genai_perf_v1`, `server_metrics`, `timeslice`, `accuracy_csv`,
  `server_metrics_parquet`, `console_txt`, `otel`, `mlflow`, and `wandb` each
  run an exporter case with `corpus_records: 100000`,
  `sample_repetitions: 16`, `processed_records: 1600000`, and
  `retained_artifact_records: 100000`; every retained member performs 16
  sequential exact passes, emits exactly 100,000 records with identical output
  digest per pass, retains one 100,000-record output artifact plus the 16
  receipts, sums active pass duration only (>=30 seconds), and divides exporter
  nanoseconds by `processed_records`; OTel additionally runs exact/folded/sketch,
  single-worker/four-worker, same-host/cross-host cellular, and telemetry off/on;
- allocator startup/steady-state/teardown and endpoint dispatch, transport
  dispatch, response reduction, capture fold, and exporter write microbenchmarks
  run for every relevant component; and
- minimal, default, full, and one all-components joint distribution each receive
  a simultaneous component-wide/full-distribution gate.

Every applicable case has exactly 30 retained pairs and participates in one
simultaneous maximum-degradation bootstrap for its component; the full
distribution gate jointly includes the union. Earlier Task-24–34 diagnostic
candidate results are never substituted for these exact Task-39a/Task-37 bytes.

- [ ] **Step 4: Verify 38a harness GREEN with deterministic fixture vectors**

Run `./scripts/run-plugin-task-gates.sh 38` twice against the checked synthetic
sample vectors. Expected: both reports are byte-identical; all negative vectors
fail for their pinned reason; no authoritative performance claim is made.

- [ ] **Step 5: Two-pass Graham review, bundle, and integrate 38a harness**

Review comparator equivalence, measurement state machine, statistics,
instrumentation, resource isolation, and evidence completeness; commit as
`bench(plugins): add exact final-candidate parity gate`, perform both mandatory
Graham passes, bundle/import, and integrate the harness so it is an ancestor of
the private 39a chain.

- [ ] **Step 6: Run 38b authoritative paper-rig A/B gate after 37b**

Run only on an otherwise-idle paper-rig with pinned disjoint CPU sets and fixed
frequency/governor. Pass only when every simultaneous ratio lower bound is
`>=0.99`, every relevant CV is `<=2%`, allocation count/bytes do not increase,
structural inspection shows no added hot-path operation, and build-isolation
proves editing one plugin does not rebuild/relink host/unrelated plugins.
Retain every sample/attempt/invalidation and exact experiment digest. Any byte,
manifest, topology, signing, or source change after this gate requires fresh
candidate packaging, conformance, and performance evidence.

Run:

```bash
cargo run -p aiperf-plugin-perf --release --bin parity -- \
  --inventory benchmarks/plugin-parity.yaml \
  --candidate-root /cargo-target/plugin-release-candidate \
  --baseline-root /cargo-target/plugin-static-baseline \
  --pairs 30 --warmups 5 --bootstrap-resamples 100000 \
  --output ../artifacts/native-plugin-parity
```

Expected: the runner executes the complete frozen case matrix above and exits
zero only for one valid simultaneous component/full-distribution PASS. The
tracker records its experiment identity and evidence-tree digest.
`38b-benchmark` changes no source, manifest, inventory, signing input, or
artifact; it creates no Git commit and integrates nothing. Any harness change
returns to 38a review, rebuilds the private 39a chain from the new ancestor, and
reruns 37b and 38b.

### Task 39: Remove every remaining static fallback after per-component gates

**Files:**
- Modify: `rust/runtime/src/extensions/mod.rs`
- Modify: `rust/runtime/src/engine/application.rs`
- Modify: `rust/runtime/src/engine/registry.rs`
- Modify: `rust/runtime/Cargo.toml`
- Modify: `rust/runtime/src/lib.rs`
- Modify: `rust/runtime/src/endpoints/registry.rs`
- Modify: `rust/runtime/src/endpoints/config.rs`
- Modify: `rust/runtime/src/export/mod.rs`
- Modify: `rust/runtime/src/config/model/transport.rs`
- Modify: `rust/runtime/src/config/validate.rs`
- Modify: `rust/runtime/src/engine/grpc_execution.rs`
- Modify: `rust/runtime/src/engine/offline_execution.rs`
- Modify: `rust/runtime/src/engine/grpc_turn_execution.rs`
- Modify: `rust/runtime/src/transport/grpc/binding.rs`
- Modify: `rust/runtime/src/engine/online_execution.rs`
- Modify: `rust/runtime/src/engine/dry_run.rs`
- Modify: `rust/runtime/src/transport/grpc/riva_binding.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs`
- Modify: `rust/cli/Cargo.toml`
- Modify: `rust/cli/src/main.rs`
- Modify: `rust/cli/src/execute_mode.rs`
- Modify: `rust/cli/src/eval/native_graph.rs`
- Modify: `rust/cli/src/yaml.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/control_hooks.rs`
- Modify: `rust/runtime/build.rs`
- Delete: every production row classified `implementation_leaf` or `asset` in
  `rust/plugin-conformance/candidate-source-inventory.toml`
- Modify: each production parent `mod.rs`, manifest, generated schema/config,
  and import site named by the Task-39a static-path manifest so deleted leaves
  are unreachable and only Task-4/6/19 host-owned contracts/capture remain
- Create: `rust/plugin-conformance/tests/no_static_paths.rs`
- Create: `rust/plugin-conformance/static-path-allowlist.toml`

**Interfaces:**
- Removes production built-in registry/factory, closed ID enum/name switches,
  static gRPC binding table, OTel accumulator/report type, direct-execution
  macro/switch, and each migrated implementation dependency from host.
- A test-only comparator may retain static implementations under distinct IDs;
  it is excluded from product packages and production search scope.
- **39a Prepare (not integrated):** in an isolated candidate worktree author one
exact removal/cutover commit per component and all no-static search fixtures;
record immutable object IDs. **39b Integrate/publish:** only after Task 37 packages
and Task 38 gates those exact candidate bytes, import those exact object IDs and
atomically switch production authority without rebuild, relink, descriptor,
  manifest, topology, or feature change.
- Task 39a’s checked static-path manifest is exhaustive over the reproducible
  searches in the Candidate Source Inventory and Static Path Inventory
  appendices. A new hit or an unlisted modified/deleted production file is a
  hard failure requiring plan/tracker amendment and fresh review before 39a.

- [ ] **Step 1: Write failing production-path search before each removal**

For each canonical first-party ID search Rust sources, Cargo graphs/features,
generated bindings/config schema, production binary symbols/imports, package
inventory, and direct execution. The test fails with exact remaining paths and
accepts only reviewed test comparator allowlist entries.
In a 39a worktree, required `AIPERF_STATIC_PATH_COMPONENT` selects exactly one
of the twelve ordered component IDs in the Implementation-unit Gate Matrix;
the test requires that component and every earlier component to have no
production path while requiring every later component to remain unchanged. An
unknown/missing component value is a test error. In 39b/final mode the variable
is absent and every generation-1 component must have no production static path.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p aiperf-plugin-conformance --test no_static_paths`

Expected: FAIL listing remaining static production paths.

- [ ] **Step 3 (39a): Prepare unintegrated exact cutover commits**

Author unintegrated candidate commits, record commit/source/Cargo.lock/artifact-
input IDs, complete Graham review and bundle each, but do not integrate or
publish. This is the sole explicit candidate-authoring exception to the literal
pre-removal performance/package requirement: before authoring each 39a commit,
all non-final SDK, four-target manifest/registration, behavior/artifact,
parent/child/cell lock, diagnostic candidate performance, build isolation,
diagnostics, ownership/import-topology, and source-search rows must be `PASS`;
Task-37 exact packaging and Task-38 exact-final performance remain `PENDING` by
construction. Remove only that component in the isolated candidate chain and
rerun affected non-final gates. Never integrate, install, publish, or promote a
39a object, and never promote static/shadowed implementation on dynamic failure.
The authoritative allocator/product composition is already integrated and
four-target/paper-rig-gated by Task 7; Task 39 must not rewrite it. The private
static-fallback removal chain order is basic exporters, Parquet, MLflow, W&B,
OTel, endpoints plus companion gRPC bindings, HTTP, gRPC,
WebSocket, dry-run, Dynosim offline, then Dynosim online. Each commit’s parent is
the preceding reviewed 39a object; Task 37 consumes the final composite tip and
the ledger retains every intermediate object ID.

- [ ] **Step 4: Verify GREEN after final removal**

Run no-static-path tests, minimal/default/full Cargo graph checks, full workspace/
all-features tests, every real-loader conformance/product/package job, and
paper-rig parity evidence digest verification. Expected: only host-owned static
categories remain; every generation-1 production ID comes from frozen plugins.

- [ ] **Step 5 (39b): Fast-forward exact gated objects and publish**

Verify imported object IDs and every platform artifact/inventory/RECORD/signing
digest equal Task 37/38 evidence, then atomically publish retained objects with
zero compile/link/sign/manifest regeneration. Any mismatch reruns Task 37,
affected conformance, and Task 38; only then update the tracker. Before each 39b
authority switch, all ten removal rows—including exact Task-37 package lifecycle
and exact Task-38 component/full-distribution performance—must be `PASS`. The
published source tree, Cargo.lock, native artifacts, manifests, inventory,
RECORD, signatures, and experiment-linked digests must be byte/object identical
to the release-candidate ledger; post-gate generation of any byte is forbidden.
The ten rows are the ten numbered predicates in the specification and are
recorded separately for every component in the execution tracker: independent
SDK build; four-platform manifest/registration conformance; behavior/artifact
parity; parent/child/cellular lock agreement; normative performance; build/
relink isolation; four-platform atomic package install/removal; missing/
incompatible/override diagnostics; allocator/compiled-crate/panic/native-
dependency import topology plus full ownership conformance; and exhaustive
production no-static-path search. A component cannot borrow a PASS from the
composite row or another component.

### Task 40: Complete documentation, whole-branch review, and release audit

**Files:**
- Create: `docs/plugins/native-author-guide.md`
- Create: `docs/plugins/native-operator-guide.md`
- Create: `docs/plugins/native-compatibility.md`
- Create: `docs/plugins/native-performance.md`
- Create: `docs/plugins/template/`
- Modify: `docs/specs/extension-registry.md`
- Modify: `docs/specs/endpoints.md`
- Modify: `docs/specs/exporters.md`
- Modify: `docs/specs/wheel-packaging.md`
- Modify: `docs/rust-architecture.md`
- Modify: `docs/module-organization.md`
- Modify: `llms.txt`
- Modify: `docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-execution-tracker.md`
- Create: `rust/plugin-sdk/tests/docs_examples.rs`
- Create: `rust/plugin-conformance/tests/final_package_and_removal.rs`

**Interfaces:**
- Documents list/validate/inspect-build/lock commands, schema 2.0, template,
  platform layouts/atomic install, trust/priority/privileged policy,
  source API versus exact universe/build identities, absence of stable/proven
  Rust ABI, panic abort behavior, diagnostics/troubleshooting, distribution
  rollback/revocation, benchmark method/results, no runtime plugin sandbox/
  isolation, and the distinct required SDK hermetic build sandbox. It documents
  no generic `--plugin-config` syntax, source-API SemVer/rebuild caveat, and the
  dynamic-loader caveats (post-call revision risk, symbol spelling is not ABI,
  exact residency/no-interposition allocation reliance, and smoke timing limits).
  The compatibility table states exactly: major may remove/change source API;
  minor is additive and source-compatible; patch preserves documented
  signatures; every level may still require an exact binary rebuild. Trust docs
  state generation-1 local-installation trust roots and absence of author
  authentication, signature/rollback/revocation authority, and that the loader
  never downloads, resolves repositories, or mutates installations.

- [ ] **Step 1: Write failing executable documentation tests**

Run every template/command/config example against clean default/full installs;
scan for banned claims (“stable Rust ABI”, plugin sandbox/isolation, dynamic
reload/unload, fallback promotion), obsolete no-dynamic-discovery claims, wrong
Python/native roots, absolute paths, and undocumented schema/identity terms.
Golden assertions require every SemVer row and every loader trust/no-download/
no-repository/no-install-mutation statement above; an empty test is rejected by
requiring at least one executed clean-install example per documented command.

- [ ] **Step 2: Verify RED**

Run:
`cargo test -p aiperf-plugin-sdk --test docs_examples && cargo test -p aiperf-cli --test plugin_commands && cargo test -p aiperf-plugin-conformance --test final_package_and_removal`

Expected: FAIL because required documents/examples are absent and architecture
records still describe static-only composition.

- [ ] **Step 3: Author docs from verified behavior and evidence**

Use actual command output, generated schema, package layouts, mismatch
diagnostics, trust rules, and retained paper-rig reports. Build the third-party
template using only allowlisted public crates. Update architecture only to
behavior proven by current code/tests.

- [ ] **Step 4: Run the full completion audit**

Map every normative spec clause, invariant, rejected-option enforcement,
manifest/algorithm field, failure row, conformance fixture, migration gate,
platform/package/performance requirement, and documentation deliverable to
current code plus fresh evidence. Run full workspace/default/engine/full/Clippy/
fmt, four-platform real loader/product/package, cross-host cellular, and exact
paper-rig A/B evidence. Missing/indirect evidence remains non-pass.
The final conformance test now executes the deferred `packaging` and
`static-removal-audit` fixture groups, verifies SDK bundle census (allowlisted
sources/versions, target/toolchain, ABI artifacts, allocator, universe record,
hermetic/linker policy, schema, and no orchestration-private crates), and checks
post-cutover package/static-path evidence against the exact Task-38 candidate
object IDs.

- [ ] **Step 5: Whole-branch Graham approval, final bundle audit, and commit**

Dispatch the most capable reviewer for the entire spec-to-branch diff and
mandatory second pass. Resolve every blocker/important/minor/style finding and
rerun affected/full gates. Verify every distinct commit bundle and exact local
object ID, require a clean worktree, update every tracker row to `PASS`, and
commit as `docs(plugins): complete native plugin release dossier`.

## Candidate Source Inventory appendix

Task 2 encodes this list in
`rust/plugin-conformance/candidate-source-inventory.toml`; post-Task-6 split
leaves begin as exact `planned/producer_task=6` rows, and Task 6 replaces them
with present BLAKE3-bound implementation rows. Tasks 24–34 copy the
listed implementation leaves and non-code assets into the corresponding
precreated candidate package while production files remain unchanged. Each
inventory row stores source path, candidate destination, owner task, BLAKE3, and
`implementation_leaf|asset|facade`. Every `implementation_leaf` and `asset` must
be byte-identical; only a row explicitly classified `facade` may differ. Missing,
extra, duplicate-owner, or digest-mismatched rows fail candidate staging and
Task 39a.
For an implementation source, the destination preserves the suffix after
`runtime/src/` under the package `src/`; comparator sources preserve the suffix
after their `tests/` root under the package `tests/baseline/`; assets retain
their relative suffix. The exact task-to-package mapping is Task 24
`plugins/export-basic`, Task 25 `plugins/export-parquet`, Task 26
`plugins/export-mlflow`, Task 27 `plugins/export-wandb`, Task 28
`plugins/export-otel`, Tasks 29–30 `plugins/endpoints`, Task 31
`plugins/transport-http`, Task 32 `plugins/transport-grpc`, Task 33
`plugins/transport-websocket` or `plugins/transport-dry-run` according to source,
and Task 34 `plugins/transport-dynosim`. The checked inventory stores every
expanded destination path; it contains no wildcard entries.
The earlier read-only inventory’s descriptive package labels map exactly as
follows and are not additional package identities:
`aiperf-plugin-exporters-basic` → `plugins/export-basic`,
`aiperf-plugin-exporter-parquet` → `plugins/export-parquet`,
`aiperf-plugin-exporter-mlflow` → `plugins/export-mlflow`,
`aiperf-plugin-exporter-wandb` → `plugins/export-wandb`,
`aiperf-plugin-exporter-otel` → `plugins/export-otel`,
`aiperf-plugin-endpoints` → `plugins/endpoints`,
`aiperf-plugin-http` → `plugins/transport-http`,
`aiperf-plugin-grpc` → `plugins/transport-grpc`,
`aiperf-plugin-websocket` → `plugins/transport-websocket`,
`aiperf-plugin-dry-run` → `plugins/transport-dry-run`, and
`aiperf-plugin-dynosim` → `plugins/transport-dynosim`.

- Task 24 basic exporters:
  `runtime/src/export/{accuracy_csv.rs,accuracy_csv/tests.rs,analysis_html.rs,analysis_txt.rs,console_txt.rs,console_txt/cell_widths.rs,console_txt/tests.rs,dataset_analysis.rs,genai_perf.rs,genai_perf/tests.rs,server_metrics/mod.rs,server_metrics/tests.rs,timeslice.rs,timeslice/tests.rs}` and every file under
  `runtime/src/export/console_txt/golden/`.
- Task 25 Parquet exporter:
  `runtime/src/export/{parquet.rs,parquet/tests.rs,parquet/units.rs,parquet_util.rs,per_record_parquet.rs}`.
- Task 26 MLflow exporter:
  `runtime/src/export/{mlflow.rs,mlflow/tests.rs}`.
- Task 27 W&B exporter:
  `runtime/src/export/wandb/{mod.rs,datastore.rs,proto.rs,tests.rs}`.
- Task 28 OTel exporter:
  post-Task-19 `runtime/src/export/{otel.rs,otel/tests.rs}`. Task 19 deletes
  `runtime/src/export/otel/accumulator.rs` and removes the report/engine side
  channel before candidate staging; it is infrastructure work and is never a
  candidate source. Task 28 copies only configuration, decoration, upload, and
  their tests; Task 39a later removes static `otel` registration/authority.
- Task 29 endpoints:
  `runtime/src/endpoints/{anthropic.rs,chat.rs,chat_chunk.rs,extraction.rs,implementation.rs,kserve.rs,riva.rs,sagemaker.rs,spec_decode.rs,tier2.rs,tier2/flexible.rs,usage.rs,vllm_generate.rs}` after Task 6 has moved the shared private helper portions of `implementation.rs` into the endpoint SDK. The candidate `mod.rs` is a reviewed facade and is not equality-hashed. `config.rs` and `models.rs` become core-owned definitions or runtime compatibility adapters; category descriptors, factory traits, and prepared handles from `metadata.rs`/`registry.rs` become plugin-API-owned definitions; remaining registry state is a runtime host adapter. None is copied into a candidate. Endpoint comparator tests are
  `runtime/tests/{endpoints_anthropic_messages.rs,endpoints_endpoints.rs,endpoints_kserve.rs,endpoints_registry.rs,endpoints_riva.rs,endpoints_tier2.rs,endpoints_vllm_generate.rs,tier2_endpoints_online.rs}`.
- Task 31 HTTP transport: every file under `runtime/src/transport/http/`.
  Comparator tests are every file under `runtime/tests/transport_http/`.
- Task 30 endpoint-owned gRPC bindings:
  `runtime/src/transport/grpc/{kserve_binding.rs,proto.rs,riva_binding.rs,riva_codec.rs,riva_proto.rs}`. Task 6 first extracts the generic factory contract and splits host registry state in `binding.rs` from concrete KServe code in `kserve_binding.rs`; Task 30 owns only the concrete KServe/Riva companion implementations beside their endpoint winners.
- Task 32 gRPC transport:
  `runtime/src/transport/grpc/{codec.rs,models.rs,raw_codec.rs,sink.rs,transport.rs}` and asset
  `runtime/tests/proto/grpc_predict_v2.proto`. Candidate `grpc/mod.rs` is a
  reviewed facade. Comparator tests are
  `runtime/tests/{transport_grpc_codec.rs,transport_grpc_riva.rs,transport_grpc_riva_transport.rs,transport_grpc_transport.rs}`.
- Task 33 WebSocket:
  post-Task-6 `runtime/src/transport/{ws.rs,ws/connector.rs,ws/dialect.rs,ws/driver.rs,ws/sink.rs}`; `runtime/src/engine/ws_execution.rs` is a host adapter and is excluded. Comparator test
  `runtime/tests/websocket_transport_config.rs`.
- Task 33 dry-run: post-Task-6 `runtime/src/transport/dry_run.rs`;
  `runtime/src/engine/dry_run.rs` is a host adapter and is excluded. Product comparators are
  `dry-run-tests/tests/{common/mod.rs,component_packages.rs,dry_run.rs,random_pool_batches.rs,timing.rs,timing_extended.rs,tracelab.rs,virtual_workers.rs}`.
- Task 34 Dynosim: post-Task-6
  `runtime/src/{dynosim/direct.rs,endpoints/dynosim.rs}`.
  `runtime/src/dynosim.rs` and `runtime/src/engine/offline_execution.rs` are
  host orchestration/adapters and are excluded. Candidate facades expose the
  Task-6 narrow direct-execution service; the two plugin leaves remain
  equality-hashed.

No candidate copies shared host contracts from `dispatch`, `clock`,
`transport::{core,measure,reduce,retry}`, body planning, metrics, scheduling, or
the engine/category factory traits. Tasks 4/6 move category factory and
transport boundary traits once into plugin API, transport-neutral value/service
contracts once into core, and shared private helper algorithms once into the
relevant category SDK; runtime keeps only compatibility adapters/re-exports.
The Task-6 leaf-ownership test proves every later equality-copied implementation
builds without `aiperf-runtime`, `RunContext`, or private engine DTOs.
No exporter candidate copies the legacy registry/config/ordering adapters or
shared report/stat/CSV/name/display/artifact behavior either. Task 4 owns the
finalized report and artifact contracts; Task 6 owns the single export-SDK
helper implementation, host-adapter split, exporter leaf hashes, and standalone
package-shaped compile gate. Tasks 24–28 consume only those frozen leaves plus
their package-internal sources and backend dependencies.

## Static Path Inventory appendix

Task 39a is the sole owner of production cutover. Its checked manifest begins
with these current production owners and expands to every additional hit found
by the reproducible searches below:

- composition and built-ins: `runtime/src/engine/application.rs`,
  `runtime/src/extensions/mod.rs`, and `runtime/src/engine/registry.rs`;
- endpoint registries and the legacy closed descriptor mapping:
  `runtime/src/endpoints/{registry.rs,config.rs}`;
- gRPC built-in/fresh binding construction:
  `runtime/src/transport/grpc/{binding.rs,riva_binding.rs}` and
  `runtime/src/engine/{grpc_execution.rs,grpc_turn_execution.rs}`;
- closed transport selection/config/direct routes:
  `runtime/src/config/model/transport.rs`,
  `runtime/src/config/validate.rs`,
  `runtime/src/engine/{online_execution.rs,offline_execution.rs,dry_run.rs,cellular_controller.rs}`,
  `runtime/src/dynosim.rs`, and
  `cli/src/{yaml.rs,load.rs,control_hooks.rs,execute_mode.rs}`;
- exporter built-ins and bypasses: `runtime/src/export/mod.rs`, including
  `register_builtins`, `with_builtin_exporters`, and `export_report`;
- CLI composition and allocator wiring: `cli/src/{main.rs,execute_mode.rs}` and
  `cli/src/eval/native_graph.rs`, plus `cli/Cargo.toml`, `runtime/Cargo.toml`,
  `runtime/build.rs`, and `runtime/src/lib.rs`.

Task 39a runs these searches before every component commit and records the
complete sorted output in its static-path manifest:

```bash
rg -n --glob '*.rs' 'Application::(stock|fresh)|BuiltinAIPerfRegistryFactory|with_builtin_extensions' cli/src runtime/src
rg -n --glob '*.rs' 'register_builtins|register_(endpoint|transport)|GrpcBindingRegistry::builtin|GrpcBindingRegistryBuilder::with_builtins' runtime/src
rg -n --glob '*.rs' 'pub enum Transport|Transport::(Http|Grpc|Websocket|DryRun|Dynosim)|legacy_descriptor_for|EndpointType::' cli/src runtime/src
rg -n --glob '*.rs' 'dry_run|dynosim|dynosim_or_unsupported|register_dynosim_transport|register_dry_run_transport' cli/src runtime/src
rg -n --glob '*.rs' 'ExporterRegistry|persist_prepared_report|global_allocator|MiMalloc|mi_option' cli/src runtime/src
```

Test-only comparator hits are permitted only when their exact file, symbol, and
reason appear in `rust/plugin-conformance/static-path-allowlist.toml`; production
owners, package manifests, generated config/schema, and product binary symbols
can never be allowlisted.

## Exact coverage additions required by the final audit

The following assertions are mandatory named fixtures, not an “exact spec”
umbrella. Task 36 owns the pre-staging rows; Task 38 owns exact candidate
performance rows; Task 40 owns deferred package/removal/documentation rows.

| Audit finding | Owner and executable assertion |
|---|---|
| H2/H14/M3 | Task 8 `abi_closure` golden classifies private fields/generated/drop dependencies, reclassifies/revokes on crossing, captures system-library identity, and proves allowlist changes alter host universe. |
| H3/M4/M15 | Task 36 `diagnostics` mutates every descriptor field and every failure-receipt tuple field, requires category/name/package/version/source API/first difference/remediation, and proves normal/debug redaction. |
| H4/M16/M19/L1 | Task 36 `initialization_shutdown`, `residency_and_poison`, and `platform_loader_policy` prove no initializer/entry registration work or detached survivor, exactly-one mapped claimant, post-load metadata diagnostic-only behavior, and each forbidden loader environment/API. |
| H5/H7/H8/H9/H11/H12/M10/M11/M12 | Tasks 6/18/19 compile-fail or golden-test `Any`/`TypeId`, receipt category, `json`/`otlp`, legacy-decoder retention, `FinalReport`/empty config, SDK bundle census, exact capture order/fields/diagnostics, and adversarial exporter ordering. |
| H10/M1/M2 | Tasks 7/12/36 test all provider symbols/version/pointer/overflow, allowed contained native allocators, platform symbol normalization, and failing-inspector pipeline exit propagation. |
| H13/M7/M8/M9 | Tasks 1/3/38 enforce complete inventory identity, joint metric/case bootstrap, legal ratio directions, three CV series, and exact invalidation replacement retention/order. |
| H6/M17/M20 | Tasks 13/17/29 test manifest-only non-executing authority, fatal explicit paths, and transparent `EndpointType` compatibility. |
| H11 | Task 37 `distribution_census` includes Kubernetes image lifecycle alongside native/wheel/container default/full. |
| L2/L3 | Task 40 `docs_examples` rejects generic `--plugin-config` claims and requires all dynamic-loader feasibility caveats. |

## Normative coverage map

This table is the plan self-review’s requirement map. Task 40 must replace each
planned reference with exact code/test/evidence references before completion.

| Spec requirement | Implementing tasks | Decisive evidence |
|---|---|---|
| Invariant 1: native Rust boundary | 5, 6, 9, 14, 36 | separately built real-loader entry/category/native-value calls on four targets plus forbidden-boundary scan |
| Invariant 2: exact build and complete closure | 7, 8, 9, 11, 12, 14, 16, 36 | compiler/sysroot/target/build/tamper/collision fixtures rejected before callable entry; loaded-module map equals locked closure |
| Invariant 3: composition before effects | 14–17, 19–21, 36 | ordered help/list/config/profile/eval/re-exec/cell ledgers with only OS baseline loading before composition |
| Invariant 4: one frozen universe | 15, 16, 20–23 | parent/child/controller/cell full-lock and plan agreement plus no alternate registry construction |
| Invariant 5: process-lifetime residency | 14, 15, 36 | retained/pinned handles, drop-recording object, initializer-failure/no-escape, attempted-unload subprocesses |
| Invariant 6: no runtime mutation | 15, 16, 36 | compile-fail freeze API; same-lock reuse/different-lock rejection; poisoned-set reuse returns original failure |
| Invariant 7: transactional packages | 13–15, 36 | multi-entry late failure and descriptor disagreement expose zero registration prefix |
| Invariant 8: deterministic override | 5, 10, 13, 16, 36 | randomized-order canonical/alias/version/tie/shadow fixtures produce byte-identical catalog/lock |
| Invariant 9: first-/third-party parity | 24–34, 37, 39 | each production ID loads through identical inventory/manifest/entry/freeze path; override behavior and static-path searches |
| Invariant 10: core reuse | 2, 4–6, 9, 36, 38 | Cargo/rustdoc allowlist and separate third-party workspace; host/unrelated rebuild isolation |
| Invariant 11: hot-path shape | 4, 6, 19, 23, 28–34, 38 | IR/call graph plus allocation/lock/spawn/channel/copy/serialization instrumentation |
| Invariant 12: measured equivalence | 1, 3, 24–34, 38 | frozen same-source comparator; authoritative paired paper-rig protocol and zero allocation increase |
| Invariant 13: open selection | 5, 6, 13, 18, 24–34, 36 | Config-v2 schema/projection/open third-party IDs; no closed enum/typed protocol copy |
| Invariant 14: trusted code | 10–14, 17, 35–37, 40 | ownership/mode/ACL/privilege/inventory/tamper/revocation fixtures and explicit no-sandbox docs |
| Invariant 15: no silent fallback | 13–17, 24–36, 39 | intended winner fixed before activation; loader/registration/runtime errors never promote shadowed/static code |
| Invariant 16: one allocator/no unwind | 7–9, 12, 14, 17, 36, 38 | direct eager `mi_*` imports and runtime origins; bidirectional ownership; host/plugin panic abnormal termination |
| Compatibility contract | 2, 5, 7–9, 12, 16, 36, 40 | canonical golden vectors, field-first mismatch diagnostics, terminology scan, private-vs-common rebuild tests |
| Crate architecture/API ownership | 2, 4–6, 9, 37–39 | final Cargo graph/feature matrix, rustdoc ownership table, published allowlist, no orchestration dependency |
| Shared allocator/provider initialization | 7, 12, 14, 36, 38 | four-target import/binding/origin/init/order/memory/performance evidence |
| Author workflow/hermetic build | 8, 9, 12, 35–37, 40 | offline external workspace, undeclared-input rejection, exact records/closure/schema/template |
| Native library entry/registrar | 5, 9, 14, 15, 36 | one exact export, handle-scoped lookup, descriptor cross-check, transactional manifest-bound registration |
| Strict manifest 2.0/Python separation | 10–13, 35–37, 40 | generated schema digest, full negative fixture matrix, exact Python diagnostic and separate install roots |
| Discovery/priority/authority | 10–13, 16, 35, 36 | OS root order, environment/explicit/hermetic modes, sorted resolution, required authority, ACL policy |
| Immutable acquisition/canonical staging | 11–14, 16, 35, 36 | no-follow race fixtures at every boundary, staged rehash, coalescing/collision proof |
| Composition/type-state/lifecycle | 13–17 | exact eleven-step state transition, consumed freeze, resident/poisoned process-global set |
| Plugin lock/bundle | 11, 13, 15, 16, 20, 21, 35–37 | full presence/failure/status/descriptor catalog, atomic bundle, first-difference reproduction |
| CLI command semantics | 17, 20, 36, 40 | root help/completion no discovery; list no code; validate/lock exact activation/publish behavior |
| Re-exec private bootstrap | 19, 20, 22, 36 | inherited FD/handle audit, no rediscovery, request-stdin identity, lock/plan pre-effect verification |
| Cellular/K8s/SLURM bootstrap | 19, 21–23, 36, 37 | fixed-0600/private pipe, image inventory, pre-Velo validation, signed digests, no code transfer |
| Endpoint category/config/gRPC | 6, 18, 19, 29, 30, 36, 38 | three endpoint shapes, strict open IDs, worker-local preparation, gRPC override, byte-stream/observer parity |
| Transport category/config/direct binding | 6, 18, 19, 31–34, 36, 38 | exact request/direct shapes, narrow services, all transport behavior, no enum/switch/static path |
| Exporter capture/config/outcomes | 6, 18, 19, 22–28, 36, 38 | exact legacy table/open order, generic projection modes, bounded cell transfer, scoped artifacts, persisted outcomes |
| Performance contract | 1, 3, 38 | same-revision comparator, structural gate, exact paired/CV/bootstrap/invalidation protocol and retained raw evidence |
| Failure/trust policy | 7, 10–17, 19–23, 35, 36 | every failure row below has authority-specific fixture and stable receipt/outcome |
| Migration/removal gates | 1–39 | per-component ten-condition D6 record, specified integration order, no simultaneous production static/dynamic ID |
| Verification/conformance fixtures | 36–40 | complete fixture inventory uses real separately built artifacts and four platform loaders |
| Documentation/tooling deliverables | 10, 17, 37, 38, 40 | generated schema, commands, template, layouts, compatibility/trust/perf docs and architecture updates |

## Rejected-alternative enforcement map

These tests keep the implementation from drifting into a rejected design while
still appearing feature-complete.

| Rejected option | Enforcing tasks/tests |
|---|---|
| Stable C ABI/function tables | Tasks 5, 6, 9, 36 reject `extern "C"` entry/category/data DTOs and generated dispatch tables; export map contains only native Rust entry symbol |
| `abi_stable` facade | Tasks 2, 5, 9, 36 dependency/source scans reject the crate/facade/container conversion surface |
| Process/RPC/IPC plugins | Tasks 6, 19, 31–34, 36, 38 prove in-process native factory/trait calls and reject request-path serialization/channel/process hops |
| One aggregate distribution library | Tasks 2, 24–34, 37 enforce separate dependency islands, four telemetry closures, and independent rebuild/install/uninstall |
| Static Cargo features or complete executables | Tasks 24–39 prove independently installed packages compose one process and editing one does not relink host |
| Rust `dylib` | Tasks 9, 12, 36 require `cdylib`, exact entry-only exports, and reject wrong artifact kind |
| Scan arbitrary libraries | Tasks 10, 13, 17, 36 discover only strict manifest basenames and prove stray libraries are not executed |
| Explicit-path-only discovery | Tasks 13, 17, 35–37 prove distribution/system/user/environment auto-discovery plus additive explicit inputs |
| Lockfile-first-only execution | Tasks 13, 16, 17, 20 show ordinary discovery always derives a lock while hermetic lock is explicit mode |
| Duplicate rejection without priority | Tasks 13, 15, 36 enforce signed priority, unique maximum, ambiguity, and manifest-bound transaction |
| Dynamic reload/unload | Tasks 14–16, 36 provide no API and prove process-lifetime residency/different-lock rejection |
| Plugins depend on orchestration runtime | Tasks 2, 4–6, 9, 36 Cargo/rustdoc/compile-fail checks reject runtime/RunContext/private imports |
| Per-library allocator/destroy callbacks | Tasks 6, 7, 36, 38 require ordinary Rust ownership through one provider and reject origin metadata/destroy tables |
| Forbid all owned values | Tasks 7 and 36 prove every ownership-table family crosses/drops both directions after allocator gate |
| Switch whole process to `System` | Tasks 7, 12, 36, 38 require pinned shared mimalloc topology and performance against static mimalloc baseline |
| AIPerf allocator wrapper/table | Tasks 7, 12, 36 structural/import/origin scans require direct `mi_*` calls with no intermediate symbol |
| Unwind containment | Tasks 7, 9, 12, 14, 17, 36 require `panic=abort`, reject `catch_unwind`, and observe abnormal subprocess termination |

## Failure-policy implementation matrix

| Condition | Required behavior | Implementing test/tasks |
|---|---|---|
| Baseline allocator/module absent, ambiguous, mismatched, or unauthenticated preload | abort startup before discovery for every authority | 7, 12, 14, 36 |
| Locked bundle/bootstrap/absence receipt/plan mismatch | fail before runtime effects | 16, 19–21, 36 |
| Missing optional default directory | ignore; fail only when authenticated distribution generation requires it | 13, 35, 36 |
| Existing unreadable default/environment root or invalid/empty environment element | fatal discovery policy error | 13, 35, 36 |
| Unreadable/invalid manifest before activation | quarantine optional; fail required/explicit | 10, 13, 16, 36 |
| Missing/wrong artifact or closure member | quarantine optional; fail required/explicit | 11–13, 16, 36 |
| Digest/embedded/universe/package mismatch | quarantine without loading optional; fail required/explicit | 8, 12, 13, 36 |
| Static dependency/search/allocator/panic failure | quarantine without loading optional; fail required/explicit | 7, 9, 12, 13, 36 |
| Same package/version with differing manifests | quarantine every optional claimant; fail any required/explicit claimant | 13, 36 |
| Same non-system loader identity with differing bytes | quarantine optional conflict group; fail any required/explicit claimant | 11–13, 36 |
| Equal maximum priority | record ambiguity; fail when selected or when required key | 13, 16, 18, 36 |
| Loader/dependency activation failure | retain returned handles, poison process, fail composition; no promotion | 14, 36 |
| Entry/descriptor/registration error after activation | rollback registry only, retain stage/handles, poison; no promotion | 14, 15, 36 |
| Post-load canonical lock mismatch | retain handles, poison, fail composition | 14, 16, 20, 21, 36 |
| Boundary panic | process abort, never typed/caught plugin error | 7, 9, 14, 17, 36 |
| Selected quarantined/ambiguous key | fail validation before effects | 13, 17–19, 36 |
| Runtime trait typed error | typed operation failure without re-resolution/promotion | 24–36 |
| Post-report exporter typed error | persist outcome and continue deterministic remaining exporters | 22–28, 36 |
| Explicit abort/undefined behavior/memory corruption | process may terminate; no isolation claim | 17, 35, 40 documentation and subprocess policy fixtures |

## Plan self-review checklist

- [x] Every spec heading and normative clause is mapped by the coverage table.
- [x] Every required conformance fixture is named in Task 36’s required
  inventory and maps to a production task.
- [x] Every rejection has an executable guard in the rejected-option table.
- [x] Every failure-policy row has authority-specific behavior and a fixture.
- [x] Every public produced interface is consumed under the same spelling/type.
- [x] Every shared file/interface has one owner in its wave; every ready
  disjoint task is explicitly fanned out in separate worktrees.
- [x] Drafting-token scan for the writing-plans skill’s complete forbidden-pattern
  list returns no matches.
- [x] Every AIPerf feature requires focused/full gates, independent Graham
  approval, per-commit bundle verification/import, and integration evidence.
- [x] Final completion requires four-platform, cross-host, packaging,
  authoritative paper-rig A/B, and whole-branch audit evidence—not agent reports.
