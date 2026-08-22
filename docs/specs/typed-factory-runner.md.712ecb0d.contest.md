# Contest ledger

> Do not edit — generated from the contest tables after each round.

- **kind:** plan_off
- **status:** failed
- **failure:** round 2 did not complete before its deadline (attempt 1)
- **round:** 2 / 6
- **artifact:** `docs/specs/typed-factory-runner.md`
- **low-friction:** yes — zero retractions and zero contested objections; unanimity is as consistent with correlated bias as with correctness, so this exchange is flagged UNVALIDATED rather than clean
- **persuasion-override rate (CW-POR):** 0.00 — the fraction of author retractions that answered an UNPROVEN objection; a high value means the author yielded to confident assertion rather than to evidence

## Seats

- **author** — `opus` (family `claude-trace`, lens —)
- **skeptic** — `gpt-5.6-sol` (family `claude-codex`, lens migration correctness — does §5's 4-step sequence land without silently changing runtime behavior, and does each step really ship green?)

## Objections

## O1 — §2/§5.4 ports a nonexistent “exactly one dataset” check. `BenchmarkRunWireV2::validate_outer` accepts every non-empty `cfg.datasets`, and `BenchmarkRunWireV2::into_authored` silently selects `.next()`. Implementing the artifact's stated `BenchmarkRun::validate()` cardinality changes behavior for two-or-more datasets instead of preserving it.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Artifact:
  - `docs/specs/typed-factory-runner.md:202` says `exactly one dataset` is enforced by `validate_outer()`.
  - `docs/specs/typed-factory-runner.md:212-215` requires `BenchmarkRun::validate()` to port `exactly one dataset`.
  - `docs/specs/typed-factory-runner.md:438-441` repeats that step 4 must land `validate_outer`'s three checks.
  
  Exact implementation:
  - `rust/runtime/src/engine/protocol_v2.rs:343-347` is only:
    `self.cfg`
    `    .datasets`
    `    .as_ref()`
    `    .is_some_and(|datasets| !datasets.is_empty()),`
    `"run.cfg.datasets must contain exactly one dataset"`
    This checks non-empty, not `len() == 1`.
  - `rust/runtime/src/engine/protocol_v2.rs:359-364` says and does:
    `// The runner builds exactly one dataset; take the first and re-serialize it`
    `let dataset = cfg`
    `    .datasets`
    `    .and_then(|datasets| datasets.into_iter().next())`
    `    .ok_or_else(|| anyhow!("run.cfg.datasets must contain one dataset"))?;`
  - `rust/runtime/src/config/model/config.rs:91-93` declares `pub datasets: Option<Vec<Dataset>>`, so a two-element value is representable.
  
  Command run:
  `rg -n "datasets.*len|exactly one dataset|datasets.*is_empty|first\\(\\)" rust/runtime/src/config rust/runtime/src/engine/protocol_v2.rs rust/cli/src`
  found no dataset cardinality check; the only relevant hits were `protocol_v2.rs:346` (`!datasets.is_empty()`) and `protocol_v2.rs:364` (`.next()`). The diagnostic is inaccurate; porting the diagnostic as a real cardinality predicate is a new rejection.

## O2 — §5.4's projection-deletion port list omits the live `--profile-export-prefix` summary/timeslice stem derivation in `BenchmarkRunWireV2::into_authored`. Pointing execution at typed `cfg.export` without reproducing this cross-field transform silently restores `profile_export_aiperf.{json,csv}` / `profile_export_aiperf_timeslices.*` names while per-record output remains under the authored custom stem; none of step 4's named tests guards it.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Exact projection behavior omitted by the artifact:
  - `rust/runtime/src/engine/protocol_v2.rs:464-473`:
    `// Derive the summary stem from the per-record path so`
    ``// `--profile-export-prefix` / `artifacts.prefix` renames``
    ``// `*_aiperf.{json,csv}` together with the jsonl.``
    `if let Some(path) = artifacts_spec.records_path.as_ref()`
    `    && let Some(name) = path.file_name().and_then(|s| s.to_str())`
    `{`
    `    let stem = name.strip_suffix(".jsonl").unwrap_or(name);`
    `    if !stem.is_empty() {`
    `        export_cfg.genai_perf.stem = stem.to_string();`
    `        export_cfg.timeslice.stem = Some(format!("{stem}_aiperf"));`
  - `rust/runtime/src/export/genai_perf.rs:150-168` consumes `cfg.genai_perf.stem` and writes `artifact_dir.join(format!("{stem}_aiperf.json"))` / `...csv`.
  - `rust/runtime/src/export/timeslice.rs:39-40` defines `pub stem: Option<String>` as the filename stem.
  - `rust/runtime/src/config/resolve.rs:1406-1413` derives `let stem = artifact_export_stem(inputs.profile_export_prefix.as_deref());` and stores `records_path: ... format!("{stem}.jsonl")`.
  - `rust/cli/src/load.rs:2231-2245` has `fn profile_export_prefix_rewrites_artifact_stem()` and asserts `arts.records_path.as_deref() == Some("myrun.jsonl")`.
  
  Artifact omission:
  - `docs/specs/typed-factory-runner.md:435-530` inventories step 4, endpoint/model transforms, and `resource_presence`, but never names `export_cfg.genai_perf.stem`, `export_cfg.timeslice.stem`, or `--profile-export-prefix`.
  - `docs/specs/typed-factory-runner.md:568-576` names the complete step-4 gate; it does not include `test_gpu_telemetry` or `test_server_metrics`, the only e2e targets found containing `--profile-export-prefix` (`rust/e2e-tests/tests/test_gpu_telemetry.rs:218-244`, `rust/e2e-tests/tests/test_server_metrics.rs:427-456`), and those tests only inspect telemetry filenames, not `*_aiperf` summaries.
  
  Command attempted:
  `cargo test -p aiperf-cli profile_export_prefix_rewrites_artifact_stem -- --exact`
  The command did not reach the test because current working-tree `rust/cli/src/kube/submission.rs:77` fails on `if let` guards (`E0658`). The source test and exact production consumers above mechanically establish the transform and the missing gate; I am not claiming the failed compile demonstrates this objection.

## O3 — §5.2's unconditional exhaustive `match` on `Transport` does not define feature-off behavior. `Transport::{Grpc,Websocket,DynosimOffline,DynosimOnline}` always deserialize, while their execution factories are `#[cfg]`-registered. Selection currently fails closed through missing registry entries; moving built-in selection out of the registry needs explicit feature-gated rejection arms and a lean-build gate. The stated “one-to-one correspondence” is false per compiled distribution.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Artifact:
  - `docs/specs/typed-factory-runner.md:374-394` moves `native_execution` selection to an exhaustive `Transport` match while keeping registry entries only for descriptors/id-addressed consumers.
  - `docs/specs/typed-factory-runner.md:406-419` claims `every in-tree registration ... registers exactly one of the six Transport variants` and says step 2 `opens no selection gap`.
  - `docs/specs/typed-factory-runner.md:532-557` gives the shared and step-2 gates; none invokes `--no-default-features`.
  
  Exact source contradiction by feature configuration:
  - `rust/runtime/src/config/model/transport.rs:15-30` declares all six variants without any `#[cfg]`:
    `pub enum Transport {`
    `    Http,`
    `    Grpc,`
    `    DynosimOffline(DynosimConfig),`
    `    DynosimOnline(DynosimConfig),`
    `    DryRun(DryRunConfig),`
    `    Websocket(WebSocketTransportConfig),`
    `}`
  - `rust/runtime/src/engine/registry.rs:758-770` gates `GrpcExtension` and `register_transport(Arc::new(OnlineGrpcTransportFactoryV2))` with `#[cfg(feature = "grpc")]`.
  - `rust/runtime/src/engine/registry.rs:775-788` gates `WebSocketExtension` / `WebSocketTransportFactoryV2` with `#[cfg(feature = "websocket")]`.
  - `rust/runtime/src/engine/registry.rs:793-806` gates `DynosimExtension` with `#[cfg(feature = "dynosim")]`.
  - `rust/runtime/src/engine/registry.rs:603-609` currently performs registry selection and therefore rejects a feature-off transport before execution.
  - `rust/runtime/Cargo.toml:10-40` has `default = ["parquet", "cellular", "grpc"]`, `websocket = [...]`, and `dynosim = [...]`; thus the artifact's shared `cargo test -p aiperf-runtime --features engine` gate is already a compiled distribution where `Transport::Websocket` and both DynoSim variants exist but their factories do not.
  - `rust/cli/Cargo.toml:24-25` explicitly supports `--no-default-features`; `rust/cli/Cargo.toml:55-57` says feature-off config parsing remains and execution `fails closed at registry selection`.
  
  An unconditional arm cannot reference feature-gated `GrpcNativeExecution`/`WebSocketNativeExecution`/DynoSim implementations in lean builds; deleting the arm under `#[cfg]` makes the always-present enum non-exhaustive. The migration must specify a feature-off error arm/predicate preserving the current `not compiled into this distribution; available: ...` behavior and verify `cargo check/test -p aiperf-cli --no-default-features`.

## O4 — The step-4 target type is self-contradictory: Purpose requires the runtime to consume `BenchmarkRun`, and the port list deliberately puts controller/run facts on `BenchmarkRun`, but §2 and §5.4 tell the implementer to drive/point the engine at `&BenchmarkConfig`. `BenchmarkConfig` cannot carry `benchmark_id`, `artifact_dir`, `planned_replay_traces`, `trial`, `variation`, `resolved`, or `variables`; following the literal migration discards the very fields §2 says must survive.
severity: critical   raised: r1   status: standing
proven: yes
evidence:
  Exact artifact contradiction:
  - `docs/specs/typed-factory-runner.md:9-15`: `the runtime consumes the typed BenchmarkRun directly`.
  - `docs/specs/typed-factory-runner.md:182-185`: `The child composition root (coordinator.rs) drives the engine against &BenchmarkConfig and its typed resolved facts`.
  - `docs/specs/typed-factory-runner.md:216-225` requires `planned_replay_traces` and typed `variation` to live on `BenchmarkRun`.
  - `docs/specs/typed-factory-runner.md:435-441`: `point coordinator.rs at BenchmarkConfig`, then requires the `BenchmarkRun` port list.
  
  Exact types:
  - `rust/runtime/src/config/model/run.rs:17-48` declares `BenchmarkRun` with `benchmark_id`, `artifact_dir`, `cfg: BenchmarkConfig`, `trial`, `variation`, `resolved`, and `variables`.
  - `rust/runtime/src/config/model/config.rs:63-148` declares `BenchmarkConfig`; none of those run/envelope fields is present.
  - `rust/runtime/src/engine/protocol_v2.rs:323-325` currently owns controller state as `pub planned_replay_traces: BTreeSet<PlannedReplayTraceInstance>` on `BenchmarkRunWireV2`, not on `BenchmarkConfig`.
  - `rust/runtime/src/engine/coordinator.rs:135-159` retains the enclosing run: `let benchmark_id = Some(envelope.run.benchmark_id.clone());`, validates `envelope`, then calls `envelope.run.into_authored()`.
  - `rust/runtime/src/engine/online_execution.rs:1216` takes `run: &AuthoredRunSpecV2`; `rust/runtime/src/engine/online_execution.rs:1280` assigns `plan.planned_replay_traces = workload.planned_replay_traces.clone()`.
  
  A coherent step 4 must point the composition root and downstream seams at `&BenchmarkRun` (accessing `&run.cfg` for component config), or explicitly define a new context carrying every non-config run fact. `&BenchmarkConfig` plus unspecified “resolved facts” is not that type and does not even include controller-authored `planned_replay_traces`.

## O5 — The round-4 O11 refinement still permits a behavior regression: §5.4 says the migration may “pick” either conflicting `resource_presence` algorithm. The product coordinator only reaches `AuthoredRunSpecV2` through `BenchmarkRunWireV2::into_authored`, whose algorithm is known. Since the artifact promises unchanged runtime behavior and deletes the alternate direct-deserialize vocabulary, step 4 must preserve the `into_authored` classification, not choose arbitrarily.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  This explicitly challenges the round-4 fix to prior O11, not its already-settled discovery of the two algorithms.
  
  Artifact:
  - `docs/specs/typed-factory-runner.md:511-525` accurately prints the `into_authored` behavior: `models: true`, `endpoints: true`, `metrics: true`, `artifacts: true`, and serialized-nonempty `sidecars`.
  - `docs/specs/typed-factory-runner.md:526-530` then says the two paths disagree and `the migration must pick one deliberately`.
  - `docs/specs/typed-factory-runner.md:364-366` requires `(no behavior change)`; `docs/specs/typed-factory-runner.md:580-582` says each step must carry behavior forward.
  
  Actual product path:
  - `rust/runtime/src/engine/coordinator.rs:159` is `let run = match envelope.run.into_authored() {`; this is the composition-root path step 4 replaces.
  - `rust/runtime/src/engine/protocol_v2.rs:496-501` is therefore the product classification:
    `models: true,`
    `endpoints: true,`
    `metrics: true,`
    `artifacts: true,`
    `sidecars: sidecars_present,`
  - `rust/runtime/src/engine/protocol_v2.rs:661-667` contains the alternate `wire.resources.X.is_some()` classification only inside `impl<'de> Deserialize<'de> for AuthoredRunSpecV2`.
  - Command `rg -n "AuthoredRunSpecV2" rust/runtime/src/engine` shows production constructors/consumers and direct `serde_json::from_value::<AuthoredRunSpecV2>` only in `rust/runtime/src/engine/registry.rs:2017`, `:2050`, and `:2078` tests; stdin is decoded as `BenchmarkRunWireV2` and then projected.
  
  Because step 4 deletes `AuthoredRunSpecV2`, there is no surviving direct-deserialize contract to balance against the self-written path. Choosing `cfg.models.is_some()` etc. would change current product workload requirement behavior. The normative choice is the `into_authored` map; “pick one deliberately” is insufficient and contradicts the migration's own no-behavior-change criterion.

## O6 — Step 4's executable gate does not guard its own four-item port list. The only source test for unknown outer fields is under `#[cfg(any())]` (permanently disabled), and no named step-4 e2e target asserts `BenchmarkRun::validate()`, `#[serde(deny_unknown_fields)]`, typed `VariationSpec`, or `planned_replay_traces` decode/retention. Deleting the DTO and its tests can leave every written gate green while omitting the prerequisites §5 calls the silent-regression path.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Artifact:
  - `docs/specs/typed-factory-runner.md:438-441` requires, before DTO deletion: `#[serde(deny_unknown_fields)]`, `validate()` with three checks, `planned_replay_traces`, and `variation: Option<VariationSpec>`.
  - `docs/specs/typed-factory-runner.md:568-576` defines the complete step-4 gate only as the full step-3 gate plus cellular e2e targets.
  
  Current tests do not pin the port:
  - `rust/runtime/src/engine/protocol_v2.rs:1284-1285` is:
    `#[cfg(any())]`
    `mod tests {`
  - The disabled module contains `fn outer_contract_rejects_unknown_fields()` at `rust/runtime/src/engine/protocol_v2.rs:1328-1335`.
  - The active module begins at `rust/runtime/src/engine/protocol_v2.rs:1410` as `#[cfg(test)] mod dispatch_mode_tests`; its active tests exercise execute-wire shape and projection/dispatch/hop behavior, not new `BenchmarkRun` strictness or validation.
  - Command:
    `rg -n "unknown outer field|benchmark_id cannot be empty|artifact_dir cannot be empty|exactly one dataset|VariationSpec|planned_replay_traces" rust/e2e-tests/tests rust/runtime/src/config/model rust/runtime/src/engine/protocol_v2.rs`
    returned no matches in `rust/e2e-tests/tests` or `rust/runtime/src/config/model`; all relevant matches are declarations/implementation in `protocol_v2.rs`, plus the disabled unknown-field test.
  - `rust/runtime/src/config/model/run.rs:50-73` has only `fn none_optionals_emit_null_not_absent()`; no port-list tests exist there.
  
  The cellular targets may exercise `planned_replay_traces` operationally, but they do not mechanically require the new field to reject unknowns, validate empty identity/path/datasets, or type `variation`. Step 4 must require direct decode/validation tests on `BenchmarkRun` before deleting `BenchmarkRunWireV2`; otherwise its own shared Cargo commands are compatible with deleting old tests and failing to add replacements.

## O7 — The round-4 O7 scoping fix leaves the dynamic-plugin non-goal false. §1 selects a closed `BenchmarkConfig.transport` and an exhaustive emergent workload match; §3 says its `RegistryId`/`RawValue` tail is the `NamedRunnerComponentSpecV2` seam; §5.4 deletes that seam. After the specified migration there is no authored transport/workload selection object carrying `{ RegistryId, RawValue }`, so the claim that runtime-loaded transport/workload plugins remain selectable through a retained tail is unsupported by—and contradictory to—the target model.
severity: medium   raised: r1   status: standing
proven: yes
evidence:
  This challenges the consequence of the prior O7 refinement, not the settled fact that current `BenchmarkConfig.transport` is closed.
  
  Exact artifact:
  - `docs/specs/typed-factory-runner.md:128-134` selects `(A) Keep Transport closed` and says §3 applies to a different runner seam.
  - `docs/specs/typed-factory-runner.md:135-153` makes workload identity an exhaustive `{Scheduled, Graph}` match (with only a possible future `StaticAccuracy` arm).
  - `docs/specs/typed-factory-runner.md:252-258` scopes §3 to `NamedRunnerComponentSpecV2 { id, config }`, `the structure §5 step 4 deletes`.
  - `docs/specs/typed-factory-runner.md:435-436` deletes `NamedRunnerComponentSpecV2`.
  - Yet `docs/specs/typed-factory-runner.md:617-625` claims runtime-loaded `transports/workloads` are in scope via `the plugin tail's RawValue config (keyed by RegistryId, decoded by the plugin's own dyn factory ...)` and `The refactor does not close the set`.
  
  Exact source confirms the tail's current owner:
  - `rust/runtime/src/engine/protocol_v2.rs:851-857` defines `NamedRunnerComponentSpecV2` with `pub id: ComponentId` and `pub config: Box<RawValue>`.
  - `rust/runtime/src/engine/protocol_v2.rs:585-588` places that type on `AuthoredRunSpecV2.transport` and `.workload`.
  - `rust/runtime/src/config/model/config.rs:73-75` instead has `pub transport: Option<Transport>`.
  - `rust/runtime/src/config/model/transport.rs:15-30` is the six-variant closed enum with no plugin arm.
  - `rust/runtime/src/config/model/workload.rs` has no authored `RegistryId`/`RawValue` workload selection; `workload_kind(&BenchmarkConfig)` is emergent as the artifact states.
  
  Registry entries retained for `--capabilities` or id-addressed NativeGraph model bindings do not create an authored workload selection path. The record must either narrow the non-goal to future categories/NativeGraph transport bindings, or define a surviving plugin-tail owner; it cannot simultaneously delete the only tail, keep both config selections closed/exhaustive, and claim dynamic transport/workload selection remains accommodated.

## O8 — Purpose's current-truth status is false: it says the record describes work that is “not built,” but commit `b7619602fb` on the stated implementation branch already ships the named step-3 DTO collapse as `WorkloadConfigV2`. This is documentation drift, and it now misleads implementers about which migration prerequisites already exist.
severity: low   raised: r1   status: standing
proven: yes
evidence:
  Artifact:
  - `docs/specs/typed-factory-runner.md:25` says: `This record is forward-looking. It describes work that is **not built**.`
  - `docs/specs/typed-factory-runner.md:420-423` defines step 3 as collapsing `ScheduledWorkloadConfigV2` / `GraphWorkloadConfigV2` while retaining all graph-only fields.
  
  Implementation worktree `/home/anthony/nvidia/projects/aiperf/ajc/rust-tfr-v2`:
  - Command `git merge-base --is-ancestor b7619602fb HEAD; printf 'workload-step-ancestor=%s\\n' "$?"` printed `workload-step-ancestor=0`.
  - `git show --stat --oneline b7619602fb` printed:
    `b7619602fb refactor(engine): collapse the two workload config DTOs into one WorkloadConfigV2`
    `rust/runtime/src/engine/registry.rs | 66 +++++++++++++++----------------------`
  - `git show b7619602fb:rust/runtime/src/engine/registry.rs | rg -n "struct WorkloadConfigV2|recorded_agent_default|system_idle_gap_cap_seconds"` printed:
    `855:pub struct WorkloadConfigV2 {`
    `876:    pub system_idle_gap_cap_seconds: Option<f64>,`
    `883:    pub recorded_agent_default: bool,`
  
  The record can remain forward-looking for unfinished steps, but “work ... not built” is no longer accurate. It should identify implemented prerequisites/step-3 collapse and remaining steps 1, 2, and 4.

## Unresolved risks

- O1 — §2/§5.4 ports a nonexistent “exactly one dataset” check. `BenchmarkRunWireV2::validate_outer` accepts every non-empty `cfg.datasets`, and `BenchmarkRunWireV2::into_authored` silently selects `.next()`. Implementing the artifact's stated `BenchmarkRun::validate()` cardinality changes behavior for two-or-more datasets instead of preserving it.
- O2 — §5.4's projection-deletion port list omits the live `--profile-export-prefix` summary/timeslice stem derivation in `BenchmarkRunWireV2::into_authored`. Pointing execution at typed `cfg.export` without reproducing this cross-field transform silently restores `profile_export_aiperf.{json,csv}` / `profile_export_aiperf_timeslices.*` names while per-record output remains under the authored custom stem; none of step 4's named tests guards it.
- O3 — §5.2's unconditional exhaustive `match` on `Transport` does not define feature-off behavior. `Transport::{Grpc,Websocket,DynosimOffline,DynosimOnline}` always deserialize, while their execution factories are `#[cfg]`-registered. Selection currently fails closed through missing registry entries; moving built-in selection out of the registry needs explicit feature-gated rejection arms and a lean-build gate. The stated “one-to-one correspondence” is false per compiled distribution.
- O4 — The step-4 target type is self-contradictory: Purpose requires the runtime to consume `BenchmarkRun`, and the port list deliberately puts controller/run facts on `BenchmarkRun`, but §2 and §5.4 tell the implementer to drive/point the engine at `&BenchmarkConfig`. `BenchmarkConfig` cannot carry `benchmark_id`, `artifact_dir`, `planned_replay_traces`, `trial`, `variation`, `resolved`, or `variables`; following the literal migration discards the very fields §2 says must survive.
- O5 — The round-4 O11 refinement still permits a behavior regression: §5.4 says the migration may “pick” either conflicting `resource_presence` algorithm. The product coordinator only reaches `AuthoredRunSpecV2` through `BenchmarkRunWireV2::into_authored`, whose algorithm is known. Since the artifact promises unchanged runtime behavior and deletes the alternate direct-deserialize vocabulary, step 4 must preserve the `into_authored` classification, not choose arbitrarily.
- O6 — Step 4's executable gate does not guard its own four-item port list. The only source test for unknown outer fields is under `#[cfg(any())]` (permanently disabled), and no named step-4 e2e target asserts `BenchmarkRun::validate()`, `#[serde(deny_unknown_fields)]`, typed `VariationSpec`, or `planned_replay_traces` decode/retention. Deleting the DTO and its tests can leave every written gate green while omitting the prerequisites §5 calls the silent-regression path.
- O7 — The round-4 O7 scoping fix leaves the dynamic-plugin non-goal false. §1 selects a closed `BenchmarkConfig.transport` and an exhaustive emergent workload match; §3 says its `RegistryId`/`RawValue` tail is the `NamedRunnerComponentSpecV2` seam; §5.4 deletes that seam. After the specified migration there is no authored transport/workload selection object carrying `{ RegistryId, RawValue }`, so the claim that runtime-loaded transport/workload plugins remain selectable through a retained tail is unsupported by—and contradictory to—the target model.
- O8 — Purpose's current-truth status is false: it says the record describes work that is “not built,” but commit `b7619602fb` on the stated implementation branch already ships the named step-3 DTO collapse as `WorkloadConfigV2`. This is documentation drift, and it now misleads implementers about which migration prerequisites already exist.
