# Contest ledger

> Do not edit — generated from the contest tables after each round.

- **kind:** plan_off
- **status:** failed
- **failure:** round 5 could not be delivered to its seat: the derived agent turn dead-lettered
- **round:** 5 / 8
- **artifact:** `docs/specs/typed-factory-runner.md`
- **low-friction:** yes — zero retractions and zero contested objections; unanimity is as consistent with correlated bias as with correctness, so this exchange is flagged UNVALIDATED rather than clean
- **persuasion-override rate (CW-POR):** 0.00 — the fraction of author retractions that answered an UNPROVEN objection; a high value means the author yielded to confident assertion rather than to evidence

## Seats

- **author** — `opus` (family `claude-trace`, lens —)
- **skeptic** — `gpt-5.6-sol` (family `claude-codex`, lens migration correctness — does §5's 4-step sequence actually land without silently changing runtime behavior, and does each step really ship green?)

## Objections

## O1 — §5.4's claim that `into_authored` injects no data absent from the typed model is false: `BenchmarkRunWireV2::planned_replay_traces` is controller-authored state that `BenchmarkRun` does not contain, and `into_authored` is the only bridge that copies it into `GraphWorkloadConfigV2::planned_replay_traces`. Deleting `BenchmarkRunWireV2` and `into_authored` as written drops cellular recorded-replay ownership.
severity: critical   raised: r1   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:263-269` says:
  - `Delete ... BenchmarkRunWireV2`
  - ``into_authored` injects **no** data absent from the typed model`
  
  Exact source:
  - `rust/runtime/src/config/model/run.rs:17-48` defines every field of `pub struct BenchmarkRun`; there is no `planned_replay_traces` field.
  - `rust/runtime/src/engine/protocol_v2.rs:323-325` defines `pub planned_replay_traces: BTreeSet<PlannedReplayTraceInstance>` on `BenchmarkRunWireV2`.
  - `rust/runtime/src/engine/protocol_v2.rs:411-415` copies it into the workload payload:
    `workload_config["planned_replay_traces"] =`
    `serde_json::to_value(&self.planned_replay_traces)`
  - `rust/runtime/src/engine/cellular_controller.rs:1973-1981` injects controller assignments into each cell run:
    `run.insert(`
    `    "planned_replay_traces".to_owned(),`
    `    serde_json::to_value(expected_replay_traces)`
  - `rust/runtime/src/engine/online_execution.rs:1280` consumes the copied value:
    `plan.planned_replay_traces = workload.planned_replay_traces.clone();`
  
  Thus this is not a copy or derivation from `BenchmarkConfig`/`BenchmarkRun`; it is per-cell controller state. §5 needs an explicit typed owner and cellular migration/test before deletion.
resolution (r2): Verified true and it is the most serious defect in the plan. `planned_replay_traces` appears ZERO times under `rust/runtime/src/config/` and `rust/cli/src/`. Its full lifecycle is: declared on `BenchmarkRunWireV2` (`protocol_v2.rs:325`), written by the controller at `cellular_controller.rs:1978` into the cell's `/run/` JSON, copied into `GraphWorkloadConfigV2` (`registry.rs:904`) by `into_authored` (`protocol_v2.rs:413-415`), consumed at `entrypoints.rs:415` as `expected_replay_traces`. So §5.4's "injects **no** data absent from the typed model — an audited per-field pass found every field is a copy or a pure derivation" was flatly false, and the deletion it authorizes would silently drop cellular graph replay's trace expectation.

Artifact changed (commit c9ef19d193): §5 step 4 now carries an explicit "Corrected (contest round 2)" paragraph tracing the full lifecycle with file:line, and states the migration precondition — the field needs a home on `BenchmarkRun` as a run-level fact alongside `cfg`, NOT a `BenchmarkConfig` field, because it is controller-derived rather than authored. The "every other field is a copy or derivation" claim is retained, scoped. §1 and the Built section were updated in the same commit to stop describing it as collapsing for free.

## O2 — §5's wire-compatibility statement identifies the wrong boundary. The actual stdin protocol is `AuthoringWire { authoring: Inputs, ... }`, not `EnvelopeV2`; `EnvelopeV2` is reconstructed only after `decode_execute_wire`. §2/§5.4 delete that dual decode and say plain `EnvelopeV2` decoding subsumes it, which cannot preserve either current CLI stdin bytes or the documented bare-resolved-run compatibility path.
severity: critical   raised: r1   status: refined
proven: yes
evidence:
  Spec claims:
  - `docs/specs/typed-factory-runner.md:115-123`: ``EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain `serde_json::from_slice`` and ``decode_execute_wire` reduces to decoding one `EnvelopeV2`; the dual authoring/bare-run accept path is subsumed`
  - `docs/specs/typed-factory-runner.md:263-265`: delete `BenchmarkRunWireV2` and reduce protocol-v2 to `EnvelopeV2`
  - `docs/specs/typed-factory-runner.md:281-282`: `Each step keeps the stdin protocol wire-compatible (the outer `EnvelopeV2` is unchanged)`
  
  Actual sender and receiver:
  - `rust/cli/src/profile.rs:159-168` defines `struct AuthoringWire<'a> { authoring: &'a load::Inputs, sweep_id: Option<String>, variation: Option<serde_json::Value>, trial: u32 }`.
  - `rust/cli/src/profile.rs:197-205` serializes that exact `AuthoringWire` and passes it to `execute::run_once`.
  - `rust/cli/src/execute_mode.rs:464-481` first calls `decode_execute_wire(input)`, then constructs `EnvelopeV2 { protocol_version: PROTOCOL_V2, operation, run }`; `EnvelopeV2` is not the stdin shape.
  - `rust/runtime/src/engine/protocol_v2.rs:169-185` documents two accepted stdin shapes: `{"authoring": <Inputs>}` and a bare `BenchmarkRunWireV2`, including external harness compatibility.
  - `rust/runtime/src/engine/protocol_v2.rs:207-232` performs `Inputs -> BenchmarkRun` resolution and sweep overlay in `resolved_run_bytes`.
  
  Plain `serde_json::from_slice::<EnvelopeV2>` cannot decode the current `{"authoring": ...}` bytes, and deleting the dual path also removes the documented bare-run shape. The sequence needs to say where resolution/sweep overlay moves and what exact stdin DTO remains compatible.
resolution (r2): Verified true. `EnvelopeV2` (`protocol_v2.rs:118`) is never deserialized from stdin — its own doc comment says it is "reconstructed around the bare `BenchmarkRunWireV2` stdin payload", and its sole construction site is `cli/src/execute_mode.rs:477`, after decode. The real stdin contract is `decode_execute_wire` (`protocol_v2.rs:191`), which accepts either an `AuthoringWireV2` (`{"authoring": <Inputs>, sweep_id, variation, trial}`, `protocol_v2.rs:155`, `deny_unknown_fields`) or a bare `BenchmarkRunWireV2`, discriminated on presence of the `authoring` key by `resolved_run_bytes` (`protocol_v2.rs:207`). So "each step keeps the stdin protocol wire-compatible (the outer `EnvelopeV2` is unchanged)" pinned a boundary that does not exist, and §2's "`EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain `serde_json::from_slice`" describes a decode that never happens.

Consequence the plan had hidden from itself: step 4 deletes `BenchmarkRunWireV2`, which IS half of the actual stdin contract. That is a wire change, not an internal-projection change.

Artifact changed (commit c9ef19d193): the §5 closing paragraph now states "The boundary to hold fixed is not `EnvelopeV2`", gives the real accept path with file:line, distinguishes the authoring arm (what AIPerf itself writes) from the bare-resolved-run arm (retained for external harnesses), and requires step 4 to either re-type the bare arm to `BenchmarkRun` or declare a stated compatibility break. It also names §2's sentence as describing a boundary that does not exist today.

## O3 — §5.4's asserted audit of `validate_run` is false. The seam consumes `run.dispatch`, `run.workload.id`, `run.artifacts.trace`, multiple artifact paths, and every sidecar field—not only `models.items` and `sidecars.live_streaming`. Repointing only the two named fields to `&BenchmarkConfig` silently removes existing rejection behavior.
severity: critical   raised: r1   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:269-270` says: `The `validate_run` seam repoints to `&BenchmarkConfig` **losslessly** (it consumes only `models.items` and `sidecars.live_streaming`).`
  
  Counterexamples in production implementations:
  - `rust/runtime/src/engine/dry_run.rs:539-552` reads `run.dispatch` and `run.workload.id`:
    `run.dispatch != crate::engine::protocol::DispatchMode::Sharded`
    `run.workload.id.as_str() != "graph"`
  - `rust/runtime/src/engine/ws_execution.rs:151-162` reads `run.artifacts.trace` plus `run.sidecars.content_server`, `run.sidecars.gpu_telemetry`, `run.sidecars.network_latency`, `run.sidecars.server_metrics`, and `run.sidecars.live_streaming`.
  - `rust/runtime/src/engine/grpc_execution.rs:149-155` reads all five sidecar fields, not only `live_streaming`.
  - `rust/runtime/src/engine/offline_execution.rs:871-879` reads `run.artifacts.records_path`, `run.artifacts.raw_path`, `run.artifacts.outputs_path`, `run.artifacts.trace`, and `run.artifacts.user_files` in `ensure_no_common_artifacts`.
  - `rust/runtime/src/engine/online_execution.rs:455-468` also reads `transport_id`, `run.models.items`, and `workload.accuracy` in static-accuracy `validate_run`.
  
  These checks produce user-visible validation failures today. §5 must enumerate and preserve all of them (including derived `dispatch` and workload identity), not repoint only `models.items`/`sidecars.live_streaming`.
resolution (r2): Verified true. `validate_run` is not one body, and I read all three. `online_execution.rs:105` is the only one matching the spec's inventory. `ws_execution.rs:151` additionally consumes `run.artifacts.trace` plus five sidecar fields, and rejects trace artifacts and all sidecars outright. `dry_run.rs:539` consumes `run.dispatch` and `run.workload.id.as_str()`, rejecting sharded dispatch and graph workloads under virtual workers. So "it consumes only `models.items` and `sidecars.live_streaming`" was a single-body inventory generalized to the seam, and the word "losslessly" was doing unearned work — a repoint audited against `online_execution` alone would silently change the `ws_execution` and `dry_run` rejection surfaces.

Artifact changed (commit c9ef19d193): §5 step 4 now says the seam is "not one body", scopes the original inventory to `online_execution.rs:105`, enumerates what the other two bodies consume and reject with file:line, and states each body repoints on its own terms.

## O4 — §5.2's claim that the audit found only two registry obligations omits the shipped NativeGraph evaluation path. `CurrentNativeGraphModelBindingResolver` and `ResolvedModelBindingSet` directly require built-in `transport_factory(...)` entries for URL-scheme validation, typed config validation, and `native_execution`. The instruction that built-ins leave the `id → factory` lookup breaks native graph before the proposed HTTP/gRPC e2e verification can detect it.
severity: critical   raised: r1   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:250-259` lists exactly two obligations and says: `Keep the `id → factory` lookup for the plugin tail; only built-ins leave it. Verify with the http + grpc e2e suites.`
  
  Unmentioned shipped consumers:
  - `rust/runtime/src/eval/native_graph/model_runtime.rs:157-164` calls `self.registry.transport_factory(binding.transport_factory_id())` and then reads `transport.descriptor().url_schemes`.
  - `rust/runtime/src/eval/native_graph/model_runtime.rs:307-335` calls `context.product_registry().transport_factory(&binding.transport_id)`, then `factory.validate(...)`, then `factory.native_execution(...)` for `engine_selected_policy_runtime_with_summary`.
  - `rust/runtime/src/eval/native_graph/model_runtime.rs:387-416` repeats the same `transport_factory` → `validate` → `native_execution` chain in `engine_inputs`.
  - `rust/runtime/src/engine/graph_execution.rs:3453` contains a built-in model binding with `transport_factory_id = "http"`.
  
  Removing built-in factories from `AIPerfRegistry::transport_factory` therefore makes the built-in `"http"` binding return `ModelRuntimeError::UnknownTransport`. §5.2 must either keep built-in execution providers lookup-addressable or migrate all three NativeGraph call sites and verify the Harbor native-graph suite.
resolution (r2): Verified. §5 step 2 named two registry obligations; there is a third. `CurrentNativeGraphModelBindingResolver::resolve` (`rust/runtime/src/eval/native_graph/model_runtime.rs:94`, with further lookups at `:309` and `:389`) calls `registry.transport_factory(binding.transport_factory_id())` for its `ModelRuntimeError::UnknownTransport` rejection, then reads `transport.descriptor().url_schemes` to drive `validate_transport_urls`. Confirmed by grep: the only `transport_factory(` callers in `runtime/src/` are `registry.rs:547` (the definition), `online_execution.rs:125` (the one §5.2 names, inside `resolve_native_execution`), and those three eval sites.

Why this matters beyond a missing bullet: this is the `aiperf eval --model-runtime` native-graph path, and §5 step 2's stated verification is "the http + grpc e2e suites" — the *profile* suites, which do not touch it. So the plan's own verification gate would have passed while `UnknownTransport` and URL-scheme validation regressed. The fix is either verifying that path separately or keeping built-ins registered for descriptor lookup even after selection leaves the registry.

Artifact changed (commit c9ef19d193): §5 step 2 gains obligation (c) with those file:line anchors, notes explicitly that the profile e2e suites do not exercise it, and extends the step's verification to "the http + grpc e2e suites **and** the native-graph eval path".

## O5 — The proposed transport representation is internally contradictory and the concrete `ComponentSpec` example is not wire-flat. §1 requires `BenchmarkConfig.transport: Transport` as a closed internally-tagged enum, while §3 requires the same discriminant to remain open `RegistryId` with a plugin tail. Its plain-derived `ComponentSpec { id, config }` expects a nested `config` key, contradicting the promised `{ "type": ..., …config… }` shape and today's flat transport wire. Step 1 therefore has no executable, wire-compatible target type.
severity: high   raised: r1   status: refined
proven: yes
evidence:
  Contradictory target statements:
  - `docs/specs/typed-factory-runner.md:77-81`: ``transport: Transport` — a `#[serde(tag = "type")]` internally-tagged enum, one variant per compiled transport`
  - `docs/specs/typed-factory-runner.md:130-139`: `The discriminant ... stays an open, normalized string, not an enum` and `This drops the manual-`Deserialize`/`untagged`-tail machinery entirely`
  - `docs/specs/typed-factory-runner.md:145-150`: `A component on the wire is therefore just `{ "type": <RegistryId>, …config… }``, followed by `struct ComponentSpec { #[serde(rename = "type")] id: RegistryId, config: Box<RawValue> }`
  
  The example has no `#[serde(flatten)]`; a plain derive maps its named `config` field to a literal nested `"config"` key, not to the remaining flat fields.
  
  Current executable shape:
  - `rust/runtime/src/config/model/transport.rs:15-31` is `#[serde(tag = "type", rename_all = "snake_case")] pub enum Transport` with no plugin-tail variant.
  - `rust/runtime/src/engine/protocol_v2.rs:523-537` implements the current flat shape by removing `"type"` from the object and wrapping the remaining object as `config` in `component_from_inline`.
  
  A closed `Transport` enum cannot accept a future unknown `RegistryId`; the shown open `ComponentSpec` cannot decode today's flat built-in config without custom flatten/buffering machinery that §3 says is unnecessary. The migration must choose and specify one real serde representation before step 1.
resolution (r2): Verified true on both halves — this is stale-draft contamination, not a live disagreement, which makes it worse: a reader implementing §1 literally builds the exact encoding §3 proves is broken.

(a) Closed enum vs open string. §1 said `transport: Transport` — "a `#[serde(tag = "type")]` internally-tagged enum, one variant per compiled transport … Feature-gated transports are `#[cfg(feature = "…")]` variants". §3 says the opposite in bold: the discriminant "**stays an open, normalized string**, not an enum", is `RegistryId`, and "drops the manual-`Deserialize`/`untagged`-tail machinery entirely" — backed by the 2026-07-26 empirical serde test showing the tagged-enum-plus-untagged-tail encoding compiles but misbehaves. §4 carried the same residue: "fail closed at serde decode of the tagged enum" and "the enum needs a small custom deserialize error".

(b) Wire shape. §1's internally-tagged enum implies flat `{type, …fields}`; §3's own prose said "built-in fields flat, as today" while the code sketch immediately below it declares `struct ComponentSpec { #[serde(rename = "type")] id: RegistryId, config: Box<RawValue> }` — nested. Today's actual shape is `NamedRunnerComponentSpecV2 { id: ComponentId, config: Box<RawValue> }`, i.e. nested. So "as today" was backwards, and §3 contradicted itself in adjacent lines.

Artifact changed (commit c9ef19d193): §1's transport bullet is rewritten to type the config payload while explicitly keeping the `RegistryId` string discriminant, names §3 as authoritative, and records that the closed-enum framing was a retired draft; feature gating is restated as applying to `match` arms. §4's two enum-decode sentences are rewritten around the built-in `match` / plugin-tail split, and the diagnostic caveat is re-aimed at the real risk — the built-in arms bypass registry lookup, so a typo'd built-in id must still produce the "available: …" list from the default arm. §3's flat/nested sentence is corrected to nested-under-`config`.

## O6 — §1/§5.3 undercount the graph-only workload state: `GraphWorkloadConfigV2` has five fields absent from `ScheduledWorkloadConfigV2`, not two. The omitted `recorded_agent_default` and `system_idle_gap_cap_seconds` are behaviorally consumed in `lower_graph`, so a migration scoped to the stated two typed optionals silently removes canonical-bundle validation and the legacy idle cap even apart from `planned_replay_traces`.
severity: high   raised: r1   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:87-91` says: `The two `*WorkloadConfigV2` DTOs differ by exactly two graph-only fields (`weka_semantics`, `ignore_trace_delays`)`.
  `docs/specs/typed-factory-runner.md:260-262` then says to collapse the DTOs into typed-optional fields without correcting that inventory.
  
  Exact DTO fields:
  - `rust/runtime/src/engine/registry.rs:887-905` adds all five graph-only fields: `weka_semantics`, `system_idle_gap_cap_seconds`, `ignore_trace_delays`, `recorded_agent_default`, and `planned_replay_traces`.
  - `rust/runtime/src/engine/registry.rs:846-859` shows `ScheduledWorkloadConfigV2` has none of those five.
  
  Behavioral consumers of two omitted fields:
  - `rust/runtime/src/engine/online_execution.rs:1251-1253`: `if workload.recorded_agent_default { validate_canonical_recorded_agent_bundle(&prepared.bundle)?; }`
  - `rust/runtime/src/engine/online_execution.rs:1263`: `system_idle_gap_cap_seconds: workload.system_idle_gap_cap_seconds`
  - `rust/runtime/src/engine/graph_execution.rs:1535`: `system_idle_gap_cap_ms: self.system_idle_gap_cap_seconds.map(|s| s * 1000.0)`
  
  The graph collapse must inventory and preserve all five fields/derivations, not only the two named in the plan.
resolution (r2): Verified true, and part of the error was mine — my earlier edit to this record's Built section (commit 20ec8346c2) preserved "the first two differ only in a graph-only field" without counting them.

`protocol_v2.rs:404-422` attaches five fields on the `WorkloadKind::Graph` arm: `weka_semantics`, `ignore_trace_delays`, `recorded_agent_default`, `planned_replay_traces`, and `system_idle_gap_cap_seconds` — the last conditionally, only when `weka_semantics` is `legacy` or `agentx`. §1's "differ by exactly two graph-only fields (`weka_semantics`, `ignore_trace_delays`) that **already exist as typed fields on `BenchmarkConfig`**; they collapse into typed-optional fields … impossible to misplace" was therefore wrong on the count AND on the collapse claim for the two fields that matter most:

- `recorded_agent_default` is not a field at all — it is the derivation `cfg.scenario.as_deref() == Some("recorded-agent-default")`, and it is behaviorally load-bearing: `online_execution.rs:1251` branches on `workload.recorded_agent_default`.
- `planned_replay_traces` is not on `BenchmarkConfig` in any form (see O1).
- `system_idle_gap_cap_seconds` is a `BenchmarkConfig` field, but its *conditional attachment* is behavior a naive typed-optional copy would lose.

"Impossible to misplace" was exactly the sentence that made the step look free.

Artifact changed (commit c9ef19d193): the Built section enumerates all five with the `protocol_v2.rs:404-422` anchor and separates copies from the derivation and the wire-only field; §1's bullet is rewritten to "**five**", splits the three that collapse from the two that do not, and names those two as "the real work of this step"; §5 step 3 now requires all five and calls out the `recorded_agent_default` derivation and the `weka_semantics`-conditional cap attachment as behavioral, with the `online_execution.rs:1251` consumer cited.

## O7 — The round-2 edit still does not define a typed `BenchmarkConfig.transport` representation. §1 says built-in config payloads are typed, but §3's only concrete wire/in-memory type is `ComponentSpec { id: RegistryId, config: Box<RawValue> }`; that leaves every built-in payload opaque on `BenchmarkConfig` and requires the same post-decode re-decode the design exists to remove. Calling the heterogeneous result “a downstream ergonomics choice” is incorrect because that result must be the type of `BenchmarkConfig.transport` before migration step 1 can compile.
severity: critical   raised: r3   status: refined
proven: yes
evidence:
  Contradictory target statements remain:
  - `docs/specs/typed-factory-runner.md:81-88`: `Every component config ... becomes a typed field on the one model` and `The **config payload** becomes typed per built-in`.
  - `docs/specs/typed-factory-runner.md:156-159`: `The config payload — becomes a typed struct, always`.
  - `docs/specs/typed-factory-runner.md:167-176` defines the only concrete component type as `struct ComponentSpec { #[serde(rename = "type")] id: RegistryId, config: Box<RawValue> }` and then decodes built-ins afterward with `HttpConfig::try_from_raw(&spec.config)` / `GrpcConfig::try_from_raw(&spec.config)`.
  - `docs/specs/typed-factory-runner.md:189-192` leaves the resulting heterogeneous type as `a downstream ergonomics choice`.
  
  Current source proves this choice is the model boundary, not downstream:
  - `rust/runtime/src/config/model/config.rs:72-74` declares `pub transport: Option<Transport>`.
  - `rust/runtime/src/config/model/transport.rs:15-31` defines that field's concrete deserialized type as `#[serde(tag = "type", rename_all = "snake_case")] pub enum Transport`.
  
  Replacing that field with the shown `ComponentSpec` makes `BenchmarkConfig` opaque again; replacing it with the result of `decode_transport` requires the artifact to specify the concrete serializable/deserializable field type and plugin-tail representation. No such type is defined, so step 1 has no compilable target.
resolution (r4): Valid, and resolving it exposed that my own round-2 fix went the WRONG DIRECTION. I record that plainly because it changes the credit: O5 asked me to reconcile §1 (closed tagged enum) against §3 (open `RegistryId`), and I reconciled §1 *toward* §3 — rewriting §1's transport bullet to keep a string discriminant. That was backwards. §1 was describing shipped reality and I edited reality to match a draft.

`rust/runtime/src/config/model/transport.rs:12-32`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Transport {
    Http,                                    // unit variant, no payload
    Grpc,                                    // unit variant, no payload
    DynosimOffline(DynosimConfig),
    DynosimOnline(DynosimConfig),
    DryRun(DryRunConfig),
    Websocket(WebSocketTransportConfig),
}
```

and `config/model/config.rs:75` is `pub transport: Option<Transport>`. So the question O7 poses — "what is the type of `BenchmarkConfig.transport` before step 1 can compile?" — already has an answer in the tree, and it is a **closed** internally-tagged enum with typed payloads. There is nothing to introduce and no `RawValue` on that field. `ComponentSpec { id: RegistryId, config: Box<RawValue> }` was never a candidate type for it; it is the shape of `NamedRunnerComponentSpecV2`, the *runner projection* this record deletes.

The real defect was that the record conflated two different objects under one word. Artifact changed (commit 6271845b4c):

- §1's transport bullet now opens "**this field already exists and is already typed**", quotes the enum with the `transport.rs:18` anchor, states the (A) keep-it-closed / (B) reopen-to-`RegistryId` fork explicitly, and **selects (A)**. The reasoning is that the field is authored config, the built-in transport set is frozen at compile time (`Application` bootstrap), and a closed enum gives exhaustiveness checking at every match — reopening it to a string would be a downgrade adopted only to match a seam that does not apply to it.
- §3 gains a scope header restricting it to the runner component seam — the structure being deleted — so a reader can no longer apply its `RegistryId`/`RawValue` prescription to the config model.
- §5 steps 1-2 are rewritten around an exhaustive `match` on `Transport`, not `match id.as_str()`. Step 1 for transports is explicitly *not* "add typed configs" (there is nothing to add) but "add a consumer reading `cfg.transport` directly, alongside the projection, asserting identical bindings."

One consequence I added that the objection did not ask for: because `Http` and `Grpc` are unit variants, the http+grpc e2e suites exercise **no** payload-bearing arm. Step 2's gate therefore must include `aiperf-dry-run-tests` and `test_websocket`, or the typed-payload path ships unverified.

## O8 — The artifact now proves its own §2 requirement false but leaves it normative and unchanged. §2 still requires plain `serde_json::from_slice::<EnvelopeV2>` and says the dual stdin path is subsumed; §5 now correctly says that boundary does not exist and that deleting `BenchmarkRunWireV2` touches stdin. The non-goal still calls `EnvelopeV2` the “outer stdio” contract. An implementer cannot satisfy both §2 and §5, so step 4 remains ambiguous despite the appended correction.
severity: critical   raised: r3   status: refined
proven: yes
evidence:
  Exact internal contradiction:
  - `docs/specs/typed-factory-runner.md:129-139` still requires: ``EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain `serde_json::from_slice``; ``decode_execute_wire` reduces to decoding one `EnvelopeV2``; `the dual authoring/bare-run accept path is subsumed`.
  - `docs/specs/typed-factory-runner.md:328-342` now says the opposite: `The boundary to hold fixed is not `EnvelopeV2``, `EnvelopeV2 ... is never deserialized from stdin`, `Step 4's deletion of `BenchmarkRunWireV2` therefore *does* touch the stdin contract`, and `§2's ... describes a boundary that does not exist today`.
  - `docs/specs/typed-factory-runner.md:357-360` still lists `the outer stdio `EnvelopeV2` contract` under `Not touched`.
  
  Executable source agrees with the correction, not §2:
  - `rust/cli/src/execute_mode.rs:464-481` calls `decode_execute_wire(input)` before constructing `EnvelopeV2 { protocol_version: PROTOCOL_V2, operation, run }`.
  - `rust/runtime/src/engine/protocol_v2.rs:191-232` implements the actual dual stdin decoder and resolution/sweep overlay.
  
  The correction must replace the false requirement/non-goal, not merely coexist with them.
resolution (r4): Valid and correctly diagnosed as a *process* failure, not just a content one. In round 2 I answered O2 by appending a "Corrected (contest round 2)" paragraph to §5 while leaving the sentence it refuted standing as normative text in §2. That is the worst of both: the record contained its own refutation and its own error, and an implementer reading top-to-bottom hits the error first. Appending a correction is not resolving an objection.

Verified state before the fix: §2 line 162 read "`EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain `serde_json::from_slice`", §2 line 168 read "`decode_execute_wire` reduces to decoding one `EnvelopeV2`; the dual authoring/bare-run accept path is subsumed", and the non-goals bullet listed "the outer stdio `EnvelopeV2` contract" as untouched. All three contradict the verified fact that `EnvelopeV2` (`protocol_v2.rs:118`) is never deserialized from stdin — its sole construction site is `cli/src/execute_mode.rs:477`, after decode.

Artifact changed (commit fb57400724), edited **in place** rather than appended:

- §2 is rewritten to open with the real contract: `decode_execute_wire` (`protocol_v2.rs:191`) remains the stdin boundary and keeps **both** arms; what changes is the *type* of the bare arm. It states in bold "**There is no `serde_json::from_slice::<EnvelopeV2>` in this design**" and names the earlier requirement as removed, not footnoted.
- The "dual accept path is subsumed" sentence is deleted.
- The non-goals bullet now reads "the stdio accept path (`decode_execute_wire`'s two arms — `EnvelopeV2` is an in-process struct built after decode, not a wire shape; see §2)".
- The Source anchors entry changes from "`EnvelopeV2` (the outer shape to keep)" to "(an in-process struct constructed after decode — not a wire shape)".
- The §5 closing paragraph no longer re-litigates §2; it points at it.

§2 and §5 now say the same thing, and step 4's ambiguity is gone.

## O9 — Retyping the bare stdin arm from `BenchmarkRunWireV2` to `BenchmarkRun` is not wire-compatible as §5 claims. The types have different strictness and field contracts: `BenchmarkRunWireV2` is `deny_unknown_fields`, uses `VariationSpec`, `usize` trial, and open `Value` resolved facts; `BenchmarkRun` accepts unknown top-level/config fields, uses open `Value` variation, `u32` trial, and typed `Resolved`. The migration must define an explicit compatibility DTO/adapter or declare the break; “re-type ... or drop” cannot coexist with “Each step keeps ... wire-compatible.”
severity: high   raised: r3   status: refined
proven: yes
evidence:
  Artifact:
  - `docs/specs/typed-factory-runner.md:328`: `Each step keeps the stdin protocol wire-compatible`.
  - `docs/specs/typed-factory-runner.md:338-340`: `the bare arm must either be re-typed to `BenchmarkRun` or dropped with a stated compatibility break`.
  
  Exact type differences:
  - `rust/runtime/src/engine/protocol_v2.rs:292-328`: `BenchmarkRunWireV2` has `#[serde(deny_unknown_fields)]`, `pub resolved: Value`, `pub variation: Option<VariationSpec>`, `pub trial: usize`, and `pub planned_replay_traces: BTreeSet<PlannedReplayTraceInstance>`.
  - `rust/runtime/src/config/model/run.rs:17-48`: `BenchmarkRun` has no `deny_unknown_fields`, `pub variation: Option<serde_json::Value>`, `pub trial: u32`, `pub resolved: Resolved`, and no current `planned_replay_traces`.
  - `rust/runtime/src/engine/protocol.rs:276-287`: `VariationSpec` itself has `#[serde(deny_unknown_fields)]` and requires `index: usize` plus `label: String`.
  - `rust/runtime/src/config/model/config.rs:2-5` explicitly says `Unknown sections are ignored during deserialization`, and `BenchmarkConfig` at `:61-62` has no `#[serde(deny_unknown_fields)]`.
  
  Therefore a direct retype both rejects previously representable `trial` values above `u32::MAX` and accepts malformed/unknown variation and config shapes previously rejected. It is a semantic wire change, not a type alias.
resolution (r4): The diagnosis is correct and every field difference checks out; the prescribed remedy does not apply to this tree, and the record now says why. Both halves are in the artifact.

**Diagnosis verified in full** — `protocol_v2.rs:293-329` vs `config/model/run.rs:17-48`, every axis you named plus one you did not:

| | `BenchmarkRunWireV2` | `BenchmarkRun` |
|---|---|---|
| strictness | `#[serde(deny_unknown_fields)]` | none |
| `resolved` | `serde_json::Value` | typed `Resolved` |
| `variation` | `Option<VariationSpec>` | `Option<serde_json::Value>` |
| `trial` | `usize` | `u32` |
| `variables` | `BTreeMap<String, Value>` | `serde_json::Map<String, Value>` |
| `planned_replay_traces` | `BTreeSet<PlannedReplayTraceInstance>` | **absent** |
| outer validation | `validate_outer()` (`protocol_v2.rs:332-350`) | **none** |

The seventh row is mine, not yours, and it is the sharpest one: `validate_outer` enforces non-empty `benchmark_id`, non-empty `artifact_dir`, and "exactly one dataset", and `BenchmarkRun` has no equivalent and no other enforcement point. Your objection undercounted its own case.

**Where the remedy diverges.** You wrote that the migration "must define an explicit compatibility DTO/adapter or declare the break." I first took option (a) and specified the DTO (commit fb57400724). The author then corrected the premise: the Rust tree is greenfield — there is no released native protocol-v2 stdin contract and no external consumer of the bare resolved-run arm. The `--execute` boundary is written and read by the same binary. So there is no third party to owe a shim to, and "declare the break" is not a concession, it is just the shape of an ordinary refactor here.

Artifact changed again (commit 084f8f44d2):

- Purpose gains an explicit **"Scope premise: the Rust tree is greenfield"** paragraph, so this cannot be re-litigated from the text. It also pins what "byte-exact against the mock server" means: **observed run output**, never stdin bytes.
- §2 deletes `BenchmarkRunWireV2` outright and re-types the bare arm to `BenchmarkRun`.
- **The table survives, re-framed as a port list rather than a contract** — because your underlying point stands even with compatibility gone: each row is a behavior that silently vanishes if the swap is naive. Per-row decisions are now stated: `deny_unknown_fields` **port it** (§2's own claim that "`deny_unknown_fields` on the typed model is the wire strictness" is only true once the attribute is actually on `BenchmarkRun`; without it this migration is a strictness *regression* performed in the name of typing); `validate_outer`'s three checks **port them** as an inherent `BenchmarkRun::validate()`; `planned_replay_traces` **port it** as a run-level field (O1); `variation` **tighten it** — the wire DTO's typed `VariationSpec` is the better shape and `BenchmarkRun`'s open `Value` should be re-typed to it rather than inherited; `resolved`/`trial`/`variables` **accept `BenchmarkRun`'s shapes**, which is the typed model winning and is the point.
- §5 step 4 now reads: delete the DTO, but only *after* the port list lands — "deleting the DTO before those land is the silent-regression path."
- The "wire-compatible at each step" claim is gone. The per-step property is now behavioral: strictness, validation, and controller-derived state carried forward, plus identical observed mock-server output.

So: your finding drove a real change, and the artifact is stronger for it. What changed is that the fix is a port list, not a shim.

## O10 — The round-2 `validate_run` correction is still incomplete: it says the original inventory holds for `online_execution.rs:105`, then enumerates only WebSocket and dry-run bodies, but omits `GrpcNativeExecution::validate_run` and both dynosim workload validation paths. Those bodies consume all sidecars and artifact policy. Deleting `AuthoredRunSpecV2` while following the new inventory still loses existing gRPC/dynosim rejection behavior.
severity: high   raised: r3   status: refined
proven: yes
evidence:
  Artifact inventory:
  - `docs/specs/typed-factory-runner.md:311-317` names only `online_execution.rs:105`, `ws_execution.rs:151`, and `dry_run.rs:539`, then says `Each body repoints on its own terms`.
  
  Missing production bodies:
  - `rust/runtime/src/engine/grpc_execution.rs:33` is `impl NativeTransportExecution for GrpcNativeExecution`.
  - `rust/runtime/src/engine/grpc_execution.rs:73-75` implements `fn validate_run(&self, run: &AuthoredRunSpecV2, context: &RunContext)`.
  - `rust/runtime/src/engine/grpc_execution.rs:149-155` rejects `run.sidecars.content_server`, `gpu_telemetry`, `network_latency`, `server_metrics`, and `live_streaming`.
  - `rust/runtime/src/engine/offline_execution.rs:871-879` defines `ensure_no_common_artifacts(run: &AuthoredRunSpecV2, ...)` and reads `run.artifacts.records_path`, `raw_path`, `outputs_path`, `trace`, and `user_files`.
  - `rust/runtime/src/engine/offline_execution.rs:887-906` calls that predicate from `dynosim_scheduled_validate_run`.
  - `rust/runtime/src/engine/offline_execution.rs:1477` is another `AuthoredRunSpecV2` consumer for the graph dynosim path.
  
  `rg -n "impl NativeTransportExecution for" rust/runtime/src/engine` returns four implementations: `grpc_execution.rs:33`, `ws_execution.rs:119`, `online_execution.rs:66`, and `dry_run.rs:494`; the artifact inventories only three.
resolution (r4): Valid, and it caught me committing the exact error I had criticized one round earlier. In round 2 I faulted the record for generalizing a single call site to a whole seam, then produced a three-body inventory by reading three files and stopping. That is the same failure with a bigger sample.

I enumerated the seam properly this time. It is not one body and **not even one trait** — there are two:

- `NativeTransportExecution::validate_run(&self, run, context)` — transport-level, 2-arg.
- `WorkloadFactory::validate_run(&self, run, context, transport, workload, transport_id)` — workload-level, 5-arg, declared `registry.rs:293`.

Seven-plus bodies:

**Transport-level.** `online_execution.rs:105` (http) — the *only* body the original "consumes only `models.items` and `sidecars.live_streaming`" inventory ever described. `grpc_execution.rs:73` → `validate_grpc_run` (`:85`) — consumes `context.default_endpoint_profile()`, `context.endpoint_profiles()`, `profile.config.urls`, and rejects **all** sidecars. `ws_execution.rs:151` — `run.artifacts.trace` plus five sidecar fields. `dry_run.rs:539` — `run.dispatch`, `run.workload.id.as_str()`.

**Workload-level.** `online_execution.rs:228` (scheduled) — delegates to the transport binding, or falls through the `dynosim_or_unsupported!` macro (`online_execution.rs:135-151`) to `offline_execution::dynosim_scheduled_validate_run` (`:887`, requiring `workload.worker_count == 1`), then runs `validate_authored_tokenizer`. `online_execution.rs:324` (graph). `online_execution.rs:447` (static accuracy) — requires `transport_id == "http"`, consumes `run.models.items.len()`.

**One correction to my own round-2 text that you did not flag and I am volunteering:** I attributed `sidecars.live_streaming` to the transport level. It is consumed by the **graph workload** body, at `online_execution.rs:337`. So the original inventory's *second* named field was on the wrong side of the trait boundary the whole time — which is precisely why a single-body audit of this seam is unsafe.

Artifact changed (commit fb57400724): §5 step 4's `validate_run` paragraph is replaced with the two-trait framing and the full bulleted inventory above, each entry with file:line and what it consumes and rejects. It closes: "A repoint audited against `online_execution.rs:105` alone silently changes the gRPC, WebSocket, dry-run, dynosim, graph, and static-accuracy rejection surfaces."

## O11 — §5.4 recognizes the `resource_presence` hazard but still does not specify the exact classification needed to execute the migration. Today `models`, `endpoints`, `metrics`, and `artifacts` are unconditionally `true`, while `sidecars` is true only for a non-empty serialized object—not `cfg.sidecars.is_some()`. Because all five typed `BenchmarkConfig` fields are `Option`, “reconstruct exactly” without this map still invites the precise Forbidden-behavior regression the paragraph warns about.
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:321-326` says only that the repoint must `reconstruct that exact classification explicitly`; it never states the classification.
  
  The exact current algorithm is:
  - `rust/runtime/src/engine/protocol_v2.rs:430-442`: `sidecars_present` is true only when the serialized sidecar object `!object.is_empty()`; `Some(Sidecars::default())` therefore remains absent.
  - `rust/runtime/src/engine/protocol_v2.rs:496-502` hardcodes:
    `models: true,`
    `endpoints: true,`
    `metrics: true,`
    `artifacts: true,`
    `sidecars: sidecars_present,`
  - `rust/runtime/src/engine/registry.rs:666-690` uses those booleans in `validate_resource_requirements`, including the `(ResourceRequirementV2::Forbidden, true)` rejection.
  - `rust/runtime/src/config/model/config.rs:63-110` represents all five corresponding sections as `Option<...>`, so a natural direct-typed implementation yields a different map.
  
  The plan's warning does not tell the implementer that four `Option` fields are always classified present while the fifth is classified by serialized non-emptiness. That is the behavior that must be written into step 4.
resolution (r4): Valid. Naming a hazard is not specifying a migration, and "reconstruct that exact classification explicitly" without printing the classification is an instruction an implementer cannot follow. Verified and now written out.

`protocol_v2.rs:496-501`:

```rust
resource_presence: ResourcePresenceV2 {
    models: true, endpoints: true, metrics: true, artifacts: true,
    sidecars: sidecars_present,
},
```

Four unconditional `true`s, regardless of whether the corresponding `Option` on `BenchmarkConfig` is `None`. And `sidecars_present` (`protocol_v2.rs:432-443`) is:

```rust
let sidecars_value = serde_json::to_value(&cfg.sidecars)?;
let (sidecars, sidecars_present) = if sidecars_value
    .as_object().is_some_and(|object| !object.is_empty())
{ (serde_json::from_value(sidecars_value)?, true) } else { (SidecarSpecV2::default(), false) };
```

I checked the one thing that decides whether your predicate distinction has teeth, and it does. Every field of `Sidecars` (`config/model/telemetry.rs:206-221`) carries `#[serde(default, skip_serializing_if = "Option::is_none")]`. So `Some(Sidecars::default())` serializes to `{}` and classifies **absent**, while `cfg.sidecars.is_some()` would call it **present**. That is a live divergence on a reachable value, not a theoretical one, and it flips the `(ResourceRequirementV2::Forbidden, true)` arm in `validate_resource_requirements`.

**One finding beyond the objection.** There is a second construction path: `AuthoredRunSpecV2::deserialize` (`protocol_v2.rs:661-667`) uses the naive `wire.resources.X.is_some()` form. So the two paths **already disagree today**. That matters for the migration in a way neither of us had stated: an implementer who "just picks whichever the code does" gets a different answer depending on which construction site they happen to read, and either choice silently changes behavior for the other path's callers.

Artifact changed (commit fb57400724): §5 step 4 hazard (b) now prints the exact map with `protocol_v2.rs:496-501`, gives `sidecars_present` verbatim with its anchor, spells out the `Some(Sidecars::default())` → `{}` → absent case against the `telemetry.rs:206-221` `skip_serializing_if` attributes, names the Forbidden-matrix consequence, and records the two-paths-already-disagree finding with the instruction that the migration must pick one **deliberately** rather than inherit whichever it happens to touch first.

## O12 — §5 still has no executable per-step verification plan, and step 4 has no gate at all despite moving controller-authored `planned_replay_traces` and changing the cellular bare-run path. “http + grpc e2e suites,” “native-graph eval path,” and “graph + scheduled e2e” name no Cargo commands or test targets; none explicitly requires the existing `test_graph_cellular`/`test_grpc_cellular` launches. The plan can therefore satisfy its written gates while breaking cellular execution, contradicting “byte-exact against the mock server at each step.”
severity: high   raised: r3   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:263` promises `byte-exact against the mock server at each step`.
  The only gates named are:
  - `docs/specs/typed-factory-runner.md:288`: `Verify with the http + grpc e2e suites **and** the native-graph eval path.`
  - `docs/specs/typed-factory-runner.md:294`: `Verify graph + scheduled e2e.`
  Step 4 (`:295-326`), which moves `planned_replay_traces`, names no verification gate at all.
  
  Concrete product tests that exercise the affected paths exist:
  - `rust/e2e-tests/tests/test_graph_cellular.rs:28` — `async fn test_graph_cellular()`
  - `rust/e2e-tests/tests/test_graph_cellular.rs:115` — `async fn test_graph_cellular_single_file_dataset_shipping()`
  - `rust/e2e-tests/tests/test_grpc_cellular.rs:68` — `async fn test_grpc_cellular_run()`
  - `rust/e2e-tests/tests/test_grpc_cellular.rs:111` — `async fn test_grpc_cellular_matches_single_cell()`
  - `rust/e2e-tests/tests/test_recorded_agent_cellular.rs:27` — `fn cellular_replay_fold_is_stable_across_cell_arrival_order()` (this is supplement-fold coverage, not a launched cellular graph run).
  
  The repository instructions require `cargo test -p aiperf-runtime --features engine`; the artifact gives no command for that gate or the `aiperf-e2e-tests` targets, and no step-4 cellular/recorded-agent launch test. Thus the stated per-step byte-exact/green property is not backed by an executable verification sequence.
resolution (r4): Valid, and the sharpest structural objection in the contest. A migration record that promises "byte-exact against the mock server at each step" while naming zero commands has an unfalsifiable gate — and the step that touches the most dangerous state had no gate at all. Verified: §5 contained exactly two verification sentences ("the http + grpc e2e suites **and** the native-graph eval path", "graph + scheduled e2e"), neither naming a target, and step 4 named nothing.

Artifact changed (commit fb57400724): §5 gains a **Verification gates (executable)** block. Shared floor at every step, from `rust/` with the venv active:

```bash
cargo fmt --check && cargo clippy --all-targets
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

with the reason stated inline — plain `cargo test -p aiperf-runtime` runs **zero** engine tests, and the `engine` feature gates the entire projection this record changes. Then per step, with targets checked to exist in `rust/e2e-tests/tests/`:

- **Step 1** — differential assertion that both paths produce identical bindings, plus `--test test_default_behavior --test test_chat_endpoint --test test_completions_endpoint --test test_kserve_grpc_endpoint`.
- **Step 2** — step-1 gate, plus `cargo test -p aiperf-dry-run-tests --test dry_run --test virtual_workers` and `--test test_websocket`, because (per O7) `Http` and `Grpc` are **unit variants** and those two suites are the only payload-bearing transport arms any suite exercises; plus `--test test_harbor_native_graph_rollout` for O4's `aiperf eval --model-runtime` path, which the profile suites do not touch.
- **Step 3** — step-2 gate, plus `--test test_conditional_graph --test test_flatgraph_parity --test test_ignore_trace_delays --test test_recorded_agent_replay --test test_dag_full_topology`. **And an admission the objection did not extract:** `recorded_agent_default` and `system_idle_gap_cap_seconds` have **no** dedicated e2e target today, and both are silent losses rather than decode errors. Step 3 must therefore *add* two tests before it may be called green — a graph run asserting `validate_canonical_recorded_agent_bundle` still rejects a non-canonical bundle, and a `weka_semantics: legacy` run asserting the idle cap still applies. Naming a missing gate is worth more than naming an existing one.
- **Step 4** — previously **no gate at all**, "which is how it could have shipped while dropping controller-authored `planned_replay_traces`" (the record now says that outright). Its gate is the full step-3 gate plus `--test test_cellular --test test_graph_cellular --test test_grpc_cellular --test test_recorded_agent_cellular --test test_cellular_dataset_shipping`, closing: "A step-4 change that has not run `test_graph_cellular` and `test_recorded_agent_cellular` is not verified, regardless of what else is green."

One scoping change from the author's greenfield correction (commit 084f8f44d2): "byte-exact against the mock server" is now pinned in Purpose to mean **observed run output**, never stdin bytes. That makes the gate you asked for actually checkable rather than a claim about a wire nobody reads.

## O13 — §5.4's “lossy endpoint/model transforms” audit is stale. Current `Endpoint` already stores `timeout_seconds` and has no `url_strategy`; YAML consumes/validates `url_strategy` before typed resolution. Current `ModelItem` has only `name` and `weight`. Therefore `endpoint_profile`'s removals and `models_from_config`'s retain are no-ops on today's typed model, not migration hazards that need reproducing or factory ownership.
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:318-321` says today's projection performs lossy transforms: `rename `timeout` → `timeout_seconds`, drop `url_strategy`, retain only `name`/`weight``.
  
  Current typed source model:
  - `rust/runtime/src/config/model/endpoint.rs:136-138` already declares `pub timeout_seconds: f64`; there is no `timeout` field.
  - `rg -n "url_strategy|pub timeout:" rust/runtime/src/config` returns no matches.
  - `rust/cli/src/yaml.rs:1670-1679` validates authored `endpoint.url_strategy` and consumes it before constructing the typed endpoint; unsupported values fail, and the only supported value is honored by construction.
  - `rust/runtime/src/config/model/models.rs:19-27` shows `ModelItem` contains exactly `pub name: String` and `pub weight: Option<f64>`—there are no extra typed model-item fields to drop.
  
  Projection helpers:
  - `rust/runtime/src/engine/protocol_v2.rs:558-561` still calls `profile.remove("timeout")` and `profile.remove("url_strategy")`, but neither key can be emitted by current `Endpoint` serialization.
  - `rust/runtime/src/engine/protocol_v2.rs:512-516` retains `name|weight`, exactly the complete current `ModelItem` field set.
  
  So hazard (a) describes pre-unification shapes, not current runtime behavior. It must be removed or replaced with actual type deltas; otherwise step 4 directs implementers to preserve nonexistent transforms and obscures real compatibility work.
resolution (r4): Right on the substance, and I have removed the stale text — but the conclusion "these are no-ops on today's typed model" overreaches by one case, and that case is the only part of the hazard worth keeping. REFINE rather than RETRACT for that reason.

**Confirmed, independently.** `Endpoint` (`config/model/endpoint.rs:117`) declares `pub timeout_seconds: f64` at `:137` and has no `timeout` field, so `profile.remove("timeout")` can never fire on it. `url_strategy` does not exist under `runtime/src/config/` at all — it lives only in the authoring layer (`cli/src/flags.rs:368`, `cli/src/yaml.rs:898`) and is consumed and validated at `yaml.rs:1673` before typed resolution. `ModelItem` (`config/model/models.rs:22`) is exactly `name` + `weight`, so `models_from_config`'s `retain(|k,_| matches!(k, "name"|"weight"))` (`protocol_v2.rs:512-516`) retains the complete field set. Three no-ops, correctly identified. The text described pre-unification shapes, and I did not re-derive it when I edited the surrounding paragraph.

**Where it overreaches.** `endpoint_profile` is not called once. `protocol_v2.rs:545-547`:

```rust
profiles.push(raw_value(endpoint_profile("default", default)?)?);
for (id, config) in additional {
    profiles.push(raw_value(endpoint_profile(&id, config)?)?);
}
```

The `default` argument is `serde_json::to_value(&cfg.endpoint)` (`:446`) — typed, and your no-op finding holds for it. But `additional` is `cfg.endpoint_profiles` (`:448-451`), and that field is declared at `config/model/config.rs:118` as:

```rust
pub endpoint_profiles: serde_json::Map<String, serde_json::Value>,
```

An **open, untyped map** — not a typed section. Config-model-unification did not reach it. An authored override profile can therefore still carry `timeout` and `url_strategy` keys, and both the rename and the removal are live for it. So the transform is not stale for `endpoint_profiles`; it is the last thing keeping override profiles consistent with the default one.

Artifact changed (commit fb57400724): hazard (a) is rewritten. It now opens by naming the earlier "lossy transforms the migration must reproduce" framing as an overstatement, proves all three no-ops against the typed model with the anchors above, states the default profile and models may simply drop them — and then carves out `endpoint_profiles` as the live exception with its `config.rs:118` type and its `:448-451`/`:547` call path, requiring the migration to either keep the transform on that open map or type it as `BTreeMap<String, Endpoint>`, noting the latter is a larger change than this record scopes.

Net: your objection deleted two-thirds of a false hazard and, by forcing me to read the call sites instead of the helper, surfaced an untyped field the record had never mentioned.

