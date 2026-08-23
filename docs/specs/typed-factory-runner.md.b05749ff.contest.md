# Contest ledger

> Do not edit — generated from the contest tables after each round.

- **kind:** spec_off
- **status:** capped
- **round:** 8 / 8
- **artifact:** `docs/specs/typed-factory-runner.md`
- **low-friction:** yes — zero retractions and zero contested objections; unanimity is as consistent with correlated bias as with correctness, so this exchange is flagged UNVALIDATED rather than clean
- **persuasion-override rate (CW-POR):** 0.00 — the fraction of author retractions that answered an UNPROVEN objection; a high value means the author yielded to confident assertion rather than to evidence

## Seats

- **author** — `opus` (family `claude-trace`, lens —)
- **skeptic** — `gpt-5.6-sol` (family `claude-codex`, lens correctness — verify every code claim mechanically against the tree; preserved-behavior gaps in the migration)

## Objections

## O1 — `Resolved` does not require its `Option` keys, so the claimed tightening and test obligation are false as written
severity: high   raised: r1   status: refined
proven: yes
evidence:
  The artifact says at `docs/specs/typed-factory-runner.md:260-266` that a present `resolved` object missing “any one of its sixteen keys — including the `Option` ones, which without `serde(default)` are required-but-nullable — ... is rejected after”, and at `docs/specs/typed-factory-runner.md:865-866` requires “a `resolved` object present but missing one key is rejected”. The target type contains optional fields such as exact declaration `pub comm_config: Option<serde_json::Value>,` at `rust/runtime/src/config/model/resolved.rs:16`, `pub dataset_file_paths: Option<Vec<String>>,` at `rust/runtime/src/config/model/resolved.rs:18`, and `pub total_expected_duration: Option<f64>,` at `rust/runtime/src/config/model/resolved.rs:44`. Serde implements the opposite behavior explicitly: `/home/anthony/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/serde-1.0.228/src/private/de.rs:22-24` says `If the missing field is of type Option<T> then treat is as None, otherwise it is an error.`, and `deserialize_option` calls `visitor.visit_none()` at `:45-50`. Thus retyping to `Resolved` rejects omission only of non-`Option` fields such as exact declarations `pub artifact_dir_created: bool,` (`resolved.rs:14`) and `pub gpu_telemetry_mode: String,` (`resolved.rs:38`); it does not enforce the module comment’s every-key-present contract. The spec must either narrow the claim/test to a required non-`Option` key or require a custom decode that enforces presence of nullable keys.
resolution (r2): You are right and the claim was wrong, not merely imprecise. I read `config/model/resolved.rs:13-46` again: sixteen fields, **fourteen** of them `Option<...>`; only `artifact_dir_created: bool` and `gpu_telemetry_mode: String` are not. A derived `Deserialize` defaults a missing `Option` field to `None` without `#[serde(default)]`, so "missing any one of its sixteen keys is rejected" was false for fourteen of them. I had argued from the module doc ("Every field is present in the wire object, including nulls") plus the absence of per-field `serde(default)` — the doc describes the producer, not the decoder, and that is exactly the reading-attributes-instead-of-decoding failure I flagged as a weakness in my own brief.

Rewritten. The bullet now states the actual narrowing: exactly three cases — omitting `artifact_dir_created`, omitting `gpu_telemetry_mode`, or a wrong JSON type in any field. It also records that `Resolved` carries no `deny_unknown_fields`, so extra keys still pass, which further shrinks the delta against `resolved: Value`. The test obligation is withdrawn: the two mandatory keys are enforced by the compiler-generated decoder and no e2e run omits them. The step-4 gate bullet is retargeted to assert the *shape* of the narrowing (missing `artifact_dir_created` rejected, missing an `Option` field still decodes) rather than "missing one key is rejected". Committed as 49a3f0b79c.

## O2 — The §2 port list omits `parse_hop_routing` and the live `run.hop_routing` repoint
severity: medium   raised: r1   status: refined
proven: yes
evidence:
  `BenchmarkRunWireV2::into_authored` calls exact symbol `parse_hop_routing(&runtime)?` at `rust/runtime/src/engine/protocol_v2.rs:387`, stores the result as exact field `hop_routing` at `protocol_v2.rs:495`, and `build_common_plan`'s request construction consumes exact expression `hop_routing: run.hop_routing,` at `rust/runtime/src/engine/online_execution.rs:1759`. The typed source is exact field `pub hop_routing: Option<HopRouting>,` at `rust/runtime/src/config/model/runtime.rs:35`; AIPerf resolution actively derives it via exact symbol `resolve_hop_routing(inputs.runtime_hop_routing, resolved_connection_reuse)` at `rust/runtime/src/config/resolve.rs:859-860` and writes `hop_routing: Some(resolved_hop_routing),` at `resolve.rs:1397`. Yet the artifact’s claimed exhaustive summary at `docs/specs/typed-factory-runner.md:774-781` enumerates `workload_kind`, `worker_count`, `planned_replay_traces`, `parse_dispatch_mode`, mandatory sections, tokenizer, metrics, export stems, `resource_presence`, and workers/phases but never `parse_hop_routing` or `hop_routing`; the Step 4 repoint inventory at `:746-772` also never identifies `online_execution.rs:1759`. Deleting `AuthoredRunSpecV2` deletes the current carrier, so Step 4 must explicitly repoint `run.hop_routing` to `run.cfg.runtime.as_ref().and_then(|runtime| runtime.hop_routing)` (or an equivalent typed derivation) and preserve the existing tests currently under disabled `#[cfg(any())] mod tests`, including exact symbols `runtime_hop_routing_absent_is_none`, `runtime_hop_routing_parses_sticky`, and `runtime_hop_routing_rejects_unknown_variant` at `protocol_v2.rs:1732`, `:1740`, and `:1767`.
resolution (r2): Fair on completeness, and I settled it rather than listing it as an open item. `parse_hop_routing` (`protocol_v2.rs:283-290`) is the sibling of `parse_dispatch_mode` and the trivial case: `Option<HopRouting>` in and out, absent stays `None`, an unrecognized string is a hard error. `Runtime.hop_routing` is already `Option<HopRouting>` (`config/model/runtime.rs:36`), so the typed decode reproduces every one of those behaviors including the rejection — nothing is at risk. The work is repointing its single consumer, `hop_routing: run.hop_routing` (`online_execution.rs:1759`), at `cfg.runtime.hop_routing`, and accepting that the error text loses the `run.cfg.runtime.hop_routing:` path prefix the manual `map_err` adds.

Added as its own entry immediately after the dispatch one, stated as a repoint with no behavior at risk beyond that message prefix, and explicitly noting it is listed because its sibling is — an omission next to a listed sibling reads as an oversight rather than a decision. Committed as 49a3f0b79c.

## O3 — Moving the port list into `BenchmarkRun::validate()` leaves the cellular-controller path unvalidated before it consumes and mutates the run
severity: high   raised: r1   status: refined
proven: yes
evidence:
  The artifact assigns mandatory-section, tokenizer, metrics, workers/phases, and cardinality behavior to exact symbol `BenchmarkRun::validate()` (`docs/specs/typed-factory-runner.md:280-306`, `:321-323`, `:362-374`, `:393-396`) and Step 4 only says to “point `coordinator.rs` at `BenchmarkRun`” (`:737-744`). But controller promotion bypasses `Coordinator::handle`: `rust/cli/src/execute_mode.rs:115-129` calls exact symbol `resolved_envelope_from_input(&input)` and then `run_controller(&wrapped)`; `run_controller` calls exact symbol `cellular_controller::run_cellular` at `execute_mode.rs:291-297`. `resolved_envelope_from_input` does only `resolved_run_bytes(input).ok()?`, `serde_json::from_slice(&bytes).ok()?`, and `serde_json::json!({ "run": run })` at `rust/runtime/src/engine/cell_launcher.rs:346-351`; there is no `BenchmarkRun::validate()` call. The controller then consumes the unvalidated envelope before any cell coordinator: exact call `let metrics_config = cellular_metrics_config(envelope)?;` at `rust/runtime/src/engine/cellular_controller.rs:862`, after multiple envelope-derived preparation decisions. Worse, `cellular_metrics_config` directly decodes `/run/cfg/metrics` using exact expression `.map(|value| serde_json::from_value(value).unwrap_or_default())` at `cellular_controller.rs:2846-2850`, preserving the very silent metrics fallback that the artifact requires `BenchmarkRun::validate()` to eliminate. `cellular_will_use_exact_fold` also treats errors as false via exact expression `cellular_metrics_config(envelope).is_ok_and(|config| {` at `cellular_controller.rs:2225`. Therefore the migration as specified does not establish one validated typed run before cellular controller policy is derived; invalid metrics can be defaulted for controller merge/storage decisions and all other semantic port checks occur only later inside launched cells. Step 4 needs a typed decode + `BenchmarkRun::validate()` before `run_controller`, and `cellular_metrics_config` should consume the already validated typed metrics (or propagate conversion errors) rather than re-decode a `Value` with `unwrap_or_default()`.
resolution (r2): Valid as a gap in the record, though the verification lands somewhere more specific than "leaves the path unvalidated": that path is unvalidated *today too*, by the same mechanism. Traced it: `cli/src/execute_mode.rs:113-129` selects the controller from `resolved_envelope_from_input` on a raw `serde_json::Value`, **before** `decode_execute_wire` at `:464`. `build_cell_envelope` mutates that `Value` (`cellular_controller.rs:1947-2010`). No `BenchmarkRunWireV2` is ever constructed on that path, so `validate_outer()` (`coordinator.rs:138`) and `into_authored` (`coordinator.rs:159`) do not run in the controller now. What holds the line before and after is that each cell revalidates its own envelope on the ordinary `run_v2` path.

So moving the port list into `BenchmarkRun::validate()` neither adds nor removes controller coverage — but the record said none of that, and a reader would reasonably infer `validate()` is a boundary every run crosses. Added a paragraph stating the mechanism and two obligations it implies: the migration must not assume `validate()` has run when the controller mutates an envelope, and it must not opportunistically re-point the controller at typed `BenchmarkRun` as a side effect of this change — that would put a validation boundary in front of the mutation for the first time and change which runs fail and where, which is a separate step. Committed as 49a3f0b79c.

## O4 — The promised `variables` ordering/duplicate-key confirmation is absent, and the accepted map types demonstrably have different ordering
severity: low   raised: r1   status: refined
proven: yes
evidence:
  The artifact says exact text “`variables`' ordering and duplicate-key behavior under `serde_json::Map` is a detail step 4 should confirm rather than assume” at `docs/specs/typed-factory-runner.md:249-252`, but the complete named Step-4 unit-test list at `:858-885` contains no `variables` test and no duplicate-key test. The behavior is not equivalent by construction in this crate: current exact field `pub variables: BTreeMap<String, Value>,` is at `rust/runtime/src/engine/protocol_v2.rs:328`, while target exact field `pub variables: serde_json::Map<String, serde_json::Value>,` is at `rust/runtime/src/config/model/run.rs:47`. `rust/runtime/Cargo.toml:110` enables exact feature list `features = ["preserve_order", "raw_value"]`; serde_json’s implementation says “Enable the `preserve_order` feature of serde_json to use `IndexMap` instead” at `/home/anthony/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/serde_json-1.0.149/src/map.rs:2-3` and selects exact alias `type MapImpl<K, V> = IndexMap<K, V>;` at `map.rs:34-35`. Its own test identifies exact outcomes `Sorted order` (`["a", "b", "c"]`) versus `Insertion order` (`["b", "a", "c"]`) at `serde_json-1.0.149/tests/map.rs:3-14`. Thus ordering does change, and the spec neither states whether that observable serialization change is accepted nor supplies the confirmation it explicitly promises; duplicate-key handling is likewise not covered. Add the named test and record the decision/result.
resolution (r2): Correct that deferring it was the wrong move, so I resolved it instead of restating the deferral. Two findings:

1. The types do differ, and by more than a coin flip: `serde_json` is built with `preserve_order` in this workspace (`runtime/Cargo.toml:110`), so `serde_json::Map` iterates in insertion order while `BTreeMap` iterates sorted. Your ordering point stands on the fact.
2. It is nevertheless inert, because `variables` has no consumer anywhere. It is decoded onto `BenchmarkRunWireV2` (`protocol_v2.rs:326-328`) and read nowhere: `into_authored` never touches it, and `AuthoredRunSpecV2` has no `variables` field (checked the full field list at `:576-606`). Nothing downstream can observe either order. Duplicate keys are last-wins in both types.

Rewrote the bullet to state both, concluding the swap is free and no step-4 test is owed — which is a stronger resolution than the test I was about to promise. Worth noting the same `preserve_order` fact is what makes your O5 bite, where there *is* a consumer: same underlying mechanism, opposite verdict, and the difference is entirely whether anything reads the map. Committed as 49a3f0b79c.

## O5 — The port list misses `endpoint_profiles` key sorting, which feeds observable profile/report order
severity: medium   raised: r1   status: refined
proven: yes
evidence:
  `BenchmarkRunWireV2::into_authored` does not pass the open map through in authored insertion order: exact expression `cfg.endpoint_profiles.into_iter().collect::<BTreeMap<_, _>>()` appears at `rust/runtime/src/engine/protocol_v2.rs:448-451`, then exact loop `for (id, config) in additional` appends profiles after `default` at `protocol_v2.rs:545-547`. The artifact discusses this path at `docs/specs/typed-factory-runner.md:464-471` only to preserve exact symbol `endpoint_profile`'s `timeout` → `timeout_seconds` / `url_strategy` transform; it never records the `BTreeMap` sort or decides whether to preserve it. This is observable beyond lookup: exact method `EndpointProfilesSpecV2::identities` promises “in authored order” and iterates the vector at `protocol_v2.rs:899-904`; `RunContext::report_run_metadata` iterates `self.endpoint_profiles` into report identities at `rust/runtime/src/engine/registry.rs:1221-1237`; and exact method `RunContext::endpoint_profiles` says it preserves sequence for “prepared endpoint tables and report run metadata” at `registry.rs:1262-1271`. The target field is exact declaration `pub endpoint_profiles: serde_json::Map<String, serde_json::Value>,` at `rust/runtime/src/config/model/config.rs:118`, and this crate enables `serde_json` feature `preserve_order` at `rust/runtime/Cargo.toml:110`, making it an insertion-ordered `IndexMap` (`serde_json-1.0.149/src/map.rs:34-35`). A direct typed repoint therefore changes additional-profile order from lexicographic to input insertion order, changing report metadata and prepared iteration order. §2’s claim that all non-copy content is enumerated (`typed-factory-runner.md:774-781`) is false until this transform is included and either preserved or explicitly accepted as a behavior change with a test.
resolution (r2): Confirmed and added. `into_authored` does `cfg.endpoint_profiles.into_iter().collect::<BTreeMap<_, _>>()` (`protocol_v2.rs:448-451`), and `endpoint_profiles` (`:540-550`) pushes `"default"` first then the rest in that sorted order. Because `serde_json` is built with `preserve_order` in this workspace (`runtime/Cargo.toml:110`), iterating `cfg.endpoint_profiles` directly yields authored order, so the `BTreeMap` collect is a real normalization rather than an incidental type choice — I had treated the whole `endpoint_profiles` handling as "keep the `endpoint_profile` transform" and missed that the ordering is a second, separate behavior riding the same lines.

Observability checked rather than assumed: profile position is the profile's identity downstream — `ValidatedEndpointProfileV2`s are indexed by `enumerate()` position (`registry.rs:1124`) and resolved back by index (`registry.rs:1244`), and the accessor is documented as "authored order" (`registry.rs:1262`). Dropping the sort reorders indices for every multi-profile run.

Added as its own port-list entry directly after the endpoint-transform paragraph, with the obligation stated as "sort the override keys at the same boundary, `default` still first", plus a step-4 gate asserting a config that authors two overrides in reverse-sorted key order still emits them sorted with the rename/removal applied. Committed as 49a3f0b79c.

## O6 — The claimed current cellular-aware `Sharded` default is unreachable in actual cell envelopes because the controller rewrites `runtime.cells` to `1`
severity: high   raised: r1   status: refined
proven: yes
evidence:
  The artifact says exact text “when it is absent the default branches on `runtime.cells`: `cells > 1` resolves `DispatchMode::Sharded`” and that a typed reader would switch “every cellular run” from `Sharded` to `Global` at `docs/specs/typed-factory-runner.md:398-418`. `parse_dispatch_mode` indeed branches on exact expression `let cells = runtime.get("cells").and_then(Value::as_u64).unwrap_or(1);` and returns `DispatchMode::Sharded` for `cells > 1` at `rust/runtime/src/engine/protocol_v2.rs:260-268`. But the actual controller’s exact symbol `build_cell_envelope` rewrites each cell payload before execution: `runtime.insert("cells".to_owned(), serde_json::Value::from(1));` at `rust/runtime/src/engine/cellular_controller.rs:1989-1995`. There is no corresponding insertion of `runtime.dispatch` in `cellular_controller.rs` or `cellular_cell.rs` (search: `rg -n '"dispatch"' rust/runtime/src/engine/cellular_controller.rs rust/runtime/src/engine/cellular_cell.rs` returns no matches). Each cell later executes the bare run through `run_v2` (`rust/cli/src/execute_mode.rs:192-194` describes that boundary), so absent dispatch reaches `parse_dispatch_mode` with `cells == 1` and resolves `DispatchMode::default()` (`Global`), not `Sharded`. The cited tests `runtime_dispatch_defaults_to_sharded_for_cellular` construct a wire with `cells = 4` directly and do not pass through `build_cell_envelope`, which is why they do not prove cellular behavior. Either the migration intentionally changes actual cellular default to `Sharded`—in which case it must be recorded as a behavior change and derive/store dispatch before the rewrite—or the claimed preserved behavior and performance-regression rationale must be removed. Add a controller-envelope test, not only a direct `parse_dispatch_mode` test.
resolution (r2): Proven, and it kills the strongest claim in the section — which I committed one round earlier off the previous contest's O12. Verified end to end:

- `cli/src/execute_mode.rs:113-129`: a run with `cells > 1` is promoted to the cellular controller *before* `decode_execute_wire`; the controller is not an issuer.
- `cellular_controller.rs:1149-1150` is the single `build_cell_envelope` call site, and `:1994` is `runtime.insert("cells", 1)` — unconditional, applied to every cell envelope before serialization, same-host and cross-host alike (cross-host cells fetch that same serialized envelope via `cellular_cell.rs:712`).
- The only consumer of the resolved value is `dispatch_mode: run.dispatch` (`online_execution.rs:1758`), in the executing process, which therefore always parses `cells == 1` and always lands on `Global`.
- Nothing in `cellular_controller.rs` or `cell_launcher.rs` reads `.dispatch` at all.

So the `cells > 1 → Sharded` arm is unreachable at execution time, and my "silently switches every cellular run from Sharded to Global, a large performance regression" was wrong: `unwrap_or_default()` changes no executing run's dispatch mode. What it does change is the controller-side projection and the CLI/runner parity assertion at `cli/src/profile.rs:1669-1675`.

Rewritten to say that plainly, including a sentence marking the earlier claim as this record's own error. The derivation is still ported, but as a projection detail rather than a performance guard. I also recorded the live gap the objection exposes — the intended cellular default is unreachable, so the ~7-8x c4-144 measurement does not describe what cells run today — and scoped it out explicitly, because changing it is a runtime behavior change and not a typing migration; the point of recording it is to stop the migration laundering it into a "preserved behavior" it never had. The step-4 gate bullet now notes the three stranded tests assert the projection, not cell behavior. Committed as 49a3f0b79c.

## O7 — The end-to-end typed / “no `Value` round-trip” headline contradicts the explicitly retained open `endpoint_profiles` transform
severity: medium   raised: r1   status: refined
proven: yes
evidence:
  The Purpose claims exact target “the config payload typed end to end: no `Value` round-trip, no per-factory re-decode” at `docs/specs/typed-factory-runner.md:23-26`, and `Built` contrasts Python’s “zero projection and zero `RawValue`” at `:109-112`. But the normative target explicitly retains exact field `cfg.endpoint_profiles` as “an open `serde_json::Map<String, Value>`, not a typed section” and requires keeping exact symbol `endpoint_profile`’s key-level transform at `typed-factory-runner.md:464-471`. The source confirms exact target declaration `pub endpoint_profiles: serde_json::Map<String, serde_json::Value>,` at `rust/runtime/src/config/model/config.rs:118`; current projection converts it through `collect::<BTreeMap<_, _>>()` and exact function `endpoint_profiles(endpoint, additional_profiles)?` at `rust/runtime/src/engine/protocol_v2.rs:448-451` and `:487`, while `endpoint_profile` performs `as_object().cloned()`, `insert`, `remove`, and returns `Value::Object(profile)` at `protocol_v2.rs:552-562`. This is literally a retained `Value` projection/transform in the config payload. It may be a justified non-goal, but the artifact must scope its headline to built-in transport/workload component configs (or list open endpoint overrides as a third residual untyped seam); as written, the target guarantee is false by its own normative design.
resolution (r2): The contradiction is real as written. "No `Value` round-trip" in Purpose reads as an absolute, and §2 then deliberately retains the `endpoint_profile` transform over the open `serde_json::Map` at `config/model/config.rs:118`. Two claims, one document.

Rewrote the headline to say what the change actually delivers: every **typed** section carried typed end to end, i.e. elimination of the typed→`Value`→`RawValue`→typed round-trip the projection creates. Added an explicit paragraph naming what stays open and why — `endpoint_profiles` is untyped *at its source* on `BenchmarkConfig`, so this is not a round-trip being tolerated but a section that was never typed; `cfg.failure_policy`'s `Option<Value>` is the same case; and the two residual `RawValue` uses are already under Non-goals. Typing `endpoint_profiles` as `BTreeMap<String, Endpoint>` is named as the separate change it would be. The distinction the record now draws is that it eliminates the round-trip *it created itself*, and claims nothing about `Value` that was authored as `Value`. Committed as 49a3f0b79c.

## O8 — The proposed typed `weka_semantics` decoder drops current trim and empty-string behavior
severity: medium   raised: r1   status: refined
proven: yes
evidence:
  The target says exact normalization for `weka_semantics` is “lower + `-`→`_` only” and mentions aliases `graph-ir`, `graphir`, and `graph_ir` at `docs/specs/typed-factory-runner.md:148-152`; it does not preserve whitespace trimming or the empty spelling. Current exact symbol `weka_wants_legacy` instead matches `semantics.map(|s| s.trim().to_ascii_lowercase()).as_deref()` at `rust/runtime/src/engine/online_execution.rs:287-290` and explicitly accepts `Some("")` as the graph-ir path while accepting whitespace-padded values after `.trim()`. Therefore authored values such as exact JSON string `" legacy "` currently resolve `Ok(true)`, and `""` currently resolves `Ok(false)`; a closed derived enum normalized only by lowercase and `-`→`_` rejects both. The artifact neither records this as a behavior change nor adds tests for it. The design must decide explicitly: preserve `.trim()` and the empty alias in the typed decoder, or state that both accepted inputs are intentionally tightened and test their rejection.
resolution (r2): Proven; the bullet was wrong twice over. Read `weka_wants_legacy` (`online_execution.rs:287-294`): it folds `semantics.map(|s| s.trim().to_ascii_lowercase())` and *then* matches `None | "" | "graph-ir" | "graphir" | "graph_ir"` → graph-ir, `"legacy" | "agentx"` → legacy, anything else → `"unknown weka semantics {other:?}; expected 'legacy' or 'graph-ir'"`.

Two errors on my side. First, `#[serde(alias)]` cannot express any of this — aliases are exact byte matches on the wire string, so they reproduce neither the trim nor the case fold, and `""` would become a decode error instead of graph-ir. Second, my stated fold ("lower + `-`→`_` only") does not even describe the code, which does lower + trim and enumerates all three spellings literally; the appeal to what Python does was irrelevant, since the Rust decoder is the thing being replaced.

Rewritten to enumerate the three dropped behaviors — trim (`" legacy "` accepted today), case fold (`"Legacy"` accepted today), and empty-string-means-graph-ir — and to state the actual requirement: a hand-written `Deserialize`/`FromStr` reproducing the fold, the error message carried verbatim, and `Option<WekaSemantics>` with absent and blank both meaning `GraphIr`. Added a step-4 gate asserting `" Legacy "`, `"AgentX"`, `""`, absent, and an unknown value. Committed as 49a3f0b79c.

## O9 — The “exactly three cases” account of `Resolved` narrowing omits every non-object JSON value accepted by the current `Value` field
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  The revised artifact says at `docs/specs/typed-factory-runner.md:294-297` that the narrowing is “exactly three cases”: omitting `artifact_dir_created`, omitting `gpu_telemetry_mode`, or giving a field the wrong JSON type, followed by “Everything else that decodes as `Value` today still decodes.” Current `BenchmarkRunWireV2` declares exact field `pub resolved: Value,` at `rust/runtime/src/engine/protocol_v2.rs:302-304`, so `null`, arrays, strings, booleans, and numbers all decode today. Target `BenchmarkRun` declares exact field `pub resolved: Resolved,` at `rust/runtime/src/config/model/run.rs:43-45`; `Resolved` is a derived struct (`rust/runtime/src/config/model/resolved.rs:12-13`) and therefore requires a JSON object when the field is present. Thus e.g. `"resolved": null` and `"resolved": []` are additional rejected classes not described by any “value for any field has the wrong JSON type” case (they have no fields). The follow-on “Everything else” claim is false. The port-list and step-4 test should state/test the top-level object narrowing as well as mandatory member omission.
resolution (r4): Valid. `resolved: Value` is total over JSON — `3`, `"x"`, `[]`, `true`, and explicit `null` all decode today; `Resolved` rejects every one. And `#[serde(default)]` covers an absent key, not an explicit `null`, so `"resolved": null` goes from `Value::Null` to a decode error. "Exactly three cases" enumerated only the object-shaped narrowings.

Artifact now splits the narrowing into a non-object part (naming the whole class, including the explicit-null case and why `default` does not cover it) and the object part (the two missing-field cases plus wrong-typed-field). Same defect class as O1 from the prior round: I reasoned about field attributes instead of about what the decoder accepts.

## O10 — The claimed two residual `RawValue` uses omit the live sidecar adapter boundary, which the migration never removes or repoints
severity: high   raised: r3   status: refined
proven: yes
evidence:
  Purpose says the opaque `RawValue` seam is “confined to the two residual uses” at `docs/specs/typed-factory-runner.md:23-26`, and Non-goals enumerates exactly those two as dynosim nested args and the dataset payload at `:1055-1062`. The source has a third first-class seam: `rust/runtime/src/engine/sidecar_input.rs:6-8` states “Sidecar bodies remain raw JSON until the selected adapter performs strict decoding”; exact struct `AuthoredSidecarInput<'a>` carries exact field `pub config: &'a RawValue` at `sidecar_input.rs:274-278`; exact trait method `fn validate(&self, raw: &RawValue)` is at `:312-317`; and exact resolver method `fn prepare(&self, authored: &[AuthoredSidecarInput<'_>])` is at `:321-323`. The current projection constructs those raw bodies through `SidecarSpecV2::authored_inputs` at `rust/runtime/src/engine/protocol_v2.rs:1127-1143`, and `coordinator.rs:212-217` calls `run.sidecars.authored_inputs()` then `self.sidecar_inputs.prepare(&authored_sidecars)`. Yet the Migration discusses transport/workload factory typing and deleting `AuthoredRunSpecV2`/`NamedRunnerComponentSpecV2`; it never changes `SidecarInputAdapter`, `AuthoredSidecarInput`, or `SidecarInputAdapterResolver`. This cannot be dismissed as source-untyped data: `cfg.sidecars` is already typed as `Option<Sidecars>` at `config/model/config.rs:109-111`, and `Sidecars` has typed built-in fields at `config/model/telemetry.rs:214-230`, so the existing sidecar path is precisely another typed→`Value`/`RawValue`→typed round-trip the headline promises to eliminate. Either the migration must add and verify a sidecar-adapter typing/repoint step, or the Purpose/Non-goals must list this third surviving `RawValue` seam and stop claiming every typed section rides typed end to end.
resolution (r4): Valid, and the sharpest structural finding of the round. Verified: `SidecarSpecV2`'s five `Option<Box<RawValue>>` fields (protocol_v2.rs:1105-1124) → `authored_inputs()` (:1132) → `AuthoredSidecarInput { id: &str, config: &RawValue }` (sidecar_input.rs:273-279) → `SidecarInputAdapter::validate(&RawValue)` (:311-318, five impls). That is the same id-plus-opaque-body shape §1 removes for transports, reached from a typed `Sidecars`, and the migration never touches it.

Artifact changes: Non-goals now lists three residual `RawValue` uses, distinguishing the two opaque leaves (dynosim nested args, dataset payload) from this one, which is a live open seam rather than a leaf. New §2 port entry states (a) the §3/§4 "step 4 deletes the `{RegistryId, RawValue}` seam" claim is true of `NamedRunnerComponentSpecV2` only, (b) feeding the adapters post-projection still requires typed `Sidecars` → `RawValue`, and selects re-serialize-and-say-so over re-pointing five adapters, and (c) `live_streaming` exists on the DTO but not on typed `Sidecars`, so it is already unreachable and always `None`.

Also recorded, since it cuts the other way: `SidecarSpecV2::validate_outer`'s "must be a JSON object" checks are dead in production — the only caller is `AuthoredRunSpecV2::validate_outer` (:722-751), whose only callers are `runtime/tests/recorded_agent_protocol.rs:63`/`:90`; the reachable path is `EnvelopeV2` → `BenchmarkRunWireV2::validate_outer`. So those bodies are listed as deliberately dropped, not ported, with the note that a valid-JSON non-object body now fails inside the adapter's own strict decode instead.

## O11 — The tokenizer port list omits the live `trust_remote_code=true` warning emitted by the decode seam being deleted
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  The artifact inventories “Three tokenizer rejections that live in the DTO's decode” at `docs/specs/typed-factory-runner.md:427-447` and says all relevant non-copy `into_authored` behavior is enumerated at `:873-880`, but the same exact decode has a fourth observable behavior. `AuthoredTokenizerV2::decode` at `rust/runtime/src/engine/online_execution.rs:960-985` checks exact condition `if config.trust_remote_code` at `:972` and emits exact `tracing::warn!` at `:979-983`: `tokenizer.trust_remote_code=true has no effect: the native tokenizer never executes repository code; loading the tokenizer normally`. `Tokenizer` retains exact typed field `pub trust_remote_code: bool` at `rust/runtime/src/config/model/tokenizer.rs:25-27`, so direct typed consumption bypassing and deleting `AuthoredTokenizerV2::decode` silently removes this user-facing warning while continuing to accept the inert option. The migration must relocate the warn-once behavior to the typed preparation/validation boundary and add coverage, or explicitly record its removal as a behavior change; porting only the three rejection checks is incomplete.
resolution (r4): Valid. `AuthoredTokenizerV2::decode` (online_execution.rs:960-985) does three things, and I listed two: the two `ensure!` rejections plus a `tracing::warn!` when `trust_remote_code` is true, telling the operator the flag is inert because the native tokenizer never executes repository code. Deleting the decode deletes the warning silently — nothing fails, the operator just stops being told.

Artifact adds it as a fourth ported item, with one scoping correction the objection did not make: the flag is not inert everywhere. `lower` still forwards it as `resolver.resolve(&self.name, &self.revision, self.trust_remote_code)` (:1059), so the warning's claim is about repository-code execution specifically, and the ported text has to preserve that scope rather than restating "the flag does nothing."

## O12 — The three dispatch-default tests are already enabled; they are not stranded under `#[cfg(any())]` as the spec repeatedly claims
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  The artifact says at `docs/specs/typed-factory-runner.md:488-492` that exact tests `runtime_dispatch_defaults_to_global_when_absent`, `runtime_dispatch_defaults_to_sharded_for_cellular`, and `runtime_explicit_dispatch_wins_over_cellular_default` “sit inside the `#[cfg(any())] mod tests` at `protocol_v2.rs:1284` and never compile,” and repeats at `:978-982` that these are “the three tests currently stranded in `#[cfg(any())] mod tests`.” Source shows two separate modules. The disabled module begins at exact lines `#[cfg(any())]` / `mod tests` in `rust/runtime/src/engine/protocol_v2.rs:1284-1285` and ends at `:1408`. A new, enabled module begins at exact lines `#[cfg(test)]` / `mod dispatch_mode_tests` at `:1410-1411`. The three named tests are inside that enabled module at exact lines `:1658`, `:1667`, and `:1680`; additional enabled hop-routing tests are there at `:1732`, `:1740`, and `:1767`. Therefore the migration does not need to “move” or re-enable these tests, and the claimed current verification gap is false. Step 4 still needs replacement coverage after deleting the helper/DTO, but the document must distinguish already-running behavior pins from the genuinely disabled stale module.
resolution (r4): Correct, and the spec was flatly wrong in four places. No refining around this one.

Mechanically confirmed: the `#[cfg(any())] mod tests` spans protocol_v2.rs:1284-1408 and contains `outer_contract_rejects_unknown_fields`; `#[cfg(test)] mod dispatch_mode_tests` opens at :1410-1411 and holds the three dispatch tests at :1658/:1667/:1680. `cargo test -p aiperf-runtime --features engine runtime_dispatch` compiles and runs them. I asserted the opposite at four points (the §2 port entry, the step-4 obligation, the verification-gates passage, and the closing cleanup note); all four are corrected, and the port entry now says what the live tests actually pin — the projection, not cell behavior, which is precisely why they stayed green while the `Sharded` arm became unreachable at execution time (the O6 finding).

One part survives unchanged and is now scoped explicitly: `outer_contract_rejects_unknown_fields` really is dark, and re-enabling or deleting that module really is a step-4 obligation. The error was extending that module's disablement to the tests below it.

Root cause worth naming: I cited a `#[cfg(any())]` line number from a prior round and then reused it three more times without re-reading the module's extent. Two rounds running, my errors came from reasoning about a remembered anchor instead of decoding the current file.

## O13 — Step 1 component parity cannot prove Step 2’s replacement binding match, and the gate executes neither dynosim arm
severity: high   raised: r3   status: refined
proven: yes
evidence:
  The artifact concludes at `docs/specs/typed-factory-runner.md:692-710` that component equality pins the binding transitively and therefore “No binding-level differential is owed at any step.” That argument is valid only while the old exact implementation remains `resolve_native_execution` → `factory.native_execution(transport, context)` (`rust/runtime/src/engine/online_execution.rs:118-128`, quoted by the artifact at `:697-702`). Step 2 explicitly replaces that selection with new exhaustive match arms over the duplicated `transport_typed: Transport` carrier (`typed-factory-runner.md:715-734`) and itself says those arms must supply `NativeTransportExecution` bindings directly (`:739-743`). Equality of the old projected `{id, config}` pair proves nothing about whether newly written direct arms return the same binding; a match that accidentally maps `Grpc` to the HTTP binding, swaps `DynosimOffline`/`DynosimOnline`, or drops variant config compiles independently of Step 1’s assertion. The verification list at `:895-908` runs profile HTTP/gRPC, dry-run, WebSocket, and eval suites plus compile checks, but executes no dynosim suite at all. The source makes that omission behaviorally significant: exact IDs `DYNOSIM_OFFLINE_ID` and `DYNOSIM_ONLINE_ID` are distinct at `rust/runtime/src/engine/offline_execution.rs:95-96`, and comments at `:707-708` state that `online` is derived from the selected transport ID, selecting virtual versus wall clock. A feature-bearing compile check cannot detect swapped bindings. During Step 2’s intentional overlap, the old registry binding and new typed-match binding are both available and must be differentially compared for all six variants (or both dynosim arms must be executed in behavior tests); the Step 1 component assertion is not a substitute for testing the new selection implementation.
resolution (r4): Valid on both halves.

Transitivity: the step-1 argument holds only because both paths terminate in the same `factory.native_execution(transport, context)`, so equal `{id, config}` gives an equal binding. Step 2 obligation (a) deletes that call — the arms supply bindings directly — so component equality proves nothing about whether a hand-written arm reproduces the registry lookup. "No binding-level differential is owed at any step" generalized a step-1 property past the step that makes it true. Artifact now states the transitivity is exactly as wide as step 1, names the step-2 differential owed (arm vs `registry.transport_factory(id).native_execution(...)`, asserted while both are still reachable), and marks the old sentence as the overreach it was.

Coverage: confirmed no dynosim test exists — `grep -rl dynosim` over `e2e-tests/` and `dry-run-tests/` matches only a doc comment in `global_dispatch_real_clock.rs`. Step 2's gate named dry_run, websocket, harbor_native_graph, and the lean build, and skipped both `Dynosim*` arms, which are the two most exposed to a match rewrite: they carry `DynosimConfig` payloads with nested `RawValue` args, so unlike the unit variants they have a config half an arm can get wrong. Gate now says so, adds the `--features dynosim` binding differential over `all_variants()` as the minimum, and names a socket-free `dynosim_offline` profile run as the real gate still owed rather than pretending coverage exists.

Also fixed the internal inconsistency you flagged: "the only payload-bearing built-in transport arm any suite exercises" immediately followed by websocket. Now "the two payload-bearing arms any suite exercises today."

## O14 — Retyping unconditional `BenchmarkRun` to the existing `VariationSpec` crosses the feature-gated engine boundary; the required type relocation is unspecified
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  The artifact requires exact change “`variation: Value` → `Option<VariationSpec>` — re-type `BenchmarkRun`” at `docs/specs/typed-factory-runner.md:263-267` and Step 4 tests that typed field at `:963`, but never gives `VariationSpec` an unconditional home. The only exact definition is `pub struct VariationSpec` at `rust/runtime/src/engine/protocol.rs:276-287`. That module is unavailable in ordinary runtime builds: `rust/runtime/src/lib.rs:40-41` gates the entire exact module with `#[cfg(feature = "engine")] pub mod engine;`, while `pub mod config;` is unconditional at `lib.rs:48`, and `config/model/run.rs` (the target `BenchmarkRun`) is likewise exported unconditionally through `config/model/mod.rs:22,32`. Therefore `config::model::run::BenchmarkRun` cannot name the existing `crate::engine::protocol::VariationSpec` without breaking every `aiperf-runtime` build that omits `engine`; the core verification explicitly runs such a build via `cargo test -p aiperf-runtime` (`typed-factory-runner.md:884-893`). Step 4 must first move/define `VariationSpec` in an unconditional config/model module and repoint the engine protocol to that shared type (including serde and `BTreeMap` ordering semantics), or choose another unconditional type. Merely saying “re-type `BenchmarkRun`” leaves a compile-blocking dependency inversion out of the migration.
resolution (r4): Valid — a compile break, not a stylistic gap. `VariationSpec` is at engine/protocol.rs:279, inside `#[cfg(feature = "engine")] pub mod engine` (lib.rs:40-41); `BenchmarkRun` is in `pub mod config`, which is unconditional (lib.rs:48). Re-typing the field as written fails to build whenever `engine` is off — the default, and the exact `cargo test -p aiperf-runtime` invocation the gates run at every step. The record said "re-type `BenchmarkRun`" and stopped there.

Artifact now specifies the move: relocate `VariationSpec` into `config/model/` with `engine::protocol` re-exporting it so existing paths keep working, carrying `deny_unknown_fields` and the `BTreeMap<String, Value> values` field unchanged. Duplicating the struct is rejected explicitly — two `deny_unknown_fields` definitions of one wire shape is the drift this arc exists to remove. The README index row records the relocation too.

## O15 — The §2 port list omits `cfg.artifacts`' live all-or-nothing fallback in `BenchmarkRunWireV2::into_authored`, so deleting the projection can silently change artifact behavior. `rust/runtime/src/engine/protocol_v2.rs:453-457` converts typed `cfg.artifacts` to `ArtifactSpecV2` and calls `.unwrap_or_default()` on every decode error. That error path is reachable because `rust/runtime/src/config/model/artifacts.rs:10-15` declares `UserFile.format: String`, while `rust/runtime/src/engine/protocol_v2.rs:1083-1104` declares `UserFileSpecV2.format: UserFileFormatV2` with only `Json`, `Yaml`, and `Text`. Thus a bare resolved run can carry `cfg.artifacts.user_files[0].format = "bogus"`; `BenchmarkConfig` accepts it, `ArtifactSpecV2` rejects it, and `.unwrap_or_default()` silently discards the entire `artifacts` section. The record makes an explicit decision for the analogous `metrics: serde_json::from_value(metrics).unwrap_or_default()` at `protocol_v2.rs:489`, but makes none for artifacts; its claim at `docs/specs/typed-factory-runner.md:969-976` that all nontrivial `into_authored` branches are enumerated is false. The migration must either preserve this fallback deliberately or, preferably under the greenfield rationale already used for metrics and dataset truncation, surface the offending `artifacts.user_files[*].format` and add a named test.
severity: high   raised: r5   status: refined
proven: yes
evidence:
  Exact current-tree source:

  `rust/runtime/src/engine/protocol_v2.rs:453-457`:
  ```rust
  let artifacts_spec: ArtifactSpecV2 = serde_json::from_value(
      serde_json::to_value(&cfg.artifacts)
          .map_err(|error| anyhow!("run.cfg.artifacts: {error}"))?,
  )
  .unwrap_or_default();
  ```

  `rust/runtime/src/config/model/artifacts.rs:10-15`:
  ```rust
  #[derive(Clone, Debug, Serialize, Deserialize)]
  pub struct UserFile {
      pub path: String,
      pub format: String,
      pub content: String,
  }
  ```

  `rust/runtime/src/engine/protocol_v2.rs:1083-1104`:
  ```rust
  pub struct UserFileSpecV2 {
      pub path: String,
      pub format: UserFileFormatV2,
      pub content: String,
  }
  ...
  pub enum UserFileFormatV2 {
      Json,
      Yaml,
      Text,
  }
  ```

  The implementation branch retains the same behavior at `ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs:469-473`:
  ```rust
  let artifacts_spec: ArtifactSpecV2 = serde_json::from_value(
      serde_json::to_value(&cfg.artifacts)
          .map_err(|error| anyhow!("run.cfg.artifacts: {error}"))?,
  )
  .unwrap_or_default();
  ```

  The artifact's exhaustive claim is `docs/specs/typed-factory-runner.md:969-976`:
  ```text
  Of `into_authored`'s own content, only `workload_kind` and `worker_count` from
  `available_parallelism` are plain copies or derivations of the typed fields. The
  rest is not: ... All are enumerated in §2's port list.
  ```
  `cfg.artifacts`' fallback is not enumerated.
resolution (r6): Valid, verified in full, and it exposed a fourth omission I found while checking it.

Confirmed: `protocol_v2.rs:454-458` decodes `ArtifactSpecV2` from the serialized typed section and calls `.unwrap_or_default()`. The error path is reachable exactly as you describe — `UserFile.format: String` (`config/model/artifacts.rs:10-15`) against the closed `UserFileFormatV2 { Json, Yaml, Text }` (`protocol_v2.rs:1083-1104`). And the consequence is worse than "loses the section": every field of `ArtifactSpecV2` is `#[serde(default)]` (`:943-983`), so the run proceeds with no `records_path`, no `raw_path`, no `inputs_path`, `trace: false`, and no dry-run analysis, with no error emitted. A bogus user-file format silently produces a run that writes nothing.

Decision recorded matches your preference: do not port the fallback; `validate()` surfaces `artifacts.user_files[i].format must be one of json, yaml, text`, with a named step-4 test. Typing `UserFile.format` as the enum is noted as the better end state and compatible, but not required — this record only forbids the silent discard.

The finding I owe you back: there are **three** `.unwrap_or_default()` section fallbacks in `into_authored`, not two. `export` at `:459-463` is the third, and it is a different animal — `cfg.export` is `Option<Export>` (`config/model/export.rs:265-285`) and the target is a separately maintained `crate::export::ExportConfig` (`export/mod.rs:279-301`), not a stricter view of the same struct. They genuinely differ: `ExportConfig` has `timeslice`, `accuracy_csv`, and a `genai_perf.stem` that typed `GenaiPerf` has no field for — which is why the stem derivation exists at all. It is `#[serde(default, deny_unknown_fields)]`, so it decodes today only because every key `Export` emits happens to be known, and it does decode (otherwise exports would be dead and the stem overwrite unobservable). The failure mode is silent and total: one key added to `Export` and not to `ExportConfig` disables every exporter at once. Recorded with its own decision.

Your last sentence is the one that mattered most, and I acted on it beyond the artifacts entry: the claim at `:969-976` that all nontrivial branches were enumerated is deleted, not repaired. It was false when written, and it is the kind of sentence that stops the next reader from re-walking the function. In its place the record now states that three separate audits each missed a behavior of the same class — one living in the function body or in the shape handed to the factory rather than in a struct field — and puts a fresh statement-level walk of `into_authored` on step 4 as an obligation: every `?`, every `unwrap_or_default`, every `Map` `insert`/`remove`, every value constructed rather than copied. The port list is now explicitly a floor.

## O16 — The claim that the default endpoint-profile transform is a no-op and “may simply drop” is mechanically false: `endpoint_profile` injects the required profile identity. `rust/runtime/src/engine/protocol_v2.rs:545` calls `endpoint_profile("default", default)`, and `rust/runtime/src/engine/protocol_v2.rs:557` executes `profile.insert("id".to_owned(), Value::String(id.to_owned()))`. Typed `Endpoint` has no `id` field (`rust/runtime/src/config/model/endpoint.rs:115-174`), while `endpoint_profile_identity` requires `id` to be a non-empty string at `rust/runtime/src/engine/protocol_v2.rs:920-927`; `RunContext::default_endpoint_profile` then resolves the literal `"default"` at `rust/runtime/src/engine/registry.rs:1257-1260`. Therefore `docs/specs/typed-factory-runner.md:619-630` is wrong to characterize the default-profile transform as a no-op and authorize dropping it. The port list must explicitly retain or replace both identity derivations: inject `"default"` for `cfg.endpoint`, and derive/overwrite each override profile's `id` from its `cfg.endpoint_profiles` map key. Sorting alone does not supply identity.
severity: high   raised: r5   status: refined
proven: yes
evidence:
  Artifact claim, `docs/specs/typed-factory-runner.md:619-630`:
  ```text
  **Endpoint and model transforms — mostly no-ops, with one live exception.**
  `endpoint_profile` (`protocol_v2.rs:552-563`) renames `timeout` →
  `timeout_seconds` and removes `url_strategy`; ... For
  the **default** profile (`serde_json::to_value(&cfg.endpoint)`,
  `protocol_v2.rs:446`) and for models, the migration may simply drop these
  transforms.
  ```

  Exact current-tree source, `rust/runtime/src/engine/protocol_v2.rs:540-562`:
  ```rust
  fn endpoint_profiles(
      default: Value,
      additional: BTreeMap<String, Value>,
  ) -> Result<EndpointProfilesSpecV2> {
      let mut profiles = Vec::with_capacity(additional.len() + 1);
      profiles.push(raw_value(endpoint_profile("default", default)?)?);
      for (id, config) in additional {
          profiles.push(raw_value(endpoint_profile(&id, config)?)?);
      }
      Ok(EndpointProfilesSpecV2 { profiles })
  }

  fn endpoint_profile(id: &str, value: Value) -> Result<Value> {
      let mut profile = value
          .as_object()
          .cloned()
          .ok_or_else(|| anyhow!("run.cfg.endpoint must be an object"))?;
      profile.insert("id".to_owned(), Value::String(id.to_owned()));
      if let Some(timeout) = profile.remove("timeout") {
          profile.insert("timeout_seconds".to_owned(), timeout);
      }
      profile.remove("url_strategy");
      Ok(Value::Object(profile))
  }
  ```

  Required downstream identity, `rust/runtime/src/engine/protocol_v2.rs:917-927`:
  ```rust
  fn endpoint_profile_identity(profile: &RawValue) -> Result<EndpointProfileIdentityV2> {
      let object: Map<String, Value> = serde_json::from_str(profile.get())
          .map_err(|error| anyhow!("must be a JSON object: {error}"))?;
      let profile_id = object
          .get("id")
          .and_then(Value::as_str)
          .ok_or_else(|| anyhow!("id must be a string"))?;
      ensure!(
          !profile_id.is_empty() && profile_id.trim() == profile_id,
          "id must be non-empty and contain no surrounding whitespace"
      );
  ```

  Literal default lookup, `rust/runtime/src/engine/registry.rs:1257-1260`:
  ```rust
  pub fn default_endpoint_profile(&self) -> Result<&ValidatedEndpointProfileV2> {
      self.endpoint_profile("default")
  }
  ```

  `rust/runtime/src/config/model/endpoint.rs:115-174` declares `pub struct Endpoint` and contains no `id` field.
resolution (r6): Correct, and the spec authorized a change that would break every cellular and multi-profile run. Not refining around this one.

Verified end to end: `endpoint_profile`'s first statement is `profile.insert("id".to_owned(), Value::String(id.to_owned()))` (`protocol_v2.rs:557`), called as `endpoint_profile("default", default)` at `:545`. I dumped every field of typed `Endpoint` (`config/model/endpoint.rs:115-174`) — `urls`, `endpoint_type`, `streaming`, `timeout_seconds`, `connection_reuse`, `ssl_verify`, `uds_path`, `extra`, `headers`, and the rest — and there is no `id`. The literal `"default"` exists nowhere in the typed model; it is manufactured by that insert. Downstream it is required: `endpoint_profile_identity` (`:917-935`) fails `id must be a string` when absent, and additionally rejects empty or whitespace-padded values; `RunContext::default_endpoint_profile` (`registry.rs:1256-1259`) resolves the literal `"default"`. Dropping the transform yields profiles with no identity and fails every default-profile lookup.

I read the transform as rename-plus-removal because that is what I had characterized it as when analyzing the *override* path, and then carried "no-op against the typed model" back onto the default path without re-reading the function. Same failure mode as O12 last round: reasoning from my own earlier characterization instead of decoding the source.

Artifact now states both identity derivations as required ports — inject the literal `"default"` for `cfg.endpoint`, derive each override profile's `id` from its `cfg.endpoint_profiles` map key — and separates them explicitly from the sort, which orders profiles but does not name them. One detail your objection did not name and the record now carries: the insert *overwrites*, so an override body that already carries its own `"id"` key has it replaced by the map key today. That is the map key being promoted into the value, which no typed re-read reproduces, and it has to be preserved as an overwrite rather than a fill-if-absent.

## O17 — The chosen metrics behavior change is incomplete for cellular execution because the controller has an independent projection-bypassing decoder that the migration never repoints. `cellular_metrics_config` at `rust/runtime/src/engine/cellular_controller.rs:2845-2855` reads `/run/cfg/metrics` directly from raw `serde_json::Value` and repeats `serde_json::from_value(value).unwrap_or_default()`. It is called on the controller's startup path at `cellular_controller.rs:860-862`, and `cellular_will_use_exact_fold` calls it again at `cellular_controller.rs:2224-2231`. `docs/specs/typed-factory-runner.md:377-392` selects the opposite behavior—surface a non-numeric SLO instead of defaulting—and `:361-375` correctly says the controller never crosses `BenchmarkRun::validate()`, but the port list gives no obligation to change `cellular_metrics_config`. Consequently Step 4 can satisfy every listed `BenchmarkRun::validate()` test while the controller still silently constructs a default merge/storage policy from invalid authored metrics before cells independently reject. The migration must repoint `cellular_metrics_config` to the same strict typed conversion/validation or explicitly retain and justify a controller-only fallback; add a controller test for a non-numeric SLO, not only the unknown-SLO-name test currently at `cellular_controller.rs:4532-4552`.
severity: high   raised: r5   status: refined
proven: yes
evidence:
  Chosen behavior, `docs/specs/typed-factory-runner.md:377-392`:
  ```text
  **The metrics fallback — do *not* port as-is; surface the error.** Today an
  invalid metric SLO silently defaults the whole metrics section. ...
  `serde_json::from_value(metrics).unwrap_or_default()` (`protocol_v2.rs:490`) ...
  `validate()` **surfaces** the error
  (`metrics.slos["x"] must be a number`) instead of defaulting.
  ```

  Controller bypass acknowledged at `docs/specs/typed-factory-runner.md:361-369`:
  ```text
  **`BenchmarkRun::validate()` is not a boundary the cellular controller crosses,
  and this record does not claim otherwise.** ... It never constructs `BenchmarkRunWireV2`,
  so `validate_outer()` ... and `into_authored` ... do not run there **today** either.
  ```

  Exact independent fallback, `rust/runtime/src/engine/cellular_controller.rs:2845-2855`:
  ```rust
  pub(crate) fn cellular_metrics_config(envelope: &serde_json::Value) -> Result<MetricsConfig> {
      let spec: crate::engine::protocol::MetricsSpec = envelope
          .pointer("/run/cfg/metrics")
          .cloned()
          .map(|value| serde_json::from_value(value).unwrap_or_default())
          .unwrap_or_default();
      let use_server_token_count = envelope
          .pointer("/run/cfg/endpoint/use_server_token_count")
          .and_then(serde_json::Value::as_bool)
          .unwrap_or(false);
      crate::engine::execute::metrics_config(&spec, use_server_token_count)
  }
  ```

  Startup consumer, `rust/runtime/src/engine/cellular_controller.rs:857-862`:
  ```rust
  let injected_seed = resolve_cellular_seed(envelope);
  // Derive the metrics policy from the envelope so the merge reproduces the
  // authored SLOs / timeslices, exactly as the single-process path does.
  let metrics_config = cellular_metrics_config(envelope)?;
  ```

  Merge-mode consumer, `rust/runtime/src/engine/cellular_controller.rs:2224-2231`:
  ```rust
  if cellular_metrics_config(envelope).is_ok_and(|config| {
      matches!(
          config.storage_mode,
          crate::metrics_core::MetricsStorageMode::Sketch { .. }
      )
  }) {
      return false;
  }
  ```

  Existing controller test, `rust/runtime/src/engine/cellular_controller.rs:4532-4552`, checks valid metrics, absence, and an unknown metric name, but does not exercise `{"slos":{"x":"not-a-number"}}`; that payload is swallowed into `MetricsSpec::default()` before `metrics_config` can reject it.
resolution (r6): Valid. This is the path I flagged in my own brief as untraced (`cellular_controller.rs:2846`, "hints at a path around the projection I have not traced"), and you traced it to the conclusion I should have reached.

Confirmed: `cellular_metrics_config` (`cellular_controller.rs:2845-2856`) does `envelope.pointer("/run/cfg/metrics").cloned().map(|value| serde_json::from_value(value).unwrap_or_default())` into `engine::protocol::MetricsSpec` — the same swallow as the projection, on the raw envelope, with no `BenchmarkRunWireV2` and no `validate()` upstream. Both call sites confirmed: `:862` on the controller startup path, deriving the `MetricsConfig` the merge folds with, and `:2225-2231` inside `cellular_will_use_exact_fold`, deciding exact-vs-sketch storage. So the metrics decision recorded at `:377-392` reaches the cells and not the controller, and step 4 could pass every listed `validate()` test while the controller builds a default merge and storage policy from metrics the cells are about to reject. The `:361-375` paragraph correctly established that the controller never crosses `validate()` — and then I failed to draw the consequence that anything the controller decodes for itself needs its own port entry.

Artifact adds it as a required port, placed immediately after that paragraph so the consequence sits with the premise: repoint `cellular_metrics_config` at the same strict typed conversion so the controller fails on the same input the cells fail on, with a step-4 controller test for a **non-numeric SLO value** — noting, as you did, that the existing test at `cellular_controller.rs:4532-4552` covers an unknown SLO *name* and passes either way. The lenient-fallback alternative is named and rejected rather than left open: a merge policy derived from config the run is about to reject has no defensible meaning, and if someone later wants it, it has to be written down as a deliberate asymmetry with a reason.

Worth flagging for the remaining rounds: this is the second controller-side bypass found in this spec's history (the first was `build_cell_envelope` rewriting `runtime.cells` to `1`, which falsified the dispatch-default claim in an earlier round). The controller reads the envelope directly in more places than these two, and I have not enumerated them. If you want a target, `cellular_controller.rs`'s other `envelope.pointer(...)` call sites are the highest-yield surface left in this record.

## O18 — The new strict artifact decision still bypasses the cellular controller, whose independent artifact decoder silently defaults the whole section; the promised step-4 test is also absent from the named gate. `run_cellular_with_startup_probe` decodes `/run/cfg/artifacts` into `crate::engine::protocol::ArtifactSpec` with `.and_then(|value| serde_json::from_value(value).ok()).unwrap_or_default()` at `rust/runtime/src/engine/cellular_controller.rs:720-725`. `ArtifactSpec` is `#[serde(deny_unknown_fields)]` at `rust/runtime/src/engine/protocol.rs:199-202` and contains no `user_files` field (repo search in that file returns no matches), while typed `Artifacts` carries `pub user_files: Option<Vec<UserFile>>` at `rust/runtime/src/config/model/artifacts.rs:37-39`. Consequently even a valid cellular config carrying both `records_path` and any valid `user_files` makes the controller decode fail and silently replaces the entire policy with `ArtifactSpec::default()`, losing the records/replay/shipping paths used by controller code. The artifact now chooses strict rejection for invalid `artifacts.user_files[i].format` and says “step 4 adds a named test” at `docs/specs/typed-factory-runner.md:353-360`, but its exhaustive named Step-4 test list at `docs/specs/typed-factory-runner.md:1173-1212` contains no artifact-format case, and it only repoints the analogous controller-side metrics fallback at `:421-446`. As with metrics, `BenchmarkRun::validate()` does not reach the controller path. Step 4 must repoint this controller artifact conversion (without dropping controller-relevant paths when `user_files` is present), add a cellular controller test, and actually add the promised named `BenchmarkRun` artifact-format test.
severity: high   raised: r7   status: refined
proven: yes
evidence:
  Exact controller decode, `rust/runtime/src/engine/cellular_controller.rs:720-725`:
  ```rust
  let artifacts: crate::engine::protocol::ArtifactSpec = envelope
      .pointer("/run/cfg/artifacts")
      .cloned()
      .and_then(|value| serde_json::from_value(value).ok())
      .unwrap_or_default();
  ```

  Exact target strictness, `rust/runtime/src/engine/protocol.rs:199-205`:
  ```rust
  /// Artifact paths relative to the exclusive run directory.
  #[derive(Clone, Debug, Default, Deserialize)]
  #[serde(deny_unknown_fields)]
  pub struct ArtifactSpec {
      /// Per-request metrics JSONL path, or absent when records are disabled.
      #[serde(default)]
      pub records_path: Option<PathBuf>,
  ```
  A literal search for `user_files` in `rust/runtime/src/engine/protocol.rs` returns no matches.

  Exact typed source, `rust/runtime/src/config/model/artifacts.rs:37-39`:
  ```rust
  /// Once-rendered user files (present when authored).
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub user_files: Option<Vec<UserFile>>,
  ```

  The controller value is live: `artifact_shipping::shippable_relatives(&artifacts)` consumes it at `cellular_controller.rs:750`; `write_controller_replay_artifacts(&artifacts, ...)` consumes it via the function declared at `cellular_controller.rs:314-351`.

  Artifact promise, `docs/specs/typed-factory-runner.md:353-360`:
  ```text
  **Decision: do not port the fallback.** ... `validate()` surfaces
  `artifacts.user_files[i].format must be one of json, yaml, text`, and step 4
  adds a named test for it.
  ```
  The named test inventory is `docs/specs/typed-factory-runner.md:1173-1212`; it names unknown fields, validation/cardinality, export stems, variation, resolved, resource presence, mandatory sections/metrics, tokenizer, dispatch, endpoint profiles, WEKA, workers, and phases, but no `artifacts.user_files` case.
resolution (r8): Valid, and worse than a documentation gap: this is a live defect on the current tree, reachable from an entirely valid config.

Verified: `cellular_controller.rs:720-725` reads `/run/cfg/artifacts` off the raw envelope with `.and_then(|value| serde_json::from_value(value).ok()).unwrap_or_default()` into `crate::engine::protocol::ArtifactSpec`, which is `#[serde(deny_unknown_fields)]` (`engine/protocol.rs:199-202`) and has no `user_files` field — confirmed by searching that file, which returns nothing. Typed `Artifacts.user_files` is `Option<Vec<UserFile>>` with `skip_serializing_if = "Option::is_none"` (`config/model/artifacts.rs:37-39`), so it is omitted only while `None`. The moment a cellular run authors one user file — every value valid — `deny_unknown_fields` trips and the controller silently uses `ArtifactSpec::default()` for the upload and concatenation decisions, losing `records_path`, `raw_path`, `outputs_path`, `inputs_path`, and `trace`. Nothing reports it.

You are also right that the step-4 test list at `:1173-1212` had no artifact-format case, so the "step 4 adds a named test" sentence in the artifacts decision pointed at nothing. That is the second time in this contest a promised obligation was written in prose and not carried into the gate list, which is a defect in how I maintain that list rather than in any one claim.

Artifact now records this immediately after the `cellular_metrics_config` port, so the two controller bypasses sit together and read as one pattern. Step 4 owes three things: repoint the conversion onto the typed section — with your caveat stated explicitly, that adding `user_files` awareness must not drop the fields the controller actually consumes; add a cellular controller test for a run carrying valid `user_files` alongside `records_path`/`raw_path`/`inputs_path`, asserting those paths survive; and add the `BenchmarkRun` artifact-format rejection test. Both new tests are now in the step-4 gate list, not only in prose.

Worth stating plainly since this is the last round: the controller reading the envelope directly is now three findings deep (`build_cell_envelope` rewriting `runtime.cells`, `cellular_metrics_config`, this). I flagged the remaining `envelope.pointer(...)` call sites as the highest-yield surface at the end of the last round and did not enumerate them myself. Whoever executes step 4 should do that enumeration first; the record says so, but it is not a substitute for having done it.

## O19 — The record treats `StaticAccuracyWorkloadConfigV2` as a selected production path and includes its factory body in the mandatory repoint inventory, but current source makes that factory unreachable: production registration installs only `scheduled` and `graph`, and the projection can emit only those two ids. `register_online_workloads` registers only `ScheduledWorkloadFactoryV2` and `GraphWorkloadFactoryV2` at `rust/runtime/src/engine/online_execution.rs:166-174`; repo-wide search finds `register_http_static_accuracy_workload(` only at its declaration `online_execution.rs:178` and a unit test call at `:2137`. Independently, `BenchmarkRunWireV2::into_authored` sets `let workload_id = workload_kind.workload_id();` and uses it at `rust/runtime/src/engine/protocol_v2.rs:367,424-426`, while `WorkloadKind::workload_id` returns only `"scheduled"` and `"graph"` at `rust/runtime/src/config/model/workload_kind.rs:55-62`. Thus `StaticAccuracyWorkloadFactoryV2::validate_run` is not “selected by the static-accuracy path” as claimed at `docs/specs/typed-factory-runner.md:151-153`; the source’s own comment correctly says static accuracy is represented through `NativeDatasetPlan::StaticAccuracy` while the emitted workload id remains `scheduled` (`workload_kind.rs:38-46`). The record must distinguish the live scheduled dataset-plan path from the unregistered/unselectable legacy factory. Repointing the latter may be chosen as dead-code/test maintenance, but it is not preserved production behavior and cannot be listed as a live `validate_run` surface without qualification.
severity: medium   raised: r7   status: refined
proven: yes
evidence:
  Exact production workload registration, `rust/runtime/src/engine/online_execution.rs:166-174`:
  ```rust
  /// Register the built-in executable workloads (`scheduled`, `graph`).
  pub fn register_online_workloads(registry: &mut crate::extensions::AIPerfRegistry) -> Result<()> {
      let tokenizers: Arc<dyn OnlineTokenizerSourceResolver> =
          Arc::new(HfHubOnlineTokenizerSourceResolver::default());
      registry.register_workload(Arc::new(ScheduledWorkloadFactoryV2 {
          tokenizers: tokenizers.clone(),
      }))?;
      registry.register_workload(Arc::new(GraphWorkloadFactoryV2 { tokenizers }))?;
      Ok(())
  }
  ```
  Repo-wide literal search for `register_http_static_accuracy_workload(` returns only `online_execution.rs:178` (the function declaration) and `online_execution.rs:2137` (unit test invocation).

  Exact projection, `rust/runtime/src/engine/protocol_v2.rs:367` and `:424-426`:
  ```rust
  let workload_id = workload_kind.workload_id();
  ...
  let workload = NamedRunnerComponentSpecV2 {
      id: workload_id.parse().expect("built-in workload ID is valid"),
      config: raw_value(workload_config)?,
  };
  ```

  Exact closed output, `rust/runtime/src/config/model/workload_kind.rs:55-62`:
  ```rust
  pub fn workload_id(self) -> &'static str {
      match self {
          WorkloadKind::Scheduled => "scheduled",
          WorkloadKind::Graph => "graph",
      }
  }
  ```

  Exact source clarification, `workload_kind.rs:38-46`:
  ```rust
  /// `StaticAccuracy` is intentionally not represented: today's projection selects
  /// a static-accuracy run through the dataset *plan*
  /// ([`NativeDatasetPlan::StaticAccuracy`](crate::engine)), not a distinct
  /// workload id — the emitted workload id is still `scheduled`.
  ```

  Contradictory artifact claim, `docs/specs/typed-factory-runner.md:151-153`:
  ```text
  `StaticAccuracyWorkloadConfigV2` remains outside `workload_kind`'s two arms — it
  is selected by the static-accuracy path, not by the classifier — and the
  exhaustive workload match must not be read as claiming otherwise.
  ```
  The artifact also calls its body part of the “verified inventory” at `docs/specs/typed-factory-runner.md:1035-1061`, specifically `online_execution.rs:447` at `:1054-1057`.
resolution (r8): Valid, and the correct conclusion is stronger than the one I wrote. The record said static accuracy "is selected by the static-accuracy path, not by the classifier," which reads as a live alternate selection route. There is no such route.

Confirmed on both halves. Registration: `register_online_workloads` (`online_execution.rs:167-175`) installs exactly `ScheduledWorkloadFactoryV2` and `GraphWorkloadFactoryV2`; a search over all of `rust/` for `register_http_static_accuracy_workload` returns its declaration at `:178`, its `_with_factories` delegate at `:189`, and a single unit-test call at `:2137` — no production caller anywhere, CLI included. Projection: `into_authored` emits `workload_kind.workload_id()` (`protocol_v2.rs:367`, `:424-426`), and that returns only `"scheduled"` or `"graph"` (`workload_kind.rs:55-62`). Either fact alone makes the factory unselectable; together they leave no path at all.

And the source told me the answer. `workload_kind.rs:38-46` states that static accuracy is represented through `NativeDatasetPlan::StaticAccuracy` while the emitted id stays `scheduled`, with a `TODO(step-1)` saying the variant should be added only once the projection emits a distinct id. I read that comment when writing the exhaustive-match argument and paraphrased it into "selected by the static-accuracy path" without checking whether the factory it names is registered.

Artifact now says the factory is unreachable, gives both proofs, and states the consequence you asked for: `StaticAccuracyWorkloadFactoryV2::validate_run` (`online_execution.rs:447`) is removed from the mandatory repoint inventory as a preserved production surface. Its `transport_id == "http"` check and `run.models.items.len()` read are reachable only from the hand-registering unit test. Repointing it is dead-code maintenance — worth doing so the file stays coherent after `AuthoredRunSpecV2` dies, and worth doing carefully because the accuracy path is expected to become selectable — but it carries no behavior-preservation obligation and no e2e gate. The live static-accuracy surface to audit is the scheduled body plus the dataset plan.

Also worth noting for the exhaustiveness argument, which this strengthens rather than weakens: the two-arm match is complete against what the wire can actually carry, and adding a third arm now would invent behavior nothing produces. The record says that explicitly now.

## O20 — The Purpose still makes the exact false count O10 was supposed to remove: it says the opaque `RawValue` seam is “confined to the two residual uses listed under Non-goals” at `docs/specs/typed-factory-runner.md:23-26`, while Non-goals explicitly says “Three uses survive this migration” and enumerates dynosim nested args, dataset payload, and the sidecar adapter boundary at `:1275-1285`. This is a direct internal contradiction in the target guarantee. Change “two” to “three” (or avoid a count) so the Purpose agrees with the normative non-goal and §2 sidecar decision.
severity: low   raised: r7   status: refined
proven: yes
evidence:
  `docs/specs/typed-factory-runner.md:23-26`:
  ```text
  What the change delivers is every **typed** section of the config carried typed
  end to end: no typed→`Value`→`RawValue`→typed round-trip, no per-factory
  re-decode, decode-time errors, and the opaque `RawValue` seam confined to the two
  residual uses listed under [Non-goals](#non-goals-and-trade-offs).
  ```

  `docs/specs/typed-factory-runner.md:1275-1285`:
  ```text
  - **"Zero `RawValue`" is a direction, not a literal end state.** Three uses
    survive this migration. Two are inner adapter inputs: the `dynosim` transport
    variants' nested Dynamo engine/router args ... and the dataset payload ... The third
    is the **sidecar adapter boundary** ...
  ```
  The §2 sidecar entry likewise says the seam survives at `docs/specs/typed-factory-runner.md:641-672`.
resolution (r8): Correct, and exactly the failure mode I warned about in my own brief: the fold-forward updated the section that owns the fact and left the count in the summary that quotes it. O10 added the sidecar adapter boundary as a third residual `RawValue` use and rewrote Non-goals to say "Three uses survive this migration"; the Purpose paragraph at `:23-26` still said "the two residual uses listed under Non-goals," pointing the reader at a list that contradicts the sentence pointing at it. A reader taking Purpose as the target guarantee would conclude the sidecar seam is out of scope, which is the opposite of what §2 decided.

Fixed by removing the count rather than incrementing it, and by naming the three inline so the two sections cannot drift apart again: dynosim nested engine/router args, the dataset payload, and the sidecar adapter boundary — with the distinction §2 draws preserved, that the last is an open seam kept by design rather than an opaque leaf.

Rated low, and it is low as a defect. It is not low as a signal: it is the only place in this rewrite where a correction landed in one section and not in the summary that restates it, and it was found by reading the document against itself rather than against the tree. That is a check I did not run.

## O21 — Deleting `AuthoredRunSpecV2`/`EndpointProfilesSpecV2` also deletes the input to the live endpoint-profile validation and normalization pipeline, but the migration never specifies its typed replacement. `validate_endpoint_profiles_v2` takes `&AuthoredRunSpecV2`, iterates `run.endpoints.profiles: Vec<Box<RawValue>>`, and strictly re-decodes each profile as private `EndpointProfileConfigV2` at `rust/runtime/src/engine/registry.rs:1285-1302`; it then enforces non-empty URLs, positive connection limit, wait-mode/session-header semantics, endpoint canonicalization/preparation, readiness policy, proxy resolution, finite time conversion, and constructs `ValidatedEndpointProfileV2` through `:1303-1418`. The artifact discusses only profile `id` injection, override rename/removal, and sorting (`docs/specs/typed-factory-runner.md:689-748`), and never names either `validate_endpoint_profiles_v2` or `EndpointProfileConfigV2` (literal searches return no matches). Yet Step 4 deletes `AuthoredRunSpecV2` and says the protocol module is reduced to outer/result types at `:1026-1033`, while Purpose promises typed sections carried typed end-to-end at `:23-35`. A direct typed default `Endpoint` path cannot simply drop this function: derived `Endpoint` decoding does not enforce `urls` non-empty, `connection_limit > 0`, or trimmed `session_header`, and it still must produce the normalized `ValidatedEndpointProfileV2`. Conversely retaining the current function requires rebuilding the forbidden typed `Endpoint` → `Value`/`RawValue` → `EndpointProfileConfigV2` round-trip. Specify and test a field-by-field typed default-profile conversion preserving every live semantic check and normalization, plus the raw/open override-profile path; repoint `coordinator.rs:189-203` accordingly.
severity: high   raised: r7   status: refined
proven: yes
evidence:
  Exact deleted carrier, `rust/runtime/src/engine/protocol_v2.rs:870-877`:
  ```rust
  /// Authored endpoint profiles shared by every transport/workload pair.
  #[derive(Debug, Default, Deserialize)]
  #[serde(deny_unknown_fields)]
  pub struct EndpointProfilesSpecV2 {
      /// Non-empty raw profiles. Each object must carry `id` and `type`; the
      /// selected endpoint factory owns every remaining key.
      pub profiles: Vec<Box<RawValue>>,
  }
  ```

  Exact live consumer, `rust/runtime/src/engine/registry.rs:1285-1305`:
  ```rust
  pub fn validate_endpoint_profiles_v2(
      run: &AuthoredRunSpecV2,
      endpoints: &EndpointRegistry,
  ) -> Result<Vec<ValidatedEndpointProfileV2>> {
      let mut validated = Vec::with_capacity(run.endpoints.profiles.len());
      for (index, authored) in run.endpoints.profiles.iter().enumerate() {
          let config = strict_decode::<EndpointProfileConfigV2>(
              authored,
              &format!("endpoint profile {index}"),
          )?;
          ensure!(
              !config.id.is_empty() && config.id.trim() == config.id,
              "endpoint profile {index}.id must be non-empty and contain no surrounding whitespace"
          );
  ```
  Further exact checks/conversions are at `registry.rs:1307-1418`: `!config.urls.is_empty()` (`:1307-1310`), `config.connection_limit > 0` (`:1311-1314`), wait-mode validation (`:1315-1321`), trimmed `session_header` (`:1322-1327`), `endpoints.canonical_id` (`:1328`), `endpoints.prepare` (`:1360`), readiness validation (`:1361-1365`), proxy resolution (`:1367-1373`), `seconds_to_optional_ns` / `seconds_to_ns` (`:1381-1407`), and construction of `ValidatedEndpointProfileV2` (`:1411-1418`).

  The current coordinator calls it at `rust/runtime/src/engine/coordinator.rs:189-203`:
  ```rust
  let endpoint_profiles =
      match validate_endpoint_profiles_v2(&run, self.product_registry.endpoints()) {
  ```

  Typed `Endpoint` fields are at `rust/runtime/src/config/model/endpoint.rs:115-207`; `urls: Vec<String>` (`:118-119`), `connection_limit: u32` (`:149-151`), and `session_header: Option<String>` (`:183-185`) have no semantic validators there.

  The artifact's endpoint entry is `docs/specs/typed-factory-runner.md:689-748`; literal searches for `validate_endpoint_profiles_v2` and `EndpointProfileConfigV2` in the artifact return no matches. Step 4 deletion is explicit at `:1026-1033`.
resolution (r8): Valid, and the most consequential finding of either contest. The record specified how `into_authored` *builds* endpoint profiles — `id` injection, override rename/removal, the sort — and said nothing about who *consumes* them. Literal searches confirm it: neither `validate_endpoint_profiles_v2` nor `EndpointProfileConfigV2` appeared anywhere in the document.

Verified: `validate_endpoint_profiles_v2` (`registry.rs:1293-1418`) takes `&AuthoredRunSpecV2`, iterates `run.endpoints.profiles: Vec<Box<RawValue>>`, and `strict_decode`s each into the private `EndpointProfileConfigV2` (`registry.rs:966-1011+`) — a **third** endpoint shape, distinct from both typed `Endpoint` and the projected profile JSON. It enforces the `id` non-empty/untrimmed check, `urls` non-empty, `connection_limit > 0`, the `wait_for_model_mode` domain, then canonicalization, readiness policy, proxy resolution, and time conversion, and produces the `ValidatedEndpointProfileV2`s the run indexes by position. Production caller `coordinator.rs:190`. Step 4 deletes both its parameter type and its input field.

Your framing of the trap is exactly right and I have adopted it: dropping the function loses real rejections — the derived `Endpoint` decoder attaches no validation to `urls` or `connection_limit`, and nothing typed produces a `ValidatedEndpointProfileV2` — while keeping it means rebuilding typed → `Value`/`RawValue` → `EndpointProfileConfigV2`, reintroducing the exact round-trip this arc exists to delete, at the boundary where it matters most.

The obligation now recorded is the split by provenance you asked for. The default profile converts field-by-field from typed `Endpoint` to `ValidatedEndpointProfileV2`, re-expressing every live check against typed fields and taking its `"default"` identity from the §2 requirement. The override profiles keep the raw path unchanged — `cfg.endpoint_profiles` stays an open `serde_json::Map<String, Value>`, so `strict_decode::<EndpointProfileConfigV2>` still applies to them verbatim, with map-key identity. `coordinator.rs:189-203` repoints onto the split. The record states plainly that this is the largest single unit of work in step 4 and that estimating the migration without it understates the step materially — which it did.

One prerequisite I added beyond your text, because it is where a converter rewrite loses things quietly: `EndpointProfileConfigV2` carries `reset_kv_cache`, `server_profiler`, `template`, `response_field`, `api_key`, `polling_interval_seconds`, and `request_content_type`, and any of those reaching the default profile today must reach it after. A field-by-field diff of the two structs is now a step-4 prerequisite, not an implementation detail.

