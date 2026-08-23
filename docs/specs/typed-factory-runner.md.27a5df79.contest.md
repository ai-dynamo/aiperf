# Contest ledger

> Do not edit — generated from the contest tables after each round.

- **kind:** spec_off
- **status:** running
- **round:** 7 / 8
- **artifact:** `docs/specs/typed-factory-runner.md`
- **low-friction:** yes — zero retractions and zero contested objections; unanimity is as consistent with correlated bias as with correctness, so this exchange is flagged UNVALIDATED rather than clean
- **persuasion-override rate (CW-POR):** 0.00 — the fraction of author retractions that answered an UNPROVEN objection; a high value means the author yielded to confident assertion rather than to evidence

## Seats

- **author** — `opus` (family `claude-trace`, lens —)
- **skeptic** — `gpt-5.6-sol` (family `claude-codex`, lens correctness — verify every code claim mechanically; preserved-behavior gaps in the migration)

## Objections

## O1 — §5 step 3 would reintroduce the just-fixed graph-ir idle-cap loss: `docs/specs/typed-factory-runner.md:516-520` requires the "`weka_semantics`-conditional `system_idle_gap_cap_seconds` attachment", while the implementation on `ajc/typed-factory-runner-v2` now attaches `system_idle_gap_cap_seconds` for every `WorkloadKind::Graph` at `rust/runtime/src/engine/protocol_v2.rs:411-427`. The record itself acknowledges at `docs/specs/typed-factory-runner.md:702-708` that the old legacy/agentx condition made the flag a silent no-op under graph-ir. Following the normative migration text would therefore preserve the obsolete bug instead of current behavior.
severity: high   raised: r1   status: refined
proven: yes
evidence:
  Command:
  `git -C '/home/anthony/nvidia/projects/aiperf/ajc/rust' show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | sed -n '400,430p'`

  Output:
  ```
     411        if workload_kind == WorkloadKind::Graph {
     412            workload_config["weka_semantics"] = serde_json::json!(cfg.weka_semantics);
     413            workload_config["ignore_trace_delays"] = serde_json::json!(cfg.ignore_trace_delays);
     414            workload_config["recorded_agent_default"] =
     415                serde_json::json!(cfg.scenario.as_deref() == Some("recorded-agent-default"));
     416            workload_config["planned_replay_traces"] =
     417                serde_json::to_value(&self.planned_replay_traces)
     418                    .map_err(|error| anyhow!("run.planned_replay_traces: {error}"))?;
     419            // Both weka arms consume the cap: `lower_legacy_agentic` threads it
     420            // into every `PhaseSpec::AgenticReplay`, and `lower_graph` puts it on
     421            // `NativeGraphDatasetPlan` where `TraceExecutor::cap_system_idle_wait_us`
     422            // applies it. Gating the projection on legacy/agentx made the flag a
     423            // silent no-op under graph-ir even though `resolve.rs` validates it and
     424            // its own rejection message names graph-ir as a supported mode.
     425            if let Some(cap) = cfg.system_idle_gap_cap_seconds {
     426                workload_config["system_idle_gap_cap_seconds"] = serde_json::json!(cap);
     427            }
  ```

  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '515,529p;702,709p'`

  Output:
  ```
     516  3. Repeat for the workload seam; collapse `ScheduledWorkloadConfigV2` /
     517     `GraphWorkloadConfigV2` into typed-optional fields — all **five** graph-only
     518     fields, including the `recorded_agent_default` derivation and the
     519     `weka_semantics`-conditional `system_idle_gap_cap_seconds` attachment. All
     520     four graph-only fields that survive the DTO are consumed inside `lower_graph`
  ...
     702    - The idle cap's own guard is written on the **graph-ir** arm, not `legacy` as an
     703      earlier draft of this bullet said: `legacy` lowering needs a HuggingFace
     704      download and so cannot run offline against the mock server. That change
     705      exposed the third defect — `into_authored` attached
     706      `system_idle_gap_cap_seconds` only under `weka_semantics` legacy/agentx, so
     707      the flag was a silent no-op under graph-ir even though `resolve.rs` validates
     708      it there, its rejection message names graph-ir as supported, and `lower_graph`
  ```
resolution (r2): Correct and confirmed independently. `git show ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs` shows the guard is now a plain `if let Some(cap) = cfg.system_idle_gap_cap_seconds` inside the `workload_kind == WorkloadKind::Graph` block, landed by `3f77a3adac` ("fix(engine): project the system idle-gap cap under graph-ir too"), while this checkout (`ajc/rust`) still has the pre-fix `matches!(cfg.weka_semantics.as_deref(), Some("legacy") | Some("agentx"))` at protocol_v2.rs:411-427. The record's own step-3 verification bullet names that condition as the third defect the guards found, so the normative migration text and the defect log were in direct contradiction — following the former would re-encode the bug.

Refined in commit 7c59f19e97, two places: the Built inventory (docs/specs/typed-factory-runner.md:61-73) now records the pre-fix conditional attachment as corrected and cites `3f77a3adac`, and §5 step 3 now says the cap attaches on the graph arm whenever `cfg.system_idle_gap_cap_seconds` is `Some`, with no `weka_semantics` predicate, and states explicitly that reintroducing the legacy/agentx guard would make the flag a silent no-op under graph-ir again. `docs/specs/README.md`'s index entry carries the same correction.

## O2 — The step order is not executable as specified: after landed step 1, `BenchmarkRunWireV2::into_authored` converts typed `Transport` into `NamedRunnerComponentSpecV2` at `rust/runtime/src/engine/protocol_v2.rs:373`, and `transport_component` serializes the variant to `Value`, removes `"type"`, and returns only `{ id, config }` at `rust/runtime/src/engine/protocol_v2.rs:534-552`. `AuthoredRunSpecV2` retains only `pub transport: NamedRunnerComponentSpecV2` at `rust/runtime/src/engine/protocol_v2.rs:617-620`. Consequently §5 step 2's required "exhaustive `Transport` match" at `docs/specs/typed-factory-runner.md:443-449` has no `Transport` value at the downstream `native_execution` selection boundary while step 4's direct-`BenchmarkRun` repoint is still unbuilt. The record must specify an intermediate retention/repoint (for example retaining typed `Transport` in `AuthoredRunSpecV2`, or moving the relevant `BenchmarkRun` plumbing before step 2); otherwise step 2 cannot be implemented without an unrecorded structural change or collapsing into step 4.
severity: high   raised: r1   status: refined
proven: yes
evidence:
  Command:
  `git -C '/home/anthony/nvidia/projects/aiperf/ajc/rust' show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | sed -n '357,380p;524,552p;608,640p'`

  Output:
  ```
     357    /// Adapt canonical Config nesting to the linked preparation seam.
     358    pub fn into_authored(self) -> Result<AuthoredRunSpecV2> {
  ...
     362        let workload_kind = workload_kind(&self.cfg);
     363        let cfg = self.cfg;
  ...
     373        let transport = transport_component(cfg.transport.as_ref())?;
  ...
     534 fn transport_component(transport: Option<&Transport>) -> Result<NamedRunnerComponentSpecV2> {
  ...
     539     let id: ComponentId = transport
     540         .canonical_id()
  ...
     543     let mut object = serde_json::to_value(transport)
  ...
     548     object.remove("type");
     549     Ok(NamedRunnerComponentSpecV2 {
     550         id,
     551         config: raw_value(Value::Object(object))?,
     552     })
  ...
     608 pub struct AuthoredRunSpecV2 {
  ...
     617     /// Open transport selection (the `{transport, clock}` execution axis).
     618     pub transport: NamedRunnerComponentSpecV2,
     619     /// Open workload selection.
     620     pub workload: NamedRunnerComponentSpecV2,
  ```

  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '431,449p;530,535p'`

  Output:
  ```
     439     for the transport family is therefore *not* "add typed configs" but "add a
     440     consumer that reads `cfg.transport` directly", run alongside the existing
     441     projection, and assert the two produce identical bindings. Dispatch is an
     442     exhaustive `match` on `Transport`, not `match id.as_str()`; the `RegistryId`
     443     string and the plugin tail belong to the seam in §3, not to this field.
     444  2. Move `native_execution` selection to the exhaustive `Transport` match for
     445     built-ins. Three obligations the audits surfaced: (a) the registry resolves not
     446     just config but
     447     the transport's **`NativeTransportExecution` binding** (`resolve_native_execution`
     448     → `transport_factory(id).native_execution(...)`), so the match arms
     449     must supply those bindings directly or every workload's prepare/`validate_run`
  ...
     531  4. Delete `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`, and
     532     `BenchmarkRunWireV2`; point `coordinator.rs` at **`BenchmarkRun`**
  ```
resolution (r2): Correct, and the gap is worse than a missing sentence — it made step 2 unimplementable as written. Verified on this checkout: `grep -n "pub transport: NamedRunnerComponentSpecV2" rust/runtime/src/engine/protocol_v2.rs` -> `586:    pub transport: NamedRunnerComponentSpecV2,`, and `grep -n "fn resolve_native_execution" rust/runtime/src/engine/online_execution.rs` -> `118:`, whose signature is `(context: &RunContext, transport: &dyn ValidatedTransportConfig, transport_id: &str)` and whose five call sites (`online_execution.rs:237,259,333,374,483`) sit inside `WorkloadFactory::validate_run` bodies handed `&AuthoredRunSpecV2`. Step 1's `transport_component` is a typed *producer* whose output is still `{ id, config: RawValue }`, so after step 1 nothing downstream holds a `Transport` to match on.

Refined in commit 7c59f19e97: §5 step 2 gains a "Prerequisite (contest O2, proven)" paragraph that states the gap with those anchors and selects the intermediate — `AuthoredRunSpecV2` gains a typed `transport_typed: Transport` populated by `into_authored` from `cfg.transport` and carried alongside the projected `NamedRunnerComponentSpecV2` (which obligations (b) and (c) still need), both dying together in step 4. It also records and rejects the alternative you named — hoisting step 4's `BenchmarkRun` repoint ahead of step 2 — because it merges the two largest steps and forfeits step 1's differential assertion. The index entry names the intermediate too.

## O3 — The load-bearing source anchor contradicts the record's corrected runner vocabulary: `docs/specs/typed-factory-runner.md:196-208` and `:531-534` require repointing `coordinator.rs` at `BenchmarkRun` and explicitly say "*not* `BenchmarkConfig`", but `docs/specs/typed-factory-runner.md:841-842` still says "the child composition root to repoint at `BenchmarkConfig`". An implementer following the source-anchor checklist would repeat the exact O4 error the normative text says is invalid.
severity: medium   raised: r1   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '196,208p;530,534p;831,845p'`

  Output:
  ```
     196 **Corrected (contest O4, proven).** Earlier drafts of this record said
     197 "`&BenchmarkConfig`" here and again in §5 step 4, while the port list two
  ...
     204 **`BenchmarkRun` is the runner vocabulary throughout this
     205 record** — that is what §2's title says and what the port list requires.
     206 `BenchmarkConfig` remains the *authored* vocabulary reached as `run.cfg`, which is
     207 what §1's typed component unions live on. Any remaining "point the engine at
     208 `BenchmarkConfig`" phrasing is an error against this paragraph.
  ...
     531 4. Delete `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`, and
     532    `BenchmarkRunWireV2`; point `coordinator.rs` at **`BenchmarkRun`** (per O4 —
     533    *not* `BenchmarkConfig`, which cannot carry the run-level facts this same step
     534    ports).
  ...
     841 - `rust/runtime/src/engine/coordinator.rs` — `envelope.run.into_authored()`, the
     842   child composition root to repoint at `BenchmarkConfig`.
  ```
resolution (r2): Conceded — I had spotted this line before your round landed and was holding it to concede here. §2's O4 correction says "Any remaining 'point the engine at `BenchmarkConfig`' phrasing is an error against this paragraph", and the Source anchors entry was exactly that phrasing, in the one section an implementer uses as a checklist. Refined in commit 7c59f19e97: the `coordinator.rs` anchor now reads "repoint at **`BenchmarkRun`**" and cites O3 plus the reason (`BenchmarkConfig` cannot carry the run-level facts the port list moves; authored sections are reached as `run.cfg`).

## O4 — The record's load-bearing target still specifies two mutually exclusive selection architectures. `docs/specs/typed-factory-runner.md:11-23`, §3 at `:315-352`, §4 at `:421-430`, and Non-goals at `:815-838` say selection remains an open `RegistryId` string with a `RawValue` plugin-tail fallback. But §1 selects the closed alternative at `:132-150`, and the record's own O7 correction at `:840-863` proves step 4 deletes the only `{ RegistryId, RawValue }` authored seam and leaves no selectable plugin tail. The live `Transport` source is closed (`rust/runtime/src/config/model/transport.rs:16-31`). An implementer cannot satisfy both the Purpose/§3/§4/Non-goals design and the normative §1/§5 migration; the stale open-tail architecture must be removed or explicitly scoped only to future work throughout, not contradicted by a correction appended beneath it.
severity: high   raised: r3   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '5,24p;131,150p;309,355p;418,435p;810,870p'`

  Output (relevant exact lines):
  ```
      11 [config-model-unification.md](config-model-unification.md) lands: **the runtime
      12 consumes the typed `BenchmarkRun` directly, and a component is selected by an
      13 open, normalized string id (`RegistryId`) whose config is a typed struct for
      14 built-ins and an opaque `RawValue` (decoded by the plugin's own factory) only for
      15 the runtime-loaded-plugin tail**
  ...
     135   - **(A) Keep `Transport` closed.** An exhaustive `match` on the enum is the
     136     honest dispatch, no `RegistryId` string and no plugin tail are involved for
     137     this field, and §3's apparatus does not apply to it.
  ...
     145   **This record selects (A).**
  ...
     315 - **The discriminant** (the component `type`/id) — **stays an open, normalized
     316   string**, not an enum.
  ...
     347 fn decode_transport(spec: ComponentSpec, reg: &Registry) -> Result<TransportBinding> {
     348     match spec.id.as_str() {
  ...
     352         _         => reg.get(&spec.id)?.validate(&spec.config)?.into(), // plugin tail: frozen-registry dyn factory
  ...
     421 `AIPerfRegistry`/`AIPerfExtension` shrinks from "transactional registry of
  ...
     427 unchanged (a built-in id resolves in the `match`'s typed arm; an unknown id falls
     428 to the plugin tail and fails closed at registry lookup, as today).
  ...
     815 - **We reintroduce a match on component id.**
     816   "never matching on a transport kind" via `dyn`. The typed design matches on the
     817   `RegistryId` string exactly once, at the selection boundary, with a plugin-tail
     818   fallthrough — not an exhaustive enum match
  ...
     830 - **Dynamic plugins are a planned future, and this design accommodates them.**
  ...
     835   bootstrap-frozen registry) is retained precisely for them. The refactor does
     836   **not** close the set;
  ...
     840   **Corrected (contest O7, proven): as specified, the migration does not actually
     841   leave a selectable tail
  ...
     843   `{ RegistryId, RawValue }` to survive. It does not: §1 selects transports from
     844   the closed `BenchmarkConfig.transport` enum and workloads from an exhaustive
     845   match, §3 identifies `NamedRunnerComponentSpecV2` as the `{ RegistryId,
     846   RawValue }` seam, and §5 step 4 deletes it.
  ...
     853   So the honest current state is: **registry openness survives for id-addressed
     854   *internal* consumers, and authored selection of a runtime-loaded plugin does
     855   not exist either before or after this migration.**
  ```

  Command:
  `git -C '/home/anthony/nvidia/projects/aiperf/ajc/rust' show 'ajc/typed-factory-runner-v2:rust/runtime/src/config/model/transport.rs' | nl -ba | sed -n '11,32p'`

  Output:
  ```
      16 #[derive(Clone, Debug, Serialize, Deserialize)]
      17 #[serde(tag = "type", rename_all = "snake_case")]
      18 pub enum Transport {
      19     /// Native HTTP/1.1 or HTTP/2 transport.
      20     Http,
      21     /// Native gRPC transport (KServe OIP / Riva).
      22     Grpc,
      23     /// Offline virtual-clock Dynamo replay (fields flat on the transport).
      24     DynosimOffline(DynosimConfig),
      25     /// Online wall-clock Dynamo replay (fields flat on the transport).
      26     DynosimOnline(DynosimConfig),
      27     /// Lightweight fake execution leaf: analytic-latency synthetic responses,
      28     /// zero network (fields flat on the transport).
      29     DryRun(DryRunConfig),
      30     /// Native persistent WebSocket transport.
      31     Websocket(WebSocketTransportConfig),
  ```
resolution (r4): Correct, and the sharpest structural finding in either contest. The record had been patched objection-by-objection and the corrections were appended *beneath* the architecture they invalidated, so Purpose/§3/§4/Non-goals still read as the normative target while §1/§5 built the opposite. Verified: `config/model/transport.rs:16-31` is a closed `#[serde(tag = "type")]` enum over six variants; §1 selects fork (A) to keep it closed; §5 step 4 deletes `NamedRunnerComponentSpecV2`; and the record's own O7 note already proved no authored surface can name a plugin id afterward. Two architectures, one implementer.

Refined in commit a625c0aa30, all four sites:
- **Purpose** now states the target as what §5 builds — exhaustive matches on the closed `Transport` enum and on `workload_kind` — with a "Corrected (contest O4, proven)" paragraph naming the superseded open-`RegistryId` framing and saying that paragraph governs wherever open-tail language survives.
- **§3** is retitled "non-normative for this migration (contest O4)": nothing in it is a task in §5, no step implements it, and an implementer building §5 reads §1 and §5 only. It is retained explicitly as the design the *future* plugin arm inherits — chiefly to preserve the empirical serde finding that rules out the enum encoding, which deleting the section would lose.
- **§4**: your objection let me find a live consequence rather than just stale prose. The paragraph promised "an unknown id falls to the plugin tail and fails closed at registry lookup" plus a diagnostic-parity caveat about producing the registry's "available: …" list from the match's default arm. With a closed `#[serde(tag = "type")]` field there is no default arm and no registry miss — an unknown id fails at *decode*, with serde's own "unknown variant, expected one of …". Both the mechanism and the diagnostic obligation are rewritten to that, leaving step 2's feature-gate refusal as the real diagnostic duty.
- **Non-goals bullet 1** said the match is "not an exhaustive enum match (the open set forbids that)". It is exactly an exhaustive enum match; corrected, with the compiler-enforced exhaustiveness named as the benefit and step 2's obligation (d) as the cost.

## O5 — The Source anchors checklist still says `rust/runtime/src/config/model/` / `config/resolve.rs` is the "resolver step that gains the typed `Transport` union" (`docs/specs/typed-factory-runner.md:893-895`), directly contradicting §1's mechanically correct statement that `BenchmarkConfig.transport: Option<Transport>` and the closed typed `Transport` enum already exist (`:111-129`) and step 1's correction that "there is nothing to introduce" (`:441-450`). The implementation branch confirms the union is already present at `config/model/config.rs:73-75` and `config/model/transport.rs:16-31`. This stale implementation anchor tells the implementer to modify the wrong layer and redo already-landed model work.
severity: medium   raised: r3   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '106,130p;437,451p;888,896p'`

  Output (relevant exact lines):
  ```
     111 - `transport` — **this field already exists and is already typed.**
     112   `BenchmarkConfig.transport: Option<Transport>` (`config/model/config.rs:75`)
  ...
     124   - **The projection is pure loss.**
  ...
     441 1. Introduce typed built-in configs decoded per id (`match id.as_str()`)
  ...
     443    typed `cfg` (no behavior change). **Corrected (round 3):** for transports there
     444    is nothing to introduce — `cfg.transport` is already the closed typed
     445    `Transport` enum
  ...
     893 - `rust/runtime/src/config/model/` and `rust/runtime/src/config/resolve.rs` — the
     894   typed `BenchmarkConfig`/`BenchmarkRun` and resolver step that gains the typed
     895   `Transport` union and typed `synthesis`/`weka_semantics`.
  ```

  Command:
  `git -C '/home/anthony/nvidia/projects/aiperf/ajc/rust' show 'ajc/typed-factory-runner-v2:rust/runtime/src/config/model/config.rs' | nl -ba | sed -n '59,80p'; git -C '/home/anthony/nvidia/projects/aiperf/ajc/rust' show 'ajc/typed-factory-runner-v2:rust/runtime/src/config/model/transport.rs' | nl -ba | sed -n '11,32p'`

  Output:
  ```
      59 /// The runner-consumed benchmark configuration.
  ...
      62 #[derive(Clone, Debug, Default, Serialize, Deserialize)]
      63 pub struct BenchmarkConfig {
  ...
      73     /// Inline transport selection (`cfg.transport`).
      74     #[serde(default, skip_serializing_if = "Option::is_none")]
      75     pub transport: Option<Transport>,
  ...
      16 #[derive(Clone, Debug, Serialize, Deserialize)]
      17 #[serde(tag = "type", rename_all = "snake_case")]
      18 pub enum Transport {
      19     /// Native HTTP/1.1 or HTTP/2 transport.
      20     Http,
      21     /// Native gRPC transport (KServe OIP / Riva).
      22     Grpc,
      23     /// Offline virtual-clock Dynamo replay (fields flat on the transport).
      24     DynosimOffline(DynosimConfig),
      25     /// Online wall-clock Dynamo replay (fields flat on the transport).
      26     DynosimOnline(DynosimConfig),
      27     /// Lightweight fake execution leaf: analytic-latency synthetic responses,
      28     /// zero network (fields flat on the transport).
      29     DryRun(DryRunConfig),
      30     /// Native persistent WebSocket transport.
      31     Websocket(WebSocketTransportConfig),
  ```
resolution (r4): Correct. `grep -n "pub transport" rust/runtime/src/config/model/config.rs` -> `75:    pub transport: Option<Transport>,`, and `config/model/transport.rs:16-31` is the closed enum — so the union the anchor told the implementer to add is already there, and §1/step 1 say so explicitly ("Nothing needs to be introduced here", "there is nothing to introduce"). Same failure mode as O3: the Source anchors section is the checklist an implementer works from, and it was still carrying pre-correction instructions.

Refined in commit a625c0aa30: the anchor now reads "the resolver step that gains typed `synthesis`/`weka_semantics`/`failure_policy`", with an O5 correction note citing `config/model/config.rs:75` and `config/model/transport.rs:16-31` and stating that the union is landed and only the untyped inner seams are outstanding.

Worth recording as a pattern: O3 and O5 are the same defect in the same section, found one round apart. The Source anchors list is where corrections go to be forgotten, because every other section is prose an author rereads and that one is a checklist nobody re-derives.

## O6 — The artifact is textually corrupted: after the final Python source-anchor sentence, `docs/specs/typed-factory-runner.md:904-905` contains literal tool-protocol debris `</parameter>` and `</invoke>`. These are not Markdown, source anchors, or part of the spec and must be removed.
severity: low   raised: r3   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '897,906p'`

  Output:
  ```
     897   (`PluginEntry` — lazy `class_path` load + `metadata: dict` validated via
     898   `get_typed_metadata`; the layer NOT ported).
     904 </parameter>
     905 </invoke>
  ```

  (The current file's exact numbering includes the final source-anchor lines at 897-903 followed immediately by the two literal closing tags at 904-905.)
resolution (r4): Confirmed and removed. `tail -8 docs/specs/typed-factory-runner.md | cat -A` showed the last two lines as `</parameter>$` and `</invoke>$` after the final Python source-anchor line — tool-call debris written into the file by an earlier session's edit. Deleted in commit a625c0aa30; the file now ends on the `get_typed_metadata` line.

## O7 — The purportedly complete step-4 port list still omits live `into_authored` behaviors. It says only four behaviors land before deletion (`docs/specs/typed-factory-runner.md:305-307`) and enumerates only strictness/three outer checks/export stem/planned traces/typed variation at `:573-584`, while the branch body also (1) hard-rejects absent `cfg.models` (`protocol_v2.rs:449-455`), (2) hard-rejects absent default `cfg.endpoint` through `endpoint_profile` (`:456-461`, `:572-588`), (3) hard-rejects absent transport (`:534-537`), and (4) converts `cfg.metrics` with `MetricsSpec::try_from(...).ok()` and deliberately defaults on a non-numeric SLO (`:462-468`). `BenchmarkRun` keeps all of these sections optional, and the specified `BenchmarkRun::validate()` tests only benchmark id, artifact dir, and dataset cardinality (`docs/specs/...:775-785`). Deleting `into_authored` under the written gate can therefore change mandatory-section and invalid-metrics behavior without any selected decision or test. The record's generic sentence that a remaining audit "must walk" the body (`:301-303`) does not port these already-demonstrated behaviors, and its later claim that every other field is only a copy/pure derivation (`:596-598`) is false for these validation/fallback branches.
severity: high   raised: r3   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | sed -n '284,308p;572,600p;761,788p'; git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | sed -n '449,478p;534,594p'`

  Output (relevant exact lines):
  ```
     301   per-record, summary, **and** timeslice artifacts from the same stem. This entry
     302   also generalizes: the remaining step-4 audit must walk `into_authored`'s body
     303   for other field-to-field transforms, not just its struct definition.
  ...
     305 What step 4 deletes is both the *projection* (`into_authored`,
     306 `AuthoredRunSpecV2`, `NamedRunnerComponentSpecV2`) and the *wire DTO*
     307 (`BenchmarkRunWireV2`), with the four ported behaviors above landing on
     308 `BenchmarkRun` first.
  ...
     578    only *after* §2's port list lands on `BenchmarkRun`:
     579    `#[serde(deny_unknown_fields)]`, an inherent `validate()` carrying
     580    `validate_outer`'s three checks ... the export-stem derivation
     581    from `artifacts.records_path` ... `planned_replay_traces` ...
     582    field, and `variation: Option<VariationSpec>`.
  ...
     596    or cellular graph replay loses its trace expectation. Every *other* field is a
     597    copy or a pure derivation (`workload_kind`, `parse_dispatch_mode`,
     598    `worker_count` from `available_parallelism`).
  ...
     775   - an unknown top-level field on a bare run payload is rejected
     776   - `BenchmarkRun::validate()` rejects empty `benchmark_id`, empty `artifact_dir`,
     777     and a two-dataset `cfg.datasets`
  ...
     782   - `variation` round-trips as a typed `VariationSpec`
     783   - `resource_presence` ...
  ```

  Implementation branch output:
  ```
     449        // Lower the authoring models to the runner spec via the typed `From`
     450        // (no `Value` round-trip); a missing models section is a hard error, as
     451        // before.
     452        let models = cfg
     453            .models
     454            .map(ModelsSpec::from)
     455            .ok_or_else(|| anyhow!("run.cfg.models must be an object"))?;
     456        let endpoint = serde_json::to_value(&cfg.endpoint)
  ...
     462        // Lower the authoring metrics to the runner spec via the typed
     463        // `TryFrom` (no untyped `Value` round-trip); default on absence or a
     464        // non-numeric SLO, matching the prior `from_value(...).unwrap_or_default()`.
     465        let metrics = cfg
     466            .metrics
     467            .and_then(|metrics| MetricsSpec::try_from(metrics).ok())
     468            .unwrap_or_default();
  ...
     534 fn transport_component(transport: Option<&Transport>) -> Result<NamedRunnerComponentSpecV2> {
     535     let Some(transport) = transport else {
     536         // Preserve the prior "transport must be an object" failure when unset.
     537         return component_from_inline(Value::Null, "run.cfg.transport");
  ...
     572 fn endpoint_profiles(
  ...
     577     profiles.push(raw_value(endpoint_profile("default", default)?)?);
  ...
     584 fn endpoint_profile(id: &str, value: Value) -> Result<Value> {
     585     let mut profile = value
     586         .as_object()
     587         .cloned()
     588         .ok_or_else(|| anyhow!("run.cfg.endpoint must be an object"))?;
  ```
resolution (r4): Correct on all four, verified independently on both branches, and it lands exactly where I flagged the record as weakest at the end of round 2 — the "must walk the body" deferral and the "every other field is a copy or a pure derivation" generalization.

- `cfg.models` absent: `ajc/rust` `protocol_v2.rs:507-511` — `as_object().ok_or_else(|| anyhow!("run.cfg.models must be an object"))?`, reached with `Value::Null`; the impl branch is `cfg.models.map(ModelsSpec::from).ok_or_else(...)` with the same message.
- default `cfg.endpoint` absent: `endpoint_profile` (`protocol_v2.rs:552-556`) rejects `Null` with `"run.cfg.endpoint must be an object"`.
- `cfg.transport` absent: `transport_component` routes `None` to `component_from_inline(Value::Null, "run.cfg.transport")` with a comment saying it exists to "preserve the prior *transport must be an object* failure when unset".
- metrics: `engine/protocol.rs:209-231` — `TryFrom` fails on the first non-numeric `slos` entry, and `.and_then(...ok()).unwrap_or_default()` therefore discards `slice_duration_seconds`, every *valid* SLO, `sketch`, and `steady_state` along with the bad one. That is worse than your summary said.

And `config/model/config.rs:66,69,75,81` holds all four sections as `Option`, so nothing on the typed model reproduces any of it.

Refined in commit a625c0aa30 as a fifth §2 port-list entry with the evidence above, two explicit decisions rather than another deferral, and tests:
- the three rejections **port verbatim**, messages included, into `BenchmarkRun::validate()` — they are the only thing making those sections mandatory;
- the metrics fallback **does not port as-is**. Silently defaulting a whole metrics section because one SLO is a string is O1's dataset-truncation class of defect, and the greenfield premise resolves it the same way: `validate()` surfaces the `TryFrom` error naming the key. Recorded as a behavior change, not smuggled in.

Step 4's prerequisite list now names all four; step 4's unit-test list gains one case per rejection plus the non-numeric-SLO rejection; and the "copy or pure derivation" sentence is replaced with an O7 correction saying three fields are and four are validation/fallback branches.

## O8 — The “landed step 1” status is false, and the migration still has no executable step-1 differential gate. The normative step says to add “a consumer that reads `cfg.transport` directly” and assert the typed and projected paths produce identical **bindings** (`docs/specs/typed-factory-runner.md:529-533`). But the claimed landed symbol `transport_component` returns `NamedRunnerComponentSpecV2`, explicitly serializes `Transport` to `Value`, strips `type`, and stores the remainder as `RawValue` (`protocol_v2.rs:534-552`); `AuthoredRunSpecV2` retains only that projected field (`:608-620`). Its test compares only `typed.id/config` against `component_from_inline.id/config` (`:1863-1881`), not a `NativeTransportExecution` binding. The spec itself admits this at `docs/specs/...:538-550`: after step 1 “there isn't” a typed `Transport` at selection and `transport_component` is only a typed producer whose output remains `{ id, config }`. The selected `transport_typed: Transport` carrier is not added until step 2 (`:552-558`). Thus `docs/specs/...:42-48` incorrectly marks the direct consumer/differential-binding work as landed, and step 1 as written cannot perform its binding comparison until the carrier currently assigned to step 2 exists. Either step 1 must add the carrier and run both actual selection paths, or it must be honestly redefined as projection-only and the binding differential moved to step 2; the current status/order causes the required safety check to be skipped.
severity: high   raised: r5   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | perl -ne 'print if ($. >= 42 && $. <= 48) || ($. >= 524 && $. <= 560)'`

  Output:
  ```
  42 **Status (contest O8, proven).** This record is *partly* built, not
  43 forward-looking. Implementation lives on `ajc/typed-factory-runner-v2`:
  44 `b7619602fb` already ships §5 step 3's DTO collapse as the unified
  45 `WorkloadConfigV2` (`engine/registry.rs:876`, `:883`), and step 1's typed
  46 `cfg.transport` consumer is `transport_component`
  47 (`engine/protocol_v2.rs`), pinned against the projection it replaces by
  48 `transport_component_matches_inline_projection`. Steps 2 and 4 are not built.
  ...
  529 Transport enum ... Step 1
  530 for the transport family is therefore *not* "add typed configs" but "add a
  531 consumer that reads `cfg.transport` directly", run alongside the existing
  532 projection, and assert the two produce identical bindings.
  ...
  538 **Prerequisite ... step 2 needs a typed `Transport` value at
  539 the selection boundary, and after step 1 there isn't one.** Step 1 lands a
  540 typed *producer* — `transport_component(cfg.transport.as_ref())` — but its
  541 output is still a `NamedRunnerComponentSpecV2 { id, config }`: it serializes the
  542 variant, removes the `"type"` tag, and hands the remainder on as `RawValue`.
  ...
  552 The intermediate this record selects: **`AuthoredRunSpecV2` gains a typed
  553 `transport_typed: Transport` field** ...
  ```

  Command:
  `git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | perl -ne 'print if $. >= 522 && $. <= 552; print if $. >= 608 && $. <= 622; print if $. >= 1860 && $. <= 1883'`

  Output:
  ```
  534 fn transport_component(transport: Option<&Transport>) -> Result<NamedRunnerComponentSpecV2> {
  ...
  543     let mut object = serde_json::to_value(transport)
  ...
  548     object.remove("type");
  549     Ok(NamedRunnerComponentSpecV2 {
  550         id,
  551         config: raw_value(Value::Object(object))?,
  552     })
  ...
  608 pub struct AuthoredRunSpecV2 {
  ...
  617     /// Open transport selection ...
  618     pub transport: NamedRunnerComponentSpecV2,
  ...
  1863 fn transport_component_matches_inline_projection() {
  1865     let typed = transport_component(Some(&transport)) ...;
  1869     let inline = component_from_inline(value, "run.cfg.transport") ...;
  1872     assert_eq!(typed.id.as_str(), inline.id.as_str(), ...);
  1877     assert_eq!(typed.config.get(), inline.config.get(), ...);
  ```
  No binding is built or compared.
resolution (r6): Proven and fixed, though the diagnosis needed one correction before the fix could be written.

Verified: `transport_component` returns `NamedRunnerComponentSpecV2`, `AuthoredRunSpecV2` retains only `pub transport: NamedRunnerComponentSpecV2` (`protocol_v2.rs:586` on ajc/rust), and the landed test compares `typed.id/typed.config.get()` against `component_from_inline`'s over `all_variants()` — six variants, all with `::default()` payloads. No `NativeTransportExecution` is constructed anywhere in it.

Where I part company with the objection is the remedy. It offers two options: add the carrier to step 1 and run both real selection paths, or redefine step 1 as projection-only and move the differential to step 2. Both concede that a binding-level differential is owed. It is not. I read the selection site:

    $ sed -n '118,128p' rust/runtime/src/engine/online_execution.rs
    fn resolve_native_execution(context, transport, transport_id) -> ... {
        let factory = context.product_registry().transport_factory(transport_id)
            .ok_or_else(|| anyhow!("transport {transport_id:?} is not registered"))?;
        factory.native_execution(transport, context)
    }

That is the whole body. The binding is a total function of `(id, config, context)`, and `context` is the same `RunContext` on both sides of the differential. So byte-exact equality of both halves of the component pins the binding transitively — constructing one adds nothing but a registry round-trip. The landed test is the right test; the spec described it wrongly.

The real defect O8 surfaces is therefore an internal contradiction in the record, and it is sharper than "the status is optimistic": step 1 as written owed an assertion that step 2's own prerequisite paragraph — added under O2, sitting eight lines below it — proves impossible in step 1. The record contained its own refutation and neither half noticed.

Refined at `docs/specs/typed-factory-runner.md`:
- Step 1's normative text now says "identical **components**", followed by a "Corrected (O8, proven)" paragraph that states why binding-level equality is unassertable in step 1, quotes `resolve_native_execution`'s body as the reason component equality suffices, names `transport_component_matches_inline_projection` as the gate step 1 owes, and states outright that no binding-level differential is owed at any step. That closes the option the objection left open — otherwise a later reader adds a useless carrier to step 1 to satisfy a word.
- The Status paragraph gains: what landed is a typed *producer* whose output is still `{ id, config }`; `AuthoredRunSpecV2` still retains only the projected field; the pin is component-level; and the status line must not be read as "typed `Transport` now reaches the selection boundary" — it does not, and does not until step 2's `transport_typed` carrier.

Committed in `3abedbb193`.

## O9 — The O4 refinement did not remove all present-tense plugin-tail claims; the record still simultaneously says the tail survives and proves it does not. Purpose promises the opaque seam is confined to “three residual uses” (`docs/specs/typed-factory-runner.md:37-40`). Non-goals says “plugin configs stay opaque” (`:921-924`), that the tail “is retained precisely” for dynamic plugins and the refactor “does not close the set” (`:931-939`), and then counts that plugin tail as one of three residual `RawValue` uses that “stay” (`:966-970`). But the immediately following O7 correction proves no authored `{ RegistryId, RawValue }` object survives step 4 and says no config can select a plugin until a future explicit tail variant is added (`:941-965`). Therefore the migration has only the two actually surviving residual uses named at `:968-970` (dynosim nested args and dataset payload), not three, and it does close authored selection to the closed enum/two-arm classifier. The blanket Purpose caveat at `:21-35` does not make these assertions true; these are still normative current-target/trade-off statements, including an explicit residual-use count. Rewrite them as future-plugin requirements or remove the nonexistent tail from the delivered end state.
severity: medium   raised: r5   status: refined
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | perl -ne 'print if ($. >= 37 && $. <= 40) || ($. >= 920 && $. <= 973)'`

  Output:
  ```
  38 What the change does deliver is the config payload typed end to end: no `Value`
  39 round-trip, no per-factory re-decode, decode-time errors, and the opaque
  40 `RawValue` seam confined to the three residual uses listed under Non-goals.
  ...
  921 - **The runtime crate gains a compile dependency on every built-in component
  922   config type.** That is the cost of typing built-in configs. It is acceptable
  923   because the built-in set is frozen at compile time; plugin configs stay opaque
  924   (`RawValue`), so the coupling does not extend to the open tail.
  ...
  931 - **Dynamic plugins are a planned future, and this design accommodates them.**
  ...
  934   core's compile time), so the plugin tail's `RawValue` config ...
  936   bootstrap-frozen registry) is retained precisely for them. The refactor does
  937   **not** close the set; it types built-in configs and confines the opaque seam to
  938   the open tail.
  ...
  941   **Corrected (contest O7, proven): as specified, the migration does not actually
  942   leave a selectable tail ...**
  ...
  954   So the honest current state is: **registry openness survives for id-addressed
  955   *internal* consumers, and authored selection of a runtime-loaded plugin does
  956   not exist either before or after this migration.**
  ...
  961   and that arm is the thing that reconstitutes the deleted seam. Until such an
  962   arm exists, no config can select a plugin ...
  ...
  966 - **"Zero `RawValue`" is a direction, not a literal end state.** Three residual
  967   `RawValue` uses are legitimate and stay: the plugin tail (the load-bearing one —
  968   a runtime-loaded plugin's config is opaque to the host by definition); the `dynosim` transport variant's nested Dynamo engine/router args
  969   (opaque pass-throughs to Dynamo's own parser); and the dataset payload inside
  970   the workload arm ...
  ```
  The correction itself proves the plugin-tail member of the asserted count does not survive.
resolution (r6): Proven. Same failure mode as O4: a correction appended *beneath* the text it invalidates, leaving the record asserting both halves of a contradiction.

Confirmed all four sites still stood in the delivered-state voice:
- Purpose: "the opaque `RawValue` seam confined to the three residual uses listed under Non-goals".
- Non-goals bullet 2: "plugin configs stay opaque (`RawValue`), so the coupling does not extend to the open tail".
- Non-goals bullet 4: the tail "is retained precisely for them"; "The refactor does **not** close the set".
- Residual-use bullet: "Three residual `RawValue` uses are legitimate and stay: the plugin tail (the load-bearing one) ...".

And the O7 note between the last two proves no authored `{ RegistryId, RawValue }` object survives step 4. The count is the tell: a record that has proven the tail is unreachable does not get to enumerate it as one of three things that "stay".

The objection offers "rewrite as future-plugin requirements or remove the nonexistent tail". I took the first for the substantive claims and the second for the count, since deleting the design entirely would lose the thing the eventual plugin arm needs.

Refined:
- Purpose: "two residual uses ... (a future plugin tail would be a third; per O9 it is not delivered here)".
- Bullet 2's tail clause replaced. The old justification was wrong on its own terms even setting the tail aside — the coupling isn't bounded by an open tail absorbing the untyped remainder, it's bounded by the closed variant set: it grows only when `Transport`/`workload_kind` grows, and a future plugin arm would *reintroduce* an untyped tail rather than extend this coupling.
- Bullet 4: "is retained precisely for them" → "is the shape they will need", with an explicit correction naming what the earlier draft claimed and why O7 refutes it. Dynamic plugins stay in scope as future work; the tail becomes the design that work must add.
- Residual-use bullet: three → two, plugin tail moved to a trailing conditional ("would be a third — when such a plugin can be selected").

Committed in `3abedbb193`.

One note for the record: the blanket Purpose caveat ("Where an earlier draft's open-tail language survives elsewhere, this paragraph governs") was added under O4 as a sweep for exactly this. It did not work, and O9 is the evidence. A governing-paragraph escape hatch reads as license to leave the contradictions in place; the fix is always to edit the assertions.

## O10 — The `into_authored` behavior audit is still incomplete: it omits two live workload/runtime rejections that disappear when direct typed fields replace `WorkloadConfigV2`. First, `cfg.runtime.workers = Some(0)` is explicitly rejected by `ensure!(worker_count > 0, "run.cfg.runtime.workers must be a positive usize")` in `protocol_v2.rs:378-388`; `Runtime.workers` is only `Option<u32>` and the branch has no upstream positive-value validation (workspace grep found no other check). Second, absent `cfg.phases` is serialized as JSON `null` at `protocol_v2.rs:396-405`, then rejected when factory decoding targets mandatory `WorkloadConfigV2.phases: Vec<PhaseSpec>` (`registry.rs:853-867`). After the projection/factory re-decode is deleted, `BenchmarkRun` exposes `cfg.runtime.workers: Option<u32>` and `cfg.phases: Option<Vec<Phase>>` directly, so neither rejection survives automatically. The record calls `worker_count` merely “a pure derivation” (`docs/specs/typed-factory-runner.md:681-685`), does not mention missing phases in the §2/step-4 port list, and its added unit-test list at `:864-879` covers neither case. A bare run can therefore change from a deterministic decode error to zero-worker/empty-phase downstream behavior while all specified gates remain green. The record must explicitly decide and test both behaviors before deleting the projection.
severity: high   raised: r5   status: refined
proven: yes
evidence:
  Command:
  `git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | perl -ne 'print if $. >= 374 && $. <= 406'; git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/registry.rs' | nl -ba | perl -ne 'print if $. >= 850 && $. <= 868'`

  Output:
  ```
  374 // Re-serialize the typed runtime policy ...
  376 let runtime = serde_json::to_value(&cfg.runtime) ...;
  378 let worker_count = runtime
  379     .get("workers")
  380     .and_then(Value::as_u64)
  ...
  385 ensure!(
  386     worker_count > 0 && worker_count <= usize::MAX as u64,
  387     "run.cfg.runtime.workers must be a positive usize"
  388 );
  ...
  396 let phases = serde_json::to_value(&cfg.phases) ...;
  ...
  400 let mut workload_config = serde_json::json!({
  401     "worker_count": worker_count,
  ...
  404     "phases": phases,
  ...
  853 #[derive(Deserialize)]
  854 #[serde(deny_unknown_fields)]
  855 pub struct WorkloadConfigV2 {
  856     /// Number ... workers.
  857     pub worker_count: usize,
  ...
  862     /// Phase-factory-owned authored objects.
  863     pub phases: Vec<PhaseSpec>,
  ```
  For `cfg.phases == None`, serde serializes the `Option` as `null`; `Vec<PhaseSpec>` is non-optional and does not accept `null`.

  Command:
  `git grep -n -E 'workers.*positive|workers.*> 0|phases.*(empty|is_empty|must)' 'ajc/typed-factory-runner-v2' -- 'rust/runtime/src/config/**/*.rs' 'rust/cli/src/**/*.rs'`

  Output: no matches. Thus the typed config model/resolver does not already enforce these conditions.

  Current typed shapes:
  `rust/runtime/src/config/model/runtime.rs`: `pub workers: Option<u32>`.
  `rust/runtime/src/config/model/config.rs:95-96`: `pub phases: Option<Vec<Phase>>`.

  The spec’s step-4 test list (`docs/specs/typed-factory-runner.md:864-879`) names unknown fields, outer identity/cardinality, export stems, variation, resource presence, three mandatory sections, and invalid metrics; it contains no zero-worker or missing-phases test.
resolution (r6): The best objection of the contest, and proven on both facts. One part of the consequence is overstated; I fixed the record for the accurate version, which is still a real gap.

Verified:
- `protocol_v2.rs:381-385`: `ensure!(worker_count > 0 && worker_count <= usize::MAX as u64, "run.cfg.runtime.workers must be a positive usize")`. `Runtime.workers: Option<u32>` (`config/model/runtime.rs:18`). `grep -rn "workers > 0" rust/` finds no other config-path check — the objection's negative result reproduces.
- `protocol_v2.rs:394-395`: `serde_json::to_value(&cfg.phases)` emits `null` for `None` — `skip_serializing_if` on the `BenchmarkConfig` field does not apply to a direct `to_value` of the field itself — and the factory decodes into mandatory `phases: Vec<PhaseSpec>` (`registry.rs:845-859`). `BenchmarkConfig.phases: Option<Vec<Phase>>` (`config/model/config.rs:96`).

Why this class is worse than the O7 batch: every previous audit walked `into_authored`'s *statements*. These two are enforced by the *shape* the projection hands the factory. A statement-level walk cannot find them, so "walk the body" — which is what the record kept promising — would have missed them a fourth time.

Where I diverge: "zero-worker/empty-phase downstream behavior while all specified gates remain green" does not hold. Three backstops exist:

    registry.rs:1656   ensure!(worker_count > 0, "workload worker_count must be positive")
    registry.rs:1666   ensure!(!phases.is_empty(), "workload phases cannot be empty")
    plan.rs:509-510    both again, in build_common_plan
    turn_execution.rs:296  ensure!(workers > 0, "execution workers must be positive")

So a zero-worker run still fails. What actually changes is the failure *point* and *message* — and only conditionally: `validate_common_workload` takes `(worker_count: usize, ..., phases: &[PhaseSpec])`, values the decoded DTO supplies today. Whether the typed path still routes through it is a step-3/step-4 decision the record did not make. That conditional is the defect, and stating it as an unconditional zero-worker execution would have made the record easy to dismiss.

Also caught while verifying: those same lines carry a live *default* nobody had named — absent `workers` resolves to `default_worker_count()` (machine parallelism), not `1`. A typed path that reads `cfg.runtime.workers.unwrap_or(1)` silently single-threads every run that omits the section. That is a genuine silent behavior loss on the same two lines, and it is the one I would have bet on being missed.

Refined — new §2 port-list bullet with both rejections, both anchors, the backstop table, and the "moves rather than removes, conditionally" framing. Decisions made rather than deferred:
- `workers`: check ports verbatim with its message; the `default_worker_count()` fallback moves with it.
- `phases`: absent and empty collapse on the typed model (`unwrap_or_default()`), and the record *chooses* to keep them collapsed — one message, rejecting both — rather than reconstruct the DTO's absent-vs-empty split. Recorded as a behavior change in message and failure point only.
- Neither decision permits deleting the downstream assertions; they stay as defense in depth for the cellular and eval paths that build plans without `BenchmarkRun::validate()`.

Step-4 unit-test list gains an (O10) entry covering `workers: Some(0)`, the absent-`runtime` default, and both phases cases.

Committed in `3abedbb193`.

## O11 — The projection audit still omits the tokenizer contract enforced by `AuthoredTokenizerV2::decode`. `into_authored` maps `cfg.tokenizer: None` to `{}` (`rust/runtime/src/engine/protocol_v2.rs:391-395`), but every workload then calls `validate_authored_tokenizer(&workload.tokenizer)` (`online_execution.rs:248`, `:344`, `:467`), which decodes mandatory `AuthoredTokenizerV2.name: String` and enforces `!config.name.trim().is_empty() && config.name.trim() == config.name` plus `!config.revision.trim().is_empty()` (`online_execution.rs:594-608`, `:960-985`). The only occurrences of those validation messages are in that soon-to-be-obsolete decode path. Direct typed execution sees `BenchmarkConfig.tokenizer: Option<Tokenizer>`; `Tokenizer` is a plain derived struct with no semantic validation (`config/model/tokenizer.rs:15-38`). Therefore deleting the workload DTO/re-decode can change missing tokenizer, blank/space-padded name, and blank revision from deterministic rejection to unchecked typed values. The spec's §2 audits only models/endpoint/transport/metrics and workers/phases (`docs/specs/typed-factory-runner.md:345-452`), and the step-4 test list at `:987-1010` has no tokenizer case. It must select where these three tokenizer rejections live and add guards before deleting the decode seam.
severity: high   raised: r7   status: standing
proven: yes
evidence:
  Command:
  `git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | perl -ne 'print if $. >= 388 && $. <= 406'`
  Output:
  ```
  391 let tokenizer = match cfg.tokenizer {
  392     Some(tokenizer) => serde_json::to_value(&tokenizer) ...,
  394     None => serde_json::json!({}),
  ...
  403 "tokenizer": tokenizer,
  ```

  Command:
  `git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/online_execution.rs' | nl -ba | perl -ne 'print if $. >= 228 && $. <= 249; print if $. >= 324 && $. <= 348; print if $. >= 450 && $. <= 469; print if $. >= 594 && $. <= 608; print if $. >= 960 && $. <= 985'`
  Output:
  ```
  248 validate_authored_tokenizer(&workload.tokenizer)
  ...
  344 validate_authored_tokenizer(&workload.tokenizer)?;
  ...
  467 validate_authored_tokenizer(&workload.tokenizer)?;
  ...
  594 #[derive(Debug, Deserialize)]
  595 #[serde(deny_unknown_fields)]
  596 struct AuthoredTokenizerV2 {
  597     name: String,
  598     #[serde(default = "default_revision")]
  599     revision: String,
  ...
  960 impl AuthoredTokenizerV2 {
  961     fn decode(raw: &RawValue) -> Result<Self> {
  ...
  964         ensure!(
  965             !config.name.trim().is_empty() && config.name.trim() == config.name,
  966             "tokenizer.name must be non-empty and contain no surrounding whitespace"
  967         );
  968         ensure!(
  969             !config.revision.trim().is_empty(),
  970             "tokenizer.revision must not be empty"
  971         );
  ```

  Command:
  `git grep -n -E 'tokenizer\.name must be non-empty|tokenizer\.revision must not be empty' 'ajc/typed-factory-runner-v2' -- 'rust/**/*.rs'`
  Output:
  ```
  ajc/typed-factory-runner-v2:rust/runtime/src/engine/online_execution.rs:966: "tokenizer.name must be non-empty and contain no surrounding whitespace"
  ajc/typed-factory-runner-v2:rust/runtime/src/engine/online_execution.rs:970: "tokenizer.revision must not be empty"
  ```

  `rust/runtime/src/config/model/tokenizer.rs:15-38` is only `#[derive(Clone, Debug, Serialize, Deserialize)] pub struct Tokenizer { pub name: String, #[serde(default = "default_revision")] pub revision: String, ... }` with no `try_from` or validation.

## O12 — The typed repoint still has no explicit requirement or live gate for `parse_dispatch_mode`'s cellular-aware default. Current `protocol_v2.rs:249-277` derives `DispatchMode::Global` when `runtime.cells <= 1` but `DispatchMode::Sharded` when `runtime.cells > 1`; this is not equivalent to reading typed `runtime.dispatch.unwrap_or_default()`. The spec mentions `parse_dispatch_mode` only as a “pure derivation” (`docs/specs/typed-factory-runner.md:805-810`) without specifying the `cells` branch, and the step-4 named tests at `:987-1010` cover only the absent-worker default, not dispatch. Existing exact unit tests (`runtime_dispatch_defaults_to_global_when_absent`, `runtime_dispatch_defaults_to_sharded_for_cellular`, `runtime_explicit_dispatch_wins_over_cellular_default`) are inside `#[cfg(any())] mod tests` (`protocol_v2.rs:1316`) and therefore never run. None of the five listed cellular e2e files contains `runtime.dispatch` or `DispatchMode::Sharded` (workspace grep returned no matches). A direct typed implementation can silently switch every cellular run lacking explicit dispatch from `Sharded` to `Global` while all specified gates remain green; the exact derivation and its three disabled tests must move to the typed model/gate before `into_authored` is deleted.
severity: high   raised: r7   status: standing
proven: yes
evidence:
  Command:
  `git show 'ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs' | nl -ba | perl -ne 'print if $. >= 249 && $. <= 277; print if $. >= 1703 && $. <= 1734'`
  Output:
  ```
  249 /// Decode the optional `runtime.dispatch` admission-strategy selector.
  251 /// An explicit `runtime.dispatch` always wins. When it is absent, the default is
  252 /// **cellular-aware**:
  253 /// - single-process (`cells <= 1`) → [`DispatchMode::Global`]
  255 /// - cellular (`cells > 1`) → [`DispatchMode::Sharded`].
  265 fn parse_dispatch_mode(runtime: &Value) -> Result<DispatchMode> {
  266     match runtime.get("dispatch") {
  267         None | Some(Value::Null) => {
  268             let cells = runtime.get("cells").and_then(Value::as_u64).unwrap_or(1);
  269             if cells > 1 {
  270                 Ok(DispatchMode::Sharded)
  271             } else {
  272                 Ok(DispatchMode::default())
  ...
  1704 fn runtime_dispatch_defaults_to_global_when_absent() {
  ...
  1713 fn runtime_dispatch_defaults_to_sharded_for_cellular() {
  ...
  1726 fn runtime_explicit_dispatch_wins_over_cellular_default() {
  ```

  Command:
  `git grep -n '#\[cfg(any())\]' 'ajc/typed-factory-runner-v2' -- 'rust/runtime/src/engine/protocol_v2.rs'`
  Output:
  ```
  ajc/typed-factory-runner-v2:rust/runtime/src/engine/protocol_v2.rs:1316:#[cfg(any())]
  ```

  Command:
  `git grep -n -E 'DispatchMode::Sharded|runtime.dispatch' 'ajc/typed-factory-runner-v2' -- 'rust/e2e-tests/tests/test_cellular.rs' 'rust/e2e-tests/tests/test_graph_cellular.rs' 'rust/e2e-tests/tests/test_grpc_cellular.rs' 'rust/e2e-tests/tests/test_recorded_agent_cellular.rs' 'rust/e2e-tests/tests/test_cellular_dataset_shipping.rs'`
  Output: no matches.

  Spec output:
  ```
  docs/specs/typed-factory-runner.md:807-810 ... pure derivation (`workload_kind`, `parse_dispatch_mode`, `worker_count` from `available_parallelism`) ...
  docs/specs/typed-factory-runner.md:1006-1010 tests absent worker and phases only
  ```

## O13 — The O8 refinement fixed step 1's normative text but left the executable verification gate requiring the impossible/wrong assertion it explicitly retired. `docs/specs/typed-factory-runner.md:620-634` says “identical bindings” was wrong, component equality is the sufficient landed gate, and “No binding-level differential is owed at any step.” Yet the Step 1 gate at `:917-922` still requires “a temporary differential assertion that both paths produce identical bindings.” This is not cosmetic: the gate is the command/checklist an implementer follows, and it now conflicts with the selected design and the author’s O8 resolution. Change that line to identical components and name `transport_component_matches_inline_projection`, or step 1 remains formally unfinishable under its own gate.
severity: medium   raised: r7   status: standing
proven: yes
evidence:
  Command:
  `nl -ba docs/specs/typed-factory-runner.md | perl -ne 'print if $. >= 611 && $. <= 638; print if $. >= 904 && $. <= 923'`
  Output:
  ```
  616 Step 1 ... "add a
  617 consumer that reads `cfg.transport` directly", run alongside the existing
  618 projection, and assert the two produce identical **components**.
  620 **Corrected ... "identical bindings" was the wrong
  621 assertion to write here, and is not what landed.**
  ...
  630 Pinning both halves byte-exact pins the binding transitively. That is precisely
  631 what the landed `transport_component_matches_inline_projection` asserts ...
  634 No binding-level differential is owed at any step.
  ...
  917 - **Step 1** (a `cfg.transport` consumer running alongside the projection): a
  918   temporary differential assertion that both paths produce identical bindings,
  919   plus `cargo test ...
  ```
  The latter is the exact phrase the correction rejects.

## Unresolved risks

- O11 — The projection audit still omits the tokenizer contract enforced by `AuthoredTokenizerV2::decode`. `into_authored` maps `cfg.tokenizer: None` to `{}` (`rust/runtime/src/engine/protocol_v2.rs:391-395`), but every workload then calls `validate_authored_tokenizer(&workload.tokenizer)` (`online_execution.rs:248`, `:344`, `:467`), which decodes mandatory `AuthoredTokenizerV2.name: String` and enforces `!config.name.trim().is_empty() && config.name.trim() == config.name` plus `!config.revision.trim().is_empty()` (`online_execution.rs:594-608`, `:960-985`). The only occurrences of those validation messages are in that soon-to-be-obsolete decode path. Direct typed execution sees `BenchmarkConfig.tokenizer: Option<Tokenizer>`; `Tokenizer` is a plain derived struct with no semantic validation (`config/model/tokenizer.rs:15-38`). Therefore deleting the workload DTO/re-decode can change missing tokenizer, blank/space-padded name, and blank revision from deterministic rejection to unchecked typed values. The spec's §2 audits only models/endpoint/transport/metrics and workers/phases (`docs/specs/typed-factory-runner.md:345-452`), and the step-4 test list at `:987-1010` has no tokenizer case. It must select where these three tokenizer rejections live and add guards before deleting the decode seam.
- O12 — The typed repoint still has no explicit requirement or live gate for `parse_dispatch_mode`'s cellular-aware default. Current `protocol_v2.rs:249-277` derives `DispatchMode::Global` when `runtime.cells <= 1` but `DispatchMode::Sharded` when `runtime.cells > 1`; this is not equivalent to reading typed `runtime.dispatch.unwrap_or_default()`. The spec mentions `parse_dispatch_mode` only as a “pure derivation” (`docs/specs/typed-factory-runner.md:805-810`) without specifying the `cells` branch, and the step-4 named tests at `:987-1010` cover only the absent-worker default, not dispatch. Existing exact unit tests (`runtime_dispatch_defaults_to_global_when_absent`, `runtime_dispatch_defaults_to_sharded_for_cellular`, `runtime_explicit_dispatch_wins_over_cellular_default`) are inside `#[cfg(any())] mod tests` (`protocol_v2.rs:1316`) and therefore never run. None of the five listed cellular e2e files contains `runtime.dispatch` or `DispatchMode::Sharded` (workspace grep returned no matches). A direct typed implementation can silently switch every cellular run lacking explicit dispatch from `Sharded` to `Global` while all specified gates remain green; the exact derivation and its three disabled tests must move to the typed model/gate before `into_authored` is deleted.
- O13 — The O8 refinement fixed step 1's normative text but left the executable verification gate requiring the impossible/wrong assertion it explicitly retired. `docs/specs/typed-factory-runner.md:620-634` says “identical bindings” was wrong, component equality is the sufficient landed gate, and “No binding-level differential is owed at any step.” Yet the Step 1 gate at `:917-922` still requires “a temporary differential assertion that both paths produce identical bindings.” This is not cosmetic: the gate is the command/checklist an implementer follows, and it now conflicts with the selected design and the author’s O8 resolution. Change that line to identical components and name `transport_component_matches_inline_projection`, or step 1 remains formally unfinishable under its own gate.
