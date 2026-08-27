# Graham review — `f53eb2fee3` fix(metrics): restore dense metric tag declaration order

Reviewer: independent adversarial pass. Base `eadd5c665f`, HEAD `f53eb2fee3`,
worktree `.worktrees/native-plugin-metrics-regression`.
Diff: 1 file, `rust/runtime/src/metrics_core/tag_id.rs`, +3/-3. Worktree clean; no
files modified by this review.

## Verdict

**NO-GO** — C0 / **I1** / **M1** / S4.

The code change itself is correct, minimal, and exactly right. Both blocking
findings are about what the change *leaves standing*, not about what it does.
Both are cheap to close.

| # | Class | Finding |
|---|---|---|
| I1 | Important | The mandated `--features engine` gate was never run for this fix, and it is red (exit 101). The commit's own verification record asserts "the suite is fully green after the fix" — that claim is false. |
| M1 | Minor | `tag_id.rs` still carries no comment that `define_builtin_metric_tags!` declaration order is a positional contract with `CATALOG`. The exact trap that produced the regression is left un-signposted in the exact file being edited. |

---

## Gate results (run by me, from the worktree `rust/`, all `RUSTFLAGS="--cfg tokio_unstable"`)

| gate | exit | result |
|---|---|---|
| `cargo test -p aiperf-runtime --lib` | **0** | 1914 passed, 0 failed, 7 ignored |
| `cargo test -p aiperf-runtime --features engine --lib` | **101** | 2531 passed, **6 failed**, 7 ignored (601.5s) |
| `cargo test -p aiperf-runtime --lib metrics_core::` | **0** | 137 passed, 0 failed |

`rustfmt --edition 2024 --check runtime/src/metrics_core/tag_id.rs` → exit 0. The
touched file is clean; the three known pre-existing fmt offenders are untouched
and not reported.

### Attribution of the six engine failures — none caused by this diff

I did not take "pre-existing" on faith. I re-ran the four non-fixture failures at
**base `eadd5c665f`** in the root repository (`/home/anthony/nvidia/projects/aiperf/ajc/rust`,
`HEAD == eadd5c665f`) with the identical invocation.

| failing test | attribution | evidence |
|---|---|---|
| `engine::graph_input::tests::recorded_agent_adapter_discovers_and_lowers_the_manifest_corpus` | **worktree environment** | Needs `rust/runtime/tests/fixtures/recorded_agent_replay/recordings/pinchbench-*.json`. `git ls-files` shows only `inspection.json` is tracked; the pinchbench recordings exist untracked in the root repo and therefore do not exist in any worktree. Not a defect. |
| `engine::graph_input::tests::recorded_agent_tool_execution_stages_pinch_task_pack_workspace_files` | **worktree environment** | Same missing untracked fixtures. |
| `engine::artifact_stream_velo::tests::velo_stream_large_artifact_round_trips_with_bounded_memory` | **pre-existing at base** | Fails at `eadd5c665f` in the root repo (with fixtures present), at both the isolated-RSS-child level and the outer level: `RSS grew 97423360 bytes during a 67108864-byte transfer`. |
| `engine::online_execution::transport_binding_differential::match_arm_binding_matches_the_registry_lookup_for_every_variant` | **pre-existing at base** | Fails at `eadd5c665f`: `online_execution.rs:2031` — `product_registry().transport_factory("http")` returns `None`. |
| `engine::registry::tests::workload_resources_fail_required_and_forbidden_presence_before_transport_prepare` | **pre-existing at base** | Fails at `eadd5c665f`: `registry.rs:1914` — `unknown variant \`acme_remote\`, expected one of \`http\`, \`grpc\`, \`dynosim_offline\`, \`dynosim_online\`, \`dry_run\`, \`websocket\``. |
| `engine::workers_characterization::tests::user_centric_workers_gt_1_thread_per_core_data_matches_single_thread` | **load-flaky** | Passes at base; passes in this worktree when run in isolation (exit 0, 3.21s). It failed only inside the 601s full-suite run while a second cargo job was competing for cores. |

Net: **zero of the six is attributable to the three-line diff.** The diff is
exonerated. The gate is still red, which is finding I1.

---

## Q1 — Does the fix restore the original discriminant order exactly, for all tags?

**Yes. Exactly, for all 138 tags, index-for-index, and the name mapping too.**

Method (mechanical, not eyeball):

1. Extracted the 138 ordered variants of the pre-refactor `pub enum MetricTag`
   from `git show f247b0102d:rust/runtime/src/metrics_core/catalog.rs`. I first
   confirmed that file is **byte-identical** to `e8456a5720^`
   (`e4404a6fdf6bbde7d06e701ee7c5ffdae4a2f8ca`), so `f247b0102d` is a valid stand-in
   for "immediately before the refactor".
2. Extracted the 138 ordered variants of `define_builtin_metric_tags!` at HEAD.
3. `diff` → **no differences.** 138 lines each, identical sequence.
4. Extracted the 138 `variant → "report_name"` pairs from the original `as_str`
   match (including the four multi-line arms — `NetworkAdjustedTimeToFirstOutputToken`,
   `EffectiveImageSamplesPerSecondPerUser`, `SpecDecodeTokenWeightedAcceptanceLength`,
   `SpecDecodeOverallDraftAcceptanceRate`) and diffed against HEAD's `=> "..."`
   literals → **identical, 138/138.** No report spelling drifted.

**The trap the brief warned about is real and is exactly what happened.** In
`f247b0102d`'s `catalog.rs`, declaration order (lines 22-160) and `as_str` match-arm
order (lines 176+) genuinely disagree. The match arms read:

```
TimeToFirstOutputToken, TimeToLastRoundTrip, AverageRoundTripTime,
InterTokenLatency, DecodeDuration, InterChunkLatency, CreditDropLatency
```

while the declaration reads `TimeToFirstOutputToken, InterTokenLatency,
InterChunkLatency, DecodeDuration, CreditDropLatency` with the two round-trip tags
at 126/127. I diffed the broken `e8456a5720` macro order against both: it matches
the **match-arm** order and not the declaration order, in both disturbed regions.
So the transcription was done off the cosmetic list, not the load-bearing one.
Root cause confirmed independently.

Corroboration from the running code rather than from source archaeology: gate 3
passes `catalog_is_discriminant_ordered` (`CATALOG[i].tag.index() == i` for all i)
and `MetricTag::COUNT == 138`. Combined with (3), `CATALOG` row order at HEAD equals
the historical enum order transitively.

Serde is unaffected in either direction: the old `MetricTag` derived `Serialize`
with no `rename_all`, so it emitted PascalCase variant names; the new
`BUILTIN_SERDE_NAMES` is `stringify!($variant)`, i.e. the same PascalCase names.
Wire and artifact spellings are byte-identical across the refactor and across this fix.

## Q2 — Is positional `CATALOG` indexing the actual mechanism?

Yes, verbatim, `rust/runtime/src/metrics_core/catalog.rs:2034-2041`:

```rust
pub const fn metric_definition(tag: MetricTag) -> &'static Definition {
    &CATALOG[tag.index()].def
}
pub const fn spec_for(tag: MetricTag) -> Option<&'static MetricSpec> {
    Some(&CATALOG[tag.index()])
}
```

`MetricTag` is now `pub type MetricTag = tag_id::MetricTagId` (catalog.rs:19), and
`MetricTagId::index()` is `self.0 as usize` (tag_id.rs:200-202) — the macro's
`BuiltinMetricTag` discriminant. `CATALOG` is a 138-row `static` whose order was
never touched. So a shift of any macro position silently re-points that tag at a
neighbouring row's unit, `MetricValueType`, `MetricConsoleGroup`, `MetricFlags`
(including `LARGER_IS_BETTER` and the rate-derivation flags) and dependency list.
Mechanism confirmed.

## Q3 — Is the ordering invariant now guarded so this cannot silently recur?

**No new guarantee is added. The fix relies entirely on the same
`catalog_is_discriminant_ordered` test that was already ignored once.**

State of the guards at HEAD (all pre-existing, none added by this commit):

- `catalog_is_discriminant_ordered` (catalog.rs:2247) — the direct tripwire.
- `websocket_lag_tags_append_after_existing_dense_identities` (catalog.rs:2253) — pins
  `TimeToLastRoundTrip.index() == ActiveTotalThroughput.index() + 1` and `+2`. This
  is already a name-to-relative-index pin and would have caught the regression too.
- `metric_definition_matches_catalog` (catalog.rs:2266) — pointer equality.
- `assert_eq!(MetricTag::COUNT, 138)` (catalog.rs:2243).

Detection coverage is therefore *adequate*; three independent tests fired. The
failure mode was **not** missing coverage — it was a red gate merged under a GO
verdict. Adding a fourth runtime test would change nothing.

**Is a structural guarantee warranted? Yes — and I want to be concrete about the
cost, because the obvious answer is more expensive than it looks.**

- *Compile-time assertion (`const _: () = { ... assert!(CATALOG[i].tag.index() == i) ... }`)* —
  **not free.** `CATALOG` is a `static`, and Rust constants cannot read statics
  ("constants cannot refer to statics"). Making this work requires converting
  `CATALOG` from `static` to `const`, which changes the borrow/promotion semantics
  of a table that is handed out as `&'static MetricSpec` on a hot lookup path. That
  is a real, reviewable change to a hot-path table, not a one-liner. I would not ask
  for it in this commit.
- *Single source of truth (generate the `CATALOG` rows from the same macro that
  declares the tags)* — the only change that makes the class of bug **impossible**
  rather than **detected**. Cost: a rewrite of 138 richly-annotated `MetricSpec`
  rows into macro invocations. Correct long-term direction; a separate PR.
- *One comment line in `tag_id.rs`* — cost ≈ zero, and it addresses the actual
  proximate cause (an editor working in `tag_id.rs` sees nothing telling them the
  list order is load-bearing). **This is finding M1 and I am asking for it here.**

Recommendation: land M1's comment now; file the macro-generated-`CATALOG` unification
as follow-up; do not attempt the `const` conversion inside a regression fix.

## Q4 — Other consumers of tag identity where positional corruption reaches an artifact or wire format?

**Yes — three, and one of them is a genuine wire format.** I grepped every
`.index()` call site and every ordered container keyed by `MetricTag`.

1. **`ColumnStore` crosses the cellular wire with tag-ordinal-positional layout.**
   `rust/runtime/src/metrics_core/store.rs:795` — `#[derive(Serialize, Deserialize)] pub struct ColumnStore`
   with `numeric: Vec<Option<NumericColumn>>` and `ragged: Vec<Option<B>>`, both
   allocated as `(0..MetricTag::COUNT)` (store.rs:873, 875) and addressed by
   `tag.index()` (store.rs:918, 930, 1143, 1154, 1168, 1179, 1191, 1274, 1284, 1289).
   That store is wrapped as `ColumnStorePartition` (`cellular/shard.rs:348`) and
   serialized with `rmp_serde` between cell and controller. The struct already treats
   its field layout as a wire contract ("New positional MessagePack fields must remain
   after every legacy field") — but **nothing pins the tag ordinal**, which is an
   equally positional part of that same payload.
2. **`SketchColumns` embeds the ordinal in a serialized map key.** store.rs:665 —
   `tags: FxHashMap<(Phase, Option<usize>, u16), TagSketch>`, where the `u16` is
   literally `tag.index() as u16` (store.rs:729, 734, 742, 752). `SketchColumns` is
   `Serialize, Deserialize` and travels inside `ColumnStore.sketch`.
   Consequence for both (1) and (2): a controller and a cell built from different
   images — entirely plausible during a K8s or SLURM rolling deploy — would fold
   metrics into the **wrong metric** with no error, no version check, and no digest.
   This is the data-integrity exposure the brief asked about, and it is latent today
   only because every process in a run is the same binary.
3. **Ordinal ordering is observable in output.** `gpu_telemetry/accumulator.rs:140` —
   `injections: BTreeMap<MetricTag, MetricValue>`. `MetricTagId` derives `Ord` on the
   `u32` index, so iteration order is discriminant order. Under the base regression
   this silently reordered GPU-telemetry injection iteration.

**Bonus defect found in the same area (not caused by this diff, S1 below):** the
interning refactor's advertised capability does not actually work end to end.
`spec_for` is typed `Option<&'static MetricSpec>` but is unconditionally
`Some(&CATALOG[tag.index()])`. For any tag registered through
`MetricTagRegistry::register` — which assigns `index >= 138` from a process-global
interner, and which `tag_id.rs`'s own test
`a_new_metric_tag_registers_without_touching_the_enum` advertises as supported — this
is an **out-of-bounds index panic**, not a `None`. The four defensive
`let Some(spec) = spec_for(tag) else { … }` sites (accumulator.rs:703, 1149, 1165,
1621) are consequently dead code that can never take their `else` branch, and
`ColumnStore.numeric[tag.index()]` panics identically. No production caller registers
a dynamic tag today, so this is latent — but it is exactly the seam the plugin work
is building toward.

**Endpoint side is clean.** I checked the sibling transcription in the same
root-cause commit (`endpoints/type_id.rs`, `EndpointTypeId`), since it applied the
same closed-enum → interned-id pattern. Its 19 built-ins are order-preserving against
the deleted enum, `canonical_id`/`from_canonical_id` round-trip correctly, and the
legacy `"chat_completions" => "chat"` alias survived (type_id.rs:130-134). No second
transcription defect there.

## Q5 — Were the 18 failures all one cause, or does the green suite hide a second defect?

**The 18 `--lib` failures were all one cause. But the green `--lib` suite does hide
defects — they are in the `--features engine` suite that gate 1 cannot reach.**

Evidence that the 18 were one cause:

- Total test count is **identical** before and after: base 1896 passed + 18 failed +
  7 ignored = 1921; HEAD 1914 passed + 0 failed + 7 ignored = 1921. No test was
  added, deleted, renamed, `#[ignore]`d, or re-baselined.
- The diff touches zero test code and zero assertion values — it is pure reordering
  of a declaration list.
- Every named failure class in the author's analysis (catalog ordering, accumulator
  spec resolution, report rate-stat derivation, downstream metrics/accuracy/gpu_telemetry
  consumers) is a direct consequence of `&CATALOG[tag.index()]` resolving to a
  neighbouring row. There is no residual failure to explain.

What the green `--lib` suite hides (all proven pre-existing at base, all invisible
without `--features engine`):

- `engine::registry::tests::workload_resources_fail_required_and_forbidden_presence_before_transport_prepare`
  — `AuthoredRunSpecV2`'s `transport.type` still decodes through a **closed serde
  enum**: `unknown variant \`acme_remote\`, expected one of \`http\`, \`grpc\`,
  \`dynosim_offline\`, \`dynosim_online\`, \`dry_run\`, \`websocket\``. This directly
  contradicts the plugin-ABI premise of the Task-6 merge that is the parent commit,
  and its sibling test `custom_selection_validates_without_a_core_enum_or_match`
  passes only because it bypasses the DTO decode. This one deserves its own
  investigation on the root branch.
- `engine::online_execution::transport_binding_differential::match_arm_binding_matches_the_registry_lookup_for_every_variant`
  — `product_registry().transport_factory("http")` returns `None`.
- `engine::artifact_stream_velo::tests::velo_stream_large_artifact_round_trips_with_bounded_memory`
  — RSS bound violated by ~45%.

None of these is a *second cause of the 18*. All three are separate root-branch red.

---

## Findings

### I1 (Important, blocks) — mandated engine gate not run; verification record claims a green it does not have

`CLAUDE.md` is explicit and unambiguous on this point:

> `aiperf-runtime`'s `engine` module … sits behind the `engine` Cargo feature, off by
> default; `cargo test -p aiperf-runtime` alone silently runs zero tests under it, so
> run **both** invocations above.

The change's verification record
(`.superpowers/sdd/…/metrics-regression-analysis.md`, "Gates" table) lists only
`cargo test -p aiperf-runtime --lib`, and states "the suite is fully green after the
fix." I ran the mandated second gate: **exit 101, 6 failures.**

I proved every one of those six is attributable to base debt, missing untracked
worktree fixtures, or load flakiness — so the *fix* is fine. That is precisely why
this is Important and not Critical. But the reasoning that exonerates the fix was
done by me, in review, not by the author before claiming green. A change to the
shared 138-row metric identity table — consumed heavily by `engine::` code that the
default feature set does not compile — is the last change where you get to skip the
engine gate. And "fully green" is a false statement about a branch that exits 101 on
a gate the project mandates.

To close: run `RUSTFLAGS="--cfg tokio_unstable" cargo test -p aiperf-runtime --features engine --lib`,
record exit 101, and record the per-test attribution (the base-control run in the root
repo at `eadd5c665f` is the cheap way to do it — it takes one command). Correct the
"fully green" claim to "green on `--lib`; engine gate red with N failures, all
attributed to base". No code change required.

### M1 (Minor, blocks) — the load-bearing ordering constraint is still undocumented in the file that carries it

`rust/runtime/src/metrics_core/tag_id.rs:41` opens `define_builtin_metric_tags! {`
with 138 entries and **no indication anywhere in the file** that the order of that
list is a contract. The module doc is one line, `//! Open metric identity.` The
warning exists — but on the *other* side of the coupling, at `catalog.rs:2031`
("`CATALOG` is ordered by declaration discriminant, guarded by the
`catalog_is_discriminant_ordered` test"), where an editor reordering `tag_id.rs` has
no reason to look.

That asymmetry *is* the regression's proximate cause. Someone transcribed a list,
tidied it into a more logical grouping, and nothing in front of them said not to.
Restoring the order without restoring the signpost leaves the trap fully armed and
re-armed for the next mechanical edit — and the demonstrated consequence is silently
wrong units and rate-derivation flags in customer artifacts.

`CLAUDE.md` makes this mandatory, not stylistic: *"Comments explain non-obvious
constraints and interactions, not syntax."* A cross-file positional coupling between
a macro list and a `static` array in another module is the textbook non-obvious
interaction.

To close — one comment above line 41, e.g.:

```rust
// Declaration order is a positional contract: `MetricTagId::index()` is this
// list's zero-based position, and `catalog::metric_definition`/`spec_for` index
// `CATALOG` with it. Reordering these entries silently re-points every shifted
// tag at a neighbouring metric's spec. Append new built-ins at the end only;
// `catalog_is_discriminant_ordered` is the tripwire.
```

### S1 (Suggestion) — `spec_for`'s `Option` is a lie, and dynamic tags panic

`spec_for` (catalog.rs:2039) is `Option<&'static MetricSpec>` but returns
unconditional `Some(&CATALOG[tag.index()])`. Any tag from
`MetricTagRegistry::register` has `index >= 138` and produces an out-of-bounds panic
instead of `None`. The four `let Some(spec) = … else` sites at accumulator.rs:703,
1149, 1165, 1621 are unreachable-`else` dead code. `ColumnStore.numeric`/`ragged`
(sized `MetricTag::COUNT`) panic the same way. Either bound-check
(`CATALOG.get(tag.index())`) and let the existing `None` branches do their job, or
drop the `Option` and rename the accessor so callers stop writing defensive code that
cannot run. Not caused by this diff; introduced with the interning refactor.

### S2 (Suggestion) — pin the tag ordinal on the cellular wire

Per Q4(1) and Q4(2), the tag ordinal is embedded positionally in `ColumnStore` and as
a `u16` map key in `SketchColumns`, both serialized across the cell↔controller
boundary via `rmp_serde`. Consider carrying a cheap ordering digest (e.g. BLAKE3 over
`BUILTIN_NAMES`) in the partition header and rejecting a mismatched peer, so
controller/cell build skew fails loudly instead of folding metrics into the wrong
slot. Same class of protection the codebase already applies elsewhere.

### S3 (Suggestion) — unify `CATALOG` with the tag macro (follow-up PR)

The only change that makes this bug class impossible rather than merely detected. See
the cost analysis in Q3: the `const`-assertion shortcut is blocked by `CATALOG` being
a `static`, so the real answer is generating the rows from the same macro. Worth
doing; not in this commit.

### S4 (Observation) — the root branch is red under the engine gate

Three genuine pre-existing failures at `eadd5c665f`, verified by me in the root repo.
The `acme_remote` closed-enum decode in particular looks like an unclosed gap in the
plugin ABI work that the parent merge commit claims to have delivered. Independent of
this fix; flagging so it is not lost.

---

## What is good here

Worth saying plainly, because the blocking findings are both about surroundings:

- The diagnosis is correct and the fix is the *right* fix — restore the data, do not
  paper over it. Three lines, one file, zero test edits, zero `#[allow]`s, no
  re-baselining. The temptation to "fix" this by reordering `CATALOG` instead, or by
  relaxing `catalog_is_discriminant_ordered`, would have been much worse and was
  correctly resisted.
- The author's written analysis correctly identifies the match-arm-vs-declaration trap
  and correctly concludes the tests were right and production was wrong.
- The restoration is exact — not approximately right, not "the failing ones", but all
  138 positions and all 138 report spellings, which I verified mechanically rather
  than by reading.
