# Native Baseten Columnar Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace native Baseten's full-row, whole-file JSON staging with projected bounded Parquet/Arrow IPC scans while preserving replay behavior and proving the improvement with the real Rust adapter.

**Architecture:** A private columnar source in the existing Baseten loader owns schema inspection and projected batch readers. A metadata pass resolves minimum time and deterministic session selection; a projected trace pass directly constructs retained `BasetenRow`s and passes the chosen grouping key through `RawRow::group_key` to the existing composer.

**Tech Stack:** Rust 2024, `arrow` IPC/arrays, `parquet` Arrow reader, Tokio current-thread tests, existing Python-compatible RNG, existing `dataset_load_bench` release adapter.

**Spec:** `docs/specs/2026-08-25-native-baseten-columnar-load.md`

## Global Constraints

- Work only in `/mnt/4tb/aiperf-origin-port-039`; do not touch the shared checkout.
- Preserve exact merge ancestry: merge commit `da917561fb` has second parent `1e32a51318af3474dc1672d225fd495471ab45df`.
- Use `/usr/bin/sccache` and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-039-target` for every Cargo command.
- Do not stage the pre-existing unstaged generated-file residuals in `docs/environment-variables.md` or `src/aiperf/config/schema/aiperf-config.schema.json`.
- Do not port outcome/fidelity columns owned by tracker #40.
- No `unwrap()`/`expect()` in production code; no new synchronization or dependency.
- Keep record batches at 128 rows and project only mode-required columns.

---

### Task 1: Columnar format detection and Arrow IPC parity

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/runtime/src/dataset/loader/baseten.rs`

**Interfaces:**
- Produces: local source detection for `.parquet`, `.arrow`, and `.ipc` with the four required columns.
- Produces: private `ColumnarSource::open(path: &Path) -> Result<Self>` and projected batch iteration used by Task 2.

- [ ] **Step 1: Write failing format and parity tests**

Add a fixture writer that emits the same required/session/timing rows through
`parquet::arrow::ArrowWriter` and `arrow::ipc::writer::FileWriter`. Assert both
Arrow suffixes pass `can_load`, a missing-column Arrow file does not, and the
composed Arrow dataset equals the Parquet dataset in session ids, timestamps,
prompt text, output caps, and extra request bodies.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-039-target \
  cargo test -p aiperf-runtime --features parquet dataset::loader::baseten --lib
```

Expected: Arrow detection/parity assertions fail because native accepts only
`.parquet` and has no IPC reader.

- [ ] **Step 3: Implement the minimal columnar source**

Enable only Arrow's existing `ipc` feature. Add a source kind enum, one retained
file descriptor, schema lookup, and `read_batches(&[&str])`. Parquet builds a
128-row `ParquetRecordBatchReader` with `ProjectionMask::columns`; IPC maps
column names to indices and creates `FileReader` with that projection. Convert
all reader errors to path-qualified `DatasetError::Validation`.

- [ ] **Step 4: Run the focused suite and verify GREEN**

Run the Step 2 command. Expected: all Baseten loader tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add rust/Cargo.toml rust/runtime/src/dataset/loader/baseten.rs
git commit -m "feat: read Baseten Arrow IPC traces"
```

### Task 2: Projected decoding and metadata-first sampling

**Files:**
- Modify: `rust/runtime/src/dataset/loader/baseten.rs`

**Interfaces:**
- Consumes: `ColumnarSource::read_batches` from Task 1.
- Produces: `SessionSelection { key, kept_ids, kept_null_ordinals }` and direct record-batch-to-`BasetenRow` decoding.
- Produces: `RawRow::group_key` as the sole grouping decision consumed by the composer.

- [ ] **Step 1: Write failing behavioral tests**

Add three focused tests:

```rust
#[tokio::test]
async fn projected_load_ignores_wide_unused_columns_and_validates_late_rows() {
    let fixture = write_columnar_fixture(131, Some((130, "input_tokens", -1)));
    let error = load_baseten(fixture.parquet, Map::new()).await.unwrap_err();
    assert!(error.to_string().contains("input_tokens"));
    assert!(take_scan_observations().iter().all(|scan| {
        scan.batch_rows <= 128 && !scan.columns.contains(&"unused_blob".to_string())
    }));
}

#[tokio::test]
async fn configured_session_column_falls_back_and_controls_grouping() {
    let fixture = write_session_policy_fixture();
    let dataset = load_baseten(fixture.parquet, Map::new()).await.unwrap();
    assert_eq!(session_turn_counts(&dataset), vec![2, 2]);
    let fallback = load_baseten(fixture.only_poor_man, Map::new()).await.unwrap();
    assert_eq!(session_turn_counts(&fallback), vec![4]);
}

#[tokio::test]
async fn sampled_metadata_keeps_whole_sessions_without_decoding_dropped_prompts() {
    let fixture = write_sampling_fixture();
    let options = serde_json::from_value(json!({
        "trace_session_sample_ratio": 0.5
    })).unwrap();
    let first = load_baseten(fixture.clone(), options.clone()).await.unwrap();
    let second = load_baseten(fixture, options).await.unwrap();
    assert_eq!(session_prompts(&first), session_prompts(&second));
    assert!(session_turn_counts(&first).iter().all(|count| *count == 2));
}
```

The fixture exposes scan projections/batch lengths through a test-only observer.
Assert the default open-loop projection is exactly required columns + chosen
session + KV hints; closed loop adds `duration_e2e_ms`; `omit_kv_hints` removes
hash/block columns; every observed batch has at most 128 rows. Put an invalid
required value at row 130 to prove validation is not sample-only.

- [ ] **Step 2: Run the focused test and verify RED**

Run the Task 1 test command. Expected: projection observer shows unused columns
decoded, the configured session policy is absent, or late malformed data is not
handled by the new direct decoder.

- [ ] **Step 3: Implement metadata and trace scans**

Parse `AIPERF_DATASET_BASETEN_SESSION_COLUMN` with exact accepted values and
schema fallback. Scan timestamp/session metadata, reproduce deterministic
session/null-row sampling with the existing RNG namespace, then project the
mode-specific trace columns. Cast batch columns to checked canonical Arrow
types once per batch, decode retained rows, normalize/filter them, construct one
intermediate `Value`, serialize that value once, and set `group_key`.

- [ ] **Step 4: Make the composer consume the grouping key**

Carry `(BasetenRow, Option<String>)` while parsing `RawRow`s. Group by the
provided key; generate `baseten_NNNNNN` only for null/no-key rows. Delete the
old post-load session scoring/sampling path after its replacement tests are
green.

- [ ] **Step 5: Run the focused suite and verify GREEN**

Run the Task 1 command. Expected: all Baseten tests pass with exact projection,
batch, validation, grouping, and deterministic sampling assertions.

- [ ] **Step 6: Commit Task 2**

```bash
git add rust/runtime/src/dataset/loader/baseten.rs
git commit -m "perf: project Baseten trace batches"
```

### Task 3: Public adapter integration and measured closure

**Files:**
- Modify: `rust/runtime/tests/dataset_load_bench.rs`
- Modify: `docs/porting-origin-main-campaign.md`
- Create: `.superpowers/sdd/2026-08-25-native-baseten-columnar-load/progress.md`
- Create: `.superpowers/sdd/2026-08-25-native-baseten-columnar-load/graham-review.md`

**Interfaces:**
- Consumes: public `dataset_load_bench::measure` and the completed loader.
- Produces: closure evidence tying exact ancestry, test mapping, and before/after benchmark samples to tracker #39.

- [ ] **Step 1: Write a failing public-adapter Arrow integration test**

Generate equivalent Parquet and Arrow IPC fixtures through shared Arrow arrays,
call `measure` for `baseten_trace`, and compare non-timing fields:
`row_count`, `conversation_count`, `turn_count`, and `total_input_tokens`.

- [ ] **Step 2: Run the integration test and verify RED, then GREEN**

Run:

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-039-target \
  cargo test -p aiperf-runtime --features engine,parquet --test dataset_load_bench baseten
```

Expected RED before Task 1 implementation: Arrow sample reports unsupported
input. Expected GREEN after implementation: Parquet and Arrow counts match.

- [ ] **Step 3: Run the pinned performance comparison**

Build the release adapter and run the exact baseline invocation three times
under `/usr/bin/time -v` with
`AIPERF_DATASET_BASETEN_SESSION_COLUMN=poor_man_session_id`. Record timed
`elapsed_ns`, peak RSS, and all semantic counts. Median time and RSS must be below
the specification baseline, and counts must remain
100,000 / 50,000 / 100,000 / 6,400,000.

- [ ] **Step 4: Run verification**

Run focused Baseten unit/integration tests, `cargo fmt --check`, relevant runtime
Clippy, and the release adapter build with the global target/sccache settings.
Record exact commands and outputs in `progress.md`.

- [ ] **Step 5: Perform Graham review and resolve every finding**

Review only `da917561fb..HEAD` using `.agents/skills/graham-code-review/SKILL.md`.
Make multiple passes over production and test hunks for error handling,
allocation, clones, blocking work, comments, naming, and minimal diff. Record all
findings and the final `GRAHAM APPROVED` verdict in `graham-review.md`; implement
and reverify any required corrections before closure.

- [ ] **Step 6: Update the campaign closure and commit evidence**

Set tracker #39 to `complete | applicable` and add a per-commit record with
exact merge ancestry, design/plan paths, upstream-to-native test mapping,
before/after benchmark samples, verification commands, and Graham verdict.

```bash
git add rust/runtime/tests/dataset_load_bench.rs \
  docs/porting-origin-main-campaign.md \
  .superpowers/sdd/2026-08-25-native-baseten-columnar-load
git commit -m "test: close Baseten loading performance port"
```
